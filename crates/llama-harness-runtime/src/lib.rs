//! A child-process-owned stdio JSONL adapter around the canonical Rust engine.
//!
//! This crate never starts an HTTP listener. SDKs launch this binary with private
//! pipes; stdout is reserved for protocol frames and stderr is diagnostics only.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use async_trait::async_trait;
use llama_harness_core::{
    agent::{AgentDefinition, RunOverrides, RunRequest},
    event::{EventRecord, EventSink, RunEvent},
    limits::{AgentLimits, GenerationOptions},
    message::{Message, MessageRole},
    policy::{ApprovalHandler, ApprovalRecord, PolicyDecision, PolicyEngine},
    tool::{Tool, ToolCallContext, ToolDefinition, ToolRegistry, ToolResult, ToolRisk},
    AgentRunner, HarnessError, ModelProvider,
};
use llama_harness_ollama::OllamaProvider;
use llama_harness_protocol::{
    decode_line, ApprovalDecisionResponse, ApprovalRequest, CancelRun, ClientHello,
    CommandAcknowledged, Envelope, Ping, PolicyDecisionRequest, PolicyDecisionResponse, Pong,
    ProtocolErrorCode, ProtocolErrorPayload, ProtocolMessage, ProviderConfiguration, RunCancelled,
    RunCompleted, RunEventPayload, RunFailed, RunStarted, RuntimeCapabilities, RuntimeHello,
    StartRun, ToolExecutionRequest, ToolResultResponse, WireAgentDefinition, WireAgentLimits,
    WireApprovalRecord, WireGenerationOptions, WireMessage, WireMessageRole, WirePolicyDecision,
    WireRunError, WireRunOverrides, WireRunRequest, WireRunResult, WireRunStatus,
    WireToolDefinition, WireToolResult, WireToolRisk, MAX_CONCURRENT_RUNS, MAX_MESSAGE_BYTES,
    MAX_PENDING_CALLBACKS, MAX_QUEUE_DEPTH,
};
use tokio::{
    io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader},
    sync::{mpsc, oneshot},
    task::JoinSet,
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

const CALLBACK_TIMEOUT: Duration = Duration::from_secs(60);
const RUNTIME_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("protocol writer stopped")]
    WriterStopped,
}

/// Creates the model provider used by a child runtime instance.
///
/// The production binary installs [`OllamaProviderFactory`]. Test-only child
/// binaries can inject a deterministic provider without adding another public
/// provider kind to protocol v1.
pub trait ProviderFactory: Send + Sync {
    fn create(
        &self,
        configuration: &ProviderConfiguration,
    ) -> Result<Arc<dyn ModelProvider>, HarnessError>;

    fn provider_names(&self) -> Vec<String> {
        vec!["ollama".into()]
    }
}

/// The production provider factory. It deliberately supports only the
/// existing loopback-only Ollama integration.
pub struct OllamaProviderFactory;

impl ProviderFactory for OllamaProviderFactory {
    fn create(
        &self,
        configuration: &ProviderConfiguration,
    ) -> Result<Arc<dyn ModelProvider>, HarnessError> {
        match configuration {
            ProviderConfiguration::Ollama { base_url } => Ok(Arc::new(
                OllamaProvider::builder().base_url(base_url).build()?,
            )),
        }
    }
}

pub async fn serve_stdio() -> Result<(), RuntimeError> {
    serve_stdio_with_factory(Arc::new(OllamaProviderFactory)).await
}

/// Serves the protocol over stdio with an application-supplied provider
/// factory. This is intended for deterministic integration-test sidecars; the
/// production binary calls [`serve_stdio`] and remains Ollama-only.
pub async fn serve_stdio_with_factory(
    provider_factory: Arc<dyn ProviderFactory>,
) -> Result<(), RuntimeError> {
    let (writer, writer_task) = start_writer();
    let state = Arc::new(RuntimeState::new(writer));
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut handshake_complete = false;
    let mut runs = JoinSet::new();

    loop {
        let line = match read_frame(&mut reader).await? {
            Some(line) => line,
            None => break,
        };
        let envelope = match decode_line(&line) {
            Ok(envelope) => envelope,
            Err(error) => {
                send_protocol_error(
                    &state,
                    "unavailable",
                    ProtocolErrorCode::InvalidMessage,
                    error.to_string(),
                )?;
                continue;
            }
        };
        if envelope.message.direction() != llama_harness_protocol::MessageDirection::ClientToRuntime
        {
            send_protocol_error(
                &state,
                &envelope.request_id,
                ProtocolErrorCode::InvalidState,
                "runtime-to-client message received on stdin",
            )?;
            continue;
        }
        match envelope.message {
            ProtocolMessage::ClientHello(hello) => {
                if handshake_complete {
                    send_protocol_error(
                        &state,
                        &envelope.request_id,
                        ProtocolErrorCode::InvalidState,
                        "client_hello is only valid once",
                    )?;
                } else {
                    handshake_complete = true;
                    send_runtime_hello(
                        &state,
                        envelope.request_id,
                        hello,
                        provider_factory.provider_names(),
                    )?;
                }
            }
            _ if !handshake_complete => {
                send_protocol_error(
                    &state,
                    &envelope.request_id,
                    ProtocolErrorCode::InvalidState,
                    "client_hello must be the first command",
                )?;
            }
            ProtocolMessage::StartRun(start) => {
                if state.active_count() >= usize::from(MAX_CONCURRENT_RUNS) {
                    send_protocol_error(
                        &state,
                        &envelope.request_id,
                        ProtocolErrorCode::QueueFull,
                        "active run limit reached",
                    )?;
                    continue;
                }
                let run_id = Uuid::new_v4().to_string();
                let cancellation = CancellationToken::new();
                state.register_run(run_id.clone(), cancellation.clone());
                acknowledge(
                    &state,
                    envelope.request_id,
                    Some(run_id.clone()),
                    "start_run",
                )?;
                let run_state = Arc::clone(&state);
                let provider_factory = Arc::clone(&provider_factory);
                runs.spawn(async move {
                    run_start(run_state, run_id, *start, cancellation, provider_factory).await;
                });
            }
            ProtocolMessage::CancelRun(CancelRun { .. }) => {
                let Some(run_id) = envelope.run_id else {
                    send_protocol_error(
                        &state,
                        &envelope.request_id,
                        ProtocolErrorCode::InvalidMessage,
                        "cancel_run requires run_id",
                    )?;
                    continue;
                };
                if state.cancel_run(&run_id) {
                    acknowledge(&state, envelope.request_id, Some(run_id), "cancel_run")?;
                } else {
                    send_protocol_error(
                        &state,
                        &envelope.request_id,
                        ProtocolErrorCode::UnknownRun,
                        "run is unknown or already terminal",
                    )?;
                }
            }
            ProtocolMessage::ToolResult(response) => resolve_callback(&state, response)?,
            ProtocolMessage::PolicyDecision(response) => resolve_policy(&state, response)?,
            ProtocolMessage::ApprovalDecision(response) => resolve_approval(&state, response)?,
            ProtocolMessage::Ping(Ping { nonce }) => {
                state.send(Envelope::new(
                    envelope.request_id,
                    None,
                    ProtocolMessage::Pong(Pong { nonce }),
                ))?;
            }
            ProtocolMessage::Shutdown(_) => {
                acknowledge(&state, envelope.request_id, None, "shutdown")?;
                break;
            }
            _ => {
                send_protocol_error(
                    &state,
                    &envelope.request_id,
                    ProtocolErrorCode::InvalidState,
                    "unsupported client command",
                )?;
            }
        }
    }

    state.cancel_all();
    while runs.join_next().await.is_some() {}
    drop(state);
    Ok(writer_task
        .await
        .map_err(|_| RuntimeError::WriterStopped)??)
}

async fn run_start(
    state: Arc<RuntimeState>,
    run_id: String,
    start: StartRun,
    cancellation: CancellationToken,
    provider_factory: Arc<dyn ProviderFactory>,
) {
    let trace_id = Uuid::new_v4().to_string();
    let run_sequence = state.next_sequence(&run_id);
    let _ = state.send(Envelope::new(
        Uuid::new_v4().to_string(),
        Some(run_id.clone()),
        ProtocolMessage::RunStarted(RunStarted {
            trace_id: trace_id.clone(),
            run_sequence,
        }),
    ));

    let result = build_runner(
        &state,
        &run_id,
        start.request,
        cancellation.clone(),
        trace_id.clone(),
        provider_factory.as_ref(),
    );
    match result {
        Ok((runner, request)) => match runner.run(request).await {
            Ok(result) => {
                let sequence = state.next_sequence(&run_id);
                let message = match result.status {
                    llama_harness_core::RunStatus::Completed
                    | llama_harness_core::RunStatus::LimitReached => {
                        ProtocolMessage::RunCompleted(RunCompleted {
                            run_sequence: sequence,
                            result: to_wire_result(result),
                        })
                    }
                    llama_harness_core::RunStatus::Cancelled => {
                        ProtocolMessage::RunCancelled(RunCancelled {
                            run_sequence: sequence,
                            reason: "run cancelled".into(),
                        })
                    }
                    llama_harness_core::RunStatus::Failed => {
                        ProtocolMessage::RunFailed(RunFailed {
                            run_sequence: sequence,
                            error: WireRunError {
                                code: "run_failed".into(),
                                message: "canonical runner failed".into(),
                            },
                        })
                    }
                };
                let _ = state.send(Envelope::new(
                    Uuid::new_v4().to_string(),
                    Some(run_id.clone()),
                    message,
                ));
            }
            Err(error) => send_run_failed(&state, &run_id, error.to_string()),
        },
        Err(error) => send_run_failed(&state, &run_id, error.to_string()),
    }
    state.remove_run(&run_id);
}

fn build_runner(
    state: &Arc<RuntimeState>,
    run_id: &str,
    request: WireRunRequest,
    cancellation: CancellationToken,
    trace_id: String,
    provider_factory: &dyn ProviderFactory,
) -> Result<(AgentRunner, RunRequest), HarnessError> {
    let provider = provider_factory.create(&request.provider)?;
    let mut tools = ToolRegistry::default();
    for definition in request.tools {
        tools.register(Arc::new(BridgeTool {
            definition: to_core_tool_definition(definition),
            callbacks: Arc::clone(state),
        }))?;
    }
    let agent = to_core_agent(request.agent);
    let core_request = RunRequest {
        agent,
        input: request.input,
        application_context: request.application_context,
        history: request.history.into_iter().map(to_core_message).collect(),
        metadata: request.metadata,
        overrides: to_core_overrides(request.overrides),
        evaluation: request.evaluation,
        cancellation,
        run_id: Some(run_id.into()),
        trace_id: Some(trace_id),
    };
    let events: Arc<dyn EventSink> = Arc::new(RuntimeEventSink {
        state: Arc::clone(state),
    });
    Ok((
        AgentRunner::builder(provider)
            .tools(tools)
            .policy(Arc::new(BridgePolicy {
                callbacks: Arc::clone(state),
            }))
            .approvals(Arc::new(BridgeApproval {
                callbacks: Arc::clone(state),
            }))
            .event_sink(events)
            .build(),
        core_request,
    ))
}

struct RuntimeState {
    writer: mpsc::Sender<Envelope>,
    runs: Mutex<HashMap<String, RunControl>>,
    callbacks: Mutex<HashMap<String, PendingCallback>>,
}

struct RunControl {
    cancellation: CancellationToken,
    sequence: AtomicU64,
}

enum CallbackResponse {
    Tool(WireToolResult),
    Policy(WirePolicyDecision),
    Approval { granted: bool, reason: String },
}
enum CallbackKind {
    Tool,
    Policy,
    Approval,
}
struct PendingCallback {
    kind: CallbackKind,
    cancellation: CancellationToken,
    sender: oneshot::Sender<CallbackResponse>,
}

impl RuntimeState {
    fn new(writer: mpsc::Sender<Envelope>) -> Self {
        Self {
            writer,
            runs: Mutex::new(HashMap::new()),
            callbacks: Mutex::new(HashMap::new()),
        }
    }
    fn send(&self, envelope: Envelope) -> Result<(), RuntimeError> {
        self.writer
            .try_send(envelope)
            .map_err(|_| RuntimeError::WriterStopped)
    }
    fn register_run(&self, run_id: String, cancellation: CancellationToken) {
        self.runs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                run_id,
                RunControl {
                    cancellation,
                    sequence: AtomicU64::new(0),
                },
            );
    }
    fn remove_run(&self, run_id: &str) {
        self.runs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(run_id);
    }
    fn active_count(&self) -> usize {
        self.runs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
    fn cancel_run(&self, run_id: &str) -> bool {
        self.runs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(run_id)
            .map(|run| run.cancellation.cancel())
            .is_some()
    }
    fn cancel_all(&self) {
        for run in self
            .runs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
        {
            run.cancellation.cancel();
        }
    }
    fn next_sequence(&self, run_id: &str) -> u64 {
        self.runs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(run_id)
            .map(|run| run.sequence.fetch_add(1, Ordering::SeqCst) + 1)
            .unwrap_or(0)
    }
    async fn request_callback(
        &self,
        context: &ToolCallContext,
        kind: CallbackKind,
        message: ProtocolMessage,
    ) -> Result<CallbackResponse, HarnessError> {
        let callback_id = match &message {
            ProtocolMessage::ToolExecutionRequested(request) => &request.callback_id,
            ProtocolMessage::PolicyDecisionRequested(request) => &request.callback_id,
            ProtocolMessage::ApprovalRequested(request) => &request.callback_id,
            _ => return Err(HarnessError::Tool("invalid callback message".into())),
        }
        .clone();
        if self
            .callbacks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
            >= usize::from(MAX_PENDING_CALLBACKS)
        {
            return Err(HarnessError::ResourceLimit(
                "pending callback limit reached".into(),
            ));
        }
        let (sender, receiver) = oneshot::channel();
        self.callbacks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                callback_id.clone(),
                PendingCallback {
                    kind,
                    cancellation: self
                        .runs
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .get(&context.run_id)
                        .map(|run| run.cancellation.clone())
                        .unwrap_or_default(),
                    sender,
                },
            );
        self.send(Envelope::new(
            Uuid::new_v4().to_string(),
            Some(context.run_id.clone()),
            message,
        ))
        .map_err(|_| HarnessError::Tool("protocol writer stopped".into()))?;
        let cancellation = self
            .runs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&context.run_id)
            .map(|run| run.cancellation.clone())
            .unwrap_or_default();
        let outcome = tokio::select! { response = receiver => response.map_err(|_| HarnessError::Cancelled), _ = cancellation.cancelled() => Err(HarnessError::Cancelled), _ = tokio::time::sleep(CALLBACK_TIMEOUT) => Err(HarnessError::TimedOut("SDK callback timed out".into())) };
        self.callbacks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&callback_id);
        outcome
    }
}

struct BridgeTool {
    definition: ToolDefinition,
    callbacks: Arc<RuntimeState>,
}
#[async_trait]
impl Tool for BridgeTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn execute(
        &self,
        _: serde_json::Value,
        _: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        Err(HarnessError::Tool(
            "bridge tools require execution context".into(),
        ))
    }
    async fn execute_with_context(
        &self,
        context: &ToolCallContext,
        arguments: serde_json::Value,
        _: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        let callback_id = Uuid::new_v4().to_string();
        let response = self
            .callbacks
            .request_callback(
                context,
                CallbackKind::Tool,
                ProtocolMessage::ToolExecutionRequested(ToolExecutionRequest {
                    run_sequence: self.callbacks.next_sequence(&context.run_id),
                    callback_id,
                    trace_id: context.trace_id.clone(),
                    call_id: context.call_id.clone(),
                    tool: to_wire_tool_definition(self.definition.clone()),
                    arguments,
                    deadline_ms: Some(CALLBACK_TIMEOUT.as_millis() as u64),
                }),
            )
            .await?;
        match response {
            CallbackResponse::Tool(result) => Ok(ToolResult {
                ok: result.ok,
                output: result.output,
                error: result.error,
            }),
            _ => Err(HarnessError::Tool("callback kind mismatch".into())),
        }
    }
}

struct BridgePolicy {
    callbacks: Arc<RuntimeState>,
}
#[async_trait]
impl PolicyEngine for BridgePolicy {
    async fn decide(
        &self,
        tool: &ToolDefinition,
        _: &serde_json::Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        if tool.read_only {
            Ok(PolicyDecision::Allow {
                reason: "read-only tool allowed by default policy".into(),
            })
        } else {
            Ok(PolicyDecision::Deny {
                reason: "state-changing tool requires an explicit policy".into(),
            })
        }
    }
    async fn decide_with_context(
        &self,
        context: &ToolCallContext,
        tool: &ToolDefinition,
        arguments: &serde_json::Value,
        request: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        let callback_id = Uuid::new_v4().to_string();
        let response = self
            .callbacks
            .request_callback(
                context,
                CallbackKind::Policy,
                ProtocolMessage::PolicyDecisionRequested(PolicyDecisionRequest {
                    run_sequence: self.callbacks.next_sequence(&context.run_id),
                    callback_id,
                    trace_id: context.trace_id.clone(),
                    call_id: context.call_id.clone(),
                    tool: to_wire_tool_definition(tool.clone()),
                    arguments: arguments.clone(),
                    deadline_ms: request.agent.limits.max_run_duration_ms,
                }),
            )
            .await?;
        match response {
            CallbackResponse::Policy(decision) => Ok(to_core_policy(decision)),
            _ => Err(HarnessError::Policy("callback kind mismatch".into())),
        }
    }
}

struct BridgeApproval {
    callbacks: Arc<RuntimeState>,
}
#[async_trait]
impl ApprovalHandler for BridgeApproval {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        _: &serde_json::Value,
        _: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        Ok(ApprovalRecord {
            call_id: String::new(),
            tool_id: tool.id.clone(),
            granted: false,
            reason: "no approval handler configured".into(),
        })
    }
    async fn approve_with_context(
        &self,
        context: &ToolCallContext,
        tool: &ToolDefinition,
        arguments: &serde_json::Value,
        request: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        let callback_id = Uuid::new_v4().to_string();
        let response = self
            .callbacks
            .request_callback(
                context,
                CallbackKind::Approval,
                ProtocolMessage::ApprovalRequested(ApprovalRequest {
                    run_sequence: self.callbacks.next_sequence(&context.run_id),
                    callback_id,
                    trace_id: context.trace_id.clone(),
                    call_id: context.call_id.clone(),
                    tool: to_wire_tool_definition(tool.clone()),
                    arguments: arguments.clone(),
                    deadline_ms: request.agent.limits.max_run_duration_ms,
                }),
            )
            .await?;
        match response {
            CallbackResponse::Approval { granted, reason } => Ok(ApprovalRecord {
                call_id: context.call_id.clone(),
                tool_id: context.tool_id.clone(),
                granted,
                reason,
            }),
            _ => Err(HarnessError::Approval("callback kind mismatch".into())),
        }
    }
}

struct RuntimeEventSink {
    state: Arc<RuntimeState>,
}
impl EventSink for RuntimeEventSink {
    fn emit(&self, record: EventRecord) {
        let run_sequence = self.state.next_sequence(&record.run_id);
        let event = to_wire_event(record.event);
        let envelope = Envelope::new(
            Uuid::new_v4().to_string(),
            Some(record.run_id),
            ProtocolMessage::RunEvent(RunEventPayload {
                trace_id: record.trace_id,
                sequence: run_sequence,
                timestamp_ms: record.timestamp_ms,
                event,
            }),
        );
        if self.state.send(envelope).is_err() {
            self.state.cancel_all();
        }
    }
}

fn resolve_callback(
    state: &RuntimeState,
    response: ToolResultResponse,
) -> Result<(), RuntimeError> {
    resolve(
        state,
        response.callback_id,
        CallbackKind::Tool,
        CallbackResponse::Tool(response.result),
    )
}
fn resolve_policy(
    state: &RuntimeState,
    response: PolicyDecisionResponse,
) -> Result<(), RuntimeError> {
    resolve(
        state,
        response.callback_id,
        CallbackKind::Policy,
        CallbackResponse::Policy(response.decision),
    )
}
fn resolve_approval(
    state: &RuntimeState,
    response: ApprovalDecisionResponse,
) -> Result<(), RuntimeError> {
    resolve(
        state,
        response.callback_id,
        CallbackKind::Approval,
        CallbackResponse::Approval {
            granted: response.granted,
            reason: response.reason,
        },
    )
}
fn resolve(
    state: &RuntimeState,
    callback_id: String,
    kind: CallbackKind,
    response: CallbackResponse,
) -> Result<(), RuntimeError> {
    let pending = state
        .callbacks
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&callback_id);
    let Some(pending) = pending else {
        return send_protocol_error(
            state,
            "unavailable",
            ProtocolErrorCode::UnknownCallback,
            "callback is unknown, stale, or already resolved",
        );
    };
    if !matches!(
        (&pending.kind, &kind),
        (CallbackKind::Tool, CallbackKind::Tool)
            | (CallbackKind::Policy, CallbackKind::Policy)
            | (CallbackKind::Approval, CallbackKind::Approval)
    ) {
        pending.cancellation.cancel();
        return send_protocol_error(
            state,
            "unavailable",
            ProtocolErrorCode::InvalidMessage,
            "callback response has the wrong kind",
        );
    }
    if pending.sender.send(response).is_err() {
        pending.cancellation.cancel();
        return send_protocol_error(
            state,
            "unavailable",
            ProtocolErrorCode::UnknownCallback,
            "callback is no longer pending",
        );
    }
    Ok(())
}

fn send_runtime_hello(
    state: &RuntimeState,
    request_id: String,
    _: ClientHello,
    providers: Vec<String>,
) -> Result<(), RuntimeError> {
    state.send(Envelope::new(
        request_id,
        None,
        ProtocolMessage::RuntimeHello(RuntimeHello {
            runtime_version: RUNTIME_VERSION.into(),
            capabilities: RuntimeCapabilities {
                supports_output_deltas: false,
                supports_structured_output: true,
                supports_trace_persistence: false,
                concurrent_runs: MAX_CONCURRENT_RUNS,
                max_pending_callbacks: MAX_PENDING_CALLBACKS,
                max_queue_depth: MAX_QUEUE_DEPTH,
            },
            providers,
        }),
    ))
}
fn acknowledge(
    state: &RuntimeState,
    request_id: String,
    run_id: Option<String>,
    command: &str,
) -> Result<(), RuntimeError> {
    state.send(Envelope::new(
        request_id,
        run_id,
        ProtocolMessage::CommandAcknowledged(CommandAcknowledged {
            command: command.into(),
        }),
    ))
}
fn send_protocol_error(
    state: &RuntimeState,
    request_id: &str,
    code: ProtocolErrorCode,
    message: impl Into<String>,
) -> Result<(), RuntimeError> {
    state.send(Envelope::new(
        request_id,
        None,
        ProtocolMessage::ProtocolError(ProtocolErrorPayload {
            code,
            message: message.into(),
            retryable: false,
        }),
    ))
}
fn send_run_failed(state: &RuntimeState, run_id: &str, message: String) {
    let _ = state.send(Envelope::new(
        Uuid::new_v4().to_string(),
        Some(run_id.into()),
        ProtocolMessage::RunFailed(RunFailed {
            run_sequence: state.next_sequence(run_id),
            error: WireRunError {
                code: "runtime_error".into(),
                message,
            },
        }),
    ));
}

fn start_writer() -> (
    mpsc::Sender<Envelope>,
    tokio::task::JoinHandle<Result<(), io::Error>>,
) {
    let (sender, mut receiver) = mpsc::channel(usize::from(MAX_QUEUE_DEPTH));
    let task = tokio::spawn(async move {
        let mut stdout = io::stdout();
        while let Some(envelope) = receiver.recv().await {
            let mut line = serde_json::to_vec(&envelope).expect("protocol envelope serializes");
            line.push(b'\n');
            stdout.write_all(&line).await?;
            stdout.flush().await?;
        }
        Ok(())
    });
    (sender, task)
}
async fn read_frame(reader: &mut BufReader<io::Stdin>) -> Result<Option<Vec<u8>>, io::Error> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Ok(Some(line))
            };
        }
        if let Some(index) = available.iter().position(|byte| *byte == b'\n') {
            if line.len() + index > MAX_MESSAGE_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "protocol frame too large",
                ));
            }
            line.extend_from_slice(&available[..index]);
            reader.consume(index + 1);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return Ok(Some(line));
        }
        if line.len() + available.len() > MAX_MESSAGE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "protocol frame too large",
            ));
        }
        let consumed = available.len();
        line.extend_from_slice(available);
        reader.consume(consumed);
    }
}

fn to_core_agent(agent: WireAgentDefinition) -> AgentDefinition {
    AgentDefinition {
        id: agent.id,
        name: agent.name,
        version: agent.version,
        system_instructions: agent.system_instructions,
        default_model: agent.default_model,
        tool_allowlist: agent.tool_allowlist,
        limits: to_core_limits(agent.limits),
        generation: to_core_generation(agent.generation),
        output_schema: agent.output_schema,
        metadata: agent.metadata,
    }
}
fn to_core_limits(limits: WireAgentLimits) -> AgentLimits {
    AgentLimits {
        max_model_calls: limits.max_model_calls.min(32),
        max_tool_calls: limits.max_tool_calls.min(64),
        max_identical_tool_calls: limits.max_identical_tool_calls.min(8),
        max_run_duration_ms: Some(limits.max_run_duration_ms.unwrap_or(300_000).min(300_000)),
        max_model_call_duration_ms: limits
            .max_model_call_duration_ms
            .map(|value| value.min(120_000)),
        max_output_repairs: limits.max_output_repairs.min(4),
        max_provider_retries: limits.max_provider_retries.min(4),
        max_input_bytes: limits.max_input_bytes.min(64 * 1024),
        max_request_payload_bytes: limits.max_request_payload_bytes.min(256 * 1024),
        max_model_response_bytes: limits.max_model_response_bytes.min(1024 * 1024),
        max_tool_arguments_bytes: limits.max_tool_arguments_bytes.min(64 * 1024),
        max_tool_result_bytes: limits.max_tool_result_bytes.min(1024 * 1024),
        max_transcript_bytes: limits.max_transcript_bytes.min(4 * 1024 * 1024),
        max_json_depth: limits.max_json_depth.min(64),
    }
}
fn to_core_generation(generation: WireGenerationOptions) -> GenerationOptions {
    GenerationOptions {
        temperature: generation.temperature,
        top_p: generation.top_p,
        max_output_tokens: generation.max_output_tokens,
    }
}
fn to_core_overrides(overrides: WireRunOverrides) -> RunOverrides {
    RunOverrides {
        model: overrides.model,
        generation: to_core_generation(overrides.generation),
    }
}
fn to_core_message(message: WireMessage) -> Message {
    Message {
        role: match message.role {
            WireMessageRole::System => MessageRole::System,
            WireMessageRole::User => MessageRole::User,
            WireMessageRole::Assistant => MessageRole::Assistant,
            WireMessageRole::Tool => MessageRole::Tool,
        },
        content: message.content,
        tool_call_id: message.tool_call_id,
        tool_calls: message
            .tool_calls
            .into_iter()
            .map(|call| llama_harness_core::ToolCall {
                id: call.id,
                tool_id: call.tool_id,
                arguments_json: call.arguments_json,
            })
            .collect(),
    }
}
fn to_core_tool_definition(definition: WireToolDefinition) -> ToolDefinition {
    ToolDefinition {
        id: definition.id,
        name: definition.name,
        description: definition.description,
        arguments_schema: definition.arguments_schema,
        risk: match definition.risk {
            WireToolRisk::Low => ToolRisk::Low,
            WireToolRisk::Medium => ToolRisk::Medium,
            WireToolRisk::High => ToolRisk::High,
        },
        idempotent: definition.idempotent,
        read_only: definition.read_only,
    }
}
fn to_wire_tool_definition(definition: ToolDefinition) -> WireToolDefinition {
    WireToolDefinition {
        id: definition.id,
        name: definition.name,
        description: definition.description,
        arguments_schema: definition.arguments_schema,
        risk: match definition.risk {
            ToolRisk::Low => WireToolRisk::Low,
            ToolRisk::Medium => WireToolRisk::Medium,
            ToolRisk::High => WireToolRisk::High,
        },
        idempotent: definition.idempotent,
        read_only: definition.read_only,
    }
}
fn to_core_policy(decision: WirePolicyDecision) -> PolicyDecision {
    match decision {
        WirePolicyDecision::Allow { reason } => PolicyDecision::Allow { reason },
        WirePolicyDecision::Deny { reason } => PolicyDecision::Deny { reason },
        WirePolicyDecision::RequireApproval { reason } => {
            PolicyDecision::RequireApproval { reason }
        }
    }
}
fn to_wire_event(event: RunEvent) -> llama_harness_protocol::WireRunEvent {
    match event {
        RunEvent::Started { trace_id, .. } => {
            llama_harness_protocol::WireRunEvent::Started { trace_id }
        }
        RunEvent::ModelRequested { call_number, model } => {
            llama_harness_protocol::WireRunEvent::ModelRequested { call_number, model }
        }
        RunEvent::ModelRetrying {
            next_call_number,
            reason,
        } => llama_harness_protocol::WireRunEvent::ModelRetrying {
            next_call_number,
            reason,
        },
        RunEvent::ModelResponded { call_number } => {
            llama_harness_protocol::WireRunEvent::ModelResponded { call_number }
        }
        RunEvent::ToolRejected {
            call_id,
            tool_id,
            reason,
        } => llama_harness_protocol::WireRunEvent::ToolRejected {
            call_id,
            tool_id,
            reason,
        },
        RunEvent::PolicyDecided { call_id, decision } => {
            llama_harness_protocol::WireRunEvent::PolicyDecided {
                call_id,
                decision: to_wire_policy(decision),
            }
        }
        RunEvent::ApprovalRequested { call_id, tool_id } => {
            llama_harness_protocol::WireRunEvent::ApprovalRequested { call_id, tool_id }
        }
        RunEvent::ToolCompleted {
            call_id,
            tool_id,
            ok,
        } => llama_harness_protocol::WireRunEvent::ToolCompleted {
            call_id,
            tool_id,
            ok,
        },
        RunEvent::Completed { status } => llama_harness_protocol::WireRunEvent::Completed {
            status: to_wire_status(status),
        },
    }
}
fn to_wire_policy(decision: PolicyDecision) -> WirePolicyDecision {
    match decision {
        PolicyDecision::Allow { reason } => WirePolicyDecision::Allow { reason },
        PolicyDecision::Deny { reason } => WirePolicyDecision::Deny { reason },
        PolicyDecision::RequireApproval { reason } => {
            WirePolicyDecision::RequireApproval { reason }
        }
    }
}
fn to_wire_status(status: llama_harness_core::RunStatus) -> WireRunStatus {
    match status {
        llama_harness_core::RunStatus::Completed => WireRunStatus::Completed,
        llama_harness_core::RunStatus::Failed => WireRunStatus::Failed,
        llama_harness_core::RunStatus::Cancelled => WireRunStatus::Cancelled,
        llama_harness_core::RunStatus::LimitReached => WireRunStatus::LimitReached,
    }
}
fn to_wire_result(result: llama_harness_core::RunResult) -> WireRunResult {
    WireRunResult {
        status: to_wire_status(result.status),
        final_output: result.final_output,
        model: result.model,
        tool_calls: result
            .tool_calls
            .into_iter()
            .map(|call| llama_harness_protocol::WireToolCall {
                id: call.id,
                tool_id: call.tool_id,
                arguments_json: call.arguments_json,
            })
            .collect(),
        policy_decisions: result
            .policy_decisions
            .into_iter()
            .map(to_wire_policy)
            .collect(),
        approvals: result
            .approvals
            .into_iter()
            .map(|approval| WireApprovalRecord {
                call_id: approval.call_id,
                tool_id: approval.tool_id,
                granted: approval.granted,
                reason: approval.reason,
            })
            .collect(),
        errors: result
            .errors
            .into_iter()
            .map(|error| WireRunError {
                code: error.code,
                message: error.message,
            })
            .collect(),
        duration_ms: result.duration_ms,
        trace_id: result.trace_id,
        model_call_limit_reached: result.model_call_limit_reached,
        tool_call_limit_reached: result.tool_call_limit_reached,
        repeated_tool_call_limit_reached: result.repeated_tool_call_limit_reached,
        cancelled: result.cancelled,
    }
}
