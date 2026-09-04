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
    AgentDefinition, AgentLimits, AgentRunner, ApprovalHandler, ApprovalRecord, CancellationSafety,
    EventRecord, EventSink, ExecutionLocation, GenerationOptions, HarnessError, IssueSafety,
    Message, MessageRole, ModelProvider, NetworkEgress, PolicyDecision, PolicyEngine, RunEvent,
    RunOverrides, RunRequest, RunStrategy, SpeculationPolicy, Tool, ToolCallContext, ToolCaller,
    ToolDefinition, ToolRegistry, ToolResult, ToolRisk,
};
use llama_harness_ollama::OllamaProvider;
use llama_harness_protocol::{
    decode_line, ApprovalDecisionResponse, ApprovalRequest, CancelRun, ClientHello,
    CommandAcknowledged, Envelope, ModelInfo as WireModelInfo, ModelInventoryResponse, Ping,
    PolicyDecisionRequest, PolicyDecisionResponse, Pong, ProtocolErrorCode, ProtocolErrorPayload,
    ProtocolMessage, ProtocolVersion, ProviderConfiguration, ProviderHealthResponse,
    ProviderInspectionRequest, RunCancelled, RunCompleted, RunEventPayload, RunFailed, RunStarted,
    RuntimeCapabilities, RuntimeHello, StartRun, ToolExecutionRequest, ToolResultResponse,
    WireAgentDefinition, WireAgentLimits, WireApprovalRecord, WireCancellationSafety,
    WireExecutionLocation, WireGenerationOptions, WireIssueSafety, WireMessage, WireMessageRole,
    WireNetworkEgress, WirePlanLifecycleOutcome, WirePlanNodeOutcome, WirePlanPhase,
    WirePolicyDecision, WireProgramLifecycleOutcome, WireProviderCapabilityLimits, WireRunError,
    WireRunOverrides, WireRunRequest, WireRunResult, WireRunStatus, WireRunStrategy,
    WireSpeculationPolicy, WireStrategyFallbackReason, WireStrategySelectionReason, WireToolCaller,
    WireToolDefinition, WireToolDiscoveryOutcome, WireToolDiscoverySelection, WireToolResult,
    WireToolRisk, MAX_CONCURRENT_RUNS, MAX_MESSAGE_BYTES, MAX_PENDING_CALLBACKS, MAX_QUEUE_DEPTH,
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
                    protocol_error_code(&error),
                    error.to_string(),
                )?;
                continue;
            }
        };
        if handshake_complete && envelope.protocol_version != state.protocol_version() {
            send_protocol_error(
                &state,
                &envelope.request_id,
                ProtocolErrorCode::IncompatibleVersion,
                "protocol_version must remain pinned after client_hello",
            )?;
            continue;
        }
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
                    state.select_protocol(envelope.protocol_version);
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
            ProtocolMessage::GetProviderHealth(request) => {
                let state = Arc::clone(&state);
                let provider_factory = Arc::clone(&provider_factory);
                let request_id = envelope.request_id;
                runs.spawn(async move {
                    report_provider_health(state, provider_factory, request_id, request).await;
                });
            }
            ProtocolMessage::GetModelInventory(request) => {
                let state = Arc::clone(&state);
                let provider_factory = Arc::clone(&provider_factory);
                let request_id = envelope.request_id;
                runs.spawn(async move {
                    report_model_inventory(state, provider_factory, request_id, request).await;
                });
            }
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
    let mut wire_request = start.request;
    let strategy_was_explicit = wire_request.strategy.is_some();
    let strategy = wire_request.strategy.unwrap_or_default();
    if state.protocol_version() == ProtocolVersion::V1_0 && strategy_was_explicit {
        send_run_failed_with_code(
            &state,
            &run_id,
            "unsupported_strategy",
            "selected protocol version 1.0 does not support the requested strategy",
        );
        state.remove_run(&run_id);
        return;
    }
    if strategy == WireRunStrategy::Programmatic {
        send_run_failed_with_code(
            &state,
            &run_id,
            "unsupported_strategy",
            "programmatic execution is unavailable because this sidecar has no configured sandbox",
        );
        state.remove_run(&run_id);
        return;
    }
    if state.protocol_version() == ProtocolVersion::V1_0 {
        project_v1_0_request(&mut wire_request);
    }
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
        wire_request,
        cancellation.clone(),
        trace_id.clone(),
        provider_factory.as_ref(),
    );
    match result {
        Ok((runner, request)) => match runner
            .run_with_strategy(request, to_core_strategy(strategy))
            .await
        {
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
                    _ => ProtocolMessage::RunFailed(RunFailed {
                        run_sequence: sequence,
                        error: WireRunError {
                            code: "unsupported_run_status".into(),
                            message: "runtime does not support this run status".into(),
                        },
                    }),
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
    protocol_version: Mutex<ProtocolVersion>,
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
            protocol_version: Mutex::new(ProtocolVersion::CURRENT),
            runs: Mutex::new(HashMap::new()),
            callbacks: Mutex::new(HashMap::new()),
        }
    }
    fn select_protocol(&self, offered: ProtocolVersion) {
        let selected = ProtocolVersion::CURRENT
            .negotiate(offered)
            .expect("validated v1 protocol major must negotiate");
        *self
            .protocol_version
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = selected;
    }
    fn protocol_version(&self) -> ProtocolVersion {
        *self
            .protocol_version
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    fn send(&self, mut envelope: Envelope) -> Result<(), RuntimeError> {
        let version = self.protocol_version();
        envelope.protocol_version = version;
        if version == ProtocolVersion::V1_0 && !project_v1_0(&mut envelope.message) {
            return Ok(());
        }
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
            CallbackResponse::Tool(result) => {
                Ok(ToolResult::new(result.ok, result.output, result.error))
            }
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
        Ok(ApprovalRecord::new(
            "",
            tool.id.clone(),
            false,
            "no approval handler configured",
        ))
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
            CallbackResponse::Approval { granted, reason } => Ok(ApprovalRecord::new(
                context.call_id.clone(),
                context.tool_id.clone(),
                granted,
                reason,
            )),
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
        let Some(event) = to_wire_event(record.event) else {
            return;
        };
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

async fn report_provider_health(
    state: Arc<RuntimeState>,
    provider_factory: Arc<dyn ProviderFactory>,
    request_id: String,
    request: ProviderInspectionRequest,
) {
    let result = match provider_factory.create(&request.provider) {
        Ok(provider) => provider.health().await,
        Err(error) => Err(error),
    };
    match result {
        Ok(health) => {
            let _ = state.send(Envelope::new(
                request_id,
                None,
                ProtocolMessage::ProviderHealth(ProviderHealthResponse {
                    healthy: health.healthy,
                    detail: health.detail,
                }),
            ));
        }
        Err(error) => {
            let _ = send_protocol_error(
                &state,
                &request_id,
                ProtocolErrorCode::RuntimeUnavailable,
                error.to_string(),
            );
        }
    }
}

async fn report_model_inventory(
    state: Arc<RuntimeState>,
    provider_factory: Arc<dyn ProviderFactory>,
    request_id: String,
    request: ProviderInspectionRequest,
) {
    let result = match provider_factory.create(&request.provider) {
        Ok(provider) => provider.list_models().await,
        Err(error) => Err(error),
    };
    match result {
        Ok(models) => {
            let _ = state.send(Envelope::new(
                request_id,
                None,
                ProtocolMessage::ModelInventory(ModelInventoryResponse {
                    models: models.into_iter().map(to_wire_model_info).collect(),
                }),
            ));
        }
        Err(error) => {
            let _ = send_protocol_error(
                &state,
                &request_id,
                ProtocolErrorCode::RuntimeUnavailable,
                error.to_string(),
            );
        }
    }
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
    send_run_failed_with_code(state, run_id, "runtime_error", message);
}
fn send_run_failed_with_code(
    state: &RuntimeState,
    run_id: &str,
    code: impl Into<String>,
    message: impl Into<String>,
) {
    let _ = state.send(Envelope::new(
        Uuid::new_v4().to_string(),
        Some(run_id.into()),
        ProtocolMessage::RunFailed(RunFailed {
            run_sequence: state.next_sequence(run_id),
            error: WireRunError {
                code: code.into(),
                message: message.into(),
            },
        }),
    ));
}

fn protocol_error_code(
    error: &llama_harness_protocol::ProtocolValidationError,
) -> ProtocolErrorCode {
    match error {
        llama_harness_protocol::ProtocolValidationError::IncompatibleVersion { .. } => {
            ProtocolErrorCode::IncompatibleVersion
        }
        llama_harness_protocol::ProtocolValidationError::UnknownMessageType(_) => {
            ProtocolErrorCode::UnknownMessageType
        }
        llama_harness_protocol::ProtocolValidationError::MessageTooLarge => {
            ProtocolErrorCode::MessageTooLarge
        }
        llama_harness_protocol::ProtocolValidationError::JsonTooDeep => {
            ProtocolErrorCode::JsonTooDeep
        }
        _ => ProtocolErrorCode::InvalidMessage,
    }
}

fn project_v1_0(message: &mut ProtocolMessage) -> bool {
    match message {
        ProtocolMessage::RunEvent(payload) => {
            if !matches!(
                payload.event,
                llama_harness_protocol::WireRunEvent::Started { .. }
                    | llama_harness_protocol::WireRunEvent::ModelRequested { .. }
                    | llama_harness_protocol::WireRunEvent::ModelRetrying { .. }
                    | llama_harness_protocol::WireRunEvent::ModelResponded { .. }
                    | llama_harness_protocol::WireRunEvent::ToolRejected { .. }
                    | llama_harness_protocol::WireRunEvent::PolicyDecided { .. }
                    | llama_harness_protocol::WireRunEvent::ApprovalRequested { .. }
                    | llama_harness_protocol::WireRunEvent::ToolCompleted { .. }
                    | llama_harness_protocol::WireRunEvent::Completed { .. }
            ) {
                return false;
            }
        }
        ProtocolMessage::PolicyDecisionRequested(request) => project_v1_0_tool(&mut request.tool),
        ProtocolMessage::ApprovalRequested(request) => project_v1_0_tool(&mut request.tool),
        ProtocolMessage::ToolExecutionRequested(request) => project_v1_0_tool(&mut request.tool),
        ProtocolMessage::ModelInventory(response) => {
            for model in &mut response.models {
                model.capabilities.supports_strict_tool_schemas = false;
                model.capabilities.supports_streaming_tool_arguments = false;
                model.capabilities.supports_parallel_tool_calls = false;
                model.capabilities.supports_structured_plans = false;
                model.capabilities.supports_programmatic_calling = false;
                model.capabilities.programmatic_conformance = None;
                model.capabilities.limits = WireProviderCapabilityLimits::default();
            }
        }
        _ => {}
    }
    true
}

fn project_v1_0_tool(tool: &mut WireToolDefinition) {
    tool.output_schema = None;
    tool.parallel_safe = false;
    tool.concurrency_key = None;
    tool.cancellation_safety = WireCancellationSafety::Unknown;
    tool.expected_latency_ms = None;
    tool.allowed_callers = std::collections::BTreeSet::from([WireToolCaller::Direct]);
    tool.speculation_policy = WireSpeculationPolicy::Disabled;
    tool.issue_safety = WireIssueSafety::Unknown;
    tool.execution_location = WireExecutionLocation::Unknown;
    tool.network_egress = WireNetworkEgress::Unknown;
}

fn project_v1_0_request(request: &mut WireRunRequest) {
    for tool in &mut request.tools {
        project_v1_0_tool(tool);
    }
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
        ..AgentLimits::default()
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
    let WireMessage {
        role,
        content,
        tool_call_id,
        tool_calls,
    } = message;
    let message = Message::new(
        match role {
            WireMessageRole::System => MessageRole::System,
            WireMessageRole::User => MessageRole::User,
            WireMessageRole::Assistant => MessageRole::Assistant,
            WireMessageRole::Tool => MessageRole::Tool,
        },
        content,
    )
    .with_tool_calls(
        tool_calls
            .into_iter()
            .map(|call| {
                llama_harness_core::ToolCall::new(call.id, call.tool_id, call.arguments_json)
            })
            .collect(),
    );
    match tool_call_id {
        Some(tool_call_id) => message.with_tool_call_id(tool_call_id),
        None => message,
    }
}
fn to_core_tool_definition(definition: WireToolDefinition) -> ToolDefinition {
    let risk = match definition.risk {
        WireToolRisk::Low => ToolRisk::Low,
        WireToolRisk::Medium => ToolRisk::Medium,
        WireToolRisk::High => ToolRisk::High,
    };
    let mut tool = ToolDefinition::new(
        definition.id,
        definition.name,
        definition.description,
        definition.arguments_schema,
    )
    .with_risk(risk)
    .with_idempotent(definition.idempotent)
    .with_read_only(definition.read_only)
    .with_parallel_safe(definition.parallel_safe)
    .with_cancellation_safety(to_core_cancellation_safety(definition.cancellation_safety))
    .with_allowed_callers(
        definition
            .allowed_callers
            .into_iter()
            .map(to_core_tool_caller),
    )
    .with_speculation_policy(to_core_speculation_policy(definition.speculation_policy))
    .with_issue_safety(to_core_issue_safety(definition.issue_safety))
    .with_execution_location(to_core_execution_location(definition.execution_location))
    .with_network_egress(to_core_network_egress(definition.network_egress));
    if let Some(output_schema) = definition.output_schema {
        tool = tool.with_output_schema(output_schema);
    }
    if let Some(concurrency_key) = definition.concurrency_key {
        tool = tool.with_concurrency_key(concurrency_key);
    }
    if let Some(expected_latency_ms) = definition.expected_latency_ms {
        tool = tool.with_expected_latency_ms(expected_latency_ms);
    }
    tool
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
            _ => WireToolRisk::High,
        },
        idempotent: definition.idempotent,
        read_only: definition.read_only,
        output_schema: definition.output_schema,
        parallel_safe: definition.parallel_safe,
        concurrency_key: definition.concurrency_key,
        cancellation_safety: to_wire_cancellation_safety(definition.cancellation_safety),
        expected_latency_ms: definition.expected_latency_ms,
        allowed_callers: definition
            .allowed_callers
            .into_iter()
            .map(to_wire_tool_caller)
            .collect(),
        speculation_policy: to_wire_speculation_policy(definition.speculation_policy),
        issue_safety: to_wire_issue_safety(definition.issue_safety),
        execution_location: to_wire_execution_location(definition.execution_location),
        network_egress: to_wire_network_egress(definition.network_egress),
    }
}
fn to_wire_model_info(info: llama_harness_core::ModelInfo) -> WireModelInfo {
    WireModelInfo {
        id: info.id,
        capabilities: llama_harness_protocol::ModelCapabilities {
            supports_tools: info.capabilities.supports_tools,
            supports_streaming: info.capabilities.supports_streaming,
            supports_structured_output: info.capabilities.supports_structured_output,
            supports_strict_tool_schemas: info.capabilities.supports_strict_tool_schemas,
            supports_streaming_tool_arguments: info.capabilities.supports_streaming_tool_arguments,
            supports_parallel_tool_calls: info.capabilities.supports_parallel_tool_calls,
            supports_structured_plans: info.capabilities.supports_structured_plans,
            // This sidecar does not configure a programmatic sandbox, so it
            // must not advertise provider support it cannot safely honor.
            supports_programmatic_calling: false,
            programmatic_conformance: None,
            limits: to_wire_provider_limits(info.capabilities.limits),
        },
    }
}
fn to_core_strategy(strategy: WireRunStrategy) -> RunStrategy {
    match strategy {
        WireRunStrategy::Adaptive => RunStrategy::Adaptive,
        WireRunStrategy::Direct => RunStrategy::Direct,
        WireRunStrategy::DeclarativePlan => RunStrategy::DeclarativePlan,
        WireRunStrategy::Programmatic => RunStrategy::Programmatic,
    }
}
fn to_core_cancellation_safety(value: WireCancellationSafety) -> CancellationSafety {
    match value {
        WireCancellationSafety::Unknown => CancellationSafety::Unknown,
        WireCancellationSafety::Cooperative => CancellationSafety::Cooperative,
        WireCancellationSafety::Guaranteed => CancellationSafety::Guaranteed,
    }
}
fn to_wire_cancellation_safety(value: CancellationSafety) -> WireCancellationSafety {
    match value {
        CancellationSafety::Unknown => WireCancellationSafety::Unknown,
        CancellationSafety::Cooperative => WireCancellationSafety::Cooperative,
        CancellationSafety::Guaranteed => WireCancellationSafety::Guaranteed,
        _ => WireCancellationSafety::Unknown,
    }
}
fn to_core_tool_caller(value: WireToolCaller) -> ToolCaller {
    match value {
        WireToolCaller::Direct => ToolCaller::Direct,
        WireToolCaller::DeclarativePlan => ToolCaller::DeclarativePlan,
        WireToolCaller::Programmatic => ToolCaller::Programmatic,
        WireToolCaller::Speculative => ToolCaller::Speculative,
    }
}
fn to_wire_tool_caller(value: ToolCaller) -> WireToolCaller {
    match value {
        ToolCaller::Direct => WireToolCaller::Direct,
        ToolCaller::DeclarativePlan => WireToolCaller::DeclarativePlan,
        ToolCaller::Programmatic => WireToolCaller::Programmatic,
        ToolCaller::Speculative => WireToolCaller::Speculative,
        _ => WireToolCaller::Direct,
    }
}
fn to_core_speculation_policy(value: WireSpeculationPolicy) -> SpeculationPolicy {
    match value {
        WireSpeculationPolicy::Disabled => SpeculationPolicy::Disabled,
        WireSpeculationPolicy::Enabled => SpeculationPolicy::Enabled,
    }
}
fn to_wire_speculation_policy(value: SpeculationPolicy) -> WireSpeculationPolicy {
    match value {
        SpeculationPolicy::Disabled => WireSpeculationPolicy::Disabled,
        SpeculationPolicy::Enabled => WireSpeculationPolicy::Enabled,
        _ => WireSpeculationPolicy::Disabled,
    }
}
fn to_core_issue_safety(value: WireIssueSafety) -> IssueSafety {
    match value {
        WireIssueSafety::Unknown => IssueSafety::Unknown,
        WireIssueSafety::Guaranteed => IssueSafety::Guaranteed,
    }
}
fn to_wire_issue_safety(value: IssueSafety) -> WireIssueSafety {
    match value {
        IssueSafety::Unknown => WireIssueSafety::Unknown,
        IssueSafety::Guaranteed => WireIssueSafety::Guaranteed,
        _ => WireIssueSafety::Unknown,
    }
}
fn to_core_execution_location(value: WireExecutionLocation) -> ExecutionLocation {
    match value {
        WireExecutionLocation::Unknown => ExecutionLocation::Unknown,
        WireExecutionLocation::LocalPrivate => ExecutionLocation::LocalPrivate,
        WireExecutionLocation::Remote => ExecutionLocation::Remote,
    }
}
fn to_wire_execution_location(value: ExecutionLocation) -> WireExecutionLocation {
    match value {
        ExecutionLocation::Unknown => WireExecutionLocation::Unknown,
        ExecutionLocation::LocalPrivate => WireExecutionLocation::LocalPrivate,
        ExecutionLocation::Remote => WireExecutionLocation::Remote,
        _ => WireExecutionLocation::Unknown,
    }
}
fn to_core_network_egress(value: WireNetworkEgress) -> NetworkEgress {
    match value {
        WireNetworkEgress::Unknown => NetworkEgress::Unknown,
        WireNetworkEgress::Prohibited => NetworkEgress::Prohibited,
        WireNetworkEgress::Permitted => NetworkEgress::Permitted,
    }
}
fn to_wire_network_egress(value: NetworkEgress) -> WireNetworkEgress {
    match value {
        NetworkEgress::Unknown => WireNetworkEgress::Unknown,
        NetworkEgress::Prohibited => WireNetworkEgress::Prohibited,
        NetworkEgress::Permitted => WireNetworkEgress::Permitted,
        _ => WireNetworkEgress::Unknown,
    }
}
fn to_wire_provider_limits(
    limits: llama_harness_core::ProviderCapabilityLimits,
) -> WireProviderCapabilityLimits {
    WireProviderCapabilityLimits {
        max_tools: limits.max_tools,
        max_tool_schema_bytes: limits.max_tool_schema_bytes,
        max_parallel_tool_calls: limits.max_parallel_tool_calls,
        max_streamed_argument_bytes: limits.max_streamed_argument_bytes,
        max_streamed_tool_calls: limits.max_streamed_tool_calls,
        max_plan_bytes: limits.max_plan_bytes,
        max_plan_nodes: limits.max_plan_nodes,
        max_program_bytes: limits.max_program_bytes,
    }
}
fn to_wire_strategy(strategy: RunStrategy) -> WireRunStrategy {
    match strategy {
        RunStrategy::Adaptive => WireRunStrategy::Adaptive,
        RunStrategy::Direct => WireRunStrategy::Direct,
        RunStrategy::DeclarativePlan => WireRunStrategy::DeclarativePlan,
        RunStrategy::Programmatic => WireRunStrategy::Programmatic,
    }
}
fn to_wire_discovery_outcome(
    outcome: llama_harness_core::ToolDiscoveryOutcome,
) -> WireToolDiscoveryOutcome {
    match outcome {
        llama_harness_core::ToolDiscoveryOutcome::Selected => WireToolDiscoveryOutcome::Selected,
        llama_harness_core::ToolDiscoveryOutcome::LimitReached => {
            WireToolDiscoveryOutcome::LimitReached
        }
        _ => WireToolDiscoveryOutcome::Selected,
    }
}
fn to_wire_discovery_selection(
    selection: llama_harness_core::ToolDiscoverySelection,
) -> WireToolDiscoverySelection {
    match selection {
        llama_harness_core::ToolDiscoverySelection::LegacyUnclassified => {
            WireToolDiscoverySelection::LegacyUnclassified
        }
        llama_harness_core::ToolDiscoverySelection::EmptyCatalog => {
            WireToolDiscoverySelection::EmptyCatalog
        }
        llama_harness_core::ToolDiscoverySelection::NoCapacity => {
            WireToolDiscoverySelection::NoCapacity
        }
        llama_harness_core::ToolDiscoverySelection::FullCatalog => {
            WireToolDiscoverySelection::FullCatalog
        }
        llama_harness_core::ToolDiscoverySelection::HotOnly => WireToolDiscoverySelection::HotOnly,
        llama_harness_core::ToolDiscoverySelection::Exact => WireToolDiscoverySelection::Exact,
        llama_harness_core::ToolDiscoverySelection::LexicalConfident => {
            WireToolDiscoverySelection::LexicalConfident
        }
        llama_harness_core::ToolDiscoverySelection::LexicalExpanded => {
            WireToolDiscoverySelection::LexicalExpanded
        }
        llama_harness_core::ToolDiscoverySelection::NoMatch => WireToolDiscoverySelection::NoMatch,
        llama_harness_core::ToolDiscoverySelection::CountLimit => {
            WireToolDiscoverySelection::CountLimit
        }
        llama_harness_core::ToolDiscoverySelection::SchemaByteLimit => {
            WireToolDiscoverySelection::SchemaByteLimit
        }
        _ => WireToolDiscoverySelection::LegacyUnclassified,
    }
}
fn to_wire_strategy_selection_reason(
    reason: llama_harness_core::StrategySelectionReason,
) -> WireStrategySelectionReason {
    match reason {
        llama_harness_core::StrategySelectionReason::Forced => WireStrategySelectionReason::Forced,
        llama_harness_core::StrategySelectionReason::AdaptivePlanner => {
            WireStrategySelectionReason::AdaptivePlanner
        }
        llama_harness_core::StrategySelectionReason::PlannerSelectedDirect => {
            WireStrategySelectionReason::PlannerSelectedDirect
        }
        llama_harness_core::StrategySelectionReason::PlannerSelectedPlan => {
            WireStrategySelectionReason::PlannerSelectedPlan
        }
        llama_harness_core::StrategySelectionReason::CapabilityDowngrade => {
            WireStrategySelectionReason::CapabilityDowngrade
        }
        llama_harness_core::StrategySelectionReason::Fallback => {
            WireStrategySelectionReason::Fallback
        }
        _ => WireStrategySelectionReason::CapabilityDowngrade,
    }
}
fn to_wire_strategy_fallback_reason(
    reason: llama_harness_core::StrategyFallbackReason,
) -> WireStrategyFallbackReason {
    match reason {
        llama_harness_core::StrategyFallbackReason::UnsupportedCapability => {
            WireStrategyFallbackReason::UnsupportedCapability
        }
        llama_harness_core::StrategyFallbackReason::InvalidPlan => {
            WireStrategyFallbackReason::InvalidPlan
        }
        llama_harness_core::StrategyFallbackReason::InvalidProgram => {
            WireStrategyFallbackReason::InvalidProgram
        }
        llama_harness_core::StrategyFallbackReason::ExecutionRecovery => {
            WireStrategyFallbackReason::ExecutionRecovery
        }
        llama_harness_core::StrategyFallbackReason::PlannerFailure => {
            WireStrategyFallbackReason::PlannerFailure
        }
        _ => WireStrategyFallbackReason::PlannerFailure,
    }
}
fn to_wire_program_lifecycle_outcome(
    outcome: llama_harness_core::ProgramLifecycleOutcome,
) -> WireProgramLifecycleOutcome {
    match outcome {
        llama_harness_core::ProgramLifecycleOutcome::Started => {
            WireProgramLifecycleOutcome::Started
        }
        llama_harness_core::ProgramLifecycleOutcome::Validated => {
            WireProgramLifecycleOutcome::Validated
        }
        llama_harness_core::ProgramLifecycleOutcome::Invalid => {
            WireProgramLifecycleOutcome::Invalid
        }
        llama_harness_core::ProgramLifecycleOutcome::Succeeded => {
            WireProgramLifecycleOutcome::Succeeded
        }
        llama_harness_core::ProgramLifecycleOutcome::Fallback => {
            WireProgramLifecycleOutcome::Fallback
        }
        llama_harness_core::ProgramLifecycleOutcome::Failed => WireProgramLifecycleOutcome::Failed,
        llama_harness_core::ProgramLifecycleOutcome::Cancelled => {
            WireProgramLifecycleOutcome::Cancelled
        }
        llama_harness_core::ProgramLifecycleOutcome::TimedOut => {
            WireProgramLifecycleOutcome::TimedOut
        }
        llama_harness_core::ProgramLifecycleOutcome::LimitReached => {
            WireProgramLifecycleOutcome::LimitReached
        }
        _ => WireProgramLifecycleOutcome::Failed,
    }
}
fn to_wire_plan_phase(phase: llama_harness_core::PlanPhase) -> WirePlanPhase {
    match phase {
        llama_harness_core::PlanPhase::Planning => WirePlanPhase::Planning,
        llama_harness_core::PlanPhase::Repair => WirePlanPhase::Repair,
        llama_harness_core::PlanPhase::Validation => WirePlanPhase::Validation,
        llama_harness_core::PlanPhase::Preflight => WirePlanPhase::Preflight,
        llama_harness_core::PlanPhase::Recovery => WirePlanPhase::Recovery,
        _ => WirePlanPhase::Validation,
    }
}
fn to_wire_plan_lifecycle_outcome(
    outcome: llama_harness_core::PlanLifecycleOutcome,
) -> WirePlanLifecycleOutcome {
    match outcome {
        llama_harness_core::PlanLifecycleOutcome::Started => WirePlanLifecycleOutcome::Started,
        llama_harness_core::PlanLifecycleOutcome::Succeeded => WirePlanLifecycleOutcome::Succeeded,
        llama_harness_core::PlanLifecycleOutcome::Invalid => WirePlanLifecycleOutcome::Invalid,
        llama_harness_core::PlanLifecycleOutcome::Rejected => WirePlanLifecycleOutcome::Rejected,
        llama_harness_core::PlanLifecycleOutcome::Failed => WirePlanLifecycleOutcome::Failed,
        llama_harness_core::PlanLifecycleOutcome::Cancelled => WirePlanLifecycleOutcome::Cancelled,
        llama_harness_core::PlanLifecycleOutcome::TimedOut => WirePlanLifecycleOutcome::TimedOut,
        llama_harness_core::PlanLifecycleOutcome::LimitReached => {
            WirePlanLifecycleOutcome::LimitReached
        }
        llama_harness_core::PlanLifecycleOutcome::Skipped => WirePlanLifecycleOutcome::Skipped,
        _ => WirePlanLifecycleOutcome::Failed,
    }
}
fn to_wire_plan_node_outcome(outcome: llama_harness_core::PlanNodeOutcome) -> WirePlanNodeOutcome {
    match outcome {
        llama_harness_core::PlanNodeOutcome::Succeeded => WirePlanNodeOutcome::Succeeded,
        llama_harness_core::PlanNodeOutcome::Failed => WirePlanNodeOutcome::Failed,
        llama_harness_core::PlanNodeOutcome::Cancelled => WirePlanNodeOutcome::Cancelled,
        llama_harness_core::PlanNodeOutcome::TimedOut => WirePlanNodeOutcome::TimedOut,
        llama_harness_core::PlanNodeOutcome::Rejected => WirePlanNodeOutcome::Rejected,
        llama_harness_core::PlanNodeOutcome::LimitReached => WirePlanNodeOutcome::LimitReached,
        llama_harness_core::PlanNodeOutcome::Reused => WirePlanNodeOutcome::Reused,
        _ => WirePlanNodeOutcome::Failed,
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
fn to_wire_event(event: RunEvent) -> Option<llama_harness_protocol::WireRunEvent> {
    Some(match event {
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
        RunEvent::ToolDiscoveryCompleted {
            caller,
            outcome,
            selection,
            candidate_count,
            selected_count,
            deferred_candidate_count,
            effective_tool_count_budget,
            effective_schema_byte_budget,
            selected_schema_bytes,
            expansion_count,
            expansion_limit,
            catalog_exceeded_budget,
            duration_ms,
        } => llama_harness_protocol::WireRunEvent::ToolDiscoveryCompleted {
            caller: to_wire_tool_caller(caller),
            outcome: to_wire_discovery_outcome(outcome),
            selection: to_wire_discovery_selection(selection),
            candidate_count,
            selected_count,
            deferred_candidate_count,
            effective_tool_count_budget,
            effective_schema_byte_budget,
            selected_schema_bytes,
            expansion_count,
            expansion_limit,
            catalog_exceeded_budget,
            duration_ms,
        },
        RunEvent::StrategySelected {
            requested,
            selected,
            reason,
        } => llama_harness_protocol::WireRunEvent::StrategySelected {
            requested: to_wire_strategy(requested),
            selected: to_wire_strategy(selected),
            reason: to_wire_strategy_selection_reason(reason),
        },
        RunEvent::StrategyFallback { from, to, reason } => {
            llama_harness_protocol::WireRunEvent::StrategyFallback {
                from: to_wire_strategy(from),
                to: to_wire_strategy(to),
                reason: to_wire_strategy_fallback_reason(reason),
            }
        }
        RunEvent::PlanLifecycle {
            phase,
            attempt,
            outcome,
        } => llama_harness_protocol::WireRunEvent::PlanLifecycle {
            phase: to_wire_plan_phase(phase),
            attempt,
            outcome: to_wire_plan_lifecycle_outcome(outcome),
        },
        RunEvent::PlanValidated {
            attempt,
            node_count,
        } => llama_harness_protocol::WireRunEvent::PlanValidated {
            attempt,
            node_count,
        },
        RunEvent::ProgramLifecycle { attempt, outcome } => {
            llama_harness_protocol::WireRunEvent::ProgramLifecycle {
                attempt,
                outcome: to_wire_program_lifecycle_outcome(outcome),
            }
        }
        RunEvent::ProgramValidated {
            attempt,
            statement_count,
            instruction_count,
        } => llama_harness_protocol::WireRunEvent::ProgramValidated {
            attempt,
            statement_count,
            instruction_count,
        },
        RunEvent::ProgramExecutionCompleted {
            attempt,
            fuel_used,
            scheduling_slices,
            tool_yields,
            branches,
            loop_iterations,
            fanout_batches,
            partial_failures,
            peak_accounted_bytes,
            duration_ms,
        } => llama_harness_protocol::WireRunEvent::ProgramExecutionCompleted {
            attempt,
            fuel_used,
            scheduling_slices,
            tool_yields,
            branches,
            loop_iterations,
            fanout_batches,
            partial_failures,
            peak_accounted_bytes,
            duration_ms,
        },
        RunEvent::PlanNodeStarted {
            node_id,
            tool_id,
            attempt,
            wave,
        } => llama_harness_protocol::WireRunEvent::PlanNodeStarted {
            node_id,
            tool_id,
            attempt,
            wave,
        },
        RunEvent::PlanNodeCompleted {
            node_id,
            tool_id,
            attempt,
            wave,
            ok,
            outcome,
            duration_ms,
        } => llama_harness_protocol::WireRunEvent::PlanNodeCompleted {
            node_id,
            tool_id,
            attempt,
            wave,
            ok,
            outcome: to_wire_plan_node_outcome(outcome),
            duration_ms,
        },
        RunEvent::ToolEffectReused { call_id, tool_id } => {
            llama_harness_protocol::WireRunEvent::ToolEffectReused { call_id, tool_id }
        }
        RunEvent::StrategyUsage {
            strategy,
            model_calls,
            planning_model_calls,
            repair_model_calls,
            recovery_model_calls,
            final_synthesis_model_calls,
            reactive_model_calls,
            tool_calls,
            tool_issued,
            tool_reused,
            tool_rejected,
            tool_pre_dispatch_aborted,
            tool_completed,
            tool_failed,
            tool_cancelled,
            duration_ms,
        } => llama_harness_protocol::WireRunEvent::StrategyUsage {
            strategy: to_wire_strategy(strategy),
            model_calls,
            planning_model_calls,
            repair_model_calls,
            recovery_model_calls,
            final_synthesis_model_calls,
            reactive_model_calls,
            tool_calls,
            tool_issued,
            tool_reused,
            tool_rejected,
            tool_pre_dispatch_aborted,
            tool_completed,
            tool_failed,
            tool_cancelled,
            duration_ms,
        },
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
        _ => return None,
    })
}
fn to_wire_policy(decision: PolicyDecision) -> WirePolicyDecision {
    match decision {
        PolicyDecision::Allow { reason } => WirePolicyDecision::Allow { reason },
        PolicyDecision::Deny { reason } => WirePolicyDecision::Deny { reason },
        PolicyDecision::RequireApproval { reason } => {
            WirePolicyDecision::RequireApproval { reason }
        }
        _ => WirePolicyDecision::Deny {
            reason: "runtime does not support this policy decision".into(),
        },
    }
}
fn to_wire_status(status: llama_harness_core::RunStatus) -> WireRunStatus {
    match status {
        llama_harness_core::RunStatus::Completed => WireRunStatus::Completed,
        llama_harness_core::RunStatus::Failed => WireRunStatus::Failed,
        llama_harness_core::RunStatus::Cancelled => WireRunStatus::Cancelled,
        llama_harness_core::RunStatus::LimitReached => WireRunStatus::LimitReached,
        _ => WireRunStatus::Failed,
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

#[cfg(test)]
mod tests {
    use super::*;
    use llama_harness_core::{
        mock::{final_response, tool_response, MockModelProvider},
        RunStatus, ToolCall,
    };
    use serde_json::{json, Value};
    use std::sync::atomic::AtomicU32;

    #[test]
    fn core_limit_mapping_preserves_wire_fields_and_defaults_programmatic_only_limits() {
        let limits = to_core_limits(WireAgentLimits {
            max_model_calls: 99,
            max_tool_calls: 99,
            max_identical_tool_calls: 99,
            max_run_duration_ms: Some(999_999),
            max_model_call_duration_ms: Some(999_999),
            max_output_repairs: 99,
            max_provider_retries: 99,
            max_input_bytes: 999_999,
            max_request_payload_bytes: 999_999,
            max_model_response_bytes: 9_999_999,
            max_tool_arguments_bytes: 999_999,
            max_tool_result_bytes: 9_999_999,
            max_transcript_bytes: 99_999_999,
            max_json_depth: 999,
        });

        assert_eq!(limits.max_model_calls, 32);
        assert_eq!(limits.max_tool_calls, 64);
        assert_eq!(limits.max_identical_tool_calls, 8);
        assert_eq!(limits.max_run_duration_ms, Some(300_000));
        assert_eq!(limits.max_model_call_duration_ms, Some(120_000));
        assert_eq!(limits.max_output_repairs, 4);
        assert_eq!(limits.max_provider_retries, 4);
        assert_eq!(limits.max_input_bytes, 64 * 1024);
        assert_eq!(limits.max_request_payload_bytes, 256 * 1024);
        assert_eq!(limits.max_model_response_bytes, 1024 * 1024);
        assert_eq!(limits.max_tool_arguments_bytes, 64 * 1024);
        assert_eq!(limits.max_tool_result_bytes, 1024 * 1024);
        assert_eq!(limits.max_transcript_bytes, 4 * 1024 * 1024);
        assert_eq!(limits.max_json_depth, 64);
        assert_eq!(
            limits.max_programmatic_program_bytes,
            AgentLimits::default().max_programmatic_program_bytes
        );
        assert_eq!(
            limits.max_programmatic_fanout_concurrency,
            AgentLimits::default().max_programmatic_fanout_concurrency
        );
    }

    struct CountingTool {
        definition: ToolDefinition,
        calls: AtomicU32,
    }

    #[async_trait]
    impl Tool for CountingTool {
        fn definition(&self) -> &ToolDefinition {
            &self.definition
        }

        async fn execute(
            &self,
            _: Value,
            _: CancellationToken,
        ) -> Result<ToolResult, HarnessError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult::success(Value::Null))
        }
    }

    #[tokio::test]
    async fn wire_result_does_not_restore_rejected_argument_values() {
        const SECRET: &str = "sentinel-runtime-argument-secret";
        let tool = Arc::new(CountingTool {
            definition: ToolDefinition::new(
                "read",
                "Read",
                "Read",
                json!({
                    "type":"object",
                    "required":["key"],
                    "properties":{"key":{"type":"string","enum":["allowed"]}},
                    "additionalProperties":false
                }),
            )
            .with_risk(ToolRisk::Low)
            .with_read_only(true)
            .with_idempotent(true),
            calls: AtomicU32::new(0),
        });
        let mut tools = ToolRegistry::default();
        tools.register(tool.clone()).unwrap();
        let provider = Arc::new(MockModelProvider::scripted([
            tool_response(ToolCall::new(
                "invalid-secret",
                "read",
                format!(r#"{{"key":"{SECRET}"}}"#),
            )),
            final_response("recovered"),
        ]));
        let mut agent = AgentDefinition::new("runtime-test", "Runtime test", "1", "mock-model");
        agent.tool_allowlist = vec!["read".into()];

        let result = AgentRunner::builder(provider)
            .tools(tools)
            .build()
            .run(RunRequest::new(agent, "read"))
            .await
            .unwrap();

        assert_eq!(result.status, RunStatus::Completed);
        assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
        assert!(!serde_json::to_string(&result).unwrap().contains(SECRET));
        let wire = to_wire_result(result);
        assert_eq!(wire.tool_calls.len(), 1);
        assert_eq!(wire.tool_calls[0].id, "invalid-secret");
        assert_eq!(wire.tool_calls[0].tool_id, "read");
        assert_eq!(wire.tool_calls[0].arguments_json, "{}");
        assert!(!serde_json::to_string(&wire).unwrap().contains(SECRET));
    }

    #[test]
    fn v1_1_tool_and_model_conversions_preserve_public_advanced_metadata() {
        let definition =
            ToolDefinition::new("tool", "Tool", "description", json!({"type":"object"}))
                .with_risk(ToolRisk::Low)
                .with_idempotent(true)
                .with_read_only(true)
                .with_output_schema(json!({"type":"string"}))
                .with_parallel_safe(true)
                .with_concurrency_key("shared")
                .with_cancellation_safety(CancellationSafety::Guaranteed)
                .with_expected_latency_ms(12)
                .with_allowed_callers([ToolCaller::Direct, ToolCaller::DeclarativePlan])
                .with_speculation_policy(SpeculationPolicy::Enabled)
                .with_issue_safety(IssueSafety::Guaranteed)
                .with_execution_location(ExecutionLocation::LocalPrivate)
                .with_network_egress(NetworkEgress::Prohibited);
        let wire = to_wire_tool_definition(definition.clone());
        assert_eq!(wire.output_schema, Some(json!({"type":"string"})));
        assert!(wire.parallel_safe);
        assert_eq!(wire.concurrency_key.as_deref(), Some("shared"));
        assert_eq!(wire.cancellation_safety, WireCancellationSafety::Guaranteed);
        assert_eq!(wire.expected_latency_ms, Some(12));
        assert!(wire
            .allowed_callers
            .contains(&WireToolCaller::DeclarativePlan));
        assert_eq!(wire.speculation_policy, WireSpeculationPolicy::Enabled);
        assert_eq!(wire.issue_safety, WireIssueSafety::Guaranteed);
        assert_eq!(wire.execution_location, WireExecutionLocation::LocalPrivate);
        assert_eq!(wire.network_egress, WireNetworkEgress::Prohibited);
        assert_eq!(to_core_tool_definition(wire), definition);

        let limits = llama_harness_core::ProviderCapabilityLimits::new()
            .with_max_tools(3)
            .with_max_tool_schema_bytes(4)
            .with_max_parallel_tool_calls(5)
            .with_max_streamed_argument_bytes(6)
            .with_max_streamed_tool_calls(7)
            .with_max_plan_bytes(8)
            .with_max_plan_nodes(9)
            .with_max_program_bytes(10);
        let capabilities = llama_harness_core::ModelCapabilities::new(true, true, true)
            .with_strict_tool_schemas(true)
            .with_streaming_tool_arguments(true)
            .with_parallel_tool_calls(true)
            .with_structured_plans(true)
            .with_programmatic_conformance(
                llama_harness_core::ProgrammaticConformance::StrictJsonAstV1,
            )
            .with_limits(limits);
        let wire_model = to_wire_model_info(
            llama_harness_core::ModelInfo::new("model").with_capabilities(capabilities),
        );
        assert!(wire_model.capabilities.supports_strict_tool_schemas);
        assert!(wire_model.capabilities.supports_streaming_tool_arguments);
        assert!(wire_model.capabilities.supports_parallel_tool_calls);
        assert!(wire_model.capabilities.supports_structured_plans);
        assert!(!wire_model.capabilities.supports_programmatic_calling);
        assert_eq!(wire_model.capabilities.programmatic_conformance, None);
        assert_eq!(wire_model.capabilities.limits.max_program_bytes, Some(10));
    }

    #[test]
    fn v1_0_projection_filters_additive_events_and_tool_metadata() {
        let mut event = ProtocolMessage::RunEvent(RunEventPayload {
            trace_id: "trace".into(),
            sequence: 1,
            timestamp_ms: 0,
            event: to_wire_event(RunEvent::StrategySelected {
                requested: RunStrategy::Adaptive,
                selected: RunStrategy::Direct,
                reason: llama_harness_core::StrategySelectionReason::CapabilityDowngrade,
            })
            .unwrap(),
        });
        assert!(!project_v1_0(&mut event));
        assert!(matches!(
            to_wire_event(RunEvent::StrategySelected {
                requested: RunStrategy::Adaptive,
                selected: RunStrategy::Direct,
                reason: llama_harness_core::StrategySelectionReason::Fallback,
            }),
            Some(llama_harness_protocol::WireRunEvent::StrategySelected {
                reason: WireStrategySelectionReason::Fallback,
                ..
            })
        ));

        let mut tool = to_wire_tool_definition(
            ToolDefinition::new("tool", "Tool", "description", json!({}))
                .with_output_schema(json!({"type":"string"}))
                .with_parallel_safe(true)
                .with_cancellation_safety(CancellationSafety::Guaranteed),
        );
        project_v1_0_tool(&mut tool);
        assert_eq!(tool.output_schema, None);
        assert!(!tool.parallel_safe);
        assert_eq!(tool.cancellation_safety, WireCancellationSafety::Unknown);
    }
}
