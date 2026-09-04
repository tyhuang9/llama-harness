use async_trait::async_trait;
use futures_util::{stream, Stream};
use llama_harness_core::{
    AgentDefinition, AgentRunner, AllowAllPolicy, ApprovalHandler, ApprovalRecord,
    CancellationSafety, EventRecord, EventSink, ExecutionLocation, HarnessError, InMemoryEventSink,
    IssueSafety, MessageRole, ModelCapabilities, ModelEventStream, ModelInfo, ModelProvider,
    ModelRequest, ModelResponse, ModelStreamEvent, NetworkEgress, PolicyDecision, PolicyEngine,
    ProviderCapabilityLimits, ProviderHealth, RunEvent, RunRequest, RunStatus, RunStrategy,
    SpeculationConfig, SpeculationMode, SpeculationPolicy, Tool, ToolCallContext, ToolCallDelta,
    ToolCaller, ToolDefinition, ToolRegistry, ToolResult, ToolRisk, Usage,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::{
    collections::VecDeque,
    pin::Pin,
    sync::Condvar,
    task::{Context, Poll},
    thread,
    time::Duration,
};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

const TOOL_ID: &str = "local.read";
const PRIVACY_CANARY: &str = "speculation-private-canary";

#[derive(Clone, Copy)]
enum StreamBehavior {
    Normal,
    RetryableBeforeCandidate,
    FailAfterCandidate,
    EmptyFlood,
    InterleavedMultiple,
    PartialPaused,
    OversizedText,
    OversizedModel,
    InvalidResponse,
    HugeArguments,
    CancellationAwareOversizedText,
    AbortAwarePendingStart,
    AbortAwarePendingTail,
    PullDrivenDelayed,
}

struct StreamingProvider {
    complete_calls: AtomicUsize,
    stream_calls: AtomicUsize,
    behavior: Mutex<StreamBehavior>,
    planner_response: Option<&'static str>,
    partial_emitted: Arc<Semaphore>,
    release_final: Arc<Semaphore>,
    stream_drop_saw_cancellation: Arc<AtomicBool>,
    stream_start_entered: Arc<Semaphore>,
    stream_start_cancellation: Mutex<Option<CancellationToken>>,
    response_delay_ms: AtomicU64,
    observe_tail: AtomicBool,
    tail_polled: Arc<Semaphore>,
}

impl StreamingProvider {
    fn new(behavior: StreamBehavior) -> Self {
        Self {
            complete_calls: AtomicUsize::new(0),
            stream_calls: AtomicUsize::new(0),
            behavior: Mutex::new(behavior),
            planner_response: None,
            partial_emitted: Arc::new(Semaphore::new(0)),
            release_final: Arc::new(Semaphore::new(0)),
            stream_drop_saw_cancellation: Arc::new(AtomicBool::new(false)),
            stream_start_entered: Arc::new(Semaphore::new(0)),
            stream_start_cancellation: Mutex::new(None),
            response_delay_ms: AtomicU64::new(0),
            observe_tail: AtomicBool::new(false),
            tail_polled: Arc::new(Semaphore::new(0)),
        }
    }

    fn adaptive_direct() -> Self {
        Self {
            planner_response: Some(r#"{"strategy":"direct"}"#),
            ..Self::new(StreamBehavior::Normal)
        }
    }

    fn adaptive_invalid_plan() -> Self {
        Self {
            planner_response: Some("not-json"),
            ..Self::new(StreamBehavior::Normal)
        }
    }

    fn declarative() -> Self {
        Self {
            planner_response: Some(
                r#"{"strategy":"declarative_plan","plan":{"nodes":[{"id":"read","tool_id":"local.read","arguments":{"query":"status"}}]}}"#,
            ),
            ..Self::new(StreamBehavior::Normal)
        }
    }

    fn set_behavior(&self, behavior: StreamBehavior) {
        *self.behavior.lock().expect("behavior lock") = behavior;
    }

    fn set_response_delay(&self, delay_ms: u64, observe_tail: bool) {
        self.response_delay_ms.store(delay_ms, Ordering::SeqCst);
        self.observe_tail.store(observe_tail, Ordering::SeqCst);
    }

    fn requests_tool(request: &ModelRequest) -> bool {
        !request
            .messages
            .iter()
            .any(|message| message.role == MessageRole::Tool)
    }

    fn tool_response(model: String) -> ModelResponse {
        ModelResponse::new(model).with_tool_calls(vec![llama_harness_core::ToolCall::new(
            "call-0",
            TOOL_ID,
            r#"{"query":"status"}"#,
        )])
    }
}

struct CancellationCheckingStream {
    events: VecDeque<Result<ModelStreamEvent, HarnessError>>,
    cancellation: CancellationToken,
    observed: Arc<AtomicBool>,
}

impl Stream for CancellationCheckingStream {
    type Item = Result<ModelStreamEvent, HarnessError>;

    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.events.pop_front())
    }
}

impl Drop for CancellationCheckingStream {
    fn drop(&mut self) {
        self.observed
            .store(self.cancellation.is_cancelled(), Ordering::SeqCst);
    }
}

struct AbortCheckingPendingStream {
    first: Option<Result<ModelStreamEvent, HarnessError>>,
    cancellation: CancellationToken,
    observed: Arc<AtomicBool>,
    tail_polled: Arc<Semaphore>,
    signalled_tail: bool,
}

impl Stream for AbortCheckingPendingStream {
    type Item = Result<ModelStreamEvent, HarnessError>;

    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(first) = self.first.take() {
            return Poll::Ready(Some(first));
        }
        if !self.signalled_tail {
            self.signalled_tail = true;
            self.tail_polled.add_permits(1);
        }
        Poll::Pending
    }
}

impl Drop for AbortCheckingPendingStream {
    fn drop(&mut self) {
        self.observed
            .store(self.cancellation.is_cancelled(), Ordering::SeqCst);
    }
}

#[async_trait]
impl ModelProvider for StreamingProvider {
    fn id(&self) -> &str {
        "streaming-test"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::new(true, true, self.planner_response.is_some())
            .with_streaming_tool_arguments(true)
            .with_parallel_tool_calls(true)
            .with_structured_plans(self.planner_response.is_some())
            .with_limits(
                ProviderCapabilityLimits::new()
                    .with_max_parallel_tool_calls(2)
                    .with_max_plan_nodes(64)
                    .with_max_plan_bytes(256 * 1024),
            )
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth::healthy())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(vec![ModelInfo::new("test-model")])
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError> {
        self.complete_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(response) = self.planner_response.filter(|_| {
            request.structured_output.as_ref().is_some_and(|output| {
                matches!(
                    output.name.as_str(),
                    "llama_harness_planner_envelope_v1" | "llama_harness_declarative_plan_v1"
                )
            })
        }) {
            return Ok(ModelResponse::new(request.model).with_final_output(response));
        }
        let requests_tool = Self::requests_tool(&request);
        if requests_tool {
            let delay_ms = self.response_delay_ms.load(Ordering::SeqCst);
            if delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }
        Ok(if requests_tool {
            Self::tool_response(request.model)
        } else {
            ModelResponse::new(request.model).with_final_output("done")
        })
    }

    async fn stream(&self, request: ModelRequest) -> Result<ModelEventStream, HarnessError> {
        let stream_ordinal = self.stream_calls.fetch_add(1, Ordering::SeqCst);
        let requests_tool = Self::requests_tool(&request);
        let model_cancellation = request.cancellation.clone();
        let behavior = *self.behavior.lock().expect("behavior lock");
        if matches!(behavior, StreamBehavior::RetryableBeforeCandidate) && stream_ordinal == 0 {
            return Err(HarnessError::RetryableProvider(
                "retry before candidate".into(),
            ));
        }
        let pending_start = matches!(behavior, StreamBehavior::AbortAwarePendingStart);
        if pending_start {
            *self
                .stream_start_cancellation
                .lock()
                .expect("stream start cancellation lock") = Some(model_cancellation.clone());
            self.stream_start_entered.add_permits(1);
            std::future::pending::<()>().await;
            unreachable!("pending stream startup cannot complete");
        }
        let model = request.model;
        let events = if !requests_tool {
            vec![
                Ok(ModelStreamEvent::TextDelta {
                    content: "done".into(),
                }),
                Ok(ModelStreamEvent::Completed {
                    model,
                    usage: Usage::default(),
                }),
            ]
        } else {
            match behavior {
                StreamBehavior::Normal => vec![
                    Ok(ModelStreamEvent::ToolCallDelta(
                        ToolCallDelta::new(0, r#"{"query":"status"}"#, true)
                            .with_call_id("call-0")
                            .with_tool_id(TOOL_ID),
                    )),
                    Ok(ModelStreamEvent::Completed {
                        model,
                        usage: Usage::default(),
                    }),
                ],
                StreamBehavior::RetryableBeforeCandidate => vec![Err(
                    HarnessError::RetryableProvider("retry before candidate".into()),
                )],
                StreamBehavior::FailAfterCandidate => vec![
                    Ok(ModelStreamEvent::ToolCallDelta(
                        ToolCallDelta::new(0, r#"{"query":"status"}"#, true)
                            .with_call_id("call-0")
                            .with_tool_id(TOOL_ID),
                    )),
                    Err(HarnessError::RetryableProvider(PRIVACY_CANARY.into())),
                ],
                StreamBehavior::EmptyFlood => vec![
                    Ok(ModelStreamEvent::TextDelta {
                        content: String::new(),
                    }),
                    Ok(ModelStreamEvent::TextDelta {
                        content: String::new(),
                    }),
                    Ok(ModelStreamEvent::TextDelta {
                        content: String::new(),
                    }),
                    Ok(ModelStreamEvent::Completed {
                        model,
                        usage: Usage::default(),
                    }),
                ],
                StreamBehavior::InterleavedMultiple => vec![
                    Ok(ModelStreamEvent::ToolCallDelta(
                        ToolCallDelta::new(1, r#"{"query":"second"}"#, true)
                            .with_call_id("call-1")
                            .with_tool_id(TOOL_ID),
                    )),
                    Ok(ModelStreamEvent::ToolCallDelta(
                        ToolCallDelta::new(0, r#"{"query":"first"}"#, true)
                            .with_call_id("call-0")
                            .with_tool_id(TOOL_ID),
                    )),
                    Ok(ModelStreamEvent::Completed {
                        model,
                        usage: Usage::default(),
                    }),
                ],
                StreamBehavior::PartialPaused => {
                    let partial_emitted = Arc::clone(&self.partial_emitted);
                    let release_final = Arc::clone(&self.release_final);
                    let events =
                        futures_util::stream::unfold((0_u8, model), move |(state, model)| {
                            let partial_emitted = Arc::clone(&partial_emitted);
                            let release_final = Arc::clone(&release_final);
                            async move {
                                match state {
                                    0 => {
                                        partial_emitted.add_permits(1);
                                        Some((
                                            Ok(ModelStreamEvent::ToolCallDelta(
                                                ToolCallDelta::new(0, "{\"query\":\"", false)
                                                    .with_call_id("call-0")
                                                    .with_tool_id(TOOL_ID),
                                            )),
                                            (1, model),
                                        ))
                                    }
                                    1 => {
                                        let permit = release_final
                                            .acquire()
                                            .await
                                            .expect("final release semaphore open");
                                        permit.forget();
                                        Some((
                                            Ok(ModelStreamEvent::ToolCallDelta(
                                                ToolCallDelta::new(0, r#"status"}"#, true),
                                            )),
                                            (2, model),
                                        ))
                                    }
                                    2 => Some((
                                        Ok(ModelStreamEvent::Completed {
                                            model: model.clone(),
                                            usage: Usage::default(),
                                        }),
                                        (3, model),
                                    )),
                                    _ => None,
                                }
                            }
                        });
                    return Ok(Box::pin(events));
                }
                StreamBehavior::OversizedText => vec![
                    Ok(ModelStreamEvent::ToolCallDelta(
                        ToolCallDelta::new(0, r#"{"query":"status"}"#, true)
                            .with_call_id("call-0")
                            .with_tool_id(TOOL_ID),
                    )),
                    Ok(ModelStreamEvent::TextDelta {
                        content: "x".repeat(256),
                    }),
                    Ok(ModelStreamEvent::Completed {
                        model,
                        usage: Usage::default(),
                    }),
                ],
                StreamBehavior::OversizedModel => vec![
                    Ok(ModelStreamEvent::ToolCallDelta(
                        ToolCallDelta::new(0, r#"{"query":"status"}"#, true)
                            .with_call_id("call-0")
                            .with_tool_id(TOOL_ID),
                    )),
                    Ok(ModelStreamEvent::Completed {
                        model: "m".repeat(256),
                        usage: Usage::default(),
                    }),
                ],
                StreamBehavior::InvalidResponse => vec![
                    Ok(ModelStreamEvent::ToolCallDelta(
                        ToolCallDelta::new(0, r#"{"query":"status"}"#, true)
                            .with_call_id("call-0")
                            .with_tool_id(TOOL_ID),
                    )),
                    Ok(ModelStreamEvent::TextDelta {
                        content: "invalid alongside a tool call".into(),
                    }),
                    Ok(ModelStreamEvent::Completed {
                        model,
                        usage: Usage::default(),
                    }),
                ],
                StreamBehavior::HugeArguments => vec![
                    Ok(ModelStreamEvent::ToolCallDelta(
                        ToolCallDelta::new(
                            0,
                            serde_json::to_string(&json!({"query":"x".repeat(1_024)}))
                                .expect("test arguments serialize"),
                            true,
                        )
                        .with_call_id("call-0")
                        .with_tool_id(TOOL_ID),
                    )),
                    Ok(ModelStreamEvent::Completed {
                        model,
                        usage: Usage::default(),
                    }),
                ],
                StreamBehavior::CancellationAwareOversizedText => {
                    return Ok(Box::pin(CancellationCheckingStream {
                        events: VecDeque::from([
                            Ok(ModelStreamEvent::ToolCallDelta(
                                ToolCallDelta::new(0, r#"{"query":"status"}"#, true)
                                    .with_call_id("call-0")
                                    .with_tool_id(TOOL_ID),
                            )),
                            Ok(ModelStreamEvent::TextDelta {
                                content: "x".repeat(256),
                            }),
                        ]),
                        cancellation: model_cancellation,
                        observed: Arc::clone(&self.stream_drop_saw_cancellation),
                    }));
                }
                StreamBehavior::AbortAwarePendingStart => {
                    unreachable!("pending stream startup is handled before response assembly")
                }
                StreamBehavior::AbortAwarePendingTail => {
                    return Ok(Box::pin(AbortCheckingPendingStream {
                        first: Some(Ok(ModelStreamEvent::ToolCallDelta(
                            ToolCallDelta::new(0, r#"{"query":"status"}"#, true)
                                .with_call_id("call-0")
                                .with_tool_id(TOOL_ID),
                        ))),
                        cancellation: model_cancellation,
                        observed: Arc::clone(&self.stream_drop_saw_cancellation),
                        tail_polled: Arc::clone(&self.tail_polled),
                        signalled_tail: false,
                    }));
                }
                StreamBehavior::PullDrivenDelayed => {
                    let delay_ms = self.response_delay_ms.load(Ordering::SeqCst);
                    let observe_tail = self.observe_tail.load(Ordering::SeqCst);
                    let tail_polled = Arc::clone(&self.tail_polled);
                    let events =
                        futures_util::stream::unfold((0_u8, model), move |(state, model)| {
                            let tail_polled = Arc::clone(&tail_polled);
                            async move {
                                match state {
                                    0 => Some((
                                        Ok(ModelStreamEvent::ToolCallDelta(
                                            ToolCallDelta::new(0, r#"{"query":"status"}"#, true)
                                                .with_call_id("call-0")
                                                .with_tool_id(TOOL_ID),
                                        )),
                                        (1, model),
                                    )),
                                    1 => {
                                        if observe_tail {
                                            tail_polled.add_permits(1);
                                        }
                                        if delay_ms > 0 {
                                            tokio::time::sleep(Duration::from_millis(delay_ms))
                                                .await;
                                        }
                                        Some((
                                            Ok(ModelStreamEvent::TextDelta {
                                                content: String::new(),
                                            }),
                                            (2, model),
                                        ))
                                    }
                                    2 => Some((
                                        Ok(ModelStreamEvent::Completed {
                                            model: model.clone(),
                                            usage: Usage::default(),
                                        }),
                                        (3, model),
                                    )),
                                    _ => None,
                                }
                            }
                        });
                    return Ok(Box::pin(events));
                }
            }
        };
        Ok(Box::pin(stream::iter(events)))
    }
}

struct TestPolicy {
    speculative_allow: bool,
    speculative_calls: AtomicUsize,
}

struct ApprovalPolicy;

#[async_trait]
impl PolicyEngine for ApprovalPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(PolicyDecision::RequireApproval {
            reason: "normal approval required".into(),
        })
    }
}

struct GrantApproval(AtomicUsize);

#[async_trait]
impl ApprovalHandler for GrantApproval {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ApprovalRecord::new("", &tool.id, true, "test grant"))
    }
}

struct CountingApproval {
    calls: AtomicUsize,
    grant: bool,
    delay_ms: u64,
}

impl CountingApproval {
    fn new(grant: bool) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            grant,
            delay_ms: 0,
        }
    }

    fn delayed(grant: bool, delay_ms: u64) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            grant,
            delay_ms,
        }
    }
}

#[async_trait]
impl ApprovalHandler for CountingApproval {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        }
        Ok(ApprovalRecord::new(
            "",
            &tool.id,
            self.grant,
            if self.grant {
                "test grant"
            } else {
                "test denial"
            },
        ))
    }
}

struct DenyOrdinaryPolicy {
    ordinary_calls: AtomicUsize,
    speculative_calls: AtomicUsize,
}

#[async_trait]
impl PolicyEngine for DenyOrdinaryPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        self.ordinary_calls.fetch_add(1, Ordering::SeqCst);
        Ok(PolicyDecision::Deny {
            reason: "ordinary policy denial".into(),
        })
    }

    async fn decide_speculative(
        &self,
        _: &ToolCallContext,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        self.speculative_calls.fetch_add(1, Ordering::SeqCst);
        Ok(PolicyDecision::Allow {
            reason: "dedicated allow must not override ordinary denial".into(),
        })
    }
}

struct CommitApprovalPolicy {
    ordinary_calls: AtomicUsize,
    speculative_calls: AtomicUsize,
    slow_commit_ms: u64,
}

struct CommitDenyPolicy {
    ordinary_calls: AtomicUsize,
    speculative_calls: AtomicUsize,
}

#[async_trait]
impl PolicyEngine for CommitDenyPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        let ordinal = self.ordinary_calls.fetch_add(1, Ordering::SeqCst);
        Ok(if ordinal >= 1_001 {
            PolicyDecision::Deny {
                reason: "commit-time ordinary denial".into(),
            }
        } else {
            PolicyDecision::Allow {
                reason: "ordinary allow".into(),
            }
        })
    }

    async fn decide_speculative(
        &self,
        _: &ToolCallContext,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        self.speculative_calls.fetch_add(1, Ordering::SeqCst);
        Ok(PolicyDecision::Allow {
            reason: "dedicated speculative allow".into(),
        })
    }
}

impl CommitApprovalPolicy {
    fn new(slow_commit_ms: u64) -> Self {
        Self {
            ordinary_calls: AtomicUsize::new(0),
            speculative_calls: AtomicUsize::new(0),
            slow_commit_ms,
        }
    }
}

#[async_trait]
impl PolicyEngine for CommitApprovalPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        let ordinal = self.ordinary_calls.fetch_add(1, Ordering::SeqCst);
        if ordinal >= 1_001 {
            if self.slow_commit_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.slow_commit_ms)).await;
            }
            Ok(PolicyDecision::RequireApproval {
                reason: "commit-time approval".into(),
            })
        } else {
            Ok(PolicyDecision::Allow {
                reason: "ordinary allow".into(),
            })
        }
    }

    async fn decide_speculative(
        &self,
        _: &ToolCallContext,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        self.speculative_calls.fetch_add(1, Ordering::SeqCst);
        Ok(PolicyDecision::Allow {
            reason: "dedicated speculative allow".into(),
        })
    }
}

struct BlockingCommitPolicy {
    speculative_calls: AtomicUsize,
    commit_entered: Arc<Semaphore>,
    release_commit: Arc<Semaphore>,
}

struct SlowEventSink {
    records: Mutex<Vec<EventRecord>>,
    slow_policy_events: AtomicBool,
    tool_completed_delay_ms: AtomicUsize,
}

struct BlockingFirstCompletionSink {
    armed: AtomicBool,
    completions: AtomicUsize,
    entered: Arc<Semaphore>,
    released: Mutex<bool>,
    release_changed: Condvar,
}

impl BlockingFirstCompletionSink {
    fn new() -> Self {
        Self {
            armed: AtomicBool::new(false),
            completions: AtomicUsize::new(0),
            entered: Arc::new(Semaphore::new(0)),
            released: Mutex::new(false),
            release_changed: Condvar::new(),
        }
    }

    fn arm(&self) {
        self.completions.store(0, Ordering::SeqCst);
        self.armed.store(true, Ordering::SeqCst);
    }

    fn release(&self) {
        *self.released.lock().expect("completion release lock") = true;
        self.release_changed.notify_all();
    }
}

impl EventSink for BlockingFirstCompletionSink {
    fn emit(&self, record: EventRecord) {
        if self.armed.load(Ordering::SeqCst)
            && matches!(record.event, RunEvent::ToolCompleted { .. })
            && self.completions.fetch_add(1, Ordering::SeqCst) == 0
        {
            self.entered.add_permits(1);
            let mut released = self.released.lock().expect("completion release lock");
            while !*released {
                released = self
                    .release_changed
                    .wait(released)
                    .expect("completion release condition");
            }
        }
    }
}

impl SlowEventSink {
    fn new() -> Self {
        Self {
            records: Mutex::new(Vec::new()),
            slow_policy_events: AtomicBool::new(false),
            tool_completed_delay_ms: AtomicUsize::new(0),
        }
    }
}

impl EventSink for SlowEventSink {
    fn emit(&self, record: EventRecord) {
        if self.slow_policy_events.load(Ordering::SeqCst)
            && matches!(record.event, RunEvent::PolicyDecided { .. })
        {
            thread::sleep(Duration::from_millis(20));
        }
        let tool_completed_delay_ms = self.tool_completed_delay_ms.load(Ordering::SeqCst);
        if tool_completed_delay_ms > 0 && matches!(record.event, RunEvent::ToolCompleted { .. }) {
            thread::sleep(Duration::from_millis(tool_completed_delay_ms as u64));
        }
        self.records
            .lock()
            .expect("event records lock")
            .push(record);
    }
}

impl BlockingCommitPolicy {
    fn new() -> Self {
        Self {
            speculative_calls: AtomicUsize::new(0),
            commit_entered: Arc::new(Semaphore::new(0)),
            release_commit: Arc::new(Semaphore::new(0)),
        }
    }
}

#[async_trait]
impl PolicyEngine for BlockingCommitPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(PolicyDecision::Allow {
            reason: "ordinary allow".into(),
        })
    }

    async fn decide_speculative(
        &self,
        _: &ToolCallContext,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        let ordinal = self.speculative_calls.fetch_add(1, Ordering::SeqCst);
        if ordinal == 1 {
            self.commit_entered.add_permits(1);
            let permit = self
                .release_commit
                .acquire()
                .await
                .expect("commit release semaphore open");
            permit.forget();
        }
        Ok(PolicyDecision::Allow {
            reason: "dedicated speculative allow".into(),
        })
    }
}

struct BlockingOrdinaryCommitPolicy {
    ordinary_calls: AtomicUsize,
    speculative_calls: AtomicUsize,
    commit_entered: Arc<Semaphore>,
    release_commit: Arc<Semaphore>,
}

impl BlockingOrdinaryCommitPolicy {
    fn new() -> Self {
        Self {
            ordinary_calls: AtomicUsize::new(0),
            speculative_calls: AtomicUsize::new(0),
            commit_entered: Arc::new(Semaphore::new(0)),
            release_commit: Arc::new(Semaphore::new(0)),
        }
    }
}

#[async_trait]
impl PolicyEngine for BlockingOrdinaryCommitPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        let ordinal = self.ordinary_calls.fetch_add(1, Ordering::SeqCst);
        if ordinal == 1_001 {
            self.commit_entered.add_permits(1);
            let permit = self
                .release_commit
                .acquire()
                .await
                .expect("ordinary commit release semaphore open");
            permit.forget();
        }
        Ok(PolicyDecision::Allow {
            reason: "ordinary allow".into(),
        })
    }

    async fn decide_speculative(
        &self,
        _: &ToolCallContext,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        self.speculative_calls.fetch_add(1, Ordering::SeqCst);
        Ok(PolicyDecision::Allow {
            reason: "dedicated speculative allow".into(),
        })
    }
}

struct FailingDedicatedCommitPolicy {
    ordinary_calls: AtomicUsize,
    speculative_calls: AtomicUsize,
}

#[async_trait]
impl PolicyEngine for FailingDedicatedCommitPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        let ordinal = self.ordinary_calls.fetch_add(1, Ordering::SeqCst);
        Ok(if ordinal >= 1_001 {
            PolicyDecision::RequireApproval {
                reason: "commit approval".into(),
            }
        } else {
            PolicyDecision::Allow {
                reason: "ordinary allow".into(),
            }
        })
    }

    async fn decide_speculative(
        &self,
        _: &ToolCallContext,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        let ordinal = self.speculative_calls.fetch_add(1, Ordering::SeqCst);
        if ordinal == 1 {
            Err(HarnessError::Policy(
                "dedicated commit policy failed".into(),
            ))
        } else {
            Ok(PolicyDecision::Allow {
                reason: "dedicated speculative allow".into(),
            })
        }
    }
}

impl TestPolicy {
    fn new(speculative_allow: bool) -> Self {
        Self {
            speculative_allow,
            speculative_calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl PolicyEngine for TestPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(PolicyDecision::Allow {
            reason: "normal test allow".into(),
        })
    }

    async fn decide_speculative(
        &self,
        _: &ToolCallContext,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        self.speculative_calls.fetch_add(1, Ordering::SeqCst);
        Ok(if self.speculative_allow {
            PolicyDecision::Allow {
                reason: "explicit speculative test allow".into(),
            }
        } else {
            PolicyDecision::Deny {
                reason: "speculation denied".into(),
            }
        })
    }
}

struct CountingTool {
    definition: ToolDefinition,
    calls: AtomicUsize,
    callers: Mutex<Vec<ToolCaller>>,
    delay_ms: AtomicU64,
    observe_delay: AtomicBool,
    delay_entered: Arc<Semaphore>,
}

impl CountingTool {
    fn eligible() -> Self {
        Self {
            definition: ToolDefinition::new(
                TOOL_ID,
                "Local read",
                "Reads stable local state",
                json!({
                    "type":"object",
                    "required":["query"],
                    "properties":{"query":{"type":"string"}},
                    "additionalProperties":false
                }),
            )
            .with_risk(ToolRisk::Low)
            .with_read_only(true)
            .with_idempotent(true)
            .with_parallel_safe(true)
            .with_cancellation_safety(CancellationSafety::Guaranteed)
            .with_allowed_callers([ToolCaller::Direct, ToolCaller::Speculative])
            .with_speculation_policy(SpeculationPolicy::Enabled)
            .with_issue_safety(IssueSafety::Guaranteed)
            .with_execution_location(ExecutionLocation::LocalPrivate)
            .with_network_egress(NetworkEgress::Prohibited)
            .with_output_schema(json!({
                "type":"object",
                "required":["value"],
                "properties":{"value":{"type":"string"}},
                "additionalProperties":false
            })),
            calls: AtomicUsize::new(0),
            callers: Mutex::new(Vec::new()),
            delay_ms: AtomicU64::new(0),
            observe_delay: AtomicBool::new(false),
            delay_entered: Arc::new(Semaphore::new(0)),
        }
    }

    fn unattested_caller_dependent() -> Self {
        let mut tool = Self::eligible();
        tool.definition.speculation_policy = SpeculationPolicy::Disabled;
        tool.definition.allowed_callers = [ToolCaller::Direct].into();
        tool
    }

    fn declarative_eligible() -> Self {
        let mut tool = Self::eligible();
        tool.definition.allowed_callers = [
            ToolCaller::Direct,
            ToolCaller::DeclarativePlan,
            ToolCaller::Speculative,
        ]
        .into();
        tool
    }

    fn set_delay(&self, delay_ms: u64, observe_delay: bool) {
        self.delay_ms.store(delay_ms, Ordering::SeqCst);
        self.observe_delay.store(observe_delay, Ordering::SeqCst);
    }
}

#[async_trait]
impl Tool for CountingTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(
        &self,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        let context = ToolCallContext::new("", "", "", TOOL_ID);
        self.execute_with_context(&context, arguments, cancellation)
            .await
    }

    async fn execute_with_context(
        &self,
        context: &ToolCallContext,
        _: Value,
        _: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let caller = context.caller.unwrap_or(ToolCaller::Direct);
        self.callers.lock().expect("caller log lock").push(caller);
        if self.observe_delay.load(Ordering::SeqCst) {
            self.delay_entered.add_permits(1);
        }
        let delay_ms = self.delay_ms.load(Ordering::SeqCst);
        if delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }
        Ok(ToolResult::success(json!({"value":"stable"})))
    }
}

fn request() -> RunRequest {
    let mut agent = AgentDefinition::new("speculation-test", "Speculation test", "1", "test-model");
    agent.tool_allowlist = vec![TOOL_ID.into()];
    RunRequest::new(agent, "status")
}

fn registry(tool: Arc<dyn Tool>) -> ToolRegistry {
    let mut registry = ToolRegistry::default();
    registry.register(tool).expect("test tool must register");
    registry
}

async fn train_and_activate(runner: &AgentRunner) {
    for _ in 0..1_000 {
        let result = runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
            .expect("shadow training run succeeds");
        assert_eq!(result.status, RunStatus::Completed);
    }
    assert_eq!(
        runner.activate_speculation(TOOL_ID).mode,
        SpeculationMode::Active
    );
}

struct BlockingSpeculativeTool {
    definition: ToolDefinition,
    calls: AtomicUsize,
    entered: Arc<Semaphore>,
    release: Arc<Semaphore>,
}

impl BlockingSpeculativeTool {
    fn new() -> Self {
        Self {
            definition: CountingTool::eligible().definition,
            calls: AtomicUsize::new(0),
            entered: Arc::new(Semaphore::new(0)),
            release: Arc::new(Semaphore::new(0)),
        }
    }
}

#[async_trait]
impl Tool for BlockingSpeculativeTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success(json!({"value":"direct"})))
    }

    async fn execute_with_context(
        &self,
        context: &ToolCallContext,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        if context.caller != Some(ToolCaller::Speculative) {
            return self.execute(arguments, cancellation).await;
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.add_permits(1);
        let permit = self
            .release
            .acquire()
            .await
            .expect("release semaphore open");
        permit.forget();
        Ok(ToolResult::success(json!({"value":"direct"})))
    }
}

struct CancellationAwareTool {
    definition: ToolDefinition,
    calls: AtomicUsize,
    observed_cancellation: AtomicUsize,
    entered: Arc<Semaphore>,
}

impl CancellationAwareTool {
    fn new() -> Self {
        Self {
            definition: CountingTool::eligible().definition,
            calls: AtomicUsize::new(0),
            observed_cancellation: AtomicUsize::new(0),
            entered: Arc::new(Semaphore::new(0)),
        }
    }
}

#[async_trait]
impl Tool for CancellationAwareTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success(json!({"value":"direct"})))
    }

    async fn execute_with_context(
        &self,
        context: &ToolCallContext,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        if context.caller != Some(ToolCaller::Speculative) {
            return self.execute(arguments, cancellation).await;
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.add_permits(1);
        cancellation.cancelled().await;
        self.observed_cancellation.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success(json!({"value":"direct"})))
    }
}

struct ToolFutureDropProbe {
    cancellation: CancellationToken,
    saw_cancellation: Arc<AtomicBool>,
}

impl Drop for ToolFutureDropProbe {
    fn drop(&mut self) {
        self.saw_cancellation
            .store(self.cancellation.is_cancelled(), Ordering::SeqCst);
    }
}

struct AbortAwareSpeculativeTool {
    definition: ToolDefinition,
    calls: AtomicUsize,
    entered: Arc<Semaphore>,
    future_drop_saw_cancellation: Arc<AtomicBool>,
}

impl AbortAwareSpeculativeTool {
    fn new() -> Self {
        Self {
            definition: CountingTool::eligible().definition,
            calls: AtomicUsize::new(0),
            entered: Arc::new(Semaphore::new(0)),
            future_drop_saw_cancellation: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[async_trait]
impl Tool for AbortAwareSpeculativeTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success(json!({"value":"stable"})))
    }

    async fn execute_with_context(
        &self,
        context: &ToolCallContext,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        if context.caller != Some(ToolCaller::Speculative) {
            return self.execute(arguments, cancellation).await;
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        let _drop_probe = ToolFutureDropProbe {
            cancellation,
            saw_cancellation: Arc::clone(&self.future_drop_saw_cancellation),
        };
        self.entered.add_permits(1);
        std::future::pending::<()>().await;
        unreachable!("pending speculative test tool cannot complete")
    }
}

#[derive(Clone, Copy)]
enum InvalidSpeculativeResult {
    Oversized,
    TooDeep,
    SchemaInvalid,
}

struct InvalidSpeculativeResultTool {
    definition: ToolDefinition,
    kind: InvalidSpeculativeResult,
    calls: AtomicUsize,
    callers: Mutex<Vec<ToolCaller>>,
}

impl InvalidSpeculativeResultTool {
    fn new(kind: InvalidSpeculativeResult) -> Self {
        let mut definition = CountingTool::eligible().definition;
        if matches!(kind, InvalidSpeculativeResult::TooDeep) {
            definition.output_schema = Some(json!({"type":"object"}));
        }
        Self {
            definition,
            kind,
            calls: AtomicUsize::new(0),
            callers: Mutex::new(Vec::new()),
        }
    }

    fn speculative_output(&self) -> Value {
        match self.kind {
            InvalidSpeculativeResult::Oversized => json!({"value":"x".repeat(512)}),
            InvalidSpeculativeResult::SchemaInvalid => json!({"unexpected":true}),
            InvalidSpeculativeResult::TooDeep => {
                let mut value = json!("leaf");
                for _ in 0..70 {
                    value = json!({"nested":value});
                }
                value
            }
        }
    }
}

#[async_trait]
impl Tool for InvalidSpeculativeResultTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.callers
            .lock()
            .expect("caller log lock")
            .push(ToolCaller::Direct);
        Ok(ToolResult::success(json!({"value":"stable"})))
    }

    async fn execute_with_context(
        &self,
        context: &ToolCallContext,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        if context.caller != Some(ToolCaller::Speculative) {
            return self.execute(arguments, cancellation).await;
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.callers
            .lock()
            .expect("caller log lock")
            .push(ToolCaller::Speculative);
        Ok(ToolResult::success(self.speculative_output()))
    }
}

#[tokio::test]
async fn omitted_config_preserves_complete_path() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::unattested_caller_dependent());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .build();

    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("disabled run succeeds");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(provider.complete_calls.load(Ordering::SeqCst), 2);
    assert_eq!(provider.stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        runner.speculation_readiness(TOOL_ID).mode,
        SpeculationMode::Disabled
    );
}

#[test]
fn disabled_readiness_reports_the_configured_activation_threshold() {
    let runner = AgentRunner::builder(Arc::new(StreamingProvider::new(StreamBehavior::Normal)))
        .speculation(SpeculationConfig {
            required_shadow_observations: 1_234,
            ..SpeculationConfig::default()
        })
        .build();

    let readiness = runner.speculation_readiness("unknown");
    assert_eq!(readiness.mode, SpeculationMode::Disabled);
    assert_eq!(readiness.required_shadow_observations, 1_234);
}

#[tokio::test]
async fn shadow_observes_without_speculative_execution_or_policy() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let policy = Arc::new(TestPolicy::new(true));
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .policy(policy.clone())
        .speculation(SpeculationConfig::default())
        .build();

    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("shadow run succeeds");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(provider.complete_calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.stream_calls.load(Ordering::SeqCst), 1);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        tool.callers.lock().unwrap().as_slice(),
        [ToolCaller::Direct]
    );
    assert_eq!(policy.speculative_calls.load(Ordering::SeqCst), 0);
    let readiness = runner.speculation_readiness(TOOL_ID);
    assert_eq!(readiness.mode, SpeculationMode::Shadow);
    assert_eq!(readiness.exact_shadow_observations, 1);
    assert!(!readiness.ready_to_activate);
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.shadow_matches, 1);
    assert_eq!(metrics.issued, 0);
}

#[tokio::test]
async fn activation_requires_threshold_and_exact_active_match_commits_once() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let policy = Arc::new(TestPolicy::new(true));
    let events = Arc::new(InMemoryEventSink::default());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .policy(policy.clone())
        .event_sink(events.clone())
        .speculation(SpeculationConfig::default())
        .build();

    assert_eq!(
        runner.activate_speculation(TOOL_ID).mode,
        SpeculationMode::Disabled
    );
    for _ in 0..999 {
        let result = runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
            .expect("shadow training run succeeds");
        assert_eq!(result.status, RunStatus::Completed);
    }
    assert_eq!(
        runner.activate_speculation(TOOL_ID).mode,
        SpeculationMode::Shadow
    );
    runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("threshold run succeeds");
    assert!(runner.speculation_readiness(TOOL_ID).ready_to_activate);
    assert_eq!(
        runner.activate_speculation(TOOL_ID).mode,
        SpeculationMode::Active
    );

    let event_start = events.events().len();
    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("active run succeeds");
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(provider.stream_calls.load(Ordering::SeqCst), 1_001);
    assert_eq!(provider.complete_calls.load(Ordering::SeqCst), 1_001);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_001);
    assert_eq!(
        tool.callers.lock().unwrap().last(),
        Some(&ToolCaller::Speculative)
    );
    assert_eq!(policy.speculative_calls.load(Ordering::SeqCst), 2);
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.committed, 1);
    assert_eq!(metrics.discarded, 0);
    assert_eq!(metrics.cancelled, 0);
    assert_eq!(
        metrics.issued,
        metrics.committed + metrics.discarded + metrics.cancelled
    );
    let all_events = events.events();
    let active_events = &all_events[event_start..];
    assert_eq!(
        active_events
            .iter()
            .filter(|record| matches!(record.event, RunEvent::ToolCompleted { .. }))
            .count(),
        1
    );
    let usage = active_events
        .iter()
        .rev()
        .find_map(|record| match record.event {
            RunEvent::StrategyUsage {
                tool_calls,
                tool_issued,
                tool_completed,
                tool_reused,
                ..
            } => Some((tool_calls, tool_issued, tool_completed, tool_reused)),
            _ => None,
        })
        .expect("active run emits strategy usage");
    assert_eq!(usage, (1, 1, 1, 0));

    assert_eq!(
        runner.return_speculation_to_shadow(TOOL_ID).mode,
        SpeculationMode::Shadow
    );
    assert_eq!(
        runner
            .speculation_readiness(TOOL_ID)
            .exact_shadow_observations,
        0
    );
}

#[tokio::test]
async fn unattested_caller_dependent_tool_never_enters_shadow_or_speculation() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::unattested_caller_dependent());
    let runner = AgentRunner::builder(provider)
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .speculation(SpeculationConfig::default())
        .build();

    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("direct-only run succeeds");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(
        tool.callers.lock().unwrap().as_slice(),
        [ToolCaller::Direct]
    );
    assert_eq!(
        runner.speculation_readiness(TOOL_ID).mode,
        SpeculationMode::Disabled
    );
    assert_eq!(runner.speculation_metrics(TOOL_ID).issued, 0);
}

#[tokio::test]
async fn ordinary_allow_policy_never_authorizes_speculation() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let runner = AgentRunner::builder(provider)
        .tools(registry(tool.clone()))
        .policy(Arc::new(AllowAllPolicy))
        .speculation(SpeculationConfig::default())
        .build();

    for _ in 0..1_000 {
        runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
            .expect("shadow training run succeeds");
    }
    assert_eq!(
        runner.activate_speculation(TOOL_ID).mode,
        SpeculationMode::Active
    );
    runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("default speculative denial falls back to Direct");

    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_001);
    assert!(tool
        .callers
        .lock()
        .unwrap()
        .iter()
        .all(|caller| *caller == ToolCaller::Direct));
    assert_eq!(runner.speculation_metrics(TOOL_ID).issued, 0);
}

#[tokio::test]
async fn granted_approval_never_authorizes_speculation() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let approvals = Arc::new(GrantApproval(AtomicUsize::new(0)));
    let runner = AgentRunner::builder(provider)
        .tools(registry(tool.clone()))
        .policy(Arc::new(ApprovalPolicy))
        .approvals(approvals.clone())
        .speculation(SpeculationConfig::default())
        .build();

    for _ in 0..1_000 {
        runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
            .expect("approved shadow run succeeds");
    }
    assert_eq!(
        runner.activate_speculation(TOOL_ID).mode,
        SpeculationMode::Active
    );
    runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("approval-only active run falls back to Direct");

    assert_eq!(approvals.0.load(Ordering::SeqCst), 1_001);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_001);
    assert_eq!(runner.speculation_metrics(TOOL_ID).issued, 0);
}

#[tokio::test]
async fn ordinary_deny_prevents_hidden_issue_even_when_dedicated_policy_allows() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let policy = Arc::new(DenyOrdinaryPolicy {
        ordinary_calls: AtomicUsize::new(0),
        speculative_calls: AtomicUsize::new(0),
    });
    let approvals = Arc::new(CountingApproval::new(true));
    let runner = AgentRunner::builder(provider)
        .tools(registry(tool.clone()))
        .policy(policy.clone())
        .approvals(approvals.clone())
        .speculation(SpeculationConfig::default())
        .build();

    train_and_activate(&runner).await;
    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("ordinary denial is represented canonically");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    assert_eq!(approvals.calls.load(Ordering::SeqCst), 0);
    assert_eq!(policy.speculative_calls.load(Ordering::SeqCst), 0);
    assert_eq!(runner.speculation_metrics(TOOL_ID).issued, 0);
}

#[tokio::test]
async fn ordinary_deny_at_commit_discards_cache_without_public_completion() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let policy = Arc::new(CommitDenyPolicy {
        ordinary_calls: AtomicUsize::new(0),
        speculative_calls: AtomicUsize::new(0),
    });
    let events = Arc::new(InMemoryEventSink::default());
    let runner = AgentRunner::builder(provider)
        .tools(registry(tool.clone()))
        .policy(policy.clone())
        .event_sink(events.clone())
        .speculation(SpeculationConfig::default())
        .build();
    train_and_activate(&runner).await;
    let event_start = events.events().len();

    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("commit-time denial is represented canonically");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_001);
    assert_eq!(policy.speculative_calls.load(Ordering::SeqCst), 1);
    let active_events = events.events()[event_start..].to_vec();
    assert_eq!(
        active_events
            .iter()
            .filter(|record| matches!(record.event, RunEvent::ToolRejected { .. }))
            .count(),
        1
    );
    assert!(active_events
        .iter()
        .all(|record| !matches!(record.event, RunEvent::ToolCompleted { .. })));
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.committed, 0);
    assert_eq!(metrics.discarded, 1);
}

#[tokio::test]
async fn commit_time_approval_is_requested_once_and_controls_cached_publication() {
    for grant in [false, true] {
        let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
        let tool = Arc::new(CountingTool::eligible());
        let policy = Arc::new(CommitApprovalPolicy::new(0));
        let approvals = Arc::new(CountingApproval::new(grant));
        let events = Arc::new(InMemoryEventSink::default());
        let runner = AgentRunner::builder(provider)
            .tools(registry(tool.clone()))
            .policy(policy.clone())
            .approvals(approvals.clone())
            .event_sink(events.clone())
            .speculation(SpeculationConfig::default())
            .build();

        train_and_activate(&runner).await;
        let event_start = events.events().len();
        let result = runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
            .expect("commit-time approval is canonical");

        assert_eq!(result.status, RunStatus::Completed);
        assert_eq!(approvals.calls.load(Ordering::SeqCst), 1);
        assert_eq!(tool.calls.load(Ordering::SeqCst), 1_001);
        let active_events = events.events()[event_start..].to_vec();
        assert_eq!(
            active_events
                .iter()
                .filter(|record| matches!(record.event, RunEvent::ApprovalRequested { .. }))
                .count(),
            1
        );
        let metrics = runner.speculation_metrics(TOOL_ID);
        if grant {
            assert_eq!(metrics.committed, 1);
            assert_eq!(metrics.discarded, 0);
            assert_eq!(policy.speculative_calls.load(Ordering::SeqCst), 2);
            assert_eq!(
                active_events
                    .iter()
                    .filter(|record| matches!(record.event, RunEvent::ToolCompleted { .. }))
                    .count(),
                1
            );
        } else {
            assert_eq!(metrics.committed, 0);
            assert_eq!(metrics.discarded, 1);
            assert_eq!(policy.speculative_calls.load(Ordering::SeqCst), 1);
            assert!(active_events
                .iter()
                .all(|record| !matches!(record.event, RunEvent::ToolCompleted { .. })));
        }
        assert_eq!(
            metrics.issued,
            metrics.committed + metrics.discarded + metrics.cancelled
        );
    }
}

#[tokio::test]
async fn runner_slot_saturation_skips_without_queueing_a_second_candidate() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(BlockingSpeculativeTool::new());
    let runner = Arc::new(
        AgentRunner::builder(provider)
            .tools(registry(tool.clone()))
            .policy(Arc::new(TestPolicy::new(true)))
            .speculation(SpeculationConfig::default())
            .build(),
    );
    for _ in 0..1_000 {
        runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
            .expect("shadow training run succeeds");
    }
    assert_eq!(
        runner.activate_speculation(TOOL_ID).mode,
        SpeculationMode::Active
    );

    let first_runner = runner.clone();
    let first = tokio::spawn(async move {
        first_runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
    });
    let entered = tool.entered.acquire().await.expect("entry semaphore open");
    entered.forget();

    let second = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("saturated candidate immediately falls back");
    assert_eq!(second.status, RunStatus::Completed);
    assert_eq!(runner.speculation_metrics(TOOL_ID).slot_saturated, 1);

    tool.release.add_permits(1);
    let first = first.await.unwrap().expect("first active run succeeds");
    assert_eq!(first.status, RunStatus::Completed);
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.committed, 1);
}

#[tokio::test]
async fn candidate_deadline_cancels_and_drains_before_direct_fallback() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CancellationAwareTool::new());
    let runner = AgentRunner::builder(provider)
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .speculation(SpeculationConfig {
            max_execution_duration_ms: 1,
            ..SpeculationConfig::default()
        })
        .build();
    for _ in 0..1_000 {
        runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
            .expect("shadow training run succeeds");
    }
    assert_eq!(
        runner.activate_speculation(TOOL_ID).mode,
        SpeculationMode::Active
    );

    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("cancelled candidate falls back to Direct");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(tool.observed_cancellation.load(Ordering::SeqCst), 1);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_002);
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.cancelled, 1);
    assert_eq!(metrics.mode, SpeculationMode::Shadow);
}

#[tokio::test]
async fn invalid_active_results_are_discarded_before_direct_fallback() {
    for kind in [
        InvalidSpeculativeResult::Oversized,
        InvalidSpeculativeResult::TooDeep,
        InvalidSpeculativeResult::SchemaInvalid,
    ] {
        let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
        let tool = Arc::new(InvalidSpeculativeResultTool::new(kind));
        let runner = AgentRunner::builder(provider)
            .tools(registry(tool.clone()))
            .policy(Arc::new(TestPolicy::new(true)))
            .speculation(SpeculationConfig::default())
            .build();
        train_and_activate(&runner).await;
        let mut active_request = request();
        if matches!(kind, InvalidSpeculativeResult::Oversized) {
            active_request.agent.limits.max_tool_result_bytes = 128;
        }

        let result = runner
            .run_with_strategy(active_request, RunStrategy::Direct)
            .await
            .expect("invalid speculative result falls back");

        assert_eq!(result.status, RunStatus::Completed);
        assert_eq!(tool.calls.load(Ordering::SeqCst), 1_002);
        assert_eq!(
            tool.callers.lock().unwrap()[1_000..],
            [ToolCaller::Speculative, ToolCaller::Direct]
        );
        let metrics = runner.speculation_metrics(TOOL_ID);
        assert_eq!(metrics.issued, 1);
        assert_eq!(metrics.committed, 0);
        assert_eq!(metrics.discarded, 1);
        assert_eq!(metrics.cancelled, 0);
    }
}

#[tokio::test]
async fn external_cancellation_signals_and_drains_an_in_flight_candidate() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CancellationAwareTool::new());
    let runner = Arc::new(
        AgentRunner::builder(provider)
            .tools(registry(tool.clone()))
            .policy(Arc::new(TestPolicy::new(true)))
            .speculation(SpeculationConfig::default())
            .build(),
    );
    train_and_activate(&runner).await;

    let cancellation = CancellationToken::new();
    let mut active_request = request();
    active_request.cancellation = cancellation.clone();
    let active_runner = Arc::clone(&runner);
    let run = tokio::spawn(async move {
        active_runner
            .run_with_strategy(active_request, RunStrategy::Direct)
            .await
    });
    let entered = tool
        .entered
        .acquire()
        .await
        .expect("candidate entry semaphore open");
    entered.forget();
    cancellation.cancel();

    let result = run.await.unwrap().expect("cancelled run is represented");
    assert_eq!(result.status, RunStatus::Cancelled);
    assert_eq!(tool.observed_cancellation.load(Ordering::SeqCst), 1);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_001);
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.cancelled, 1);
    assert_eq!(metrics.committed, 0);
    assert_eq!(metrics.discarded, 0);
}

#[tokio::test]
async fn nonfinal_partial_arguments_never_execute_before_the_final_boundary() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry(tool.clone()))
            .policy(Arc::new(TestPolicy::new(true)))
            .speculation(SpeculationConfig::default())
            .build(),
    );
    train_and_activate(&runner).await;
    provider.set_behavior(StreamBehavior::PartialPaused);

    let active_runner = Arc::clone(&runner);
    let run = tokio::spawn(async move {
        active_runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
    });
    let partial = provider
        .partial_emitted
        .acquire()
        .await
        .expect("partial signal semaphore open");
    partial.forget();
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_000);
    assert_eq!(runner.speculation_metrics(TOOL_ID).issued, 0);

    provider.release_final.add_permits(1);
    let result = run.await.unwrap().expect("finalized call succeeds");
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_001);
    assert_eq!(runner.speculation_metrics(TOOL_ID).committed, 1);
}

#[tokio::test]
async fn interleaved_stream_calls_execute_in_index_order_and_only_index_zero_trains() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::InterleavedMultiple));
    let tool = Arc::new(CountingTool::eligible());
    let runner = AgentRunner::builder(provider)
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .speculation(SpeculationConfig::default())
        .build();

    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("interleaved stream run succeeds");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.tool_calls.len(), 2);
    assert_eq!(result.tool_calls[0].id, "call-0");
    assert_eq!(result.tool_calls[1].id, "call-1");
    assert_eq!(tool.calls.load(Ordering::SeqCst), 2);
    assert_eq!(runner.speculation_metrics(TOOL_ID).shadow_matches, 1);
}

#[tokio::test]
async fn active_multi_call_stream_commits_only_index_zero_then_executes_index_one() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .speculation(SpeculationConfig::default())
        .build();
    train_and_activate(&runner).await;
    provider.set_behavior(StreamBehavior::InterleavedMultiple);

    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("active multi-call run succeeds");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.tool_calls.len(), 2);
    assert_eq!(result.tool_calls[0].id, "call-0");
    assert_eq!(result.tool_calls[1].id, "call-1");
    assert_eq!(
        tool.callers.lock().unwrap()[1_000..],
        [ToolCaller::Speculative, ToolCaller::Direct]
    );
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.committed, 1);
}

#[tokio::test]
async fn terminal_stream_failure_after_candidate_is_value_free_and_never_retried() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::FailAfterCandidate));
    let tool = Arc::new(CountingTool::eligible());
    let events = Arc::new(InMemoryEventSink::default());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .event_sink(events.clone())
        .speculation(SpeculationConfig::default())
        .build();

    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("terminal stream failures are represented in the run result");

    assert_eq!(result.status, RunStatus::Failed);
    assert_eq!(provider.complete_calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.stream_calls.load(Ordering::SeqCst), 1);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    assert!(result
        .errors
        .iter()
        .any(|error| error.code == "model_stream.upstream_provider_failure"));
    assert!(!events
        .events()
        .iter()
        .any(|record: &EventRecord| { matches!(record.event, RunEvent::ModelRetrying { .. }) }));
    assert_eq!(
        runner.speculation_metrics(TOOL_ID).terminal_stream_failures,
        1
    );
    assert!(!serde_json::to_string(&result)
        .unwrap()
        .contains(PRIVACY_CANARY));
    assert!(!serde_json::to_string(&events.events())
        .unwrap()
        .contains(PRIVACY_CANARY));
    assert!(!serde_json::to_string(&runner.speculation_metrics(TOOL_ID))
        .unwrap()
        .contains(PRIVACY_CANARY));
}

#[tokio::test]
async fn retry_after_speculative_stream_failure_is_sequential() {
    let provider = Arc::new(StreamingProvider::new(
        StreamBehavior::RetryableBeforeCandidate,
    ));
    let tool = Arc::new(CountingTool::eligible());
    let events = Arc::new(InMemoryEventSink::default());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .event_sink(events.clone())
        .speculation(SpeculationConfig::default())
        .build();

    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("retry completes through the sequential provider path");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(provider.stream_calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.complete_calls.load(Ordering::SeqCst), 2);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    assert_eq!(runner.speculation_metrics(TOOL_ID).issued, 0);
    assert_eq!(
        events
            .events()
            .iter()
            .filter(|record| matches!(record.event, RunEvent::ModelRetrying { .. }))
            .count(),
        1
    );
}

#[tokio::test]
async fn every_post_issue_stream_or_response_validation_error_settles_the_candidate() {
    for behavior in [
        StreamBehavior::OversizedText,
        StreamBehavior::OversizedModel,
        StreamBehavior::InvalidResponse,
    ] {
        let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
        let tool = Arc::new(CountingTool::eligible());
        let runner = AgentRunner::builder(provider.clone())
            .tools(registry(tool.clone()))
            .policy(Arc::new(TestPolicy::new(true)))
            .speculation(SpeculationConfig::default())
            .build();
        train_and_activate(&runner).await;
        provider.set_behavior(behavior);

        let mut active_request = request();
        if matches!(
            behavior,
            StreamBehavior::OversizedText | StreamBehavior::OversizedModel
        ) {
            active_request.agent.limits.max_model_response_bytes = 128;
        }
        let result = runner
            .run_with_strategy(active_request, RunStrategy::Direct)
            .await
            .expect("post-issue validation failure is represented");

        assert_ne!(result.status, RunStatus::Cancelled);
        assert_eq!(tool.calls.load(Ordering::SeqCst), 1_001);
        let metrics = runner.speculation_metrics(TOOL_ID);
        assert_eq!(metrics.issued, 1);
        assert_eq!(metrics.committed, 0);
        assert_eq!(metrics.discarded, 1);
        assert_eq!(metrics.cancelled, 0);
        assert_eq!(
            metrics.issued,
            metrics.committed + metrics.discarded + metrics.cancelled
        );
    }
}

#[tokio::test]
async fn transcript_overflow_after_issue_settles_without_publishing_the_candidate() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .speculation(SpeculationConfig::default())
        .build();
    train_and_activate(&runner).await;
    provider.set_behavior(StreamBehavior::HugeArguments);
    let mut active_request = request();
    active_request.agent.limits.max_transcript_bytes = 600;

    let result = runner
        .run_with_strategy(active_request, RunStrategy::Direct)
        .await
        .expect("transcript overflow is represented");

    assert_eq!(result.status, RunStatus::LimitReached);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_001);
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.discarded, 1);
    assert_eq!(metrics.committed, 0);
}

#[tokio::test]
async fn adaptive_invalid_plan_fallback_never_starts_speculation() {
    let provider = Arc::new(StreamingProvider::adaptive_invalid_plan());
    let tool = Arc::new(CountingTool::declarative_eligible());
    let events = Arc::new(InMemoryEventSink::default());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .event_sink(events.clone())
        .speculation(SpeculationConfig::default())
        .build();

    let result = runner
        .run(request())
        .await
        .expect("adaptive invalid plan falls back to direct");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(provider.complete_calls.load(Ordering::SeqCst), 4);
    assert_eq!(provider.stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    assert_eq!(runner.speculation_metrics(TOOL_ID).issued, 0);
    assert!(events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyFallback {
            reason: llama_harness_core::StrategyFallbackReason::InvalidPlan,
            ..
        }
    )));
}

#[tokio::test]
async fn deactivation_during_commit_discards_cache_and_executes_authoritative_direct_once() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let policy = Arc::new(BlockingCommitPolicy::new());
    let runner = Arc::new(
        AgentRunner::builder(provider)
            .tools(registry(tool.clone()))
            .policy(policy.clone())
            .speculation(SpeculationConfig::default())
            .build(),
    );
    train_and_activate(&runner).await;

    let active_runner = Arc::clone(&runner);
    let run = tokio::spawn(async move {
        active_runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
    });
    let entered = policy
        .commit_entered
        .acquire()
        .await
        .expect("commit policy entry semaphore open");
    entered.forget();
    assert_eq!(
        runner.return_speculation_to_shadow(TOOL_ID).mode,
        SpeculationMode::Shadow
    );
    policy.release_commit.add_permits(1);

    let result = run.await.unwrap().expect("deactivated run falls back");
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_002);
    assert_eq!(
        tool.callers.lock().unwrap()[1_000..],
        [ToolCaller::Speculative, ToolCaller::Direct]
    );
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.discarded, 1);
    assert_eq!(metrics.committed, 0);
}

#[tokio::test(start_paused = true)]
async fn candidate_deadline_blocks_reuse_after_slow_commit_policy_or_approval() {
    for (slow_policy_ms, slow_approval_ms) in [(20, 0), (0, 20)] {
        let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
        let tool = Arc::new(CountingTool::eligible());
        let policy = Arc::new(CommitApprovalPolicy::new(slow_policy_ms));
        let approvals = Arc::new(CountingApproval::delayed(true, slow_approval_ms));
        let runner = AgentRunner::builder(provider)
            .tools(registry(tool.clone()))
            .policy(policy)
            .approvals(approvals.clone())
            .speculation(SpeculationConfig {
                max_execution_duration_ms: 5,
                ..SpeculationConfig::default()
            })
            .build();
        train_and_activate(&runner).await;

        let result = runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
            .expect("expired cache executes the already-authorized Direct call");

        assert_eq!(result.status, RunStatus::Completed);
        assert_eq!(approvals.calls.load(Ordering::SeqCst), 1);
        assert_eq!(tool.calls.load(Ordering::SeqCst), 1_002);
        assert_eq!(
            tool.callers.lock().unwrap()[1_000..],
            [ToolCaller::Speculative, ToolCaller::Direct]
        );
        let metrics = runner.speculation_metrics(TOOL_ID);
        assert_eq!(metrics.issued, 1);
        assert_eq!(metrics.committed, 0);
        assert_eq!(metrics.discarded, 1);
    }
}

#[tokio::test]
async fn candidate_deadline_blocks_reuse_after_a_slow_normal_event_sink() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let events = Arc::new(SlowEventSink::new());
    let runner = AgentRunner::builder(provider)
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .event_sink(events.clone())
        .speculation(SpeculationConfig {
            max_execution_duration_ms: 5,
            ..SpeculationConfig::default()
        })
        .build();
    train_and_activate(&runner).await;
    events.slow_policy_events.store(true, Ordering::SeqCst);

    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("expired cache executes Direct after the normal event boundary");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_002);
    assert_eq!(
        tool.callers.lock().unwrap()[1_000..],
        [ToolCaller::Speculative, ToolCaller::Direct]
    );
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.committed, 0);
    assert_eq!(metrics.discarded, 1);
}

#[tokio::test(start_paused = true)]
async fn candidate_lease_expires_while_canonical_prepare_continues_once() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let mut keyed_tool = CountingTool::eligible();
    keyed_tool.definition.concurrency_key = Some("lease-key".into());
    let tool = Arc::new(keyed_tool);
    let policy = Arc::new(BlockingOrdinaryCommitPolicy::new());
    let runner = Arc::new(
        AgentRunner::builder(provider)
            .tools(registry(tool.clone()))
            .policy(policy.clone())
            .speculation(SpeculationConfig {
                max_execution_duration_ms: 5,
                ..SpeculationConfig::default()
            })
            .build(),
    );
    train_and_activate(&runner).await;

    let active_runner = Arc::clone(&runner);
    let run = tokio::spawn(async move {
        active_runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
    });
    let entered = policy
        .commit_entered
        .acquire()
        .await
        .expect("ordinary policy entry semaphore open");
    entered.forget();
    tokio::time::advance(Duration::from_millis(6)).await;
    tokio::task::yield_now().await;

    let expired = runner.speculation_metrics(TOOL_ID);
    assert_eq!(expired.issued, 1);
    assert_eq!(expired.discarded, 1);
    assert_eq!(expired.committed, 0);
    assert_eq!(expired.cancelled, 0);
    assert_eq!(expired.mode, SpeculationMode::Shadow);
    assert!(
        !run.is_finished(),
        "canonical policy future remains in flight"
    );
    assert_eq!(
        runner.activate_speculation(TOOL_ID).mode,
        SpeculationMode::Shadow
    );

    policy.release_commit.add_permits(1);
    let result = run
        .await
        .unwrap()
        .expect("canonical Direct prepare completes");
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(policy.ordinary_calls.load(Ordering::SeqCst), 1_002);
    assert_eq!(policy.speculative_calls.load(Ordering::SeqCst), 1);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_002);
    assert_eq!(
        tool.callers.lock().unwrap()[1_000..],
        [ToolCaller::Speculative, ToolCaller::Direct]
    );
}

#[tokio::test]
async fn atomic_live_take_precedes_slow_tool_completed_publication() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let events = Arc::new(SlowEventSink::new());
    let runner = AgentRunner::builder(provider)
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .event_sink(events.clone())
        .speculation(SpeculationConfig {
            max_execution_duration_ms: 50,
            ..SpeculationConfig::default()
        })
        .build();
    train_and_activate(&runner).await;
    events.tool_completed_delay_ms.store(100, Ordering::SeqCst);

    let started = std::time::Instant::now();
    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("committed publication completes");

    assert!(started.elapsed() >= Duration::from_millis(100));
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_001);
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.committed, 1);
    assert_eq!(metrics.discarded, 0);
    assert_eq!(metrics.cancelled, 0);
}

#[tokio::test]
async fn aborted_run_cancels_tool_future_before_drop_and_settles_once() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(AbortAwareSpeculativeTool::new());
    let runner = Arc::new(
        AgentRunner::builder(provider)
            .tools(registry(tool.clone()))
            .policy(Arc::new(TestPolicy::new(true)))
            .speculation(SpeculationConfig::default())
            .build(),
    );
    train_and_activate(&runner).await;

    let active_runner = Arc::clone(&runner);
    let run = tokio::spawn(async move {
        active_runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
    });
    let entered = tool
        .entered
        .acquire()
        .await
        .expect("abort tool entry semaphore open");
    entered.forget();
    run.abort();
    assert!(run
        .await
        .expect_err("aborted task must not complete")
        .is_cancelled());

    assert!(tool.future_drop_saw_cancellation.load(Ordering::SeqCst));
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.cancelled, 1);
    assert_eq!(metrics.committed, 0);
    assert_eq!(metrics.discarded, 0);
    assert_eq!(metrics.mode, SpeculationMode::Shadow);
    assert_eq!(
        runner.activate_speculation(TOOL_ID).mode,
        SpeculationMode::Shadow
    );

    let fallback = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("subsequent call remains sequential Direct");
    assert_eq!(fallback.status, RunStatus::Completed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_002);
    assert_eq!(runner.speculation_metrics(TOOL_ID).issued, 1);
}

#[tokio::test]
async fn aborted_direct_stream_cancels_model_before_provider_drop() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(AbortAwareSpeculativeTool::new());
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry(tool.clone()))
            .policy(Arc::new(TestPolicy::new(true)))
            .speculation(SpeculationConfig::default())
            .build(),
    );
    train_and_activate(&runner).await;
    provider.set_behavior(StreamBehavior::AbortAwarePendingTail);

    let active_runner = Arc::clone(&runner);
    let run = tokio::spawn(async move {
        active_runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
    });
    let tool_entered = tool
        .entered
        .acquire()
        .await
        .expect("abort tool entry semaphore open");
    tool_entered.forget();
    let tail_polled = provider
        .tail_polled
        .acquire()
        .await
        .expect("pending provider tail semaphore open");
    tail_polled.forget();

    run.abort();
    assert!(run
        .await
        .expect_err("aborted task must not complete")
        .is_cancelled());

    assert!(provider.stream_drop_saw_cancellation.load(Ordering::SeqCst));
    assert!(tool.future_drop_saw_cancellation.load(Ordering::SeqCst));
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.in_flight, 0);
    assert_eq!(metrics.cancelled, 1);
    assert_eq!(metrics.committed, 0);
    assert_eq!(metrics.discarded, 0);
    assert_eq!(metrics.mode, SpeculationMode::Shadow);
    assert_eq!(
        metrics.issued,
        metrics.in_flight + metrics.committed + metrics.discarded + metrics.cancelled
    );
}

#[tokio::test]
async fn aborted_direct_stream_start_cancels_model() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry(tool))
            .policy(Arc::new(TestPolicy::new(true)))
            .speculation(SpeculationConfig::default())
            .build(),
    );
    train_and_activate(&runner).await;
    provider.set_behavior(StreamBehavior::AbortAwarePendingStart);

    let active_runner = Arc::clone(&runner);
    let run = tokio::spawn(async move {
        active_runner
            .run_with_strategy(request(), RunStrategy::Direct)
            .await
    });
    let entered = provider
        .stream_start_entered
        .acquire()
        .await
        .expect("pending stream startup semaphore open");
    entered.forget();

    run.abort();
    assert!(run
        .await
        .expect_err("aborted task must not complete")
        .is_cancelled());
    assert!(provider
        .stream_start_cancellation
        .lock()
        .expect("stream start cancellation lock")
        .as_ref()
        .expect("provider observed model cancellation token")
        .is_cancelled());

    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.mode, SpeculationMode::Active);
    assert_eq!(metrics.issued, 0);
    assert_eq!(metrics.in_flight, 0);
}

#[tokio::test]
async fn dedicated_commit_policy_error_executes_same_authorized_prepare_once() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let policy = Arc::new(FailingDedicatedCommitPolicy {
        ordinary_calls: AtomicUsize::new(0),
        speculative_calls: AtomicUsize::new(0),
    });
    let approvals = Arc::new(CountingApproval::new(true));
    let events = Arc::new(InMemoryEventSink::default());
    let runner = AgentRunner::builder(provider)
        .tools(registry(tool.clone()))
        .policy(policy.clone())
        .approvals(approvals.clone())
        .event_sink(events.clone())
        .speculation(SpeculationConfig::default())
        .build();
    train_and_activate(&runner).await;
    let event_start = events.events().len();

    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("dedicated failure falls back without re-prepare");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(policy.ordinary_calls.load(Ordering::SeqCst), 1_002);
    assert_eq!(policy.speculative_calls.load(Ordering::SeqCst), 2);
    assert_eq!(approvals.calls.load(Ordering::SeqCst), 1);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1_002);
    let active_events = events.events()[event_start..].to_vec();
    assert_eq!(
        active_events
            .iter()
            .filter(|record| matches!(record.event, RunEvent::PolicyDecided { .. }))
            .count(),
        1
    );
    assert_eq!(
        active_events
            .iter()
            .filter(|record| matches!(record.event, RunEvent::ApprovalRequested { .. }))
            .count(),
        1
    );
    assert_eq!(
        active_events
            .iter()
            .filter(|record| matches!(record.event, RunEvent::ToolCompleted { .. }))
            .count(),
        1
    );
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.discarded, 1);
    assert_eq!(metrics.mode, SpeculationMode::Shadow);
}

#[tokio::test]
async fn post_issue_assembly_error_cancels_provider_stream_before_drop() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .speculation(SpeculationConfig::default())
        .build();
    train_and_activate(&runner).await;
    provider.set_behavior(StreamBehavior::CancellationAwareOversizedText);
    let mut active_request = request();
    active_request.agent.limits.max_model_response_bytes = 128;

    let result = runner
        .run_with_strategy(active_request, RunStrategy::Direct)
        .await
        .expect("assembly failure is represented");

    assert_eq!(result.status, RunStatus::LimitReached);
    assert!(provider.stream_drop_saw_cancellation.load(Ordering::SeqCst));
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.discarded, 1);
    assert_eq!(metrics.committed, 0);
    assert_eq!(metrics.mode, SpeculationMode::Shadow);
}

#[tokio::test(start_paused = true)]
async fn active_polls_pull_driven_tail_while_tool_is_pending_and_overlaps_latency() {
    let active_provider = Arc::new(StreamingProvider::new(StreamBehavior::PullDrivenDelayed));
    let active_tool = Arc::new(CountingTool::eligible());
    let active_runner = Arc::new(
        AgentRunner::builder(active_provider.clone())
            .tools(registry(active_tool.clone()))
            .policy(Arc::new(TestPolicy::new(true)))
            .speculation(SpeculationConfig::default())
            .build(),
    );
    train_and_activate(&active_runner).await;
    active_provider.set_response_delay(10, true);
    active_tool.set_delay(10, true);

    let active_started = tokio::time::Instant::now();
    let active_run = {
        let runner = Arc::clone(&active_runner);
        tokio::spawn(async move {
            runner
                .run_with_strategy(request(), RunStrategy::Direct)
                .await
        })
    };
    let tool_entered = active_tool
        .delay_entered
        .acquire()
        .await
        .expect("delayed tool entry semaphore open");
    tool_entered.forget();
    tokio::task::yield_now().await;
    assert_eq!(
        active_provider.tail_polled.available_permits(),
        1,
        "the provider tail must be polled before the blocked tool completes"
    );
    let tail_polled = active_provider
        .tail_polled
        .try_acquire()
        .expect("tail poll signal is available");
    tail_polled.forget();
    let in_flight = active_runner.speculation_metrics(TOOL_ID);
    assert_eq!(in_flight.issued, 1);
    assert_eq!(in_flight.in_flight, 1);
    assert_eq!(
        in_flight.issued,
        in_flight.in_flight + in_flight.committed + in_flight.discarded + in_flight.cancelled
    );

    tokio::time::advance(Duration::from_millis(10)).await;
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    let active_result = active_run
        .await
        .expect("active run task completes")
        .expect("active overlap run succeeds");
    let active_elapsed = tokio::time::Instant::now() - active_started;
    assert_eq!(active_result.status, RunStatus::Completed);
    assert_eq!(active_elapsed, Duration::from_millis(10));
    let active_metrics = active_runner.speculation_metrics(TOOL_ID);
    assert_eq!(active_metrics.in_flight, 0);
    assert_eq!(active_metrics.committed, 1);
    assert_eq!(active_metrics.execution_duration_ms.count, 1);
    assert_eq!(active_metrics.publication_wait_ms.count, 1);

    let shadow_provider = Arc::new(StreamingProvider::new(StreamBehavior::PullDrivenDelayed));
    shadow_provider.set_response_delay(10, false);
    let shadow_tool = Arc::new(CountingTool::eligible());
    shadow_tool.set_delay(10, false);
    let shadow_runner = Arc::new(
        AgentRunner::builder(shadow_provider)
            .tools(registry(shadow_tool))
            .policy(Arc::new(TestPolicy::new(true)))
            .speculation(SpeculationConfig::default())
            .build(),
    );
    let shadow_started = tokio::time::Instant::now();
    let shadow_run = {
        let runner = Arc::clone(&shadow_runner);
        tokio::spawn(async move {
            runner
                .run_with_strategy(request(), RunStrategy::Direct)
                .await
        })
    };
    tokio::task::yield_now().await;
    tokio::time::advance(Duration::from_millis(10)).await;
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    assert!(!shadow_run.is_finished());
    tokio::time::advance(Duration::from_millis(10)).await;
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        shadow_run
            .await
            .expect("shadow run task completes")
            .expect("shadow run succeeds")
            .status,
        RunStatus::Completed
    );
    let shadow_elapsed = tokio::time::Instant::now() - shadow_started;

    let disabled_provider = Arc::new(StreamingProvider::new(StreamBehavior::PullDrivenDelayed));
    disabled_provider.set_response_delay(10, false);
    let disabled_tool = Arc::new(CountingTool::eligible());
    disabled_tool.set_delay(10, false);
    let disabled_runner = Arc::new(
        AgentRunner::builder(disabled_provider)
            .tools(registry(disabled_tool))
            .policy(Arc::new(TestPolicy::new(true)))
            .build(),
    );
    let disabled_started = tokio::time::Instant::now();
    let disabled_run = {
        let runner = Arc::clone(&disabled_runner);
        tokio::spawn(async move {
            runner
                .run_with_strategy(request(), RunStrategy::Direct)
                .await
        })
    };
    tokio::task::yield_now().await;
    tokio::time::advance(Duration::from_millis(10)).await;
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    assert!(!disabled_run.is_finished());
    tokio::time::advance(Duration::from_millis(10)).await;
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        disabled_run
            .await
            .expect("disabled run task completes")
            .expect("disabled run succeeds")
            .status,
        RunStatus::Completed
    );
    let disabled_elapsed = tokio::time::Instant::now() - disabled_started;

    assert_eq!(shadow_elapsed, Duration::from_millis(20));
    assert_eq!(disabled_elapsed, Duration::from_millis(20));
    assert!(active_elapsed < shadow_elapsed);
    assert!(active_elapsed < disabled_elapsed);
}

#[tokio::test]
async fn terminal_stream_error_cancels_and_drains_a_pending_attempt() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CancellationAwareTool::new());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .speculation(SpeculationConfig::default())
        .build();
    train_and_activate(&runner).await;
    provider.set_behavior(StreamBehavior::FailAfterCandidate);

    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("terminal stream error is represented");

    assert_eq!(result.status, RunStatus::Failed);
    assert_eq!(provider.stream_calls.load(Ordering::SeqCst), 1_001);
    assert_eq!(tool.observed_cancellation.load(Ordering::SeqCst), 1);
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 1);
    assert_eq!(metrics.in_flight, 0);
    assert_eq!(metrics.cancelled, 1);
    assert_eq!(metrics.terminal_stream_failures, 1);
    assert_eq!(metrics.mode, SpeculationMode::Shadow);
}

#[tokio::test]
async fn committed_candidate_releases_key_before_slow_tool_completed_sink() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let mut keyed = CountingTool::eligible();
    keyed.definition.concurrency_key = Some("publication-key".into());
    let tool = Arc::new(keyed);
    let events = Arc::new(BlockingFirstCompletionSink::new());
    let runner = Arc::new(
        AgentRunner::builder(provider)
            .tools(registry(tool.clone()))
            .policy(Arc::new(TestPolicy::new(true)))
            .event_sink(events.clone())
            .speculation(SpeculationConfig::default())
            .build(),
    );
    train_and_activate(&runner).await;
    events.arm();

    let first = {
        let runner = Arc::clone(&runner);
        std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("first-run test runtime builds")
                .block_on(runner.run_with_strategy(request(), RunStrategy::Direct))
        })
    };
    let entered = events
        .entered
        .acquire()
        .await
        .expect("first completion sink entry semaphore open");
    entered.forget();

    let second_result = tokio::time::timeout(
        Duration::from_secs(2),
        runner.run_with_strategy(request(), RunStrategy::Direct),
    )
    .await
    .expect("second same-key run must not wait for first event sink")
    .expect("second same-key run succeeds");
    assert_eq!(second_result.status, RunStatus::Completed);
    assert!(!first.is_finished());

    events.release();
    assert_eq!(
        first
            .join()
            .expect("first run thread completes")
            .expect("first same-key run succeeds")
            .status,
        RunStatus::Completed
    );
    let metrics = runner.speculation_metrics(TOOL_ID);
    assert_eq!(metrics.issued, 2);
    assert_eq!(metrics.committed, 2);
    assert_eq!(metrics.in_flight, 0);
    assert_eq!(metrics.key_saturated, 0);
}

#[tokio::test]
async fn stream_event_and_run_argument_limits_prevent_dispatch() {
    let flood_provider = Arc::new(StreamingProvider::new(StreamBehavior::EmptyFlood));
    let flood_tool = Arc::new(CountingTool::eligible());
    let flood_runner = AgentRunner::builder(flood_provider.clone())
        .tools(registry(flood_tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .speculation(SpeculationConfig {
            max_stream_events: 2,
            ..SpeculationConfig::default()
        })
        .build();
    let flood = flood_runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("event flood becomes a bounded run result");
    assert_eq!(flood.status, RunStatus::LimitReached);
    assert_eq!(flood_tool.calls.load(Ordering::SeqCst), 0);
    assert_eq!(flood_provider.stream_calls.load(Ordering::SeqCst), 1);

    let limited_provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let limited_tool = Arc::new(CountingTool::eligible());
    let limited_runner = AgentRunner::builder(limited_provider)
        .tools(registry(limited_tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .speculation(SpeculationConfig::default())
        .build();
    let mut limited_request = request();
    limited_request.agent.limits.max_tool_arguments_bytes = 2;
    let limited = limited_runner
        .run_with_strategy(limited_request, RunStrategy::Direct)
        .await
        .expect("argument overflow becomes a bounded run result");
    assert_eq!(limited.status, RunStatus::LimitReached);
    assert_eq!(limited_tool.calls.load(Ordering::SeqCst), 0);
    assert_eq!(limited_runner.speculation_metrics(TOOL_ID).issued, 0);
}

#[tokio::test]
async fn adaptive_planner_selected_direct_never_starts_speculation() {
    let provider = Arc::new(StreamingProvider::adaptive_direct());
    let tool = Arc::new(CountingTool::declarative_eligible());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .speculation(SpeculationConfig::default())
        .build();

    let result = runner
        .run(request())
        .await
        .expect("adaptive direct run succeeds");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(provider.complete_calls.load(Ordering::SeqCst), 3);
    assert_eq!(provider.stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        tool.callers.lock().unwrap().as_slice(),
        [ToolCaller::Direct]
    );
    assert_eq!(runner.speculation_metrics(TOOL_ID).shadow_matches, 0);
    assert_eq!(runner.speculation_metrics(TOOL_ID).issued, 0);
}

#[tokio::test]
async fn adaptive_capability_downgrade_never_starts_speculation() {
    let provider = Arc::new(StreamingProvider::new(StreamBehavior::Normal));
    let tool = Arc::new(CountingTool::eligible());
    let events = Arc::new(InMemoryEventSink::default());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .event_sink(events.clone())
        .speculation(SpeculationConfig::default())
        .build();

    let result = runner
        .run(request())
        .await
        .expect("adaptive capability downgrade succeeds sequentially");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(provider.complete_calls.load(Ordering::SeqCst), 2);
    assert_eq!(provider.stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    assert_eq!(runner.speculation_metrics(TOOL_ID).issued, 0);
    assert!(events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyFallback {
            reason: llama_harness_core::StrategyFallbackReason::UnsupportedCapability,
            ..
        }
    )));
}

#[tokio::test]
async fn declarative_execution_never_streams_or_registers_speculation() {
    let provider = Arc::new(StreamingProvider::declarative());
    let tool = Arc::new(CountingTool::declarative_eligible());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .policy(Arc::new(TestPolicy::new(true)))
        .speculation(SpeculationConfig::default())
        .build();

    let result = runner
        .run_with_strategy(request(), RunStrategy::DeclarativePlan)
        .await
        .expect("declarative run succeeds");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(provider.complete_calls.load(Ordering::SeqCst), 2);
    assert_eq!(provider.stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        tool.callers.lock().unwrap().as_slice(),
        [ToolCaller::DeclarativePlan]
    );
    assert_eq!(
        runner.speculation_readiness(TOOL_ID).mode,
        SpeculationMode::Disabled
    );
}
