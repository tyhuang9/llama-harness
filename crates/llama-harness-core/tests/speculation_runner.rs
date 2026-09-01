use async_trait::async_trait;
use futures_util::stream;
use llama_harness_core::{
    AgentDefinition, AgentRunner, AllowAllPolicy, ApprovalHandler, ApprovalRecord,
    CancellationSafety, EventRecord, ExecutionLocation, HarnessError, InMemoryEventSink,
    IssueSafety, MessageRole, ModelCapabilities, ModelEventStream, ModelInfo, ModelProvider,
    ModelRequest, ModelResponse, ModelStreamEvent, NetworkEgress, PolicyDecision, PolicyEngine,
    ProviderCapabilityLimits, ProviderHealth, RunEvent, RunRequest, RunStatus, RunStrategy,
    SpeculationConfig, SpeculationMode, SpeculationPolicy, Tool, ToolCallContext, ToolCallDelta,
    ToolCaller, ToolDefinition, ToolRegistry, ToolResult, ToolRisk, Usage,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

const TOOL_ID: &str = "local.read";
const PRIVACY_CANARY: &str = "speculation-private-canary";

#[derive(Clone, Copy)]
enum StreamBehavior {
    Normal,
    FailAfterCandidate,
    EmptyFlood,
    InterleavedMultiple,
}

struct StreamingProvider {
    complete_calls: AtomicUsize,
    stream_calls: AtomicUsize,
    behavior: StreamBehavior,
    planner_response: Option<&'static str>,
}

impl StreamingProvider {
    fn new(behavior: StreamBehavior) -> Self {
        Self {
            complete_calls: AtomicUsize::new(0),
            stream_calls: AtomicUsize::new(0),
            behavior,
            planner_response: None,
        }
    }

    fn adaptive_direct() -> Self {
        Self {
            planner_response: Some(r#"{"strategy":"direct"}"#),
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
        let ordinal = self.complete_calls.fetch_add(1, Ordering::SeqCst);
        if ordinal == 0 {
            if let Some(response) = self.planner_response {
                return Ok(ModelResponse::new(request.model).with_final_output(response));
            }
        }
        Ok(if Self::requests_tool(&request) {
            Self::tool_response(request.model)
        } else {
            ModelResponse::new(request.model).with_final_output("done")
        })
    }

    async fn stream(&self, request: ModelRequest) -> Result<ModelEventStream, HarnessError> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        let requests_tool = Self::requests_tool(&request);
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
            match self.behavior {
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
        Ok(ToolResult::success(json!({
            "value": if caller == ToolCaller::Speculative { "speculative" } else { "direct" }
        })))
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
}

impl CancellationAwareTool {
    fn new() -> Self {
        Self {
            definition: CountingTool::eligible().definition,
            calls: AtomicUsize::new(0),
            observed_cancellation: AtomicUsize::new(0),
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
        cancellation.cancelled().await;
        self.observed_cancellation.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success(json!({"value":"direct"})))
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
    assert_eq!(provider.complete_calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.stream_calls.load(Ordering::SeqCst), 2);
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
    let runner = AgentRunner::builder(provider)
        .tools(registry(tool.clone()))
        .policy(policy.clone())
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

    let result = runner
        .run_with_strategy(request(), RunStrategy::Direct)
        .await
        .expect("active run succeeds");
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.tool_calls.len(), 1);
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
async fn adaptive_direct_uses_the_same_shadow_stream_helper() {
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
    assert_eq!(provider.complete_calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.stream_calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        tool.callers.lock().unwrap().as_slice(),
        [ToolCaller::Direct]
    );
    assert_eq!(runner.speculation_metrics(TOOL_ID).shadow_matches, 1);
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
