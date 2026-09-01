#![cfg(feature = "programmatic")]

use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, MockModelProvider},
    AgentDefinition, AgentRunner, AllowAllPolicy, ApprovalHandler, ApprovalRecord, HarnessError,
    InMemoryEventSink, ModelCapabilities, ModelInfo, ModelProvider, ModelRequest, ModelResponse,
    PolicyDecision, PolicyEngine, ProgramLifecycleOutcome, ProgrammaticConformance,
    ProgrammaticHostConfig, ProviderCapabilityLimits, ProviderHealth, RunEvent, RunRequest,
    RunResult, RunStatus, RunStrategy, SpeculationConfig, SpeculationMode, StrategyFallbackReason,
    Tool, ToolCallContext, ToolCaller, ToolDefinition, ToolRegistry, ToolResult, ToolRisk,
    HARD_MAX_PROGRAMMATIC_FANOUT_CONCURRENCY, HARD_MAX_PROGRAMMATIC_PROGRAM_BYTES,
};
use llama_harness_programmatic_sandbox::{SandboxLimits, HARD_LIMITS};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc, Mutex,
};
use std::time::Duration;
use tokio::sync::{Notify, Semaphore};
use tokio_util::sync::CancellationToken;

fn capabilities() -> ModelCapabilities {
    ModelCapabilities::new(true, false, true)
        .with_programmatic_conformance(ProgrammaticConformance::StrictJsonAstV1)
        .with_limits(ProviderCapabilityLimits::new().with_max_program_bytes(64 * 1024))
}

fn request(tool_ids: &[&str]) -> RunRequest {
    let mut agent = AgentDefinition::new("programmatic", "Programmatic", "1", "mock-model");
    agent.tool_allowlist = tool_ids.iter().map(|id| (*id).into()).collect();
    agent.limits.max_model_calls = 4;
    RunRequest::new(agent, "complete the task").with_run_id("programmatic-test-run")
}

struct HostDeadlineFallbackProvider {
    calls: AtomicU32,
    fallback_started: Arc<Notify>,
}

struct ConcurrentProgramProvider {
    program: String,
}

struct AdmissionProgramProvider {
    program: String,
    generated: Arc<Semaphore>,
}

struct SynthesisBarrierProvider {
    program: String,
    planning_calls: AtomicU32,
    synthesis_entered: Arc<Semaphore>,
    synthesis_releases: Arc<Semaphore>,
}

#[async_trait]
impl ModelProvider for ConcurrentProgramProvider {
    fn id(&self) -> &str {
        "concurrent-program"
    }

    fn capabilities(&self) -> ModelCapabilities {
        capabilities()
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth::healthy())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(vec![
            ModelInfo::new("mock-model").with_capabilities(capabilities())
        ])
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError> {
        Ok(if request.tools.is_empty() {
            ModelResponse::new("mock-model").with_final_output("concurrent done")
        } else {
            ModelResponse::new("mock-model").with_final_output(self.program.clone())
        })
    }
}

#[async_trait]
impl ModelProvider for AdmissionProgramProvider {
    fn id(&self) -> &str {
        "admission-program"
    }

    fn capabilities(&self) -> ModelCapabilities {
        capabilities()
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth::healthy())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(vec![
            ModelInfo::new("mock-model").with_capabilities(capabilities())
        ])
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError> {
        if request.tools.is_empty() {
            Ok(ModelResponse::new("mock-model").with_final_output("admission done"))
        } else {
            self.generated.add_permits(1);
            Ok(ModelResponse::new("mock-model").with_final_output(self.program.clone()))
        }
    }
}

#[async_trait]
impl ModelProvider for SynthesisBarrierProvider {
    fn id(&self) -> &str {
        "synthesis-barrier"
    }

    fn capabilities(&self) -> ModelCapabilities {
        capabilities()
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth::healthy())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(vec![
            ModelInfo::new("mock-model").with_capabilities(capabilities())
        ])
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError> {
        if request.tools.is_empty() {
            self.synthesis_entered.add_permits(1);
            let permit = self
                .synthesis_releases
                .acquire()
                .await
                .expect("synthesis barrier remains open");
            permit.forget();
            Ok(ModelResponse::new("mock-model").with_final_output("synthesized"))
        } else {
            self.planning_calls.fetch_add(1, Ordering::SeqCst);
            Ok(ModelResponse::new("mock-model").with_final_output(self.program.clone()))
        }
    }
}

#[async_trait]
impl ModelProvider for HostDeadlineFallbackProvider {
    fn id(&self) -> &str {
        "host-deadline-fallback"
    }

    fn capabilities(&self) -> ModelCapabilities {
        capabilities()
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth::healthy())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(vec![
            ModelInfo::new("mock-model").with_capabilities(capabilities())
        ])
    }

    async fn complete(&self, _: ModelRequest) -> Result<ModelResponse, HarnessError> {
        match self.calls.fetch_add(1, Ordering::SeqCst) {
            0 | 1 => Ok(ModelResponse::new("mock-model").with_final_output("not a program")),
            2 => {
                self.fallback_started.notify_waiters();
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok(ModelResponse::new("mock-model").with_final_output("late direct answer"))
            }
            _ => Err(HarnessError::Provider("unexpected model request".into())),
        }
    }
}

fn return_program_with_exact_serialized_bytes(bytes: usize) -> String {
    let empty = json!({"version":1,"body":[{
        "kind":"return",
        "value":{"kind":"string","value":""}
    }]})
    .to_string();
    assert!(bytes >= empty.len());
    let source = json!({"version":1,"body":[{
        "kind":"return",
        "value":{"kind":"string","value":"x".repeat(bytes - empty.len())}
    }]})
    .to_string();
    assert_eq!(source.len(), bytes);
    source
}

fn two_read_fanout_program() -> String {
    read_fanout_program(&[1, 2])
}

fn read_fanout_program(values: &[u64]) -> String {
    json!({"version":1,"body":[
        {"kind":"fan_out","name":"results","tool_id":"read","item":"i",
         "collection":{"kind":"array","items":values.iter().map(|value| json!({"kind":"integer","value":value})).collect::<Vec<_>>()},
         "max_calls":values.len().max(1),
         "arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"variable","name":"i"}}]}},
        {"kind":"return","value":{"kind":"variable","name":"results"}}
    ]})
    .to_string()
}

struct CountingTool {
    definition: ToolDefinition,
    calls: AtomicU32,
}

struct AdmissionBarrierTool {
    definition: ToolDefinition,
    calls: AtomicU32,
    entered: Arc<Semaphore>,
    releases: Arc<Semaphore>,
}

impl AdmissionBarrierTool {
    fn new() -> Self {
        Self {
            definition: ToolDefinition::new(
                "read",
                "read",
                "suspends programmatic VMs for admission tests",
                json!({"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]}),
            )
            .with_risk(ToolRisk::Low)
            .with_read_only(true)
            .with_parallel_safe(true)
            .with_allowed_callers([ToolCaller::Programmatic]),
            calls: AtomicU32::new(0),
            entered: Arc::new(Semaphore::new(0)),
            releases: Arc::new(Semaphore::new(0)),
        }
    }
}

#[async_trait]
impl Tool for AdmissionBarrierTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(
        &self,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.add_permits(1);
        tokio::select! {
            _ = cancellation.cancelled() => Ok(ToolResult::success(json!({"cancelled": true}))),
            permit = self.releases.acquire() => {
                permit.expect("admission test release semaphore remains open").forget();
                Ok(ToolResult::success(arguments))
            }
        }
    }
}

struct ReentrantProgrammaticTool {
    definition: ToolDefinition,
    calls: AtomicU32,
    runner: Mutex<Option<Arc<AgentRunner>>>,
    nested_result: Mutex<Option<RunResult>>,
    nested_finished: Arc<Notify>,
}

impl ReentrantProgrammaticTool {
    fn new() -> Self {
        Self {
            definition: ToolDefinition::new(
                "reenter",
                "reenter",
                "starts a nested programmatic run",
                json!({"type":"object","additionalProperties":false}),
            )
            .with_risk(ToolRisk::Low)
            .with_read_only(true)
            .with_parallel_safe(true)
            .with_allowed_callers([ToolCaller::Programmatic]),
            calls: AtomicU32::new(0),
            runner: Mutex::new(None),
            nested_result: Mutex::new(None),
            nested_finished: Arc::new(Notify::new()),
        }
    }

    fn attach_runner(&self, runner: Arc<AgentRunner>) {
        *self.runner.lock().unwrap() = Some(runner);
    }
}

#[async_trait]
impl Tool for ReentrantProgrammaticTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let runner = self
            .runner
            .lock()
            .unwrap()
            .clone()
            .expect("reentrant test tool is attached to a runner");
        let nested_result = runner
            .run_with_strategy(
                request(&[]).with_run_id("reentrant-nested-programmatic"),
                RunStrategy::Programmatic,
            )
            .await?;
        *self.nested_result.lock().unwrap() = Some(nested_result);
        self.nested_finished.notify_one();
        Ok(ToolResult::success(json!({"nested_run_finished": true})))
    }
}

impl CountingTool {
    fn new(id: &str, read_only: bool, parallel_safe: bool) -> Self {
        Self {
            definition: ToolDefinition::new(
                id,
                id,
                "test tool",
                json!({"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]}),
            )
            .with_risk(ToolRisk::Low)
            .with_read_only(read_only)
            .with_parallel_safe(parallel_safe)
            .with_allowed_callers([ToolCaller::Programmatic]),
            calls: AtomicU32::new(0),
        }
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
        _: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success(arguments))
    }
}

struct RecordingTool {
    definition: ToolDefinition,
    result: ToolResult,
    calls: AtomicU32,
    contexts: Mutex<Vec<ToolCallContext>>,
}

impl RecordingTool {
    fn new(
        id: &str,
        result: ToolResult,
        read_only: bool,
        parallel_safe: bool,
        callers: impl IntoIterator<Item = ToolCaller>,
    ) -> Self {
        Self {
            definition: ToolDefinition::new(
                id,
                id,
                "recording test tool",
                json!({"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]}),
            )
            .with_risk(ToolRisk::Low)
            .with_read_only(read_only)
            .with_parallel_safe(parallel_safe)
            .with_allowed_callers(callers),
            result,
            calls: AtomicU32::new(0),
            contexts: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Tool for RecordingTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(
        &self,
        arguments: Value,
        _: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success(arguments))
    }

    async fn execute_with_context(
        &self,
        context: &ToolCallContext,
        _: Value,
        _: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.contexts.lock().unwrap().push(context.clone());
        Ok(self.result.clone())
    }
}

struct CancellationBarrierTool {
    definition: ToolDefinition,
    calls: AtomicU32,
    entered: Arc<Notify>,
    observed_cancellation: AtomicBool,
}

impl CancellationBarrierTool {
    fn mutation(id: &str) -> Self {
        Self {
            definition: ToolDefinition::new(
                id,
                id,
                "cooperatively cancellable mutation",
                json!({"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]}),
            )
            .with_risk(ToolRisk::High)
            .with_allowed_callers([ToolCaller::Programmatic]),
            calls: AtomicU32::new(0),
            entered: Arc::new(Notify::new()),
            observed_cancellation: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl Tool for CancellationBarrierTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(
        &self,
        _: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.notify_waiters();
        cancellation.cancelled().await;
        self.observed_cancellation.store(true, Ordering::SeqCst);
        Ok(ToolResult::success(json!({"cancelled": true})))
    }
}

struct ControlledFanoutTool {
    definition: ToolDefinition,
    calls: AtomicU32,
    entered: Arc<Semaphore>,
    releases: Vec<Arc<Semaphore>>,
    finished: Arc<Semaphore>,
    active: AtomicU32,
    maximum: AtomicU32,
    completed: Mutex<Vec<u64>>,
}

impl ControlledFanoutTool {
    fn new(call_count: usize) -> Self {
        Self {
            definition: ToolDefinition::new(
                "read",
                "read",
                "controlled parallel read",
                json!({"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]}),
            )
            .with_risk(ToolRisk::Low)
            .with_read_only(true)
            .with_parallel_safe(true)
            .with_allowed_callers([ToolCaller::Programmatic]),
            calls: AtomicU32::new(0),
            entered: Arc::new(Semaphore::new(0)),
            releases: (0..call_count)
                .map(|_| Arc::new(Semaphore::new(0)))
                .collect(),
            finished: Arc::new(Semaphore::new(0)),
            active: AtomicU32::new(0),
            maximum: AtomicU32::new(0),
            completed: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Tool for ControlledFanoutTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(
        &self,
        arguments: Value,
        _: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        let value = arguments["value"]
            .as_u64()
            .expect("test fanout argument must be an integer");
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.maximum.fetch_max(active, Ordering::SeqCst);
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.add_permits(1);
        let permit = self.releases[value as usize - 1]
            .acquire()
            .await
            .expect("test fanout release semaphore remains open");
        permit.forget();
        self.completed.lock().unwrap().push(value);
        self.finished.add_permits(1);
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(ToolResult::success(arguments))
    }
}

struct CancellationFanoutTool {
    definition: ToolDefinition,
    calls: AtomicU32,
    entered: Arc<Semaphore>,
    observed_cancellation: AtomicU32,
}

struct NonCooperativeMutationTool {
    definition: ToolDefinition,
    calls: AtomicU32,
    entered: Arc<Notify>,
}

impl NonCooperativeMutationTool {
    fn new() -> Self {
        Self {
            definition: ToolDefinition::new(
                "write",
                "write",
                "non-cooperative mutation",
                json!({"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]}),
            )
            .with_risk(ToolRisk::High)
            .with_allowed_callers([ToolCaller::Programmatic]),
            calls: AtomicU32::new(0),
            entered: Arc::new(Notify::new()),
        }
    }
}

#[async_trait]
impl Tool for NonCooperativeMutationTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.notify_waiters();
        std::future::pending().await
    }
}

impl CancellationFanoutTool {
    fn new() -> Self {
        Self {
            definition: ToolDefinition::new(
                "read",
                "read",
                "cooperatively cancellable fanout read",
                json!({"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]}),
            )
            .with_risk(ToolRisk::Low)
            .with_read_only(true)
            .with_parallel_safe(true)
            .with_allowed_callers([ToolCaller::Programmatic]),
            calls: AtomicU32::new(0),
            entered: Arc::new(Semaphore::new(0)),
            observed_cancellation: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl Tool for CancellationFanoutTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(
        &self,
        _: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.add_permits(1);
        cancellation.cancelled().await;
        self.observed_cancellation.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success(json!({"cancelled": true})))
    }
}

struct ApprovalOnWrite {
    seen_arguments: Mutex<Vec<Value>>,
    granted: bool,
}

#[async_trait]
impl PolicyEngine for ApprovalOnWrite {
    async fn decide(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(if tool.id == "write" {
            PolicyDecision::RequireApproval {
                reason: "write requires approval".into(),
            }
        } else {
            PolicyDecision::Allow {
                reason: "read is allowed".into(),
            }
        })
    }
}

#[async_trait]
impl ApprovalHandler for ApprovalOnWrite {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        arguments: &Value,
        _: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        self.seen_arguments.lock().unwrap().push(arguments.clone());
        Ok(ApprovalRecord::new(
            "untrusted-call-id",
            tool.id.clone(),
            self.granted,
            "test approval",
        ))
    }
}

#[tokio::test]
async fn forced_programmatic_runs_model_program_broker_and_final_synthesis() {
    let program = json!({"version":1,"body":[
        {"kind":"invoke","name":"result","tool_id":"write","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":7}}]}},
        {"kind":"return","value":{"kind":"variable","name":"result"}}
    ]});
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(serde_json::to_string(&program).unwrap()),
            final_response("finished"),
        ])
        .with_capabilities(capabilities()),
    );
    let tool = Arc::new(CountingTool::new("write", false, false));
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry)
        .policy(Arc::new(AllowAllPolicy))
        .programmatic(ProgrammaticHostConfig::default())
        .build();

    let result = runner
        .run_with_strategy(request(&["write"]), RunStrategy::Programmatic)
        .await
        .unwrap();
    assert!(matches!(result.status, RunStatus::Completed));
    assert_eq!(result.final_output.as_deref(), Some("finished"));
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.tool_calls.len(), 1);
    assert!(result.tool_calls[0].id.starts_with("programmatic-0-"));
    assert_eq!(result.tool_calls[0].arguments_json, r#"{"value":7}"#);
    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].tools.len(), 1);
    assert!(requests[1].tools.is_empty());
}

#[tokio::test]
async fn programmatic_generation_uses_bounded_completion_even_when_streaming_is_advertised() {
    let program = json!({"version":1,"body":[
        {"kind":"return","value":{"kind":"string","value":"safe"}}
    ]});
    // `MockModelProvider` only implements `complete`; its trait-default `stream` fails.
    // A successful run with streaming advertised therefore proves this path remains bounded
    // completion-based until a distinct provider streaming contract is introduced.
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(serde_json::to_string(&program).unwrap()),
            final_response("done"),
        ])
        .with_capabilities(
            ModelCapabilities::new(true, true, true)
                .with_programmatic_conformance(ProgrammaticConformance::StrictJsonAstV1)
                .with_limits(ProviderCapabilityLimits::new().with_max_program_bytes(64 * 1024)),
        ),
    );
    let runner = AgentRunner::builder(provider.clone())
        .speculation(SpeculationConfig::default())
        .programmatic(ProgrammaticHostConfig::default())
        .build();
    let result = runner
        .run_with_strategy(request(&[]), RunStrategy::Programmatic)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(provider.requests().len(), 2);
    assert_eq!(
        runner.speculation_readiness("read").mode,
        SpeculationMode::Disabled
    );
    assert_eq!(runner.speculation_metrics("read").issued, 0);
}

#[tokio::test]
async fn identical_mutations_are_distinct_program_occurrences() {
    let program = json!({"version":1,"body":[
        {"kind":"invoke","name":"first","tool_id":"write","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":7}}]}},
        {"kind":"invoke","name":"second","tool_id":"write","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":7}}]}},
        {"kind":"return","value":{"kind":"array","items":[{"kind":"variable","name":"first"},{"kind":"variable","name":"second"}]}}
    ]});
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(serde_json::to_string(&program).unwrap()),
            final_response("finished"),
        ])
        .with_capabilities(capabilities()),
    );
    let tool = Arc::new(CountingTool::new("write", false, false));
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let runner = AgentRunner::builder(provider)
        .tools(registry)
        .policy(Arc::new(AllowAllPolicy))
        .programmatic(ProgrammaticHostConfig::default())
        .build();
    let result = runner
        .run_with_strategy(request(&["write"]), RunStrategy::Programmatic)
        .await
        .unwrap();
    assert!(matches!(result.status, RunStatus::Completed));
    assert_eq!(tool.calls.load(Ordering::SeqCst), 2);
    assert_ne!(result.tool_calls[0].id, result.tool_calls[1].id);
}

#[tokio::test]
async fn repeated_public_run_ids_still_receive_distinct_programmatic_effect_nonces() {
    let program = json!({"version":1,"body":[
        {"kind":"invoke","name":"write","tool_id":"write","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":7}}]}},
        {"kind":"return","value":{"kind":"variable","name":"write"}}
    ]})
    .to_string();
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(program.clone()),
            final_response("first"),
            final_response(program),
            final_response("second"),
        ])
        .with_capabilities(capabilities()),
    );
    let tool = Arc::new(RecordingTool::new(
        "write",
        ToolResult::success(json!({"value": 7})),
        false,
        false,
        [ToolCaller::Programmatic],
    ));
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let runner = AgentRunner::builder(provider)
        .tools(registry)
        .policy(Arc::new(AllowAllPolicy))
        .programmatic(ProgrammaticHostConfig::default())
        .build();

    let first = runner
        .run_with_strategy(
            request(&["write"]).with_run_id("reused-public-run-id"),
            RunStrategy::Programmatic,
        )
        .await
        .unwrap();
    let second = runner
        .run_with_strategy(
            request(&["write"]).with_run_id("reused-public-run-id"),
            RunStrategy::Programmatic,
        )
        .await
        .unwrap();

    assert_eq!(tool.calls.load(Ordering::SeqCst), 2);
    assert_ne!(first.tool_calls[0].id, second.tool_calls[0].id);
    let first_nonce = &first.tool_calls[0].id[15..51];
    let second_nonce = &second.tool_calls[0].id[15..51];
    assert!(uuid::Uuid::parse_str(first_nonce).is_ok());
    assert!(uuid::Uuid::parse_str(second_nonce).is_ok());
    assert_ne!(first_nonce, second_nonce);
    let contexts = tool.contexts.lock().unwrap();
    assert_eq!(contexts.len(), 2);
    assert_eq!(contexts[0].run_id, contexts[1].run_id);
    assert_ne!(contexts[0].effect_key, contexts[1].effect_key);
}

#[tokio::test]
async fn concurrent_reused_public_run_ids_keep_full_execution_nonces_distinct() {
    let program = json!({"version":1,"body":[
        {"kind":"invoke","name":"write","tool_id":"write","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":7}}]}},
        {"kind":"return","value":{"kind":"variable","name":"write"}}
    ]})
    .to_string();
    let provider = Arc::new(ConcurrentProgramProvider { program });
    let tool = Arc::new(RecordingTool::new(
        "write",
        ToolResult::success(json!({"value": 7})),
        false,
        false,
        [ToolCaller::Programmatic],
    ));
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let runner = Arc::new(
        AgentRunner::builder(provider)
            .tools(registry)
            .policy(Arc::new(AllowAllPolicy))
            .programmatic(ProgrammaticHostConfig::default())
            .build(),
    );
    let first_runner = runner.clone();
    let first = tokio::spawn(async move {
        first_runner
            .run_with_strategy(
                request(&["write"]).with_run_id("concurrent-public-run-id"),
                RunStrategy::Programmatic,
            )
            .await
            .unwrap()
    });
    let second = tokio::spawn(async move {
        runner
            .run_with_strategy(
                request(&["write"]).with_run_id("concurrent-public-run-id"),
                RunStrategy::Programmatic,
            )
            .await
            .unwrap()
    });
    let (first, second) = tokio::join!(first, second);
    let first = first.unwrap();
    let second = second.unwrap();

    assert_eq!(tool.calls.load(Ordering::SeqCst), 2);
    assert_ne!(first.tool_calls[0].id, second.tool_calls[0].id);
    for call_id in [&first.tool_calls[0].id, &second.tool_calls[0].id] {
        assert!(uuid::Uuid::parse_str(&call_id[15..51]).is_ok());
    }
}

#[tokio::test(start_paused = true)]
async fn live_vm_admission_rejects_n_plus_one_and_reuses_released_capacity() {
    const MAX_ACTIVE_VMS: usize = 2;
    const PER_VM_LIVE_BYTES: usize = 4 * 1024 * 1024;
    let program = json!({"version":1,"body":[
        {"kind":"invoke","name":"result","tool_id":"read","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":1}}]}},
        {"kind":"return","value":{"kind":"variable","name":"result"}}
    ]})
    .to_string();
    let generated = Arc::new(Semaphore::new(0));
    let provider = Arc::new(AdmissionProgramProvider {
        program,
        generated: generated.clone(),
    });
    let tool = Arc::new(AdmissionBarrierTool::new());
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let events = Arc::new(InMemoryEventSink::default());
    let config = ProgrammaticHostConfig {
        limits: SandboxLimits {
            max_live_bytes: PER_VM_LIVE_BYTES,
            max_cumulative_bytes: 16 * 1024 * 1024,
            ..SandboxLimits::default()
        },
        max_active_vms: MAX_ACTIVE_VMS,
        ..ProgrammaticHostConfig::default()
    };
    let runner = Arc::new(
        AgentRunner::builder(provider)
            .tools(registry)
            .event_sink(events.clone())
            .programmatic(config)
            .build(),
    );

    let first_runner = runner.clone();
    let mut first = tokio::spawn(async move {
        first_runner
            .run_with_strategy(
                request(&["read"]).with_run_id("live-vm-first"),
                RunStrategy::Programmatic,
            )
            .await
            .unwrap()
    });
    let second_runner = runner.clone();
    let mut second = tokio::spawn(async move {
        second_runner
            .run_with_strategy(
                request(&["read"]).with_run_id("live-vm-second"),
                RunStrategy::Programmatic,
            )
            .await
            .unwrap()
    });
    for _ in 0..MAX_ACTIVE_VMS {
        tool.entered.acquire().await.unwrap().forget();
        generated.acquire().await.unwrap().forget();
    }

    let third_runner = runner.clone();
    let third = tokio::spawn(async move {
        third_runner
            .run_with_strategy(
                request(&["read"]).with_run_id("live-vm-third"),
                RunStrategy::Programmatic,
            )
            .await
            .unwrap()
    });
    let third = tokio::time::timeout(Duration::from_millis(1), third)
        .await
        .expect("the N+1 candidate must fail without waiting for a live VM slot")
        .unwrap();
    assert_eq!(third.status, RunStatus::LimitReached);
    assert!(third.final_output.is_none());
    assert_eq!(third.errors.len(), 1);
    assert_eq!(third.errors[0].code, "resource_limit");
    assert_eq!(tool.calls.load(Ordering::SeqCst), MAX_ACTIVE_VMS as u32);
    assert!(
        generated.try_acquire().is_err(),
        "the rejected candidate must not spend a planning model call"
    );

    let third_events = events
        .events()
        .into_iter()
        .filter(|record| record.run_id == "live-vm-third")
        .collect::<Vec<_>>();
    let lifecycle = third_events
        .iter()
        .filter_map(|record| match record.event {
            RunEvent::ProgramLifecycle { attempt, outcome } => Some((attempt, outcome)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle,
        vec![
            (1, ProgramLifecycleOutcome::Started),
            (1, ProgramLifecycleOutcome::LimitReached),
        ]
    );
    assert!(third_events.iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyUsage {
            strategy: RunStrategy::Programmatic,
            model_calls: 0,
            planning_model_calls: 0,
            repair_model_calls: 0,
            final_synthesis_model_calls: 0,
            tool_calls: 0,
            tool_issued: 0,
            ..
        }
    )));
    assert!(!third_events
        .iter()
        .any(|record| matches!(record.event, RunEvent::ProgramExecutionCompleted { .. })));
    assert_eq!(
        third_events
            .iter()
            .filter(|record| {
                matches!(
                    record.event,
                    RunEvent::Completed {
                        status: RunStatus::LimitReached
                    }
                )
            })
            .count(),
        1
    );

    tool.releases.add_permits(1);
    let first_finished = tokio::select! {
        result = &mut first => {
            let result = result.unwrap();
            assert_eq!(result.status, RunStatus::Completed, "{:?}", result.errors);
            true
        }
        result = &mut second => {
            let result = result.unwrap();
            assert_eq!(result.status, RunStatus::Completed, "{:?}", result.errors);
            false
        }
    };
    let replacement_runner = runner.clone();
    let replacement = tokio::spawn(async move {
        replacement_runner
            .run_with_strategy(
                request(&["read"]).with_run_id("live-vm-replacement"),
                RunStrategy::Programmatic,
            )
            .await
            .unwrap()
    });
    generated.acquire().await.unwrap().forget();
    tool.entered.acquire().await.unwrap().forget();
    assert_eq!(tool.calls.load(Ordering::SeqCst), 3);

    tool.releases.add_permits(2);
    if !first_finished {
        assert_eq!(first.await.unwrap().status, RunStatus::Completed);
    }
    if first_finished {
        assert_eq!(second.await.unwrap().status, RunStatus::Completed);
    }
    assert_eq!(replacement.await.unwrap().status, RunStatus::Completed);

    let peaks = events
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            RunEvent::ProgramExecutionCompleted {
                peak_accounted_bytes,
                ..
            } => Some(peak_accounted_bytes as usize),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(peaks.len(), 3);
    assert!(peaks.iter().all(|peak| *peak <= PER_VM_LIVE_BYTES));
    assert_eq!(
        MAX_ACTIVE_VMS * PER_VM_LIVE_BYTES,
        8 * 1024 * 1024,
        "active slots bound aggregate retained sandbox memory"
    );
}

#[tokio::test(start_paused = true)]
async fn live_run_admission_holds_capacity_through_final_synthesis() {
    const MAX_ACTIVE_RUNS: usize = 2;
    let program = json!({"version":1,"body":[
        {"kind":"invoke","name":"result","tool_id":"read","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":1}}]}},
        {"kind":"return","value":{"kind":"variable","name":"result"}}
    ]})
    .to_string();
    let provider = Arc::new(SynthesisBarrierProvider {
        program,
        planning_calls: AtomicU32::new(0),
        synthesis_entered: Arc::new(Semaphore::new(0)),
        synthesis_releases: Arc::new(Semaphore::new(0)),
    });
    let tool = Arc::new(CountingTool::new("read", true, true));
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry)
            .programmatic(ProgrammaticHostConfig {
                max_active_vms: MAX_ACTIVE_RUNS,
                ..ProgrammaticHostConfig::default()
            })
            .build(),
    );

    let first_runner = runner.clone();
    let mut first = tokio::spawn(async move {
        first_runner
            .run_with_strategy(
                request(&["read"]).with_run_id("synthesis-held-first"),
                RunStrategy::Programmatic,
            )
            .await
            .unwrap()
    });
    let second_runner = runner.clone();
    let mut second = tokio::spawn(async move {
        second_runner
            .run_with_strategy(
                request(&["read"]).with_run_id("synthesis-held-second"),
                RunStrategy::Programmatic,
            )
            .await
            .unwrap()
    });
    for _ in 0..MAX_ACTIVE_RUNS {
        provider.synthesis_entered.acquire().await.unwrap().forget();
    }
    assert_eq!(tool.calls.load(Ordering::SeqCst), MAX_ACTIVE_RUNS as u32);
    assert_eq!(provider.planning_calls.load(Ordering::SeqCst), 2);

    let third_runner = runner.clone();
    let third = tokio::spawn(async move {
        third_runner
            .run_with_strategy(
                request(&["read"]).with_run_id("synthesis-held-third"),
                RunStrategy::Programmatic,
            )
            .await
            .unwrap()
    });
    let third = tokio::time::timeout(Duration::from_millis(1), third)
        .await
        .expect("N+1 run must fail while earlier runs are in final synthesis")
        .unwrap();
    assert_eq!(third.status, RunStatus::LimitReached);
    assert!(third.final_output.is_none());
    assert!(third.tool_calls.is_empty());
    assert_eq!(provider.planning_calls.load(Ordering::SeqCst), 2);
    assert_eq!(tool.calls.load(Ordering::SeqCst), MAX_ACTIVE_RUNS as u32);

    provider.synthesis_releases.add_permits(1);
    let first_finished = tokio::select! {
        result = &mut first => {
            assert_eq!(result.unwrap().status, RunStatus::Completed);
            true
        }
        result = &mut second => {
            assert_eq!(result.unwrap().status, RunStatus::Completed);
            false
        }
    };
    let replacement_runner = runner.clone();
    let replacement = tokio::spawn(async move {
        replacement_runner
            .run_with_strategy(
                request(&["read"]).with_run_id("synthesis-held-replacement"),
                RunStrategy::Programmatic,
            )
            .await
            .unwrap()
    });
    provider.synthesis_entered.acquire().await.unwrap().forget();
    assert_eq!(provider.planning_calls.load(Ordering::SeqCst), 3);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 3);

    provider.synthesis_releases.add_permits(2);
    if first_finished {
        assert_eq!(second.await.unwrap().status, RunStatus::Completed);
    } else {
        assert_eq!(first.await.unwrap().status, RunStatus::Completed);
    }
    assert_eq!(replacement.await.unwrap().status, RunStatus::Completed);
}

#[tokio::test(flavor = "current_thread")]
async fn sliced_programmatic_execution_yields_to_cancellation_before_completion() {
    let body = (0..64)
        .map(|index| {
            json!({
                "kind":"let",
                "name":format!("value_{index}"),
                "value":{"kind":"integer","value":index},
            })
        })
        .chain(std::iter::once(json!({
            "kind":"return",
            "value":{"kind":"variable","name":"value_63"},
        })))
        .collect::<Vec<_>>();
    let program = json!({"version":1,"body":body}).to_string();
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(program),
            final_response("must not synthesize"),
        ])
        .with_capabilities(capabilities()),
    );
    let runner = AgentRunner::builder(provider.clone())
        .programmatic(ProgrammaticHostConfig {
            limits: SandboxLimits {
                max_slice_fuel: 1,
                ..SandboxLimits::default()
            },
            ..ProgrammaticHostConfig::default()
        })
        .build();
    let run_request = request(&[]).with_run_id("slice-fairness-cancellation");
    let cancellation = run_request.cancellation.clone();
    let canceller = tokio::spawn(async move {
        tokio::task::yield_now().await;
        cancellation.cancel();
    });

    let result = runner
        .run_with_strategy(run_request, RunStrategy::Programmatic)
        .await
        .unwrap();
    canceller.await.unwrap();

    assert_eq!(result.status, RunStatus::Cancelled);
    assert!(result.final_output.is_none());
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn reentrant_programmatic_admission_fails_fast_without_nested_effects_and_reuses_capacity() {
    let outer_program = json!({"version":1,"body":[
        {"kind":"invoke","name":"nested","tool_id":"reenter","arguments":{"kind":"object","entries":[]}},
        {"kind":"return","value":{"kind":"variable","name":"nested"}}
    ]})
    .to_string();
    let later_program = json!({"version":1,"body":[{
        "kind":"return","value":{"kind":"string","value":"reused"}
    }]})
    .to_string();
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(outer_program),
            final_response("outer synthesized"),
            final_response(later_program),
            final_response("reused synthesized"),
        ])
        .with_capabilities(capabilities()),
    );
    let events = Arc::new(InMemoryEventSink::default());
    let tool = Arc::new(ReentrantProgrammaticTool::new());
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry)
            .event_sink(events.clone())
            .programmatic(ProgrammaticHostConfig {
                max_active_vms: 1,
                ..ProgrammaticHostConfig::default()
            })
            .build(),
    );
    tool.attach_runner(runner.clone());

    let nested_finished = tool.nested_finished.clone();
    let nested_wait = nested_finished.notified();
    let outer_runner = runner.clone();
    let outer = tokio::spawn(async move {
        outer_runner
            .run_with_strategy(
                request(&["reenter"]).with_run_id("reentrant-outer-programmatic"),
                RunStrategy::Programmatic,
            )
            .await
            .unwrap()
    });

    tokio::time::timeout(Duration::from_millis(1), nested_wait)
        .await
        .expect("nested run must fail without waiting for the outer tool-held VM slot");
    let outer = outer.await.unwrap();
    assert_eq!(outer.status, RunStatus::Completed, "{:?}", outer.errors);
    assert_eq!(outer.final_output.as_deref(), Some("outer synthesized"));
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);

    let nested = tool
        .nested_result
        .lock()
        .unwrap()
        .take()
        .expect("outer tool completed its nested run");
    assert_eq!(nested.status, RunStatus::LimitReached);
    assert!(nested.final_output.is_none());
    assert!(nested.tool_calls.is_empty());
    assert_eq!(nested.errors.len(), 1);
    assert_eq!(nested.errors[0].code, "resource_limit");
    assert_eq!(
        provider.requests().len(),
        2,
        "the nested capacity rejection must not call the model"
    );

    let nested_events = events
        .events()
        .into_iter()
        .filter(|record| record.run_id == "reentrant-nested-programmatic")
        .collect::<Vec<_>>();
    let lifecycle = nested_events
        .iter()
        .filter_map(|record| match record.event {
            RunEvent::ProgramLifecycle { attempt, outcome } => Some((attempt, outcome)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle,
        vec![
            (1, ProgramLifecycleOutcome::Started),
            (1, ProgramLifecycleOutcome::LimitReached),
        ]
    );
    assert!(nested_events.iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyUsage {
            strategy: RunStrategy::Programmatic,
            model_calls: 0,
            planning_model_calls: 0,
            repair_model_calls: 0,
            final_synthesis_model_calls: 0,
            tool_calls: 0,
            tool_issued: 0,
            ..
        }
    )));
    assert!(!nested_events
        .iter()
        .any(|record| matches!(record.event, RunEvent::ProgramExecutionCompleted { .. })));
    assert_eq!(
        nested_events
            .iter()
            .filter(|record| {
                matches!(
                    record.event,
                    RunEvent::Completed {
                        status: RunStatus::LimitReached
                    }
                )
            })
            .count(),
        1
    );

    let reused = runner
        .run_with_strategy(
            request(&[]).with_run_id("reentrant-capacity-reused"),
            RunStrategy::Programmatic,
        )
        .await
        .unwrap();
    assert_eq!(reused.status, RunStatus::Completed, "{:?}", reused.errors);
    assert_eq!(reused.final_output.as_deref(), Some("reused synthesized"));
    assert_eq!(provider.requests().len(), 4);
}

#[tokio::test]
async fn cancellation_releases_live_vm_admission() {
    let program = json!({"version":1,"body":[
        {"kind":"invoke","name":"result","tool_id":"read","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":1}}]}},
        {"kind":"return","value":{"kind":"variable","name":"result"}}
    ]})
    .to_string();
    let provider = Arc::new(AdmissionProgramProvider {
        program,
        generated: Arc::new(Semaphore::new(0)),
    });
    let tool = Arc::new(AdmissionBarrierTool::new());
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let runner = Arc::new(
        AgentRunner::builder(provider)
            .tools(registry)
            .programmatic(ProgrammaticHostConfig {
                max_active_vms: 1,
                ..ProgrammaticHostConfig::default()
            })
            .build(),
    );
    let first_request = request(&["read"]).with_run_id("cancelled-live-vm");
    let cancellation = first_request.cancellation.clone();
    let first_runner = runner.clone();
    let first = tokio::spawn(async move {
        first_runner
            .run_with_strategy(first_request, RunStrategy::Programmatic)
            .await
            .unwrap()
    });
    tool.entered.acquire().await.unwrap().forget();
    cancellation.cancel();
    assert_eq!(first.await.unwrap().status, RunStatus::Failed);

    let second_runner = runner.clone();
    let second = tokio::spawn(async move {
        second_runner
            .run_with_strategy(
                request(&["read"]).with_run_id("replacement-live-vm"),
                RunStrategy::Programmatic,
            )
            .await
            .unwrap()
    });
    tool.entered.acquire().await.unwrap().forget();
    tool.releases.add_permits(1);
    assert_eq!(second.await.unwrap().status, RunStatus::Completed);
}

#[tokio::test]
async fn invalid_program_attempts_release_live_vm_admission_before_fallback() {
    let valid = json!({"version":1,"body":[{
        "kind":"return","value":{"kind":"string","value":"verified"}
    }]})
    .to_string();
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response("not json"),
            final_response("still not json"),
            final_response("direct fallback"),
            final_response(valid),
            final_response("final synthesis"),
        ])
        .with_capabilities(capabilities()),
    );
    let runner = AgentRunner::builder(provider)
        .programmatic(ProgrammaticHostConfig {
            max_active_vms: 1,
            ..ProgrammaticHostConfig::default()
        })
        .build();
    let first = runner
        .run_with_strategy(
            request(&[]).with_run_id("invalid-live-vm"),
            RunStrategy::Programmatic,
        )
        .await
        .unwrap();
    assert_eq!(first.status, RunStatus::Completed);
    let second = tokio::time::timeout(
        Duration::from_secs(1),
        runner.run_with_strategy(
            request(&[]).with_run_id("replacement-after-invalid-live-vm"),
            RunStrategy::Programmatic,
        ),
    )
    .await
    .expect("invalid parse and repair must release the live VM slot")
    .unwrap();
    assert_eq!(second.status, RunStatus::Completed);
    assert_eq!(second.final_output.as_deref(), Some("final synthesis"));
}

#[tokio::test]
async fn result_bound_mutation_requires_approval_for_the_final_immutable_arguments() {
    let program = json!({"version":1,"body":[
        {"kind":"invoke","name":"read_result","tool_id":"read","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":1}}]}},
        {"kind":"invoke","name":"write_result","tool_id":"write","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"path","value":{"kind":"variable","name":"read_result"},"pointer":"/output/value"}}]}},
        {"kind":"return","value":{"kind":"variable","name":"write_result"}}
    ]});
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(serde_json::to_string(&program).unwrap()),
            final_response("finished"),
        ])
        .with_capabilities(capabilities()),
    );
    let read = Arc::new(RecordingTool::new(
        "read",
        ToolResult::success(json!({"value": 7})),
        true,
        true,
        [ToolCaller::Programmatic],
    ));
    let write = Arc::new(RecordingTool::new(
        "write",
        ToolResult::success(json!({"value": 7})),
        false,
        false,
        [ToolCaller::Programmatic],
    ));
    let approvals = Arc::new(ApprovalOnWrite {
        seen_arguments: Mutex::new(Vec::new()),
        granted: true,
    });
    let mut registry = ToolRegistry::default();
    registry.register(read.clone()).unwrap();
    registry.register(write.clone()).unwrap();
    let runner = AgentRunner::builder(provider)
        .tools(registry)
        .policy(approvals.clone())
        .approvals(approvals.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build();

    let result = runner
        .run_with_strategy(request(&["read", "write"]), RunStrategy::Programmatic)
        .await
        .unwrap();
    assert!(matches!(result.status, RunStatus::Completed));
    assert_eq!(write.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        approvals.seen_arguments.lock().unwrap().as_slice(),
        &[json!({"value": 7})]
    );
    let contexts = write.contexts.lock().unwrap();
    assert_eq!(contexts.len(), 1);
    assert_eq!(contexts[0].caller, Some(ToolCaller::Programmatic));
    assert!(contexts[0].program_attempt.is_some());
    assert!(contexts[0].static_call_site.is_some());
    assert!(contexts[0].dynamic_ordinal.is_some());
    assert_eq!(
        contexts[0].effect_key.as_deref(),
        Some(contexts[0].call_id.as_str())
    );
}

#[tokio::test]
async fn programmatic_caller_cannot_bypass_a_direct_only_tool_or_trigger_approval() {
    let program = json!({"version":1,"body":[
        {"kind":"invoke","name":"result","tool_id":"direct_only","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":7}}]}},
        {"kind":"return","value":{"kind":"variable","name":"result"}}
    ]});
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(serde_json::to_string(&program).unwrap()),
            final_response("finished"),
        ])
        .with_capabilities(capabilities()),
    );
    let tool = Arc::new(RecordingTool::new(
        "direct_only",
        ToolResult::success(json!({"value": 7})),
        false,
        false,
        [ToolCaller::Direct],
    ));
    let approvals = Arc::new(ApprovalOnWrite {
        seen_arguments: Mutex::new(Vec::new()),
        granted: true,
    });
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let runner = AgentRunner::builder(provider)
        .tools(registry)
        .policy(approvals.clone())
        .approvals(approvals.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build();

    let result = runner
        .run_with_strategy(request(&["direct_only"]), RunStrategy::Programmatic)
        .await
        .unwrap();
    assert!(matches!(result.status, RunStatus::Completed));
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    assert!(approvals.seen_arguments.lock().unwrap().is_empty());
    assert!(result
        .errors
        .iter()
        .any(|error| error.code == "tool_rejected"));
}

#[tokio::test]
async fn forced_programmatic_fails_closed_without_every_opt_in() {
    let provider = Arc::new(MockModelProvider::scripted([]).with_capabilities(capabilities()));
    let no_host = AgentRunner::builder(provider.clone()).build();
    assert!(matches!(
        no_host
            .run_with_strategy(request(&[]), RunStrategy::Programmatic)
            .await,
        Err(HarnessError::UnsupportedCapability(_))
    ));

    let missing_conformance = Arc::new(
        MockModelProvider::scripted([]).with_capabilities(
            ModelCapabilities::new(true, false, true)
                .with_programmatic_calling(true)
                .with_limits(ProviderCapabilityLimits::new().with_max_program_bytes(1024)),
        ),
    );
    let runner = AgentRunner::builder(missing_conformance)
        .programmatic(ProgrammaticHostConfig::default())
        .build();
    assert!(matches!(
        runner
            .run_with_strategy(request(&[]), RunStrategy::Programmatic)
            .await,
        Err(HarnessError::UnsupportedCapability(_))
    ));
}

#[tokio::test]
async fn forced_programmatic_fails_closed_for_each_missing_or_zero_provider_declaration() {
    let strict_without_tools = ModelCapabilities::new(false, false, true)
        .with_programmatic_conformance(ProgrammaticConformance::StrictJsonAstV1)
        .with_limits(ProviderCapabilityLimits::new().with_max_program_bytes(1024));
    let missing_programmatic = ModelCapabilities::new(true, false, true)
        .with_limits(ProviderCapabilityLimits::new().with_max_program_bytes(1024));
    let missing_conformance = ModelCapabilities::new(true, false, true)
        .with_programmatic_calling(true)
        .with_limits(ProviderCapabilityLimits::new().with_max_program_bytes(1024));
    let missing_program_limit = ModelCapabilities::new(true, false, true)
        .with_programmatic_conformance(ProgrammaticConformance::StrictJsonAstV1);
    let zero_program_limit = ModelCapabilities::new(true, false, true)
        .with_programmatic_conformance(ProgrammaticConformance::StrictJsonAstV1)
        .with_limits(ProviderCapabilityLimits::new().with_max_program_bytes(0));

    for advertised in [
        strict_without_tools,
        missing_programmatic,
        missing_conformance,
        missing_program_limit,
        zero_program_limit,
    ] {
        let provider = Arc::new(MockModelProvider::scripted([]).with_capabilities(advertised));
        let runner = AgentRunner::builder(provider.clone())
            .programmatic(ProgrammaticHostConfig::default())
            .build();
        assert!(matches!(
            runner
                .run_with_strategy(request(&[]), RunStrategy::Programmatic)
                .await,
            Err(HarnessError::UnsupportedCapability(_))
        ));
        assert!(provider.requests().is_empty());
    }

    let provider = Arc::new(MockModelProvider::scripted([]).with_capabilities(capabilities()));
    let runner = AgentRunner::builder(provider.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build();
    let mut insufficient_model_budget = request(&[]);
    insufficient_model_budget.agent.limits.max_model_calls = 1;
    assert!(matches!(
        runner
            .run_with_strategy(insufficient_model_budget, RunStrategy::Programmatic)
            .await,
        Err(HarnessError::UnsupportedCapability(_))
    ));
    assert!(provider.requests().is_empty());
}

#[tokio::test]
async fn adaptive_never_selects_programmatic_even_when_available() {
    let provider = Arc::new(
        MockModelProvider::scripted([final_response("direct")]).with_capabilities(capabilities()),
    );
    let runner = AgentRunner::builder(provider.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build();
    let result = runner.run(request(&[])).await.unwrap();
    assert_eq!(result.final_output.as_deref(), Some("direct"));
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn fanout_rejects_mutation_before_any_dispatch() {
    let program = json!({"version":1,"body":[
        {"kind":"fan_out","name":"results","tool_id":"write","item":"i","collection":{"kind":"array","items":[{"kind":"integer","value":1},{"kind":"integer","value":2}]},"max_calls":2,
         "arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"variable","name":"i"}}]}},
        {"kind":"return","value":{"kind":"variable","name":"results"}}
    ]});
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(serde_json::to_string(&program).unwrap())])
            .with_capabilities(capabilities()),
    );
    let tool = Arc::new(CountingTool::new("write", false, false));
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let approvals = Arc::new(ApprovalOnWrite {
        seen_arguments: Mutex::new(Vec::new()),
        granted: true,
    });
    let runner = AgentRunner::builder(provider)
        .tools(registry)
        .policy(approvals.clone())
        .approvals(approvals.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build();
    let result = runner
        .run_with_strategy(request(&["write"]), RunStrategy::Programmatic)
        .await
        .unwrap();
    assert!(matches!(result.status, RunStatus::Failed));
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    assert!(approvals.seen_arguments.lock().unwrap().is_empty());
}

#[tokio::test]
async fn fanout_rejects_unsafe_reads_and_provider_cap_excess_before_any_dispatch() {
    let program = json!({"version":1,"body":[
        {"kind":"fan_out","name":"results","tool_id":"read","item":"i","collection":{"kind":"array","items":[{"kind":"integer","value":1},{"kind":"integer","value":2}]},"max_calls":2,
         "arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"variable","name":"i"}}]}},
        {"kind":"return","value":{"kind":"variable","name":"results"}}
    ]});
    for (tool, provider_capabilities) in [
        (
            Arc::new(CountingTool::new("read", true, false)),
            capabilities().with_parallel_tool_calls(true).with_limits(
                ProviderCapabilityLimits::new()
                    .with_max_program_bytes(64 * 1024)
                    .with_max_parallel_tool_calls(2),
            ),
        ),
        (
            Arc::new(CountingTool::new("read", true, true)),
            capabilities().with_parallel_tool_calls(true).with_limits(
                ProviderCapabilityLimits::new()
                    .with_max_program_bytes(64 * 1024)
                    .with_max_parallel_tool_calls(1),
            ),
        ),
    ] {
        let provider = Arc::new(
            MockModelProvider::scripted([final_response(serde_json::to_string(&program).unwrap())])
                .with_capabilities(provider_capabilities),
        );
        let mut registry = ToolRegistry::default();
        registry.register(tool.clone()).unwrap();
        let runner = AgentRunner::builder(provider)
            .tools(registry)
            .policy(Arc::new(AllowAllPolicy))
            .programmatic(ProgrammaticHostConfig::default())
            .build();
        let result = runner
            .run_with_strategy(request(&["read"]), RunStrategy::Programmatic)
            .await
            .unwrap();
        assert!(matches!(
            result.status,
            RunStatus::Failed | RunStatus::LimitReached
        ));
        assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn agent_programmatic_byte_limit_is_inclusive_and_wins_the_cap_intersection() {
    let base = return_program_with_exact_serialized_bytes(160);
    let source_n_minus_one = return_program_with_exact_serialized_bytes(base.len() - 1);
    let source_n_plus_one = return_program_with_exact_serialized_bytes(base.len() + 1);

    for (source, agent_cap, expected_model_calls) in [
        (&source_n_minus_one, source_n_minus_one.len() as u64, 2),
        (&base, base.len() as u64, 2),
        (&source_n_plus_one, base.len() as u64, 3),
    ] {
        let provider = Arc::new(
            MockModelProvider::scripted([
                final_response(source.clone()),
                final_response(return_program_with_exact_serialized_bytes(base.len() - 1)),
                final_response("done"),
            ])
            .with_capabilities(capabilities().with_limits(
                ProviderCapabilityLimits::new().with_max_program_bytes((base.len() + 1) as u64),
            )),
        );
        let mut config = ProgrammaticHostConfig::default();
        config.limits.max_program_bytes = base.len() + 1;
        let mut run_request = request(&[]);
        run_request.agent.limits.max_programmatic_program_bytes = agent_cap;
        let result = AgentRunner::builder(provider.clone())
            .programmatic(config)
            .build()
            .run_with_strategy(run_request, RunStrategy::Programmatic)
            .await
            .unwrap();

        assert_eq!(result.status, RunStatus::Completed);
        assert_eq!(provider.requests().len(), expected_model_calls);
    }
}

#[tokio::test]
async fn programmatic_fanout_uses_the_minimum_of_host_agent_and_provider_caps() {
    let program = two_read_fanout_program();
    for (host_cap, agent_cap, provider_cap, completes) in [
        (2usize, 2u32, 2u32, true),
        (1, 2, 2, false),
        (2, 1, 2, false),
        (2, 2, 1, false),
    ] {
        let provider = Arc::new(
            MockModelProvider::scripted([final_response(program.clone()), final_response("done")])
                .with_capabilities(
                    capabilities().with_parallel_tool_calls(true).with_limits(
                        ProviderCapabilityLimits::new()
                            .with_max_program_bytes(64 * 1024)
                            .with_max_parallel_tool_calls(provider_cap),
                    ),
                ),
        );
        let tool = Arc::new(CountingTool::new("read", true, true));
        let mut registry = ToolRegistry::default();
        registry.register(tool.clone()).unwrap();
        let config = ProgrammaticHostConfig {
            max_fanout_concurrency: host_cap,
            ..ProgrammaticHostConfig::default()
        };
        let mut run_request = request(&["read"]);
        run_request.agent.limits.max_programmatic_fanout_concurrency = agent_cap;
        let result = AgentRunner::builder(provider.clone())
            .tools(registry)
            .policy(Arc::new(AllowAllPolicy))
            .programmatic(config)
            .build()
            .run_with_strategy(run_request, RunStrategy::Programmatic)
            .await
            .unwrap();

        if completes {
            assert_eq!(result.status, RunStatus::Completed);
            assert_eq!(tool.calls.load(Ordering::SeqCst), 2);
            assert_eq!(provider.requests().len(), 2);
        } else {
            assert!(matches!(
                result.status,
                RunStatus::Failed | RunStatus::LimitReached
            ));
            assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
            assert_eq!(provider.requests().len(), 1);
        }
    }
}

#[tokio::test]
async fn empty_programmatic_fanout_completes_without_broker_or_tool_calls() {
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(read_fanout_program(&[])),
            final_response("done"),
        ])
        .with_capabilities(capabilities().with_parallel_tool_calls(true)),
    );
    let tool = Arc::new(CountingTool::new("read", true, true));
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let events = Arc::new(InMemoryEventSink::default());

    let result = AgentRunner::builder(provider.clone())
        .tools(registry)
        .policy(Arc::new(AllowAllPolicy))
        .event_sink(events.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build()
        .run_with_strategy(request(&["read"]), RunStrategy::Programmatic)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.final_output.as_deref(), Some("done"));
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.requests().len(), 2);
    let records = events.events();
    assert!(records.iter().all(|record| {
        !matches!(
            record.event,
            RunEvent::ToolEffectReused { .. }
                | RunEvent::ToolRejected { .. }
                | RunEvent::PolicyDecided { .. }
                | RunEvent::ApprovalRequested { .. }
                | RunEvent::ToolCompleted { .. }
        )
    }));
    assert!(records.iter().any(|record| matches!(
        record.event,
        RunEvent::ProgramExecutionCompleted {
            tool_yields: 0,
            fanout_batches: 0,
            ..
        }
    )));
    assert!(records
        .iter()
        .any(|record| matches!(record.event, RunEvent::StrategyUsage { tool_calls: 0, .. })));
}

#[tokio::test]
async fn transcript_envelope_rejects_a_fanout_before_policy_or_dispatch() {
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(two_read_fanout_program())]).with_capabilities(
            capabilities().with_parallel_tool_calls(true).with_limits(
                ProviderCapabilityLimits::new()
                    .with_max_program_bytes(64 * 1024)
                    .with_max_parallel_tool_calls(2),
            ),
        ),
    );
    let tool = Arc::new(CountingTool::new("read", true, true));
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let mut run_request = request(&["read"]);
    // One potential 1 MiB program return fits, but reserving two possible
    // 1 MiB canonical tool responses does not. This must stop before policy,
    // approval, or a tool dispatch can begin.
    run_request.agent.limits.max_transcript_bytes = 2 * 1024 * 1024;
    let result = AgentRunner::builder(provider.clone())
        .tools(registry)
        .policy(Arc::new(AllowAllPolicy))
        .programmatic(ProgrammaticHostConfig::default())
        .build()
        .run_with_strategy(run_request, RunStrategy::Programmatic)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::LimitReached);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn programmatic_parallel_fanout_preserves_input_order_through_reverse_completion() {
    let program = read_fanout_program(&[1, 2, 3]);
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(program), final_response("done")])
            .with_capabilities(
                capabilities().with_parallel_tool_calls(true).with_limits(
                    ProviderCapabilityLimits::new()
                        .with_max_program_bytes(64 * 1024)
                        .with_max_parallel_tool_calls(3),
                ),
            ),
    );
    let tool = Arc::new(ControlledFanoutTool::new(3));
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let config = ProgrammaticHostConfig {
        max_fanout_concurrency: 3,
        ..ProgrammaticHostConfig::default()
    };
    let mut run_request = request(&["read"]);
    run_request.agent.limits.max_programmatic_fanout_concurrency = 3;
    // This test isolates ordering rather than the bounded final-synthesis
    // envelope. Three possible 1 MiB responses plus a 1 MiB program return
    // require more than the default 4 MiB transcript allowance.
    run_request.agent.limits.max_transcript_bytes = 16 * 1024 * 1024;
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry)
            .policy(Arc::new(AllowAllPolicy))
            .programmatic(config)
            .build(),
    );
    let entered = tool.entered.clone();
    let finished = tool.finished.clone();
    let run = tokio::spawn(async move {
        runner
            .run_with_strategy(run_request, RunStrategy::Programmatic)
            .await
    });

    for _ in 0..3 {
        entered.acquire().await.unwrap().forget();
    }
    for value in [3usize, 2, 1] {
        tool.releases[value - 1].add_permits(1);
        finished.acquire().await.unwrap().forget();
    }
    let result = run.await.unwrap().unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 3);
    assert_eq!(tool.maximum.load(Ordering::SeqCst), 3);
    assert_eq!(*tool.completed.lock().unwrap(), vec![3, 2, 1]);
    assert_eq!(result.tool_calls.len(), 3);
    let requests = provider.requests();
    let synthesis = requests[1]
        .messages
        .last()
        .expect("final synthesis receives the canonical broker transcript");
    let canonical: Value = serde_json::from_str(&synthesis.content).unwrap();
    assert_eq!(
        canonical["broker_calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|call| call["arguments"]["value"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(
        canonical["program_return"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value["output"]["value"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}

#[tokio::test]
async fn cancelled_programmatic_fanout_drains_each_active_read_without_synthesis_or_replay() {
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(two_read_fanout_program()),
            final_response("must not synthesize"),
            final_response("must not fall back"),
        ])
        .with_capabilities(
            capabilities().with_parallel_tool_calls(true).with_limits(
                ProviderCapabilityLimits::new()
                    .with_max_program_bytes(64 * 1024)
                    .with_max_parallel_tool_calls(2),
            ),
        ),
    );
    let tool = Arc::new(CancellationFanoutTool::new());
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry)
            .policy(Arc::new(AllowAllPolicy))
            .programmatic(ProgrammaticHostConfig::default())
            .build(),
    );
    let run_request = request(&["read"]);
    let cancellation = run_request.cancellation.clone();
    let entered = tool.entered.clone();
    let run = tokio::spawn(async move {
        runner
            .run_with_strategy(run_request, RunStrategy::Programmatic)
            .await
    });

    for _ in 0..2 {
        entered.acquire().await.unwrap().forget();
    }
    cancellation.cancel();
    let result = run.await.unwrap().unwrap();

    assert_eq!(result.status, RunStatus::Failed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 2);
    assert_eq!(tool.observed_cancellation.load(Ordering::SeqCst), 2);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn each_programmatic_byte_cap_winner_enforces_n_minus_one_n_and_n_plus_one() {
    enum Winner {
        Host,
        Agent,
        Provider,
        Library,
    }

    for winner in [
        Winner::Host,
        Winner::Agent,
        Winner::Provider,
        Winner::Library,
    ] {
        let cap = if matches!(winner, Winner::Library) {
            HARD_MAX_PROGRAMMATIC_PROGRAM_BYTES as usize
        } else {
            160
        };
        let sources = [
            return_program_with_exact_serialized_bytes(cap - 1),
            return_program_with_exact_serialized_bytes(cap),
            return_program_with_exact_serialized_bytes(cap + 1),
        ];
        for (index, source) in sources.iter().enumerate() {
            let mut host = ProgrammaticHostConfig::default();
            host.limits.max_program_bytes = if matches!(winner, Winner::Host) {
                cap
            } else {
                HARD_MAX_PROGRAMMATIC_PROGRAM_BYTES as usize
            };
            let mut run_request = request(&[]);
            run_request.agent.limits.max_programmatic_program_bytes =
                if matches!(winner, Winner::Agent) {
                    cap as u64
                } else {
                    HARD_MAX_PROGRAMMATIC_PROGRAM_BYTES
                };
            if matches!(winner, Winner::Library) {
                // At the immutable 256 KiB program ceiling, the deliberately
                // rejected N+1 response must still fit in the repair
                // transcript and literal-retention accounting. Raise only
                // those unrelated limits so this test exercises the effective
                // program-byte boundary.
                let repair_transcript_cap = (cap as u64).saturating_mul(4);
                run_request.agent.limits.max_input_bytes = repair_transcript_cap;
                run_request.agent.limits.max_request_payload_bytes = repair_transcript_cap;
                host.limits.max_constant_bytes = HARD_LIMITS.max_constant_bytes;
                host.limits.max_fuel = HARD_LIMITS.max_fuel;
                host.limits.max_slice_fuel = HARD_LIMITS.max_slice_fuel;
            }
            let provider_cap = if matches!(winner, Winner::Provider) {
                cap as u64
            } else {
                HARD_MAX_PROGRAMMATIC_PROGRAM_BYTES
            };
            let provider = Arc::new(
                MockModelProvider::scripted([
                    final_response(source.clone()),
                    final_response(return_program_with_exact_serialized_bytes(cap - 1)),
                    final_response("done"),
                ])
                .with_capabilities(capabilities().with_limits(
                    ProviderCapabilityLimits::new().with_max_program_bytes(provider_cap),
                )),
            );
            let result = AgentRunner::builder(provider.clone())
                .programmatic(host)
                .build()
                .run_with_strategy(run_request, RunStrategy::Programmatic)
                .await
                .unwrap();
            assert_eq!(result.status, RunStatus::Completed);
            assert_eq!(provider.requests().len(), if index < 2 { 2 } else { 3 });
        }
    }
}

#[tokio::test]
async fn each_configurable_fanout_cap_winner_enforces_n_minus_one_n_and_n_plus_one() {
    enum Winner {
        Host,
        Agent,
        Provider,
    }

    for winner in [Winner::Host, Winner::Agent, Winner::Provider] {
        for call_count in 1..=3usize {
            let host_cap = if matches!(winner, Winner::Host) { 2 } else { 3 };
            let agent_cap = if matches!(winner, Winner::Agent) {
                2
            } else {
                3
            };
            let provider_cap = if matches!(winner, Winner::Provider) {
                2
            } else {
                3
            };
            let provider = Arc::new(
                MockModelProvider::scripted([
                    final_response(read_fanout_program(
                        &(1..=call_count as u64).collect::<Vec<_>>(),
                    )),
                    final_response("done"),
                ])
                .with_capabilities(
                    capabilities().with_parallel_tool_calls(true).with_limits(
                        ProviderCapabilityLimits::new()
                            .with_max_program_bytes(64 * 1024)
                            .with_max_parallel_tool_calls(provider_cap),
                    ),
                ),
            );
            let tool = Arc::new(CountingTool::new("read", true, true));
            let mut registry = ToolRegistry::default();
            registry.register(tool.clone()).unwrap();
            let config = ProgrammaticHostConfig {
                max_fanout_concurrency: host_cap,
                ..ProgrammaticHostConfig::default()
            };
            let mut run_request = request(&["read"]);
            run_request.agent.limits.max_programmatic_fanout_concurrency = agent_cap as u32;
            let result = AgentRunner::builder(provider.clone())
                .tools(registry)
                .policy(Arc::new(AllowAllPolicy))
                .programmatic(config)
                .build()
                .run_with_strategy(run_request, RunStrategy::Programmatic)
                .await
                .unwrap();
            if call_count <= 2 {
                assert_eq!(result.status, RunStatus::Completed);
                assert_eq!(tool.calls.load(Ordering::SeqCst), call_count as u32);
                assert_eq!(provider.requests().len(), 2);
            } else {
                assert!(matches!(
                    result.status,
                    RunStatus::Failed | RunStatus::LimitReached
                ));
                assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
                assert_eq!(provider.requests().len(), 1);
            }
        }
    }
    assert_eq!(HARD_MAX_PROGRAMMATIC_FANOUT_CONCURRENCY, 8);
}

#[tokio::test]
async fn immutable_fanout_cap_accepts_n_minus_one_and_n_but_rejects_n_plus_one_before_dispatch() {
    for call_count in [
        HARD_MAX_PROGRAMMATIC_FANOUT_CONCURRENCY as usize - 1,
        HARD_MAX_PROGRAMMATIC_FANOUT_CONCURRENCY as usize,
    ] {
        let provider = Arc::new(
            MockModelProvider::scripted([
                final_response(read_fanout_program(
                    &(1..=call_count as u64).collect::<Vec<_>>(),
                )),
                final_response("done"),
            ])
            .with_capabilities(
                capabilities().with_parallel_tool_calls(true).with_limits(
                    ProviderCapabilityLimits::new()
                        .with_max_program_bytes(64 * 1024)
                        .with_max_parallel_tool_calls(HARD_MAX_PROGRAMMATIC_FANOUT_CONCURRENCY),
                ),
            ),
        );
        let tool = Arc::new(CountingTool::new("read", true, true));
        let mut registry = ToolRegistry::default();
        registry.register(tool.clone()).unwrap();
        let mut run_request = request(&["read"]);
        run_request.agent.limits.max_programmatic_fanout_concurrency =
            HARD_MAX_PROGRAMMATIC_FANOUT_CONCURRENCY;
        // The effective fan-out cap is eight. Reserve eight potential 1 MiB
        // tool responses plus a 1 MiB program return so this test remains a
        // fan-out-cap boundary, not a transcript-cap boundary.
        run_request.agent.limits.max_transcript_bytes = 16 * 1024 * 1024;
        let result = AgentRunner::builder(provider.clone())
            .tools(registry)
            .policy(Arc::new(AllowAllPolicy))
            .programmatic(ProgrammaticHostConfig::default())
            .build()
            .run_with_strategy(run_request, RunStrategy::Programmatic)
            .await
            .unwrap();
        assert_eq!(result.status, RunStatus::Completed);
        assert_eq!(tool.calls.load(Ordering::SeqCst), call_count as u32);
    }

    let n_plus_one = HARD_MAX_PROGRAMMATIC_FANOUT_CONCURRENCY as usize + 1;
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(read_fanout_program(
                &(1..=n_plus_one as u64).collect::<Vec<_>>(),
            )),
            final_response("still invalid"),
            final_response("direct fallback"),
        ])
        .with_capabilities(
            capabilities().with_parallel_tool_calls(true).with_limits(
                ProviderCapabilityLimits::new()
                    .with_max_program_bytes(64 * 1024)
                    .with_max_parallel_tool_calls(HARD_MAX_PROGRAMMATIC_FANOUT_CONCURRENCY),
            ),
        ),
    );
    let tool = Arc::new(CountingTool::new("read", true, true));
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let result = AgentRunner::builder(provider.clone())
        .tools(registry)
        .policy(Arc::new(AllowAllPolicy))
        .programmatic(ProgrammaticHostConfig::default())
        .build()
        .run_with_strategy(request(&["read"]), RunStrategy::Programmatic)
        .await
        .unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.final_output.as_deref(), Some("direct fallback"));
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.requests().len(), 3);
}

#[tokio::test]
async fn post_dispatch_mutation_failure_is_terminal_and_never_replayed_or_fallen_back() {
    let program = json!({"version":1,"body":[
        {"kind":"invoke","name":"result","tool_id":"write","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":7}}]}},
        {"kind":"return","value":{"kind":"variable","name":"result"}}
    ]});
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(serde_json::to_string(&program).unwrap()),
            final_response("must not synthesize"),
            final_response("must not fall back"),
        ])
        .with_capabilities(capabilities()),
    );
    let tool = Arc::new(RecordingTool::new(
        "write",
        ToolResult::failure("MUTATION_RESULT_CANARY"),
        false,
        false,
        [ToolCaller::Programmatic],
    ));
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry)
        .policy(Arc::new(AllowAllPolicy))
        .programmatic(ProgrammaticHostConfig::default())
        .build();

    let result = runner
        .run_with_strategy(request(&["write"]), RunStrategy::Programmatic)
        .await
        .unwrap();
    assert!(matches!(result.status, RunStatus::Failed));
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.requests().len(), 1);
    assert!(result
        .errors
        .iter()
        .all(|error| !error.message.contains("MUTATION_RESULT_CANARY")));
}

#[tokio::test]
async fn cancellation_during_an_active_mutation_is_terminal_uncertain_without_replay() {
    let program = json!({"version":1,"body":[
        {"kind":"invoke","name":"result","tool_id":"write","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":7}}]}},
        {"kind":"return","value":{"kind":"variable","name":"result"}}
    ]});
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(serde_json::to_string(&program).unwrap()),
            final_response("must not synthesize"),
            final_response("must not fall back"),
        ])
        .with_capabilities(capabilities()),
    );
    let tool = Arc::new(CancellationBarrierTool::mutation("write"));
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let events = Arc::new(InMemoryEventSink::default());
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry)
            .policy(Arc::new(AllowAllPolicy))
            .event_sink(events.clone())
            .programmatic(ProgrammaticHostConfig::default())
            .build(),
    );
    let run_request = request(&["write"]);
    let cancellation = run_request.cancellation.clone();
    let entered = tool.entered.clone();
    let entered_wait = entered.notified();
    let run = tokio::spawn(async move {
        runner
            .run_with_strategy(run_request, RunStrategy::Programmatic)
            .await
    });

    entered_wait.await;
    cancellation.cancel();
    let result = run.await.unwrap().unwrap();

    assert_eq!(result.status, RunStatus::Failed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    assert!(tool.observed_cancellation.load(Ordering::SeqCst));
    assert_eq!(provider.requests().len(), 1);
    assert!(events.events().iter().all(|record| {
        !matches!(
            record.event,
            RunEvent::StrategyFallback { .. } | RunEvent::ProgramExecutionCompleted { .. }
        )
    }));
}

#[tokio::test(start_paused = true)]
async fn active_programmatic_mutation_deadline_drains_cooperatively_then_stays_terminal() {
    let program = json!({"version":1,"body":[
        {"kind":"invoke","name":"result","tool_id":"write","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":7}}]}},
        {"kind":"return","value":{"kind":"variable","name":"result"}}
    ]});
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(serde_json::to_string(&program).unwrap()),
            final_response("must not synthesize"),
            final_response("must not fall back"),
        ])
        .with_capabilities(capabilities()),
    );
    let tool = Arc::new(CancellationBarrierTool::mutation("write"));
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let config = ProgrammaticHostConfig {
        max_duration_ms: 1,
        ..ProgrammaticHostConfig::default()
    };
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry)
            .policy(Arc::new(AllowAllPolicy))
            .programmatic(config)
            .build(),
    );
    let entered = tool.entered.clone();
    let entered_wait = entered.notified();
    let run = tokio::spawn(async move {
        runner
            .run_with_strategy(request(&["write"]), RunStrategy::Programmatic)
            .await
    });

    entered_wait.await;
    tokio::time::advance(std::time::Duration::from_millis(2)).await;
    let result = run.await.unwrap().unwrap();

    assert_eq!(result.status, RunStatus::Failed);
    assert!(tool.observed_cancellation.load(Ordering::SeqCst));
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn non_cooperative_programmatic_mutation_cleanup_grace_is_terminal_uncertain() {
    let program = json!({"version":1,"body":[
        {"kind":"invoke","name":"result","tool_id":"write","arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"integer","value":7}}]}},
        {"kind":"return","value":{"kind":"variable","name":"result"}}
    ]});
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(serde_json::to_string(&program).unwrap()),
            final_response("must not synthesize"),
            final_response("must not fall back"),
        ])
        .with_capabilities(capabilities()),
    );
    let tool = Arc::new(NonCooperativeMutationTool::new());
    let mut registry = ToolRegistry::default();
    registry.register(tool.clone()).unwrap();
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry)
            .policy(Arc::new(AllowAllPolicy))
            .programmatic(ProgrammaticHostConfig::default())
            .build(),
    );
    let run_request = request(&["write"]);
    let cancellation = run_request.cancellation.clone();
    let entered = tool.entered.clone();
    let entered_wait = entered.notified();
    let run = tokio::spawn(async move {
        runner
            .run_with_strategy(run_request, RunStrategy::Programmatic)
            .await
    });

    entered_wait.await;
    cancellation.cancel();
    let result = run.await.unwrap().unwrap();

    assert_eq!(result.status, RunStatus::Failed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn invalid_program_gets_exactly_one_repair_before_dispatch() {
    let repaired =
        json!({"version":1,"body":[{"kind":"return","value":{"kind":"string","value":"safe"}}]});
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response("not json"),
            final_response(serde_json::to_string(&repaired).unwrap()),
            final_response("done"),
        ])
        .with_capabilities(capabilities()),
    );
    let runner = AgentRunner::builder(provider.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build();
    let result = runner
        .run_with_strategy(request(&[]), RunStrategy::Programmatic)
        .await
        .unwrap();
    assert!(matches!(result.status, RunStatus::Completed));
    assert_eq!(provider.requests().len(), 3);
}

#[tokio::test]
async fn two_call_programmatic_budget_skips_repair_and_preserves_direct_fallback_capacity() {
    let provider = Arc::new(
        MockModelProvider::scripted([final_response("not json"), final_response("fallback")])
            .with_capabilities(capabilities()),
    );
    let events = Arc::new(InMemoryEventSink::default());
    let mut run_request = request(&[]);
    run_request.agent.limits.max_model_calls = 2;
    let result = AgentRunner::builder(provider.clone())
        .event_sink(events.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build()
        .run_with_strategy(run_request, RunStrategy::Programmatic)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.final_output.as_deref(), Some("fallback"));
    assert_eq!(provider.requests().len(), 2);
    let usage = events
        .events()
        .iter()
        .find_map(|record| match &record.event {
            RunEvent::StrategyUsage {
                strategy,
                model_calls,
                planning_model_calls,
                repair_model_calls,
                recovery_model_calls,
                final_synthesis_model_calls,
                reactive_model_calls,
                ..
            } => Some((
                *strategy,
                *model_calls,
                *planning_model_calls,
                *repair_model_calls,
                *recovery_model_calls,
                *final_synthesis_model_calls,
                *reactive_model_calls,
            )),
            _ => None,
        });
    assert_eq!(usage, Some((RunStrategy::Direct, 2, 1, 0, 0, 0, 1)));
    let (_, model_calls, planning, repair, recovery, synthesis, reactive) = usage.unwrap();
    assert_eq!(
        model_calls,
        planning + repair + recovery + synthesis + reactive
    );
    let records = events.events();
    assert_eq!(
        records
            .iter()
            .filter(|record| matches!(record.event, RunEvent::Started { .. }))
            .count(),
        1
    );
    assert_eq!(
        records
            .iter()
            .filter(|record| matches!(record.event, RunEvent::Completed { .. }))
            .count(),
        1
    );
    assert!(records.iter().any(|record| matches!(
        record.event,
        RunEvent::StrategySelected {
            requested: RunStrategy::Programmatic,
            selected: RunStrategy::Direct,
            ..
        }
    )));
}

#[tokio::test]
async fn three_call_programmatic_budget_allows_repair_and_reserves_final_synthesis() {
    let repaired =
        json!({"version":1,"body":[{"kind":"return","value":{"kind":"string","value":"safe"}}]});
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response("not json"),
            final_response(serde_json::to_string(&repaired).unwrap()),
            final_response("synthesized"),
        ])
        .with_capabilities(capabilities()),
    );
    let events = Arc::new(InMemoryEventSink::default());
    let mut run_request = request(&[]);
    run_request.agent.limits.max_model_calls = 3;
    let result = AgentRunner::builder(provider.clone())
        .event_sink(events.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build()
        .run_with_strategy(run_request, RunStrategy::Programmatic)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.final_output.as_deref(), Some("synthesized"));
    assert_eq!(provider.requests().len(), 3);
    let usage = events
        .events()
        .iter()
        .find_map(|record| match &record.event {
            RunEvent::StrategyUsage {
                strategy,
                model_calls,
                planning_model_calls,
                repair_model_calls,
                recovery_model_calls,
                final_synthesis_model_calls,
                reactive_model_calls,
                ..
            } => Some((
                *strategy,
                *model_calls,
                *planning_model_calls,
                *repair_model_calls,
                *recovery_model_calls,
                *final_synthesis_model_calls,
                *reactive_model_calls,
            )),
            _ => None,
        });
    assert_eq!(usage, Some((RunStrategy::Programmatic, 3, 1, 1, 0, 1, 0)));
    let (_, model_calls, planning, repair, recovery, synthesis, reactive) = usage.unwrap();
    assert_eq!(
        model_calls,
        planning + repair + recovery + synthesis + reactive
    );
}

#[tokio::test]
async fn program_lifecycle_succeeds_only_after_final_synthesis() {
    let program =
        json!({"version":1,"body":[{"kind":"return","value":{"kind":"string","value":"safe"}}]});
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(serde_json::to_string(&program).unwrap()),
            final_response("synthesized"),
        ])
        .with_capabilities(capabilities()),
    );
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider)
        .event_sink(events.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build()
        .run_with_strategy(request(&[]), RunStrategy::Programmatic)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    let records = events.events();
    let lifecycle = records
        .iter()
        .filter_map(|record| match record.event {
            RunEvent::ProgramLifecycle { attempt, outcome } => Some((attempt, outcome)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle,
        vec![
            (1, ProgramLifecycleOutcome::Started),
            (1, ProgramLifecycleOutcome::Validated),
            (1, ProgramLifecycleOutcome::Succeeded),
        ]
    );
    let vm_completed = records
        .iter()
        .position(|record| matches!(record.event, RunEvent::ProgramExecutionCompleted { .. }))
        .unwrap();
    let vm_metrics = records.iter().find_map(|record| match record.event {
        RunEvent::ProgramExecutionCompleted {
            scheduling_slices,
            tool_yields,
            ..
        } => Some((scheduling_slices, tool_yields)),
        _ => None,
    });
    assert!(matches!(vm_metrics, Some((slices, 0)) if slices > 0));
    let succeeded = records
        .iter()
        .position(|record| {
            matches!(
                record.event,
                RunEvent::ProgramLifecycle {
                    outcome: ProgramLifecycleOutcome::Succeeded,
                    ..
                }
            )
        })
        .unwrap();
    assert!(vm_completed < succeeded);
}

#[tokio::test]
async fn synthesis_failure_emits_only_the_program_failure_terminal_lifecycle() {
    let program =
        json!({"version":1,"body":[{"kind":"return","value":{"kind":"string","value":"safe"}}]});
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(serde_json::to_string(&program).unwrap()),
            final_response("   "),
        ])
        .with_capabilities(capabilities()),
    );
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider)
        .event_sink(events.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build()
        .run_with_strategy(request(&[]), RunStrategy::Programmatic)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Failed);
    let records = events.events();
    let lifecycle = records
        .iter()
        .filter_map(|record| match record.event {
            RunEvent::ProgramLifecycle { attempt, outcome } => Some((attempt, outcome)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle,
        vec![
            (1, ProgramLifecycleOutcome::Started),
            (1, ProgramLifecycleOutcome::Validated),
            (1, ProgramLifecycleOutcome::Failed),
        ]
    );
    assert!(records
        .iter()
        .any(|record| matches!(record.event, RunEvent::ProgramExecutionCompleted { .. })));
}

#[tokio::test]
async fn programmatic_parser_canary_never_enters_public_result_or_events() {
    const PARSER_CANARY: &str = "program-parser-canary";
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(PARSER_CANARY),
            final_response(PARSER_CANARY),
            final_response("direct fallback"),
        ])
        .with_capabilities(capabilities()),
    );
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider)
        .event_sink(events.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build()
        .run_with_strategy(request(&[]), RunStrategy::Programmatic)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert!(!serde_json::to_string(&result)
        .unwrap()
        .contains(PARSER_CANARY));
    assert!(!format!("{result:?}").contains(PARSER_CANARY));
    assert!(!serde_json::to_string(&events.events())
        .unwrap()
        .contains(PARSER_CANARY));
}

#[tokio::test]
async fn invalid_repaired_program_falls_back_once_to_fresh_direct_scope_before_effects() {
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response("not json"),
            final_response("still not json"),
            final_response("direct fallback"),
        ])
        .with_capabilities(capabilities()),
    );
    let events = Arc::new(InMemoryEventSink::default());
    let runner = AgentRunner::builder(provider.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .event_sink(events.clone())
        .build();

    let result = runner
        .run_with_strategy(
            request(&[])
                .with_run_id("explicit-fallback-run")
                .with_trace_id("explicit-fallback-trace"),
            RunStrategy::Programmatic,
        )
        .await
        .unwrap();
    assert!(matches!(result.status, RunStatus::Completed));
    assert_eq!(result.id, "explicit-fallback-run");
    assert_eq!(result.trace_id, "explicit-fallback-trace");
    assert_eq!(result.final_output.as_deref(), Some("direct fallback"));
    assert_eq!(provider.requests().len(), 3);
    assert!(events.events().iter().any(|record| {
        matches!(
            record.event,
            RunEvent::StrategyFallback {
                reason: StrategyFallbackReason::InvalidProgram,
                ..
            }
        )
    }));
    let records = events.events();
    assert_eq!(
        records
            .iter()
            .filter(|record| matches!(record.event, RunEvent::Started { .. }))
            .count(),
        1
    );
    assert_eq!(
        records
            .iter()
            .filter(|record| matches!(record.event, RunEvent::Completed { .. }))
            .count(),
        1
    );
    assert!(records
        .iter()
        .enumerate()
        .all(|(index, record)| record.sequence == (index + 1) as u64));
}

#[tokio::test(start_paused = true)]
async fn direct_fallback_keeps_the_tighter_programmatic_host_deadline() {
    let provider = Arc::new(HostDeadlineFallbackProvider {
        calls: AtomicU32::new(0),
        fallback_started: Arc::new(Notify::new()),
    });
    let config = ProgrammaticHostConfig {
        max_duration_ms: 1,
        ..ProgrammaticHostConfig::default()
    };
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .programmatic(config)
            .build(),
    );
    let mut run_request = request(&[]);
    // The application deadline is intentionally looser than the host's
    // programmatic deadline. The fallback must retain the latter.
    run_request.agent.limits.max_run_duration_ms = Some(100);
    let fallback_started = provider.fallback_started.clone();
    let run = tokio::spawn(async move {
        runner
            .run_with_strategy(run_request, RunStrategy::Programmatic)
            .await
    });

    fallback_started.notified().await;
    tokio::time::advance(Duration::from_millis(2)).await;
    let result = run.await.unwrap().unwrap();

    assert_eq!(result.status, RunStatus::Failed);
    assert!(result.errors.iter().any(|error| error.code == "timed_out"));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
}

#[test]
fn invalid_vm_admission_configuration_never_panics_while_building() {
    for max_active_vms in [0, 17, usize::MAX] {
        let config = ProgrammaticHostConfig {
            max_active_vms,
            ..ProgrammaticHostConfig::default()
        };
        let provider = Arc::new(MockModelProvider::scripted([]).with_capabilities(capabilities()));
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            AgentRunner::builder(provider).programmatic(config).build()
        }));
        assert!(built.is_ok(), "builder panicked for {max_active_vms}");
    }
}

#[tokio::test]
async fn invalid_vm_admission_configuration_fails_closed_when_run() {
    for max_active_vms in [0, 17, usize::MAX] {
        let config = ProgrammaticHostConfig {
            max_active_vms,
            ..ProgrammaticHostConfig::default()
        };
        let provider = Arc::new(MockModelProvider::scripted([]).with_capabilities(capabilities()));
        let result = AgentRunner::builder(provider)
            .programmatic(config)
            .build()
            .run_with_strategy(request(&[]), RunStrategy::Programmatic)
            .await;
        assert!(matches!(result, Err(HarnessError::InvalidRequest(_))));
    }
}
