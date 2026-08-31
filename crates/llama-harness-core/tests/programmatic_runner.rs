#![cfg(feature = "programmatic")]

use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, MockModelProvider},
    AgentDefinition, AgentRunner, AllowAllPolicy, ApprovalHandler, ApprovalRecord, HarnessError,
    InMemoryEventSink, ModelCapabilities, PolicyDecision, PolicyEngine, ProgrammaticConformance,
    ProgrammaticHostConfig, ProviderCapabilityLimits, RunEvent, RunRequest, RunStatus, RunStrategy,
    StrategyFallbackReason, Tool, ToolCallContext, ToolCaller, ToolDefinition, ToolRegistry,
    ToolResult, ToolRisk,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex,
};
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

struct CountingTool {
    definition: ToolDefinition,
    calls: AtomicU32,
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
    assert_eq!(result.tool_calls[0].arguments_json, "{}");
    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].tools.len(), 1);
    assert!(requests[1].tools.is_empty());
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
    let runner = AgentRunner::builder(provider)
        .tools(registry)
        .policy(Arc::new(AllowAllPolicy))
        .programmatic(ProgrammaticHostConfig::default())
        .build();
    let result = runner
        .run_with_strategy(request(&["write"]), RunStrategy::Programmatic)
        .await
        .unwrap();
    assert!(matches!(result.status, RunStatus::Failed));
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
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
        .run_with_strategy(request(&[]), RunStrategy::Programmatic)
        .await
        .unwrap();
    assert!(matches!(result.status, RunStatus::Completed));
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
}
