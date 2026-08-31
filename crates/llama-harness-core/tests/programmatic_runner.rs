#![cfg(feature = "programmatic")]

use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, MockModelProvider},
    AgentDefinition, AgentRunner, AllowAllPolicy, HarnessError, InMemoryEventSink,
    ModelCapabilities, ProgrammaticConformance, ProgrammaticHostConfig, ProviderCapabilityLimits,
    RunEvent, RunRequest, RunStatus, RunStrategy, StrategyFallbackReason, Tool, ToolCaller,
    ToolDefinition, ToolRegistry, ToolResult, ToolRisk,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
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
