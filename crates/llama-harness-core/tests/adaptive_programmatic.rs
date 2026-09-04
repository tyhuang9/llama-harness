#![cfg(feature = "programmatic")]

use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, MockModelProvider},
    AgentDefinition, AgentRunner, AllowAllPolicy, HarnessError, InMemoryEventSink,
    ModelCapabilities, ProgrammaticConformance, ProgrammaticHostConfig, ProgrammaticWorkloadClass,
    ProviderCapabilityLimits, RunEvent, RunRequest, RunStatus, RunStrategy, StrategyFallbackReason,
    StrategySelectionReason, Tool, ToolCaller, ToolDefinition, ToolDiscoveryLimits, ToolRegistry,
    ToolResult, ToolRisk,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

const LEGACY_PLANNER_PROMPT: &str = "Select the safest efficient tool strategy. Return only one strict JSON object: {\"strategy\":\"direct\"} when no finite safe plan is justified, or {\"strategy\":\"declarative_plan\",\"plan\":{\"nodes\":[...]}} for a finite dependency DAG. Use only the supplied tools. Every plan node requires id, tool_id, and schema-valid arguments. Optional fields are depends_on, result_bindings, concurrency, approval_barrier, and commit_boundary. Choose direct for mutations, approval-sensitive work, ambiguity, or an uncertain next step.";

fn capabilities() -> ModelCapabilities {
    ModelCapabilities::new(true, false, true)
        .with_structured_plans(true)
        .with_programmatic_conformance(ProgrammaticConformance::StrictJsonAstV1)
        .with_limits(
            ProviderCapabilityLimits::new()
                .with_max_plan_nodes(64)
                .with_max_plan_bytes(256 * 1024)
                .with_max_program_bytes(64 * 1024),
        )
}

fn request(tool_ids: &[&str], max_model_calls: u32) -> RunRequest {
    let mut agent = AgentDefinition::new("adaptive-programmatic", "Adaptive", "1", "mock-model");
    agent.tool_allowlist = tool_ids.iter().map(|id| (*id).to_owned()).collect();
    agent.limits.max_model_calls = max_model_calls;
    RunRequest::new(agent, "complete the task")
        .with_run_id("adaptive-programmatic-run")
        .with_trace_id("adaptive-programmatic-trace")
}

fn proposal(class: ProgrammaticWorkloadClass) -> String {
    json!({"strategy":"programmatic","workload_class":class}).to_string()
}

fn return_program() -> String {
    json!({"version":1,"body":[
        {"kind":"return","value":{"kind":"string","value":"program result"}}
    ]})
    .to_string()
}

struct TestTool {
    definition: ToolDefinition,
    fail: bool,
    calls: AtomicU32,
}

struct BlockingTool {
    definition: ToolDefinition,
    entered: Arc<Semaphore>,
    release: Arc<Semaphore>,
}

#[async_trait]
impl Tool for BlockingTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        self.entered.add_permits(1);
        self.release.acquire().await.unwrap().forget();
        Ok(ToolResult::success(json!({"ok":true})))
    }
}

impl TestTool {
    fn new(id: &str, callers: impl IntoIterator<Item = ToolCaller>, fail: bool) -> Self {
        Self {
            definition: ToolDefinition::new(id, id, "test tool", json!({"type":"object"}))
                .with_risk(ToolRisk::Low)
                .with_read_only(true)
                .with_idempotent(true)
                .with_parallel_safe(true)
                .with_allowed_callers(callers),
            fail,
            calls: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl Tool for TestTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            Err(HarnessError::Tool("post-dispatch failure".into()))
        } else {
            Ok(ToolResult::success(json!({"ok":true})))
        }
    }
}

fn registry(tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> ToolRegistry {
    let mut registry = ToolRegistry::default();
    for tool in tools {
        registry.register(tool).unwrap();
    }
    registry
}

#[test]
fn workload_classes_are_stable_snake_case_values() {
    let cases = [
        (ProgrammaticWorkloadClass::Loop, "loop"),
        (ProgrammaticWorkloadClass::FanOut, "fan_out"),
        (ProgrammaticWorkloadClass::Filter, "filter"),
        (ProgrammaticWorkloadClass::Aggregation, "aggregation"),
        (
            ProgrammaticWorkloadClass::LargeIntermediateData,
            "large_intermediate_data",
        ),
    ];
    for (class, encoded) in cases {
        assert_eq!(
            serde_json::to_string(&class).unwrap(),
            format!("\"{encoded}\"")
        );
        assert_eq!(
            serde_json::from_str::<ProgrammaticWorkloadClass>(&format!("\"{encoded}\"")).unwrap(),
            class
        );
    }
}

#[tokio::test]
async fn empty_allowlist_preserves_legacy_planner_prompt_schema_and_name() {
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(r#"{"strategy":"direct"}"#),
            final_response("done"),
        ])
        .with_capabilities(capabilities()),
    );
    let tool = Arc::new(TestTool::new(
        "both",
        [ToolCaller::Direct, ToolCaller::DeclarativePlan],
        false,
    ));
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([tool as Arc<dyn Tool>]))
        .programmatic(ProgrammaticHostConfig::default())
        .build()
        .run(request(&["both"], 3))
        .await
        .unwrap();

    assert_eq!(result.final_output.as_deref(), Some("done"));
    let requests = provider.requests();
    assert_eq!(requests[0].messages[0].content, LEGACY_PLANNER_PROMPT);
    let structured = requests[0].structured_output.as_ref().unwrap();
    assert_eq!(structured.name, "llama_harness_planner_envelope_v1");
    assert!(!structured.schema.to_string().contains("programmatic"));
}

#[tokio::test]
async fn empty_allowlist_preserves_legacy_no_tools_direct_path() {
    let provider = Arc::new(
        MockModelProvider::scripted([final_response("done")]).with_capabilities(capabilities()),
    );
    let result = AgentRunner::builder(provider.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build()
        .run(request(&[], 3))
        .await
        .unwrap();

    assert_eq!(result.final_output.as_deref(), Some("done"));
    let requests = provider.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].structured_output.is_none());
}

#[tokio::test]
async fn every_allowlisted_workload_class_can_be_promoted() {
    let classes = [
        ProgrammaticWorkloadClass::Loop,
        ProgrammaticWorkloadClass::FanOut,
        ProgrammaticWorkloadClass::Filter,
        ProgrammaticWorkloadClass::Aggregation,
        ProgrammaticWorkloadClass::LargeIntermediateData,
    ];
    for class in classes {
        let provider = Arc::new(
            MockModelProvider::scripted([
                final_response(proposal(class)),
                final_response(return_program()),
                final_response("done"),
            ])
            .with_capabilities(capabilities()),
        );
        let events = Arc::new(InMemoryEventSink::default());
        let result = AgentRunner::builder(provider.clone())
            .event_sink(events.clone())
            .programmatic(ProgrammaticHostConfig::default())
            .adaptive_programmatic_allowlist([class])
            .build()
            .run(request(&[], 3))
            .await
            .unwrap();

        assert_eq!(result.status, RunStatus::Completed, "{class:?}");
        assert_eq!(result.final_output.as_deref(), Some("done"));
        let requests = provider.requests();
        assert_eq!(requests.len(), 3);
        let planner = requests[0].structured_output.as_ref().unwrap();
        assert_eq!(
            planner.name,
            "llama_harness_adaptive_programmatic_envelope_v1"
        );
        let schema = planner.schema.to_string();
        assert!(schema.contains(class_name(class)));
        assert!(events.events().iter().any(|record| matches!(
            record.event,
            RunEvent::StrategySelected {
                requested: RunStrategy::Adaptive,
                selected: RunStrategy::Programmatic,
                reason: StrategySelectionReason::AdaptivePlanner,
            }
        )));
    }
}

fn class_name(class: ProgrammaticWorkloadClass) -> &'static str {
    match class {
        ProgrammaticWorkloadClass::Loop => "loop",
        ProgrammaticWorkloadClass::FanOut => "fan_out",
        ProgrammaticWorkloadClass::Filter => "filter",
        ProgrammaticWorkloadClass::Aggregation => "aggregation",
        ProgrammaticWorkloadClass::LargeIntermediateData => "large_intermediate_data",
        _ => unreachable!("new workload classes must be added to this contract test"),
    }
}

#[tokio::test]
async fn known_unpromoted_class_falls_back_without_repair_or_program_generation() {
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(proposal(ProgrammaticWorkloadClass::FanOut)),
            final_response("direct"),
        ])
        .with_capabilities(capabilities()),
    );
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider.clone())
        .event_sink(events.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .adaptive_programmatic_allowlist([ProgrammaticWorkloadClass::Loop])
        .build()
        .run(request(&[], 3))
        .await
        .unwrap();

    assert_eq!(result.final_output.as_deref(), Some("direct"));
    assert_eq!(provider.requests().len(), 2);
    assert!(!events
        .events()
        .iter()
        .any(|record| matches!(record.event, RunEvent::ProgramLifecycle { .. })));
    assert!(events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyFallback {
            from: RunStrategy::Programmatic,
            to: RunStrategy::Direct,
            reason: StrategyFallbackReason::UnsupportedCapability,
        }
    )));
}

#[tokio::test]
async fn unknown_workload_class_uses_the_single_existing_planner_repair() {
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(r#"{"strategy":"programmatic","workload_class":"unknown"}"#),
            final_response(proposal(ProgrammaticWorkloadClass::Loop)),
            final_response(return_program()),
            final_response("done"),
        ])
        .with_capabilities(capabilities()),
    );
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider.clone())
        .event_sink(events.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .adaptive_programmatic_allowlist([ProgrammaticWorkloadClass::Loop])
        .build()
        .run(request(&[], 4))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(provider.requests().len(), 4);
    assert!(events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyUsage {
            model_calls: 4,
            planning_model_calls: 2,
            repair_model_calls: 1,
            final_synthesis_model_calls: 1,
            reactive_model_calls: 0,
            ..
        }
    )));
}

#[tokio::test]
async fn forced_declarative_rejects_a_programmatic_envelope() {
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(proposal(ProgrammaticWorkloadClass::Loop))])
            .with_capabilities(capabilities()),
    );
    let result = AgentRunner::builder(provider.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .adaptive_programmatic_allowlist([ProgrammaticWorkloadClass::Loop])
        .build()
        .run_with_strategy(request(&[], 2), RunStrategy::DeclarativePlan)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Failed);
    let requests = provider.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].messages[0].content, LEGACY_PLANNER_PROMPT);
    let structured = requests[0].structured_output.as_ref().unwrap();
    assert_eq!(structured.name, "llama_harness_declarative_plan_v1");
    assert!(!structured.schema.to_string().contains("programmatic"));
}

#[tokio::test]
async fn missing_host_capability_byte_or_remaining_call_budget_falls_back_before_generation() {
    for gate in ["host", "capability", "bytes", "calls"] {
        let mut provider_capabilities = capabilities();
        if gate == "capability" {
            provider_capabilities.supports_programmatic_calling = false;
            provider_capabilities.programmatic_conformance = None;
        } else if gate == "bytes" {
            provider_capabilities.limits.max_program_bytes = None;
        }
        let provider = Arc::new(
            MockModelProvider::scripted([final_response("direct")])
                .with_capabilities(provider_capabilities),
        );
        let mut builder = AgentRunner::builder(provider.clone())
            .adaptive_programmatic_allowlist([ProgrammaticWorkloadClass::Loop]);
        if gate != "host" {
            builder = builder.programmatic(ProgrammaticHostConfig::default());
        }
        let result = builder
            .build()
            .run(request(&[], if gate == "calls" { 2 } else { 3 }))
            .await
            .unwrap();

        assert_eq!(result.final_output.as_deref(), Some("direct"), "{gate}");
        assert_eq!(provider.requests().len(), 1, "{gate}");
        assert!(provider.requests()[0].structured_output.is_none(), "{gate}");
    }
}

#[tokio::test]
async fn invalid_host_config_is_terminal_after_the_adaptive_proposal() {
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(proposal(ProgrammaticWorkloadClass::Loop))])
            .with_capabilities(capabilities()),
    );
    let result = AgentRunner::builder(provider.clone())
        .programmatic(ProgrammaticHostConfig {
            max_active_vms: 0,
            ..ProgrammaticHostConfig::default()
        })
        .adaptive_programmatic_allowlist([ProgrammaticWorkloadClass::Loop])
        .build()
        .run(request(&[], 3))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Failed);
    assert_eq!(provider.requests().len(), 1);
    assert!(!result.errors.is_empty());
}

#[tokio::test]
async fn promotion_selects_a_fresh_programmatic_caller_scope() {
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(proposal(ProgrammaticWorkloadClass::Loop)),
            final_response(return_program()),
            final_response("done"),
        ])
        .with_capabilities(capabilities()),
    );
    let program_only = Arc::new(TestTool::new(
        "program_only",
        [ToolCaller::Programmatic],
        false,
    ));
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([program_only as Arc<dyn Tool>]))
        .event_sink(events.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .adaptive_programmatic_allowlist([ProgrammaticWorkloadClass::Loop])
        .build()
        .run(request(&["program_only"], 3))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    let requests = provider.requests();
    assert_eq!(requests[0].tools.len(), 1);
    assert_eq!(requests[0].tools[0].id, "program_only");
    assert_eq!(requests[1].tools.len(), 1);
    assert_eq!(requests[1].tools[0].id, "program_only");
    let callers = events
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            RunEvent::ToolDiscoveryCompleted { caller, .. } => Some(caller),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        callers,
        [
            ToolCaller::DeclarativePlan,
            ToolCaller::Direct,
            ToolCaller::Programmatic
        ]
    );
}

#[tokio::test]
async fn programmatic_promotion_does_not_require_declarative_plan_support() {
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(proposal(ProgrammaticWorkloadClass::Loop)),
            final_response(return_program()),
            final_response("done"),
        ])
        .with_capabilities(capabilities().with_structured_plans(false)),
    );
    let program_only = Arc::new(TestTool::new(
        "program_only",
        [ToolCaller::Programmatic],
        false,
    ));
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([program_only as Arc<dyn Tool>]))
        .event_sink(events.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .adaptive_programmatic_allowlist([ProgrammaticWorkloadClass::Loop])
        .build()
        .run(request(&["program_only"], 3))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.final_output.as_deref(), Some("done"));
    let requests = provider.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].tools.len(), 1);
    assert_eq!(requests[0].tools[0].id, "program_only");
    assert_eq!(requests[1].tools.len(), 1);
    assert_eq!(requests[1].tools[0].id, "program_only");
    let planner = requests[0].structured_output.as_ref().unwrap();
    assert_eq!(
        planner.name,
        "llama_harness_adaptive_programmatic_envelope_v1"
    );
    let schema = planner.schema.to_string();
    assert!(schema.contains("programmatic"));
    assert!(!schema.contains("declarative_plan"));
    assert!(!requests[0].messages[0]
        .content
        .contains("finite dependency DAG"));
    assert!(events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::StrategySelected {
            requested: RunStrategy::Adaptive,
            selected: RunStrategy::Programmatic,
            reason: StrategySelectionReason::AdaptivePlanner,
        }
    )));
}

#[tokio::test]
async fn programmatic_planner_scope_limit_stops_before_model_or_tool_work() {
    let provider = Arc::new(MockModelProvider::scripted([]).with_capabilities(capabilities()));
    let one = Arc::new(TestTool::new("one", [ToolCaller::Programmatic], false));
    let two = Arc::new(TestTool::new("two", [ToolCaller::Programmatic], false));
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([one as Arc<dyn Tool>, two as Arc<dyn Tool>]))
        .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(1))
        .programmatic(ProgrammaticHostConfig::default())
        .adaptive_programmatic_allowlist([ProgrammaticWorkloadClass::Loop])
        .build()
        .run(request(&["one", "two"], 3))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::LimitReached);
    assert_eq!(result.final_output, None);
    assert!(provider.requests().is_empty());
}

#[tokio::test]
async fn adaptive_programmatic_admission_saturation_is_fail_fast() {
    let invoke = json!({"version":1,"body":[
        {"kind":"invoke","name":"result","tool_id":"block","arguments":{"kind":"object","entries":[]}},
        {"kind":"return","value":{"kind":"variable","name":"result"}}
    ]})
    .to_string();
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(proposal(ProgrammaticWorkloadClass::Loop)),
            final_response(invoke),
            final_response(proposal(ProgrammaticWorkloadClass::Loop)),
            final_response("done"),
        ])
        .with_capabilities(capabilities()),
    );
    let entered = Arc::new(Semaphore::new(0));
    let release = Arc::new(Semaphore::new(0));
    let tool = Arc::new(BlockingTool {
        definition: ToolDefinition::new(
            "block",
            "block",
            "blocking tool",
            json!({"type":"object"}),
        )
        .with_risk(ToolRisk::Low)
        .with_read_only(true)
        .with_idempotent(true)
        .with_allowed_callers([ToolCaller::Programmatic]),
        entered: entered.clone(),
        release: release.clone(),
    });
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry([tool as Arc<dyn Tool>]))
            .policy(Arc::new(AllowAllPolicy))
            .programmatic(ProgrammaticHostConfig {
                max_active_vms: 1,
                ..ProgrammaticHostConfig::default()
            })
            .adaptive_programmatic_allowlist([ProgrammaticWorkloadClass::Loop])
            .build(),
    );
    let first = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(request(&["block"], 3)).await.unwrap() })
    };
    entered.acquire().await.unwrap().forget();

    let second = runner.run(request(&["block"], 3)).await.unwrap();
    assert_eq!(second.status, RunStatus::LimitReached);
    assert_eq!(provider.requests().len(), 3);

    release.add_permits(1);
    assert_eq!(first.await.unwrap().status, RunStatus::Completed);
    assert_eq!(provider.requests().len(), 4);
}

#[tokio::test]
async fn invalid_program_fallback_keeps_one_run_emitter_scope_and_counters() {
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(proposal(ProgrammaticWorkloadClass::Loop)),
            final_response("not a program"),
            final_response("still not a program"),
            final_response("direct"),
        ])
        .with_capabilities(capabilities()),
    );
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider.clone())
        .event_sink(events.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .adaptive_programmatic_allowlist([ProgrammaticWorkloadClass::Loop])
        .build()
        .run(request(&[], 4))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.id, "adaptive-programmatic-run");
    assert_eq!(result.trace_id, "adaptive-programmatic-trace");
    assert_eq!(provider.requests().len(), 4);
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
    assert_eq!(
        records
            .iter()
            .filter(|record| matches!(record.event, RunEvent::StrategyUsage { .. }))
            .count(),
        1
    );
    let execution_id = &records[0].execution_id;
    assert!(records.iter().enumerate().all(|(index, record)| {
        record.run_id == result.id
            && record.trace_id == result.trace_id
            && &record.execution_id == execution_id
            && record.sequence == (index + 1) as u64
    }));
    assert!(records.iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyFallback {
            from: RunStrategy::Programmatic,
            to: RunStrategy::Direct,
            reason: StrategyFallbackReason::InvalidProgram,
        }
    )));
    assert!(records.iter().any(|record| matches!(
        record.event,
        RunEvent::StrategySelected {
            requested: RunStrategy::Adaptive,
            selected: RunStrategy::Direct,
            reason: StrategySelectionReason::CapabilityDowngrade,
        }
    )));
    assert!(records.iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyUsage {
            strategy: RunStrategy::Direct,
            model_calls: 4,
            planning_model_calls: 2,
            repair_model_calls: 1,
            final_synthesis_model_calls: 0,
            reactive_model_calls: 1,
            ..
        }
    )));
    assert_eq!(
        records
            .iter()
            .filter(|record| matches!(
                record.event,
                RunEvent::ToolDiscoveryCompleted {
                    caller: ToolCaller::Direct,
                    ..
                }
            ))
            .count(),
        1,
        "the prepared Direct scope must be reused"
    );
}

#[tokio::test]
async fn a_post_dispatch_failure_never_falls_back_or_replays() {
    let invoke = json!({"version":1,"body":[
        {"kind":"invoke","name":"result","tool_id":"fail","arguments":{"kind":"object","entries":[]}},
        {"kind":"return","value":{"kind":"variable","name":"result"}}
    ]})
    .to_string();
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(proposal(ProgrammaticWorkloadClass::Loop)),
            final_response(invoke),
        ])
        .with_capabilities(capabilities()),
    );
    let tool = Arc::new(TestTool::new("fail", [ToolCaller::Programmatic], true));
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([tool.clone() as Arc<dyn Tool>]))
        .policy(Arc::new(AllowAllPolicy))
        .event_sink(events.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .adaptive_programmatic_allowlist([ProgrammaticWorkloadClass::Loop])
        .build()
        .run(request(&["fail"], 4))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Failed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.requests().len(), 2);
    assert!(!events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyFallback {
            reason: StrategyFallbackReason::InvalidProgram,
            ..
        }
    )));
    assert!(!result.errors.is_empty());
}
