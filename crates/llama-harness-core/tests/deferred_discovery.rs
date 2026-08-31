use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, tool_response, MockModelProvider, MockStep},
    AgentDefinition, AgentRunner, HarnessError, InMemoryEventSink, ModelCapabilities, ModelInfo,
    ModelProvider, ModelRequest, ModelResponse, ProviderCapabilityLimits, ProviderHealth, RunEvent,
    RunRequest, RunResult, RunStatus, RunStrategy, Tool, ToolCall, ToolCaller, ToolDefinition,
    ToolDiscoveryLimits, ToolDiscoveryMetadata, ToolDiscoveryOutcome, ToolDiscoverySelection,
    ToolRegistry, ToolResult,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};
use tokio_util::sync::CancellationToken;

struct CountingTool {
    definition: ToolDefinition,
    calls: AtomicU32,
    result: ToolResult,
}

impl CountingTool {
    fn new(id: &str) -> Self {
        Self {
            definition: ToolDefinition::new(
                id,
                id.replace('.', " "),
                "safe catalog description",
                json!({"type": "object", "additionalProperties": false}),
            )
            .with_read_only(true)
            .with_idempotent(true)
            .with_parallel_safe(true)
            .with_allowed_callers([ToolCaller::Direct, ToolCaller::DeclarativePlan]),
            calls: AtomicU32::new(0),
            result: ToolResult::success(json!({"value": 42})),
        }
    }

    fn with_result(mut self, result: ToolResult) -> Self {
        self.result = result;
        self
    }
}

#[async_trait]
impl Tool for CountingTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.result.clone())
    }
}

struct ObservedProvider {
    capabilities: ModelCapabilities,
    capability_calls: AtomicU32,
    completion_calls: AtomicU32,
}

#[async_trait]
impl ModelProvider for ObservedProvider {
    fn id(&self) -> &str {
        "observed"
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.capability_calls.fetch_add(1, Ordering::SeqCst);
        self.capabilities.clone()
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth::healthy())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(Vec::new())
    }

    async fn complete(&self, _: ModelRequest) -> Result<ModelResponse, HarnessError> {
        self.completion_calls.fetch_add(1, Ordering::SeqCst);
        Ok(ModelResponse::new("mock-model").with_final_output("done"))
    }
}

fn capabilities(max_tools: u32, planning: bool) -> ModelCapabilities {
    capabilities_with_schema_bytes(max_tools, planning, 64 * 1024)
}

fn capabilities_with_schema_bytes(
    max_tools: u32,
    planning: bool,
    schema_bytes: u64,
) -> ModelCapabilities {
    ModelCapabilities::new(true, false, true)
        .with_structured_plans(planning)
        .with_limits(
            ProviderCapabilityLimits::new()
                .with_max_tools(max_tools)
                .with_max_tool_schema_bytes(schema_bytes)
                .with_max_plan_nodes(16)
                .with_max_plan_bytes(64 * 1024),
        )
}

fn registry(count: usize) -> (ToolRegistry, Vec<Arc<CountingTool>>) {
    let mut registry = ToolRegistry::default();
    let mut tools = Vec::new();
    for index in 0..count {
        let id = format!("catalog.tool.{index:03}");
        let tool = Arc::new(CountingTool::new(&id));
        registry
            .register_with_discovery(
                tool.clone(),
                ToolDiscoveryMetadata::deferred()
                    .with_namespace("catalog")
                    .with_aliases([format!("alias.{index:03}")]),
            )
            .unwrap();
        tools.push(tool);
    }
    (registry, tools)
}

fn request(count: usize, input: &str) -> RunRequest {
    let mut agent = AgentDefinition::new("discovery", "Discovery", "1", "mock-model");
    agent.tool_allowlist = (0..count)
        .map(|index| format!("catalog.tool.{index:03}"))
        .collect();
    RunRequest::new(agent, input)
}

#[tokio::test]
async fn legacy_hot_registration_preserves_unrestricted_small_catalog_definitions() {
    let definitions = vec![
        ToolDefinition::new(
            "  Mixed Case/工具 !?.  ",
            "Unicode display 名称",
            "legacy hot definition",
            json!({"type": "object"}),
        )
        .with_read_only(true)
        .with_allowed_callers([ToolCaller::Direct]),
        ToolDefinition::new(
            "punctuation:@#$%^&*()[]{}",
            " punctuation\nand spaces ",
            "legacy hot definition",
            json!({"type": "object"}),
        )
        .with_read_only(true)
        .with_allowed_callers([ToolCaller::Direct]),
        ToolDefinition::new(
            format!("long-{}", "x".repeat(300)),
            "",
            "legacy hot definition",
            json!({"type": "object"}),
        )
        .with_read_only(true)
        .with_allowed_callers([ToolCaller::Direct]),
    ];
    let mut registry = ToolRegistry::default();
    let mut tools = Vec::new();
    for definition in &definitions {
        let tool = Arc::new(CountingTool {
            definition: definition.clone(),
            calls: AtomicU32::new(0),
            result: ToolResult::success(json!({"ok": true})),
        });
        registry.register(tool.clone()).unwrap();
        tools.push(tool);
    }
    let provider = Arc::new(
        MockModelProvider::scripted([
            tool_response(ToolCall::new("legacy-call", &definitions[0].id, "{}")),
            final_response("done"),
        ])
        .with_capabilities(capabilities(10, false)),
    );
    let mut agent = AgentDefinition::new("legacy", "Legacy", "1", "mock-model");
    agent.tool_allowlist = definitions
        .iter()
        .map(|definition| definition.id.clone())
        .collect();
    agent
        .tool_allowlist
        .push(definitions[0].id.trim().to_owned());
    let result = AgentRunner::builder(provider.clone())
        .tools(registry)
        .build()
        .run_with_strategy(
            RunRequest::new(agent, "use legacy tools"),
            RunStrategy::Direct,
        )
        .await
        .unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    for request in requests {
        assert_eq!(request.tools, definitions);
        assert_eq!(
            serde_json::to_vec(&request.tools).unwrap(),
            serde_json::to_vec(&definitions).unwrap()
        );
    }
    assert_eq!(tools[0].calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn direct_e2e_reuses_one_immutable_selected_scope_and_emits_private_counters() {
    let count = 1_000;
    let target = "catalog.tool.733";
    let provider = Arc::new(
        MockModelProvider::scripted([
            tool_response(ToolCall::new("call-1", target, "{}")),
            final_response("done"),
        ])
        .with_capabilities(capabilities(2, false)),
    );
    let (registry, tools) = registry(count);
    let events = Arc::new(InMemoryEventSink::default());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry)
        .event_sink(events.clone())
        .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(4))
        .build();

    let result = runner
        .run_with_strategy(request(count, target), RunStrategy::Direct)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(tools[733].calls.load(Ordering::SeqCst), 1);
    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    for model_request in &requests {
        assert_eq!(model_request.tools.len(), 1);
        assert_eq!(model_request.tools[0].id, target);
        assert_eq!(
            model_request
                .prepared_tools
                .as_ref()
                .unwrap()
                .serialized_definitions(),
            serde_json::to_vec(&model_request.tools).unwrap()
        );
    }
    assert!(Arc::ptr_eq(
        requests[0].prepared_tools.as_ref().unwrap(),
        requests[1].prepared_tools.as_ref().unwrap()
    ));
    let discovery = events
        .events()
        .into_iter()
        .find_map(|record| match record.event {
            RunEvent::ToolDiscoveryCompleted {
                candidate_count,
                selected_count,
                deferred_candidate_count,
                catalog_exceeded_budget,
                ..
            } => Some((
                candidate_count,
                selected_count,
                deferred_candidate_count,
                catalog_exceeded_budget,
                serde_json::to_string(&record.event).unwrap(),
            )),
            _ => None,
        })
        .expect("discovery event");
    assert_eq!(discovery.0, 1_000);
    assert_eq!(discovery.1, 1);
    assert_eq!(discovery.2, 1_000);
    assert!(discovery.3);
    for forbidden in [
        target,
        "safe catalog description",
        "\"query\":",
        "\"tool_ids\":",
        "\"name\":",
        "\"namespace\":",
        "\"aliases\":",
        "\"description\":",
        "\"schema\":",
        "fingerprint",
        "cache_hit",
        "cache_build",
        "eviction",
        "model_output",
        "raw_error",
    ] {
        assert!(!discovery.4.contains(forbidden), "leaked {forbidden}");
    }
}

#[tokio::test]
async fn planner_repair_and_final_synthesis_reuse_selected_scopes() {
    let count = 30;
    let target = "catalog.tool.017";
    let repaired = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "node", "tool_id": target, "arguments": {}}]}
    })
    .to_string();
    let bypass = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "bypass", "tool_id": "catalog.tool.018", "arguments": {}}]}
    })
    .to_string();
    let provider = Arc::new(
        MockModelProvider::scripted([
            MockStep::Response(ModelResponse::new("mock-model").with_final_output(bypass)),
            MockStep::Response(ModelResponse::new("mock-model").with_final_output(repaired)),
            final_response("synthesized"),
        ])
        .with_capabilities(capabilities(2, true)),
    );
    let (registry, tools) = registry(count);
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry)
        .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(4))
        .build();

    let result = runner
        .run_with_strategy(request(count, target), RunStrategy::DeclarativePlan)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(tools[17].calls.load(Ordering::SeqCst), 1);
    assert_eq!(tools[18].calls.load(Ordering::SeqCst), 0);
    let requests = provider.requests();
    assert_eq!(requests.len(), 3);
    assert!(requests
        .iter()
        .all(|request| request.tools.len() == 1 && request.tools[0].id == target));
    let prepared = requests[0].prepared_tools.as_ref().unwrap();
    assert!(requests
        .iter()
        .all(|request| Arc::ptr_eq(prepared, request.prepared_tools.as_ref().unwrap())));
}

#[tokio::test]
async fn execution_recovery_reuses_the_prepared_plan_scope() {
    let failed_plan = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "failed", "tool_id": "recovery.fail", "arguments": {}}]}
    })
    .to_string();
    let recovered_plan = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "recovered", "tool_id": "recovery.good", "arguments": {}}]}
    })
    .to_string();
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(failed_plan),
            final_response(recovered_plan),
            final_response("synthesized"),
        ])
        .with_capabilities(capabilities(2, true)),
    );
    let failed = Arc::new(
        CountingTool::new("recovery.fail").with_result(ToolResult::failure("expected failure")),
    );
    let recovered = Arc::new(CountingTool::new("recovery.good"));
    let mut registry = ToolRegistry::default();
    let mut allowlist = vec!["recovery.fail".into(), "recovery.good".into()];
    for tool in [&failed, &recovered] {
        registry
            .register_with_discovery(
                tool.clone(),
                ToolDiscoveryMetadata::deferred().with_namespace("recovery"),
            )
            .unwrap();
    }
    for index in 0..30 {
        let id = format!("distractor.tool.{index:03}");
        allowlist.push(id.clone());
        registry
            .register_with_discovery(
                Arc::new(CountingTool::new(&id)),
                ToolDiscoveryMetadata::deferred().with_namespace("distractor"),
            )
            .unwrap();
    }
    let mut agent = AgentDefinition::new("recovery", "Recovery", "1", "mock-model");
    agent.tool_allowlist = allowlist;
    let request = RunRequest::new(agent, "recovery");
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider.clone())
        .tools(registry)
        .event_sink(events.clone())
        .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(2))
        .build()
        .run(request)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(failed.calls.load(Ordering::SeqCst), 1);
    assert_eq!(recovered.calls.load(Ordering::SeqCst), 1);
    let expected = vec!["recovery.fail".to_owned(), "recovery.good".to_owned()];
    let requests = provider.requests();
    assert_eq!(requests.len(), 3);
    assert!(requests.iter().all(|request| {
        request
            .tools
            .iter()
            .map(|tool| tool.id.clone())
            .collect::<Vec<_>>()
            == expected
    }));
    assert!(events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyFallback {
            reason: llama_harness_core::StrategyFallbackReason::ExecutionRecovery,
            ..
        }
    )));
}

#[tokio::test]
async fn adaptive_invalid_plan_and_planner_failure_fallbacks_reuse_selected_scopes() {
    let count = 30;
    let target = "catalog.tool.011";
    let invalid_events = Arc::new(InMemoryEventSink::default());
    let invalid_provider = Arc::new(
        MockModelProvider::scripted([
            final_response("not-json"),
            final_response("still-not-json"),
            final_response("fallback"),
        ])
        .with_capabilities(capabilities(1, true)),
    );
    let (invalid_registry, _) = registry(count);
    let invalid = AgentRunner::builder(invalid_provider.clone())
        .tools(invalid_registry)
        .event_sink(invalid_events.clone())
        .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(1))
        .build()
        .run(request(count, target))
        .await
        .unwrap();
    assert_eq!(invalid.status, RunStatus::Completed);
    let invalid_requests = invalid_provider.requests();
    assert_eq!(invalid_requests.len(), 3);
    assert!(invalid_requests
        .iter()
        .all(|request| { request.tools.len() == 1 && request.tools[0].id == target }));
    let invalid_discovery = invalid_events
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            RunEvent::ToolDiscoveryCompleted { caller, .. } => Some(caller),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        invalid_discovery,
        vec![ToolCaller::DeclarativePlan, ToolCaller::Direct]
    );

    let failure_events = Arc::new(InMemoryEventSink::default());
    let failure_provider = Arc::new(
        MockModelProvider::scripted([
            MockStep::Error(HarnessError::Provider("expected planner failure".into())),
            final_response("fallback"),
        ])
        .with_capabilities(capabilities(1, true)),
    );
    let (failure_registry, _) = registry(count);
    let failure = AgentRunner::builder(failure_provider.clone())
        .tools(failure_registry)
        .event_sink(failure_events.clone())
        .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(1))
        .build()
        .run(request(count, target))
        .await
        .unwrap();
    assert_eq!(failure.status, RunStatus::Completed);
    let failure_requests = failure_provider.requests();
    assert_eq!(failure_requests.len(), 2);
    assert!(failure_requests
        .iter()
        .all(|request| { request.tools.len() == 1 && request.tools[0].id == target }));
    assert_eq!(
        failure_events
            .events()
            .into_iter()
            .filter_map(|record| match record.event {
                RunEvent::ToolDiscoveryCompleted { caller, .. } => Some(caller),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec![ToolCaller::DeclarativePlan, ToolCaller::Direct]
    );
}

#[tokio::test]
async fn known_unselected_and_unknown_calls_have_the_same_external_rejection() {
    async fn rejection(known_probe: bool) -> (Value, Vec<Value>, Vec<Value>, u32) {
        let target = "catalog.target";
        let probe = "catalog.probe";
        let replacement = "catalog.replacement";
        let provider = Arc::new(
            MockModelProvider::scripted([
                tool_response(ToolCall::new("bypass", probe, "{}")),
                final_response("done"),
            ])
            .with_capabilities(capabilities(1, false)),
        );
        let events = Arc::new(InMemoryEventSink::default());
        let mut registry = ToolRegistry::default();
        let mut tools = Vec::new();
        let mut allowlist = vec![target.into(), probe.into(), replacement.into()];
        for id in std::iter::once(target.to_owned())
            .chain((0..28).map(|index| format!("catalog.distractor.{index:03}")))
            .chain(std::iter::once(if known_probe {
                probe.to_owned()
            } else {
                replacement.to_owned()
            }))
        {
            if id.starts_with("catalog.distractor") {
                allowlist.push(id.clone());
            }
            let tool = Arc::new(CountingTool::new(&id));
            registry
                .register_with_discovery(
                    tool.clone(),
                    ToolDiscoveryMetadata::deferred()
                        .with_aliases([format!("alias-{}", tools.len())]),
                )
                .unwrap();
            tools.push(tool);
        }
        let mut agent = AgentDefinition::new("equivalence", "Equivalence", "1", "mock-model");
        agent.tool_allowlist = allowlist;
        let run_request = RunRequest::new(agent, target)
            .with_run_id("equivalent-run")
            .with_trace_id("equivalent-trace");
        let runner = AgentRunner::builder(provider.clone())
            .tools(registry)
            .event_sink(events.clone())
            .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(1))
            .build();
        let result = runner
            .run_with_strategy(run_request, RunStrategy::Direct)
            .await
            .unwrap();
        let result = json!({
            "status": result.status,
            "final_output": result.final_output,
            "tool_calls": result.tool_calls,
            "errors": result.errors,
        });
        let requests = provider
            .requests()
            .into_iter()
            .map(|request| serde_json::to_value(request).unwrap())
            .collect::<Vec<_>>();
        let event_values = events
            .events()
            .into_iter()
            .map(|record| {
                let mut value = serde_json::to_value(record.event).unwrap();
                if matches!(
                    value.get("type").and_then(Value::as_str),
                    Some("strategy_usage" | "tool_discovery_completed")
                ) {
                    value["duration_ms"] = json!(0);
                }
                value
            })
            .collect::<Vec<_>>();
        let calls = tools
            .iter()
            .map(|tool| tool.calls.load(Ordering::SeqCst))
            .sum();
        (result, requests, event_values, calls)
    }

    let known = rejection(true).await;
    let unknown = rejection(false).await;
    assert_eq!(known, unknown);
    assert_eq!(known.3, 0);
}

#[tokio::test]
async fn hot_overflow_fails_before_model_use_and_zero_provider_capacity_is_no_tool() {
    let provider = Arc::new(
        MockModelProvider::scripted([final_response("unused")])
            .with_capabilities(capabilities(1, false)),
    );
    let mut hot = ToolRegistry::default();
    for index in 0..2 {
        hot.register(Arc::new(CountingTool::new(&format!("hot.{index}"))))
            .unwrap();
    }
    let mut hot_request = request(0, "hot.0");
    hot_request.agent.tool_allowlist = vec!["hot.0".into(), "hot.1".into()];
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider.clone())
        .tools(hot)
        .event_sink(events.clone())
        .build()
        .run_with_strategy(hot_request, RunStrategy::Direct)
        .await
        .unwrap();
    assert_eq!(result.status, RunStatus::LimitReached);
    assert!(provider.requests().is_empty());
    assert!(matches!(
        events.events()[1].event,
        RunEvent::ToolDiscoveryCompleted {
            outcome: ToolDiscoveryOutcome::LimitReached,
            selection: ToolDiscoverySelection::CountLimit,
            candidate_count: 2,
            selected_count: 2,
            ..
        }
    ));

    let zero_provider = Arc::new(
        MockModelProvider::scripted([final_response("no tools")])
            .with_capabilities(capabilities(0, false)),
    );
    let (zero_registry, _) = registry(30);
    let result = AgentRunner::builder(zero_provider.clone())
        .tools(zero_registry)
        .build()
        .run_with_strategy(request(30, "catalog.tool.001"), RunStrategy::Direct)
        .await
        .unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    assert!(zero_provider.requests()[0].tools.is_empty());
    assert!(zero_provider.requests()[0].prepared_tools.is_none());

    for schema_bytes in [0, 1] {
        let provider = Arc::new(
            MockModelProvider::scripted([final_response("no tools")])
                .with_capabilities(capabilities_with_schema_bytes(8, true, schema_bytes)),
        );
        let (registry, _) = registry(30);
        let result = AgentRunner::builder(provider.clone())
            .tools(registry)
            .build()
            .run(request(30, "catalog.tool.001"))
            .await
            .unwrap();
        assert_eq!(result.status, RunStatus::Completed);
        assert!(provider.requests()[0].prepared_tools.is_none());
        let requests = provider.requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].tools.is_empty());
    }
}

#[tokio::test]
async fn exact_namespace_overflow_is_a_zero_effect_terminal_limit() {
    for strategy in [
        RunStrategy::Direct,
        RunStrategy::Adaptive,
        RunStrategy::DeclarativePlan,
    ] {
        let provider = Arc::new(
            MockModelProvider::scripted([final_response("unused")])
                .with_capabilities(capabilities(1, true)),
        );
        let events = Arc::new(InMemoryEventSink::default());
        let mut registry = ToolRegistry::default();
        let mut tools = Vec::new();
        for id in ["exact.one", "exact.two"] {
            let tool = Arc::new(CountingTool::new(id));
            registry
                .register_with_discovery(
                    tool.clone(),
                    ToolDiscoveryMetadata::deferred().with_namespace("exact"),
                )
                .unwrap();
            tools.push(tool);
        }
        let mut agent = AgentDefinition::new("exact", "Exact", "1", "mock-model");
        agent.tool_allowlist = vec!["exact.one".into(), "exact.two".into()];
        let result = AgentRunner::builder(provider.clone())
            .tools(registry)
            .event_sink(events.clone())
            .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(1))
            .build()
            .run_with_strategy(RunRequest::new(agent, "exact"), strategy)
            .await
            .unwrap();
        assert_eq!(result.status, RunStatus::LimitReached);
        assert!(result.final_output.is_none());
        assert!(result.tool_calls.is_empty());
        assert!(result.policy_decisions.is_empty());
        assert!(result.approvals.is_empty());
        assert!(provider.requests().is_empty());
        let records = events.events();
        let discoveries = records
            .iter()
            .filter(|record| matches!(record.event, RunEvent::ToolDiscoveryCompleted { .. }))
            .collect::<Vec<_>>();
        assert!(!discoveries.is_empty());
        let RunEvent::ToolDiscoveryCompleted {
            outcome,
            selection,
            effective_tool_count_budget,
            selected_count,
            expansion_limit,
            ..
        } = discoveries.last().unwrap().event
        else {
            unreachable!()
        };
        assert_eq!(outcome, ToolDiscoveryOutcome::LimitReached);
        assert_eq!(selection, ToolDiscoverySelection::CountLimit);
        assert_eq!(effective_tool_count_budget, 1);
        assert_eq!(selected_count, 2);
        assert_eq!(expansion_limit, 8);
        assert!(matches!(records[0].event, RunEvent::Started { .. }));
        assert!(matches!(
            records[records.len() - 2].event,
            RunEvent::StrategyUsage {
                model_calls: 0,
                tool_calls: 0,
                ..
            }
        ));
        assert!(matches!(
            records.last().unwrap().event,
            RunEvent::Completed {
                status: RunStatus::LimitReached
            }
        ));
        assert_eq!(
            tools
                .iter()
                .map(|tool| tool.calls.load(Ordering::SeqCst))
                .sum::<u32>(),
            0
        );
    }
}

#[tokio::test]
async fn discovery_events_cover_every_value_free_selection_category() {
    async fn selection_for(
        registry: ToolRegistry,
        allowlist: Vec<String>,
        input: &str,
        limits: ToolDiscoveryLimits,
        capabilities: ModelCapabilities,
    ) -> (RunResult, RunEvent) {
        let provider = Arc::new(
            MockModelProvider::scripted([final_response("done")]).with_capabilities(capabilities),
        );
        let events = Arc::new(InMemoryEventSink::default());
        let mut agent = AgentDefinition::new("categories", "Categories", "1", "mock-model");
        agent.tool_allowlist = allowlist;
        let result = AgentRunner::builder(provider)
            .tools(registry)
            .event_sink(events.clone())
            .discovery_limits(limits)
            .build()
            .run_with_strategy(RunRequest::new(agent, input), RunStrategy::Direct)
            .await
            .unwrap();
        let discoveries = events
            .events()
            .into_iter()
            .filter_map(|record| {
                matches!(record.event, RunEvent::ToolDiscoveryCompleted { .. })
                    .then_some(record.event)
            })
            .collect::<Vec<_>>();
        assert_eq!(discoveries.len(), 1);
        (result, discoveries.into_iter().next().unwrap())
    }

    let (_, empty) = selection_for(
        ToolRegistry::default(),
        vec![],
        "nothing",
        ToolDiscoveryLimits::new(),
        capabilities(8, false),
    )
    .await;
    assert!(matches!(
        empty,
        RunEvent::ToolDiscoveryCompleted {
            selection: ToolDiscoverySelection::EmptyCatalog,
            candidate_count: 0,
            selected_count: 0,
            selected_schema_bytes: 2,
            catalog_exceeded_budget: false,
            ..
        }
    ));

    let (full_registry, _) = registry(1);
    let (_, full) = selection_for(
        full_registry,
        vec!["catalog.tool.000".into()],
        "anything",
        ToolDiscoveryLimits::new(),
        capabilities(8, false),
    )
    .await;
    assert!(matches!(
        full,
        RunEvent::ToolDiscoveryCompleted {
            selection: ToolDiscoverySelection::FullCatalog,
            candidate_count: 1,
            selected_count: 1,
            catalog_exceeded_budget: false,
            ..
        }
    ));

    let (no_capacity_registry, _) = registry(1);
    let (_, no_capacity) = selection_for(
        no_capacity_registry,
        vec!["catalog.tool.000".into()],
        "catalog.tool.000",
        ToolDiscoveryLimits::new(),
        capabilities(0, false),
    )
    .await;
    assert!(matches!(
        no_capacity,
        RunEvent::ToolDiscoveryCompleted {
            selection: ToolDiscoverySelection::NoCapacity,
            effective_tool_count_budget: 0,
            selected_count: 0,
            ..
        }
    ));

    let mut hot_registry = ToolRegistry::default();
    let hot = Arc::new(CountingTool::new("always.hot"));
    let deferred = Arc::new(CountingTool::new("deferred.weather"));
    hot_registry.register(hot).unwrap();
    hot_registry
        .register_with_discovery(deferred, ToolDiscoveryMetadata::deferred())
        .unwrap();
    let (_, hot_only) = selection_for(
        hot_registry,
        vec!["always.hot".into(), "deferred.weather".into()],
        "unmatched query",
        ToolDiscoveryLimits::new().with_max_tools(1),
        capabilities(8, false),
    )
    .await;
    assert!(matches!(
        hot_only,
        RunEvent::ToolDiscoveryCompleted {
            selection: ToolDiscoverySelection::HotOnly,
            selected_count: 1,
            deferred_candidate_count: 1,
            catalog_exceeded_budget: true,
            ..
        }
    ));

    let (exact_registry, _) = registry(30);
    let (_, exact) = selection_for(
        exact_registry,
        (0..30)
            .map(|index| format!("catalog.tool.{index:03}"))
            .collect(),
        "catalog.tool.017",
        ToolDiscoveryLimits::new().with_max_tools(2),
        capabilities(8, false),
    )
    .await;
    assert!(matches!(
        exact,
        RunEvent::ToolDiscoveryCompleted {
            selection: ToolDiscoverySelection::Exact,
            selected_count: 1,
            expansion_count: 0,
            ..
        }
    ));

    let (lexical_registry, _) = registry(30);
    let (_, lexical) = selection_for(
        lexical_registry,
        (0..30)
            .map(|index| format!("catalog.tool.{index:03}"))
            .collect(),
        "please use catalog tool 017 now",
        ToolDiscoveryLimits::new().with_max_tools(2),
        capabilities(8, false),
    )
    .await;
    assert!(matches!(
        lexical,
        RunEvent::ToolDiscoveryCompleted {
            selection: ToolDiscoverySelection::LexicalConfident,
            selected_count: 1,
            ..
        }
    ));

    let mut expanded_registry = ToolRegistry::default();
    for id in ["weather.one", "weather.two"] {
        expanded_registry
            .register_with_discovery(
                Arc::new(CountingTool::new(id)),
                ToolDiscoveryMetadata::deferred().with_aliases(["weather-forecast"]),
            )
            .unwrap();
    }
    expanded_registry
        .register_with_discovery(
            Arc::new(CountingTool::new("calendar.only")),
            ToolDiscoveryMetadata::deferred(),
        )
        .unwrap();
    let (_, expanded) = selection_for(
        expanded_registry,
        vec![
            "weather.one".into(),
            "weather.two".into(),
            "calendar.only".into(),
        ],
        "weather",
        ToolDiscoveryLimits::new()
            .with_max_tools(2)
            .with_max_expansion_tools(2),
        capabilities(8, false),
    )
    .await;
    assert!(
        matches!(
            &expanded,
            RunEvent::ToolDiscoveryCompleted {
                selection: ToolDiscoverySelection::LexicalExpanded,
                selected_count: 2,
                expansion_count: 2,
                expansion_limit: 2,
                ..
            }
        ),
        "{expanded:?}"
    );

    let (no_match_registry, _) = registry(30);
    let (_, no_match) = selection_for(
        no_match_registry,
        (0..30)
            .map(|index| format!("catalog.tool.{index:03}"))
            .collect(),
        "unrelated gibberish",
        ToolDiscoveryLimits::new().with_max_tools(2),
        capabilities(8, false),
    )
    .await;
    assert!(matches!(
        no_match,
        RunEvent::ToolDiscoveryCompleted {
            selection: ToolDiscoverySelection::NoMatch,
            selected_count: 0,
            catalog_exceeded_budget: true,
            ..
        }
    ));
}

#[tokio::test]
async fn schema_budget_limit_is_terminal_and_duration_reconciles_for_every_strategy() {
    for strategy in [
        RunStrategy::Direct,
        RunStrategy::Adaptive,
        RunStrategy::DeclarativePlan,
    ] {
        let provider = Arc::new(
            MockModelProvider::scripted([final_response("unused")])
                .with_capabilities(capabilities(8, true)),
        );
        let events = Arc::new(InMemoryEventSink::default());
        let tool = Arc::new(CountingTool::new("mandatory.hot"));
        let mut registry = ToolRegistry::default();
        registry.register(tool.clone()).unwrap();
        let mut agent = AgentDefinition::new("schema-limit", "Schema limit", "1", "mock-model");
        agent.tool_allowlist = vec!["mandatory.hot".into()];
        let result = AgentRunner::builder(provider.clone())
            .tools(registry)
            .event_sink(events.clone())
            .discovery_limits(
                ToolDiscoveryLimits::new()
                    .with_max_tools(8)
                    .with_max_tool_schema_bytes(2),
            )
            .build()
            .run_with_strategy(RunRequest::new(agent, "read"), strategy)
            .await
            .unwrap();

        assert_eq!(result.status, RunStatus::LimitReached);
        assert!(result.final_output.is_none());
        assert!(result.tool_calls.is_empty());
        assert!(result.policy_decisions.is_empty());
        assert!(result.approvals.is_empty());
        assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
        assert!(provider.requests().is_empty());

        let records = events.events();
        assert!(matches!(records[0].event, RunEvent::Started { .. }));
        let discovery_index = records
            .iter()
            .position(|record| {
                matches!(
                    record.event,
                    RunEvent::ToolDiscoveryCompleted {
                        outcome: ToolDiscoveryOutcome::LimitReached,
                        selection: ToolDiscoverySelection::SchemaByteLimit,
                        effective_schema_byte_budget: 2,
                        selected_count: 1,
                        catalog_exceeded_budget: true,
                        ..
                    }
                )
            })
            .expect("schema limit discovery event");
        assert_eq!(discovery_index, 1);
        let usage = records
            .iter()
            .find_map(|record| match record.event {
                RunEvent::StrategyUsage {
                    model_calls,
                    tool_calls,
                    duration_ms,
                    ..
                } => Some((model_calls, tool_calls, duration_ms)),
                _ => None,
            })
            .unwrap();
        assert_eq!(usage, (0, 0, result.duration_ms));
        assert!(matches!(
            records.last().unwrap().event,
            RunEvent::Completed {
                status: RunStatus::LimitReached
            }
        ));
    }
}

#[tokio::test]
async fn empty_and_no_capacity_scopes_emit_once_for_every_attempted_caller() {
    for no_capacity in [false, true] {
        for strategy in [
            RunStrategy::Direct,
            RunStrategy::Adaptive,
            RunStrategy::DeclarativePlan,
        ] {
            let responses = if strategy == RunStrategy::DeclarativePlan {
                vec![final_response("invalid"), final_response("still invalid")]
            } else {
                vec![final_response("done")]
            };
            let max_tools = if no_capacity { 0 } else { 8 };
            let provider = Arc::new(
                MockModelProvider::scripted(responses)
                    .with_capabilities(capabilities(max_tools, true)),
            );
            let events = Arc::new(InMemoryEventSink::default());
            let mut registry = ToolRegistry::default();
            let mut agent = AgentDefinition::new("empty", "Empty", "1", "mock-model");
            if no_capacity {
                registry
                    .register_with_discovery(
                        Arc::new(CountingTool::new("capacity.tool")),
                        ToolDiscoveryMetadata::deferred(),
                    )
                    .unwrap();
                agent.tool_allowlist = vec!["capacity.tool".into()];
            }
            let result = AgentRunner::builder(provider)
                .tools(registry)
                .event_sink(events.clone())
                .build()
                .run_with_strategy(RunRequest::new(agent, "answer"), strategy)
                .await
                .unwrap();
            if strategy == RunStrategy::DeclarativePlan {
                assert_eq!(result.status, RunStatus::Failed);
            } else {
                assert_eq!(result.status, RunStatus::Completed);
            }
            let discoveries = events
                .events()
                .into_iter()
                .filter_map(|record| match record.event {
                    RunEvent::ToolDiscoveryCompleted {
                        caller,
                        outcome,
                        selection,
                        ..
                    } => Some((caller, outcome, selection)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let expected_callers = if strategy == RunStrategy::Direct {
                vec![ToolCaller::Direct]
            } else {
                vec![ToolCaller::DeclarativePlan, ToolCaller::Direct]
            };
            assert_eq!(
                discoveries
                    .iter()
                    .map(|(caller, _, _)| *caller)
                    .collect::<Vec<_>>(),
                expected_callers
            );
            let expected_selection = if no_capacity {
                ToolDiscoverySelection::NoCapacity
            } else {
                ToolDiscoverySelection::EmptyCatalog
            };
            assert!(discoveries.iter().all(|(_, outcome, selection)| {
                *outcome == ToolDiscoveryOutcome::Selected && *selection == expected_selection
            }));
        }
    }
}

#[tokio::test]
async fn adaptive_capability_fallback_matches_forced_direct_discovery() {
    let count = 100;
    let target = "catalog.tool.042";
    let direct_provider = Arc::new(
        MockModelProvider::scripted([final_response("direct")])
            .with_capabilities(capabilities(2, false)),
    );
    let adaptive_provider = Arc::new(
        MockModelProvider::scripted([final_response("adaptive")])
            .with_capabilities(capabilities(2, false)),
    );
    let (direct_registry, _) = registry(count);
    let (adaptive_registry, _) = registry(count);
    AgentRunner::builder(direct_provider.clone())
        .tools(direct_registry)
        .build()
        .run_with_strategy(request(count, target), RunStrategy::Direct)
        .await
        .unwrap();
    AgentRunner::builder(adaptive_provider.clone())
        .tools(adaptive_registry)
        .build()
        .run(request(count, target))
        .await
        .unwrap();

    let direct_ids = direct_provider.requests()[0]
        .tools
        .iter()
        .map(|tool| tool.id.clone())
        .collect::<Vec<_>>();
    let adaptive_ids = adaptive_provider.requests()[0]
        .tools
        .iter()
        .map(|tool| tool.id.clone())
        .collect::<Vec<_>>();
    assert_eq!(direct_ids, adaptive_ids);
    assert_eq!(direct_ids, vec![target]);
}

#[tokio::test]
async fn forced_direct_and_declarative_use_the_same_selected_definition() {
    let count = 100;
    let target = "catalog.tool.042";
    let direct_provider = Arc::new(
        MockModelProvider::scripted([final_response("direct")])
            .with_capabilities(capabilities(1, true)),
    );
    let plan = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "selected", "tool_id": target, "arguments": {}}]}
    })
    .to_string();
    let declarative_provider = Arc::new(
        MockModelProvider::scripted([final_response(plan), final_response("declarative")])
            .with_capabilities(capabilities(1, true)),
    );
    let (direct_registry, _) = registry(count);
    let (declarative_registry, tools) = registry(count);
    AgentRunner::builder(direct_provider.clone())
        .tools(direct_registry)
        .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(1))
        .build()
        .run_with_strategy(request(count, target), RunStrategy::Direct)
        .await
        .unwrap();
    AgentRunner::builder(declarative_provider.clone())
        .tools(declarative_registry)
        .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(1))
        .build()
        .run_with_strategy(request(count, target), RunStrategy::DeclarativePlan)
        .await
        .unwrap();

    let direct_tools = direct_provider.requests()[0].tools.clone();
    let declarative_requests = declarative_provider.requests();
    assert_eq!(direct_tools, declarative_requests[0].tools);
    assert!(declarative_requests
        .iter()
        .all(|request| request.tools == direct_tools));
    assert!(Arc::ptr_eq(
        declarative_requests[0].prepared_tools.as_ref().unwrap(),
        declarative_requests[1].prepared_tools.as_ref().unwrap()
    ));
    assert_eq!(direct_tools.len(), 1);
    assert_eq!(direct_tools[0].id, target);
    assert_eq!(tools[42].calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn direct_and_adaptive_preflight_rejects_before_large_catalog_discovery() {
    let count = 1_000;
    let target = "catalog.tool.733";
    let provider = Arc::new(ObservedProvider {
        capabilities: capabilities(1, true),
        capability_calls: AtomicU32::new(0),
        completion_calls: AtomicU32::new(0),
    });
    let (registry, tools) = registry(count);
    let events = Arc::new(InMemoryEventSink::default());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry)
        .event_sink(events.clone())
        .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(1))
        .build();

    for strategy in [
        RunStrategy::Direct,
        RunStrategy::Adaptive,
        RunStrategy::DeclarativePlan,
    ] {
        let empty = request(count, "");
        assert!(matches!(
            runner.run_with_strategy(empty, strategy).await,
            Err(HarnessError::InvalidRequest(_))
        ));

        let mut oversized = request(count, target);
        oversized.agent.limits.max_input_bytes = 1;
        assert!(matches!(
            runner.run_with_strategy(oversized, strategy).await,
            Err(HarnessError::InvalidRequest(_))
        ));

        let cancelled = request(count, target);
        cancelled.cancellation.cancel();
        let cancelled = runner.run_with_strategy(cancelled, strategy).await.unwrap();
        assert_eq!(cancelled.status, RunStatus::Cancelled);
        assert!(cancelled.cancelled);

        let mut expired = request(count, target);
        expired.agent.limits.max_run_duration_ms = Some(0);
        let expired = runner.run_with_strategy(expired, strategy).await.unwrap();
        assert_eq!(expired.status, RunStatus::Failed);
        assert!(expired.errors.iter().any(|error| error.code == "timed_out"));

        let mut exhausted = request(count, target);
        exhausted.agent.limits.max_model_calls = 0;
        assert!(matches!(
            runner.run_with_strategy(exhausted, strategy).await,
            Err(HarnessError::InvalidRequest(_))
        ));
    }

    assert_eq!(provider.capability_calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.completion_calls.load(Ordering::SeqCst), 0);
    assert!(events.events().is_empty());
    assert_eq!(
        tools
            .iter()
            .map(|tool| tool.calls.load(Ordering::SeqCst))
            .sum::<u32>(),
        0
    );

    let result = runner
        .run_with_strategy(request(count, target), RunStrategy::Direct)
        .await
        .unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(provider.capability_calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.completion_calls.load(Ordering::SeqCst), 1);
    assert!(events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::ToolDiscoveryCompleted {
            caller: ToolCaller::Direct,
            ..
        }
    )));
}
