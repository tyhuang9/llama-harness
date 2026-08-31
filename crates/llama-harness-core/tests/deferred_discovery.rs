use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, tool_response, MockModelProvider, MockStep},
    AgentDefinition, AgentRunner, HarnessError, InMemoryEventSink, ModelCapabilities, ModelInfo,
    ModelProvider, ModelRequest, ModelResponse, ProviderCapabilityLimits, ProviderHealth, RunEvent,
    RunRequest, RunStatus, RunStrategy, Tool, ToolCall, ToolCaller, ToolDefinition,
    ToolDiscoveryLimits, ToolDiscoveryMetadata, ToolRegistry, ToolResult,
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
    }
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
    assert!(!discovery.4.contains(target));
    assert!(!discovery.4.contains("alias"));
    assert!(!discovery.4.contains("fingerprint"));
    assert!(!discovery.4.contains("cache"));
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
                if value.get("type") == Some(&Value::String("strategy_usage".into())) {
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
    let error = AgentRunner::builder(provider.clone())
        .tools(hot)
        .build()
        .run_with_strategy(hot_request, RunStrategy::Direct)
        .await
        .unwrap_err();
    assert!(matches!(error, HarnessError::ResourceLimit(_)));
    assert!(provider.requests().is_empty());

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
        let requests = provider.requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].tools.is_empty());
    }
}

#[tokio::test]
async fn exact_namespace_overflow_fails_before_provider_model_tool_or_events() {
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
        let error = AgentRunner::builder(provider.clone())
            .tools(registry)
            .event_sink(events.clone())
            .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(1))
            .build()
            .run_with_strategy(RunRequest::new(agent, "exact"), strategy)
            .await
            .unwrap_err();
        assert!(matches!(error, HarnessError::ResourceLimit(_)));
        assert!(provider.requests().is_empty());
        assert!(events.events().is_empty());
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

    for strategy in [RunStrategy::Direct, RunStrategy::Adaptive] {
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
