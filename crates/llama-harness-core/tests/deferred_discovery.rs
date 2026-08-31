use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, tool_response, MockModelProvider, MockStep},
    AgentDefinition, AgentRunner, HarnessError, InMemoryEventSink, ModelCapabilities,
    ModelResponse, ProviderCapabilityLimits, RunEvent, RunRequest, RunStatus, RunStrategy, Tool,
    ToolCall, ToolCaller, ToolDefinition, ToolDiscoveryLimits, ToolDiscoveryMetadata, ToolRegistry,
    ToolResult,
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
        }
    }
}

#[async_trait]
impl Tool for CountingTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success(json!({"value": 42})))
    }
}

fn capabilities(max_tools: u32, planning: bool) -> ModelCapabilities {
    ModelCapabilities::new(true, false, true)
        .with_structured_plans(planning)
        .with_limits(
            ProviderCapabilityLimits::new()
                .with_max_tools(max_tools)
                .with_max_tool_schema_bytes(64 * 1024)
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
async fn known_unselected_and_unknown_calls_have_the_same_external_rejection() {
    async fn rejection(tool_id: &str) -> (String, u32) {
        let count = 30;
        let target = "catalog.tool.005";
        let provider = Arc::new(
            MockModelProvider::scripted([
                tool_response(ToolCall::new("bypass", tool_id, "{}")),
                final_response("done"),
            ])
            .with_capabilities(capabilities(1, false)),
        );
        let (registry, tools) = registry(count);
        let runner = AgentRunner::builder(provider)
            .tools(registry)
            .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(1))
            .build();
        let result = runner
            .run_with_strategy(request(count, target), RunStrategy::Direct)
            .await
            .unwrap();
        let rejection = result
            .errors
            .iter()
            .find(|error| error.message.contains("tool unavailable"))
            .expect("unavailable error")
            .message
            .clone();
        let calls = tools
            .iter()
            .map(|tool| tool.calls.load(Ordering::SeqCst))
            .sum();
        (rejection, calls)
    }

    let known = rejection("catalog.tool.006").await;
    let unknown = rejection("catalog.tool.999").await;
    assert_eq!(known, unknown);
    assert_eq!(known.1, 0);
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
    let (registry, _) = registry(30);
    let result = AgentRunner::builder(zero_provider.clone())
        .tools(registry)
        .build()
        .run_with_strategy(request(30, "catalog.tool.001"), RunStrategy::Direct)
        .await
        .unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    assert!(zero_provider.requests()[0].tools.is_empty());
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
