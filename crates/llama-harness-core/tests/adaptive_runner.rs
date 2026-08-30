use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, tool_response, MockModelProvider},
    AgentDefinition, AgentRunner, AllowAllPolicy, ApprovalHandler, ApprovalRecord, HarnessError,
    InMemoryEventSink, ModelCapabilities, PlanConcurrency, PolicyDecision, PolicyEngine,
    ProviderCapabilityLimits, RunEvent, RunRequest, RunStatus, RunStrategy, Tool, ToolCall,
    ToolCaller, ToolDefinition, ToolRegistry, ToolResult, ToolRisk,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex,
};
use tokio::sync::Barrier;
use tokio_util::sync::CancellationToken;

fn planning_capabilities(parallel: u32) -> ModelCapabilities {
    ModelCapabilities::new(true, false, true)
        .with_structured_plans(true)
        .with_parallel_tool_calls(parallel > 1)
        .with_limits(
            ProviderCapabilityLimits::new()
                .with_max_parallel_tool_calls(parallel)
                .with_max_plan_nodes(64)
                .with_max_plan_bytes(256 * 1024),
        )
}

fn request(tool_ids: &[&str]) -> RunRequest {
    let mut agent = AgentDefinition::new("adaptive", "Adaptive", "1", "mock-model");
    agent.tool_allowlist = tool_ids.iter().map(|id| (*id).to_owned()).collect();
    RunRequest::new(agent, "complete the task")
}

fn definition(id: &str, read_only: bool, parallel_safe: bool) -> ToolDefinition {
    ToolDefinition::new(id, id, "adaptive test tool", json!({"type": "object"}))
        .with_risk(if read_only {
            ToolRisk::Low
        } else {
            ToolRisk::High
        })
        .with_read_only(read_only)
        .with_idempotent(read_only)
        .with_parallel_safe(parallel_safe)
        .with_allowed_callers([ToolCaller::Direct, ToolCaller::DeclarativePlan])
}

fn registry(tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> ToolRegistry {
    let mut registry = ToolRegistry::default();
    for tool in tools {
        registry.register(tool).unwrap();
    }
    registry
}

struct FixedTool {
    definition: ToolDefinition,
    output: ToolResult,
    calls: AtomicU32,
    arguments: Mutex<Vec<Value>>,
}

impl FixedTool {
    fn new(id: &str, read_only: bool, output: ToolResult) -> Self {
        Self {
            definition: definition(id, read_only, read_only),
            output,
            calls: AtomicU32::new(0),
            arguments: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Tool for FixedTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(
        &self,
        arguments: Value,
        _: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.arguments
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(arguments);
        Ok(self.output.clone())
    }
}

struct BarrierTool {
    definition: ToolDefinition,
    barrier: Arc<Barrier>,
    active: Arc<AtomicU32>,
    maximum: Arc<AtomicU32>,
}

#[async_trait]
impl Tool for BarrierTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.maximum.fetch_max(active, Ordering::SeqCst);
        self.barrier.wait().await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(ToolResult::success(json!({"ok": true})))
    }
}

struct GrantApproval(AtomicU32);

#[async_trait]
impl ApprovalHandler for GrantApproval {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ApprovalRecord::new(
            "",
            tool.id.clone(),
            true,
            "test approval",
        ))
    }
}

struct RequireApproval;

#[async_trait]
impl PolicyEngine for RequireApproval {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(PolicyDecision::RequireApproval {
            reason: "test policy".into(),
        })
    }
}

#[tokio::test]
async fn adaptive_planner_can_choose_no_tool_direct_response() {
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(r#"{"strategy":"direct"}"#),
            final_response("done"),
        ])
        .with_capabilities(planning_capabilities(2)),
    );
    let unused = Arc::new(FixedTool::new(
        "unused",
        true,
        ToolResult::success(json!({})),
    ));
    let events = Arc::new(InMemoryEventSink::default());
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry([unused.clone() as Arc<dyn Tool>]))
        .event_sink(events.clone())
        .build();

    let result = runner.run(request(&["unused"])).await.unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.final_output.as_deref(), Some("done"));
    assert_eq!(unused.calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.requests().len(), 2);
    assert!(events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::StrategySelected {
            selected: RunStrategy::Direct,
            ..
        }
    )));
}

#[tokio::test]
async fn adaptive_downgrades_to_direct_when_provider_cannot_plan() {
    let provider = Arc::new(MockModelProvider::scripted([final_response("direct")]));
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider.clone())
        .event_sink(events.clone())
        .build()
        .run(request(&[]))
        .await
        .unwrap();

    assert_eq!(result.final_output.as_deref(), Some("direct"));
    assert_eq!(provider.requests().len(), 1);
    assert!(events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyFallback {
            reason: llama_harness_core::StrategyFallbackReason::UnsupportedCapability,
            ..
        }
    )));
}

#[tokio::test]
async fn adaptive_downgrades_when_model_budget_cannot_finalize_a_plan() {
    let provider = Arc::new(
        MockModelProvider::scripted([final_response("direct")])
            .with_capabilities(planning_capabilities(2)),
    );
    let tool = Arc::new(FixedTool::new("read", true, ToolResult::success(json!({}))));
    let mut run_request = request(&["read"]);
    run_request.agent.limits.max_model_calls = 1;
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([tool as Arc<dyn Tool>]))
        .build()
        .run(run_request)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn adaptive_direct_recovery_redacts_invalid_arguments_before_feedback() {
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(r#"{"strategy":"direct"}"#),
            tool_response(ToolCall::new(
                "invalid",
                "read",
                r#"{"secret":"must-not-return","required":4}"#,
            )),
            final_response("done"),
        ])
        .with_capabilities(planning_capabilities(2)),
    );
    let mut tool = FixedTool::new("read", true, ToolResult::success(json!({})));
    tool.definition.arguments_schema = json!({
        "type": "object",
        "required": ["required"],
        "properties": {"required": {"type": "string"}},
        "additionalProperties": false
    });
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([Arc::new(tool) as Arc<dyn Tool>]))
        .build()
        .run(request(&["read"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    let requests = provider.requests();
    assert_eq!(requests.len(), 3);
    let serialized = serde_json::to_string(&requests[2].messages).unwrap();
    assert!(!serialized.contains("must-not-return"));
    let recorded = requests[2]
        .messages
        .iter()
        .flat_map(|message| message.tool_calls.iter())
        .find(|call| call.id == "invalid")
        .unwrap();
    assert_eq!(recorded.arguments_json, "{}");
}

#[tokio::test]
async fn forced_declarative_plan_fails_closed_without_capability() {
    let provider = Arc::new(MockModelProvider::scripted([final_response("unused")]));
    let error = AgentRunner::builder(provider.clone())
        .build()
        .run_with_strategy(request(&[]), RunStrategy::DeclarativePlan)
        .await
        .unwrap_err();

    assert!(matches!(error, HarnessError::UnsupportedCapability(_)));
    assert!(provider.requests().is_empty());
}

#[tokio::test]
async fn independent_safe_nodes_execute_in_one_parallel_wave() {
    let plan = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "one", "tool_id": "read", "arguments": {}},
            {"id": "two", "tool_id": "read", "arguments": {}}
        ]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(plan.to_string()), final_response("done")])
            .with_capabilities(planning_capabilities(2)),
    );
    let barrier = Arc::new(Barrier::new(2));
    let active = Arc::new(AtomicU32::new(0));
    let maximum = Arc::new(AtomicU32::new(0));
    let tool = Arc::new(BarrierTool {
        definition: definition("read", true, true),
        barrier,
        active,
        maximum: maximum.clone(),
    });
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider)
        .tools(registry([tool as Arc<dyn Tool>]))
        .event_sink(events.clone())
        .build()
        .run(request(&["read"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(maximum.load(Ordering::SeqCst), 2);
    let waves = events
        .events()
        .iter()
        .filter_map(|record| match record.event {
            RunEvent::PlanNodeStarted { wave, .. } => Some(wave),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(waves, vec![1, 1]);
}

#[tokio::test]
async fn dependent_result_binding_is_revalidated_before_execution() {
    let plan = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "lookup", "tool_id": "lookup", "arguments": {}},
            {
                "id": "consume",
                "tool_id": "consume",
                "arguments": {"input": ""},
                "depends_on": ["lookup"],
                "result_bindings": [{
                    "target_pointer": "/input",
                    "source": {"node_id": "lookup", "output_pointer": "/value"}
                }]
            }
        ]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(plan.to_string()), final_response("done")])
            .with_capabilities(planning_capabilities(4)),
    );
    let lookup = Arc::new(FixedTool::new(
        "lookup",
        true,
        ToolResult::success(json!({"value": "bound"})),
    ));
    let mut consume_tool = FixedTool::new(
        "consume",
        true,
        ToolResult::success(json!({"accepted": true})),
    );
    consume_tool.definition.arguments_schema = json!({
        "type": "object",
        "required": ["input"],
        "properties": {"input": {"type": "string"}}
    });
    let consume = Arc::new(consume_tool);
    let result = AgentRunner::builder(provider)
        .tools(registry([
            lookup as Arc<dyn Tool>,
            consume.clone() as Arc<dyn Tool>,
        ]))
        .build()
        .run(request(&["lookup", "consume"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(
        consume
            .arguments
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_slice(),
        &[json!({"input": "bound"})]
    );
}

#[tokio::test]
async fn approval_and_mutation_nodes_are_serialized_from_safe_reads() {
    let plan = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "read-one", "tool_id": "read", "arguments": {}},
            {"id": "write", "tool_id": "write", "arguments": {}, "approval_barrier": true, "commit_boundary": true},
            {"id": "read-two", "tool_id": "read", "arguments": {}}
        ]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(plan.to_string()), final_response("done")])
            .with_capabilities(planning_capabilities(4)),
    );
    let read = Arc::new(FixedTool::new(
        "read",
        true,
        ToolResult::success(json!({"value": 1})),
    ));
    let write = Arc::new(FixedTool::new(
        "write",
        false,
        ToolResult::success(json!({"written": true})),
    ));
    let approvals = Arc::new(GrantApproval(AtomicU32::new(0)));
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider)
        .tools(registry([read as Arc<dyn Tool>, write as Arc<dyn Tool>]))
        .policy(Arc::new(AllowAllPolicy))
        .approvals(approvals.clone())
        .event_sink(events.clone())
        .build()
        .run(request(&["read", "write"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(approvals.0.load(Ordering::SeqCst), 1);
    let write_wave = events
        .events()
        .iter()
        .find_map(|record| match &record.event {
            RunEvent::PlanNodeStarted { node_id, wave, .. } if node_id == "write" => Some(*wave),
            _ => None,
        });
    assert_eq!(write_wave, Some(2));
}

#[tokio::test]
async fn malformed_plan_gets_one_repair_then_falls_back_before_effects() {
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response("not-json"),
            final_response("still-not-json"),
            final_response("fallback"),
        ])
        .with_capabilities(planning_capabilities(2)),
    );
    let tool = Arc::new(FixedTool::new("read", true, ToolResult::success(json!({}))));
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([tool.clone() as Arc<dyn Tool>]))
        .event_sink(events.clone())
        .build()
        .run(request(&["read"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.final_output.as_deref(), Some("fallback"));
    assert_eq!(provider.requests().len(), 3);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    assert!(events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyFallback {
            reason: llama_harness_core::StrategyFallbackReason::InvalidPlan,
            ..
        }
    )));
}

#[tokio::test]
async fn recovery_reuses_committed_mutation_instead_of_executing_it_twice() {
    let initial = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "write", "tool_id": "write", "arguments": {}},
            {"id": "fail", "tool_id": "fail", "arguments": {}, "depends_on": ["write"]}
        ]}
    });
    let recovery = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "write-again", "tool_id": "write", "arguments": {}}
        ]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(initial.to_string()),
            final_response(recovery.to_string()),
            final_response("done"),
        ])
        .with_capabilities(planning_capabilities(2)),
    );
    let write = Arc::new(FixedTool::new(
        "write",
        false,
        ToolResult::success(json!({"written": true})),
    ));
    let fail = Arc::new(FixedTool::new(
        "fail",
        true,
        ToolResult::failure("expected failure"),
    ));
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider)
        .tools(registry([
            write.clone() as Arc<dyn Tool>,
            fail as Arc<dyn Tool>,
        ]))
        .policy(Arc::new(AllowAllPolicy))
        .event_sink(events.clone())
        .build()
        .run(request(&["write", "fail"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(write.calls.load(Ordering::SeqCst), 1);
    assert!(events
        .events()
        .iter()
        .any(|record| matches!(record.event, RunEvent::ToolEffectReused { .. })));
}

#[tokio::test]
async fn provider_parallel_limit_bounds_each_deterministic_wave() {
    let nodes = (0..6)
        .map(|index| {
            json!({
                "id": format!("node-{index}"),
                "tool_id": "read",
                "arguments": {"index": index}
            })
        })
        .collect::<Vec<_>>();
    let envelope = json!({"strategy": "declarative_plan", "plan": {"nodes": nodes}});
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(envelope.to_string()), final_response("done")])
            .with_capabilities(planning_capabilities(3)),
    );
    let maximum = Arc::new(AtomicU32::new(0));
    let tool = Arc::new(BarrierTool {
        definition: definition("read", true, true),
        barrier: Arc::new(Barrier::new(3)),
        active: Arc::new(AtomicU32::new(0)),
        maximum: maximum.clone(),
    });
    let result = AgentRunner::builder(provider)
        .tools(registry([tool as Arc<dyn Tool>]))
        .build()
        .run(request(&["read"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(maximum.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn cancellation_stops_before_planner_or_tool_side_effects() {
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(r#"{"strategy":"direct"}"#)])
            .with_capabilities(planning_capabilities(2)),
    );
    let tool = Arc::new(FixedTool::new("read", true, ToolResult::success(json!({}))));
    let run_request = request(&["read"]);
    run_request.cancellation.cancel();
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([tool.clone() as Arc<dyn Tool>]))
        .build()
        .run(run_request)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Cancelled);
    assert!(result.cancelled);
    assert!(provider.requests().is_empty());
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn plan_concurrency_contract_remains_explicitly_serializable() {
    assert_eq!(
        serde_json::to_value(PlanConcurrency::Serial).unwrap(),
        json!("serial")
    );
}

#[tokio::test]
async fn policy_required_approval_is_honored_during_whole_plan_preflight() {
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "read", "tool_id": "read", "arguments": {}}]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(envelope.to_string()), final_response("done")])
            .with_capabilities(planning_capabilities(1)),
    );
    let tool = Arc::new(FixedTool::new("read", true, ToolResult::success(json!({}))));
    let approvals = Arc::new(GrantApproval(AtomicU32::new(0)));
    let result = AgentRunner::builder(provider)
        .tools(registry([tool as Arc<dyn Tool>]))
        .policy(Arc::new(RequireApproval))
        .approvals(approvals.clone())
        .build()
        .run(request(&["read"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(approvals.0.load(Ordering::SeqCst), 1);
}
