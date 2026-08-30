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
use std::time::Duration;
use tokio::sync::{Barrier, Semaphore};
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

struct CapturingApproval(Mutex<Vec<Value>>);

#[async_trait]
impl ApprovalHandler for CapturingApproval {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        arguments: &Value,
        _: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(arguments.clone());
        Ok(ApprovalRecord::new(
            "",
            tool.id.clone(),
            true,
            "captured exact arguments",
        ))
    }
}

struct ErrorTool {
    definition: ToolDefinition,
    error: HarnessError,
    calls: AtomicU32,
}

#[async_trait]
impl Tool for ErrorTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(self.error.clone())
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

struct DenyToolPolicy(&'static str);

#[async_trait]
impl PolicyEngine for DenyToolPolicy {
    async fn decide(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        if tool.id == self.0 {
            Ok(PolicyDecision::Deny {
                reason: "security test denial".into(),
            })
        } else {
            Ok(PolicyDecision::Allow {
                reason: "security test allow".into(),
            })
        }
    }
}

struct HeldTool {
    definition: ToolDefinition,
    entered: Arc<Semaphore>,
    release: Arc<Semaphore>,
    active: Arc<AtomicU32>,
    maximum: Arc<AtomicU32>,
}

#[async_trait]
impl Tool for HeldTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.maximum.fetch_max(active, Ordering::SeqCst);
        self.entered.add_permits(1);
        let permit = self
            .release
            .acquire()
            .await
            .map_err(|_| HarnessError::Tool("test release semaphore closed".into()))?;
        permit.forget();
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(ToolResult::success(json!({"ok": true})))
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
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([tool as Arc<dyn Tool>]))
        .event_sink(events.clone())
        .build()
        .run(request(&["read"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(maximum.load(Ordering::SeqCst), 2);
    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    let final_messages = &requests[1].messages;
    let planned_calls = final_messages
        .iter()
        .find(|message| message.tool_calls.len() == 2)
        .expect("the final request must retain the complete planned call batch");
    let planned_call_ids = planned_calls
        .tool_calls
        .iter()
        .map(|call| call.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(planned_call_ids, vec!["plan-1-node-1", "plan-1-node-2"]);
    let correlated_results = final_messages
        .iter()
        .filter_map(|message| {
            message.tool_call_id.as_deref().map(|call_id| {
                (
                    call_id,
                    serde_json::from_str::<ToolResult>(&message.content),
                )
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(correlated_results.len(), 2);
    assert!(correlated_results
        .iter()
        .all(|(call_id, result)| planned_call_ids.contains(call_id)
            && result.as_ref().is_ok_and(|result| result.ok)));
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
            RunEvent::PlanNodeStarted { node_id, wave, .. } if node_id == "plan-1-node-2" => {
                Some(*wave)
            }
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
    let mut run_request = request(&["write", "fail"]);
    run_request.agent.limits.max_identical_tool_calls = 1;
    let result = AgentRunner::builder(provider)
        .tools(registry([
            write.clone() as Arc<dyn Tool>,
            fail as Arc<dyn Tool>,
        ]))
        .policy(Arc::new(AllowAllPolicy))
        .event_sink(events.clone())
        .build()
        .run(run_request)
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
async fn late_effect_reuse_is_charged_before_a_later_mutation() {
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "first-write", "tool_id": "write", "arguments": {"input": "same"}},
            {"id": "lookup", "tool_id": "lookup", "arguments": {}},
            {
                "id": "reused-write",
                "tool_id": "write",
                "arguments": {"input": "placeholder"},
                "depends_on": ["lookup"],
                "result_bindings": [{
                    "target_pointer": "/input",
                    "source": {"node_id": "lookup", "output_pointer": "/value"}
                }]
            },
            {
                "id": "later-write",
                "tool_id": "write",
                "arguments": {"input": "distinct"},
                "depends_on": ["reused-write"]
            }
        ]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(envelope.to_string())])
            .with_capabilities(planning_capabilities(1)),
    );
    let lookup_result = ToolResult::success(json!({"value": "same"}));
    let write_result = ToolResult::success(json!({"written": true}));
    let lookup = Arc::new(FixedTool::new("lookup", true, lookup_result.clone()));
    let write = Arc::new(FixedTool::new("write", false, write_result.clone()));
    let serialized_result_bytes =
        |result: &ToolResult| u64::try_from(serde_json::to_vec(result).unwrap().len()).unwrap();
    let exact_entry_bytes = |call_id: &str, tool_id: &str, arguments: &str, result: &ToolResult| {
        (call_id.len() as u64) * 2
            + tool_id.len() as u64
            + arguments.len() as u64
            + serialized_result_bytes(result)
    };
    let max_result_bytes =
        serialized_result_bytes(&lookup_result).max(serialized_result_bytes(&write_result));
    let lookup_entry = exact_entry_bytes("plan-1-node-2", "lookup", "{}", &lookup_result);
    let write_entry = exact_entry_bytes(
        "plan-1-node-1",
        "write",
        r#"{"input":"same"}"#,
        &write_result,
    );
    let later_worst_case = ("plan-1-node-4".len() as u64) * 2
        + "write".len() as u64
        + r#"{"input":"distinct"}"#.len() as u64
        + max_result_bytes;

    let mut run_request = request(&["lookup", "write"]);
    run_request.agent.limits.max_tool_result_bytes = max_result_bytes;
    run_request.agent.limits.max_transcript_bytes =
        run_request.input.len() as u64 + lookup_entry + write_entry + later_worst_case;
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([
            lookup.clone() as Arc<dyn Tool>,
            write.clone() as Arc<dyn Tool>,
        ]))
        .policy(Arc::new(AllowAllPolicy))
        .event_sink(events.clone())
        .build()
        .run(run_request)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::LimitReached);
    assert_eq!(lookup.calls.load(Ordering::SeqCst), 1);
    assert_eq!(write.calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.requests().len(), 1);
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

#[tokio::test]
async fn failed_mutation_is_uncertain_and_never_enters_recovery() {
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "write", "tool_id": "write", "arguments": {}}]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(envelope.to_string())])
            .with_capabilities(planning_capabilities(1)),
    );
    let write = Arc::new(FixedTool::new(
        "write",
        false,
        ToolResult::failure("outcome unknown"),
    ));
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([write.clone() as Arc<dyn Tool>]))
        .policy(Arc::new(AllowAllPolicy))
        .event_sink(events.clone())
        .build()
        .run(request(&["write"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Failed);
    assert_eq!(write.calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.requests().len(), 1);
    assert!(!events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyFallback {
            reason: llama_harness_core::StrategyFallbackReason::ExecutionRecovery,
            ..
        }
    )));
}

#[tokio::test]
async fn invalid_mutation_output_is_uncertain_and_never_enters_recovery() {
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "write", "tool_id": "write", "arguments": {}}]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(envelope.to_string())])
            .with_capabilities(planning_capabilities(1)),
    );
    let mut write_tool = FixedTool::new(
        "write",
        false,
        ToolResult::success(json!({"accepted": "not-a-boolean"})),
    );
    write_tool.definition = write_tool.definition.with_output_schema(json!({
        "type": "object",
        "required": ["accepted"],
        "properties": {"accepted": {"type": "boolean"}}
    }));
    let write = Arc::new(write_tool);
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([write.clone() as Arc<dyn Tool>]))
        .policy(Arc::new(AllowAllPolicy))
        .build()
        .run(request(&["write"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Failed);
    assert_eq!(write.calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.requests().len(), 1);
    assert!(result.errors.iter().any(|error| error.code == "tool_error"));
}

#[tokio::test]
async fn post_dispatch_cancellation_is_terminal_and_never_replayed() {
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "write", "tool_id": "write", "arguments": {}}]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(envelope.to_string())])
            .with_capabilities(planning_capabilities(1)),
    );
    let write = Arc::new(ErrorTool {
        definition: definition("write", false, false),
        error: HarnessError::Cancelled,
        calls: AtomicU32::new(0),
    });
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([write.clone() as Arc<dyn Tool>]))
        .policy(Arc::new(AllowAllPolicy))
        .build()
        .run(request(&["write"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Cancelled);
    assert_eq!(write.calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn bound_approval_receives_exact_executed_arguments() {
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "lookup", "tool_id": "lookup", "arguments": {}},
            {
                "id": "write",
                "tool_id": "write",
                "arguments": {"input": "placeholder"},
                "depends_on": ["lookup"],
                "result_bindings": [{
                    "target_pointer": "/input",
                    "source": {"node_id": "lookup", "output_pointer": "/value"}
                }],
                "approval_barrier": true
            }
        ]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(envelope.to_string()), final_response("done")])
            .with_capabilities(planning_capabilities(2)),
    );
    let lookup = Arc::new(FixedTool::new(
        "lookup",
        true,
        ToolResult::success(json!({"value": "bound"})),
    ));
    let mut write_tool = FixedTool::new(
        "write",
        false,
        ToolResult::success(json!({"written": true})),
    );
    write_tool.definition.arguments_schema = json!({
        "type": "object",
        "required": ["input"],
        "properties": {"input": {"type": "string"}}
    });
    let write = Arc::new(write_tool);
    let approvals = Arc::new(CapturingApproval(Mutex::new(Vec::new())));
    let result = AgentRunner::builder(provider)
        .tools(registry([
            lookup as Arc<dyn Tool>,
            write.clone() as Arc<dyn Tool>,
        ]))
        .policy(Arc::new(AllowAllPolicy))
        .approvals(approvals.clone())
        .build()
        .run(request(&["lookup", "write"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(
        approvals
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_slice(),
        &[json!({"input": "bound"})]
    );
    assert_eq!(
        write
            .arguments
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_slice(),
        &[json!({"input": "bound"})]
    );
}

#[tokio::test]
async fn bound_calls_enforce_repeat_limit_on_final_signature() {
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "lookup", "tool_id": "lookup", "arguments": {}},
            {
                "id": "consume-one", "tool_id": "consume", "arguments": {"input": "one"},
                "depends_on": ["lookup"],
                "result_bindings": [{"target_pointer": "/input", "source": {"node_id": "lookup", "output_pointer": "/value"}}]
            },
            {
                "id": "consume-two", "tool_id": "consume", "arguments": {"input": "two"},
                "depends_on": ["lookup"],
                "result_bindings": [{"target_pointer": "/input", "source": {"node_id": "lookup", "output_pointer": "/value"}}]
            }
        ]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(envelope.to_string())])
            .with_capabilities(planning_capabilities(2)),
    );
    let lookup = Arc::new(FixedTool::new(
        "lookup",
        true,
        ToolResult::success(json!({"value": "same"})),
    ));
    let consume = Arc::new(FixedTool::new(
        "consume",
        true,
        ToolResult::success(json!({"ok": true})),
    ));
    let mut run_request = request(&["lookup", "consume"]);
    run_request.agent.limits.max_identical_tool_calls = 1;
    let result = AgentRunner::builder(provider)
        .tools(registry([
            lookup as Arc<dyn Tool>,
            consume.clone() as Arc<dyn Tool>,
        ]))
        .build()
        .run(run_request)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::LimitReached);
    assert!(result.repeated_tool_call_limit_reached);
    assert_eq!(consume.calls.load(Ordering::SeqCst), 0);
    assert_eq!(result.tool_calls[1].arguments_json, r#"{"input":"same"}"#);
    assert_eq!(result.tool_calls[2].arguments_json, r#"{"input":"same"}"#);
}

#[tokio::test]
async fn plan_transcript_limit_is_terminal_before_final_model_call() {
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "read", "tool_id": "read", "arguments": {}}]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(envelope.to_string()),
            final_response("must not run"),
        ])
        .with_capabilities(planning_capabilities(1)),
    );
    let read = Arc::new(FixedTool::new(
        "read",
        true,
        ToolResult::success(json!({"large": "payload"})),
    ));
    let mut run_request = request(&["read"]);
    run_request.agent.limits.max_transcript_bytes = run_request.input.len() as u64 + 1;
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([read as Arc<dyn Tool>]))
        .build()
        .run(run_request)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::LimitReached);
    assert_eq!(provider.requests().len(), 1);
    assert!(result.final_output.is_none());
}

#[tokio::test]
async fn shared_concurrency_key_forces_deterministic_waves() {
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "one", "tool_id": "read", "arguments": {"index": 1}},
            {"id": "two", "tool_id": "read", "arguments": {"index": 2}}
        ]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(envelope.to_string()), final_response("done")])
            .with_capabilities(planning_capabilities(2)),
    );
    let mut read_tool = FixedTool::new("read", true, ToolResult::success(json!({"ok": true})));
    read_tool.definition = read_tool.definition.with_concurrency_key("shared");
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider)
        .tools(registry([Arc::new(read_tool) as Arc<dyn Tool>]))
        .event_sink(events.clone())
        .build()
        .run(request(&["read"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    let waves = events
        .events()
        .iter()
        .filter_map(|record| match record.event {
            RunEvent::PlanNodeStarted { wave, .. } => Some(wave),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(waves, vec![1, 2]);
}

#[tokio::test]
async fn partial_parallel_wave_records_success_for_recovery() {
    let initial = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "kept", "tool_id": "good", "arguments": {}},
            {"id": "failed", "tool_id": "fail", "arguments": {}}
        ]}
    });
    let recovery = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "finish", "tool_id": "good", "arguments": {"recovery": true}}]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(initial.to_string()),
            final_response(recovery.to_string()),
            final_response("done"),
        ])
        .with_capabilities(planning_capabilities(2)),
    );
    let good = Arc::new(FixedTool::new(
        "good",
        true,
        ToolResult::success(json!({"value": "kept"})),
    ));
    let fail = Arc::new(FixedTool::new(
        "fail",
        true,
        ToolResult::failure("expected"),
    ));
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([
            good.clone() as Arc<dyn Tool>,
            fail as Arc<dyn Tool>,
        ]))
        .build()
        .run(request(&["good", "fail"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(good.calls.load(Ordering::SeqCst), 2);
    let recovery_request = serde_json::to_string(&provider.requests()[1].messages).unwrap();
    assert!(recovery_request.contains("kept"));
}

#[tokio::test]
async fn invalid_plan_repair_does_not_consume_execution_recovery() {
    let initial = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "failed", "tool_id": "fail", "arguments": {}}]}
    });
    let recovery = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "recovered", "tool_id": "good", "arguments": {}}]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response("invalid"),
            final_response(initial.to_string()),
            final_response(recovery.to_string()),
            final_response("done"),
        ])
        .with_capabilities(planning_capabilities(1)),
    );
    let fail = Arc::new(FixedTool::new(
        "fail",
        true,
        ToolResult::failure("expected"),
    ));
    let good = Arc::new(FixedTool::new(
        "good",
        true,
        ToolResult::success(json!({"ok": true})),
    ));
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([fail as Arc<dyn Tool>, good as Arc<dyn Tool>]))
        .build()
        .run(request(&["fail", "good"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(provider.requests().len(), 4);
}

#[tokio::test]
async fn forced_direct_emits_selection_and_usage_metadata() {
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
        "done",
    )])))
    .event_sink(events.clone())
    .build()
    .run_with_strategy(request(&[]), RunStrategy::Direct)
    .await
    .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert!(events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::StrategySelected {
            requested: RunStrategy::Direct,
            selected: RunStrategy::Direct,
            reason: llama_harness_core::StrategySelectionReason::Forced,
        }
    )));
    assert!(events.events().iter().any(|record| matches!(
        record.event,
        RunEvent::StrategyUsage {
            strategy: RunStrategy::Direct,
            ..
        }
    )));
}

#[tokio::test]
async fn binding_copy_budget_rejects_amplification_before_invocation() {
    let bindings = (0..32)
        .map(|index| {
            json!({
                "target_pointer": format!("/values/{index}"),
                "source": {"node_id": "lookup", "output_pointer": "/large"}
            })
        })
        .collect::<Vec<_>>();
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "lookup", "tool_id": "lookup", "arguments": {}},
            {
                "id": "amplify",
                "tool_id": "consume",
                "arguments": {"values": vec![Value::Null; 32]},
                "depends_on": ["lookup"],
                "result_bindings": bindings
            }
        ]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(envelope.to_string())])
            .with_capabilities(planning_capabilities(1)),
    );
    let lookup = Arc::new(FixedTool::new(
        "lookup",
        true,
        ToolResult::success(json!({"large": "x".repeat(4096)})),
    ));
    let consume = Arc::new(FixedTool::new(
        "consume",
        true,
        ToolResult::success(json!({"ok": true})),
    ));
    let mut run_request = request(&["lookup", "consume"]);
    run_request.agent.limits.max_tool_arguments_bytes = 8 * 1024;
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([
            lookup.clone() as Arc<dyn Tool>,
            consume.clone() as Arc<dyn Tool>,
        ]))
        .build()
        .run(run_request)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::LimitReached);
    assert_eq!(lookup.calls.load(Ordering::SeqCst), 1);
    assert_eq!(consume.calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn cumulative_plan_result_budget_stops_before_later_wave() {
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "one", "tool_id": "read", "arguments": {"index": 1}},
            {"id": "two", "tool_id": "read", "arguments": {"index": 2}, "depends_on": ["one"]},
            {"id": "three", "tool_id": "read", "arguments": {"index": 3}, "depends_on": ["two"]}
        ]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(envelope.to_string())])
            .with_capabilities(planning_capabilities(1)),
    );
    let read = Arc::new(FixedTool::new(
        "read",
        true,
        ToolResult::success(json!({"payload": "x".repeat(700)})),
    ));
    let mut run_request = request(&["read"]);
    run_request.agent.limits.max_tool_result_bytes = 1024;
    run_request.agent.limits.max_transcript_bytes = run_request.input.len() as u64 + 2300;
    let result = AgentRunner::builder(provider.clone())
        .tools(registry([read.clone() as Arc<dyn Tool>]))
        .build()
        .run(run_request)
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::LimitReached);
    assert_eq!(read.calls.load(Ordering::SeqCst), 2);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn mutation_gate_denial_prevents_every_mutation_dispatch() {
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "early-secret", "tool_id": "early", "arguments": {}},
            {"id": "lookup-secret", "tool_id": "lookup", "arguments": {}},
            {
                "id": "late-secret",
                "tool_id": "late",
                "arguments": {"input": "placeholder"},
                "depends_on": ["lookup-secret"],
                "result_bindings": [{
                    "target_pointer": "/input",
                    "source": {"node_id": "lookup-secret", "output_pointer": "/value"}
                }]
            }
        ]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(envelope.to_string()),
            final_response("invalid recovery"),
        ])
        .with_capabilities(planning_capabilities(1)),
    );
    let early = Arc::new(FixedTool::new(
        "early",
        false,
        ToolResult::success(json!({"ok": true})),
    ));
    let lookup = Arc::new(FixedTool::new(
        "lookup",
        true,
        ToolResult::success(json!({"value": "resolved"})),
    ));
    let late = Arc::new(FixedTool::new(
        "late",
        false,
        ToolResult::success(json!({"ok": true})),
    ));
    let result = AgentRunner::builder(provider)
        .tools(registry([
            early.clone() as Arc<dyn Tool>,
            lookup.clone() as Arc<dyn Tool>,
            late.clone() as Arc<dyn Tool>,
        ]))
        .policy(Arc::new(DenyToolPolicy("late")))
        .build()
        .run_with_strategy(
            request(&["early", "lookup", "late"]),
            RunStrategy::DeclarativePlan,
        )
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Failed);
    assert_eq!(lookup.calls.load(Ordering::SeqCst), 1);
    assert_eq!(early.calls.load(Ordering::SeqCst), 0);
    assert_eq!(late.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn bound_mutation_after_mutation_is_repaired_or_fails_before_effects() {
    let invalid = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [
            {"id": "first-secret", "tool_id": "first", "arguments": {}},
            {
                "id": "second-secret",
                "tool_id": "second",
                "arguments": {"input": "placeholder"},
                "depends_on": ["first-secret"],
                "result_bindings": [{
                    "target_pointer": "/input",
                    "source": {"node_id": "first-secret", "output_pointer": "/value"}
                }]
            }
        ]}
    });
    let first = Arc::new(FixedTool::new(
        "first",
        false,
        ToolResult::success(json!({"value": "mutated"})),
    ));
    let second = Arc::new(FixedTool::new(
        "second",
        false,
        ToolResult::success(json!({"ok": true})),
    ));

    let adaptive_provider = Arc::new(
        MockModelProvider::scripted([
            final_response(invalid.to_string()),
            final_response(invalid.to_string()),
            final_response("direct fallback"),
        ])
        .with_capabilities(planning_capabilities(1)),
    );
    let adaptive = AgentRunner::builder(adaptive_provider.clone())
        .tools(registry([
            first.clone() as Arc<dyn Tool>,
            second.clone() as Arc<dyn Tool>,
        ]))
        .policy(Arc::new(AllowAllPolicy))
        .build()
        .run(request(&["first", "second"]))
        .await
        .unwrap();
    assert_eq!(adaptive.status, RunStatus::Completed);
    assert_eq!(adaptive.final_output.as_deref(), Some("direct fallback"));
    assert_eq!(first.calls.load(Ordering::SeqCst), 0);
    assert_eq!(second.calls.load(Ordering::SeqCst), 0);

    let forced_provider = Arc::new(
        MockModelProvider::scripted([
            final_response(invalid.to_string()),
            final_response(invalid.to_string()),
        ])
        .with_capabilities(planning_capabilities(1)),
    );
    let forced = AgentRunner::builder(forced_provider)
        .tools(registry([
            first.clone() as Arc<dyn Tool>,
            second.clone() as Arc<dyn Tool>,
        ]))
        .policy(Arc::new(AllowAllPolicy))
        .build()
        .run_with_strategy(request(&["first", "second"]), RunStrategy::DeclarativePlan)
        .await
        .unwrap();
    assert_eq!(forced.status, RunStatus::Failed);
    assert_eq!(first.calls.load(Ordering::SeqCst), 0);
    assert_eq!(second.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn same_concurrency_key_is_exclusive_across_runs() {
    let provider = Arc::new(MockModelProvider::scripted([
        tool_response(ToolCall::new("one", "held", "{}")),
        tool_response(ToolCall::new("two", "held", "{}")),
        final_response("done"),
        final_response("done"),
    ]));
    let entered = Arc::new(Semaphore::new(0));
    let release = Arc::new(Semaphore::new(0));
    let active = Arc::new(AtomicU32::new(0));
    let maximum = Arc::new(AtomicU32::new(0));
    let tool = Arc::new(HeldTool {
        definition: definition("held", true, true).with_concurrency_key("shared"),
        entered: entered.clone(),
        release: release.clone(),
        active,
        maximum: maximum.clone(),
    });
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry([tool as Arc<dyn Tool>]))
            .build(),
    );
    let first_runner = runner.clone();
    let first = tokio::spawn(async move {
        first_runner
            .run_with_strategy(request(&["held"]), RunStrategy::Direct)
            .await
    });
    entered.acquire().await.unwrap().forget();
    let second_runner = runner.clone();
    let second = tokio::spawn(async move {
        second_runner
            .run_with_strategy(request(&["held"]), RunStrategy::Direct)
            .await
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while provider.requests().len() < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(50), entered.acquire())
            .await
            .is_err()
    );
    release.add_permits(1);
    tokio::time::timeout(Duration::from_secs(1), entered.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    release.add_permits(1);
    assert_eq!(first.await.unwrap().unwrap().status, RunStatus::Completed);
    assert_eq!(second.await.unwrap().unwrap().status, RunStatus::Completed);
    assert_eq!(maximum.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn direct_and_declarative_runs_share_concurrency_keys() {
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": "planned", "tool_id": "held", "arguments": {}}]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([
            final_response(envelope.to_string()),
            tool_response(ToolCall::new("direct", "held", "{}")),
            final_response("done"),
            final_response("done"),
        ])
        .with_capabilities(planning_capabilities(1)),
    );
    let entered = Arc::new(Semaphore::new(0));
    let release = Arc::new(Semaphore::new(0));
    let maximum = Arc::new(AtomicU32::new(0));
    let tool = Arc::new(HeldTool {
        definition: definition("held", true, true).with_concurrency_key("shared"),
        entered: entered.clone(),
        release: release.clone(),
        active: Arc::new(AtomicU32::new(0)),
        maximum: maximum.clone(),
    });
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry([tool as Arc<dyn Tool>]))
            .build(),
    );

    let planned_runner = runner.clone();
    let planned = tokio::spawn(async move {
        planned_runner
            .run_with_strategy(request(&["held"]), RunStrategy::DeclarativePlan)
            .await
    });
    entered.acquire().await.unwrap().forget();
    let direct_runner = runner.clone();
    let direct = tokio::spawn(async move {
        direct_runner
            .run_with_strategy(request(&["held"]), RunStrategy::Direct)
            .await
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while provider.requests().len() < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(50), entered.acquire())
            .await
            .is_err()
    );
    release.add_permits(1);
    tokio::time::timeout(Duration::from_secs(1), entered.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    release.add_permits(1);

    assert_eq!(planned.await.unwrap().unwrap().status, RunStatus::Completed);
    assert_eq!(direct.await.unwrap().unwrap().status, RunStatus::Completed);
    assert_eq!(maximum.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancellation_while_waiting_for_a_concurrency_key_never_invokes_the_tool() {
    let provider = Arc::new(MockModelProvider::scripted([
        tool_response(ToolCall::new("first", "held", "{}")),
        tool_response(ToolCall::new("queued", "held", "{}")),
        final_response("done"),
    ]));
    let entered = Arc::new(Semaphore::new(0));
    let release = Arc::new(Semaphore::new(0));
    let maximum = Arc::new(AtomicU32::new(0));
    let tool = Arc::new(HeldTool {
        definition: definition("held", true, true).with_concurrency_key("shared"),
        entered: entered.clone(),
        release: release.clone(),
        active: Arc::new(AtomicU32::new(0)),
        maximum: maximum.clone(),
    });
    let runner = Arc::new(
        AgentRunner::builder(provider.clone())
            .tools(registry([tool as Arc<dyn Tool>]))
            .build(),
    );

    let first_runner = runner.clone();
    let first = tokio::spawn(async move {
        first_runner
            .run_with_strategy(request(&["held"]), RunStrategy::Direct)
            .await
    });
    entered.acquire().await.unwrap().forget();
    let queued_runner = runner.clone();
    let queued_request = request(&["held"]);
    let queued_cancellation = queued_request.cancellation.clone();
    let queued = tokio::spawn(async move {
        queued_runner
            .run_with_strategy(queued_request, RunStrategy::Direct)
            .await
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while provider.requests().len() < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    queued_cancellation.cancel();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), queued)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .status,
        RunStatus::Cancelled
    );
    assert_eq!(entered.available_permits(), 0);
    release.add_permits(1);
    assert_eq!(first.await.unwrap().unwrap().status, RunStatus::Completed);
    assert_eq!(maximum.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn different_concurrency_keys_can_overlap_across_runs() {
    let provider = Arc::new(MockModelProvider::scripted([
        tool_response(ToolCall::new("one", "one", "{}")),
        tool_response(ToolCall::new("two", "two", "{}")),
        final_response("done"),
        final_response("done"),
    ]));
    let entered = Arc::new(Semaphore::new(0));
    let release = Arc::new(Semaphore::new(0));
    let active = Arc::new(AtomicU32::new(0));
    let maximum = Arc::new(AtomicU32::new(0));
    let one = Arc::new(HeldTool {
        definition: definition("one", true, true).with_concurrency_key("one-key"),
        entered: entered.clone(),
        release: release.clone(),
        active: active.clone(),
        maximum: maximum.clone(),
    });
    let two = Arc::new(HeldTool {
        definition: definition("two", true, true).with_concurrency_key("two-key"),
        entered: entered.clone(),
        release: release.clone(),
        active,
        maximum: maximum.clone(),
    });
    let runner = Arc::new(
        AgentRunner::builder(provider)
            .tools(registry([one as Arc<dyn Tool>, two as Arc<dyn Tool>]))
            .build(),
    );
    let first_runner = runner.clone();
    let first = tokio::spawn(async move {
        first_runner
            .run_with_strategy(request(&["one", "two"]), RunStrategy::Direct)
            .await
    });
    entered.acquire().await.unwrap().forget();
    let second_runner = runner.clone();
    let second = tokio::spawn(async move {
        second_runner
            .run_with_strategy(request(&["one", "two"]), RunStrategy::Direct)
            .await
    });
    tokio::time::timeout(Duration::from_secs(1), entered.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    assert_eq!(maximum.load(Ordering::SeqCst), 2);
    release.add_permits(2);
    assert_eq!(first.await.unwrap().unwrap().status, RunStatus::Completed);
    assert_eq!(second.await.unwrap().unwrap().status, RunStatus::Completed);
}

#[tokio::test]
async fn plan_event_identifiers_do_not_persist_model_node_ids() {
    let secret_node_id = "secret-customer-token-123";
    let envelope = json!({
        "strategy": "declarative_plan",
        "plan": {"nodes": [{"id": secret_node_id, "tool_id": "read", "arguments": {}}]}
    });
    let provider = Arc::new(
        MockModelProvider::scripted([final_response(envelope.to_string()), final_response("done")])
            .with_capabilities(planning_capabilities(1)),
    );
    let read = Arc::new(FixedTool::new("read", true, ToolResult::success(json!({}))));
    let events = Arc::new(InMemoryEventSink::default());
    let result = AgentRunner::builder(provider)
        .tools(registry([read as Arc<dyn Tool>]))
        .event_sink(events.clone())
        .build()
        .run(request(&["read"]))
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert!(!serde_json::to_string(&events.events())
        .unwrap()
        .contains(secret_node_id));
    assert!(result
        .tool_calls
        .iter()
        .all(|call| !call.id.contains(secret_node_id)));
}
