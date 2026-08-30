use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, tool_response, MockModelProvider, MockStep},
    AgentDefinition, AgentLimits, AgentRunner, ApprovalHandler, ApprovalRecord, EventRecord,
    GenerationOptions, HarnessError, InMemoryEventSink, JsonMap, Message, MessageRole,
    ModelCapabilities, ModelInfo, ModelProvider, ModelRequest, ModelResponse, PolicyDecision,
    PolicyEngine, ProviderHealth, RunEvent, RunOverrides, RunRequest, RunStatus, RunStrategy, Tool,
    ToolCall, ToolCaller, ToolDefinition, ToolRegistry, ToolResult, ToolRisk,
};
use serde_json::{json, Value};
use std::{
    future::pending,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};
use tokio_util::sync::CancellationToken;

fn request() -> RunRequest {
    RunRequest {
        agent: AgentDefinition {
            id: "external-test".into(),
            name: "External test".into(),
            version: "1".into(),
            system_instructions: String::new(),
            default_model: "mock-model".into(),
            tool_allowlist: vec!["read".into()],
            limits: AgentLimits::default(),
            generation: GenerationOptions::default(),
            output_schema: None,
            metadata: JsonMap::new(),
        },
        input: "hello".into(),
        application_context: JsonMap::new(),
        history: vec![],
        metadata: JsonMap::new(),
        overrides: RunOverrides::default(),
        evaluation: JsonMap::new(),
        cancellation: CancellationToken::new(),
        run_id: None,
        trace_id: None,
    }
}

fn call(id: &str, tool_id: &str, arguments_json: &str) -> ToolCall {
    ToolCall::new(id, tool_id, arguments_json)
}

#[tokio::test]
async fn host_supplied_run_and_trace_ids_preserve_external_correlation() {
    let provider = Arc::new(MockModelProvider::scripted([final_response("done")]));
    let runner = AgentRunner::builder(provider).build();
    let result = runner
        .run(
            request()
                .with_run_id("runtime-run-1")
                .with_trace_id("runtime-trace-1"),
        )
        .await
        .expect("the canonical runner must accept transport correlations");

    assert_eq!(result.id, "runtime-run-1");
    assert_eq!(result.trace_id, "runtime-trace-1");
}

fn response(final_output: Option<&str>, tool_calls: Vec<ToolCall>) -> ModelResponse {
    let mut response = ModelResponse::new("mock-model").with_tool_calls(tool_calls);
    response.final_output = final_output.map(str::to_owned);
    response
}

enum ToolBehavior {
    Return(ToolResult),
    Error(HarnessError),
    Pending,
}

struct TestTool {
    definition: ToolDefinition,
    calls: AtomicU32,
    behavior: ToolBehavior,
}

impl TestTool {
    fn read(id: &str, schema: Value) -> Self {
        Self {
            definition: ToolDefinition::new(id, id, "test tool", schema)
                .with_risk(ToolRisk::Low)
                .with_idempotent(true)
                .with_read_only(true),
            calls: AtomicU32::new(0),
            behavior: ToolBehavior::Return(ToolResult::success(json!({"value": "ok"}))),
        }
    }

    fn state_changing(id: &str) -> Self {
        let mut tool = Self::read(id, json!({"type": "object"}));
        tool.definition.read_only = false;
        tool.definition.idempotent = false;
        tool.definition.risk = ToolRisk::High;
        tool
    }

    fn returning(mut self, result: ToolResult) -> Self {
        self.behavior = ToolBehavior::Return(result);
        self
    }

    fn failing(mut self, error: HarnessError) -> Self {
        self.behavior = ToolBehavior::Error(error);
        self
    }

    fn pending(mut self) -> Self {
        self.behavior = ToolBehavior::Pending;
        self
    }
}

#[async_trait]
impl Tool for TestTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match &self.behavior {
            ToolBehavior::Return(result) => Ok(result.clone()),
            ToolBehavior::Error(error) => Err(error.clone()),
            ToolBehavior::Pending => pending().await,
        }
    }
}

fn registry(tool: Arc<dyn Tool>) -> ToolRegistry {
    let mut registry = ToolRegistry::default();
    registry.register(tool).unwrap();
    registry
}

struct FixedPolicy(Result<PolicyDecision, HarnessError>);

#[async_trait]
impl PolicyEngine for FixedPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        self.0.clone()
    }
}

struct PendingPolicy;

#[async_trait]
impl PolicyEngine for PendingPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        pending().await
    }
}

struct CancellingPolicy;

#[async_trait]
impl PolicyEngine for CancellingPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        request: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        request.cancellation.cancel();
        Ok(PolicyDecision::Allow {
            reason: "cancel immediately before tool".into(),
        })
    }
}

enum ApprovalBehavior {
    Grant(bool),
    Error(HarnessError),
    Pending,
}

struct TestApproval(ApprovalBehavior);

#[async_trait]
impl ApprovalHandler for TestApproval {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        match &self.0 {
            ApprovalBehavior::Grant(granted) => Ok(ApprovalRecord::new(
                "",
                tool.id.clone(),
                *granted,
                "test approval",
            )),
            ApprovalBehavior::Error(error) => Err(error.clone()),
            ApprovalBehavior::Pending => pending().await,
        }
    }
}

struct PendingProvider;

#[async_trait]
impl ModelProvider for PendingProvider {
    fn id(&self) -> &str {
        "pending"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::default()
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth::healthy())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(vec![])
    }

    async fn complete(&self, _: ModelRequest) -> Result<ModelResponse, HarnessError> {
        pending().await
    }
}

fn completed_events(records: &[EventRecord]) -> Vec<RunStatus> {
    records
        .iter()
        .filter_map(|record| match &record.event {
            RunEvent::Completed { status } => Some(status.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn public_api_runs_a_scripted_final_response() {
    let runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
        "ok",
    )])))
    .build();
    let result = runner.run(request()).await.unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.final_output.as_deref(), Some("ok"));
}

#[tokio::test]
async fn multi_step_tool_feedback_preserves_message_and_event_order() {
    let provider = Arc::new(MockModelProvider::scripted([
        MockStep::Response(response(
            None,
            vec![
                call("first", "read", r#"{"key":"one"}"#),
                call("second", "read", r#"{"key":"two"}"#),
            ],
        )),
        final_response("done"),
    ]));
    let tool = Arc::new(TestTool::read(
        "read",
        json!({"type":"object","required":["key"],"properties":{"key":{"type":"string"}}}),
    ));
    let events = Arc::new(InMemoryEventSink::default());
    let mut run_request = request();
    run_request.agent.system_instructions = "system".into();
    run_request.history = vec![Message::user("prior")];
    let runner = AgentRunner::builder(provider.clone())
        .tools(registry(tool.clone()))
        .event_sink(events.clone())
        .build();

    let result = runner
        .run_with_strategy(run_request, RunStrategy::Direct)
        .await
        .unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 2);
    assert_eq!(result.tool_calls[0].arguments_json, r#"{"key":"one"}"#);
    assert_eq!(result.tool_calls[1].arguments_json, r#"{"key":"two"}"#);
    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0]
            .messages
            .iter()
            .map(|message| &message.role)
            .collect::<Vec<_>>(),
        vec![&MessageRole::System, &MessageRole::User, &MessageRole::User]
    );
    let feedback = &requests[1].messages[3..];
    assert_eq!(feedback.len(), 3);
    assert_eq!(feedback[0].role, MessageRole::Assistant);
    assert_eq!(feedback[0].tool_calls.len(), 2);
    assert_eq!(feedback[0].tool_calls[0].id, "first");
    assert_eq!(feedback[0].tool_calls[1].id, "second");
    assert_eq!(feedback[1].tool_call_id.as_deref(), Some("first"));
    assert_eq!(feedback[2].tool_call_id.as_deref(), Some("second"));
    assert_eq!(
        events
            .events()
            .iter()
            .map(|record| std::mem::discriminant(&record.event))
            .collect::<Vec<_>>(),
        vec![
            std::mem::discriminant(&RunEvent::Started {
                run_id: String::new(),
                trace_id: String::new(),
            }),
            std::mem::discriminant(&RunEvent::StrategySelected {
                requested: RunStrategy::Direct,
                selected: RunStrategy::Direct,
                reason: llama_harness_core::StrategySelectionReason::Forced,
            }),
            std::mem::discriminant(&RunEvent::ModelRequested {
                call_number: 0,
                model: String::new(),
            }),
            std::mem::discriminant(&RunEvent::ModelResponded { call_number: 0 }),
            std::mem::discriminant(&RunEvent::PolicyDecided {
                call_id: String::new(),
                decision: PolicyDecision::Allow {
                    reason: String::new(),
                },
            }),
            std::mem::discriminant(&RunEvent::ToolCompleted {
                call_id: String::new(),
                tool_id: String::new(),
                ok: true,
            }),
            std::mem::discriminant(&RunEvent::PolicyDecided {
                call_id: String::new(),
                decision: PolicyDecision::Allow {
                    reason: String::new(),
                },
            }),
            std::mem::discriminant(&RunEvent::ToolCompleted {
                call_id: String::new(),
                tool_id: String::new(),
                ok: true,
            }),
            std::mem::discriminant(&RunEvent::ModelRequested {
                call_number: 0,
                model: String::new(),
            }),
            std::mem::discriminant(&RunEvent::ModelResponded { call_number: 0 }),
            std::mem::discriminant(&RunEvent::StrategyUsage {
                strategy: RunStrategy::Direct,
                model_calls: 0,
                planning_model_calls: 0,
                repair_model_calls: 0,
                recovery_model_calls: 0,
                reactive_model_calls: 0,
                tool_calls: 0,
                tool_issued: 0,
                tool_reused: 0,
                tool_rejected: 0,
                tool_pre_dispatch_aborted: 0,
                tool_completed: 0,
                tool_failed: 0,
                tool_cancelled: 0,
                duration_ms: 0,
            }),
            std::mem::discriminant(&RunEvent::Completed {
                status: RunStatus::Completed,
            }),
        ]
    );
}

#[tokio::test]
async fn concurrent_runs_have_isolated_monotonic_event_envelopes() {
    let sink = Arc::new(InMemoryEventSink::default());
    let runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
        final_response("one"),
        final_response("two"),
    ])))
    .event_sink(sink.clone())
    .build();
    let (one, two) = tokio::join!(runner.run(request()), runner.run(request()));
    let results = [one.unwrap(), two.unwrap()];
    let records = sink.events();

    assert_ne!(results[0].id, results[1].id);
    for result in results {
        let run_records = records
            .iter()
            .filter(|record| record.run_id == result.id)
            .collect::<Vec<_>>();
        assert!(!run_records.is_empty());
        assert!(run_records
            .iter()
            .all(|record| record.trace_id == result.trace_id));
        assert_eq!(
            run_records
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            (1..=run_records.len() as u64).collect::<Vec<_>>()
        );
        assert!(run_records
            .windows(2)
            .all(|window| window[0].timestamp_ms <= window[1].timestamp_ms));
        assert!(matches!(run_records[0].event, RunEvent::Started { .. }));
        assert!(matches!(
            run_records.last().unwrap().event,
            RunEvent::Completed {
                status: RunStatus::Completed
            }
        ));
    }
}

#[tokio::test]
async fn retryable_provider_errors_are_bounded_and_nonretryable_errors_are_not_retried() {
    let retrying = Arc::new(MockModelProvider::scripted([
        MockStep::Error(HarnessError::RetryableProvider("busy".into())),
        final_response("recovered"),
    ]));
    let mut retry_request = request();
    retry_request.agent.limits.max_provider_retries = 1;
    let recovered = AgentRunner::builder(retrying.clone())
        .build()
        .run(retry_request)
        .await
        .unwrap();
    assert_eq!(recovered.status, RunStatus::Completed);
    assert_eq!(retrying.requests().len(), 2);

    let ordinary = Arc::new(MockModelProvider::scripted([
        MockStep::Error(HarnessError::Provider("bad request".into())),
        final_response("must not run"),
    ]));
    let failed = AgentRunner::builder(ordinary.clone())
        .build()
        .run(request())
        .await
        .unwrap();
    assert_eq!(failed.status, RunStatus::Failed);
    assert_eq!(ordinary.requests().len(), 1);

    let exhausted = Arc::new(MockModelProvider::scripted([
        MockStep::Error(HarnessError::RetryableProvider("one".into())),
        MockStep::Error(HarnessError::RetryableProvider("two".into())),
        final_response("must not run"),
    ]));
    let mut exhausted_request = request();
    exhausted_request.agent.limits.max_provider_retries = 1;
    let failed = AgentRunner::builder(exhausted.clone())
        .build()
        .run(exhausted_request)
        .await
        .unwrap();
    assert_eq!(failed.status, RunStatus::Failed);
    assert_eq!(exhausted.requests().len(), 2);
}

#[tokio::test]
async fn tools_are_never_retried_even_for_retryable_provider_errors() {
    let tool = Arc::new(
        TestTool::read("read", json!({"type": "object"}))
            .failing(HarnessError::RetryableProvider("do not retry tools".into())),
    );
    let result = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
        tool_response(call("one", "read", "{}")),
        final_response("must not run"),
    ])))
    .tools(registry(tool.clone()))
    .build()
    .run(request())
    .await
    .unwrap();
    assert_eq!(result.status, RunStatus::Failed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn active_provider_cancellation_returns_cancelled_with_terminal_event() {
    let sink = Arc::new(InMemoryEventSink::default());
    let run_request = request();
    let cancellation = run_request.cancellation.clone();
    tokio::spawn(async move {
        tokio::task::yield_now().await;
        cancellation.cancel();
    });
    let result = AgentRunner::builder(Arc::new(PendingProvider))
        .event_sink(sink.clone())
        .build()
        .run(run_request)
        .await
        .unwrap();
    assert_eq!(result.status, RunStatus::Cancelled);
    assert!(result.cancelled);
    assert_eq!(completed_events(&sink.events()), vec![RunStatus::Cancelled]);
}

#[tokio::test]
async fn active_cancellation_interrupts_policy_approval_and_tool_awaits() {
    fn cancel_after_yield(cancellation: CancellationToken) {
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            cancellation.cancel();
        });
    }

    let policy_request = request();
    cancel_after_yield(policy_request.cancellation.clone());
    let policy_result =
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([tool_response(
            call("policy", "read", "{}"),
        )])))
        .tools(registry(Arc::new(TestTool::read(
            "read",
            json!({"type": "object"}),
        ))))
        .policy(Arc::new(PendingPolicy))
        .build()
        .run(policy_request)
        .await
        .unwrap();
    assert_eq!(policy_result.status, RunStatus::Cancelled);

    let approval_request = request();
    cancel_after_yield(approval_request.cancellation.clone());
    let approval_result =
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([tool_response(
            call("approval", "read", "{}"),
        )])))
        .tools(registry(Arc::new(TestTool::read(
            "read",
            json!({"type": "object"}),
        ))))
        .policy(Arc::new(FixedPolicy(Ok(PolicyDecision::RequireApproval {
            reason: "ask".into(),
        }))))
        .approvals(Arc::new(TestApproval(ApprovalBehavior::Pending)))
        .build()
        .run(approval_request)
        .await
        .unwrap();
    assert_eq!(approval_result.status, RunStatus::Cancelled);

    let pending_tool = Arc::new(TestTool::read("read", json!({"type": "object"})).pending());
    let tool_request = request();
    cancel_after_yield(tool_request.cancellation.clone());
    let tool_result = AgentRunner::builder(Arc::new(MockModelProvider::scripted([tool_response(
        call("tool", "read", "{}"),
    )])))
    .tools(registry(pending_tool.clone()))
    .build()
    .run(tool_request)
    .await
    .unwrap();
    assert_eq!(tool_result.status, RunStatus::Cancelled);
    assert_eq!(pending_tool.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn provider_per_call_timeout_and_absolute_deadline_are_active() {
    let mut provider_timeout = request();
    provider_timeout.agent.limits.max_model_call_duration_ms = Some(2);
    let result = AgentRunner::builder(Arc::new(PendingProvider))
        .build()
        .run(provider_timeout)
        .await
        .unwrap();
    assert_eq!(result.status, RunStatus::Failed);
    assert_eq!(result.errors[0].code, "timed_out");

    let mut absolute_timeout = request();
    absolute_timeout.agent.limits.max_run_duration_ms = Some(2);
    absolute_timeout.agent.limits.max_model_call_duration_ms = Some(1000);
    let result = AgentRunner::builder(Arc::new(PendingProvider))
        .build()
        .run(absolute_timeout)
        .await
        .unwrap();
    assert_eq!(result.status, RunStatus::Failed);
    assert_eq!(result.errors[0].code, "timed_out");
}

#[tokio::test(start_paused = true)]
async fn policy_approval_and_tool_awaits_obey_the_absolute_deadline() {
    async fn run_with(
        policy: Arc<dyn PolicyEngine>,
        approval: Arc<dyn ApprovalHandler>,
        tool: Arc<TestTool>,
    ) -> llama_harness_core::RunResult {
        let mut run_request = request();
        run_request.agent.limits.max_run_duration_ms = Some(2);
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([tool_response(
            call("one", "read", "{}"),
        )])))
        .tools(registry(tool))
        .policy(policy)
        .approvals(approval)
        .build()
        .run(run_request)
        .await
        .unwrap()
    }

    let timed_out_policy = run_with(
        Arc::new(PendingPolicy),
        Arc::new(TestApproval(ApprovalBehavior::Grant(true))),
        Arc::new(TestTool::read("read", json!({"type": "object"}))),
    )
    .await;
    assert_eq!(timed_out_policy.status, RunStatus::Failed);
    assert_eq!(timed_out_policy.errors.last().unwrap().code, "timed_out");

    let approval_policy = Arc::new(FixedPolicy(Ok(PolicyDecision::RequireApproval {
        reason: "ask".into(),
    })));
    let timed_out_approval = run_with(
        approval_policy,
        Arc::new(TestApproval(ApprovalBehavior::Pending)),
        Arc::new(TestTool::read("read", json!({"type": "object"}))),
    )
    .await;
    assert_eq!(timed_out_approval.status, RunStatus::Failed);
    assert_eq!(timed_out_approval.errors.last().unwrap().code, "timed_out");

    let pending_tool = Arc::new(TestTool::read("read", json!({"type": "object"})).pending());
    let timed_out_tool = run_with(
        Arc::new(FixedPolicy(Ok(PolicyDecision::Allow {
            reason: "allow".into(),
        }))),
        Arc::new(TestApproval(ApprovalBehavior::Grant(true))),
        pending_tool.clone(),
    )
    .await;
    assert_eq!(timed_out_tool.status, RunStatus::Failed);
    assert_eq!(timed_out_tool.errors.last().unwrap().code, "timed_out");
    assert_eq!(pending_tool.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancellation_is_rechecked_immediately_before_tool_invocation() {
    let tool = Arc::new(TestTool::read("read", json!({"type": "object"})));
    let result = AgentRunner::builder(Arc::new(MockModelProvider::scripted([tool_response(
        call("one", "read", "{}"),
    )])))
    .tools(registry(tool.clone()))
    .policy(Arc::new(CancellingPolicy))
    .build()
    .run(request())
    .await
    .unwrap();
    assert_eq!(result.status, RunStatus::Cancelled);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn callback_errors_return_failed_results_and_terminal_events() {
    let provider_sink = Arc::new(InMemoryEventSink::default());
    let provider_result =
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([MockStep::Error(
            HarnessError::Provider("provider boom".into()),
        )])))
        .event_sink(provider_sink.clone())
        .build()
        .run(request())
        .await
        .unwrap();
    assert_eq!(provider_result.status, RunStatus::Failed);
    assert_eq!(
        completed_events(&provider_sink.events()),
        vec![RunStatus::Failed]
    );

    let policy_sink = Arc::new(InMemoryEventSink::default());
    let policy_result =
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([tool_response(
            call("one", "read", "{}"),
        )])))
        .tools(registry(Arc::new(TestTool::read(
            "read",
            json!({"type": "object"}),
        ))))
        .policy(Arc::new(FixedPolicy(Err(HarnessError::Policy(
            "policy boom".into(),
        )))))
        .event_sink(policy_sink.clone())
        .build()
        .run(request())
        .await
        .unwrap();
    assert_eq!(policy_result.status, RunStatus::Failed);
    assert_eq!(
        completed_events(&policy_sink.events()),
        vec![RunStatus::Failed]
    );

    let approval_sink = Arc::new(InMemoryEventSink::default());
    let approval_result =
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([tool_response(
            call("one", "read", "{}"),
        )])))
        .tools(registry(Arc::new(TestTool::read(
            "read",
            json!({"type": "object"}),
        ))))
        .policy(Arc::new(FixedPolicy(Ok(PolicyDecision::RequireApproval {
            reason: "ask".into(),
        }))))
        .approvals(Arc::new(TestApproval(ApprovalBehavior::Error(
            HarnessError::Approval("approval boom".into()),
        ))))
        .event_sink(approval_sink.clone())
        .build()
        .run(request())
        .await
        .unwrap();
    assert_eq!(approval_result.status, RunStatus::Failed);
    assert_eq!(
        completed_events(&approval_sink.events()),
        vec![RunStatus::Failed]
    );

    let tool_sink = Arc::new(InMemoryEventSink::default());
    let tool = Arc::new(
        TestTool::read("read", json!({"type": "object"}))
            .failing(HarnessError::Tool("tool boom".into())),
    );
    let tool_result = AgentRunner::builder(Arc::new(MockModelProvider::scripted([tool_response(
        call("one", "read", "{}"),
    )])))
    .tools(registry(tool))
    .event_sink(tool_sink.clone())
    .build()
    .run(request())
    .await
    .unwrap();
    assert_eq!(tool_result.status, RunStatus::Failed);
    assert_eq!(
        completed_events(&tool_sink.events()),
        vec![RunStatus::Failed]
    );
}

#[tokio::test]
async fn policy_denial_approval_grant_and_approval_denial_are_recorded() {
    let denied_tool = Arc::new(TestTool::read("read", json!({"type": "object"})));
    let denied = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
        tool_response(call("denied", "read", "{}")),
        final_response("done"),
    ])))
    .tools(registry(denied_tool.clone()))
    .policy(Arc::new(FixedPolicy(Ok(PolicyDecision::Deny {
        reason: "blocked".into(),
    }))))
    .build()
    .run(request())
    .await
    .unwrap();
    assert_eq!(denied.status, RunStatus::Completed);
    assert_eq!(denied_tool.calls.load(Ordering::SeqCst), 0);
    assert_eq!(denied.errors[0].code, "tool_rejected");

    for granted in [true, false] {
        let tool = Arc::new(TestTool::read("read", json!({"type": "object"})));
        let result = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
            tool_response(call("approval", "read", "{}")),
            final_response("done"),
        ])))
        .tools(registry(tool.clone()))
        .policy(Arc::new(FixedPolicy(Ok(PolicyDecision::RequireApproval {
            reason: "ask".into(),
        }))))
        .approvals(Arc::new(TestApproval(ApprovalBehavior::Grant(granted))))
        .build()
        .run(request())
        .await
        .unwrap();
        assert_eq!(result.status, RunStatus::Completed);
        assert_eq!(result.approvals[0].granted, granted);
        assert_eq!(tool.calls.load(Ordering::SeqCst), u32::from(granted));
    }
}

#[tokio::test]
async fn default_policy_denies_state_changing_tools() {
    let tool = Arc::new(TestTool::state_changing("read"));
    let result = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
        tool_response(call("one", "read", "{}")),
        final_response("done"),
    ])))
    .tools(registry(tool.clone()))
    .build()
    .run(request())
    .await
    .unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    assert!(matches!(
        result.policy_decisions[0],
        PolicyDecision::Deny { .. }
    ));
}

#[tokio::test]
async fn duplicate_registration_preserves_the_original_tool() {
    let original = Arc::new(TestTool::read("read", json!({"type": "object"})));
    let duplicate = Arc::new(TestTool::read("read", json!({"type": "object"})));
    let mut tools = registry(original.clone());
    let error = tools.register(duplicate.clone()).unwrap_err();
    assert!(matches!(error, HarnessError::InvalidTool(_)));

    let result = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
        tool_response(call("one", "read", "{}")),
        final_response("done"),
    ])))
    .tools(tools)
    .build()
    .run(request())
    .await
    .unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(original.calls.load(Ordering::SeqCst), 1);
    assert_eq!(duplicate.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn tool_schemas_reject_all_non_fragment_references_without_resolution() {
    for reference in [
        "https://127.0.0.1/schema.json",
        "http://example.invalid/schema.json",
        "file:///etc/passwd",
        "relative/schema.json",
    ] {
        let tool = Arc::new(TestTool::read("read", json!({"$ref": reference})));
        let error = ToolRegistry::default().register(tool).unwrap_err();
        assert!(error
            .to_string()
            .contains("external schema reference is disabled"));
    }

    let recursive = Arc::new(TestTool::read(
        "read",
        json!({"$recursiveRef": "https://example.invalid/schema.json"}),
    ));
    assert!(ToolRegistry::default().register(recursive).is_err());

    let local = Arc::new(TestTool::read(
        "read",
        json!({
            "$defs": {"value": {"type": "string"}},
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/value"}}
        }),
    ));
    ToolRegistry::default().register(local).unwrap();
}

#[tokio::test]
async fn output_schemas_reject_http_and_file_references_before_provider_use() {
    for reference in ["https://127.0.0.1/schema.json", "file:///etc/passwd"] {
        let provider = Arc::new(MockModelProvider::scripted([final_response("{}")]));
        let mut run_request = request();
        run_request.agent.output_schema = Some(json!({"$ref": reference}));
        let error = AgentRunner::builder(provider.clone())
            .build()
            .run(run_request)
            .await
            .unwrap_err();
        assert!(matches!(error, HarnessError::InvalidRequest(_)));
        assert!(error
            .to_string()
            .contains("external schema reference is disabled"));
        assert!(provider.requests().is_empty());
    }
}

#[tokio::test]
async fn schema_repair_succeeds_and_exhaustion_fails_at_the_exact_limit() {
    let schema = json!({"type":"object","required":["answer"]});
    let repaired_provider = Arc::new(MockModelProvider::scripted([
        final_response("not json"),
        final_response(r#"{"answer":"done"}"#),
    ]));
    let mut repair_request = request();
    repair_request.agent.output_schema = Some(schema.clone());
    repair_request.agent.limits.max_output_repairs = 1;
    let repaired = AgentRunner::builder(repaired_provider.clone())
        .build()
        .run(repair_request)
        .await
        .unwrap();
    assert_eq!(repaired.status, RunStatus::Completed);
    assert_eq!(repaired_provider.requests().len(), 2);
    assert_eq!(
        repaired_provider.requests()[1]
            .messages
            .last()
            .unwrap()
            .role,
        MessageRole::System
    );

    let exhausted_provider = Arc::new(MockModelProvider::scripted([
        final_response("bad one"),
        final_response("bad two"),
        final_response(r#"{"answer":"too late"}"#),
    ]));
    let mut exhausted_request = request();
    exhausted_request.agent.output_schema = Some(schema);
    exhausted_request.agent.limits.max_output_repairs = 1;
    let exhausted = AgentRunner::builder(exhausted_provider.clone())
        .build()
        .run(exhausted_request)
        .await
        .unwrap();
    assert_eq!(exhausted.status, RunStatus::Failed);
    assert_eq!(exhausted_provider.requests().len(), 2);
    assert_eq!(exhausted.errors.last().unwrap().code, "invalid_output");
}

#[tokio::test]
async fn input_and_transcript_limits_are_inclusive() {
    let mut exact_input = request();
    exact_input.input = "four".into();
    exact_input.agent.limits.max_input_bytes = 4;
    assert_eq!(
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
            "ok"
        )])))
        .build()
        .run(exact_input)
        .await
        .unwrap()
        .status,
        RunStatus::Completed
    );

    let mut over_input = request();
    over_input.input = "five!".into();
    over_input.agent.limits.max_input_bytes = 4;
    assert!(matches!(
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
            "unused"
        )])))
        .build()
        .run(over_input)
        .await,
        Err(HarnessError::InvalidRequest(_))
    ));

    let mut exact_transcript = request();
    exact_transcript.agent.limits.max_transcript_bytes = 5;
    assert_eq!(
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
            "ok"
        )])))
        .build()
        .run(exact_transcript)
        .await
        .unwrap()
        .status,
        RunStatus::Completed
    );
    let mut over_transcript = request();
    over_transcript.agent.limits.max_transcript_bytes = 4;
    assert!(matches!(
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
            "unused"
        )])))
        .build()
        .run(over_transcript)
        .await,
        Err(HarnessError::InvalidRequest(_))
    ));
}

#[tokio::test]
async fn request_and_model_response_byte_limits_are_inclusive() {
    let mut exact_request = request();
    loop {
        let length = serde_json::to_vec(&exact_request).unwrap().len() as u64;
        if exact_request.agent.limits.max_request_payload_bytes == length {
            break;
        }
        exact_request.agent.limits.max_request_payload_bytes = length;
    }
    let exact_request_limit = exact_request.agent.limits.max_request_payload_bytes;
    assert_eq!(
        serde_json::to_vec(&exact_request).unwrap().len() as u64,
        exact_request_limit
    );
    assert_eq!(
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
            "ok"
        )])))
        .build()
        .run(exact_request.clone())
        .await
        .unwrap()
        .status,
        RunStatus::Completed
    );
    exact_request.agent.limits.max_request_payload_bytes = exact_request_limit - 1;
    assert!(matches!(
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
            "unused"
        )])))
        .build()
        .run(exact_request)
        .await,
        Err(HarnessError::InvalidRequest(_))
    ));

    let model_response = response(Some("ok"), vec![]);
    let response_bytes = serde_json::to_vec(&model_response).unwrap().len() as u64;
    let mut exact_model = request();
    exact_model.agent.limits.max_model_response_bytes = response_bytes;
    assert_eq!(
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([MockStep::Response(
            model_response.clone()
        ),])))
        .build()
        .run(exact_model)
        .await
        .unwrap()
        .status,
        RunStatus::Completed
    );
    let mut over_model = request();
    over_model.agent.limits.max_model_response_bytes = response_bytes - 1;
    let limited =
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([MockStep::Response(
            model_response,
        )])))
        .build()
        .run(over_model)
        .await
        .unwrap();
    assert_eq!(limited.status, RunStatus::LimitReached);
}

#[tokio::test]
async fn tool_argument_and_result_byte_limits_are_inclusive() {
    let exact_arguments = "{}";
    let exact_tool = Arc::new(TestTool::read("read", json!({"type": "object"})));
    let mut exact_request = request();
    exact_request.agent.limits.max_tool_arguments_bytes = exact_arguments.len() as u64;
    let exact = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
        tool_response(call("one", "read", exact_arguments)),
        final_response("done"),
    ])))
    .tools(registry(exact_tool.clone()))
    .build()
    .run(exact_request)
    .await
    .unwrap();
    assert_eq!(exact.status, RunStatus::Completed);
    assert_eq!(exact_tool.calls.load(Ordering::SeqCst), 1);

    let oversized_tool = Arc::new(TestTool::read("read", json!({"type": "object"})));
    let mut oversized_request = request();
    oversized_request.agent.limits.max_tool_arguments_bytes = 1;
    let oversized = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
        tool_response(call("one", "read", exact_arguments)),
        final_response("done"),
    ])))
    .tools(registry(oversized_tool.clone()))
    .build()
    .run(oversized_request)
    .await
    .unwrap();
    assert_eq!(oversized.status, RunStatus::Completed);
    assert_eq!(oversized_tool.calls.load(Ordering::SeqCst), 0);
    assert_eq!(oversized.errors[0].code, "tool_rejected");

    let tool_result = ToolResult::success(json!({"value": "ok"}));
    let result_bytes = serde_json::to_vec(&tool_result).unwrap().len() as u64;
    let exact_result_tool =
        Arc::new(TestTool::read("read", json!({"type": "object"})).returning(tool_result.clone()));
    let mut exact_result_request = request();
    exact_result_request.agent.limits.max_tool_result_bytes = result_bytes;
    let exact = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
        tool_response(call("one", "read", "{}")),
        final_response("done"),
    ])))
    .tools(registry(exact_result_tool))
    .build()
    .run(exact_result_request)
    .await
    .unwrap();
    assert_eq!(exact.status, RunStatus::Completed);

    let over_result_tool =
        Arc::new(TestTool::read("read", json!({"type": "object"})).returning(tool_result));
    let mut over_result_request = request();
    over_result_request.agent.limits.max_tool_result_bytes = result_bytes - 1;
    let limited = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
        tool_response(call("one", "read", "{}")),
        final_response("must not run"),
    ])))
    .tools(registry(over_result_tool))
    .build()
    .run(over_result_request)
    .await
    .unwrap();
    assert_eq!(limited.status, RunStatus::LimitReached);
}

#[tokio::test]
async fn json_depth_limits_are_inclusive_for_request_and_tool_payloads() {
    let mut exact_request = request();
    exact_request
        .application_context
        .insert("outer".into(), json!({"inner": 1}));
    exact_request.agent.limits.max_json_depth = 2;
    assert_eq!(
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
            "ok"
        )])))
        .build()
        .run(exact_request)
        .await
        .unwrap()
        .status,
        RunStatus::Completed
    );
    let mut over_request = request();
    over_request
        .application_context
        .insert("outer".into(), json!({"inner": 1}));
    over_request.agent.limits.max_json_depth = 1;
    assert!(matches!(
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
            "unused"
        )])))
        .build()
        .run(over_request)
        .await,
        Err(HarnessError::InvalidRequest(_))
    ));

    let tool = Arc::new(TestTool::read("read", json!({"type": "object"})));
    let mut tool_request = request();
    tool_request.agent.limits.max_json_depth = 1;
    let result = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
        tool_response(call("one", "read", r#"{"outer":{"inner":1}}"#)),
        final_response("done"),
    ])))
    .tools(registry(tool.clone()))
    .build()
    .run(tool_request)
    .await
    .unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    assert_eq!(result.errors[0].code, "tool_rejected");
}

#[tokio::test]
async fn transcript_growth_is_stopped_before_another_model_call() {
    let provider = Arc::new(MockModelProvider::scripted([
        tool_response(call("one", "read", "{}")),
        final_response("must not run"),
    ]));
    let tool = Arc::new(
        TestTool::read("read", json!({"type": "object"}))
            .returning(ToolResult::success(json!({"large": "payload"}))),
    );
    let mut run_request = request();
    run_request.agent.limits.max_transcript_bytes = run_request.input.len() as u64 + 1;
    let result = AgentRunner::builder(provider.clone())
        .tools(registry(tool))
        .build()
        .run(run_request)
        .await
        .unwrap();
    assert_eq!(result.status, RunStatus::LimitReached);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn malformed_unknown_disallowed_and_schema_invalid_calls_feed_back_without_execution() {
    let read = Arc::new(TestTool::read(
        "read",
        json!({"type":"object","required":["key"]}),
    ));
    let write = Arc::new(TestTool::read("write", json!({"type": "object"})));
    let mut tools = registry(read.clone());
    tools.register(write.clone()).unwrap();
    let provider = Arc::new(MockModelProvider::scripted([
        MockStep::Response(response(
            None,
            vec![
                call("malformed", "read", "not-json"),
                call("unknown", "missing", "{}"),
                call("disallowed", "write", "{}"),
                call("schema", "read", "{}"),
            ],
        )),
        final_response("done"),
    ]));
    let result = AgentRunner::builder(provider.clone())
        .tools(tools)
        .build()
        .run(request())
        .await
        .unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(read.calls.load(Ordering::SeqCst), 0);
    assert_eq!(write.calls.load(Ordering::SeqCst), 0);
    assert_eq!(result.errors.len(), 4);
    assert!(provider.requests()[1]
        .messages
        .iter()
        .rev()
        .take(4)
        .all(|message| { message.role == MessageRole::Tool && message.tool_call_id.is_some() }));
}

#[tokio::test]
async fn invalid_argument_values_are_redacted_from_errors_events_and_transcript() {
    const SECRET: &str = "sentinel-atomic-argument-secret";
    let read = Arc::new(TestTool::read(
        "read",
        json!({
            "type":"object",
            "required":["key"],
            "properties":{"key":{"type":"string","enum":["allowed"]}},
            "additionalProperties":false
        }),
    ));
    let provider = Arc::new(MockModelProvider::scripted([
        tool_response(call(
            "invalid-secret",
            "read",
            &format!(r#"{{"key":"{SECRET}"}}"#),
        )),
        final_response("recovered"),
    ]));
    let events = Arc::new(InMemoryEventSink::default());

    let result = AgentRunner::builder(provider.clone())
        .tools(registry(read.clone()))
        .event_sink(events.clone())
        .build()
        .run(request())
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(read.calls.load(Ordering::SeqCst), 0);
    assert!(result
        .errors
        .iter()
        .any(|error| error.code == "tool_rejected"
            && error.message.contains("arguments failed validation")));
    assert!(result
        .errors
        .iter()
        .all(|error| !error.message.contains(SECRET)));
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].id, "invalid-secret");
    assert_eq!(result.tool_calls[0].tool_id, "read");
    assert_eq!(result.tool_calls[0].arguments_json, "{}");
    assert!(!serde_json::to_string(&result).unwrap().contains(SECRET));

    let records = events.events();
    assert!(records.iter().any(|record| matches!(
        &record.event,
        RunEvent::ToolRejected { reason, .. }
            if reason.contains("arguments failed validation") && !reason.contains(SECRET)
    )));
    assert!(!serde_json::to_string(&records).unwrap().contains(SECRET));

    let transcript = &provider.requests()[1].messages;
    assert!(!serde_json::to_string(transcript).unwrap().contains(SECRET));
    let assistant_call = transcript
        .iter()
        .find(|message| message.role == MessageRole::Assistant)
        .and_then(|message| message.tool_calls.first())
        .unwrap();
    assert_eq!(assistant_call.arguments_json, "{}");
}

#[tokio::test]
async fn direct_runner_hides_and_rejects_tools_without_direct_permission() {
    let mut hidden = TestTool::read("read", json!({"type":"object"}));
    hidden.definition.allowed_callers = [ToolCaller::Programmatic].into();
    let hidden = Arc::new(hidden);
    let provider = Arc::new(MockModelProvider::scripted([
        tool_response(call("hidden", "read", "{}")),
        final_response("done"),
    ]));

    let result = AgentRunner::builder(provider.clone())
        .tools(registry(hidden.clone()))
        .build()
        .run(request())
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert!(provider.requests()[0].tools.is_empty());
    assert_eq!(hidden.calls.load(Ordering::SeqCst), 0);
    assert!(result
        .errors
        .iter()
        .any(|error| error.message.contains("does not allow direct calls")));
    let feedback: ToolResult = serde_json::from_str(
        &provider.requests()[1]
            .messages
            .iter()
            .find(|message| message.role == MessageRole::Tool)
            .unwrap()
            .content,
    )
    .unwrap();
    assert!(!feedback.ok);
}

#[tokio::test]
async fn tool_output_schema_accepts_valid_success() {
    let mut tool = TestTool::read("read", json!({"type":"object"}));
    tool.definition.output_schema = Some(json!({
        "type":"object",
        "required":["value"],
        "properties":{"value":{"type":"string"}},
        "additionalProperties":false
    }));
    let tool = Arc::new(tool);
    let provider = Arc::new(MockModelProvider::scripted([
        tool_response(call("valid", "read", "{}")),
        final_response("done"),
    ]));

    let result = AgentRunner::builder(provider.clone())
        .tools(registry(tool))
        .build()
        .run(request())
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    let feedback: ToolResult = serde_json::from_str(
        &provider.requests()[1]
            .messages
            .iter()
            .find(|message| message.role == MessageRole::Tool)
            .unwrap()
            .content,
    )
    .unwrap();
    assert!(feedback.ok);
    assert_eq!(feedback.output, json!({"value":"ok"}));
}

#[tokio::test]
async fn invalid_success_output_fails_closed_without_entering_transcript() {
    let mut tool = TestTool::read("read", json!({"type":"object"}))
        .returning(ToolResult::success(json!({"secret":"must-not-leak"})));
    tool.definition.output_schema = Some(json!({
        "type":"object",
        "required":["value"],
        "properties":{"value":{"type":"string"}},
        "additionalProperties":false
    }));
    let tool = Arc::new(tool);
    let provider = Arc::new(MockModelProvider::scripted([
        tool_response(call("invalid", "read", "{}")),
        final_response("recovered"),
    ]));
    let events = Arc::new(InMemoryEventSink::default());

    let result = AgentRunner::builder(provider.clone())
        .tools(registry(tool))
        .event_sink(events.clone())
        .build()
        .run(request())
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert!(result.errors.iter().any(|error| {
        error.code == "tool_error" && error.message.contains("output failed validation")
    }));
    assert!(result
        .errors
        .iter()
        .all(|error| !error.message.contains("must-not-leak")));
    let requests = provider.requests();
    let feedback_message = requests[1]
        .messages
        .iter()
        .find(|message| message.role == MessageRole::Tool)
        .unwrap();
    assert!(!feedback_message.content.contains("must-not-leak"));
    let feedback: ToolResult = serde_json::from_str(&feedback_message.content).unwrap();
    assert!(!feedback.ok);
    assert!(events.events().iter().any(|record| matches!(
        &record.event,
        RunEvent::ToolCompleted { call_id, ok, .. } if call_id == "invalid" && !ok
    )));
}

#[tokio::test]
async fn declared_failure_bypasses_success_output_schema() {
    let mut tool = TestTool::read("read", json!({"type":"object"}))
        .returning(ToolResult::failure("declared failure"));
    tool.definition.output_schema = Some(json!({"type":"string"}));
    let provider = Arc::new(MockModelProvider::scripted([
        tool_response(call("failure", "read", "{}")),
        final_response("done"),
    ]));

    let result = AgentRunner::builder(provider.clone())
        .tools(registry(Arc::new(tool)))
        .build()
        .run(request())
        .await
        .unwrap();

    assert_eq!(result.status, RunStatus::Completed);
    assert!(result
        .errors
        .iter()
        .any(|error| error.message == "declared failure"));
    let feedback: ToolResult = serde_json::from_str(
        &provider.requests()[1]
            .messages
            .iter()
            .find(|message| message.role == MessageRole::Tool)
            .unwrap()
            .content,
    )
    .unwrap();
    assert!(!feedback.ok);
    assert_eq!(feedback.error.as_deref(), Some("declared failure"));
}

#[tokio::test]
async fn exact_model_tool_and_repeat_call_limits_stop_before_the_extra_call() {
    let model_provider = Arc::new(MockModelProvider::scripted([
        tool_response(call("one", "read", "{}")),
        final_response("must not run"),
    ]));
    let mut model_request = request();
    model_request.agent.limits.max_model_calls = 1;
    let model_limited = AgentRunner::builder(model_provider.clone())
        .tools(registry(Arc::new(TestTool::read(
            "read",
            json!({"type": "object"}),
        ))))
        .build()
        .run(model_request)
        .await
        .unwrap();
    assert!(model_limited.model_call_limit_reached);
    assert_eq!(model_provider.requests().len(), 1);

    let tool = Arc::new(TestTool::read("read", json!({"type": "object"})));
    let mut tool_request = request();
    tool_request.agent.limits.max_tool_calls = 1;
    let tool_limited =
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([MockStep::Response(
            response(
                None,
                vec![call("one", "read", "{}"), call("two", "read", r#"{"x":1}"#)],
            ),
        )])))
        .tools(registry(tool.clone()))
        .build()
        .run(tool_request)
        .await
        .unwrap();
    assert!(tool_limited.tool_call_limit_reached);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);

    let repeated_tool = Arc::new(TestTool::read("read", json!({"type": "object"})));
    let mut repeated_request = request();
    repeated_request.agent.limits.max_identical_tool_calls = 1;
    let repeated =
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([MockStep::Response(
            response(
                None,
                vec![call("one", "read", "{}"), call("two", "read", "{}")],
            ),
        )])))
        .tools(registry(repeated_tool.clone()))
        .build()
        .run(repeated_request)
        .await
        .unwrap();
    assert!(repeated.repeated_tool_call_limit_reached);
    assert_eq!(repeated_tool.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn invalid_requests_provider_failures_and_empty_responses_are_deterministic() {
    let mut invalid = request();
    invalid.input.clear();
    assert!(matches!(
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
            "unused"
        )])))
        .build()
        .run(invalid)
        .await,
        Err(HarnessError::InvalidRequest(_))
    ));

    let provider_failure =
        AgentRunner::builder(Arc::new(MockModelProvider::scripted([MockStep::Error(
            HarnessError::Provider("offline".into()),
        )])))
        .build()
        .run(request())
        .await
        .unwrap();
    assert_eq!(provider_failure.status, RunStatus::Failed);
    assert_eq!(provider_failure.errors[0].code, "provider_error");

    for empty in [response(None, vec![]), response(Some(""), vec![])] {
        let result =
            AgentRunner::builder(Arc::new(MockModelProvider::scripted([MockStep::Response(
                empty,
            )])))
            .build()
            .run(request())
            .await
            .unwrap();
        assert_eq!(result.status, RunStatus::Failed);
        assert_eq!(result.errors[0].code, "empty_model_response");
    }
}

#[tokio::test]
async fn mock_provider_inventory_and_request_capture_remain_public() {
    let provider = Arc::new(MockModelProvider::scripted([final_response("done")]));
    assert_eq!(provider.list_models().await.unwrap()[0].id, "mock-model");
    AgentRunner::builder(provider.clone())
        .build()
        .run(request())
        .await
        .unwrap();
    assert_eq!(provider.requests().len(), 1);
}
