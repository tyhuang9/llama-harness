use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use llama_harness::{
    async_trait,
    serde_json::{json, Value},
    AgentDefinition, AgentRunner, ApprovalHandler, ApprovalRecord, CancellationToken, EventRecord,
    EventSink, HarnessError, JsonValue, MessageRole, ModelCapabilities, ModelInfo, ModelProvider,
    ModelRequest, ModelResponse, PolicyDecision, PolicyEngine, ProviderHealth, RunEvent,
    RunRequest, RunStatus, Tool, ToolCall, ToolDefinition, ToolRegistry, ToolResult, ToolRisk,
};
use tokio::sync::Notify;

#[derive(Default)]
struct CustomProvider {
    calls: AtomicUsize,
}

#[async_trait]
impl ModelProvider for CustomProvider {
    fn id(&self) -> &str {
        "custom"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::new(true, false, true)
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth::healthy())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(vec![ModelInfo::new("custom-model")])
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError> {
        if request.cancellation.is_cancelled() {
            return Err(HarnessError::Cancelled);
        }
        Ok(if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            ModelResponse::new(request.model).with_tool_calls(vec![ToolCall::new(
                "call-1",
                "status.read",
                r#"{"scope":"local"}"#,
            )])
        } else {
            ModelResponse::new(request.model).with_final_output("complete")
        })
    }
}

struct CustomTool {
    definition: ToolDefinition,
    executions: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for CustomTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(
        &self,
        arguments: JsonValue,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        if cancellation.is_cancelled() {
            return Err(HarnessError::Cancelled);
        }
        self.executions.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success(arguments))
    }
}

struct CustomPolicy;

#[async_trait]
impl PolicyEngine for CustomPolicy {
    async fn decide(
        &self,
        tool: &ToolDefinition,
        _arguments: &Value,
        _request: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(PolicyDecision::RequireApproval {
            reason: format!("application approval for {}", tool.id),
        })
    }
}

struct CustomApproval;

#[async_trait]
impl ApprovalHandler for CustomApproval {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        _arguments: &Value,
        _request: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        Ok(ApprovalRecord::new("call-1", &tool.id, true, "approved"))
    }
}

#[derive(Default)]
struct CustomEventSink(Mutex<Vec<EventRecord>>);

impl EventSink for CustomEventSink {
    fn emit(&self, record: EventRecord) {
        self.0.lock().expect("event sink lock").push(record);
    }
}

struct BlockingProvider {
    started: Arc<Notify>,
}

#[async_trait]
impl ModelProvider for BlockingProvider {
    fn id(&self) -> &str {
        "blocking"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::default()
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth::healthy())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(vec![ModelInfo::new("blocking-model")])
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError> {
        self.started.notify_one();
        request.cancellation.cancelled().await;
        Err(HarnessError::Cancelled)
    }
}

#[tokio::test]
async fn facade_wires_the_complete_extension_contract() {
    let provider: Arc<dyn ModelProvider> = Arc::new(CustomProvider::default());
    assert_eq!(provider.id(), "custom");
    assert!(provider.health().await.expect("provider health").healthy);

    let definition = ToolDefinition::new(
        "status.read",
        "Read status",
        "Reads status",
        json!({
            "type": "object",
            "properties": {"scope": {"type": "string"}},
            "required": ["scope"],
            "additionalProperties": false
        }),
    )
    .with_risk(ToolRisk::Low)
    .with_idempotent(true)
    .with_read_only(true);
    let executions = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::default();
    tools
        .register(Arc::new(CustomTool {
            definition,
            executions: Arc::clone(&executions),
        }))
        .expect("register custom tool");
    let events = Arc::new(CustomEventSink::default());
    let runner = AgentRunner::builder(provider)
        .tools(tools)
        .policy(Arc::new(CustomPolicy))
        .approvals(Arc::new(CustomApproval))
        .event_sink(Arc::clone(&events) as Arc<dyn EventSink>)
        .build();
    let mut agent = AgentDefinition::new("custom", "Custom", "1", "custom-model");
    agent.tool_allowlist.push("status.read".into());
    let result = runner
        .run(RunRequest::new(agent, "Read status"))
        .await
        .expect("facade run");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.final_output.as_deref(), Some("complete"));
    assert_eq!(executions.load(Ordering::SeqCst), 1);
    assert_eq!(result.approvals.len(), 1);
    assert!(result.approvals[0].granted);
    assert!(events
        .0
        .lock()
        .expect("event sink lock")
        .iter()
        .any(|record| matches!(record.event, RunEvent::ToolCompleted { ok: true, .. })));

    let _: JsonValue = json!({"supported": true});
    let _: MessageRole = MessageRole::User;
}

#[tokio::test]
async fn facade_propagates_cancellation_to_a_custom_provider() {
    let started = Arc::new(Notify::new());
    let events = Arc::new(CustomEventSink::default());
    let runner = AgentRunner::builder(Arc::new(BlockingProvider {
        started: Arc::clone(&started),
    }))
    .event_sink(Arc::clone(&events) as Arc<dyn EventSink>)
    .build();
    let cancellation = CancellationToken::new();
    let request = RunRequest::new(
        AgentDefinition::new("blocking", "Blocking", "1", "blocking-model"),
        "Wait",
    );
    let mut request = request;
    request.cancellation = cancellation.clone();
    let running = tokio::spawn(async move { runner.run(request).await });
    started.notified().await;
    cancellation.cancel();
    let result = running.await.expect("runner task").expect("cancelled run");

    assert_eq!(result.status, RunStatus::Cancelled);
    assert!(result.cancelled);
    assert!(events
        .0
        .lock()
        .expect("event sink lock")
        .iter()
        .any(|record| matches!(
            record.event,
            RunEvent::Completed {
                status: RunStatus::Cancelled
            }
        )));
}
