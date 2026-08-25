use std::sync::{Arc, Mutex};

use llama_harness::{
    async_trait,
    serde_json::{json, Value},
    ApprovalHandler, ApprovalRecord, CancellationToken, EventRecord, EventSink, HarnessError,
    JsonValue, MessageRole, ModelCapabilities, ModelInfo, ModelProvider, ModelRequest,
    ModelResponse, PolicyDecision, PolicyEngine, ProviderHealth, RunRequest, Tool, ToolDefinition,
    ToolResult, ToolRisk,
};

struct CustomProvider;

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
        Ok(ModelResponse::new(request.model).with_final_output("complete"))
    }
}

struct CustomTool(ToolDefinition);

#[async_trait]
impl Tool for CustomTool {
    fn definition(&self) -> &ToolDefinition {
        &self.0
    }

    async fn execute(
        &self,
        arguments: JsonValue,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        if cancellation.is_cancelled() {
            return Err(HarnessError::Cancelled);
        }
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
        Ok(if tool.read_only {
            PolicyDecision::Allow {
                reason: "read-only".into(),
            }
        } else {
            PolicyDecision::RequireApproval {
                reason: "state-changing".into(),
            }
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
        Ok(ApprovalRecord::new("call", &tool.id, false, "not approved"))
    }
}

#[derive(Default)]
struct CustomEventSink(Mutex<Vec<EventRecord>>);

impl EventSink for CustomEventSink {
    fn emit(&self, record: EventRecord) {
        self.0.lock().expect("event sink lock").push(record);
    }
}

#[tokio::test]
async fn facade_contains_the_complete_extension_contract() {
    let provider: Arc<dyn ModelProvider> = Arc::new(CustomProvider);
    assert_eq!(provider.id(), "custom");
    assert!(provider.health().await.expect("provider health").healthy);

    let definition = ToolDefinition::new(
        "status.read",
        "Read status",
        "Reads status",
        json!({"type": "object", "additionalProperties": false}),
    )
    .with_risk(ToolRisk::Low)
    .with_idempotent(true)
    .with_read_only(true);
    let tool: Arc<dyn Tool> = Arc::new(CustomTool(definition.clone()));
    let output = tool
        .execute(json!({"scope": "local"}), CancellationToken::new())
        .await
        .expect("tool output");
    assert!(output.ok);

    let _: Arc<dyn PolicyEngine> = Arc::new(CustomPolicy);
    let _: Arc<dyn ApprovalHandler> = Arc::new(CustomApproval);
    let _: Arc<dyn EventSink> = Arc::new(CustomEventSink::default());
    let _: JsonValue = json!({"supported": true});
    let _: MessageRole = MessageRole::User;

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert!(cancellation.is_cancelled());
}
