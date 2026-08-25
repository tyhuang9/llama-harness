//! Compile-only consumer used to validate the published facade.

use std::sync::{Arc, Mutex};

use llama_harness::{
    async_trait, serde_json::json, AgentRunner, ApprovalHandler, ApprovalRecord, CancellationToken,
    EventRecord, EventSink, HarnessError, JsonValue, ModelCapabilities, ModelInfo, ModelProvider,
    ModelRequest, ModelResponse, PolicyDecision, PolicyEngine, ProviderHealth, RunRequest, Tool,
    ToolDefinition, ToolRegistry, ToolResult,
};

/// Application-defined provider implemented through facade exports only.
pub struct ConsumerProvider;

#[async_trait]
impl ModelProvider for ConsumerProvider {
    fn id(&self) -> &str {
        "packaged-consumer"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::new(true, false, true)
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth::healthy())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(vec![ModelInfo::new("consumer-model")])
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError> {
        if request.cancellation.is_cancelled() {
            return Err(HarnessError::Cancelled);
        }
        Ok(ModelResponse::new(request.model).with_final_output("consumer response"))
    }
}

/// Application-defined tool implemented through facade exports only.
pub struct ConsumerTool {
    definition: ToolDefinition,
}

impl ConsumerTool {
    /// Creates the compile-test tool with a closed argument schema.
    pub fn new() -> Self {
        Self {
            definition: ToolDefinition::new(
                "consumer.status",
                "Consumer status",
                "Read packaged consumer status",
                json!({"type": "object", "additionalProperties": false}),
            ),
        }
    }
}

impl Default for ConsumerTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ConsumerTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(
        &self,
        _arguments: JsonValue,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        if cancellation.is_cancelled() {
            return Err(HarnessError::Cancelled);
        }
        Ok(ToolResult::success(json!({"status": "ready"})))
    }
}

/// Application-defined policy implemented through facade exports only.
pub struct ConsumerPolicy;

#[async_trait]
impl PolicyEngine for ConsumerPolicy {
    async fn decide(
        &self,
        _tool: &ToolDefinition,
        _arguments: &JsonValue,
        _request: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(PolicyDecision::Allow {
            reason: "fixture permits its read-only tool".into(),
        })
    }
}

/// Application-defined approval handler implemented through facade exports only.
pub struct ConsumerApprovals;

#[async_trait]
impl ApprovalHandler for ConsumerApprovals {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        _arguments: &JsonValue,
        _request: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        Ok(ApprovalRecord::new(
            "fixture-call",
            tool.id.clone(),
            true,
            "fixture approval",
        ))
    }
}

/// Application-defined event sink implemented through facade exports only.
#[derive(Default)]
pub struct ConsumerEvents {
    records: Mutex<Vec<EventRecord>>,
}

impl ConsumerEvents {
    /// Returns the number of records observed by the sink.
    pub fn len(&self) -> usize {
        self.records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Returns whether no records have been observed.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl EventSink for ConsumerEvents {
    fn emit(&self, record: EventRecord) {
        self.records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(record);
    }
}

/// Builds a runner wired entirely through types exported by the facade.
pub fn configured_runner() -> Result<(AgentRunner, Arc<ConsumerEvents>), HarnessError> {
    let mut tools = ToolRegistry::default();
    tools.register(Arc::new(ConsumerTool::new()))?;
    let events = Arc::new(ConsumerEvents::default());
    let runner = AgentRunner::builder(Arc::new(ConsumerProvider))
        .tools(tools)
        .policy(Arc::new(ConsumerPolicy))
        .approvals(Arc::new(ConsumerApprovals))
        .event_sink(events.clone())
        .build();
    Ok((runner, events))
}

/// Exercises facade-provided cooperative cancellation without a core dependency.
pub fn cancelled_token() -> CancellationToken {
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert!(cancellation.is_cancelled());
    cancellation
}

#[cfg(feature = "ollama")]
pub use llama_harness::ollama;

#[cfg(feature = "observability")]
pub use llama_harness::observability;

#[cfg(feature = "evals")]
pub use llama_harness::evals;

#[cfg(feature = "tauri")]
pub use llama_harness::tauri;
