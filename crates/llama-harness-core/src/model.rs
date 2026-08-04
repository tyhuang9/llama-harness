use crate::{GenerationOptions, HarnessError, JsonMap, Message, ToolDefinition};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub generation: GenerationOptions,
    #[serde(default)]
    pub metadata: JsonMap,
    #[serde(skip)]
    pub cancellation: CancellationToken,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelResponse {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_output: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<crate::ToolCall>,
    #[serde(default)]
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub supports_tools: bool,
    pub supports_streaming: bool,
    pub supports_structured_output: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub capabilities: ModelCapabilities,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderHealth {
    pub healthy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn id(&self) -> &str;
    fn capabilities(&self) -> ModelCapabilities;
    async fn health(&self) -> Result<ProviderHealth, HarnessError>;
    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError>;
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError>;
}
