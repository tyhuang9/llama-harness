use crate::{GenerationOptions, HarnessError, JsonMap, Message, ToolDefinition};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
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

impl ModelRequest {
    /// Creates a provider request with empty messages and conservative generation defaults.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: Vec::new(),
            tools: Vec::new(),
            generation: GenerationOptions::default(),
            metadata: JsonMap::new(),
            cancellation: CancellationToken::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct ModelResponse {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_output: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<crate::ToolCall>,
    #[serde(default)]
    pub usage: Usage,
}

impl ModelResponse {
    /// Creates an empty provider response for the named model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            final_output: None,
            tool_calls: Vec::new(),
            usage: Usage::default(),
        }
    }

    /// Sets the provider's final assistant output.
    pub fn with_final_output(mut self, output: impl Into<String>) -> Self {
        self.final_output = Some(output.into());
        self
    }

    /// Sets tool calls requested by the model.
    pub fn with_tool_calls(mut self, tool_calls: Vec<crate::ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    /// Sets provider-reported token usage.
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = usage;
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl Usage {
    /// Creates provider-reported token usage.
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelCapabilities {
    pub supports_tools: bool,
    pub supports_streaming: bool,
    pub supports_structured_output: bool,
}

impl ModelCapabilities {
    /// Creates an explicit provider capability declaration.
    pub fn new(
        supports_tools: bool,
        supports_streaming: bool,
        supports_structured_output: bool,
    ) -> Self {
        Self {
            supports_tools,
            supports_streaming,
            supports_structured_output,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub capabilities: ModelCapabilities,
}

impl ModelInfo {
    /// Creates model inventory metadata with conservative capabilities.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            capabilities: ModelCapabilities::default(),
        }
    }

    /// Sets the model capabilities reported by its provider.
    pub fn with_capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProviderHealth {
    pub healthy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ProviderHealth {
    /// Creates provider health with an optional diagnostic detail.
    pub fn new(healthy: bool, detail: Option<String>) -> Self {
        Self { healthy, detail }
    }

    /// Creates a healthy provider status.
    pub fn healthy() -> Self {
        Self {
            healthy: true,
            detail: None,
        }
    }

    /// Creates an unhealthy provider status with a diagnostic detail.
    pub fn unhealthy(detail: impl Into<String>) -> Self {
        Self {
            healthy: false,
            detail: Some(detail.into()),
        }
    }
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn id(&self) -> &str;
    fn capabilities(&self) -> ModelCapabilities;
    async fn health(&self) -> Result<ProviderHealth, HarnessError>;
    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError>;
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError>;
}
