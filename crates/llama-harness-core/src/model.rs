use crate::{GenerationOptions, HarnessError, JsonMap, Message, ToolDefinition};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
/// Request sent from the runner to a model provider.
pub struct ModelRequest {
    /// Model identifier requested for completion.
    pub model: String,
    /// Transcript messages sent to the model.
    pub messages: Vec<Message>,
    /// Tools available to the model for this request.
    pub tools: Vec<ToolDefinition>,
    /// Generation settings for this request.
    pub generation: GenerationOptions,
    #[serde(default)]
    /// Provider-specific request metadata.
    pub metadata: JsonMap,
    #[serde(skip)]
    /// Cooperative cancellation token for the request.
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
/// Response returned by a model provider.
pub struct ModelResponse {
    /// Model identifier that produced the response.
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Final assistant text, when present.
    pub final_output: Option<String>,
    #[serde(default)]
    /// Tool calls requested by the model.
    pub tool_calls: Vec<crate::ToolCall>,
    #[serde(default)]
    /// Token usage reported by the provider.
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
/// Token usage reported for a model call.
pub struct Usage {
    /// Number of input tokens.
    pub input_tokens: u64,
    /// Number of output tokens.
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
/// Capabilities advertised by a model provider.
pub struct ModelCapabilities {
    /// Whether the model can request tools.
    pub supports_tools: bool,
    /// Whether the provider supports streaming responses.
    pub supports_streaming: bool,
    /// Whether the model supports structured output.
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
/// Inventory metadata for one available model.
pub struct ModelInfo {
    /// Provider-specific model identifier.
    pub id: String,
    #[serde(default)]
    /// Capabilities reported for the model.
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
/// Health status reported by a model provider.
pub struct ProviderHealth {
    /// Whether the provider is currently healthy.
    pub healthy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional diagnostic detail about the health state.
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
/// Interface implemented by model backends used by the runner.
pub trait ModelProvider: Send + Sync {
    /// Returns the stable provider identifier.
    fn id(&self) -> &str;
    /// Returns capabilities shared by the provider's models.
    fn capabilities(&self) -> ModelCapabilities;
    /// Checks provider health.
    async fn health(&self) -> Result<ProviderHealth, HarnessError>;
    /// Lists models available from the provider.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError>;
    /// Completes one model request.
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError>;
}
