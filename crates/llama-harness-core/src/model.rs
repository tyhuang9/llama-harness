use crate::{GenerationOptions, HarnessError, JsonMap, Message, ModelEventStream, ToolDefinition};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
/// Immutable tool definitions and their exact provider-consumable JSON.
pub struct PreparedToolCatalog {
    definitions: Arc<[ToolDefinition]>,
    serialized_definitions: Arc<[u8]>,
    provider_tools: Box<RawValue>,
}

impl PartialEq for PreparedToolCatalog {
    fn eq(&self, other: &Self) -> bool {
        self.definitions == other.definitions
            && self.serialized_definitions == other.serialized_definitions
            && self.provider_tools.get() == other.provider_tools.get()
    }
}

impl PreparedToolCatalog {
    /// Prepares an immutable catalog for hosts constructing model requests directly.
    pub fn from_definitions(definitions: Vec<ToolDefinition>) -> Result<Self, HarnessError> {
        #[derive(Serialize)]
        struct ProviderTool<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            function: ProviderFunction<'a>,
        }
        #[derive(Serialize)]
        struct ProviderFunction<'a> {
            name: &'a str,
            description: &'a str,
            parameters: &'a serde_json::Value,
        }
        let serialized_definitions = serde_json::to_vec(&definitions)
            .map(Arc::<[u8]>::from)
            .map_err(|error| HarnessError::InvalidRequest(error.to_string()))?;
        let provider_tools = definitions
            .iter()
            .map(|definition| ProviderTool {
                kind: "function",
                function: ProviderFunction {
                    name: &definition.id,
                    description: &definition.description,
                    parameters: &definition.arguments_schema,
                },
            })
            .collect::<Vec<_>>();
        let provider_tools = RawValue::from_string(
            serde_json::to_string(&provider_tools)
                .map_err(|error| HarnessError::InvalidRequest(error.to_string()))?,
        )
        .map_err(|error| HarnessError::InvalidRequest(error.to_string()))?;
        Ok(Self::new(
            Arc::from(definitions),
            serialized_definitions,
            provider_tools,
        ))
    }

    pub(crate) fn new(
        definitions: Arc<[ToolDefinition]>,
        serialized_definitions: Arc<[u8]>,
        provider_tools: Box<RawValue>,
    ) -> Self {
        Self {
            definitions,
            serialized_definitions,
            provider_tools,
        }
    }

    /// Returns the selected definitions in provider order.
    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    /// Returns the exact serialized `ToolDefinition` array used for discovery budgets.
    pub fn serialized_definitions(&self) -> &[u8] {
        &self.serialized_definitions
    }

    /// Returns exact JSON for a standard function-tool array without reserializing schemas.
    pub fn provider_tools_json(&self) -> &RawValue {
        &self.provider_tools
    }
}

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
    #[serde(skip)]
    /// Immutable prepared form of `tools`, when supplied by the core runner.
    pub prepared_tools: Option<Arc<PreparedToolCatalog>>,
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
            prepared_tools: None,
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
    #[serde(default)]
    /// Whether the provider can constrain tool arguments with strict schemas.
    pub supports_strict_tool_schemas: bool,
    #[serde(default)]
    /// Whether tool-call arguments may arrive incrementally while streaming.
    pub supports_streaming_tool_arguments: bool,
    #[serde(default)]
    /// Whether the provider can return multiple tool calls in one response.
    pub supports_parallel_tool_calls: bool,
    #[serde(default)]
    /// Whether the provider can generate provider-native structured plans.
    pub supports_structured_plans: bool,
    #[serde(default)]
    /// Whether the provider can generate programmatic tool workflows.
    pub supports_programmatic_calling: bool,
    #[serde(default)]
    /// Provider-advertised resource limits for advanced calling features.
    pub limits: ProviderCapabilityLimits,
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
            supports_strict_tool_schemas: false,
            supports_streaming_tool_arguments: false,
            supports_parallel_tool_calls: false,
            supports_structured_plans: false,
            supports_programmatic_calling: false,
            limits: ProviderCapabilityLimits::default(),
        }
    }

    /// Declares support for strict tool schemas.
    pub fn with_strict_tool_schemas(mut self, supported: bool) -> Self {
        self.supports_strict_tool_schemas = supported;
        self
    }

    /// Declares support for incrementally streamed tool arguments.
    pub fn with_streaming_tool_arguments(mut self, supported: bool) -> Self {
        self.supports_streaming_tool_arguments = supported;
        self
    }

    /// Declares support for multiple tool calls in one model response.
    pub fn with_parallel_tool_calls(mut self, supported: bool) -> Self {
        self.supports_parallel_tool_calls = supported;
        self
    }

    /// Declares support for provider-native structured plans.
    pub fn with_structured_plans(mut self, supported: bool) -> Self {
        self.supports_structured_plans = supported;
        self
    }

    /// Declares support for programmatic tool workflows.
    pub fn with_programmatic_calling(mut self, supported: bool) -> Self {
        self.supports_programmatic_calling = supported;
        self
    }

    /// Sets resource limits advertised by the provider.
    pub fn with_limits(mut self, limits: ProviderCapabilityLimits) -> Self {
        self.limits = limits;
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(default)]
#[non_exhaustive]
/// Optional provider limits for advanced tool-calling capabilities.
pub struct ProviderCapabilityLimits {
    /// Maximum number of tools accepted in one request.
    pub max_tools: Option<u32>,
    /// Maximum total serialized tool-schema bytes accepted in one request.
    pub max_tool_schema_bytes: Option<u64>,
    /// Maximum parallel tool calls returned in one response.
    pub max_parallel_tool_calls: Option<u32>,
    /// Maximum streamed tool-argument bytes returned in one call.
    pub max_streamed_argument_bytes: Option<u64>,
    /// Maximum streamed tool calls returned in one response.
    pub max_streamed_tool_calls: Option<u32>,
    /// Maximum serialized structured-plan bytes.
    pub max_plan_bytes: Option<u64>,
    /// Maximum nodes in a structured plan.
    pub max_plan_nodes: Option<u32>,
    /// Maximum generated program bytes.
    pub max_program_bytes: Option<u64>,
}

impl ProviderCapabilityLimits {
    /// Creates an empty provider-limit declaration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of tools accepted in one request.
    pub fn with_max_tools(mut self, max_tools: u32) -> Self {
        self.max_tools = Some(max_tools);
        self
    }

    /// Sets the maximum total serialized tool-schema bytes in one request.
    pub fn with_max_tool_schema_bytes(mut self, max_tool_schema_bytes: u64) -> Self {
        self.max_tool_schema_bytes = Some(max_tool_schema_bytes);
        self
    }

    /// Sets the maximum parallel tool calls returned in one response.
    pub fn with_max_parallel_tool_calls(mut self, max_parallel_tool_calls: u32) -> Self {
        self.max_parallel_tool_calls = Some(max_parallel_tool_calls);
        self
    }

    /// Sets the maximum streamed argument bytes returned in one tool call.
    pub fn with_max_streamed_argument_bytes(mut self, max_streamed_argument_bytes: u64) -> Self {
        self.max_streamed_argument_bytes = Some(max_streamed_argument_bytes);
        self
    }

    /// Sets the maximum streamed tool calls returned in one response.
    pub fn with_max_streamed_tool_calls(mut self, max_streamed_tool_calls: u32) -> Self {
        self.max_streamed_tool_calls = Some(max_streamed_tool_calls);
        self
    }

    /// Sets the maximum serialized structured-plan bytes.
    pub fn with_max_plan_bytes(mut self, max_plan_bytes: u64) -> Self {
        self.max_plan_bytes = Some(max_plan_bytes);
        self
    }

    /// Sets the maximum nodes in a structured plan.
    pub fn with_max_plan_nodes(mut self, max_plan_nodes: u32) -> Self {
        self.max_plan_nodes = Some(max_plan_nodes);
        self
    }

    /// Sets the maximum generated program bytes.
    pub fn with_max_program_bytes(mut self, max_program_bytes: u64) -> Self {
        self.max_program_bytes = Some(max_program_bytes);
        self
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
    /// Streams one model request when the provider implements streaming.
    async fn stream(&self, _: ModelRequest) -> Result<ModelEventStream, HarnessError> {
        Err(HarnessError::UnsupportedCapability(format!(
            "provider {} does not support streaming",
            self.id()
        )))
    }
}
