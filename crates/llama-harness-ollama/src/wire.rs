use llama_harness_core::{
    GenerationOptions, HarnessError, Message, MessageRole, ToolCall, ToolDefinition, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize)]
pub(crate) struct ChatRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<WireMessage>,
    pub(crate) stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) keep_alive: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<WireToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) options: Option<WireOptions>,
}

#[derive(Debug, Serialize)]
pub(crate) struct WireMessage {
    pub(crate) role: &'static str,
    pub(crate) content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) tool_calls: Vec<WireToolCall>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct WireToolCall {
    pub(crate) function: WireFunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct WireFunctionCall {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) arguments: Value,
}

#[derive(Debug, Serialize)]
pub(crate) struct WireToolDefinition {
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) function: WireToolFunction,
}

#[derive(Debug, Serialize)]
pub(crate) struct WireToolFunction {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: Value,
}

#[derive(Debug, Serialize)]
pub(crate) struct WireOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) num_predict: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatResponse {
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) message: Option<WireResponseMessage>,
    #[serde(default)]
    pub(crate) done: bool,
    pub(crate) prompt_eval_count: Option<u64>,
    pub(crate) eval_count: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct WireResponseMessage {
    #[serde(default)]
    pub(crate) content: String,
    #[serde(default)]
    pub(crate) tool_calls: Vec<WireToolCall>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TagsResponse {
    #[serde(default)]
    pub(crate) models: Vec<TagModel>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TagModel {
    pub(crate) name: String,
}

pub(crate) fn chat_request(
    model: String,
    messages: &[Message],
    tools: &[ToolDefinition],
    generation: &GenerationOptions,
    keep_alive: Option<&str>,
    stream: bool,
) -> Result<ChatRequest, HarnessError> {
    Ok(ChatRequest {
        model,
        messages: messages
            .iter()
            .map(wire_message)
            .collect::<Result<Vec<_>, _>>()?,
        stream,
        keep_alive: keep_alive.map(str::to_owned),
        tools: (!tools.is_empty()).then(|| tools.iter().map(wire_tool).collect()),
        options: options(generation),
    })
}

fn wire_message(message: &Message) -> Result<WireMessage, HarnessError> {
    let role = match message.role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
        _ => {
            return Err(HarnessError::InvalidRequest(
                "unsupported message role for the Ollama provider".into(),
            ))
        }
    };
    Ok(WireMessage {
        role,
        content: message.content.clone(),
        tool_calls: message
            .tool_calls
            .iter()
            .map(|call| WireToolCall {
                function: WireFunctionCall {
                    name: call.tool_id.clone(),
                    // Preserve malformed arguments as a string so the model can receive the
                    // tool-result feedback and repair its own previous proposal. Dropping the
                    // assistant turn would make the recovery conversation incoherent.
                    arguments: serde_json::from_str(&call.arguments_json)
                        .unwrap_or_else(|_| Value::String(call.arguments_json.clone())),
                },
            })
            .collect(),
    })
}

fn wire_tool(tool: &ToolDefinition) -> WireToolDefinition {
    WireToolDefinition {
        kind: "function",
        function: WireToolFunction {
            name: tool.id.clone(),
            description: tool.description.clone(),
            parameters: tool.arguments_schema.clone(),
        },
    }
}

fn options(generation: &GenerationOptions) -> Option<WireOptions> {
    (generation.temperature.is_some()
        || generation.top_p.is_some()
        || generation.max_output_tokens.is_some())
    .then_some(WireOptions {
        temperature: generation.temperature,
        top_p: generation.top_p,
        num_predict: generation.max_output_tokens,
    })
}

pub(crate) fn tool_calls(calls: &[WireToolCall]) -> Vec<ToolCall> {
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            ToolCall::new(
                format!("ollama-{index}"),
                call.function.name.clone(),
                match &call.function.arguments {
                    Value::String(arguments) => arguments.clone(),
                    value => serde_json::to_string(value).unwrap_or_else(|_| "null".into()),
                },
            )
        })
        .collect()
}

pub(crate) fn usage(response: &ChatResponse) -> Usage {
    Usage::new(
        response.prompt_eval_count.unwrap_or_default(),
        response.eval_count.unwrap_or_default(),
    )
}
