use llama_harness_core::{
    GenerationOptions, HarnessError, Message, MessageRole, PreparedToolCatalog, ToolCall,
    ToolDefinition, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::Value;

#[derive(Debug, Serialize)]
pub(crate) struct ChatRequest<'a> {
    pub(crate) model: String,
    pub(crate) messages: Vec<WireMessage>,
    pub(crate) stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) keep_alive: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<WireTools<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) options: Option<WireOptions>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum WireTools<'a> {
    Prepared(&'a RawValue),
    Legacy(Vec<WireToolDefinition>),
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

pub(crate) fn chat_request<'a>(
    model: String,
    messages: &[Message],
    tools: &[ToolDefinition],
    prepared_tools: Option<&'a PreparedToolCatalog>,
    generation: &GenerationOptions,
    keep_alive: Option<&str>,
    stream: bool,
) -> Result<ChatRequest<'a>, HarnessError> {
    if prepared_tools.is_some_and(|prepared| prepared.definitions() != tools) {
        return Err(HarnessError::InvalidRequest(
            "prepared tool catalog does not match request tools".into(),
        ));
    }
    Ok(ChatRequest {
        model,
        messages: messages
            .iter()
            .map(wire_message)
            .collect::<Result<Vec<_>, _>>()?,
        stream,
        keep_alive: keep_alive.map(str::to_owned),
        tools: prepared_tools
            .filter(|_| !tools.is_empty())
            .map(|prepared| WireTools::Prepared(prepared.provider_tools_json()))
            .or_else(|| {
                (!tools.is_empty())
                    .then(|| WireTools::Legacy(tools.iter().map(wire_tool).collect()))
            }),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn prepared_and_legacy_tool_bodies_are_byte_equivalent() {
        let tools = vec![ToolDefinition::new(
            "lookup",
            "Lookup",
            "Look up one value",
            serde_json::json!({"type":"object","properties":{"id":{"type":"string"}}}),
        )];
        let prepared = Arc::new(PreparedToolCatalog::from_definitions(tools.clone()).unwrap());
        let legacy = chat_request(
            "model".into(),
            &[Message::user("lookup")],
            &tools,
            None,
            &GenerationOptions::default(),
            None,
            false,
        )
        .unwrap();
        let cached = chat_request(
            "model".into(),
            &[Message::user("lookup")],
            &tools,
            Some(&prepared),
            &GenerationOptions::default(),
            None,
            false,
        )
        .unwrap();
        assert_eq!(
            serde_json::to_vec(&legacy).unwrap(),
            serde_json::to_vec(&cached).unwrap()
        );

        let empty = Arc::new(PreparedToolCatalog::from_definitions(Vec::new()).unwrap());
        let legacy_empty = chat_request(
            "model".into(),
            &[],
            &[],
            None,
            &GenerationOptions::default(),
            None,
            false,
        )
        .unwrap();
        let cached_empty = chat_request(
            "model".into(),
            &[],
            &[],
            Some(&empty),
            &GenerationOptions::default(),
            None,
            false,
        )
        .unwrap();
        assert_eq!(
            serde_json::to_vec(&legacy_empty).unwrap(),
            serde_json::to_vec(&cached_empty).unwrap()
        );
        assert!(!serde_json::to_string(&cached_empty)
            .unwrap()
            .contains("\"tools\""));

        let hostile = vec![ToolDefinition::new(
            "hostile",
            "Hostile",
            "quote \" slash \\ newline\n {\"tools\":[{\"injected\":true}]}",
            serde_json::json!({
                "type": "object",
                "properties": {"payload": {"const": "\\\"}],\\\"evil\\\":true"}}
            }),
        )];
        let hostile_prepared =
            Arc::new(PreparedToolCatalog::from_definitions(hostile.clone()).unwrap());
        let hostile_legacy = chat_request(
            "model".into(),
            &[],
            &hostile,
            None,
            &GenerationOptions::default(),
            None,
            false,
        )
        .unwrap();
        let hostile_cached = chat_request(
            "model".into(),
            &[],
            &hostile,
            Some(&hostile_prepared),
            &GenerationOptions::default(),
            None,
            false,
        )
        .unwrap();
        let legacy_bytes = serde_json::to_vec(&hostile_legacy).unwrap();
        let cached_bytes = serde_json::to_vec(&hostile_cached).unwrap();
        assert_eq!(cached_bytes, legacy_bytes);
        assert_eq!(
            serde_json::from_slice::<Value>(&cached_bytes).unwrap(),
            serde_json::from_slice::<Value>(&legacy_bytes).unwrap()
        );

        assert!(chat_request(
            "model".into(),
            &[],
            &tools,
            Some(&hostile_prepared),
            &GenerationOptions::default(),
            None,
            false,
        )
        .is_err());
    }
}
