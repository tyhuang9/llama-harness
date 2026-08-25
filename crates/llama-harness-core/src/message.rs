use crate::tool::{ToolCall, ToolResult};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl Message {
    /// Creates a transcript message without tool correlation data.
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_call_id: None,
            tool_calls: vec![],
        }
    }

    /// Associates this message with the tool call it answers.
    pub fn with_tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }

    /// Associates model-requested tool calls with this message.
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageRole::User, content)
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageRole::System, content)
    }

    pub(crate) fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }

    pub(crate) fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self::new(MessageRole::Assistant, "").with_tool_calls(tool_calls)
    }

    pub(crate) fn tool(call_id: String, result: &ToolResult) -> Result<Self, serde_json::Error> {
        Ok(Self::new(MessageRole::Tool, serde_json::to_string(result)?).with_tool_call_id(call_id))
    }

    pub(crate) fn transcript_bytes(&self) -> u64 {
        self.content.len() as u64
            + self
                .tool_call_id
                .as_ref()
                .map_or(0, |call_id| call_id.len() as u64)
            + self
                .tool_calls
                .iter()
                .map(|call| (call.id.len() + call.tool_id.len() + call.arguments_json.len()) as u64)
                .sum::<u64>()
    }
}
