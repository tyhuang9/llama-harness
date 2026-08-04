use crate::tool::ToolResult;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            tool_call_id: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            tool_call_id: None,
        }
    }

    pub(crate) fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            tool_call_id: None,
        }
    }

    pub(crate) fn tool(call_id: String, result: &ToolResult) -> Result<Self, serde_json::Error> {
        Ok(Self {
            role: MessageRole::Tool,
            content: serde_json::to_string(result)?,
            tool_call_id: Some(call_id),
        })
    }

    pub(crate) fn transcript_bytes(&self) -> u64 {
        self.content.len() as u64
            + self
                .tool_call_id
                .as_ref()
                .map_or(0, |call_id| call_id.len() as u64)
    }
}
