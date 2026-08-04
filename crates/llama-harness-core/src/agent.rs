use crate::{
    limits::{AgentLimits, GenerationOptions},
    message::Message,
    policy::{ApprovalRecord, PolicyDecision},
    tool::ToolCall,
    RunError,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

pub type JsonMap = serde_json::Map<String, Value>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentDefinition {
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub system_instructions: String,
    pub default_model: String,
    #[serde(default)]
    pub tool_allowlist: Vec<String>,
    #[serde(default)]
    pub limits: AgentLimits,
    #[serde(default)]
    pub generation: GenerationOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    #[serde(default)]
    pub metadata: JsonMap,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct RunOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub generation: GenerationOptions,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunRequest {
    pub agent: AgentDefinition,
    pub input: String,
    #[serde(default)]
    pub application_context: JsonMap,
    #[serde(default)]
    pub history: Vec<Message>,
    #[serde(default)]
    pub metadata: JsonMap,
    #[serde(default)]
    pub overrides: RunOverrides,
    #[serde(default)]
    pub evaluation: JsonMap,
    #[serde(skip, default = "CancellationToken::new")]
    pub cancellation: CancellationToken,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Completed,
    Failed,
    Cancelled,
    LimitReached,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunResult {
    pub id: String,
    pub status: RunStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_output: Option<String>,
    pub model: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub policy_decisions: Vec<PolicyDecision>,
    #[serde(default)]
    pub approvals: Vec<ApprovalRecord>,
    #[serde(default)]
    pub errors: Vec<RunError>,
    pub duration_ms: u64,
    pub trace_id: String,
    pub model_call_limit_reached: bool,
    pub tool_call_limit_reached: bool,
    pub repeated_tool_call_limit_reached: bool,
    pub cancelled: bool,
}
