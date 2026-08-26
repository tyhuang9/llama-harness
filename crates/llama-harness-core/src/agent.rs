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

impl AgentDefinition {
    /// Creates a minimal agent definition with the conservative runtime defaults.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        version: impl Into<String>,
        default_model: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            version: version.into(),
            system_instructions: String::new(),
            default_model: default_model.into(),
            tool_allowlist: vec![],
            limits: AgentLimits::default(),
            generation: GenerationOptions::default(),
            output_schema: None,
            metadata: JsonMap::new(),
        }
    }
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
    #[serde(skip)]
    pub run_id: Option<String>,
    #[serde(skip)]
    pub trace_id: Option<String>,
}

impl RunRequest {
    /// Creates a run request with empty host-provided context and a fresh cancellation token.
    pub fn new(agent: AgentDefinition, input: impl Into<String>) -> Self {
        Self {
            agent,
            input: input.into(),
            application_context: JsonMap::new(),
            history: vec![],
            metadata: JsonMap::new(),
            overrides: RunOverrides::default(),
            evaluation: JsonMap::new(),
            cancellation: CancellationToken::new(),
            run_id: None,
            trace_id: None,
        }
    }

    /// Sets a host-generated run ID, used by child-process adapters to correlate
    /// their lifecycle with the canonical embedded run.
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Sets a host-generated trace ID when a transport needs a stable external
    /// correlation value for the canonical run.
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RunStatus {
    Completed,
    Failed,
    Cancelled,
    LimitReached,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
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

impl RunResult {
    /// Creates an empty run result that adapters and evaluation fixtures can populate.
    pub fn new(
        id: impl Into<String>,
        status: RunStatus,
        model: impl Into<String>,
        trace_id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            status,
            final_output: None,
            model: model.into(),
            tool_calls: Vec::new(),
            policy_decisions: Vec::new(),
            approvals: Vec::new(),
            errors: Vec::new(),
            duration_ms: 0,
            trace_id: trace_id.into(),
            model_call_limit_reached: false,
            tool_call_limit_reached: false,
            repeated_tool_call_limit_reached: false,
            cancelled: false,
        }
    }
}
