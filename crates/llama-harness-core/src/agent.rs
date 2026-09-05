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

/// A JSON object map used for application-defined metadata and context.
pub type JsonMap = serde_json::Map<String, Value>;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
/// Strategy identifier used by planning and evaluation contracts.
pub enum RunStrategy {
    /// Evaluates adaptive strategy selection.
    #[default]
    Adaptive,
    /// Evaluates direct model-and-tool execution.
    Direct,
    /// Evaluates declarative plan execution.
    DeclarativePlan,
    /// Evaluates programmatic orchestration.
    Programmatic,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Describes an agent and its runtime defaults.
pub struct AgentDefinition {
    /// Stable identifier for the agent.
    pub id: String,
    /// Human-readable agent name.
    pub name: String,
    /// Application-defined agent version.
    pub version: String,
    #[serde(default)]
    /// System instructions supplied to the model.
    pub system_instructions: String,
    /// Default model identifier used for runs.
    pub default_model: String,
    #[serde(default)]
    /// Tool IDs the agent may use.
    pub tool_allowlist: Vec<String>,
    #[serde(default)]
    /// Resource and payload limits for runs.
    pub limits: AgentLimits,
    #[serde(default)]
    /// Default generation settings for model calls.
    pub generation: GenerationOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional JSON schema for validating final output.
    pub output_schema: Option<Value>,
    #[serde(default)]
    /// Application-defined agent metadata.
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
/// Per-run settings that override an agent's defaults.
pub struct RunOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional model identifier for this run.
    pub model: Option<String>,
    #[serde(default)]
    /// Generation settings for this run.
    pub generation: GenerationOptions,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Input and host context for one agent run.
pub struct RunRequest {
    /// Agent definition to execute.
    pub agent: AgentDefinition,
    /// User or application input presented to the agent.
    pub input: String,
    #[serde(default)]
    /// Structured context supplied by the host application.
    pub application_context: JsonMap,
    #[serde(default)]
    /// Previous transcript messages supplied to the run.
    pub history: Vec<Message>,
    #[serde(default)]
    /// Application-defined request metadata.
    pub metadata: JsonMap,
    #[serde(default)]
    /// Per-run model and generation overrides.
    pub overrides: RunOverrides,
    #[serde(default)]
    /// Evaluation metadata associated with the run.
    pub evaluation: JsonMap,
    #[serde(skip, default = "CancellationToken::new")]
    /// Cooperative cancellation token for the run.
    pub cancellation: CancellationToken,
    #[serde(skip)]
    /// Optional externally supplied run identifier.
    pub run_id: Option<String>,
    #[serde(skip)]
    /// Optional externally supplied trace identifier.
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
/// Terminal state reported for an agent run.
pub enum RunStatus {
    /// The run completed successfully.
    Completed,
    /// The run failed after starting.
    Failed,
    /// The run was cancelled.
    Cancelled,
    /// A configured resource limit stopped the run.
    LimitReached,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
/// Result and telemetry captured for one agent run.
pub struct RunResult {
    /// Run identifier used for correlation.
    pub id: String,
    /// Terminal status of the run.
    pub status: RunStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Final assistant output, when one was produced.
    pub final_output: Option<String>,
    /// Model identifier selected for the run.
    pub model: String,
    #[serde(default)]
    /// Tool calls requested by the model.
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    /// Policy decisions recorded during the run.
    pub policy_decisions: Vec<PolicyDecision>,
    #[serde(default)]
    /// Approval records recorded during the run.
    pub approvals: Vec<ApprovalRecord>,
    #[serde(default)]
    /// Errors captured during execution.
    pub errors: Vec<RunError>,
    /// Elapsed run duration in milliseconds.
    pub duration_ms: u64,
    /// Trace identifier used for correlation.
    pub trace_id: String,
    /// Whether the model-call limit stopped the run.
    pub model_call_limit_reached: bool,
    /// Whether the tool-call limit stopped the run.
    pub tool_call_limit_reached: bool,
    /// Whether the repeated-identical-tool-call limit stopped the run.
    pub repeated_tool_call_limit_reached: bool,
    /// Whether cancellation stopped the run.
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
