use std::{collections::BTreeSet, fmt, str::FromStr};

use schemars::JsonSchema;
use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

pub const PROTOCOL_MAJOR: u16 = 1;
pub const PROTOCOL_MINOR: u16 = 0;
pub const MAX_MESSAGE_BYTES: usize = 256 * 1024;
pub const MAX_JSON_DEPTH: u32 = 64;
pub const MAX_PENDING_CALLBACKS: u16 = 128;
pub const MAX_CONCURRENT_RUNS: u16 = 16;
pub const MAX_QUEUE_DEPTH: u16 = 256;
pub const MAX_IDENTIFIER_BYTES: usize = 256;
pub const MAX_ERROR_MESSAGE_BYTES: usize = 4 * 1024;

pub type Extensions = Map<String, Value>;

/// The protocol version is intentionally independent from crate and SDK versions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, JsonSchema)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProtocolVersion {
    pub const V1: Self = Self {
        major: PROTOCOL_MAJOR,
        minor: PROTOCOL_MINOR,
    };

    pub const fn is_compatible_with(self, other: Self) -> bool {
        self.major == other.major
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}", self.major, self.minor)
    }
}

impl FromStr for ProtocolVersion {
    type Err = ProtocolVersionParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut parts = value.split('.');
        let major = parts
            .next()
            .ok_or(ProtocolVersionParseError)?
            .parse()
            .map_err(|_| ProtocolVersionParseError)?;
        let minor = parts
            .next()
            .ok_or(ProtocolVersionParseError)?
            .parse()
            .map_err(|_| ProtocolVersionParseError)?;
        if parts.next().is_some() {
            return Err(ProtocolVersionParseError);
        }
        Ok(Self { major, minor })
    }
}

impl Serialize for ProtocolVersion {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for ProtocolVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(D::Error::custom)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ProtocolVersionParseError;

impl fmt::Display for ProtocolVersionParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("protocol version must be <major>.<minor>")
    }
}

impl std::error::Error for ProtocolVersionParseError {}

/// A single JSONL frame. `request_id` correlates commands, acknowledgements,
/// errors, and callbacks; `run_id` scopes commands and events to one run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Envelope {
    pub protocol_version: ProtocolVersion,
    pub request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(flatten)]
    pub message: ProtocolMessage,
}

impl Envelope {
    pub fn new(
        request_id: impl Into<String>,
        run_id: Option<String>,
        message: ProtocolMessage,
    ) -> Self {
        Self {
            protocol_version: ProtocolVersion::V1,
            request_id: request_id.into(),
            run_id,
            message,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum ProtocolMessage {
    ClientHello(ClientHello),
    StartRun(Box<StartRun>),
    CancelRun(CancelRun),
    ToolResult(ToolResultResponse),
    PolicyDecision(PolicyDecisionResponse),
    ApprovalDecision(ApprovalDecisionResponse),
    GetProviderHealth(ProviderInspectionRequest),
    GetModelInventory(ProviderInspectionRequest),
    Ping(Ping),
    Shutdown(Shutdown),
    RuntimeHello(RuntimeHello),
    CommandAcknowledged(CommandAcknowledged),
    ProtocolError(ProtocolErrorPayload),
    RunStarted(RunStarted),
    RunEvent(RunEventPayload),
    PolicyDecisionRequested(PolicyDecisionRequest),
    ApprovalRequested(ApprovalRequest),
    ToolExecutionRequested(ToolExecutionRequest),
    RunCompleted(RunCompleted),
    RunFailed(RunFailed),
    RunCancelled(RunCancelled),
    Pong(Pong),
    ProviderHealth(ProviderHealthResponse),
    ModelInventory(ModelInventoryResponse),
}

impl ProtocolMessage {
    pub const fn direction(&self) -> MessageDirection {
        match self {
            Self::ClientHello(_)
            | Self::StartRun(_)
            | Self::CancelRun(_)
            | Self::ToolResult(_)
            | Self::PolicyDecision(_)
            | Self::ApprovalDecision(_)
            | Self::GetProviderHealth(_)
            | Self::GetModelInventory(_)
            | Self::Ping(_)
            | Self::Shutdown(_) => MessageDirection::ClientToRuntime,
            _ => MessageDirection::RuntimeToClient,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageDirection {
    ClientToRuntime,
    RuntimeToClient,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ClientHello {
    pub sdk: ClientIdentity,
    #[serde(default)]
    pub capabilities: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ClientIdentity {
    pub name: String,
    pub version: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimeHello {
    pub runtime_version: String,
    pub capabilities: RuntimeCapabilities,
    #[serde(default)]
    pub providers: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimeCapabilities {
    pub supports_output_deltas: bool,
    pub supports_structured_output: bool,
    pub supports_trace_persistence: bool,
    pub concurrent_runs: u16,
    pub max_pending_callbacks: u16,
    pub max_queue_depth: u16,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StartRun {
    pub request: WireRunRequest,
}

/// A provider inspection command is separate from an agent run and therefore
/// does not allocate a run ID or invoke application tools.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ProviderInspectionRequest {
    pub provider: ProviderConfiguration,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WireRunRequest {
    pub provider: ProviderConfiguration,
    pub agent: WireAgentDefinition,
    pub input: String,
    #[serde(default)]
    pub application_context: Extensions,
    #[serde(default)]
    pub history: Vec<WireMessage>,
    #[serde(default)]
    pub metadata: Extensions,
    #[serde(default)]
    pub overrides: WireRunOverrides,
    #[serde(default)]
    pub evaluation: Extensions,
    #[serde(default)]
    pub tools: Vec<WireToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_database_path: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProviderConfiguration {
    Ollama { base_url: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WireAgentDefinition {
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub system_instructions: String,
    pub default_model: String,
    #[serde(default)]
    pub tool_allowlist: Vec<String>,
    #[serde(default)]
    pub limits: WireAgentLimits,
    #[serde(default)]
    pub generation: WireGenerationOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    #[serde(default)]
    pub metadata: Extensions,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct WireAgentLimits {
    pub max_model_calls: u32,
    pub max_tool_calls: u32,
    pub max_identical_tool_calls: u32,
    pub max_run_duration_ms: Option<u64>,
    pub max_model_call_duration_ms: Option<u64>,
    pub max_output_repairs: u32,
    pub max_provider_retries: u32,
    pub max_input_bytes: u64,
    pub max_request_payload_bytes: u64,
    pub max_model_response_bytes: u64,
    pub max_tool_arguments_bytes: u64,
    pub max_tool_result_bytes: u64,
    pub max_transcript_bytes: u64,
    pub max_json_depth: u32,
}

impl Default for WireAgentLimits {
    fn default() -> Self {
        Self {
            max_model_calls: 8,
            max_tool_calls: 16,
            max_identical_tool_calls: 2,
            max_run_duration_ms: None,
            max_model_call_duration_ms: None,
            max_output_repairs: 1,
            max_provider_retries: 2,
            max_input_bytes: 64 * 1024,
            max_request_payload_bytes: 256 * 1024,
            max_model_response_bytes: 1024 * 1024,
            max_tool_arguments_bytes: 64 * 1024,
            max_tool_result_bytes: 1024 * 1024,
            max_transcript_bytes: 4 * 1024 * 1024,
            max_json_depth: 64,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WireGenerationOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WireRunOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub generation: WireGenerationOptions,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WireMessage {
    pub role: WireMessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<WireToolCall>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireMessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WireToolDefinition {
    pub id: String,
    pub name: String,
    pub description: String,
    pub arguments_schema: Value,
    pub risk: WireToolRisk,
    pub idempotent: bool,
    pub read_only: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireToolRisk {
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WireToolCall {
    pub id: String,
    pub tool_id: String,
    pub arguments_json: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WireToolResult {
    pub ok: bool,
    pub output: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CancelRun {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolResultResponse {
    pub callback_id: String,
    pub result: WireToolResult,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PolicyDecisionResponse {
    pub callback_id: String,
    pub decision: WirePolicyDecision,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ApprovalDecisionResponse {
    pub callback_id: String,
    pub granted: bool,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Ping {
    pub nonce: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Shutdown {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CommandAcknowledged {
    pub command: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ProtocolErrorPayload {
    pub code: ProtocolErrorCode,
    pub message: String,
    #[serde(default)]
    pub retryable: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolErrorCode {
    IncompatibleVersion,
    InvalidMessage,
    MessageTooLarge,
    JsonTooDeep,
    UnknownMessageType,
    InvalidState,
    UnknownRun,
    UnknownCallback,
    DuplicateCallback,
    QueueFull,
    CallbackTimedOut,
    RuntimeUnavailable,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RunStarted {
    pub trace_id: String,
    /// Monotonic within a run. Every run-scoped runtime message carries one.
    pub run_sequence: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunEventPayload {
    pub trace_id: String,
    /// Monotonic within a run, starting at one.
    pub sequence: u64,
    pub timestamp_ms: u64,
    pub event: WireRunEvent,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireRunEvent {
    Started {
        trace_id: String,
    },
    ModelRequested {
        call_number: u32,
        model: String,
    },
    ModelRetrying {
        next_call_number: u32,
        reason: String,
    },
    ModelResponded {
        call_number: u32,
    },
    ToolRejected {
        call_id: String,
        tool_id: String,
        reason: String,
    },
    PolicyDecided {
        call_id: String,
        decision: WirePolicyDecision,
    },
    ApprovalRequested {
        call_id: String,
        tool_id: String,
    },
    ToolCompleted {
        call_id: String,
        tool_id: String,
        ok: bool,
    },
    Completed {
        status: WireRunStatus,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum WirePolicyDecision {
    Allow { reason: String },
    Deny { reason: String },
    RequireApproval { reason: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PolicyDecisionRequest {
    pub run_sequence: u64,
    pub callback_id: String,
    pub trace_id: String,
    pub call_id: String,
    pub tool: WireToolDefinition,
    pub arguments: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ApprovalRequest {
    pub run_sequence: u64,
    pub callback_id: String,
    pub trace_id: String,
    pub call_id: String,
    pub tool: WireToolDefinition,
    pub arguments: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolExecutionRequest {
    pub run_sequence: u64,
    pub callback_id: String,
    pub trace_id: String,
    pub call_id: String,
    pub tool: WireToolDefinition,
    pub arguments: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunCompleted {
    pub run_sequence: u64,
    pub result: WireRunResult,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RunFailed {
    pub run_sequence: u64,
    pub error: WireRunError,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RunCancelled {
    pub run_sequence: u64,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WireRunResult {
    pub status: WireRunStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_output: Option<String>,
    pub model: String,
    #[serde(default)]
    pub tool_calls: Vec<WireToolCall>,
    #[serde(default)]
    pub policy_decisions: Vec<WirePolicyDecision>,
    #[serde(default)]
    pub approvals: Vec<WireApprovalRecord>,
    #[serde(default)]
    pub errors: Vec<WireRunError>,
    pub duration_ms: u64,
    pub trace_id: String,
    pub model_call_limit_reached: bool,
    pub tool_call_limit_reached: bool,
    pub repeated_tool_call_limit_reached: bool,
    pub cancelled: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireRunStatus {
    Completed,
    Failed,
    Cancelled,
    LimitReached,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WireApprovalRecord {
    pub call_id: String,
    pub tool_id: String,
    pub granted: bool,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WireRunError {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Pong {
    pub nonce: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ProviderHealthResponse {
    pub healthy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModelInventoryResponse {
    pub models: Vec<ModelInfo>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModelInfo {
    pub id: String,
    pub capabilities: ModelCapabilities,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModelCapabilities {
    pub supports_tools: bool,
    pub supports_streaming: bool,
    pub supports_structured_output: bool,
}
