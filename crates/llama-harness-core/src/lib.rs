//! Embedded, provider-neutral agent runtime for application-owned integrations.

#![deny(missing_docs)]

mod agent;
mod agent_manifest;
mod error;
mod event;
mod limits;
mod message;
/// Deterministic mock provider and scripted response helpers.
pub mod mock;
mod model;
mod policy;
mod runner;
mod tool;

pub use agent::{AgentDefinition, JsonMap, RunOverrides, RunRequest, RunResult, RunStatus};
pub use agent_manifest::{
    load_agent_manifest, load_agent_manifest_path, AgentManifest, AgentManifestError,
    AGENT_MANIFEST_VERSION,
};
pub use error::{HarnessError, RunError};
pub use event::{EventRecord, EventSink, InMemoryEventSink, RunEvent};
pub use limits::{AgentLimits, GenerationOptions};
pub use message::{Message, MessageRole};
pub use model::{
    ModelCapabilities, ModelInfo, ModelProvider, ModelRequest, ModelResponse, ProviderHealth, Usage,
};
pub use policy::{
    AllowAllPolicy, ApprovalHandler, ApprovalRecord, DenyApproval, PolicyDecision, PolicyEngine,
    SafeDefaultPolicy,
};
pub use runner::{AgentRunner, AgentRunnerBuilder};
pub use tool::{
    Tool, ToolCall, ToolCallContext, ToolDefinition, ToolRegistry, ToolResult, ToolRisk,
};
