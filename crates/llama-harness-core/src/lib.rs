//! Embedded, provider-neutral agent runtime for application-owned integrations.

mod runtime;

pub use runtime::*;

pub mod agent {
    pub use crate::{AgentDefinition, RunOverrides, RunRequest, RunResult, RunStatus};
}

pub mod message {
    pub use crate::{Message, MessageRole};
}

pub mod model {
    pub use crate::{
        ModelCapabilities, ModelInfo, ModelProvider, ModelRequest, ModelResponse, ProviderHealth,
        Usage,
    };
}

pub mod tool {
    pub use crate::{Tool, ToolCall, ToolDefinition, ToolRegistry, ToolResult, ToolRisk};
}

pub mod policy {
    pub use crate::{
        AllowAllPolicy, ApprovalHandler, ApprovalRecord, DenyApproval, PolicyDecision, PolicyEngine,
    };
}

pub mod event {
    pub use crate::{EventSink, InMemoryEventSink, RunEvent};
}

pub mod limits {
    pub use crate::{AgentLimits, GenerationOptions};
}

pub mod error {
    pub use crate::{HarnessError, RunError};
}

pub mod runner {
    pub use crate::{AgentRunner, AgentRunnerBuilder};
}
