use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
/// Serializable error information captured in a run result.
pub struct RunError {
    /// Stable machine-readable error code.
    pub code: String,
    /// Human-readable error message.
    pub message: String,
}

impl RunError {
    /// Creates a stable, serializable run error.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
/// Errors raised while validating or executing a run.
pub enum HarnessError {
    #[error("invalid request: {0}")]
    /// The run request is invalid.
    InvalidRequest(String),
    #[error("invalid tool: {0}")]
    /// A requested tool is unknown or improperly registered.
    InvalidTool(String),
    #[error("invalid tool arguments: {0}")]
    /// Tool arguments failed schema validation.
    InvalidArguments(String),
    #[error("provider error: {0}")]
    /// The model provider returned a non-retryable error.
    Provider(String),
    #[error("retryable provider error: {0}")]
    /// The model provider returned an error eligible for retry.
    RetryableProvider(String),
    #[error("policy error: {0}")]
    /// Policy evaluation failed.
    Policy(String),
    #[error("approval error: {0}")]
    /// Approval handling failed.
    Approval(String),
    #[error("tool error: {0}")]
    /// Tool execution failed.
    Tool(String),
    #[error("cancelled")]
    /// Cooperative cancellation stopped the operation.
    Cancelled,
    #[error("timed out: {0}")]
    /// A configured operation timeout elapsed.
    TimedOut(String),
    #[error("resource limit reached: {0}")]
    /// A configured resource limit was exceeded.
    ResourceLimit(String),
    #[error("invalid structured output: {0}")]
    /// Final output failed structured-output validation.
    InvalidOutput(String),
}

impl HarnessError {
    pub(crate) fn run_error(&self) -> RunError {
        let code = match self {
            Self::InvalidRequest(_) => "invalid_request",
            Self::InvalidTool(_) => "invalid_tool",
            Self::InvalidArguments(_) => "invalid_arguments",
            Self::Provider(_) | Self::RetryableProvider(_) => "provider_error",
            Self::Policy(_) => "policy_error",
            Self::Approval(_) => "approval_error",
            Self::Tool(_) => "tool_error",
            Self::Cancelled => "cancelled",
            Self::TimedOut(_) => "timed_out",
            Self::ResourceLimit(_) => "resource_limit",
            Self::InvalidOutput(_) => "invalid_output",
        };
        RunError {
            code: code.into(),
            message: self.to_string(),
        }
    }
}
