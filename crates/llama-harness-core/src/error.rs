use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunError {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum HarnessError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("invalid tool: {0}")]
    InvalidTool(String),
    #[error("invalid tool arguments: {0}")]
    InvalidArguments(String),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("retryable provider error: {0}")]
    RetryableProvider(String),
    #[error("policy error: {0}")]
    Policy(String),
    #[error("approval error: {0}")]
    Approval(String),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("cancelled")]
    Cancelled,
    #[error("timed out: {0}")]
    TimedOut(String),
    #[error("resource limit reached: {0}")]
    ResourceLimit(String),
    #[error("invalid structured output: {0}")]
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
