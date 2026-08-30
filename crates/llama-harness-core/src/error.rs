use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Stable, value-free reasons why a provider-neutral model stream failed.
pub enum ModelStreamFailureKind {
    /// An event arrived after the stream completed successfully.
    EventAfterCompletion,
    /// An event arrived after the stream had already failed.
    EventAfterFailure,
    /// EOF was observed after the stream had already failed.
    EofAfterFailure,
    /// EOF arrived before an explicit completion event.
    EofBeforeCompletion,
    /// Completion arrived while a tool call was still incomplete.
    CompletionWithPendingCall,
    /// EOF arrived while a tool call was still incomplete.
    EofWithPendingCall,
    /// Direct assembler finalization found an incomplete tool call.
    IncompletePendingCall,
    /// A fragment arrived after its indexed tool call had completed.
    FragmentAfterCallCompletion,
    /// A streamed call identifier was empty or exceeded its bound.
    InvalidCallId,
    /// A streamed tool identifier was empty or exceeded its bound.
    InvalidToolId,
    /// A streamed call identifier changed between fragments.
    ConflictingCallId,
    /// A streamed tool identifier changed between fragments.
    ConflictingToolId,
    /// A final tool call did not provide a call identifier.
    MissingCallId,
    /// A final tool call did not provide a tool identifier.
    MissingToolId,
    /// Two completed tool calls used the same call identifier.
    DuplicateCallId,
    /// A completed call selected a tool outside the allowed catalog.
    UnknownTool,
    /// Completed tool arguments were not valid JSON.
    MalformedArgumentsJson,
    /// Completed tool arguments did not satisfy the registered schema.
    ArgumentsSchemaMismatch,
    /// The upstream provider stream returned a non-resource failure.
    UpstreamProviderFailure,
}

impl ModelStreamFailureKind {
    /// Returns the stable machine-readable run-error code.
    pub const fn code(self) -> &'static str {
        match self {
            Self::EventAfterCompletion => "model_stream.event_after_completion",
            Self::EventAfterFailure => "model_stream.event_after_failure",
            Self::EofAfterFailure => "model_stream.eof_after_failure",
            Self::EofBeforeCompletion => "model_stream.eof_before_completion",
            Self::CompletionWithPendingCall => "model_stream.completion_with_pending_call",
            Self::EofWithPendingCall => "model_stream.eof_with_pending_call",
            Self::IncompletePendingCall => "model_stream.incomplete_pending_call",
            Self::FragmentAfterCallCompletion => "model_stream.fragment_after_call_completion",
            Self::InvalidCallId => "model_stream.invalid_call_id",
            Self::InvalidToolId => "model_stream.invalid_tool_id",
            Self::ConflictingCallId => "model_stream.conflicting_call_id",
            Self::ConflictingToolId => "model_stream.conflicting_tool_id",
            Self::MissingCallId => "model_stream.missing_call_id",
            Self::MissingToolId => "model_stream.missing_tool_id",
            Self::DuplicateCallId => "model_stream.duplicate_call_id",
            Self::UnknownTool => "model_stream.unknown_tool",
            Self::MalformedArgumentsJson => "model_stream.malformed_arguments_json",
            Self::ArgumentsSchemaMismatch => "model_stream.arguments_schema_mismatch",
            Self::UpstreamProviderFailure => "model_stream.upstream_provider_failure",
        }
    }

    /// Returns the stable human-readable message without provider-controlled data.
    pub const fn message(self) -> &'static str {
        match self {
            Self::EventAfterCompletion => "model stream event received after completion",
            Self::EventAfterFailure => "model stream event received after terminal failure",
            Self::EofAfterFailure => "model stream EOF received after terminal failure",
            Self::EofBeforeCompletion => "model stream reached EOF before completion",
            Self::CompletionWithPendingCall => {
                "model stream completed with an incomplete tool call"
            }
            Self::EofWithPendingCall => "model stream reached EOF with an incomplete tool call",
            Self::IncompletePendingCall => "model stream ended with an incomplete tool call",
            Self::FragmentAfterCallCompletion => {
                "model stream tool-call fragment received after call completion"
            }
            Self::InvalidCallId => "model stream provided an invalid call identifier",
            Self::InvalidToolId => "model stream provided an invalid tool identifier",
            Self::ConflictingCallId => "model stream changed a tool-call identifier",
            Self::ConflictingToolId => "model stream changed a tool identifier",
            Self::MissingCallId => "model stream omitted a final call identifier",
            Self::MissingToolId => "model stream omitted a final tool identifier",
            Self::DuplicateCallId => "model stream reused a tool-call identifier",
            Self::UnknownTool => "model stream selected an unknown tool",
            Self::MalformedArgumentsJson => "model stream produced malformed tool arguments",
            Self::ArgumentsSchemaMismatch => "model stream tool arguments failed schema validation",
            Self::UpstreamProviderFailure => "upstream model stream failed",
        }
    }
}

impl fmt::Display for ModelStreamFailureKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message())
    }
}

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
    #[error("{kind}")]
    /// A provider-neutral model stream failed with a stable value-free reason.
    ModelStream {
        /// Stable reason for the stream failure.
        kind: ModelStreamFailureKind,
    },
    #[error("unsupported capability: {0}")]
    /// The selected provider does not implement a requested optional capability.
    UnsupportedCapability(String),
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
            Self::ModelStream { kind } => kind.code(),
            Self::UnsupportedCapability(_) => "unsupported_capability",
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

#[cfg(test)]
mod tests {
    use super::{HarnessError, ModelStreamFailureKind};

    #[test]
    fn model_stream_failures_have_exact_value_free_run_errors() {
        let expected = [
            (
                ModelStreamFailureKind::EventAfterCompletion,
                "model_stream.event_after_completion",
                "model stream event received after completion",
            ),
            (
                ModelStreamFailureKind::EventAfterFailure,
                "model_stream.event_after_failure",
                "model stream event received after terminal failure",
            ),
            (
                ModelStreamFailureKind::EofAfterFailure,
                "model_stream.eof_after_failure",
                "model stream EOF received after terminal failure",
            ),
            (
                ModelStreamFailureKind::EofBeforeCompletion,
                "model_stream.eof_before_completion",
                "model stream reached EOF before completion",
            ),
            (
                ModelStreamFailureKind::CompletionWithPendingCall,
                "model_stream.completion_with_pending_call",
                "model stream completed with an incomplete tool call",
            ),
            (
                ModelStreamFailureKind::EofWithPendingCall,
                "model_stream.eof_with_pending_call",
                "model stream reached EOF with an incomplete tool call",
            ),
            (
                ModelStreamFailureKind::IncompletePendingCall,
                "model_stream.incomplete_pending_call",
                "model stream ended with an incomplete tool call",
            ),
            (
                ModelStreamFailureKind::FragmentAfterCallCompletion,
                "model_stream.fragment_after_call_completion",
                "model stream tool-call fragment received after call completion",
            ),
            (
                ModelStreamFailureKind::InvalidCallId,
                "model_stream.invalid_call_id",
                "model stream provided an invalid call identifier",
            ),
            (
                ModelStreamFailureKind::InvalidToolId,
                "model_stream.invalid_tool_id",
                "model stream provided an invalid tool identifier",
            ),
            (
                ModelStreamFailureKind::ConflictingCallId,
                "model_stream.conflicting_call_id",
                "model stream changed a tool-call identifier",
            ),
            (
                ModelStreamFailureKind::ConflictingToolId,
                "model_stream.conflicting_tool_id",
                "model stream changed a tool identifier",
            ),
            (
                ModelStreamFailureKind::MissingCallId,
                "model_stream.missing_call_id",
                "model stream omitted a final call identifier",
            ),
            (
                ModelStreamFailureKind::MissingToolId,
                "model_stream.missing_tool_id",
                "model stream omitted a final tool identifier",
            ),
            (
                ModelStreamFailureKind::DuplicateCallId,
                "model_stream.duplicate_call_id",
                "model stream reused a tool-call identifier",
            ),
            (
                ModelStreamFailureKind::UnknownTool,
                "model_stream.unknown_tool",
                "model stream selected an unknown tool",
            ),
            (
                ModelStreamFailureKind::MalformedArgumentsJson,
                "model_stream.malformed_arguments_json",
                "model stream produced malformed tool arguments",
            ),
            (
                ModelStreamFailureKind::ArgumentsSchemaMismatch,
                "model_stream.arguments_schema_mismatch",
                "model stream tool arguments failed schema validation",
            ),
            (
                ModelStreamFailureKind::UpstreamProviderFailure,
                "model_stream.upstream_provider_failure",
                "upstream model stream failed",
            ),
        ];

        for (kind, code, message) in expected {
            let error = HarnessError::ModelStream { kind };
            assert_eq!(kind.code(), code);
            assert_eq!(kind.message(), message);
            assert_eq!(error.to_string(), message);
            let run_error = error.run_error();
            assert_eq!(run_error.code, code);
            assert_eq!(run_error.message, message);
            let serialized = serde_json::to_string(&run_error).unwrap();
            assert!(!serialized.contains("sentinel-provider-controlled-value"));
        }
    }
}
