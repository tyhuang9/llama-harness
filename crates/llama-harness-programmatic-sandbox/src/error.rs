use alloc::string::String;
use core::fmt;

/// Stable machine-readable sandbox error category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SandboxErrorCode {
    /// The JSON input is malformed or violates the strict AST schema.
    InvalidProgram,
    /// A caller-provided limit is zero or exceeds an immutable library cap.
    InvalidLimits,
    /// A bounded program resource was exhausted.
    ResourceLimit,
    /// Static verification rejected the program.
    Verification,
    /// Deterministic execution failed.
    Execution,
    /// A resume token did not match the suspended execution.
    InvalidResume,
}

/// An error safe to expose without embedding program source or runtime values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxError {
    code: SandboxErrorCode,
    message: String,
    kind: SandboxErrorKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SandboxErrorKind {
    General,
    OutputLimit,
}

impl SandboxError {
    pub(crate) fn new(code: SandboxErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            kind: SandboxErrorKind::General,
        }
    }

    pub(crate) fn output_limit() -> Self {
        Self {
            code: SandboxErrorCode::ResourceLimit,
            message: "output byte limit exceeded".into(),
            kind: SandboxErrorKind::OutputLimit,
        }
    }

    /// Returns the stable error category.
    pub const fn code(&self) -> SandboxErrorCode {
        self.code
    }

    /// Returns whether this error is specifically a program-return size
    /// violation rather than another sandbox resource limit.
    pub const fn is_output_limit(&self) -> bool {
        matches!(self.kind, SandboxErrorKind::OutputLimit)
    }

    /// Returns a bounded, source-free diagnostic.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SandboxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl core::error::Error for SandboxError {}
