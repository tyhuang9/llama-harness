#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! A deterministic, resource-bounded language for programmatic tool orchestration.
//!
//! The public boundary accepts only the versioned JSON syntax tree. Compiled
//! instructions remain private and are never accepted from or persisted by a
//! caller. The crate owns no external capabilities: tool requests are inert data
//! that an embedding host may independently authorize and execute.
//!
//! The cached runtime graph uses [`alloc::sync::Arc`], so supported targets
//! must provide pointer-width atomic operations. This keeps suspended
//! [`Execution`] values safe to move across async task boundaries without
//! introducing a `std` dependency.

#[cfg(not(target_has_atomic = "ptr"))]
compile_error!("llama-harness-programmatic-sandbox requires pointer-width atomics for Arc");

extern crate alloc;
#[cfg(test)]
extern crate std;

mod accounting;
mod ast;
mod compiler;
mod error;
mod limits;
mod parser;
mod value;
mod vm;

pub(crate) use ast::{BinaryOperator, Expression, ObjectEntry, Statement, UnaryOperator};
pub use ast::{Program, PROGRAM_VERSION_V1};
pub use compiler::VerifiedProgram;
pub use error::{SandboxError, SandboxErrorCode};
pub use limits::{SandboxLimits, HARD_LIMITS};
pub use vm::{
    Execution, ExecutionId, ExecutionMetrics, ResumeToken, StepOutcome, ToolBatch, ToolRequest,
    ToolResponse, MAX_ATOMIC_KEY_BYTES, MAX_ATOMIC_STRING_BYTES,
};
