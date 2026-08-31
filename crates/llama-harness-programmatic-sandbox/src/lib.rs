#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! A deterministic, resource-bounded language for programmatic tool orchestration.
//!
//! The public boundary accepts only the versioned JSON syntax tree. Compiled
//! instructions remain private and are never accepted from or persisted by a
//! caller. The crate owns no external capabilities: tool requests are inert data
//! that an embedding host may independently authorize and execute.

extern crate alloc;
#[cfg(test)]
extern crate std;

mod accounting;
mod ast;
mod compiler;
mod error;
mod limits;
mod parser;
mod vm;

pub use ast::{
    BinaryOperator, Expression, ObjectEntry, Program, Statement, UnaryOperator, PROGRAM_VERSION_V1,
};
pub use compiler::VerifiedProgram;
pub use error::{SandboxError, SandboxErrorCode};
pub use limits::{SandboxLimits, HARD_LIMITS};
pub use vm::{
    Execution, ExecutionId, ExecutionMetrics, ResumeToken, StepOutcome, ToolBatch, ToolRequest,
    ToolResponse,
};
