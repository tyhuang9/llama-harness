//! Stable, language-neutral contracts for the llama-harness managed sidecar.
//!
//! Protocol v1 uses newline-delimited JSON over a child process's standard input
//! and output. Standard output contains protocol frames only; diagnostics belong
//! on standard error. Rust and Tauri consumers embed the engine directly and do
//! not use this protocol.

mod contracts;
mod validation;

pub use contracts::*;
pub use validation::*;
