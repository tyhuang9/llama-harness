//! Local, redacted trace persistence for the embedded Llama Harness runtime.
//!
//! This crate persists structured causal events, never model chain-of-thought.

mod redaction;
mod sqlite;

pub use redaction::{RedactionConfig, REDACTED_VALUE};
pub use sqlite::{
    AppendOutcome, ExportedRun, PersistedEvent, RetentionPolicy, RetentionResult, RunListQuery,
    RunSummary, SqliteEventSink, TraceStoreConfig, TraceStoreError,
};
