//! Local, redacted trace persistence for the embedded Llama Harness runtime.
//!
//! This crate persists structured causal events, never model chain-of-thought.

#![deny(missing_docs)]

mod redaction;
mod sqlite;

pub use redaction::{RedactionConfig, REDACTED_VALUE};
pub use sqlite::{
    AppendOutcome, ExportedRun, PersistedEvent, PersistenceFailure, PersistenceFailureCategory,
    PersistenceFailureHandler, PersistenceHealth, RetentionPolicy, RetentionResult, RunListQuery,
    RunSummary, SqliteEventSink, TraceStoreConfig, TraceStoreError,
};
