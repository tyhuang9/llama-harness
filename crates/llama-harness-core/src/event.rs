use crate::{PolicyDecision, RunStatus};
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
/// An ordered event emitted during a run.
pub struct EventRecord {
    /// Identifier of the run that emitted the event.
    pub run_id: String,
    /// Trace identifier associated with the run.
    pub trace_id: String,
    /// Monotonic within one run, starting at one.
    pub sequence: u64,
    /// Milliseconds since the Unix epoch. Never decreases within one emitter.
    pub timestamp_ms: u64,
    /// Event payload.
    pub event: RunEvent,
}

impl EventRecord {
    /// Creates one ordered event record for a run and trace.
    pub fn new(
        run_id: impl Into<String>,
        trace_id: impl Into<String>,
        sequence: u64,
        timestamp_ms: u64,
        event: RunEvent,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            trace_id: trace_id.into(),
            sequence,
            timestamp_ms,
            event,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
/// Lifecycle event emitted by the runner.
pub enum RunEvent {
    /// A run has started.
    Started {
        /// Identifier of the started run.
        run_id: String,
        /// Trace identifier for the started run.
        trace_id: String,
    },
    /// The runner is requesting a model completion.
    ModelRequested {
        /// One-based model call number.
        call_number: u32,
        /// Model identifier being called.
        model: String,
    },
    /// A retryable provider error caused another model call to be scheduled.
    ModelRetrying {
        /// Number of the next model call.
        next_call_number: u32,
        /// Reason the previous call will be retried.
        reason: String,
    },
    /// The model returned a response.
    ModelResponded {
        /// Model call number that returned.
        call_number: u32,
    },
    /// A tool call was rejected before execution.
    ToolRejected {
        /// Identifier of the rejected call.
        call_id: String,
        /// Identifier of the rejected tool.
        tool_id: String,
        /// Reason for rejection.
        reason: String,
    },
    /// A policy decision was recorded for a tool call.
    PolicyDecided {
        /// Identifier of the related tool call.
        call_id: String,
        /// Policy outcome.
        decision: PolicyDecision,
    },
    /// Approval was requested for a tool call.
    ApprovalRequested {
        /// Identifier of the tool call awaiting approval.
        call_id: String,
        /// Identifier of the tool requiring approval.
        tool_id: String,
    },
    /// A tool call finished execution.
    ToolCompleted {
        /// Identifier of the completed call.
        call_id: String,
        /// Identifier of the completed tool.
        tool_id: String,
        /// Whether the tool returned a successful result.
        ok: bool,
    },
    /// The run reached a terminal status.
    Completed {
        /// Terminal run status.
        status: RunStatus,
    },
}

/// Receives ordered events from an agent runner.
pub trait EventSink: Send + Sync {
    /// Records one emitted event.
    fn emit(&self, record: EventRecord);
}

#[derive(Default)]
/// Thread-safe in-memory event collector.
pub struct InMemoryEventSink {
    events: Mutex<Vec<EventRecord>>,
}

impl InMemoryEventSink {
    /// Returns a snapshot of all events collected so far.
    pub fn events(&self) -> Vec<EventRecord> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl EventSink for InMemoryEventSink {
    fn emit(&self, record: EventRecord) {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(record);
    }
}

pub(crate) struct EventEmitter {
    run_id: String,
    trace_id: String,
    sequence: u64,
    last_timestamp_ms: u64,
    sink: Arc<dyn EventSink>,
}

impl EventEmitter {
    pub(crate) fn new(run_id: String, trace_id: String, sink: Arc<dyn EventSink>) -> Self {
        Self {
            run_id,
            trace_id,
            sequence: 0,
            last_timestamp_ms: 0,
            sink,
        }
    }

    pub(crate) fn emit(&mut self, event: RunEvent) {
        self.sequence = self.sequence.saturating_add(1);
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis() as u64)
            .max(self.last_timestamp_ms);
        self.last_timestamp_ms = timestamp_ms;
        self.sink.emit(EventRecord {
            run_id: self.run_id.clone(),
            trace_id: self.trace_id.clone(),
            sequence: self.sequence,
            timestamp_ms,
            event,
        });
    }
}
