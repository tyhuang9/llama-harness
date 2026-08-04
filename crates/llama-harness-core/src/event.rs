use crate::{PolicyDecision, RunStatus};
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EventRecord {
    pub run_id: String,
    pub trace_id: String,
    /// Monotonic within one run, starting at one.
    pub sequence: u64,
    /// Milliseconds since the Unix epoch. Never decreases within one emitter.
    pub timestamp_ms: u64,
    pub event: RunEvent,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunEvent {
    Started {
        run_id: String,
        trace_id: String,
    },
    ModelRequested {
        call_number: u32,
        model: String,
    },
    ModelRetrying {
        next_call_number: u32,
        reason: String,
    },
    ModelResponded {
        call_number: u32,
    },
    ToolRejected {
        call_id: String,
        tool_id: String,
        reason: String,
    },
    PolicyDecided {
        call_id: String,
        decision: PolicyDecision,
    },
    ApprovalRequested {
        call_id: String,
        tool_id: String,
    },
    ToolCompleted {
        call_id: String,
        tool_id: String,
        ok: bool,
    },
    Completed {
        status: RunStatus,
    },
}

pub trait EventSink: Send + Sync {
    fn emit(&self, record: EventRecord);
}

#[derive(Default)]
pub struct InMemoryEventSink {
    events: Mutex<Vec<EventRecord>>,
}

impl InMemoryEventSink {
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
