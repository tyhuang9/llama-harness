//! Minimal reusable pieces for an embedded Tauri agent host.
//!
//! An application still owns its `ModelProvider`, Rust `Tool` implementations,
//! policy, and runner construction. This example only wires canonical events,
//! approvals, and cancellation into Tauri.

use std::sync::Arc;

use llama_harness_core::{EventSink, RunRequest};
use llama_harness_tauri::{ApprovalRouter, RunRegistry, TauriEmitter, TauriEventSink};
use tauri::{AppHandle, Runtime};

pub struct EmbeddedAgentHost<R: Runtime> {
    pub runs: Arc<RunRegistry>,
    pub approvals: Arc<ApprovalRouter<TauriEmitter<R>>>,
    pub events: Arc<TauriEventSink<TauriEmitter<R>>>,
}

impl<R: Runtime> EmbeddedAgentHost<R> {
    pub fn new(app: AppHandle<R>) -> Self {
        let emitter = TauriEmitter::new(app);
        Self {
            runs: Arc::new(RunRegistry::default()),
            approvals: Arc::new(ApprovalRouter::new(emitter.clone())),
            events: Arc::new(TauriEventSink::new(emitter)),
        }
    }

    /// Register before starting `AgentRunner::run`; expose this ID to a Tauri
    /// cancellation command, then call `runs.complete` when the future ends.
    pub fn register(
        &self,
        request: &mut RunRequest,
    ) -> Result<String, llama_harness_tauri::RunRegistryError> {
        self.runs.register(request)
    }

    pub fn event_sink(&self) -> Arc<dyn EventSink> {
        Arc::clone(&self.events) as Arc<dyn EventSink>
    }

    /// Route a frontend decision from a narrow `#[tauri::command]` in the host.
    pub fn respond_to_approval(
        &self,
        approval_id: &str,
        granted: bool,
        reason: impl Into<String>,
    ) -> bool {
        self.approvals.respond(approval_id, granted, reason)
    }

    /// Cancel both the core run and any pending frontend approval on shutdown or
    /// through a host command.
    pub fn cancel(&self, run_id: &str) -> bool {
        self.approvals.cancel_run(run_id);
        self.runs.cancel(run_id)
    }
}
