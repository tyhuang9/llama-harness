//! Minimal reusable pieces for an embedded Tauri agent host.
//!
//! An application still owns its `ModelProvider`, Rust `Tool` implementations,
//! policy, and runner construction. This example only wires canonical events,
//! approvals, and cancellation into Tauri.

use std::sync::Arc;

use llama_harness::{
    tauri::{
        ApprovalRouter, RunRegistry, RunRegistryError, TauriAppHandle, TauriEventSink,
        TauriRuntime, TauriTargetEmitter,
    },
    EventSink, RunRequest,
};

/// The only window permitted to receive run and approval payloads.
pub const MAIN_WINDOW_LABEL: &str = "main";

pub struct EmbeddedAgentHost<R: TauriRuntime> {
    pub runs: Arc<RunRegistry>,
    pub approvals: Arc<ApprovalRouter<TauriTargetEmitter<R>>>,
    pub events: Arc<TauriEventSink<TauriTargetEmitter<R>>>,
}

impl<R: TauriRuntime> EmbeddedAgentHost<R> {
    pub fn new(app: TauriAppHandle<R>) -> Self {
        let emitter = TauriTargetEmitter::new(app, MAIN_WINDOW_LABEL);
        Self {
            runs: Arc::new(RunRegistry::default()),
            approvals: Arc::new(ApprovalRouter::new(emitter.clone())),
            events: Arc::new(TauriEventSink::new(emitter)),
        }
    }

    /// Register before starting `AgentRunner::run`; expose this ID to a Tauri
    /// cancellation command, then call `runs.complete` when the future ends.
    pub fn register(&self, request: &mut RunRequest) -> Result<String, RunRegistryError> {
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
