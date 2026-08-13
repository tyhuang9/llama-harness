//! Tauri helpers for embedded llama-harness applications.
//!
//! The helpers compose with the canonical embedded [`llama_harness_core::AgentRunner`].
//! They do not start a daemon or move tool execution into a webview.

use std::{
    collections::HashMap,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use llama_harness_core::{
    ApprovalHandler, ApprovalRecord, EventRecord, EventSink, HarnessError, RunRequest,
    ToolCallContext, ToolDefinition,
};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Runtime};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub use tauri::{AppHandle as TauriAppHandle, Runtime as TauriRuntime};

/// Event name used by [`TauriEventSink`] unless a host supplies another name.
pub const DEFAULT_RUN_EVENT_NAME: &str = "llama-harness://run-event";
/// Event name used by [`ApprovalRouter`] unless a host supplies another name.
pub const DEFAULT_APPROVAL_EVENT_NAME: &str = "llama-harness://approval-requested";

/// Emits serializable payloads to a frontend. The small trait keeps the routing
/// and approval helpers testable without creating a Tauri application.
pub trait FrontendEmitter: Send + Sync + Clone + 'static {
    fn emit<P: Serialize + Clone>(&self, event: &str, payload: P) -> Result<(), String>;
}

/// Broadcast [`FrontendEmitter`] implementation backed by a Tauri app handle.
///
/// This compatibility adapter sends every payload to every webview. Do not use
/// it for sensitive multi-window run or approval events; use
/// [`TauriTargetEmitter`] with an explicit window label instead.
pub struct TauriEmitter<R: Runtime> {
    app: AppHandle<R>,
}

/// [`FrontendEmitter`] implementation that sends every payload to one named
/// Tauri target through [`Emitter::emit_to`].
pub struct TauriTargetEmitter<R: Runtime> {
    app: AppHandle<R>,
    target: String,
}

impl<R: Runtime> Clone for TauriTargetEmitter<R> {
    fn clone(&self) -> Self {
        Self {
            app: self.app.clone(),
            target: self.target.clone(),
        }
    }
}

impl<R: Runtime> TauriTargetEmitter<R> {
    pub fn new(app: AppHandle<R>, target: impl Into<String>) -> Self {
        Self {
            app,
            target: target.into(),
        }
    }

    pub fn target(&self) -> &str {
        &self.target
    }
}

impl<R: Runtime> FrontendEmitter for TauriTargetEmitter<R> {
    fn emit<P: Serialize + Clone>(&self, event: &str, payload: P) -> Result<(), String> {
        self.app
            .emit_to(&self.target, event, payload)
            .map_err(|error| error.to_string())
    }
}

impl<R: Runtime> Clone for TauriEmitter<R> {
    fn clone(&self) -> Self {
        Self {
            app: self.app.clone(),
        }
    }
}

impl<R: Runtime> TauriEmitter<R> {
    pub fn new(app: AppHandle<R>) -> Self {
        Self { app }
    }
}

impl<R: Runtime> FrontendEmitter for TauriEmitter<R> {
    fn emit<P: Serialize + Clone>(&self, event: &str, payload: P) -> Result<(), String> {
        self.app
            .emit(event, payload)
            .map_err(|error| error.to_string())
    }
}

/// Forwards each redacted canonical event to a frontend event channel.
#[derive(Clone)]
pub struct TauriEventSink<E: FrontendEmitter> {
    emitter: E,
    event_name: String,
    last_error: Arc<Mutex<Option<String>>>,
}

impl<E: FrontendEmitter> TauriEventSink<E> {
    pub fn new(emitter: E) -> Self {
        Self::with_event_name(emitter, DEFAULT_RUN_EVENT_NAME)
    }

    pub fn with_event_name(emitter: E, event_name: impl Into<String>) -> Self {
        Self {
            emitter,
            event_name: event_name.into(),
            last_error: Arc::new(Mutex::new(None)),
        }
    }

    /// Returns the latest frontend emission error. Event delivery failures do
    /// not make the canonical runner fail or expose event payloads in logs.
    pub fn last_emit_error(&self) -> Option<String> {
        self.last_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl<E: FrontendEmitter> EventSink for TauriEventSink<E> {
    fn emit(&self, record: EventRecord) {
        if let Err(error) = self.emitter.emit(&self.event_name, record) {
            *self
                .last_error
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
        }
    }
}

/// Sends one event record to every registered sink. This is useful for a
/// Tauri frontend plus local redacted SQLite persistence.
#[derive(Clone, Default)]
pub struct FanoutEventSink {
    sinks: Vec<Arc<dyn EventSink>>,
}

impl FanoutEventSink {
    pub fn new(sinks: impl IntoIterator<Item = Arc<dyn EventSink>>) -> Self {
        Self {
            sinks: sinks.into_iter().collect(),
        }
    }
}

impl EventSink for FanoutEventSink {
    fn emit(&self, record: EventRecord) {
        for sink in &self.sinks {
            sink.emit(record.clone());
        }
    }
}

/// Tracks embedded runs by their application-visible ID for Tauri commands.
#[derive(Default)]
pub struct RunRegistry {
    runs: Mutex<HashMap<String, CancellationToken>>,
}

impl RunRegistry {
    /// Assigns a run ID when needed and makes its existing cancellation token
    /// available to `cancel`. Call [`Self::complete`] after `AgentRunner::run`.
    pub fn register(&self, request: &mut RunRequest) -> Result<String, RunRegistryError> {
        let run_id = request
            .run_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let mut runs = self
            .runs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if runs.contains_key(&run_id) {
            return Err(RunRegistryError::DuplicateRun(run_id));
        }
        request.run_id = Some(run_id.clone());
        runs.insert(run_id.clone(), request.cancellation.clone());
        Ok(run_id)
    }

    /// Requests cooperative cancellation. Side effects that already started in
    /// an application-owned tool cannot be reversed.
    pub fn cancel(&self, run_id: &str) -> bool {
        self.runs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(run_id)
            .map(|token| token.cancel())
            .is_some()
    }

    pub fn complete(&self, run_id: &str) -> bool {
        self.runs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(run_id)
            .is_some()
    }

    pub fn cancel_all(&self) {
        for token in self
            .runs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
        {
            token.cancel();
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RunRegistryError {
    #[error("run {0} is already registered")]
    DuplicateRun(String),
}

/// Payload emitted to the frontend when the canonical runner requests a user
/// approval. The frontend returns the opaque `approval_id` to
/// [`ApprovalRouter::respond`].
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingApproval {
    pub approval_id: String,
    pub run_id: String,
    pub trace_id: String,
    pub call_id: String,
    pub tool: ToolDefinition,
    pub arguments: serde_json::Value,
}

struct PendingApprovalEntry {
    run_id: String,
    sender: oneshot::Sender<ApprovalRecord>,
}

/// Bounded approval-routing configuration. Defaults are intentionally small
/// enough to fail closed when a host loses its approval surface.
#[derive(Clone, Debug)]
pub struct ApprovalRouterSettings {
    pub max_pending: usize,
    pub timeout: std::time::Duration,
    pub max_approval_id_bytes: usize,
    pub max_reason_bytes: usize,
}

impl Default for ApprovalRouterSettings {
    fn default() -> Self {
        Self {
            max_pending: 64,
            timeout: std::time::Duration::from_secs(5 * 60),
            max_approval_id_bytes: 128,
            max_reason_bytes: 1024,
        }
    }
}

/// Turns canonical approval requests into frontend events while preserving a
/// Rust-owned, opaque correlation ID. It never grants an approval by default.
#[derive(Clone)]
pub struct ApprovalRouter<E: FrontendEmitter> {
    emitter: E,
    event_name: String,
    pending: Arc<Mutex<HashMap<String, PendingApprovalEntry>>>,
    settings: ApprovalRouterSettings,
}

impl<E: FrontendEmitter> ApprovalRouter<E> {
    pub fn new(emitter: E) -> Self {
        Self::with_event_name(emitter, DEFAULT_APPROVAL_EVENT_NAME)
    }

    pub fn with_event_name(emitter: E, event_name: impl Into<String>) -> Self {
        Self::with_settings(emitter, event_name, ApprovalRouterSettings::default())
    }

    pub fn with_settings(
        emitter: E,
        event_name: impl Into<String>,
        settings: ApprovalRouterSettings,
    ) -> Self {
        Self {
            emitter,
            event_name: event_name.into(),
            pending: Arc::new(Mutex::new(HashMap::new())),
            settings,
        }
    }

    pub fn pending_count(&self) -> usize {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Supplies a user's decision for an emitted approval. Returns false for a
    /// stale, cancelled, or previously consumed request.
    pub fn respond(&self, approval_id: &str, granted: bool, reason: impl Into<String>) -> bool {
        if approval_id.len() > self.settings.max_approval_id_bytes {
            return false;
        }
        let reason = reason.into();
        if reason.len() > self.settings.max_reason_bytes {
            return false;
        }
        let entry = self
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(approval_id);
        entry
            .map(|entry| {
                entry
                    .sender
                    .send(ApprovalRecord {
                        call_id: String::new(),
                        tool_id: String::new(),
                        granted,
                        reason,
                    })
                    .is_ok()
            })
            .unwrap_or(false)
    }

    /// Fails unresolved approvals for a run. Invoke this from the same host
    /// cancellation/shutdown path that cancels the [`RunRegistry`].
    pub fn cancel_run(&self, run_id: &str) -> usize {
        let ids = self
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter(|(_, entry)| entry.run_id == run_id)
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        let mut cancelled = 0;
        for id in ids {
            if self
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&id)
                .is_some()
            {
                cancelled += 1;
            }
        }
        cancelled
    }

    pub fn cancel_all(&self) -> usize {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let count = pending.len();
        pending.clear();
        count
    }
}

#[async_trait]
impl<E: FrontendEmitter> ApprovalHandler for ApprovalRouter<E> {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        _: &serde_json::Value,
        _: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        Ok(ApprovalRecord {
            call_id: String::new(),
            tool_id: tool.id.clone(),
            granted: false,
            reason: "approval routing requires a tool call context".into(),
        })
    }

    async fn approve_with_context(
        &self,
        context: &ToolCallContext,
        tool: &ToolDefinition,
        arguments: &serde_json::Value,
        request: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        let approval_id = Uuid::new_v4().to_string();
        let (sender, receiver) = oneshot::channel();
        {
            let mut pending = self
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if pending.len() >= self.settings.max_pending {
                return Err(HarnessError::ResourceLimit(format!(
                    "pending approvals exceed {}",
                    self.settings.max_pending
                )));
            }
            pending.insert(
                approval_id.clone(),
                PendingApprovalEntry {
                    run_id: context.run_id.clone(),
                    sender,
                },
            );
        }
        let payload = PendingApproval {
            approval_id: approval_id.clone(),
            run_id: context.run_id.clone(),
            trace_id: context.trace_id.clone(),
            call_id: context.call_id.clone(),
            tool: tool.clone(),
            arguments: arguments.clone(),
        };
        if let Err(error) = self.emitter.emit(&self.event_name, payload) {
            self.pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&approval_id);
            return Err(HarnessError::Approval(format!(
                "could not emit approval request: {error}"
            )));
        }
        let outcome = tokio::select! {
            result = receiver => result.map_err(|_| HarnessError::Cancelled),
            _ = request.cancellation.cancelled() => Err(HarnessError::Cancelled),
            _ = tokio::time::sleep(self.settings.timeout) => Err(HarnessError::TimedOut("approval response".into())),
        };
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&approval_id);
        let record = outcome?;
        Ok(ApprovalRecord {
            call_id: context.call_id.clone(),
            tool_id: context.tool_id.clone(),
            granted: record.granted,
            reason: record.reason,
        })
    }
}

/// Returns an application-controlled SQLite filename beneath a selected Tauri
/// data directory. Rejects roots, `..`, and multi-component names so callers
/// cannot accidentally persist traces outside their chosen application data.
pub fn trace_database_path(
    data_directory: impl AsRef<Path>,
    file_name: impl AsRef<Path>,
) -> Result<PathBuf, TracePathError> {
    let file_name = file_name.as_ref();
    if file_name.as_os_str().is_empty()
        || file_name.components().count() != 1
        || file_name
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(TracePathError::InvalidFileName);
    }
    let path = data_directory.as_ref().join(file_name);
    if path.extension().and_then(|value| value.to_str()) != Some("sqlite") {
        return Err(TracePathError::UnsupportedExtension);
    }
    Ok(path)
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TracePathError {
    #[error("trace database file name must be one relative file name")]
    InvalidFileName,
    #[error("trace database must use the .sqlite extension")]
    UnsupportedExtension,
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use llama_harness_core::{
        AgentDefinition, EventRecord, InMemoryEventSink, RunEvent, RunRequest, ToolRisk,
    };

    #[derive(Clone, Default)]
    struct TestEmitter(Arc<Mutex<Vec<(String, serde_json::Value)>>>);

    impl FrontendEmitter for TestEmitter {
        fn emit<P: Serialize + Clone>(&self, event: &str, payload: P) -> Result<(), String> {
            self.0
                .lock()
                .unwrap()
                .push((event.into(), serde_json::to_value(payload).unwrap()));
            Ok(())
        }
    }

    fn tool() -> ToolDefinition {
        ToolDefinition {
            id: "notes.write".into(),
            name: "Write note".into(),
            description: "Writes a note".into(),
            arguments_schema: serde_json::json!({"type":"object"}),
            risk: ToolRisk::Medium,
            idempotent: false,
            read_only: false,
        }
    }

    #[test]
    fn registry_assigns_and_cancels_a_run() {
        let registry = RunRegistry::default();
        let mut request =
            RunRequest::new(AgentDefinition::new("agent", "Agent", "1", "mock"), "hello");
        let id = registry.register(&mut request).unwrap();
        assert_eq!(request.run_id.as_deref(), Some(id.as_str()));
        assert!(registry.cancel(&id));
        assert!(request.cancellation.is_cancelled());
        assert!(registry.complete(&id));
        assert!(!registry.cancel(&id));
    }

    #[test]
    fn fanout_preserves_the_same_event_for_each_sink() {
        let first = Arc::new(InMemoryEventSink::default());
        let second = Arc::new(InMemoryEventSink::default());
        let sink = FanoutEventSink::new([
            Arc::clone(&first) as Arc<dyn EventSink>,
            Arc::clone(&second) as Arc<dyn EventSink>,
        ]);
        let record = EventRecord {
            run_id: "run".into(),
            trace_id: "trace".into(),
            sequence: 1,
            timestamp_ms: 1,
            event: RunEvent::Started {
                run_id: "run".into(),
                trace_id: "trace".into(),
            },
        };
        sink.emit(record.clone());
        assert_eq!(first.events(), vec![record.clone()]);
        assert_eq!(second.events(), vec![record]);
    }

    #[tokio::test]
    async fn approval_router_uses_opaque_one_time_ids() {
        let emitter = TestEmitter::default();
        let router = ApprovalRouter::new(emitter.clone());
        let context = ToolCallContext {
            run_id: "run".into(),
            trace_id: "trace".into(),
            call_id: "call".into(),
            tool_id: "notes.write".into(),
        };
        let request = RunRequest::new(AgentDefinition::new("agent", "Agent", "1", "mock"), "hello");
        let definition = tool();
        let arguments = serde_json::json!({"title":"One"});
        let waiting = router.approve_with_context(&context, &definition, &arguments, &request);
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(1), &mut waiting)
                .await
                .is_err()
        );
        let event = emitter.0.lock().unwrap()[0].1.clone();
        let id = event["approvalId"].as_str().unwrap().to_owned();
        assert!(router.respond(&id, true, "approved"));
        let record = waiting.await.unwrap();
        assert!(record.granted);
        assert_eq!(record.call_id, "call");
        assert!(!router.respond(&id, true, "duplicate"));
    }

    #[tokio::test]
    async fn approval_router_fails_closed_on_capacity_timeout_and_cancellation() {
        let emitter = TestEmitter::default();
        let router = ApprovalRouter::with_settings(
            emitter,
            DEFAULT_APPROVAL_EVENT_NAME,
            ApprovalRouterSettings {
                max_pending: 1,
                timeout: std::time::Duration::from_millis(50),
                ..ApprovalRouterSettings::default()
            },
        );
        let context = ToolCallContext {
            run_id: "run".into(),
            trace_id: "trace".into(),
            call_id: "call".into(),
            tool_id: "notes.write".into(),
        };
        let request = RunRequest::new(AgentDefinition::new("agent", "Agent", "1", "mock"), "hello");
        let definition = tool();
        let arguments = serde_json::json!({});
        let waiting = router.approve_with_context(&context, &definition, &arguments, &request);
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(1), &mut waiting)
                .await
                .is_err()
        );
        assert_eq!(router.pending_count(), 1);
        assert!(matches!(
            router
                .approve_with_context(&context, &definition, &arguments, &request)
                .await,
            Err(HarnessError::ResourceLimit(_))
        ));
        assert!(matches!(waiting.await, Err(HarnessError::TimedOut(_))));
        assert_eq!(router.pending_count(), 0);

        let cancelling = router.approve_with_context(&context, &definition, &arguments, &request);
        tokio::pin!(cancelling);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(1), &mut cancelling)
                .await
                .is_err()
        );
        request.cancellation.cancel();
        assert!(matches!(cancelling.await, Err(HarnessError::Cancelled)));
        assert_eq!(router.pending_count(), 0);
    }

    #[test]
    fn approval_router_rejects_oversize_response_fields() {
        let router = ApprovalRouter::with_settings(
            TestEmitter::default(),
            DEFAULT_APPROVAL_EVENT_NAME,
            ApprovalRouterSettings {
                max_approval_id_bytes: 2,
                max_reason_bytes: 2,
                ..ApprovalRouterSettings::default()
            },
        );
        assert!(!router.respond("too-long", true, "ok"));
        assert!(!router.respond("ok", true, "too-long"));
    }

    #[test]
    fn trace_path_rejects_escape_attempts() {
        assert_eq!(
            trace_database_path("data", "traces.sqlite").unwrap(),
            PathBuf::from("data").join("traces.sqlite")
        );
        assert_eq!(
            trace_database_path("data", "nested/traces.sqlite"),
            Err(TracePathError::InvalidFileName)
        );
        assert_eq!(
            trace_database_path("data", "traces.db"),
            Err(TracePathError::UnsupportedExtension)
        );
    }
}
