//! Optional, provider-neutral Model Context Protocol tool integration.
//!
//! This crate imports a complete validated MCP catalog through the ordinary
//! `ToolRegistry` boundary. It intentionally contains no JSON-RPC client.

#![deny(missing_docs)]

use async_trait::async_trait;
use llama_harness_core::{
    CancellationSafety, ExecutionLocation, GroupToolRegistration, HarnessError, NetworkEgress,
    SpeculationPolicy, Tool, ToolCallContext, ToolCaller, ToolDefinition, ToolDiscoveryMetadata,
    ToolRegistrationGroup, ToolRegistry, ToolResult, ToolRisk,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, OnceLock, Weak,
    },
    time::{Duration, Instant},
};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

const MAX_SERVER_GATES: usize = 256;

struct ServerCallGate {
    permits: tokio::sync::Semaphore,
    waiters: AtomicUsize,
    max_waiters: usize,
}

struct ServerWaiter<'a>(&'a AtomicUsize);

impl Drop for ServerWaiter<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

static SERVER_CALL_GATES: OnceLock<Mutex<HashMap<String, Weak<ServerCallGate>>>> = OnceLock::new();

fn server_call_gate_with_limit(
    server_id: &str,
    max_gates: usize,
    max_waiters: usize,
) -> Result<Arc<ServerCallGate>, McpError> {
    let gates = SERVER_CALL_GATES.get_or_init(|| Mutex::new(HashMap::new()));
    server_call_gate_from(gates, server_id, max_gates, max_waiters)
}

fn server_call_gate_from(
    gates: &Mutex<HashMap<String, Weak<ServerCallGate>>>,
    server_id: &str,
    max_gates: usize,
    max_waiters: usize,
) -> Result<Arc<ServerCallGate>, McpError> {
    validate_server_waiter_limit(max_waiters)?;
    let mut gates = gates.lock().expect("MCP server gate mutex");
    gates.retain(|_, gate| gate.strong_count() != 0);
    if let Some(gate) = gates.get(server_id).and_then(Weak::upgrade) {
        if gate.max_waiters != max_waiters {
            return Err(McpError::InvalidConfiguration(
                "all live managers for one server must use the same max_server_waiters".into(),
            ));
        }
        return Ok(gate);
    }
    if gates.len() >= max_gates {
        return Err(McpError::ResourceLimit);
    }
    let gate = Arc::new(ServerCallGate {
        permits: tokio::sync::Semaphore::new(1),
        waiters: AtomicUsize::new(0),
        max_waiters,
    });
    gates.insert(server_id.to_owned(), Arc::downgrade(&gate));
    Ok(gate)
}

fn server_call_gate(server_id: &str, max_waiters: usize) -> Result<Arc<ServerCallGate>, McpError> {
    server_call_gate_with_limit(server_id, MAX_SERVER_GATES, max_waiters)
}

/// Explicit MCP protocol era negotiated with a server.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum McpProtocolEra {
    /// Stateless discovery-era protocol.
    Modern20260728,
    /// Initialize-era protocol.
    Legacy20251125,
}
/// Negotiated protocol context. Per-request context is retained for modern servers.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct McpContext {
    /// Protocol era.
    pub era: McpProtocolEra,
    /// Negotiated version.
    pub version: String,
    /// Server capability names.
    pub capabilities: BTreeSet<String>,
    /// Opaque transport context; never placed in tool metadata.
    pub request_context: Option<String>,
}
/// Lifecycle operation whose transport dispatch failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpDispatchState {
    /// No request was dispatched.
    NotDispatched,
    /// Request might have reached the server.
    PossiblyDispatched,
    /// Server responded.
    Responded,
}
/// Sanitized typed transport failure.
#[derive(Clone, Debug, Error)]
#[error("MCP transport {operation:?} failed after {dispatch:?}")]
pub struct McpTransportError {
    /// Operation class.
    pub operation: McpOperation,
    /// Dispatch certainty.
    pub dispatch: McpDispatchState,
}
/// Transport operation class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum McpOperation {
    /// Negotiation.
    Connect,
    /// Tool listing.
    ListTools,
    /// Tool invocation.
    CallTool,
    /// Shutdown.
    Close,
}
/// Semantic transport boundary; implementations own JSON-RPC and credentials.
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Implementations MUST enforce their host-controlled [`McpWireLimits`]
    /// before and while reading and parsing wire frames. This semantic
    /// boundary receives values that have already been allocated by the host;
    /// the catalog manager separately performs bounded, non-recursive
    /// admission before retaining or cloning any server values.
    /// Negotiates one supported MCP context.
    async fn connect(
        &self,
        cancellation: CancellationToken,
    ) -> Result<McpContext, McpTransportError>;
    /// Returns one page from a tools/list catalog for the negotiated context.
    async fn list_tools(
        &self,
        context: &McpContext,
        cursor: Option<&str>,
        cancellation: CancellationToken,
    ) -> Result<McpToolPage, McpTransportError>;
    /// Executes exactly one native tool call; callers never retry automatically.
    async fn call_tool(
        &self,
        context: &McpContext,
        request: McpCallRequest,
        cancellation: CancellationToken,
    ) -> Result<McpCallResult, McpTransportError>;
    /// Closes the transport context.
    async fn close(
        &self,
        context: McpContext,
        cancellation: CancellationToken,
    ) -> Result<(), McpTransportError>;
}

/// Host-side limits for raw MCP wire frames.
///
/// These limits are intentionally not passed through [`McpTransport`]: the
/// transport owns framing and parsing. Implementers must apply equivalent
/// limits before semantic values cross that boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct McpWireLimits {
    /// Maximum encoded frame size accepted by the transport.
    pub max_frame_bytes: usize,
    /// Maximum JSON nesting accepted while parsing a frame.
    pub max_json_depth: usize,
}

impl Default for McpWireLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 512 * 1024,
            max_json_depth: 32,
        }
    }
}

/// Closes a negotiated context with a bounded deadline. Hosts should call this
/// after a catalog is no longer used; it never exposes transport details.
pub async fn close_context(
    transport: &dyn McpTransport,
    context: McpContext,
    cancellation: CancellationToken,
    timeout: Duration,
) -> Result<(), McpTransportError> {
    tokio::time::timeout(timeout, transport.close(context, cancellation))
        .await
        .map_err(|_| McpTransportError {
            operation: McpOperation::Close,
            dispatch: McpDispatchState::PossiblyDispatched,
        })?
}
/// One remote tool declaration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpTool {
    /// Native server tool name.
    pub name: String,
    /// Untrusted display description.
    pub description: String,
    /// JSON Schema 2020-12 input schema.
    pub input_schema: Value,
    /// Optional JSON Schema output schema.
    pub output_schema: Option<Value>,
}
/// One bounded tools/list response page.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpToolPage {
    /// Tools.
    pub tools: Vec<McpTool>,
    /// Opaque next cursor.
    pub next_cursor: Option<String>,
    /// Modern cache TTL hint.
    pub ttl_ms: Option<u64>,
    /// Modern cache-scope hint.
    pub cache_scope: Option<String>,
}
/// Remote call request.
#[derive(Clone, Debug)]
pub struct McpCallRequest {
    /// Exact native tool name.
    pub name: String,
    /// Validated JSON arguments.
    pub arguments: Value,
    /// Core correlation only.
    pub context: ToolCallContext,
}
/// Remote call response.
#[derive(Clone, Debug)]
pub struct McpCallResult {
    /// Structured result when supplied.
    pub structured_content: Option<Value>,
    /// Unstructured content normalized to JSON.
    pub content: Option<Value>,
    /// Server-reported tool failure.
    pub is_error: bool,
}
/// Conservative import bounds.
#[derive(Clone, Debug)]
pub struct McpLimits {
    /// Max list pages.
    pub max_pages: usize,
    /// Max tools.
    pub max_tools: usize,
    /// Max schema/catalog bytes.
    pub max_catalog_bytes: usize,
    /// Max schema JSON depth.
    pub max_json_depth: usize,
    /// Max aggregate bytes accepted from one tool response.
    pub max_response_bytes: usize,
    /// Max unstructured MCP content blocks accepted from one response.
    pub max_content_blocks: usize,
    /// Maximum JSON nodes admitted from any one server value.
    pub max_json_nodes: usize,
    /// Maximum object properties admitted from any one server value.
    pub max_schema_properties: usize,
    /// Maximum UTF-8 bytes in any admitted string.
    pub max_string_bytes: usize,
    /// Maximum negotiated capability names admitted from one context.
    pub max_context_capabilities: usize,
    /// Maximum aggregate pending, retired, and in-progress contexts retained
    /// by this manager. Refresh fails before connecting when this bound is
    /// exhausted.
    pub max_retired_generations: usize,
    /// Maximum queued local calls for one server before calls fail closed.
    ///
    /// The value must be in `1..usize::MAX`. The first live manager for a
    /// canonical server ID establishes the process-wide bound; every other
    /// live manager for that server must configure the same value.
    pub max_server_waiters: usize,
}

/// Bounded lifecycle deadlines. Calls are never retried after a timeout.
#[derive(Clone, Debug)]
pub struct McpTimeouts {
    /// Connection and negotiation deadline.
    pub connect: Duration,
    /// Per-page listing deadline.
    pub list: Duration,
    /// Per-tool invocation deadline.
    pub call: Duration,
    /// Shutdown deadline.
    pub close: Duration,
}

impl Default for McpTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(10),
            list: Duration::from_secs(10),
            call: Duration::from_secs(30),
            close: Duration::from_secs(5),
        }
    }
}
impl Default for McpLimits {
    fn default() -> Self {
        Self {
            max_pages: 32,
            max_tools: 256,
            max_catalog_bytes: 512 * 1024,
            max_json_depth: 32,
            max_response_bytes: 512 * 1024,
            max_content_blocks: 128,
            max_json_nodes: 16 * 1024,
            max_schema_properties: 8 * 1024,
            max_string_bytes: 64 * 1024,
            max_context_capabilities: 128,
            max_retired_generations: 8,
            max_server_waiters: 32,
        }
    }
}
/// Errors while negotiating or importing a catalog.
#[derive(Debug, Error)]
pub enum McpError {
    /// Transport failure.
    #[error(transparent)]
    Transport(#[from] McpTransportError),
    /// Untrusted server data was rejected.
    #[error("invalid MCP catalog: {0}")]
    InvalidCatalog(String),
    /// Host configuration is invalid or conflicts with live process-global state.
    #[error("invalid MCP configuration: {0}")]
    InvalidConfiguration(String),
    /// Core registration failed.
    #[error(transparent)]
    Core(#[from] HarnessError),
    /// A catalog is unavailable, expired, invalidated, or already closed.
    #[error("MCP catalog unavailable")]
    CatalogUnavailable,
    /// Shutdown cleanup is still owned by an in-progress drain worker.
    #[error("MCP shutdown cleanup pending")]
    CleanupPending,
    /// A local admission bound was exhausted before transport dispatch.
    #[error("MCP resource limit exceeded")]
    ResourceLimit,
    /// The configured clock was non-monotonic or overflowed a deadline.
    #[error("MCP catalog clock failure")]
    Clock,
}

/// Source of monotonically increasing milliseconds for catalog cache decisions.
pub trait McpClock: Send + Sync {
    /// Returns the current monotonic millisecond tick.
    fn now_ms(&self) -> u64;
}

#[derive(Debug)]
struct SystemMcpClock {
    origin: Instant,
}

impl Default for SystemMcpClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl McpClock for SystemMcpClock {
    fn now_ms(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

/// Host policy for legacy cache age and explicit stale use.
#[derive(Clone, Debug)]
pub struct McpCachePolicy {
    /// Host-assigned cache lifetime for legacy protocol catalogs.
    pub legacy_ttl: Duration,
    /// Additional time an expired catalog may be used. `None` fails closed.
    pub max_stale: Option<Duration>,
}

impl Default for McpCachePolicy {
    fn default() -> Self {
        Self {
            legacy_ttl: Duration::from_secs(300),
            max_stale: None,
        }
    }
}

/// Sanitized cache scope supplied by a modern server.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpCacheScope {
    /// Cache entry can be shared across authorization contexts.
    Public,
}

impl McpCacheScope {
    fn parse(value: &str) -> Result<Self, McpError> {
        match value {
            "public" => Ok(Self::Public),
            _ => Err(McpError::InvalidCatalog(
                "unsupported or private cache scope".into(),
            )),
        }
    }
}

/// Stable lifecycle operation class for metadata-only observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpLifecycleOperation {
    /// Protocol negotiation.
    Negotiation,
    /// Complete catalog refresh.
    Refresh,
    /// Native tool dispatch.
    Call,
    /// Transport shutdown.
    Close,
}

/// Stable, value-free lifecycle outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpLifecycleOutcome {
    /// Operation completed successfully.
    Succeeded,
    /// Operation failed without exposing server error text.
    Failed,
    /// Operation exceeded its bounded deadline.
    TimedOut,
    /// Operation was cancelled.
    Cancelled,
    /// Local lifecycle policy rejected the operation before dispatch.
    Rejected,
}

/// Non-sensitive call correlation preserved from the core call context.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct McpCallCorrelation {
    /// Run correlation identifier.
    pub run_id: String,
    /// Trace correlation identifier.
    pub trace_id: String,
    /// Call correlation identifier.
    pub call_id: String,
}

impl From<&ToolCallContext> for McpCallCorrelation {
    fn from(value: &ToolCallContext) -> Self {
        Self {
            run_id: value.run_id.clone(),
            trace_id: value.trace_id.clone(),
            call_id: value.call_id.clone(),
        }
    }
}

/// Metadata-only MCP lifecycle event. It deliberately contains no server or tool metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpLifecycleEvent {
    /// Operation class.
    pub operation: McpLifecycleOperation,
    /// Stable local outcome.
    pub outcome: McpLifecycleOutcome,
    /// Local elapsed duration in milliseconds.
    pub duration_ms: u64,
    /// Number of accepted catalog entries or content blocks.
    pub count: u64,
    /// Number of normalized bytes.
    pub bytes: u64,
    /// Number of catalog pages.
    pub pages: u64,
    /// Whether the invocation used explicitly permitted stale catalog data.
    pub stale: bool,
    /// Whether cancellation was observed.
    pub cancelled: bool,
    /// Dispatch certainty.
    pub dispatch: McpDispatchState,
    /// Correlation for a call, when available.
    pub correlation: Option<McpCallCorrelation>,
}

/// Value-free observer failure.
#[derive(Clone, Copy, Debug, Error)]
#[error("MCP observer failed")]
pub struct McpObserverError;

/// Receives metadata-only MCP lifecycle events.
///
/// Implementations must be nonblocking and complete in constant time. The
/// manager invokes this callback after releasing lifecycle and dispatch
/// ownership; slow observers still delay their own caller and are therefore
/// treated as a host integration fault rather than a flow-control mechanism.
pub trait McpObserver: Send + Sync {
    /// Observes one lifecycle event. Errors never affect MCP execution.
    fn observe(&self, event: &McpLifecycleEvent) -> Result<(), McpObserverError>;
}

/// Cumulative health of lifecycle observation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct McpObserverHealth {
    /// Events offered to the observer.
    pub attempted: u64,
    /// Observer failures suppressed by the adapter.
    pub failures: u64,
    /// Observer callbacks exceeding the small host-integration budget.
    pub slow: u64,
}

/// Local shutdown cleanup health without transport details.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct McpShutdownHealth {
    /// Contexts retained for a later independent shutdown retry.
    pub pending_contexts: u64,
    /// Contexts owned by deferred drain workers.
    pub in_progress_contexts: u64,
    /// Bounded close attempts that did not complete.
    pub close_failures: u64,
}

#[derive(Default)]
struct ObserverHub {
    observer: Option<Arc<dyn McpObserver>>,
    attempted: AtomicU64,
    failures: AtomicU64,
    slow: AtomicU64,
}

impl ObserverHub {
    fn emit(&self, event: McpLifecycleEvent) {
        if let Some(observer) = &self.observer {
            self.attempted.fetch_add(1, Ordering::Relaxed);
            let started = Instant::now();
            if catch_unwind(AssertUnwindSafe(|| observer.observe(&event)))
                .map_or(true, |result| result.is_err())
            {
                self.failures.fetch_add(1, Ordering::Relaxed);
            }
            if started.elapsed() > Duration::from_millis(10) {
                self.slow.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn health(&self) -> McpObserverHealth {
        McpObserverHealth {
            attempted: self.attempted.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
            slow: self.slow.load(Ordering::Relaxed),
        }
    }
}

/// Immutable, locally safe summary of the active catalog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct McpCatalogSnapshot {
    /// Monotonically increasing local generation.
    pub generation: u64,
    /// Negotiated protocol era.
    pub era: McpProtocolEra,
    /// Modern cache scope, if negotiated.
    pub cache_scope: Option<McpCacheScope>,
    /// Number of tools in the validated catalog.
    pub tool_count: usize,
}

struct ActiveCatalog {
    summary: McpCatalogSnapshot,
    context: Arc<McpContext>,
    in_flight: Arc<InFlightCalls>,
    dispatch_gate: Arc<tokio::sync::RwLock<()>>,
    expires_at_ms: u64,
    stale_until_ms: u64,
    tools: Vec<Arc<ManagedMcpToolAdapter>>,
}

#[derive(Default)]
struct InFlightCalls {
    count: AtomicU64,
    drained: tokio::sync::Notify,
}

struct InFlightLease(Arc<InFlightCalls>);

impl Drop for InFlightLease {
    fn drop(&mut self) {
        if self.0.count.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.drained.notify_waiters();
        }
    }
}

async fn wait_for_drain(calls: &InFlightCalls) {
    loop {
        if calls.count.load(Ordering::Acquire) == 0 {
            return;
        }
        let notified = calls.drained.notified();
        if calls.count.load(Ordering::Acquire) == 0 {
            return;
        }
        notified.await;
    }
}

struct RetiredContext {
    context: Arc<McpContext>,
    in_flight: Arc<InFlightCalls>,
}

struct CatalogState {
    active: Option<Arc<ActiveCatalog>>,
    closed: bool,
    invalidated: bool,
    last_now_ms: u64,
    next_generation: u64,
    pending_refresh_contexts: Vec<PendingRefreshContext>,
    refresh_tokens: HashSet<u64>,
    next_refresh_token: u64,
    retired_contexts: Vec<RetiredContext>,
    close_failures: u64,
    draining_contexts: u64,
}

type McpIdDeriver = dyn Fn(&str, &str) -> String + Send + Sync;

impl Default for CatalogState {
    fn default() -> Self {
        Self {
            active: None,
            closed: false,
            invalidated: false,
            last_now_ms: 0,
            next_generation: 1,
            pending_refresh_contexts: Vec::new(),
            refresh_tokens: HashSet::new(),
            next_refresh_token: 1,
            retired_contexts: Vec::new(),
            close_failures: 0,
            draining_contexts: 0,
        }
    }
}

struct PendingRefreshContext {
    token: u64,
    context: McpContext,
}

struct RefreshOwnership {
    state: Arc<Mutex<CatalogState>>,
    token: u64,
}

impl Drop for RefreshOwnership {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.refresh_tokens.remove(&self.token);
        }
    }
}

fn owned_context_count(state: &CatalogState) -> Option<usize> {
    state
        .pending_refresh_contexts
        .len()
        .checked_add(state.retired_contexts.len())?
        .checked_add(usize::try_from(state.draining_contexts).ok()?)
}

/// Owns complete immutable catalog snapshots for one MCP server.
///
/// A refresh validates every page before replacing the active snapshot. Tools
/// from an earlier snapshot consult this manager before dispatch and therefore
/// cannot call a server after replacement, invalidation, expiry, or close.
pub struct McpCatalogManager {
    server_id: String,
    transport: Arc<dyn McpTransport>,
    limits: McpLimits,
    timeouts: McpTimeouts,
    cache_policy: McpCachePolicy,
    clock: Arc<dyn McpClock>,
    state: Arc<Mutex<CatalogState>>,
    refresh_lock: tokio::sync::Mutex<()>,
    close_lock: tokio::sync::Mutex<()>,
    observer: Arc<ObserverHub>,
    registration_group: ToolRegistrationGroup,
    server_call_gate: Arc<ServerCallGate>,
    id_deriver: Arc<McpIdDeriver>,
}

impl McpCatalogManager {
    /// Creates a manager using the system monotonic clock and no observer.
    pub fn new(
        transport: Arc<dyn McpTransport>,
        server_id: impl Into<String>,
        limits: McpLimits,
    ) -> Result<Self, McpError> {
        Self::with_configuration(
            transport,
            server_id,
            limits,
            McpTimeouts::default(),
            McpCachePolicy::default(),
            Arc::new(SystemMcpClock::default()),
            None,
        )
    }

    /// Creates a manager with host-controlled lifecycle dependencies.
    pub fn with_configuration(
        transport: Arc<dyn McpTransport>,
        server_id: impl Into<String>,
        limits: McpLimits,
        timeouts: McpTimeouts,
        cache_policy: McpCachePolicy,
        clock: Arc<dyn McpClock>,
        observer: Option<Arc<dyn McpObserver>>,
    ) -> Result<Self, McpError> {
        let server_id = validate_server_id(server_id.into())?;
        validate_duration(cache_policy.legacy_ttl)?;
        if let Some(max_stale) = cache_policy.max_stale {
            validate_duration(max_stale)?;
        }
        let server_call_gate = server_call_gate(&server_id, limits.max_server_waiters)?;
        Ok(Self {
            server_id: server_id.clone(),
            transport,
            limits,
            timeouts,
            cache_policy,
            clock,
            state: Arc::new(Mutex::new(CatalogState::default())),
            refresh_lock: tokio::sync::Mutex::new(()),
            close_lock: tokio::sync::Mutex::new(()),
            observer: Arc::new(ObserverHub {
                observer,
                ..ObserverHub::default()
            }),
            registration_group: ToolRegistrationGroup::new(format!("mcp:{server_id}"))
                .map_err(McpError::Core)?,
            server_call_gate,
            id_deriver: Arc::new(generated_id),
        })
    }

    /// Refreshes a complete catalog and atomically installs it on success.
    pub async fn refresh(
        &self,
        cancellation: CancellationToken,
    ) -> Result<McpCatalogSnapshot, McpError> {
        let refresh_guard = self.refresh_lock.lock().await;
        let ownership = self.begin_refresh()?;
        let started = Instant::now();
        let mut lifecycle_events = Vec::with_capacity(2);
        let result = self
            .refresh_locked(
                ownership.token,
                cancellation,
                started,
                &mut lifecycle_events,
            )
            .await;
        drop(ownership);
        drop(refresh_guard);
        for lifecycle_event in lifecycle_events {
            self.observer.emit(lifecycle_event);
        }
        result
    }

    async fn refresh_locked(
        &self,
        token: u64,
        cancellation: CancellationToken,
        started: Instant,
        lifecycle_events: &mut Vec<McpLifecycleEvent>,
    ) -> Result<McpCatalogSnapshot, McpError> {
        let built = self
            .fetch_complete(token, cancellation.child_token(), lifecycle_events)
            .await;
        let elapsed = elapsed_ms(started);
        let (context, tools, scope, bytes, pages) = match built {
            Ok(value) => value,
            Err(error) => {
                let cancelled = cancellation.is_cancelled();
                lifecycle_events.push(event(
                    McpLifecycleOperation::Refresh,
                    if cancelled {
                        McpLifecycleOutcome::Cancelled
                    } else {
                        outcome_for_error(&error)
                    },
                    elapsed,
                    0,
                    0,
                    0,
                    false,
                    cancelled,
                    dispatch_for_error(&error),
                    None,
                ));
                return Err(error);
            }
        };
        let now = match self.checked_now() {
            Ok(now) => now,
            Err(error) => {
                return self
                    .fail_refresh_context(token, error, started, bytes, pages, lifecycle_events)
                    .await
            }
        };
        let ttl = match context.era {
            McpProtocolEra::Modern20260728 => match tools.cache_ttl {
                Some(ttl) => ttl,
                None => {
                    return self
                        .fail_refresh_context(
                            token,
                            McpError::InvalidCatalog("modern catalog omitted ttl".into()),
                            started,
                            bytes,
                            pages,
                            lifecycle_events,
                        )
                        .await
                }
            },
            McpProtocolEra::Legacy20251125 => match duration_ms(self.cache_policy.legacy_ttl) {
                Ok(ttl) => ttl,
                Err(error) => {
                    return self
                        .fail_refresh_context(token, error, started, bytes, pages, lifecycle_events)
                        .await
                }
            },
        };
        let expires_at_ms = match now.checked_add(ttl) {
            Some(value) => value,
            None => {
                return self
                    .fail_refresh_context(
                        token,
                        McpError::Clock,
                        started,
                        bytes,
                        pages,
                        lifecycle_events,
                    )
                    .await
            }
        };
        let max_stale = match self.cache_policy.max_stale.map(duration_ms).transpose() {
            Ok(value) => value.unwrap_or(0),
            Err(error) => {
                return self
                    .fail_refresh_context(token, error, started, bytes, pages, lifecycle_events)
                    .await
            }
        };
        let stale_until_ms = expires_at_ms.checked_add(max_stale).ok_or(McpError::Clock);
        let stale_until_ms = match stale_until_ms {
            Ok(value) => value,
            Err(error) => {
                return self
                    .fail_refresh_context(token, error, started, bytes, pages, lifecycle_events)
                    .await
            }
        };
        let generation = match (|| {
            let mut state = self
                .state
                .lock()
                .map_err(|_| McpError::CatalogUnavailable)?;
            if state.closed {
                return Err(McpError::CatalogUnavailable);
            }
            let generation = state.next_generation;
            state.next_generation = state
                .next_generation
                .checked_add(1)
                .ok_or(McpError::Clock)?;
            Ok(generation)
        })() {
            Ok(generation) => generation,
            Err(error) => {
                return self
                    .fail_refresh_context(token, error, started, bytes, pages, lifecycle_events)
                    .await
            }
        };
        let context = Arc::new(context);
        let in_flight = Arc::new(InFlightCalls::default());
        let dispatch_gate = Arc::new(tokio::sync::RwLock::new(()));
        let managed = tools
            .items
            .into_iter()
            .map(|tool| {
                Arc::new(ManagedMcpToolAdapter::new(
                    Arc::new(McpToolAdapter::new(
                        (self.id_deriver)(&self.server_id, &tool.name),
                        self.server_id.clone(),
                        tool,
                        Arc::clone(&context),
                        Arc::clone(&self.transport),
                        self.timeouts.call,
                        self.limits.clone(),
                        Arc::clone(&self.observer),
                        Arc::clone(&self.server_call_gate),
                    )),
                    generation,
                    Arc::downgrade(&self.state),
                    Arc::clone(&self.clock),
                    Arc::clone(&in_flight),
                    Arc::clone(&dispatch_gate),
                ))
            })
            .collect::<Vec<_>>();
        let mut generated_ids = HashSet::with_capacity(managed.len());
        if managed
            .iter()
            .any(|tool| !generated_ids.insert(tool.definition().id.clone()))
        {
            return self
                .fail_refresh_context(
                    token,
                    McpError::InvalidCatalog("generated ID collision".into()),
                    started,
                    bytes,
                    pages,
                    lifecycle_events,
                )
                .await;
        }
        let snapshot = Arc::new(ActiveCatalog {
            summary: McpCatalogSnapshot {
                generation,
                era: context.era,
                cache_scope: scope,
                tool_count: managed.len(),
            },
            context: Arc::clone(&context),
            in_flight,
            dispatch_gate,
            expires_at_ms,
            stale_until_ms,
            tools: managed,
        });
        let retiring = self.state.lock().ok().and_then(|state| {
            state
                .active
                .as_ref()
                .map(|active| Arc::clone(&active.dispatch_gate))
        });
        let _retire_permit = match retiring {
            Some(gate) => Some(gate.write_owned().await),
            None => None,
        };
        let previous_result = (|| {
            let mut state = self
                .state
                .lock()
                .map_err(|_| McpError::CatalogUnavailable)?;
            if state.closed {
                return Err(McpError::CatalogUnavailable);
            }
            if let Some(index) = state
                .pending_refresh_contexts
                .iter()
                .rposition(|candidate| candidate.token == token)
            {
                state.pending_refresh_contexts.remove(index);
            } else {
                return Err(McpError::CatalogUnavailable);
            }
            let previous = state.active.replace(Arc::clone(&snapshot));
            state.invalidated = false;
            Ok(previous)
        })();
        // Observer delivery and cleanup are deliberately outside the
        // retirement gate: observers are host code and never participate in
        // dispatch/lifecycle ownership.
        drop(_retire_permit);
        let previous = match previous_result {
            Ok(previous) => previous,
            Err(error) => {
                return self
                    .fail_refresh_context(token, error, started, bytes, pages, lifecycle_events)
                    .await
            }
        };
        if let Some(previous) = previous {
            self.schedule_close_after_drain(
                Arc::clone(&previous.context),
                Arc::clone(&previous.in_flight),
            );
        }
        lifecycle_events.push(event(
            McpLifecycleOperation::Refresh,
            McpLifecycleOutcome::Succeeded,
            elapsed,
            snapshot.summary.tool_count as u64,
            bytes,
            pages,
            false,
            false,
            McpDispatchState::Responded,
            None,
        ));
        Ok(snapshot.summary.clone())
    }

    /// Explicitly invalidates the current list after a server list-change notice.
    pub fn invalidate_list_change(&self) {
        // This uses the same state lock as post-permit call admission: work
        // already admitted may finish, but a call that observes this update
        // is rejected before it reaches the transport.
        if let Ok(mut state) = self.state.lock() {
            state.invalidated = true;
        }
    }

    /// Returns the current immutable snapshot summary, if still installed.
    pub fn active_snapshot(&self) -> Option<McpCatalogSnapshot> {
        self.state.lock().ok().and_then(|state| {
            state
                .active
                .as_ref()
                .map(|snapshot| snapshot.summary.clone())
        })
    }

    /// Returns the active adapters. Callers should register only this complete snapshot.
    pub fn active_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.state
            .lock()
            .ok()
            .and_then(|state| {
                state.active.as_ref().map(|snapshot| {
                    snapshot
                        .tools
                        .iter()
                        .cloned()
                        .map(|tool| tool as Arc<dyn Tool>)
                        .collect()
                })
            })
            .unwrap_or_default()
    }

    /// Returns a new registry with the active snapshot atomically replacing this manager's group.
    pub fn replace_registered(&self, registry: &ToolRegistry) -> Result<ToolRegistry, McpError> {
        let tools = self.active_tools();
        if tools.is_empty() {
            return Err(McpError::CatalogUnavailable);
        }
        registry
            .replace_group(
                &self.registration_group,
                tools.into_iter().map(|tool| {
                    GroupToolRegistration::new(tool, ToolDiscoveryMetadata::deferred())
                }),
            )
            .map_err(McpError::Core)
    }

    /// Closes the manager. The local state is closed before bounded transport shutdown.
    pub async fn close(&self, cancellation: CancellationToken) -> Result<(), McpError> {
        let _close = self.close_lock.lock().await;
        // Closing marks the state unavailable under the same lock used by a
        // call's post-permit admission check. An already admitted call keeps
        // its read permit and in-flight lease; close therefore stays bounded
        // instead of waiting for a remote call to return.
        let contexts = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| McpError::CatalogUnavailable)?;
            if !state.closed {
                state.closed = true;
                state.invalidated = true;
                if let Some(snapshot) = state.active.take() {
                    state.retired_contexts.push(RetiredContext {
                        context: Arc::clone(&snapshot.context),
                        in_flight: Arc::clone(&snapshot.in_flight),
                    });
                }
                for pending in std::mem::take(&mut state.pending_refresh_contexts) {
                    state.retired_contexts.push(RetiredContext {
                        context: Arc::new(pending.context),
                        in_flight: Arc::new(InFlightCalls::default()),
                    });
                }
            }
            state
                .retired_contexts
                .iter()
                .map(|entry| RetiredContext {
                    context: Arc::clone(&entry.context),
                    in_flight: Arc::clone(&entry.in_flight),
                })
                .collect::<Vec<_>>()
        };
        let started = Instant::now();
        if contexts.len() > 1 {
            for context in contexts {
                self.schedule_close_after_drain(context.context, context.in_flight);
            }
            drop(_close);
            self.observer.emit(event(
                McpLifecycleOperation::Close,
                McpLifecycleOutcome::Failed,
                elapsed_ms(started),
                0,
                0,
                0,
                false,
                cancellation.is_cancelled(),
                McpDispatchState::PossiblyDispatched,
                None,
            ));
            return Err(McpError::CleanupPending);
        }
        let mut failed = false;
        for context in contexts {
            if tokio::time::timeout(self.timeouts.close, wait_for_drain(&context.in_flight))
                .await
                .is_err()
            {
                self.schedule_close_after_drain(context.context, context.in_flight);
                failed = true;
                continue;
            }
            if !self.try_close_retained(&context).await {
                failed = true;
            }
        }
        let unresolved = self.state.lock().map_or(true, |state| {
            owned_context_count(&state) != Some(0) || !state.refresh_tokens.is_empty()
        });
        let result = if unresolved {
            Err(McpError::CleanupPending)
        } else if failed {
            Err(McpError::Transport(McpTransportError {
                operation: McpOperation::Close,
                dispatch: McpDispatchState::PossiblyDispatched,
            }))
        } else {
            Ok(())
        };
        drop(_close);
        self.observer.emit(event(
            McpLifecycleOperation::Close,
            result
                .as_ref()
                .map(|_| McpLifecycleOutcome::Succeeded)
                .unwrap_or(McpLifecycleOutcome::Failed),
            elapsed_ms(started),
            0,
            0,
            0,
            false,
            cancellation.is_cancelled(),
            McpDispatchState::PossiblyDispatched,
            None,
        ));
        result
    }

    /// Returns observer delivery health without exposing observer errors.
    pub fn observer_health(&self) -> McpObserverHealth {
        self.observer.health()
    }

    /// Returns retained shutdown work without transport details.
    pub fn shutdown_health(&self) -> McpShutdownHealth {
        self.state.lock().map_or_else(
            |_| McpShutdownHealth {
                pending_contexts: u64::MAX,
                in_progress_contexts: u64::MAX,
                close_failures: u64::MAX,
            },
            |state| McpShutdownHealth {
                pending_contexts: u64::try_from(owned_context_count(&state).unwrap_or(usize::MAX))
                    .unwrap_or(u64::MAX),
                in_progress_contexts: state.draining_contexts,
                close_failures: state.close_failures,
            },
        )
    }

    /// Attempts a context already retained in manager ownership. Keeping the
    /// original entry in state until transport close succeeds makes a dropped
    /// caller future unable to lose a connected context.
    async fn try_close_retained(&self, context: &RetiredContext) -> bool {
        let closed = close_context(
            self.transport.as_ref(),
            (*context.context).clone(),
            CancellationToken::new(),
            self.timeouts.close,
        )
        .await
        .is_ok();
        if let Ok(mut state) = self.state.lock() {
            if closed {
                if let Some(index) = state.retired_contexts.iter().position(|entry| {
                    Arc::ptr_eq(&entry.context, &context.context)
                        && Arc::ptr_eq(&entry.in_flight, &context.in_flight)
                }) {
                    state.retired_contexts.remove(index);
                }
            } else {
                state.close_failures = state.close_failures.saturating_add(1);
            }
        }
        closed
    }

    async fn fail_refresh_context<T>(
        &self,
        token: u64,
        error: McpError,
        started: Instant,
        bytes: u64,
        pages: u64,
        lifecycle_events: &mut Vec<McpLifecycleEvent>,
    ) -> Result<T, McpError> {
        self.cleanup_pending_context(token).await;
        lifecycle_events.push(event(
            McpLifecycleOperation::Refresh,
            outcome_for_error(&error),
            elapsed_ms(started),
            0,
            bytes,
            pages,
            false,
            false,
            McpDispatchState::Responded,
            None,
        ));
        Err(error)
    }

    fn begin_refresh(&self) -> Result<RefreshOwnership, McpError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| McpError::CatalogUnavailable)?;
        if state.closed
            || owned_context_count(&state)
                .is_none_or(|count| count >= self.limits.max_retired_generations)
        {
            return Err(McpError::CatalogUnavailable);
        }
        let token = state.next_refresh_token;
        state.next_refresh_token = state
            .next_refresh_token
            .checked_add(1)
            .ok_or(McpError::Clock)?;
        if !state.refresh_tokens.insert(token) {
            return Err(McpError::Clock);
        }
        Ok(RefreshOwnership {
            state: Arc::clone(&self.state),
            token,
        })
    }

    fn retain_pending_context(&self, token: u64, context: McpContext) {
        if let Ok(mut state) = self.state.lock() {
            if state.refresh_tokens.contains(&token) {
                state
                    .pending_refresh_contexts
                    .push(PendingRefreshContext { token, context });
            }
        }
    }

    fn take_pending_context(&self, token: u64) -> Option<McpContext> {
        self.state.lock().ok().and_then(|mut state| {
            state
                .pending_refresh_contexts
                .iter()
                .rposition(|candidate| candidate.token == token)
                .map(|index| state.pending_refresh_contexts.remove(index).context)
        })
    }

    async fn cleanup_pending_context(&self, token: u64) {
        let Some(context) = self.take_pending_context(token) else {
            return;
        };
        let retained = RetiredContext {
            context: Arc::new(context),
            in_flight: Arc::new(InFlightCalls::default()),
        };
        if let Ok(mut state) = self.state.lock() {
            state.retired_contexts.push(RetiredContext {
                context: Arc::clone(&retained.context),
                in_flight: Arc::clone(&retained.in_flight),
            });
        }
        self.try_close_retained(&retained).await;
    }

    fn schedule_close_after_drain(&self, context: Arc<McpContext>, calls: Arc<InFlightCalls>) {
        let transport = Arc::clone(&self.transport);
        let timeout = self.timeouts.close;
        let observer = Arc::clone(&self.observer);
        let state = Arc::clone(&self.state);
        if let Ok(mut state) = state.lock() {
            if let Some(index) = state.retired_contexts.iter().position(|entry| {
                Arc::ptr_eq(&entry.context, &context) && Arc::ptr_eq(&entry.in_flight, &calls)
            }) {
                state.retired_contexts.remove(index);
            }
            state.draining_contexts = state.draining_contexts.saturating_add(1);
        }
        tokio::spawn(async move {
            wait_for_drain(&calls).await;
            let started = Instant::now();
            let result = close_context(
                transport.as_ref(),
                (*context).clone(),
                CancellationToken::new(),
                timeout,
            )
            .await;
            if let Ok(mut state) = state.lock() {
                state.draining_contexts = state.draining_contexts.saturating_sub(1);
                if result.is_err() {
                    state.retired_contexts.push(RetiredContext {
                        context,
                        in_flight: calls,
                    });
                    state.close_failures = state.close_failures.saturating_add(1);
                }
            }
            observer.emit(event(
                McpLifecycleOperation::Close,
                if result.is_ok() {
                    McpLifecycleOutcome::Succeeded
                } else {
                    McpLifecycleOutcome::Failed
                },
                elapsed_ms(started),
                0,
                0,
                0,
                false,
                false,
                McpDispatchState::PossiblyDispatched,
                None,
            ));
        });
    }

    fn checked_now(&self) -> Result<u64, McpError> {
        let now = self.clock.now_ms();
        let mut state = self
            .state
            .lock()
            .map_err(|_| McpError::CatalogUnavailable)?;
        if now < state.last_now_ms {
            return Err(McpError::Clock);
        }
        state.last_now_ms = now;
        Ok(now)
    }

    async fn fetch_complete(
        &self,
        token: u64,
        cancellation: CancellationToken,
        lifecycle_events: &mut Vec<McpLifecycleEvent>,
    ) -> Result<(McpContext, FetchedTools, Option<McpCacheScope>, u64, u64), McpError> {
        let negotiation_started = Instant::now();
        let context = tokio::select! {
            _ = cancellation.cancelled() => return Err(McpError::Transport(McpTransportError { operation: McpOperation::Connect, dispatch: McpDispatchState::NotDispatched })),
            result = tokio::time::timeout(self.timeouts.connect, self.transport.connect(cancellation.child_token())) => result.map_err(|_| McpTransportError { operation: McpOperation::Connect, dispatch: McpDispatchState::PossiblyDispatched })??,
        };
        // Record ownership before the first post-connect await. If the
        // refresh future is abandoned during validation/listing, close() can
        // still reclaim this handle.
        self.retain_pending_context(token, context.clone());
        if let Err(error) = validate_context(&context, &self.limits) {
            self.cleanup_pending_context(token).await;
            return Err(error);
        }
        if !supported_context(&context) {
            // Once connected, this function owns the context until it is
            // transferred in its successful return value.
            self.cleanup_pending_context(token).await;
            return Err(McpError::InvalidCatalog(
                "unsupported version or missing tools capability".into(),
            ));
        }
        lifecycle_events.push(event(
            McpLifecycleOperation::Negotiation,
            McpLifecycleOutcome::Succeeded,
            elapsed_ms(negotiation_started),
            0,
            0,
            0,
            false,
            false,
            McpDispatchState::Responded,
            None,
        ));
        let fetched = fetch_tools(
            &*self.transport,
            &context,
            cancellation,
            &self.limits,
            self.timeouts.list,
        )
        .await;
        let fetched = match fetched {
            Ok(fetched) => fetched,
            Err(error) => {
                self.cleanup_pending_context(token).await;
                return Err(error);
            }
        };
        let scope = fetched.cache_scope;
        let bytes = fetched.bytes;
        let pages = fetched.pages;
        Ok((context, fetched, scope, bytes, pages))
    }
}

struct FetchedTools {
    items: Vec<McpTool>,
    cache_ttl: Option<u64>,
    cache_scope: Option<McpCacheScope>,
    bytes: u64,
    pages: u64,
}

struct ManagedMcpToolAdapter {
    inner: Arc<McpToolAdapter>,
    generation: u64,
    state: Weak<Mutex<CatalogState>>,
    clock: Arc<dyn McpClock>,
    in_flight: Arc<InFlightCalls>,
    dispatch_gate: Arc<tokio::sync::RwLock<()>>,
}

impl ManagedMcpToolAdapter {
    fn new(
        inner: Arc<McpToolAdapter>,
        generation: u64,
        state: Weak<Mutex<CatalogState>>,
        clock: Arc<dyn McpClock>,
        in_flight: Arc<InFlightCalls>,
        dispatch_gate: Arc<tokio::sync::RwLock<()>>,
    ) -> Self {
        Self {
            inner,
            generation,
            state,
            clock,
            in_flight,
            dispatch_gate,
        }
    }

    fn check_active(&self) -> Result<(bool, InFlightLease), HarnessError> {
        let now = self.clock.now_ms();
        let state = self
            .state
            .upgrade()
            .ok_or_else(|| HarnessError::InvalidTool("MCP catalog unavailable".into()))?;
        let mut state = state
            .lock()
            .map_err(|_| HarnessError::InvalidTool("MCP catalog unavailable".into()))?;
        if now < state.last_now_ms {
            return Err(HarnessError::InvalidTool(
                "MCP catalog clock unavailable".into(),
            ));
        }
        state.last_now_ms = now;
        let snapshot = state
            .active
            .as_ref()
            .ok_or_else(|| HarnessError::InvalidTool("MCP catalog unavailable".into()))?;
        if state.closed || state.invalidated || snapshot.summary.generation != self.generation {
            return Err(HarnessError::InvalidTool(
                "MCP catalog is no longer active".into(),
            ));
        }
        if now < snapshot.expires_at_ms {
            self.in_flight.count.fetch_add(1, Ordering::AcqRel);
            return Ok((false, InFlightLease(Arc::clone(&self.in_flight))));
        }
        if now < snapshot.stale_until_ms {
            self.in_flight.count.fetch_add(1, Ordering::AcqRel);
            return Ok((true, InFlightLease(Arc::clone(&self.in_flight))));
        }
        Err(HarnessError::InvalidTool("MCP catalog expired".into()))
    }
}

#[async_trait]
impl Tool for ManagedMcpToolAdapter {
    fn definition(&self) -> &ToolDefinition {
        self.inner.definition()
    }
    async fn execute(
        &self,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        self.execute_with_context(
            &ToolCallContext::new("", "", "", &self.definition().id),
            arguments,
            cancellation,
        )
        .await
    }
    async fn execute_with_context(
        &self,
        call: &ToolCallContext,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        let server_permit = match self.inner.acquire_server(&cancellation).await {
            Ok(permit) => permit,
            Err(error) => {
                let (outcome, cancelled) = match error {
                    HarnessError::Cancelled => (McpLifecycleOutcome::Cancelled, true),
                    _ => (McpLifecycleOutcome::Rejected, false),
                };
                self.inner.observer.emit(event(
                    McpLifecycleOperation::Call,
                    outcome,
                    0,
                    0,
                    0,
                    0,
                    false,
                    cancelled,
                    McpDispatchState::NotDispatched,
                    Some(McpCallCorrelation::from(call)),
                ));
                return Err(error);
            }
        };
        // This permit stays alive through the transport await. Retirement
        // obtains the write side before changing generation state, so a call
        // is either admitted before retirement or rejected before dispatch.
        let dispatch_permit = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                drop(server_permit);
                self.inner.observer.emit(event(
                    McpLifecycleOperation::Call,
                    McpLifecycleOutcome::Cancelled,
                    0, 0, 0, 0, false, true,
                    McpDispatchState::NotDispatched,
                    Some(McpCallCorrelation::from(call)),
                ));
                return Err(HarnessError::Cancelled);
            },
            permit = self.dispatch_gate.read() => permit,
        };
        let (stale, lease) = self.check_active()?;
        if cancellation.is_cancelled() {
            drop(lease);
            drop(dispatch_permit);
            drop(server_permit);
            return Err(HarnessError::Cancelled);
        }
        let (result, lifecycle) = self
            .inner
            .execute_managed(call, arguments, cancellation, stale, Some(server_permit))
            .await;
        drop(lease);
        drop(dispatch_permit);
        self.inner.observer.emit(lifecycle);
        result
    }
}

fn supported_context(context: &McpContext) -> bool {
    matches!(
        (context.era, context.version.as_str()),
        (McpProtocolEra::Modern20260728, "2026-07-28")
            | (McpProtocolEra::Legacy20251125, "2025-11-25")
    ) && context.capabilities.contains("tools")
}

async fn fetch_tools(
    transport: &dyn McpTransport,
    context: &McpContext,
    cancellation: CancellationToken,
    limits: &McpLimits,
    timeout: Duration,
) -> Result<FetchedTools, McpError> {
    let mut cursor = None;
    let mut seen_cursors = HashSet::new();
    let mut seen_names = HashSet::new();
    let mut items = Vec::new();
    let mut bytes = 0usize;
    let mut pages = 0usize;
    let mut cache_ttl: Option<u64> = None;
    let mut cache_scope = None;
    for _ in 0..limits.max_pages {
        let page = tokio::select! {
            _ = cancellation.cancelled() => return Err(McpError::Transport(McpTransportError { operation: McpOperation::ListTools, dispatch: McpDispatchState::NotDispatched })),
            result = tokio::time::timeout(timeout, transport.list_tools(context, cursor.as_deref(), cancellation.child_token())) => result.map_err(|_| McpTransportError { operation: McpOperation::ListTools, dispatch: McpDispatchState::PossiblyDispatched })??,
        };
        pages = pages.checked_add(1).ok_or(McpError::Clock)?;
        let remaining_tools = limits
            .max_tools
            .checked_sub(items.len())
            .ok_or_else(|| McpError::InvalidCatalog("tool limit exceeded".into()))?;
        if page.tools.len() > remaining_tools {
            return Err(McpError::InvalidCatalog("tool limit exceeded".into()));
        }
        let page_bytes = admit_page(&page, limits)?;
        bytes = bytes
            .checked_add(page_bytes)
            .ok_or(McpError::InvalidCatalog(
                "catalog exceeds byte limit".into(),
            ))?;
        if bytes > limits.max_catalog_bytes {
            return Err(McpError::InvalidCatalog(
                "catalog exceeds byte limit".into(),
            ));
        }
        if context.era == McpProtocolEra::Modern20260728 {
            let ttl = page
                .ttl_ms
                .ok_or_else(|| McpError::InvalidCatalog("modern page omitted ttl".into()))?;
            let scope = McpCacheScope::parse(page.cache_scope.as_deref().ok_or_else(|| {
                McpError::InvalidCatalog("modern page omitted cache scope".into())
            })?)?;
            cache_ttl = Some(cache_ttl.map_or(ttl, |previous| previous.min(ttl)));
            if cache_scope
                .replace(scope)
                .is_some_and(|previous| previous != scope)
            {
                return Err(McpError::InvalidCatalog(
                    "inconsistent modern cache metadata".into(),
                ));
            }
        } else if page.ttl_ms.is_some() || page.cache_scope.is_some() {
            return Err(McpError::InvalidCatalog(
                "legacy page supplied modern cache metadata".into(),
            ));
        }
        for tool in page.tools {
            if !seen_names.insert(tool.name.clone()) {
                return Err(McpError::InvalidCatalog(
                    "duplicate native tool name".into(),
                ));
            }
            validate_tool(&tool, limits)?;
            items.push(tool);
        }
        cursor = page.next_cursor;
        if let Some(next) = &cursor {
            if !seen_cursors.insert(next.clone()) {
                return Err(McpError::InvalidCatalog("cursor cycle".into()));
            }
        } else {
            break;
        }
    }
    if cursor.is_some() {
        return Err(McpError::InvalidCatalog("page limit exceeded".into()));
    }
    Ok(FetchedTools {
        items,
        cache_ttl,
        cache_scope,
        bytes: u64::try_from(bytes).unwrap_or(u64::MAX),
        pages: u64::try_from(pages).unwrap_or(u64::MAX),
    })
}

fn validate_duration(duration: Duration) -> Result<(), McpError> {
    duration_ms(duration).map(|_| ())
}

fn validate_server_waiter_limit(max_waiters: usize) -> Result<(), McpError> {
    if max_waiters == 0 || max_waiters == usize::MAX {
        Err(McpError::InvalidConfiguration(
            "max_server_waiters must be finite and nonzero".into(),
        ))
    } else {
        Ok(())
    }
}

fn duration_ms(duration: Duration) -> Result<u64, McpError> {
    u64::try_from(duration.as_millis()).map_err(|_| McpError::Clock)
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn outcome_for_error(error: &McpError) -> McpLifecycleOutcome {
    match error {
        McpError::Transport(McpTransportError { .. }) => McpLifecycleOutcome::Failed,
        McpError::CatalogUnavailable
        | McpError::CleanupPending
        | McpError::Clock
        | McpError::InvalidConfiguration(_)
        | McpError::InvalidCatalog(_)
        | McpError::ResourceLimit => McpLifecycleOutcome::Rejected,
        McpError::Core(_) => McpLifecycleOutcome::Failed,
    }
}

fn dispatch_for_error(error: &McpError) -> McpDispatchState {
    match error {
        McpError::Transport(error) => error.dispatch,
        McpError::CatalogUnavailable
        | McpError::CleanupPending
        | McpError::Clock
        | McpError::InvalidConfiguration(_)
        | McpError::InvalidCatalog(_)
        | McpError::ResourceLimit
        | McpError::Core(_) => McpDispatchState::NotDispatched,
    }
}

#[allow(clippy::too_many_arguments)]
fn event(
    operation: McpLifecycleOperation,
    outcome: McpLifecycleOutcome,
    duration_ms: u64,
    count: u64,
    bytes: u64,
    pages: u64,
    stale: bool,
    cancelled: bool,
    dispatch: McpDispatchState,
    correlation: Option<McpCallCorrelation>,
) -> McpLifecycleEvent {
    McpLifecycleEvent {
        operation,
        outcome,
        duration_ms,
        count,
        bytes,
        pages,
        stale,
        cancelled,
        dispatch,
        correlation,
    }
}

struct McpToolAdapter {
    definition: ToolDefinition,
    transport: Arc<dyn McpTransport>,
    context: Arc<McpContext>,
    native_name: String,
    call_timeout: Duration,
    limits: McpLimits,
    observer: Arc<ObserverHub>,
    server_call_gate: Arc<ServerCallGate>,
}
impl McpToolAdapter {
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: String,
        server_id: String,
        tool: McpTool,
        context: Arc<McpContext>,
        transport: Arc<dyn McpTransport>,
        call_timeout: Duration,
        limits: McpLimits,
        observer: Arc<ObserverHub>,
        server_call_gate: Arc<ServerCallGate>,
    ) -> Self {
        let native_name = tool.name.clone();
        let mut d = ToolDefinition::new(id, tool.name, tool.description, tool.input_schema)
            .with_risk(ToolRisk::High)
            .with_idempotent(false)
            .with_read_only(false)
            .with_parallel_safe(false)
            .with_concurrency_key(format!("mcp:{server_id}"))
            .with_cancellation_safety(CancellationSafety::Unknown)
            .with_allowed_callers([ToolCaller::Direct])
            .with_speculation_policy(SpeculationPolicy::Disabled)
            .with_execution_location(ExecutionLocation::Remote)
            .with_network_egress(NetworkEgress::Permitted);
        if let Some(schema) = tool.output_schema {
            d = d.with_output_schema(schema);
        }
        Self {
            definition: d,
            transport,
            context,
            native_name,
            call_timeout,
            limits,
            observer,
            server_call_gate,
        }
    }
}
#[async_trait]
impl Tool for McpToolAdapter {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn execute(
        &self,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        self.execute_with_context(
            &ToolCallContext::new("", "", "", &self.definition.id),
            arguments,
            cancellation,
        )
        .await
    }
    async fn execute_with_context(
        &self,
        call: &ToolCallContext,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        let (result, lifecycle) = self
            .execute_managed(call, arguments, cancellation, false, None)
            .await;
        self.observer.emit(lifecycle);
        result
    }
}

impl McpToolAdapter {
    async fn acquire_server(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<tokio::sync::SemaphorePermit<'_>, HarnessError> {
        let queued = self.server_call_gate.waiters.fetch_add(1, Ordering::AcqRel);
        if queued >= self.server_call_gate.max_waiters {
            self.server_call_gate.waiters.fetch_sub(1, Ordering::AcqRel);
            return Err(HarnessError::ResourceLimit(
                "MCP server queue is full".into(),
            ));
        }
        let waiter = ServerWaiter(&self.server_call_gate.waiters);
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                drop(waiter);
                Err(HarnessError::Cancelled)
            }
            permit = self.server_call_gate.permits.acquire() => {
                drop(waiter);
                permit.map_err(|_| HarnessError::InvalidTool("MCP server gate unavailable".into()))
            }
        }
    }

    async fn execute_managed(
        &self,
        call: &ToolCallContext,
        arguments: Value,
        cancellation: CancellationToken,
        stale: bool,
        preacquired: Option<tokio::sync::SemaphorePermit<'_>>,
    ) -> (Result<ToolResult, HarnessError>, McpLifecycleEvent) {
        if call.caller == Some(ToolCaller::Speculative) {
            return (
                Err(HarnessError::InvalidTool(
                    "MCP tools cannot be speculative".into(),
                )),
                event(
                    McpLifecycleOperation::Call,
                    McpLifecycleOutcome::Rejected,
                    0,
                    0,
                    0,
                    0,
                    stale,
                    false,
                    McpDispatchState::NotDispatched,
                    Some(McpCallCorrelation::from(call)),
                ),
            );
        }
        if cancellation.is_cancelled() {
            return (
                Err(HarnessError::Cancelled),
                event(
                    McpLifecycleOperation::Call,
                    McpLifecycleOutcome::Cancelled,
                    0,
                    0,
                    0,
                    0,
                    stale,
                    true,
                    McpDispatchState::NotDispatched,
                    Some(McpCallCorrelation::from(call)),
                ),
            );
        }
        let started = Instant::now();
        let correlation = Some(McpCallCorrelation::from(call));
        let _server_permit = match preacquired {
            Some(permit) => permit,
            None => match self.acquire_server(&cancellation).await {
                Ok(permit) => permit,
                Err(error) => {
                    let (outcome, cancelled) = match error {
                        HarnessError::Cancelled => (McpLifecycleOutcome::Cancelled, true),
                        _ => (McpLifecycleOutcome::Rejected, false),
                    };
                    return (
                        Err(error),
                        event(
                            McpLifecycleOperation::Call,
                            outcome,
                            elapsed_ms(started),
                            0,
                            0,
                            0,
                            stale,
                            cancelled,
                            McpDispatchState::NotDispatched,
                            correlation,
                        ),
                    );
                }
            },
        };
        enum CallFailure {
            Cancelled,
            TimedOut,
            Transport(McpTransportError),
        }
        let result = tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err(CallFailure::Cancelled),
            result = tokio::time::timeout(self.call_timeout, self.transport.call_tool(&self.context, McpCallRequest { name: self.native_name.clone(), arguments, context: call.clone() }, cancellation.child_token())) => match result {
                Ok(Ok(result)) => Ok(result),
                Ok(Err(error)) => Err(CallFailure::Transport(error)),
                Err(_) => Err(CallFailure::TimedOut),
            },
        };
        let elapsed = elapsed_ms(started);
        let (returned, lifecycle) = match result {
            Ok(result) => {
                let normalized = normalize_result(
                    result,
                    &self.limits,
                    self.definition.output_schema.is_some(),
                );
                let (outcome, count, bytes) = match &normalized {
                    Ok((_, bytes, count)) => (McpLifecycleOutcome::Succeeded, *count, *bytes),
                    Err(_) => (McpLifecycleOutcome::Rejected, 0, 0),
                };
                let lifecycle = event(
                    McpLifecycleOperation::Call,
                    outcome,
                    elapsed,
                    count,
                    bytes,
                    0,
                    stale,
                    false,
                    McpDispatchState::Responded,
                    correlation,
                );
                (normalized.map(|(result, _, _)| result), lifecycle)
            }
            Err(error) => {
                let (outcome, cancelled, dispatch, returned) = match error {
                    CallFailure::Cancelled => (
                        McpLifecycleOutcome::Cancelled,
                        true,
                        McpDispatchState::PossiblyDispatched,
                        HarnessError::Cancelled,
                    ),
                    CallFailure::TimedOut => (
                        McpLifecycleOutcome::TimedOut,
                        false,
                        McpDispatchState::PossiblyDispatched,
                        HarnessError::TimedOut("MCP tool call".into()),
                    ),
                    CallFailure::Transport(error) => (
                        McpLifecycleOutcome::Failed,
                        false,
                        error.dispatch,
                        HarnessError::Tool("MCP transport failure".into()),
                    ),
                };
                let lifecycle = event(
                    McpLifecycleOperation::Call,
                    outcome,
                    elapsed,
                    0,
                    0,
                    0,
                    stale,
                    cancelled,
                    dispatch,
                    correlation,
                );
                (Err(returned), lifecycle)
            }
        };
        drop(_server_permit);
        (returned, lifecycle)
    }
}

fn normalize_result(
    result: McpCallResult,
    limits: &McpLimits,
    has_output_schema: bool,
) -> Result<(ToolResult, u64, u64), HarnessError> {
    let (bytes, blocks) = admit_call_result(&result, limits)?;
    if !result.is_error && has_output_schema && result.structured_content.is_none() {
        return Err(HarnessError::Tool(
            "MCP structured content is required".into(),
        ));
    }
    let output = result
        .structured_content
        .or(result.content)
        .unwrap_or(Value::Null);
    if result.is_error {
        Ok((
            ToolResult::new(false, output, Some("MCP tool reported failure".into())),
            bytes,
            blocks,
        ))
    } else {
        Ok((ToolResult::success(output), bytes, blocks))
    }
}

fn admit_call_result(
    result: &McpCallResult,
    limits: &McpLimits,
) -> Result<(u64, u64), HarnessError> {
    let blocks = result
        .content
        .as_ref()
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    if blocks > limits.max_content_blocks {
        return Err(HarnessError::ResourceLimit(
            "MCP response exceeds content block limit".into(),
        ));
    }
    let mut bytes = 0usize;
    for value in [result.structured_content.as_ref(), result.content.as_ref()]
        .into_iter()
        .flatten()
    {
        bytes = bytes
            .checked_add(
                admit_json_value(value, limits, limits.max_response_bytes, false).map_err(
                    |_| {
                        HarnessError::ResourceLimit(
                            "MCP response exceeds configured structural limit".into(),
                        )
                    },
                )?,
            )
            .ok_or_else(|| HarnessError::ResourceLimit("MCP response exceeds byte limit".into()))?;
        if bytes > limits.max_response_bytes {
            return Err(HarnessError::ResourceLimit(
                "MCP response exceeds byte limit".into(),
            ));
        }
    }
    Ok((
        u64::try_from(bytes).unwrap_or(u64::MAX),
        u64::try_from(blocks).unwrap_or(u64::MAX),
    ))
}
fn generated_id(server: &str, native: &str) -> String {
    let slug: String = native
        .chars()
        .filter_map(|c| {
            if c.is_ascii_alphanumeric() {
                Some(c.to_ascii_lowercase())
            } else if c == '-' || c == '_' {
                Some('-')
            } else {
                None
            }
        })
        .take(48)
        .collect();
    format!(
        "mcp-{server}-{}-{}",
        if slug.is_empty() { "tool" } else { &slug },
        &blake3::hash(native.as_bytes()).to_hex()[..32]
    )
}
fn validate_server_id(id: String) -> Result<String, McpError> {
    if id.is_empty()
        || id.len() > 64
        || !id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        Err(McpError::InvalidCatalog(
            "invalid canonical server id".into(),
        ))
    } else {
        Ok(id)
    }
}
fn validate_tool(tool: &McpTool, limits: &McpLimits) -> Result<(), McpError> {
    if tool.name.is_empty()
        || tool.name.len() > 256
        || !tool.name.chars().all(|c| !c.is_control())
        || !tool.input_schema.is_object()
    {
        return Err(McpError::InvalidCatalog(
            "malformed tool identifier or input schema".into(),
        ));
    }
    for schema in [&tool.input_schema]
        .into_iter()
        .chain(tool.output_schema.iter())
    {
        if admit_json_value(schema, limits, limits.max_catalog_bytes, true).is_err() {
            return Err(McpError::InvalidCatalog(
                "unsafe or oversized schema".into(),
            ));
        }
    }
    Ok(())
}

fn admit_page(page: &McpToolPage, limits: &McpLimits) -> Result<usize, McpError> {
    // Serde serializes every public struct field. Count that framing here so
    // catalog admission remains conservative without serializing the page.
    let mut bytes = 2usize
        .checked_add(3)
        .ok_or_else(|| McpError::InvalidCatalog("catalog exceeds byte limit".into()))?; // braces plus four field separators
    let mut add = |value: usize| -> Result<(), McpError> {
        bytes = bytes
            .checked_add(value)
            .ok_or_else(|| McpError::InvalidCatalog("catalog exceeds byte limit".into()))?;
        if bytes > limits.max_catalog_bytes {
            return Err(McpError::InvalidCatalog(
                "catalog exceeds byte limit".into(),
            ));
        }
        Ok(())
    };
    add(json_field_prefix("tools")?)?;
    add(2usize
        .checked_add(page.tools.len().saturating_sub(1))
        .ok_or_else(|| McpError::InvalidCatalog("catalog exceeds byte limit".into()))?)?;
    for tool in &page.tools {
        if tool.name.len() > limits.max_string_bytes
            || tool.description.len() > limits.max_string_bytes
        {
            return Err(McpError::InvalidCatalog(
                "catalog string exceeds limit".into(),
            ));
        }
        add(2usize
            .checked_add(3)
            .ok_or_else(|| McpError::InvalidCatalog("catalog exceeds byte limit".into()))?)?;
        add(json_field_prefix("name")?)?;
        add(json_string_len(&tool.name)
            .map_err(|_| McpError::InvalidCatalog("catalog exceeds byte limit".into()))?)?;
        add(json_field_prefix("description")?)?;
        add(json_string_len(&tool.description)
            .map_err(|_| McpError::InvalidCatalog("catalog exceeds byte limit".into()))?)?;
        add(json_field_prefix("input_schema")?)?;
        add(
            admit_json_value(&tool.input_schema, limits, limits.max_catalog_bytes, true)
                .map_err(|_| McpError::InvalidCatalog("unsafe or oversized schema".into()))?,
        )?;
        add(json_field_prefix("output_schema")?)?;
        if let Some(schema) = &tool.output_schema {
            add(
                admit_json_value(schema, limits, limits.max_catalog_bytes, true)
                    .map_err(|_| McpError::InvalidCatalog("unsafe or oversized schema".into()))?,
            )?;
        } else {
            add(4)?;
        }
    }
    add(json_field_prefix("next_cursor")?)?;
    add(optional_json_string(page.next_cursor.as_deref(), limits)?)?;
    add(json_field_prefix("ttl_ms")?)?;
    add(page.ttl_ms.map_or(4, |value| value.to_string().len()))?;
    add(json_field_prefix("cache_scope")?)?;
    add(optional_json_string(page.cache_scope.as_deref(), limits)?)?;
    Ok(bytes)
}

fn json_field_prefix(name: &str) -> Result<usize, McpError> {
    json_string_len(name)
        .and_then(|length| length.checked_add(1).ok_or(()))
        .map_err(|_| McpError::InvalidCatalog("catalog exceeds byte limit".into()))
}

fn optional_json_string(value: Option<&str>, limits: &McpLimits) -> Result<usize, McpError> {
    match value {
        Some(value) if value.len() <= limits.max_string_bytes => json_string_len(value)
            .map_err(|_| McpError::InvalidCatalog("catalog exceeds byte limit".into())),
        Some(_) => Err(McpError::InvalidCatalog(
            "catalog string exceeds limit".into(),
        )),
        None => Ok(4),
    }
}

fn validate_context(context: &McpContext, limits: &McpLimits) -> Result<(), McpError> {
    if context.capabilities.len() > limits.max_context_capabilities
        || context.version.len() > limits.max_string_bytes
        || context
            .request_context
            .as_ref()
            .is_some_and(|value| value.len() > limits.max_string_bytes)
    {
        return Err(McpError::InvalidCatalog(
            "negotiated context exceeds limit".into(),
        ));
    }
    let mut bytes = json_string_len(&context.version)
        .map_err(|_| McpError::InvalidCatalog("negotiated context exceeds limit".into()))?;
    let mut nodes = 1usize;
    for capability in &context.capabilities {
        if capability.len() > limits.max_string_bytes {
            return Err(McpError::InvalidCatalog(
                "negotiated context exceeds limit".into(),
            ));
        }
        bytes = bytes
            .checked_add(
                json_string_len(capability).map_err(|_| {
                    McpError::InvalidCatalog("negotiated context exceeds limit".into())
                })?,
            )
            .ok_or_else(|| McpError::InvalidCatalog("negotiated context exceeds limit".into()))?;
        nodes = nodes
            .checked_add(1)
            .ok_or_else(|| McpError::InvalidCatalog("negotiated context exceeds limit".into()))?;
    }
    if let Some(request_context) = &context.request_context {
        bytes = bytes
            .checked_add(
                json_string_len(request_context).map_err(|_| {
                    McpError::InvalidCatalog("negotiated context exceeds limit".into())
                })?,
            )
            .ok_or_else(|| McpError::InvalidCatalog("negotiated context exceeds limit".into()))?;
        nodes = nodes
            .checked_add(1)
            .ok_or_else(|| McpError::InvalidCatalog("negotiated context exceeds limit".into()))?;
    }
    if bytes > limits.max_catalog_bytes
        || nodes > limits.max_json_nodes
        || limits.max_json_depth < 2
    {
        return Err(McpError::InvalidCatalog(
            "negotiated context exceeds limit".into(),
        ));
    }
    Ok(())
}

/// Estimates JSON's encoded size while enforcing structural limits without
/// serialization, recursion, or cloning of untrusted server values.
fn admit_json_value(
    root: &Value,
    limits: &McpLimits,
    max_bytes: usize,
    reject_external_refs: bool,
) -> Result<usize, ()> {
    let mut bytes = 0usize;
    let mut nodes = 0usize;
    let mut properties = 0usize;
    let mut stack = vec![(root, 1usize)];
    while let Some((value, depth)) = stack.pop() {
        if depth > limits.max_json_depth {
            return Err(());
        }
        nodes = nodes.checked_add(1).ok_or(())?;
        if nodes > limits.max_json_nodes {
            return Err(());
        }
        let mut add = |count: usize| -> Result<(), ()> {
            bytes = bytes.checked_add(count).ok_or(())?;
            (bytes <= max_bytes).then_some(()).ok_or(())
        };
        match value {
            Value::Null => add(4)?,
            Value::Bool(true) => add(4)?,
            Value::Bool(false) => add(5)?,
            Value::Number(number) => add(number.to_string().len())?,
            Value::String(string) => {
                if string.len() > limits.max_string_bytes {
                    return Err(());
                }
                add(json_string_len(string)?)?;
            }
            Value::Array(values) => {
                add(2usize
                    .checked_add(values.len().saturating_sub(1))
                    .ok_or(())?)?;
                ensure_child_capacity(values.len(), nodes, stack.len(), limits.max_json_nodes)?;
                for child in values.iter().rev() {
                    stack.push((child, depth.checked_add(1).ok_or(())?));
                }
            }
            Value::Object(values) => {
                properties = properties.checked_add(values.len()).ok_or(())?;
                if properties > limits.max_schema_properties {
                    return Err(());
                }
                add(2usize
                    .checked_add(values.len().saturating_sub(1))
                    .ok_or(())?)?;
                ensure_child_capacity(values.len(), nodes, stack.len(), limits.max_json_nodes)?;
                for (key, child) in values.iter().rev() {
                    if key.len() > limits.max_string_bytes {
                        return Err(());
                    }
                    add(json_string_len(key)?.checked_add(1).ok_or(())?)?;
                    if reject_external_refs
                        && matches!(key.as_str(), "$ref" | "$dynamicRef" | "$recursiveRef")
                        && child
                            .as_str()
                            .is_some_and(|reference| !reference.starts_with('#'))
                    {
                        return Err(());
                    }
                    stack.push((child, depth.checked_add(1).ok_or(())?));
                }
            }
        }
    }
    Ok(bytes)
}

fn ensure_child_capacity(
    child_count: usize,
    visited_nodes: usize,
    queued_nodes: usize,
    max_nodes: usize,
) -> Result<(), ()> {
    let remaining = max_nodes.checked_sub(visited_nodes).ok_or(())?;
    if queued_nodes.checked_add(child_count).ok_or(())? > remaining {
        return Err(());
    }
    Ok(())
}

fn json_string_len(value: &str) -> Result<usize, ()> {
    let mut bytes = 2usize;
    for byte in value.bytes() {
        bytes = bytes
            .checked_add(match byte {
                b'"' | b'\\' | 0x08 | 0x0c | b'\n' | b'\r' | b'\t' => 2,
                0x00..=0x1f => 6,
                _ => 1,
            })
            .ok_or(())?;
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64};

    #[derive(Default)]
    struct FakeClock(AtomicU64);

    impl FakeClock {
        fn set(&self, now: u64) {
            self.0.store(now, Ordering::Relaxed);
        }
    }

    impl McpClock for FakeClock {
        fn now_ms(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    struct FakeTransport {
        context: McpContext,
        pages: Mutex<Vec<McpToolPage>>,
        calls: AtomicU64,
        closes: AtomicU64,
        yield_close: AtomicBool,
        hang_close: AtomicBool,
        fail_close: AtomicBool,
        connects: AtomicU64,
        block_connect: AtomicBool,
        connect_started: AtomicBool,
        connect_release: tokio::sync::Notify,
        block_list: AtomicBool,
        list_started: AtomicBool,
        list_release: tokio::sync::Notify,
        hang_call: AtomicBool,
        block_call: AtomicBool,
        call_started: AtomicU64,
        call_release: tokio::sync::Notify,
        active_calls: AtomicU64,
        max_active_calls: AtomicU64,
        fail_list: AtomicBool,
        call_failure: Mutex<Option<McpTransportError>>,
        result: Mutex<McpCallResult>,
    }

    impl FakeTransport {
        fn modern(ttl_ms: u64, tools: Vec<McpTool>) -> Self {
            Self {
                context: McpContext {
                    era: McpProtocolEra::Modern20260728,
                    version: "2026-07-28".into(),
                    capabilities: BTreeSet::from(["tools".into()]),
                    request_context: None,
                },
                pages: Mutex::new(vec![McpToolPage {
                    tools,
                    next_cursor: None,
                    ttl_ms: Some(ttl_ms),
                    cache_scope: Some("public".into()),
                }]),
                calls: AtomicU64::new(0),
                closes: AtomicU64::new(0),
                yield_close: AtomicBool::new(false),
                hang_close: AtomicBool::new(false),
                fail_close: AtomicBool::new(false),
                connects: AtomicU64::new(0),
                block_connect: AtomicBool::new(false),
                connect_started: AtomicBool::new(false),
                connect_release: tokio::sync::Notify::new(),
                block_list: AtomicBool::new(false),
                list_started: AtomicBool::new(false),
                list_release: tokio::sync::Notify::new(),
                hang_call: AtomicBool::new(false),
                block_call: AtomicBool::new(false),
                call_started: AtomicU64::new(0),
                call_release: tokio::sync::Notify::new(),
                active_calls: AtomicU64::new(0),
                max_active_calls: AtomicU64::new(0),
                fail_list: AtomicBool::new(false),
                call_failure: Mutex::new(None),
                result: Mutex::new(McpCallResult {
                    structured_content: Some(serde_json::json!({"ok":true})),
                    content: None,
                    is_error: false,
                }),
            }
        }

        fn legacy(tools: Vec<McpTool>) -> Self {
            let mut transport = Self::modern(1, tools);
            transport.context.era = McpProtocolEra::Legacy20251125;
            transport.context.version = "2025-11-25".into();
            transport.pages.get_mut().expect("test mutex")[0].ttl_ms = None;
            transport.pages.get_mut().expect("test mutex")[0].cache_scope = None;
            transport
        }

        fn replace_tools(&self, tools: Vec<McpTool>) {
            self.pages.lock().expect("test mutex")[0].tools = tools;
        }
    }

    struct FakeCallLease<'a>(&'a FakeTransport);

    impl Drop for FakeCallLease<'_> {
        fn drop(&mut self) {
            self.0.active_calls.fetch_sub(1, Ordering::AcqRel);
        }
    }

    #[async_trait]
    impl McpTransport for FakeTransport {
        async fn connect(&self, _: CancellationToken) -> Result<McpContext, McpTransportError> {
            self.connects.fetch_add(1, Ordering::Relaxed);
            if self.block_connect.load(Ordering::Relaxed) {
                self.connect_started.store(true, Ordering::Release);
                self.connect_release.notified().await;
            }
            Ok(self.context.clone())
        }
        async fn list_tools(
            &self,
            _: &McpContext,
            _: Option<&str>,
            _: CancellationToken,
        ) -> Result<McpToolPage, McpTransportError> {
            if self.block_list.load(Ordering::Relaxed) {
                self.list_started.store(true, Ordering::Release);
                self.list_release.notified().await;
            }
            if self.fail_list.load(Ordering::Relaxed) {
                return Err(McpTransportError {
                    operation: McpOperation::ListTools,
                    dispatch: McpDispatchState::NotDispatched,
                });
            }
            Ok(self.pages.lock().expect("test mutex")[0].clone())
        }
        async fn call_tool(
            &self,
            _: &McpContext,
            _: McpCallRequest,
            _: CancellationToken,
        ) -> Result<McpCallResult, McpTransportError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let active = self.active_calls.fetch_add(1, Ordering::AcqRel) + 1;
            self.max_active_calls.fetch_max(active, Ordering::AcqRel);
            let _lease = FakeCallLease(self);
            if let Some(error) = self.call_failure.lock().expect("test mutex").clone() {
                return Err(error);
            }
            if self.block_call.load(Ordering::Relaxed) {
                self.call_started.fetch_add(1, Ordering::Release);
                self.call_release.notified().await;
            }
            if self.hang_call.load(Ordering::Relaxed) {
                std::future::pending::<()>().await;
                unreachable!("pending future completed")
            }
            Ok(self.result.lock().expect("test mutex").clone())
        }
        async fn close(
            &self,
            _: McpContext,
            _: CancellationToken,
        ) -> Result<(), McpTransportError> {
            self.closes.fetch_add(1, Ordering::Relaxed);
            if self.hang_close.load(Ordering::Relaxed) {
                std::future::pending::<()>().await;
            }
            if self.yield_close.load(Ordering::Relaxed) {
                tokio::task::yield_now().await;
            }
            if self.fail_close.load(Ordering::Relaxed) {
                Err(McpTransportError {
                    operation: McpOperation::Close,
                    dispatch: McpDispatchState::NotDispatched,
                })
            } else {
                Ok(())
            }
        }
    }

    struct RecordingObserver(Mutex<Vec<McpLifecycleEvent>>, bool);

    impl McpObserver for RecordingObserver {
        fn observe(&self, event: &McpLifecycleEvent) -> Result<(), McpObserverError> {
            self.0.lock().expect("test mutex").push(event.clone());
            if self.1 {
                Err(McpObserverError)
            } else {
                Ok(())
            }
        }
    }

    struct PanicObserver;

    impl McpObserver for PanicObserver {
        fn observe(&self, _: &McpLifecycleEvent) -> Result<(), McpObserverError> {
            panic!("observer panic must not escape")
        }
    }

    struct SlowObserver {
        manager: Mutex<Option<std::sync::Weak<McpCatalogManager>>>,
        saw_snapshot: AtomicBool,
    }

    impl McpObserver for SlowObserver {
        fn observe(&self, _: &McpLifecycleEvent) -> Result<(), McpObserverError> {
            if let Some(manager) = self
                .manager
                .lock()
                .expect("test mutex")
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
            {
                self.saw_snapshot
                    .store(manager.active_snapshot().is_some(), Ordering::Release);
            }
            std::thread::sleep(Duration::from_millis(15));
            Ok(())
        }
    }

    struct BlockingLifecycleObserver {
        operation: McpLifecycleOperation,
        armed: AtomicBool,
        blocked: AtomicBool,
        events: Mutex<Vec<McpLifecycleEvent>>,
        entered: std::sync::mpsc::Sender<()>,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl BlockingLifecycleObserver {
        fn arm(&self) {
            self.armed.store(true, Ordering::Release);
        }

        fn clear_events(&self) {
            self.events.lock().expect("test mutex").clear();
        }

        fn events(&self) -> Vec<McpLifecycleEvent> {
            self.events.lock().expect("test mutex").clone()
        }
    }

    impl McpObserver for BlockingLifecycleObserver {
        fn observe(&self, event: &McpLifecycleEvent) -> Result<(), McpObserverError> {
            self.events.lock().expect("test mutex").push(event.clone());
            if event.operation == self.operation
                && self.armed.load(Ordering::Acquire)
                && !self.blocked.swap(true, Ordering::AcqRel)
            {
                let _ = self.entered.send(());
                let _ = self.release.lock().expect("test mutex").recv();
            }
            Ok(())
        }
    }

    fn blocking_observer(
        operation: McpLifecycleOperation,
    ) -> (
        Arc<BlockingLifecycleObserver>,
        std::sync::mpsc::Receiver<()>,
        std::sync::mpsc::Sender<()>,
    ) {
        let (entered, entered_receiver) = std::sync::mpsc::channel();
        let (release_sender, release) = std::sync::mpsc::channel();
        (
            Arc::new(BlockingLifecycleObserver {
                operation,
                armed: AtomicBool::new(false),
                blocked: AtomicBool::new(false),
                events: Mutex::new(Vec::new()),
                entered,
                release: Mutex::new(release),
            }),
            entered_receiver,
            release_sender,
        )
    }

    fn unique_test_server_id(prefix: &str) -> String {
        static NEXT_TEST_SERVER: AtomicU64 = AtomicU64::new(1);
        format!(
            "{prefix}-{}",
            NEXT_TEST_SERVER.fetch_add(1, Ordering::Relaxed)
        )
    }

    fn manager_for_server(
        transport: Arc<dyn McpTransport>,
        clock: Arc<dyn McpClock>,
        server_id: String,
        observer: Option<Arc<dyn McpObserver>>,
    ) -> McpCatalogManager {
        McpCatalogManager::with_configuration(
            transport,
            server_id,
            McpLimits::default(),
            McpTimeouts::default(),
            McpCachePolicy {
                legacy_ttl: Duration::from_millis(20),
                max_stale: None,
            },
            clock,
            observer,
        )
        .expect("valid manager")
    }

    fn tool(name: &str) -> McpTool {
        McpTool {
            name: name.into(),
            description: "untrusted description".into(),
            input_schema: serde_json::json!({"type":"object"}),
            output_schema: None,
        }
    }

    fn manager(
        transport: Arc<dyn McpTransport>,
        clock: Arc<dyn McpClock>,
        max_stale: Option<Duration>,
        observer: Option<Arc<dyn McpObserver>>,
    ) -> McpCatalogManager {
        McpCatalogManager::with_configuration(
            transport,
            "server",
            McpLimits::default(),
            McpTimeouts::default(),
            McpCachePolicy {
                legacy_ttl: Duration::from_millis(20),
                max_stale,
            },
            clock,
            observer,
        )
        .expect("valid manager")
    }

    async fn call_first(manager: &McpCatalogManager) -> Result<ToolResult, HarnessError> {
        manager.active_tools()[0]
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
    }
    #[test]
    fn ids_are_stable_and_bounded() {
        let id = generated_id("server", "unsafe name / 1");
        assert!(id.starts_with("mcp-server-unsafename1-"));
        assert!(id.len() < 100);
    }
    #[test]
    fn external_schema_reference_is_rejected() {
        assert!(admit_json_value(
            &serde_json::json!({"$ref":"https://invalid/schema"}),
            &McpLimits::default(),
            McpLimits::default().max_catalog_bytes,
            true,
        )
        .is_err());
    }

    #[test]
    fn iterative_admission_rejects_adversarial_schema_without_serializing() {
        let limits = McpLimits::default();
        let mut deep = Value::Null;
        for _ in 0..(limits.max_json_depth + 2) {
            deep = Value::Array(vec![deep]);
        }
        assert!(admit_json_value(&deep, &limits, limits.max_catalog_bytes, true).is_err());

        let wide = Value::Array(
            (0..(limits.max_json_nodes + 1))
                .map(|_| Value::Null)
                .collect(),
        );
        assert!(admit_json_value(&wide, &limits, limits.max_catalog_bytes, true).is_err());

        let large = Value::String("x".repeat(limits.max_string_bytes + 1));
        assert!(admit_json_value(&large, &limits, limits.max_catalog_bytes, true).is_err());
    }

    #[test]
    fn iterative_admission_accepts_node_limit_and_rejects_n_plus_one_before_queueing() {
        let limits = McpLimits {
            max_json_nodes: 3,
            ..McpLimits::default()
        };
        assert!(admit_json_value(
            &Value::Array(vec![Value::Null, Value::Null]),
            &limits,
            limits.max_catalog_bytes,
            true,
        )
        .is_ok());
        assert!(admit_json_value(
            &Value::Array(vec![Value::Null, Value::Null, Value::Null]),
            &limits,
            limits.max_catalog_bytes,
            true,
        )
        .is_err());
    }

    #[tokio::test]
    async fn wide_page_tool_limit_rejects_before_schema_walks() {
        let context = McpContext {
            era: McpProtocolEra::Modern20260728,
            version: "2026-07-28".into(),
            capabilities: BTreeSet::from(["tools".into()]),
            request_context: None,
        };
        let transport = FakeTransport::modern(100, vec![tool("one"), tool("two")]);
        let limits = McpLimits {
            max_tools: 1,
            ..McpLimits::default()
        };
        assert!(matches!(
            fetch_tools(
                &transport,
                &context,
                CancellationToken::new(),
                &limits,
                Duration::from_millis(1),
            )
            .await,
            Err(McpError::InvalidCatalog(_))
        ));
    }

    #[test]
    fn page_estimator_counts_structural_wrappers_at_the_byte_boundary() {
        let page = McpToolPage {
            tools: Vec::new(),
            next_cursor: None,
            ttl_ms: None,
            cache_scope: None,
        };
        let full = admit_page(&page, &McpLimits::default()).expect("page estimate");
        let at_limit = McpLimits {
            max_catalog_bytes: full,
            ..McpLimits::default()
        };
        assert_eq!(admit_page(&page, &at_limit).expect("exact boundary"), full);
        let beyond_limit = McpLimits {
            max_catalog_bytes: full - 1,
            ..McpLimits::default()
        };
        assert!(admit_page(&page, &beyond_limit).is_err());
    }

    #[test]
    fn iterative_result_admission_rejects_deep_wide_and_large_values() {
        let limits = McpLimits::default();
        let mut deep = Value::Null;
        for _ in 0..(limits.max_json_depth + 2) {
            deep = Value::Array(vec![deep]);
        }
        assert!(normalize_result(
            McpCallResult {
                structured_content: Some(deep),
                content: None,
                is_error: false
            },
            &limits,
            false
        )
        .is_err());
        let wide = Value::Array(
            (0..(limits.max_json_nodes + 1))
                .map(|_| Value::Null)
                .collect(),
        );
        assert!(normalize_result(
            McpCallResult {
                structured_content: Some(wide),
                content: None,
                is_error: false
            },
            &limits,
            false
        )
        .is_err());
        let large = Value::String("x".repeat(limits.max_string_bytes + 1));
        assert!(normalize_result(
            McpCallResult {
                structured_content: Some(large),
                content: None,
                is_error: false
            },
            &limits,
            false
        )
        .is_err());
    }

    #[test]
    fn negotiation_context_limits_are_bounded() {
        let mut limits = McpLimits {
            max_context_capabilities: 1,
            ..McpLimits::default()
        };
        let context = McpContext {
            era: McpProtocolEra::Modern20260728,
            version: "2026-07-28".into(),
            capabilities: BTreeSet::from(["tools".into(), "large".into()]),
            request_context: None,
        };
        assert!(validate_context(&context, &limits).is_err());
        limits.max_context_capabilities = 2;
        limits.max_string_bytes = 2;
        assert!(validate_context(&context, &limits).is_err());
    }

    #[tokio::test]
    async fn adapters_share_the_snapshot_context_arc() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one"), tool("two")]));
        let manager = manager(transport, clock, None, None);
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        let state = manager.state.lock().expect("test mutex");
        let active = state.active.as_ref().expect("active snapshot");
        assert!(Arc::ptr_eq(&active.context, &active.tools[0].inner.context));
        assert!(Arc::ptr_eq(&active.context, &active.tools[1].inner.context));
    }

    #[test]
    fn same_server_managers_with_matching_waiter_limits_share_the_global_gate() {
        let clock = Arc::new(FakeClock::default());
        let server_id = unique_test_server_id("matching-waiter-limit");
        let limits = McpLimits {
            max_server_waiters: 3,
            ..McpLimits::default()
        };
        let first = McpCatalogManager::with_configuration(
            Arc::new(FakeTransport::modern(100, vec![tool("one")])),
            server_id.clone(),
            limits.clone(),
            McpTimeouts::default(),
            McpCachePolicy::default(),
            clock.clone(),
            None,
        )
        .expect("first manager");
        let second = McpCatalogManager::with_configuration(
            Arc::new(FakeTransport::modern(100, vec![tool("two")])),
            server_id,
            limits,
            McpTimeouts::default(),
            McpCachePolicy::default(),
            clock,
            None,
        )
        .expect("second manager");
        assert!(Arc::ptr_eq(
            &first.server_call_gate,
            &second.server_call_gate
        ));
        assert_eq!(first.server_call_gate.max_waiters, 3);
    }

    #[test]
    fn same_server_managers_reject_conflicting_waiter_limits() {
        let clock = Arc::new(FakeClock::default());
        let server_id = unique_test_server_id("conflicting-waiter-limit");
        let first = McpCatalogManager::with_configuration(
            Arc::new(FakeTransport::modern(100, vec![tool("one")])),
            server_id.clone(),
            McpLimits {
                max_server_waiters: 2,
                ..McpLimits::default()
            },
            McpTimeouts::default(),
            McpCachePolicy::default(),
            clock.clone(),
            None,
        )
        .expect("first manager establishes the process-wide bound");
        for max_server_waiters in [1, 4] {
            let conflicting = McpCatalogManager::with_configuration(
                Arc::new(FakeTransport::modern(100, vec![tool("two")])),
                server_id.clone(),
                McpLimits {
                    max_server_waiters,
                    ..McpLimits::default()
                },
                McpTimeouts::default(),
                McpCachePolicy::default(),
                clock.clone(),
                None,
            );
            assert!(matches!(
                conflicting,
                Err(McpError::InvalidConfiguration(_))
            ));
        }
        assert_eq!(first.server_call_gate.max_waiters, 2);
    }

    #[test]
    fn server_waiter_limit_must_be_finite_and_nonzero() {
        for max_server_waiters in [0, usize::MAX] {
            let result = McpCatalogManager::with_configuration(
                Arc::new(FakeTransport::modern(100, vec![tool("one")])),
                unique_test_server_id("invalid-waiter-limit"),
                McpLimits {
                    max_server_waiters,
                    ..McpLimits::default()
                },
                McpTimeouts::default(),
                McpCachePolicy::default(),
                Arc::new(FakeClock::default()),
                None,
            );
            assert!(matches!(result, Err(McpError::InvalidConfiguration(_))));
        }
    }

    #[test]
    fn server_gate_prunes_dead_keys_and_enforces_live_bound() {
        let gates = Mutex::new(HashMap::new());
        let first = server_call_gate_from(&gates, "one", 1, 2).expect("first gate");
        assert!(matches!(
            server_call_gate_from(&gates, "two", 1, 2),
            Err(McpError::ResourceLimit)
        ));
        drop(first);
        let second = server_call_gate_from(&gates, "two", 1, 2).expect("pruned gate");
        assert_eq!(second.waiters.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn two_managers_serialize_calls_to_the_same_server() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let first = manager(transport.clone(), clock.clone(), None, None);
        let second = manager(transport.clone(), clock, None, None);
        first
            .refresh(CancellationToken::new())
            .await
            .expect("first refresh");
        second
            .refresh(CancellationToken::new())
            .await
            .expect("second refresh");
        transport.block_call.store(true, Ordering::Release);
        let first_tool = first.active_tools().remove(0);
        let second_tool = second.active_tools().remove(0);
        let first_call = tokio::spawn(async move {
            first_tool
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
        });
        while transport.call_started.load(Ordering::Acquire) != 1 {
            tokio::task::yield_now().await;
        }
        let second_call = tokio::spawn(async move {
            second_tool
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
        });
        for _ in 0..4 {
            tokio::task::yield_now().await;
        }
        assert_eq!(transport.calls.load(Ordering::Acquire), 1);
        assert_eq!(transport.max_active_calls.load(Ordering::Acquire), 1);
        transport.block_call.store(false, Ordering::Release);
        transport.call_release.notify_waiters();
        first_call.await.expect("first call").expect("first result");
        second_call
            .await
            .expect("second call")
            .expect("second result");
        assert_eq!(transport.max_active_calls.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn server_queue_limit_rejects_before_transport_dispatch() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let manager = McpCatalogManager::with_configuration(
            transport.clone(),
            "bounded-queue-server",
            McpLimits {
                max_server_waiters: 1,
                ..McpLimits::default()
            },
            McpTimeouts::default(),
            McpCachePolicy::default(),
            clock,
            None,
        )
        .expect("manager");
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        transport.block_call.store(true, Ordering::Release);
        let tool = manager.active_tools().remove(0);
        let first_tool = Arc::clone(&tool);
        let first = tokio::spawn(async move {
            first_tool
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
        });
        while transport.call_started.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let second_tool = Arc::clone(&tool);
        let second = tokio::spawn(async move {
            second_tool
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
        });
        for _ in 0..4 {
            tokio::task::yield_now().await;
        }
        assert!(matches!(
            tool.execute(serde_json::json!({}), CancellationToken::new())
                .await,
            Err(HarnessError::ResourceLimit(_))
        ));
        assert_eq!(transport.calls.load(Ordering::Acquire), 1);
        transport.block_call.store(false, Ordering::Release);
        transport.call_release.notify_waiters();
        first.await.expect("first task").expect("first result");
        second.await.expect("second task").expect("second result");
        assert_eq!(transport.calls.load(Ordering::Acquire), 2);
    }

    #[tokio::test]
    async fn slow_observer_is_measured_after_lifecycle_locks_are_released() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let observer = Arc::new(SlowObserver {
            manager: Mutex::new(None),
            saw_snapshot: AtomicBool::new(false),
        });
        let manager = Arc::new(manager(transport, clock, None, Some(observer.clone())));
        *observer.manager.lock().expect("test mutex") = Some(Arc::downgrade(&manager));
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        assert!(observer.saw_snapshot.load(Ordering::Acquire));
        assert!(manager.observer_health().slow >= 1);
    }

    #[tokio::test]
    async fn blocked_call_observer_releases_lifecycle_and_admission_guards() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let (observer, entered, release) = blocking_observer(McpLifecycleOperation::Call);
        let manager = Arc::new(manager_for_server(
            transport.clone(),
            clock,
            unique_test_server_id("blocked-call-observer"),
            Some(observer.clone()),
        ));
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("initial refresh");
        observer.arm();

        let tool = manager.active_tools().remove(0);
        let call = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .expect("test runtime")
                .block_on(tool.execute(serde_json::json!({}), CancellationToken::new()))
        });
        entered
            .recv_timeout(Duration::from_secs(1))
            .expect("call observer entered after outcome emission");

        let initial = manager
            .state
            .lock()
            .expect("test mutex")
            .active
            .as_ref()
            .expect("active catalog")
            .clone();
        assert_eq!(initial.in_flight.count.load(Ordering::Acquire), 0);
        assert_eq!(manager.server_call_gate.waiters.load(Ordering::Acquire), 0);
        assert_eq!(manager.server_call_gate.permits.available_permits(), 1);

        manager.invalidate_list_change();
        let refreshed = tokio::time::timeout(
            Duration::from_secs(1),
            manager.refresh(CancellationToken::new()),
        )
        .await
        .expect("refresh is not blocked by call observation")
        .expect("refresh succeeds while observer is blocked");
        assert_eq!(refreshed.generation, 2);
        tokio::time::timeout(Duration::from_secs(1), async {
            while transport.closes.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("retired generation cleanup");
        tokio::time::timeout(
            Duration::from_secs(1),
            manager.close(CancellationToken::new()),
        )
        .await
        .expect("close is not blocked by call observation")
        .expect("close succeeds while observer is blocked");

        assert!(observer.blocked.load(Ordering::Acquire));
        release.send(()).expect("release call observer");
        call.join()
            .expect("call thread")
            .expect("call result after observer release");
        assert_eq!(transport.calls.load(Ordering::Acquire), 1);
        assert_eq!(transport.closes.load(Ordering::Acquire), 2);
    }

    #[tokio::test]
    async fn blocked_refresh_observer_releases_refresh_lock_before_callback() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let (observer, entered, release) = blocking_observer(McpLifecycleOperation::Refresh);
        let manager = Arc::new(manager_for_server(
            transport,
            clock,
            unique_test_server_id("blocked-refresh-observer"),
            Some(observer.clone()),
        ));
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("initial refresh");
        observer.arm();

        let first_manager = Arc::clone(&manager);
        let first = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .expect("test runtime")
                .block_on(first_manager.refresh(CancellationToken::new()))
        });
        entered
            .recv_timeout(Duration::from_secs(1))
            .expect("refresh observer entered after lock release");

        let second = tokio::time::timeout(
            Duration::from_secs(1),
            manager.refresh(CancellationToken::new()),
        )
        .await
        .expect("second refresh is not blocked by observer callback")
        .expect("second refresh succeeds");
        assert_eq!(second.generation, 3);

        release.send(()).expect("release refresh observer");
        assert_eq!(
            first
                .join()
                .expect("refresh thread")
                .expect("first refresh result")
                .generation,
            2
        );
    }

    #[tokio::test]
    async fn blocked_negotiation_observer_allows_a_concurrent_refresh_to_complete() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let (observer, entered, release) = blocking_observer(McpLifecycleOperation::Negotiation);
        let manager = Arc::new(manager_for_server(
            transport,
            clock,
            unique_test_server_id("blocked-negotiation-observer"),
            Some(observer.clone()),
        ));
        observer.arm();

        let first_manager = Arc::clone(&manager);
        let first = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .expect("test runtime")
                .block_on(first_manager.refresh(CancellationToken::new()))
        });
        entered
            .recv_timeout(Duration::from_secs(1))
            .expect("negotiation observer entered after refresh lock release");

        let second = tokio::time::timeout(
            Duration::from_secs(1),
            manager.refresh(CancellationToken::new()),
        )
        .await
        .expect("second refresh is not blocked by negotiation observation")
        .expect("second refresh succeeds");
        assert_eq!(second.generation, 2);
        assert!(manager
            .state
            .lock()
            .expect("test mutex")
            .refresh_tokens
            .is_empty());

        release.send(()).expect("release negotiation observer");
        assert_eq!(
            first
                .join()
                .expect("refresh thread")
                .expect("first refresh result")
                .generation,
            1
        );
        let refresh_events = observer
            .events()
            .into_iter()
            .filter(|event| {
                matches!(
                    event.operation,
                    McpLifecycleOperation::Negotiation | McpLifecycleOperation::Refresh
                )
            })
            .map(|event| event.operation)
            .collect::<Vec<_>>();
        assert_eq!(
            refresh_events,
            vec![
                McpLifecycleOperation::Negotiation,
                McpLifecycleOperation::Negotiation,
                McpLifecycleOperation::Refresh,
                McpLifecycleOperation::Refresh,
            ]
        );
    }

    #[tokio::test]
    async fn blocked_post_connect_clock_failure_observer_allows_close_to_complete() {
        let clock = Arc::new(FakeClock::default());
        clock.set(10);
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let (observer, entered, release) = blocking_observer(McpLifecycleOperation::Refresh);
        let manager = Arc::new(manager_for_server(
            transport.clone(),
            clock.clone(),
            unique_test_server_id("blocked-clock-failure-observer"),
            Some(observer.clone()),
        ));
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("initial refresh");
        observer.clear_events();
        clock.set(9);
        observer.arm();

        let failing_manager = Arc::clone(&manager);
        let failing = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .expect("test runtime")
                .block_on(failing_manager.refresh(CancellationToken::new()))
        });
        entered
            .recv_timeout(Duration::from_secs(1))
            .expect("failed refresh observer entered after cleanup and lock release");
        assert_eq!(transport.closes.load(Ordering::Acquire), 1);
        assert!(manager
            .state
            .lock()
            .expect("test mutex")
            .refresh_tokens
            .is_empty());

        tokio::time::timeout(
            Duration::from_secs(1),
            manager.close(CancellationToken::new()),
        )
        .await
        .expect("close is not blocked by failed refresh observation")
        .expect("close succeeds");

        release.send(()).expect("release failed refresh observer");
        assert!(matches!(
            failing.join().expect("refresh thread"),
            Err(McpError::Clock)
        ));
        let refresh_events = observer
            .events()
            .into_iter()
            .filter(|event| {
                matches!(
                    event.operation,
                    McpLifecycleOperation::Negotiation | McpLifecycleOperation::Refresh
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(refresh_events.len(), 2);
        assert_eq!(
            refresh_events[0].operation,
            McpLifecycleOperation::Negotiation
        );
        assert_eq!(refresh_events[1].operation, McpLifecycleOperation::Refresh);
        assert_eq!(refresh_events[1].outcome, McpLifecycleOutcome::Rejected);
        assert_eq!(transport.closes.load(Ordering::Acquire), 2);
    }

    #[tokio::test]
    async fn blocked_close_observer_releases_close_lock_before_callback() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let (observer, entered, release) = blocking_observer(McpLifecycleOperation::Close);
        let manager = Arc::new(manager_for_server(
            transport,
            clock,
            unique_test_server_id("blocked-close-observer"),
            Some(observer.clone()),
        ));
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("initial refresh");
        observer.arm();

        let first_manager = Arc::clone(&manager);
        let first = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .expect("test runtime")
                .block_on(first_manager.close(CancellationToken::new()))
        });
        entered
            .recv_timeout(Duration::from_secs(1))
            .expect("close observer entered after lock release");

        tokio::time::timeout(
            Duration::from_secs(1),
            manager.close(CancellationToken::new()),
        )
        .await
        .expect("second close is not blocked by observer callback")
        .expect("second close succeeds");

        release.send(()).expect("release close observer");
        first
            .join()
            .expect("close thread")
            .expect("first close result");
    }

    #[tokio::test]
    async fn default_catalog_limit_accepts_exactly_256_tools_in_one_pass() {
        let clock = Arc::new(FakeClock::default());
        let tools = (0..256)
            .map(|index| tool(&format!("tool-{index}")))
            .collect::<Vec<_>>();
        let transport = Arc::new(FakeTransport::modern(100, tools));
        let manager = manager(transport, clock, None, None);
        let snapshot = manager
            .refresh(CancellationToken::new())
            .await
            .expect("catalog");
        assert_eq!(snapshot.tool_count, 256);
    }

    #[tokio::test]
    async fn retired_generation_bound_fails_before_connect() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let limits = McpLimits {
            max_retired_generations: 0,
            ..McpLimits::default()
        };
        let manager = McpCatalogManager::with_configuration(
            transport.clone(),
            "server",
            limits,
            McpTimeouts::default(),
            McpCachePolicy::default(),
            clock,
            None,
        )
        .expect("manager");
        assert!(matches!(
            manager.refresh(CancellationToken::new()).await,
            Err(McpError::CatalogUnavailable)
        ));
        assert!(!transport.connect_started.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn modern_ttl_expires_without_sleep_and_prevents_dispatch() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(10, vec![tool("one")]));
        let clock_manager = manager(transport.clone(), clock.clone(), None, None);
        clock_manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        assert!(call_first(&clock_manager).await.expect("fresh call").ok);
        clock.set(11);
        assert!(matches!(
            call_first(&clock_manager).await,
            Err(HarnessError::InvalidTool(_))
        ));
        assert_eq!(transport.calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn modern_zero_ttl_is_immediately_stale() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(0, vec![tool("one")]));
        let manager = manager(transport.clone(), clock, None, None);
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        assert!(matches!(
            call_first(&manager).await,
            Err(HarnessError::InvalidTool(_))
        ));
        assert_eq!(transport.calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn private_catalog_is_rejected_until_execution_time_auth_context_exists() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        transport.pages.lock().expect("test mutex")[0].cache_scope = Some("private".into());
        let unsupported_manager = manager(transport.clone(), clock, None, None);
        assert!(matches!(
            unsupported_manager.refresh(CancellationToken::new()).await,
            Err(McpError::InvalidCatalog(_))
        ));
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn every_post_connect_refresh_failure_releases_the_new_context() {
        let clock = Arc::new(FakeClock::default());
        let mut unsupported_transport = FakeTransport::modern(100, vec![tool("one")]);
        unsupported_transport.context.capabilities.clear();
        let transport = Arc::new(unsupported_transport);
        let unsupported_manager = manager(transport.clone(), clock, None, None);
        assert!(matches!(
            unsupported_manager.refresh(CancellationToken::new()).await,
            Err(McpError::InvalidCatalog(_))
        ));
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);

        let clock = Arc::new(FakeClock::default());
        clock.set(10);
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let clock_manager = manager(transport.clone(), clock.clone(), None, None);
        clock_manager
            .refresh(CancellationToken::new())
            .await
            .expect("first refresh");
        clock.set(9);
        assert!(matches!(
            clock_manager.refresh(CancellationToken::new()).await,
            Err(McpError::Clock)
        ));
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);

        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        transport.pages.lock().expect("test mutex")[0].ttl_ms = None;
        let ttl_manager = manager(transport.clone(), clock, None, None);
        assert!(matches!(
            ttl_manager.refresh(CancellationToken::new()).await,
            Err(McpError::InvalidCatalog(_))
        ));
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);

        let clock = Arc::new(FakeClock::default());
        clock.set(u64::MAX);
        let transport = Arc::new(FakeTransport::modern(1, vec![tool("one")]));
        let overflow_manager = manager(transport.clone(), clock, None, None);
        assert!(matches!(
            overflow_manager.refresh(CancellationToken::new()).await,
            Err(McpError::Clock)
        ));
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);

        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let generation_manager = manager(transport.clone(), clock, None, None);
        generation_manager
            .state
            .lock()
            .expect("test mutex")
            .next_generation = u64::MAX;
        assert!(matches!(
            generation_manager.refresh(CancellationToken::new()).await,
            Err(McpError::Clock)
        ));
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn concurrent_close_after_connect_releases_the_uninstalled_context() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        transport.block_connect.store(true, Ordering::Relaxed);
        let manager = manager(transport.clone(), clock, None, None);
        let refresh = manager.refresh(CancellationToken::new());
        let close = async {
            while !transport.connect_started.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
            assert!(matches!(
                manager.close(CancellationToken::new()).await,
                Err(McpError::CleanupPending)
            ));
            transport.connect_release.notify_waiters();
        };
        let (refreshed, ()) = tokio::join!(refresh, close);
        assert!(matches!(refreshed, Err(McpError::CatalogUnavailable)));
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn close_after_connect_before_listing_takes_refresh_context_once() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        transport.block_list.store(true, Ordering::Relaxed);
        let manager = Arc::new(manager(transport.clone(), clock, None, None));
        let refresh_manager = Arc::clone(&manager);
        let refresh =
            tokio::spawn(async move { refresh_manager.refresh(CancellationToken::new()).await });
        while !transport.list_started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
        assert!(matches!(
            manager.close(CancellationToken::new()).await,
            Err(McpError::CleanupPending)
        ));
        refresh.abort();
        assert!(refresh.await.expect_err("aborted refresh").is_cancelled());
        manager
            .close(CancellationToken::new())
            .await
            .expect("retry closes retained context");
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn aborted_refreshes_accumulate_under_the_aggregate_context_bound() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        transport.block_list.store(true, Ordering::Relaxed);
        let manager = Arc::new(
            McpCatalogManager::with_configuration(
                transport.clone(),
                "server",
                McpLimits {
                    max_retired_generations: 2,
                    ..McpLimits::default()
                },
                McpTimeouts::default(),
                McpCachePolicy::default(),
                clock,
                None,
            )
            .expect("manager"),
        );
        for expected in 1..=2 {
            transport.list_started.store(false, Ordering::Release);
            let refresh_manager = Arc::clone(&manager);
            let refresh =
                tokio::spawn(
                    async move { refresh_manager.refresh(CancellationToken::new()).await },
                );
            while !transport.list_started.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
            refresh.abort();
            assert!(refresh.await.expect_err("aborted refresh").is_cancelled());
            assert_eq!(manager.shutdown_health().pending_contexts, expected);
        }
        assert_eq!(transport.connects.load(Ordering::Relaxed), 2);
        assert!(matches!(
            manager.refresh(CancellationToken::new()).await,
            Err(McpError::CatalogUnavailable)
        ));
        assert_eq!(transport.connects.load(Ordering::Relaxed), 2);
        transport.block_list.store(false, Ordering::Relaxed);
        assert!(matches!(
            manager.close(CancellationToken::new()).await,
            Err(McpError::CleanupPending)
        ));
        tokio::time::timeout(Duration::from_secs(1), async {
            while manager.shutdown_health().pending_contexts != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("drain retained contexts");
        manager
            .close(CancellationToken::new())
            .await
            .expect("closed");
        assert_eq!(manager.shutdown_health().pending_contexts, 0);
        assert_eq!(transport.closes.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn aggregate_context_bound_counts_pending_retired_and_draining_before_connect() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let manager = McpCatalogManager::with_configuration(
            transport.clone(),
            "server",
            McpLimits {
                max_retired_generations: 3,
                ..McpLimits::default()
            },
            McpTimeouts::default(),
            McpCachePolicy::default(),
            clock,
            None,
        )
        .expect("manager");
        let context = transport.context.clone();
        {
            let mut state = manager.state.lock().expect("test mutex");
            state.pending_refresh_contexts.push(PendingRefreshContext {
                token: 1,
                context: context.clone(),
            });
            state.retired_contexts.push(RetiredContext {
                context: Arc::new(context),
                in_flight: Arc::new(InFlightCalls::default()),
            });
            state.draining_contexts = 1;
        }
        assert!(matches!(
            manager.refresh(CancellationToken::new()).await,
            Err(McpError::CatalogUnavailable)
        ));
        assert_eq!(transport.connects.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn retirement_waits_for_an_admitted_dispatch_and_rejects_new_work() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        transport.block_call.store(true, Ordering::Relaxed);
        let manager = manager(transport.clone(), clock, None, None);
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("first refresh");
        let first_tool = manager.active_tools().remove(0);
        let stale_tool = Arc::clone(&first_tool);
        let active_call = tokio::spawn(async move {
            first_tool
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
        });
        while transport.call_started.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let manager = Arc::new(manager);
        let replacement_manager = Arc::clone(&manager);
        let replacement =
            tokio::spawn(
                async move { replacement_manager.refresh(CancellationToken::new()).await },
            );
        tokio::task::yield_now().await;
        assert!(
            !replacement.is_finished(),
            "retirement must wait for the admitted old-generation dispatch"
        );
        assert_eq!(transport.calls.load(Ordering::Acquire), 1);
        transport.block_call.store(false, Ordering::Relaxed);
        transport.call_release.notify_waiters();
        active_call
            .await
            .expect("old call task")
            .expect("old call result");
        replacement
            .await
            .expect("replacement task")
            .expect("replacement refresh");
        assert!(stale_tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .is_err());
        assert_eq!(transport.calls.load(Ordering::Acquire), 1);
        manager
            .close(CancellationToken::new())
            .await
            .expect("all contexts closed");
    }

    #[tokio::test]
    async fn queued_old_generation_call_is_rejected_after_refresh_wins() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let manager = Arc::new(manager(transport.clone(), clock, None, None));
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        transport.block_call.store(true, Ordering::Release);
        let old = manager.active_tools().remove(0);
        let admitted = Arc::clone(&old);
        let admitted_call = tokio::spawn(async move {
            admitted
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
        });
        while transport.call_started.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let queued = Arc::clone(&old);
        let queued_call = tokio::spawn(async move {
            queued
                .execute(serde_json::json!({}), CancellationToken::new())
                .await
        });
        let refresh_manager = Arc::clone(&manager);
        let refresh =
            tokio::spawn(async move { refresh_manager.refresh(CancellationToken::new()).await });
        for _ in 0..4 {
            tokio::task::yield_now().await;
        }
        transport.block_call.store(false, Ordering::Release);
        transport.call_release.notify_waiters();
        admitted_call
            .await
            .expect("admitted task")
            .expect("admitted result");
        refresh
            .await
            .expect("refresh task")
            .expect("refresh result");
        assert!(matches!(
            queued_call.await.expect("queued task"),
            Err(HarnessError::InvalidTool(_))
        ));
        assert_eq!(transport.calls.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn generated_id_collision_rejects_the_complete_snapshot_and_releases_context() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(
            100,
            vec![tool("first"), tool("second")],
        ));
        let mut manager = manager(transport.clone(), clock, None, None);
        manager.id_deriver = Arc::new(|_, _| "forced-generated-id".into());
        assert!(matches!(
            manager.refresh(CancellationToken::new()).await,
            Err(McpError::InvalidCatalog(_))
        ));
        assert!(manager.active_snapshot().is_none());
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn post_fetch_rejections_emit_one_responded_refresh_event_after_cleanup() {
        let clock = Arc::new(FakeClock::default());
        clock.set(10);
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let observer = Arc::new(RecordingObserver(Mutex::new(Vec::new()), false));
        let manager = manager(
            transport.clone(),
            clock.clone(),
            None,
            Some(observer.clone()),
        );
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("first refresh");
        observer.0.lock().expect("test mutex").clear();
        clock.set(9);
        assert!(manager.refresh(CancellationToken::new()).await.is_err());
        let events = observer.0.lock().expect("test mutex");
        assert_eq!(events.len(), 2, "{events:?}");
        let refresh = events.last().expect("refresh event");
        assert_eq!(refresh.operation, McpLifecycleOperation::Refresh);
        assert_eq!(refresh.outcome, McpLifecycleOutcome::Rejected);
        assert_eq!(refresh.dispatch, McpDispatchState::Responded);
        drop(events);
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn close_retains_failed_cleanup_and_retries_independently_of_caller_cancellation() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        transport.hang_close.store(true, Ordering::Relaxed);
        let mut manager = manager(transport.clone(), clock, None, None);
        manager.timeouts.close = Duration::ZERO;
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert!(manager.close(cancelled).await.is_err());
        assert_eq!(manager.shutdown_health().pending_contexts, 1);
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);

        transport.hang_close.store(false, Ordering::Relaxed);
        manager.timeouts.close = Duration::from_millis(50);
        assert!(manager.close(CancellationToken::new()).await.is_ok());
        assert_eq!(manager.shutdown_health().pending_contexts, 0);
        assert_eq!(transport.closes.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn permanent_close_failure_remains_observable_and_is_not_double_closed() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        transport.fail_close.store(true, Ordering::Relaxed);
        let manager = manager(transport.clone(), clock, None, None);
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        assert!(manager.close(CancellationToken::new()).await.is_err());
        assert_eq!(manager.shutdown_health().pending_contexts, 1);
        assert_eq!(manager.shutdown_health().close_failures, 1);
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);
        assert!(manager.close(CancellationToken::new()).await.is_err());
        assert_eq!(manager.shutdown_health().pending_contexts, 1);
        assert_eq!(transport.closes.load(Ordering::Relaxed), 2);
        transport.fail_close.store(false, Ordering::Relaxed);
        assert!(manager.close(CancellationToken::new()).await.is_ok());
        assert_eq!(manager.shutdown_health().pending_contexts, 0);
        assert_eq!(transport.closes.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn legacy_uses_host_age_policy() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::legacy(vec![tool("one")]));
        let manager = manager(transport.clone(), clock.clone(), None, None);
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        clock.set(21);
        assert!(matches!(
            call_first(&manager).await,
            Err(HarnessError::InvalidTool(_))
        ));
        assert_eq!(transport.calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn list_change_and_old_generation_reject_before_dispatch() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("old")]));
        let manager = manager(transport.clone(), clock, None, None);
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("first refresh");
        let old = manager.active_tools().remove(0);
        manager.invalidate_list_change();
        assert!(matches!(
            old.execute(serde_json::json!({}), CancellationToken::new())
                .await,
            Err(HarnessError::InvalidTool(_))
        ));
        transport.replace_tools(vec![tool("new")]);
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("second refresh");
        assert!(matches!(
            old.execute(serde_json::json!({}), CancellationToken::new())
                .await,
            Err(HarnessError::InvalidTool(_))
        ));
        assert_eq!(transport.calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn failed_refresh_preserves_prior_immutable_snapshot() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let manager = manager(transport.clone(), clock, None, None);
        let first = manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        transport.fail_list.store(true, Ordering::Relaxed);
        assert!(manager.refresh(CancellationToken::new()).await.is_err());
        assert_eq!(
            manager.active_snapshot().expect("snapshot").generation,
            first.generation
        );
        assert!(
            call_first(&manager)
                .await
                .expect("prior snapshot remains callable")
                .ok
        );
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn explicit_bounded_stale_allowance_is_honored() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(10, vec![tool("one")]));
        let manager = manager(
            transport.clone(),
            clock.clone(),
            Some(Duration::from_millis(5)),
            None,
        );
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        clock.set(12);
        assert!(call_first(&manager).await.expect("bounded stale call").ok);
        clock.set(16);
        assert!(matches!(
            call_first(&manager).await,
            Err(HarnessError::InvalidTool(_))
        ));
        assert_eq!(transport.calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn close_rejects_existing_tools_before_dispatch() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let manager = manager(transport.clone(), clock, None, None);
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        let old = manager.active_tools().remove(0);
        manager
            .close(CancellationToken::new())
            .await
            .expect("close");
        assert!(matches!(
            old.execute(serde_json::json!({}), CancellationToken::new())
                .await,
            Err(HarnessError::InvalidTool(_))
        ));
        assert_eq!(transport.calls.load(Ordering::Relaxed), 0);
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn close_timeout_retains_context_until_the_inflight_call_drains() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        transport.hang_call.store(true, Ordering::Relaxed);
        let mut manager = McpCatalogManager::with_configuration(
            transport.clone(),
            "server",
            McpLimits::default(),
            McpTimeouts {
                close: Duration::ZERO,
                ..McpTimeouts::default()
            },
            McpCachePolicy::default(),
            clock,
            None,
        )
        .expect("manager");
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        let tool = manager.active_tools().remove(0);
        let call_cancellation = CancellationToken::new();
        let task_cancellation = call_cancellation.clone();
        let task =
            tokio::spawn(
                async move { tool.execute(serde_json::json!({}), task_cancellation).await },
            );
        while transport.calls.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        assert!(matches!(
            manager.close(CancellationToken::new()).await,
            Err(McpError::CleanupPending)
        ));
        assert_eq!(transport.closes.load(Ordering::Relaxed), 0);
        assert_eq!(manager.shutdown_health().in_progress_contexts, 1);
        assert!(matches!(
            manager.close(CancellationToken::new()).await,
            Err(McpError::CleanupPending)
        ));
        assert_eq!(transport.closes.load(Ordering::Relaxed), 0);
        manager.timeouts.close = Duration::from_millis(50);
        call_cancellation.cancel();
        assert!(matches!(
            task.await.expect("call task"),
            Err(HarnessError::Cancelled)
        ));
        tokio::time::timeout(Duration::from_secs(1), async {
            while transport.closes.load(Ordering::Relaxed) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("deferred close after drain");
        assert_eq!(transport.closes.load(Ordering::Relaxed), 1);
        assert_eq!(manager.shutdown_health().in_progress_contexts, 0);
        assert_eq!(manager.shutdown_health().pending_contexts, 0);
    }

    #[tokio::test]
    async fn cancelled_call_does_not_dispatch_or_retry() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let manager = manager(transport.clone(), clock, None, None);
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        assert!(matches!(
            manager.active_tools()[0]
                .execute(serde_json::json!({}), cancellation)
                .await,
            Err(HarnessError::Cancelled)
        ));
        assert_eq!(transport.calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn cancellation_after_server_permit_before_dispatch_never_calls_transport() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let observer = Arc::new(RecordingObserver(Mutex::new(Vec::new()), false));
        let manager = Arc::new(manager_for_server(
            transport.clone(),
            clock,
            unique_test_server_id("permit-cancellation"),
            Some(observer.clone()),
        ));
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        let (tool, dispatch_gate) = {
            let state = manager.state.lock().expect("test mutex");
            let active = state.active.as_ref().expect("active catalog");
            (
                Arc::clone(&active.tools[0]),
                Arc::clone(&active.dispatch_gate),
            )
        };
        let dispatch_barrier = dispatch_gate.write_owned().await;
        let cancellation = CancellationToken::new();
        let call_cancellation = cancellation.clone();
        let call =
            tokio::spawn(
                async move { tool.execute(serde_json::json!({}), call_cancellation).await },
            );
        while manager.server_call_gate.permits.available_permits() != 0 {
            tokio::task::yield_now().await;
        }

        cancellation.cancel();
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(1), call)
                .await
                .expect("cancellation wins before dispatch")
                .expect("call task"),
            Err(HarnessError::Cancelled)
        ));
        assert_eq!(transport.calls.load(Ordering::Acquire), 0);
        assert_eq!(manager.server_call_gate.waiters.load(Ordering::Acquire), 0);
        assert_eq!(manager.server_call_gate.permits.available_permits(), 1);
        assert!(observer.0.lock().expect("test mutex").iter().any(|event| {
            event.operation == McpLifecycleOperation::Call
                && event.outcome == McpLifecycleOutcome::Cancelled
                && event.dispatch == McpDispatchState::NotDispatched
        }));
        drop(dispatch_barrier);
    }

    #[tokio::test]
    async fn timed_out_call_dispatches_once_and_never_retries() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        transport.hang_call.store(true, Ordering::Relaxed);
        let manager = McpCatalogManager::with_configuration(
            transport.clone(),
            "server",
            McpLimits::default(),
            McpTimeouts {
                call: Duration::ZERO,
                ..McpTimeouts::default()
            },
            McpCachePolicy::default(),
            clock,
            None,
        )
        .expect("manager");
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        assert!(matches!(
            call_first(&manager).await,
            Err(HarnessError::TimedOut(_))
        ));
        assert_eq!(transport.calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn observer_is_redacted_and_failures_are_nonfatal() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("secret-native-name")]));
        let observer = Arc::new(RecordingObserver(Mutex::new(Vec::new()), true));
        let manager = manager(transport, clock, None, Some(observer.clone()));
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh despite observer failure");
        let events = observer.0.lock().expect("test mutex");
        let encoded = serde_json::to_string(&*events).expect("event json");
        assert!(!encoded.contains("secret-native-name"));
        assert!(!encoded.contains("untrusted description"));
        assert!(manager.observer_health().failures > 0);
    }

    #[tokio::test]
    async fn observer_panics_are_contained_and_counted() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let manager = manager(transport, clock, None, Some(Arc::new(PanicObserver)));
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("observer panic is isolated");
        assert!(manager.observer_health().failures > 0);
    }

    #[tokio::test]
    async fn observer_preserves_pre_dispatch_refresh_failure_state() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        transport.fail_list.store(true, Ordering::Relaxed);
        let observer = Arc::new(RecordingObserver(Mutex::new(Vec::new()), false));
        let manager = manager(transport, clock, None, Some(observer.clone()));
        assert!(manager.refresh(CancellationToken::new()).await.is_err());
        let events = observer.0.lock().expect("test mutex");
        let refresh = events
            .iter()
            .find(|event| event.operation == McpLifecycleOperation::Refresh)
            .expect("refresh event");
        assert_eq!(refresh.outcome, McpLifecycleOutcome::Failed);
        assert_eq!(refresh.dispatch, McpDispatchState::NotDispatched);
        assert!(!refresh.cancelled);
    }

    #[tokio::test]
    async fn observer_reports_sanitized_call_outcomes_and_dispatch_certainty() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        let observer = Arc::new(RecordingObserver(Mutex::new(Vec::new()), false));
        let manager = manager(transport.clone(), clock, None, Some(observer.clone()));
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        let tool = manager.active_tools().remove(0);

        let mut speculative = ToolCallContext::new("run", "trace", "call", "tool");
        speculative.caller = Some(ToolCaller::Speculative);
        assert!(matches!(
            tool.execute_with_context(
                &speculative,
                serde_json::json!({}),
                CancellationToken::new()
            )
            .await,
            Err(HarnessError::InvalidTool(_))
        ));

        *transport.call_failure.lock().expect("test mutex") = Some(McpTransportError {
            operation: McpOperation::CallTool,
            dispatch: McpDispatchState::NotDispatched,
        });
        assert!(matches!(
            tool.execute(serde_json::json!({}), CancellationToken::new())
                .await,
            Err(HarnessError::Tool(_))
        ));
        *transport.call_failure.lock().expect("test mutex") = Some(McpTransportError {
            operation: McpOperation::CallTool,
            dispatch: McpDispatchState::PossiblyDispatched,
        });
        assert!(matches!(
            tool.execute(serde_json::json!({}), CancellationToken::new())
                .await,
            Err(HarnessError::Tool(_))
        ));
        *transport.call_failure.lock().expect("test mutex") = None;
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert!(matches!(
            tool.execute(serde_json::json!({}), cancelled).await,
            Err(HarnessError::Cancelled)
        ));

        let events = observer.0.lock().expect("test mutex");
        let calls = events
            .iter()
            .filter(|event| event.operation == McpLifecycleOperation::Call)
            .collect::<Vec<_>>();
        assert!(calls.iter().any(|event| {
            event.outcome == McpLifecycleOutcome::Rejected
                && event.dispatch == McpDispatchState::NotDispatched
        }));
        assert!(calls.iter().any(|event| {
            event.outcome == McpLifecycleOutcome::Failed
                && event.dispatch == McpDispatchState::NotDispatched
        }));
        assert!(calls.iter().any(|event| {
            event.outcome == McpLifecycleOutcome::Failed
                && event.dispatch == McpDispatchState::PossiblyDispatched
        }));
        assert!(calls.iter().any(|event| {
            event.outcome == McpLifecycleOutcome::Cancelled
                && event.cancelled
                && event.dispatch == McpDispatchState::NotDispatched
        }));
    }

    #[tokio::test]
    async fn observer_reports_timeout_as_possibly_dispatched() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(100, vec![tool("one")]));
        transport.hang_call.store(true, Ordering::Relaxed);
        let observer = Arc::new(RecordingObserver(Mutex::new(Vec::new()), false));
        let manager = McpCatalogManager::with_configuration(
            transport,
            "server",
            McpLimits::default(),
            McpTimeouts {
                call: Duration::ZERO,
                ..McpTimeouts::default()
            },
            McpCachePolicy::default(),
            clock,
            Some(observer.clone()),
        )
        .expect("manager");
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        assert!(matches!(
            call_first(&manager).await,
            Err(HarnessError::TimedOut(_))
        ));
        let event = observer
            .0
            .lock()
            .expect("test mutex")
            .iter()
            .find(|event| event.operation == McpLifecycleOperation::Call)
            .expect("call event")
            .clone();
        assert_eq!(event.outcome, McpLifecycleOutcome::TimedOut);
        assert_eq!(event.dispatch, McpDispatchState::PossiblyDispatched);
    }

    #[test]
    fn normalization_enforces_bytes_depth_blocks_and_schema() {
        let mut limits = McpLimits {
            max_response_bytes: 3,
            ..McpLimits::default()
        };
        assert!(matches!(
            normalize_result(
                McpCallResult {
                    structured_content: Some(serde_json::json!("toolong")),
                    content: None,
                    is_error: false
                },
                &limits,
                false
            ),
            Err(HarnessError::ResourceLimit(_))
        ));
        limits.max_response_bytes = 1024;
        limits.max_json_depth = 1;
        assert!(matches!(
            normalize_result(
                McpCallResult {
                    structured_content: Some(serde_json::json!({"nested": true})),
                    content: None,
                    is_error: false
                },
                &limits,
                false
            ),
            Err(HarnessError::ResourceLimit(_))
        ));
        limits.max_json_depth = 32;
        limits.max_content_blocks = 1;
        assert!(matches!(
            normalize_result(
                McpCallResult {
                    structured_content: None,
                    content: Some(serde_json::json!([1, 2])),
                    is_error: false
                },
                &limits,
                false
            ),
            Err(HarnessError::ResourceLimit(_))
        ));
        limits.max_content_blocks = 2;
        assert!(matches!(
            normalize_result(
                McpCallResult {
                    structured_content: None,
                    content: Some(serde_json::json!("x")),
                    is_error: false
                },
                &limits,
                true
            ),
            Err(HarnessError::Tool(_))
        ));
    }

    #[test]
    fn error_results_use_static_error_with_bounded_normalized_content() {
        let result = normalize_result(
            McpCallResult {
                structured_content: None,
                content: Some(serde_json::json!([{"text":"server controlled"}])),
                is_error: true,
            },
            &McpLimits::default(),
            true,
        )
        .expect("normalizes failure");
        assert!(!result.0.ok);
        assert_eq!(result.0.error.as_deref(), Some("MCP tool reported failure"));
        assert_ne!(result.0.output, Value::Null);
    }
}
