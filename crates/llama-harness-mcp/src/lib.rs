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
    collections::{BTreeSet, HashSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, Weak,
    },
    time::{Duration, Instant},
};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

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
    /// Core registration failed.
    #[error(transparent)]
    Core(#[from] HarnessError),
    /// A catalog is unavailable, expired, invalidated, or already closed.
    #[error("MCP catalog unavailable")]
    CatalogUnavailable,
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
}

#[derive(Default)]
struct ObserverHub {
    observer: Option<Arc<dyn McpObserver>>,
    attempted: AtomicU64,
    failures: AtomicU64,
}

impl ObserverHub {
    fn emit(&self, event: McpLifecycleEvent) {
        if let Some(observer) = &self.observer {
            self.attempted.fetch_add(1, Ordering::Relaxed);
            if observer.observe(&event).is_err() {
                self.failures.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn health(&self) -> McpObserverHealth {
        McpObserverHealth {
            attempted: self.attempted.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
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
    context: McpContext,
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

struct CatalogState {
    active: Option<Arc<ActiveCatalog>>,
    closed: bool,
    invalidated: bool,
    last_now_ms: u64,
    next_generation: u64,
}

impl Default for CatalogState {
    fn default() -> Self {
        Self {
            active: None,
            closed: false,
            invalidated: false,
            last_now_ms: 0,
            next_generation: 1,
        }
    }
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
    observer: Arc<ObserverHub>,
    registration_group: ToolRegistrationGroup,
    in_flight: Arc<InFlightCalls>,
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
        Ok(Self {
            server_id: server_id.clone(),
            transport,
            limits,
            timeouts,
            cache_policy,
            clock,
            state: Arc::new(Mutex::new(CatalogState::default())),
            refresh_lock: tokio::sync::Mutex::new(()),
            observer: Arc::new(ObserverHub {
                observer,
                ..ObserverHub::default()
            }),
            registration_group: ToolRegistrationGroup::new(format!("mcp:{server_id}"))
                .map_err(McpError::Core)?,
            in_flight: Arc::new(InFlightCalls::default()),
        })
    }

    /// Refreshes a complete catalog and atomically installs it on success.
    pub async fn refresh(
        &self,
        cancellation: CancellationToken,
    ) -> Result<McpCatalogSnapshot, McpError> {
        let _refresh = self.refresh_lock.lock().await;
        if self.is_closed() {
            return Err(McpError::CatalogUnavailable);
        }
        let started = Instant::now();
        let built = self.fetch_complete(cancellation.child_token()).await;
        let elapsed = elapsed_ms(started);
        let (context, tools, scope, bytes, pages) = match built {
            Ok(value) => value,
            Err(error) => {
                let cancelled = cancellation.is_cancelled();
                self.observer.emit(event(
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
            Err(error) => return self.fail_refresh_context(context, error).await,
        };
        let ttl = match context.era {
            McpProtocolEra::Modern20260728 => match tools.cache_ttl {
                Some(ttl) => ttl,
                None => {
                    return self
                        .fail_refresh_context(
                            context,
                            McpError::InvalidCatalog("modern catalog omitted ttl".into()),
                        )
                        .await
                }
            },
            McpProtocolEra::Legacy20251125 => match duration_ms(self.cache_policy.legacy_ttl) {
                Ok(ttl) => ttl,
                Err(error) => return self.fail_refresh_context(context, error).await,
            },
        };
        let expires_at_ms = match now.checked_add(ttl) {
            Some(value) => value,
            None => return self.fail_refresh_context(context, McpError::Clock).await,
        };
        let max_stale = match self.cache_policy.max_stale.map(duration_ms).transpose() {
            Ok(value) => value.unwrap_or(0),
            Err(error) => return self.fail_refresh_context(context, error).await,
        };
        let stale_until_ms = expires_at_ms.checked_add(max_stale).ok_or(McpError::Clock);
        let stale_until_ms = match stale_until_ms {
            Ok(value) => value,
            Err(error) => return self.fail_refresh_context(context, error).await,
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
            Err(error) => return self.fail_refresh_context(context, error).await,
        };
        let managed = tools
            .items
            .into_iter()
            .map(|tool| {
                Arc::new(ManagedMcpToolAdapter::new(
                    Arc::new(McpToolAdapter::new(
                        generated_id(&self.server_id, &tool.name),
                        self.server_id.clone(),
                        tool,
                        context.clone(),
                        Arc::clone(&self.transport),
                        self.timeouts.call,
                        self.limits.clone(),
                        Arc::clone(&self.observer),
                    )),
                    generation,
                    Arc::downgrade(&self.state),
                    Arc::clone(&self.clock),
                    Arc::clone(&self.in_flight),
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
                    context,
                    McpError::InvalidCatalog("generated ID collision".into()),
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
            context,
            expires_at_ms,
            stale_until_ms,
            tools: managed,
        });
        let previous = match (|| {
            let mut state = self
                .state
                .lock()
                .map_err(|_| McpError::CatalogUnavailable)?;
            if state.closed {
                return Err(McpError::CatalogUnavailable);
            }
            let previous = state.active.replace(Arc::clone(&snapshot));
            state.invalidated = false;
            Ok(previous)
        })() {
            Ok(previous) => previous,
            Err(error) => {
                return self
                    .fail_refresh_context(snapshot.context.clone(), error)
                    .await
            }
        };
        if let Some(previous) = previous {
            self.schedule_close_after_drain(previous.context.clone());
        }
        self.observer.emit(event(
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
        let context = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| McpError::CatalogUnavailable)?;
            if state.closed {
                return Ok(());
            }
            state.closed = true;
            state.invalidated = true;
            state.active.take().map(|snapshot| snapshot.context.clone())
        };
        let started = Instant::now();
        let result = match context {
            Some(context) => {
                if tokio::time::timeout(self.timeouts.close, wait_for_drain(&self.in_flight))
                    .await
                    .is_err()
                {
                    self.schedule_close_after_drain(context);
                    Err(McpError::Transport(McpTransportError {
                        operation: McpOperation::Close,
                        dispatch: McpDispatchState::PossiblyDispatched,
                    }))
                } else {
                    close_context(
                        self.transport.as_ref(),
                        context,
                        cancellation.child_token(),
                        self.timeouts.close,
                    )
                    .await
                    .map_err(McpError::from)
                }
            }
            None => Ok(()),
        };
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

    fn is_closed(&self) -> bool {
        self.state.lock().map(|state| state.closed).unwrap_or(true)
    }

    async fn discard_context(&self, context: McpContext) {
        let _ = close_context(
            self.transport.as_ref(),
            context,
            CancellationToken::new(),
            self.timeouts.close,
        )
        .await;
    }

    async fn fail_refresh_context<T>(
        &self,
        context: McpContext,
        error: McpError,
    ) -> Result<T, McpError> {
        self.discard_context(context).await;
        Err(error)
    }

    fn schedule_close_after_drain(&self, context: McpContext) {
        let transport = Arc::clone(&self.transport);
        let calls = Arc::clone(&self.in_flight);
        let timeout = self.timeouts.close;
        let observer = Arc::clone(&self.observer);
        tokio::spawn(async move {
            wait_for_drain(&calls).await;
            let started = Instant::now();
            let result = close_context(
                transport.as_ref(),
                context,
                CancellationToken::new(),
                timeout,
            )
            .await;
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
        cancellation: CancellationToken,
    ) -> Result<(McpContext, FetchedTools, Option<McpCacheScope>, u64, u64), McpError> {
        let negotiation_started = Instant::now();
        let context = tokio::select! {
            _ = cancellation.cancelled() => return Err(McpError::Transport(McpTransportError { operation: McpOperation::Connect, dispatch: McpDispatchState::NotDispatched })),
            result = tokio::time::timeout(self.timeouts.connect, self.transport.connect(cancellation.child_token())) => result.map_err(|_| McpTransportError { operation: McpOperation::Connect, dispatch: McpDispatchState::PossiblyDispatched })??,
        };
        if !supported_context(&context) {
            return Err(McpError::InvalidCatalog(
                "unsupported version or missing tools capability".into(),
            ));
        }
        self.observer.emit(event(
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
                let _ = close_context(
                    self.transport.as_ref(),
                    context,
                    CancellationToken::new(),
                    self.timeouts.close,
                )
                .await;
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
}

impl ManagedMcpToolAdapter {
    fn new(
        inner: Arc<McpToolAdapter>,
        generation: u64,
        state: Weak<Mutex<CatalogState>>,
        clock: Arc<dyn McpClock>,
        in_flight: Arc<InFlightCalls>,
    ) -> Self {
        Self {
            inner,
            generation,
            state,
            clock,
            in_flight,
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
        let (stale, _lease) = self.check_active()?;
        self.inner
            .execute_managed(call, arguments, cancellation, stale)
            .await
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
        let page_bytes = serde_json::to_vec(&page)
            .map_err(|_| McpError::InvalidCatalog("unserializable page".into()))?
            .len();
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
            if items.len() > limits.max_tools {
                return Err(McpError::InvalidCatalog("tool limit exceeded".into()));
            }
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

fn duration_ms(duration: Duration) -> Result<u64, McpError> {
    u64::try_from(duration.as_millis()).map_err(|_| McpError::Clock)
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn outcome_for_error(error: &McpError) -> McpLifecycleOutcome {
    match error {
        McpError::Transport(McpTransportError {
            operation: _,
            dispatch: McpDispatchState::NotDispatched,
        }) => McpLifecycleOutcome::Cancelled,
        McpError::CatalogUnavailable | McpError::Clock | McpError::InvalidCatalog(_) => {
            McpLifecycleOutcome::Rejected
        }
        McpError::Transport(_) | McpError::Core(_) => McpLifecycleOutcome::Failed,
    }
}

fn dispatch_for_error(error: &McpError) -> McpDispatchState {
    match error {
        McpError::Transport(error) => error.dispatch,
        McpError::CatalogUnavailable
        | McpError::Clock
        | McpError::InvalidCatalog(_)
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
    context: McpContext,
    native_name: String,
    call_timeout: Duration,
    limits: McpLimits,
    observer: Arc<ObserverHub>,
}
impl McpToolAdapter {
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: String,
        server_id: String,
        tool: McpTool,
        context: McpContext,
        transport: Arc<dyn McpTransport>,
        call_timeout: Duration,
        limits: McpLimits,
        observer: Arc<ObserverHub>,
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
        self.execute_managed(call, arguments, cancellation, false)
            .await
    }
}

impl McpToolAdapter {
    async fn execute_managed(
        &self,
        call: &ToolCallContext,
        arguments: Value,
        cancellation: CancellationToken,
        stale: bool,
    ) -> Result<ToolResult, HarnessError> {
        if call.caller == Some(ToolCaller::Speculative) {
            return Err(HarnessError::InvalidTool(
                "MCP tools cannot be speculative".into(),
            ));
        }
        if cancellation.is_cancelled() {
            self.observer.emit(event(
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
            ));
            return Err(HarnessError::Cancelled);
        }
        let started = Instant::now();
        let correlation = Some(McpCallCorrelation::from(call));
        let result = tokio::select! {
            _ = cancellation.cancelled() => Err(HarnessError::Cancelled),
            result = tokio::time::timeout(self.call_timeout, self.transport.call_tool(&self.context, McpCallRequest { name: self.native_name.clone(), arguments, context: call.clone() }, cancellation.child_token())) => match result {
                Ok(Ok(result)) => Ok(result),
                Ok(Err(_)) => Err(HarnessError::Tool("MCP transport failure".into())),
                Err(_) => Err(HarnessError::TimedOut("MCP tool call".into())),
            },
        };
        let elapsed = elapsed_ms(started);
        match result {
            Ok(result) => {
                let normalized = normalize_result(
                    &result,
                    &self.limits,
                    self.definition.output_schema.is_some(),
                );
                let (outcome, count, bytes) = match &normalized {
                    Ok(_) => (
                        McpLifecycleOutcome::Succeeded,
                        content_block_count(&result),
                        result_bytes(&result).unwrap_or(0),
                    ),
                    Err(_) => (McpLifecycleOutcome::Rejected, 0, 0),
                };
                self.observer.emit(event(
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
                ));
                normalized
            }
            Err(error) => {
                let outcome = match error {
                    HarnessError::Cancelled => McpLifecycleOutcome::Cancelled,
                    HarnessError::TimedOut(_) => McpLifecycleOutcome::TimedOut,
                    _ => McpLifecycleOutcome::Failed,
                };
                let cancelled = matches!(error, HarnessError::Cancelled);
                self.observer.emit(event(
                    McpLifecycleOperation::Call,
                    outcome,
                    elapsed,
                    0,
                    0,
                    0,
                    stale,
                    cancelled,
                    McpDispatchState::PossiblyDispatched,
                    correlation,
                ));
                Err(error)
            }
        }
    }
}

fn normalize_result(
    result: &McpCallResult,
    limits: &McpLimits,
    has_output_schema: bool,
) -> Result<ToolResult, HarnessError> {
    let bytes = result_bytes(result)?;
    if usize::try_from(bytes).unwrap_or(usize::MAX) > limits.max_response_bytes {
        return Err(HarnessError::ResourceLimit(
            "MCP response exceeds byte limit".into(),
        ));
    }
    let values = [result.structured_content.as_ref(), result.content.as_ref()];
    if values
        .iter()
        .flatten()
        .any(|value| json_depth(value) > limits.max_json_depth)
    {
        return Err(HarnessError::ResourceLimit(
            "MCP response exceeds depth limit".into(),
        ));
    }
    if usize::try_from(content_block_count(result)).unwrap_or(usize::MAX)
        > limits.max_content_blocks
    {
        return Err(HarnessError::ResourceLimit(
            "MCP response exceeds content block limit".into(),
        ));
    }
    if !result.is_error && has_output_schema && result.structured_content.is_none() {
        return Err(HarnessError::Tool(
            "MCP structured content is required".into(),
        ));
    }
    let output = result
        .structured_content
        .clone()
        .or_else(|| result.content.clone())
        .unwrap_or(Value::Null);
    if result.is_error {
        Ok(ToolResult::new(
            false,
            output,
            Some("MCP tool reported failure".into()),
        ))
    } else {
        Ok(ToolResult::success(output))
    }
}

fn result_bytes(result: &McpCallResult) -> Result<u64, HarnessError> {
    let structured = result
        .structured_content
        .as_ref()
        .map(serde_json::to_vec)
        .transpose()
        .map_err(|_| HarnessError::Tool("MCP result normalization failed".into()))?
        .map_or(0usize, |bytes| bytes.len());
    let content = result
        .content
        .as_ref()
        .map(serde_json::to_vec)
        .transpose()
        .map_err(|_| HarnessError::Tool("MCP result normalization failed".into()))?
        .map_or(0usize, |bytes| bytes.len());
    u64::try_from(
        structured
            .checked_add(content)
            .ok_or_else(|| HarnessError::ResourceLimit("MCP response exceeds byte limit".into()))?,
    )
    .map_err(|_| HarnessError::ResourceLimit("MCP response exceeds byte limit".into()))
}

fn content_block_count(result: &McpCallResult) -> u64 {
    result
        .content
        .as_ref()
        .and_then(Value::as_array)
        .map_or(0, |blocks| u64::try_from(blocks.len()).unwrap_or(u64::MAX))
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
        let encoded = serde_json::to_vec(schema)
            .map_err(|_| McpError::InvalidCatalog("invalid schema".into()))?;
        if encoded.len() > limits.max_catalog_bytes
            || json_depth(schema) > limits.max_json_depth
            || has_external_reference(schema)
        {
            return Err(McpError::InvalidCatalog(
                "unsafe or oversized schema".into(),
            ));
        }
    }
    Ok(())
}
fn json_depth(v: &Value) -> usize {
    match v {
        Value::Array(a) => 1 + a.iter().map(json_depth).max().unwrap_or(0),
        Value::Object(o) => 1 + o.values().map(json_depth).max().unwrap_or(0),
        _ => 1,
    }
}
fn has_external_reference(v: &Value) -> bool {
    match v {
        Value::Object(o) => o.iter().any(|(k, v)| {
            matches!(k.as_str(), "$ref" | "$dynamicRef" | "$recursiveRef")
                && v.as_str().is_some_and(|s| !s.starts_with('#'))
                || has_external_reference(v)
        }),
        Value::Array(a) => a.iter().any(has_external_reference),
        _ => false,
    }
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
        hang_call: AtomicBool,
        fail_list: AtomicBool,
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
                hang_call: AtomicBool::new(false),
                fail_list: AtomicBool::new(false),
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

    #[async_trait]
    impl McpTransport for FakeTransport {
        async fn connect(&self, _: CancellationToken) -> Result<McpContext, McpTransportError> {
            Ok(self.context.clone())
        }
        async fn list_tools(
            &self,
            _: &McpContext,
            _: Option<&str>,
            _: CancellationToken,
        ) -> Result<McpToolPage, McpTransportError> {
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
            Ok(())
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
        assert!(has_external_reference(
            &serde_json::json!({"$ref":"https://invalid/schema"})
        ));
    }

    #[tokio::test]
    async fn modern_ttl_expires_without_sleep_and_prevents_dispatch() {
        let clock = Arc::new(FakeClock::default());
        let transport = Arc::new(FakeTransport::modern(10, vec![tool("one")]));
        let manager = manager(transport.clone(), clock.clone(), None, None);
        manager
            .refresh(CancellationToken::new())
            .await
            .expect("refresh");
        assert!(call_first(&manager).await.expect("fresh call").ok);
        clock.set(11);
        assert!(matches!(
            call_first(&manager).await,
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
        let manager = manager(transport, clock, None, None);
        assert!(matches!(
            manager.refresh(CancellationToken::new()).await,
            Err(McpError::InvalidCatalog(_))
        ));
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

    #[test]
    fn normalization_enforces_bytes_depth_blocks_and_schema() {
        let mut limits = McpLimits {
            max_response_bytes: 3,
            ..McpLimits::default()
        };
        assert!(matches!(
            normalize_result(
                &McpCallResult {
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
                &McpCallResult {
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
                &McpCallResult {
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
                &McpCallResult {
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
            &McpCallResult {
                structured_content: None,
                content: Some(serde_json::json!([{"text":"server controlled"}])),
                is_error: true,
            },
            &McpLimits::default(),
            true,
        )
        .expect("normalizes failure");
        assert!(!result.ok);
        assert_eq!(result.error.as_deref(), Some("MCP tool reported failure"));
        assert_ne!(result.output, Value::Null);
    }
}
