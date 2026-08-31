//! Optional, provider-neutral Model Context Protocol tool integration.
//!
//! This crate imports a complete validated MCP catalog through the ordinary
//! `ToolRegistry` boundary. It intentionally contains no JSON-RPC client.

#![deny(missing_docs)]

use async_trait::async_trait;
use llama_harness_core::{
    CancellationSafety, ExecutionLocation, HarnessError, NetworkEgress, SpeculationPolicy, Tool,
    ToolCallContext, ToolCaller, ToolDefinition, ToolDiscoveryMetadata, ToolRegistry, ToolResult,
    ToolRisk,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeSet, HashSet},
    sync::Arc,
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
/// Immutable provenance retained by each MCP adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct McpToolProvenance {
    /// Host canonical server identifier.
    pub server_id: String,
    /// Exact native name, not model-visible metadata.
    pub native_name: String,
    /// Negotiated era.
    pub era: McpProtocolEra,
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
}
impl Default for McpLimits {
    fn default() -> Self {
        Self {
            max_pages: 32,
            max_tools: 256,
            max_catalog_bytes: 512 * 1024,
            max_json_depth: 32,
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
}

/// Retrieves and validates a complete catalog, then registers it atomically with respect to validation.
pub async fn import_catalog(
    registry: &mut ToolRegistry,
    transport: Arc<dyn McpTransport>,
    server_id: impl Into<String>,
    cancellation: CancellationToken,
    limits: McpLimits,
) -> Result<Vec<McpToolProvenance>, McpError> {
    let server_id = validate_server_id(server_id.into())?;
    let context = transport.connect(cancellation.child_token()).await?;
    if !matches!(
        (context.era, context.version.as_str()),
        (McpProtocolEra::Modern20260728, "2026-07-28")
            | (McpProtocolEra::Legacy20251125, "2025-11-25")
    ) || !context.capabilities.contains("tools")
    {
        return Err(McpError::InvalidCatalog(
            "unsupported version or missing tools capability".into(),
        ));
    }
    let mut cursor = None;
    let mut seen_cursors = HashSet::new();
    let mut seen_names = HashSet::new();
    let mut tools = Vec::new();
    let mut bytes = 0usize;
    for _ in 0..limits.max_pages {
        let page = transport
            .list_tools(&context, cursor.as_deref(), cancellation.child_token())
            .await?;
        bytes = bytes.saturating_add(
            serde_json::to_vec(&page)
                .map_err(|_| McpError::InvalidCatalog("unserializable page".into()))?
                .len(),
        );
        if bytes > limits.max_catalog_bytes {
            return Err(McpError::InvalidCatalog(
                "catalog exceeds byte limit".into(),
            ));
        }
        for tool in page.tools {
            if !seen_names.insert(tool.name.clone()) {
                return Err(McpError::InvalidCatalog(
                    "duplicate native tool name".into(),
                ));
            }
            validate_tool(&tool, &limits)?;
            tools.push(tool);
            if tools.len() > limits.max_tools {
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
    let mut staged = ToolRegistry::default();
    let mut adapters = Vec::new();
    let mut ids = HashSet::new();
    for tool in tools {
        let id = generated_id(&server_id, &tool.name);
        if !ids.insert(id.clone()) || registry.get(&id).is_some() {
            return Err(McpError::InvalidCatalog("generated ID collision".into()));
        }
        let adapter = Arc::new(McpToolAdapter::new(
            id,
            server_id.clone(),
            tool,
            context.clone(),
            Arc::clone(&transport),
        ));
        staged.register_with_discovery(adapter.clone(), ToolDiscoveryMetadata::deferred())?;
        adapters.push(adapter);
    }
    for adapter in &adapters {
        registry.register_with_discovery(adapter.clone(), ToolDiscoveryMetadata::deferred())?;
    }
    Ok(adapters.into_iter().map(|a| a.provenance.clone()).collect())
}

struct McpToolAdapter {
    definition: ToolDefinition,
    provenance: McpToolProvenance,
    transport: Arc<dyn McpTransport>,
    context: McpContext,
    native_name: String,
}
impl McpToolAdapter {
    fn new(
        id: String,
        server_id: String,
        tool: McpTool,
        context: McpContext,
        transport: Arc<dyn McpTransport>,
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
            provenance: McpToolProvenance {
                server_id,
                native_name: native_name.clone(),
                era: context.era,
            },
            transport,
            context,
            native_name,
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
        if call.caller == Some(ToolCaller::Speculative) {
            return Err(HarnessError::InvalidTool(
                "MCP tools cannot be speculative".into(),
            ));
        }
        let result = self
            .transport
            .call_tool(
                &self.context,
                McpCallRequest {
                    name: self.native_name.clone(),
                    arguments,
                    context: call.clone(),
                },
                cancellation,
            )
            .await
            .map_err(|_| HarnessError::Tool("MCP transport failure".into()))?;
        let output = result
            .structured_content
            .or(result.content)
            .unwrap_or(Value::Null);
        Ok(if result.is_error {
            ToolResult::failure("MCP tool reported failure")
        } else {
            ToolResult::success(output)
        })
    }
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
        &blake3::hash(native.as_bytes()).to_hex()[..12]
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
    #[test]
    fn ids_are_stable_and_bounded() {
        let id = generated_id("server", "unsafe name / 1");
        assert!(id.starts_with("mcp-server-unsafe-name-1-"));
        assert!(id.len() < 100);
    }
    #[test]
    fn external_schema_reference_is_rejected() {
        assert!(has_external_reference(
            &serde_json::json!({"$ref":"https://invalid/schema"})
        ));
    }
}
