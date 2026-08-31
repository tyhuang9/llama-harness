# llama-harness-mcp

Optional, transport-neutral MCP catalog adapter for `llama-harness`.

The crate deliberately does not implement JSON-RPC or select a network client. Hosts supply an
`McpTransport`; validated remote tools are registered as normal deferred core tools.

For long-lived integrations, use `McpCatalogManager`. It builds a complete
validated immutable snapshot before replacing the active catalog, rejects tools
from invalidated, expired, replaced, or closed generations before native
dispatch, and only permits stale use when the host configures a bounded
`McpCachePolicy::max_stale`. Modern catalogs require consistent `ttl_ms` and
`cache_scope`; legacy catalogs use the host-assigned age policy.

`McpObserver` receives metadata-only lifecycle events. Events include local
duration, bounded counts and bytes, page/cache/stale/cancellation/dispatch
state, and core run/trace/call correlation. They intentionally exclude server
metadata, requests, results, schemas, frames, endpoints, credentials, and
server error text. Observer failures are suppressed and exposed through
`McpCatalogManager::observer_health`.
