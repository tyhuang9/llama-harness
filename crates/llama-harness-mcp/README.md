# llama-harness-mcp

Optional, transport-neutral MCP catalog adapter for `llama-harness`.

The crate deliberately does not implement JSON-RPC or select a network client. Hosts supply an
`McpTransport`; validated remote tools are registered as normal deferred core tools.

## Compatibility and host boundary

The adapter explicitly accepts only the modern `2026-07-28` discovery era and
the legacy `2025-11-25` initialize era, and requires the negotiated `tools`
capability. A host owns endpoint selection, authentication, credentials,
connection lifetime, and the transport implementation. It must never treat
server descriptions, annotations, schemas, cache hints, errors, or request
context as trusted authority.

Imported tools are high-risk, state-changing, non-idempotent, non-parallel,
unknown-cancellation, Direct-only tools. They are deferred for normal core
discovery, remain subject to the application allowlist, argument validation,
policy, approval, call limits, cancellation, output validation, and the
existing `ToolBroker`; there is no supported MCP execution bypass. Calls are
never retried after dispatch because a remote effect may be uncertain.

`McpCatalogManager` installs only a fully validated immutable snapshot. Modern
`tools/list` pages use MCP's `ttlMs` hint (zero is immediately stale) and a
consistent public `cacheScope`; the manager uses the shortest page TTL for a
complete catalog. Private cache scope is deliberately rejected in this release:
the core call context does not yet carry an enforceable execution-time
authorization capability. A legacy snapshot uses the host-selected age.
List-change invalidation and expiry fail closed unless a host deliberately
configures a bounded stale allowance. Refresh never mutates an active runner,
and old, invalidated, expired, or closed snapshots reject before dispatch.

Lifecycle observation is metadata-only. It deliberately redacts endpoints,
credentials, native names, arguments, results, schemas, raw frames, request
contexts, and server-controlled error text. The crate is intended for
same-process hosts or host-provided remote transports; it does not supply a
reference network transport, task/resource/prompt import, speculative calls,
an OAuth UI, or a real MCP client.

For long-lived integrations, use `McpCatalogManager`. It builds a complete
validated immutable snapshot before replacing the active catalog, rejects tools
from invalidated, expired, replaced, or closed generations before native
dispatch, and only permits stale use when the host configures a bounded
`McpCachePolicy::max_stale`. Modern catalogs require consistent `cacheScope`
and honor per-page `ttlMs`; legacy catalogs use the host-assigned age policy.

Registry-integrated hosts should use `refresh_registered`, passing their latest
immutable `ToolRegistry` and adopting the returned registry. It validates the
candidate through the core registry's actual group replacement before changing
the active MCP generation, so an invalid schema, invalid generated adapter, or
collision with a host tool leaves the prior generation installed and callable.
A successful empty catalog removes only this manager's group and preserves
unrelated tools. `refresh` is the catalog-only workflow; `replace_registered`
can bind one already-refreshed catalog (including an active empty catalog) for
migration. After either registry operation succeeds, catalog-only `refresh`
rejects before transport I/O and all later refreshes must use
`refresh_registered`.

`McpObserver` receives metadata-only lifecycle events. Events include local
duration, bounded counts and bytes, page/cache/stale/cancellation/dispatch
state, and core run/trace/call correlation. They intentionally exclude server
metadata, requests, results, schemas, frames, endpoints, credentials, and
server error text. Observer failures are suppressed and exposed through
`McpCatalogManager::observer_health`. A refresh buffers its successful
Negotiation event and terminal Refresh event in that order, then delivers them
only after refresh ownership, lifecycle locks, and dispatch permits have been
released. Observer callbacks must still be nonblocking because delivery remains
synchronous for the caller receiving the outcome.

Calls to one canonical server ID share a process-global `ServerCallGate`, so
serialization and queue admission apply across manager instances. The first
live manager for a server establishes its finite, nonzero
`McpLimits::max_server_waiters` bound. Additional live managers for that server
must configure the same value and are rejected with
`McpError::InvalidConfiguration` on a mismatch; they cannot weaken or expand
the established bound. After all managers for that server are dropped, the
unused gate may be pruned and a later manager may establish a new bound.
