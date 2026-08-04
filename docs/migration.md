# Migration from the legacy service

The existing Axum server remains supported and independently testable during this migration. This branch does not route production traffic through the core library yet.

| Existing service responsibility | Migration status |
| --- | --- |
| Axum routes, SSE, bearer tokens, pairing, and local admin UI | Retain as legacy transport/UI. A later adapter may call the embedded core. |
| `server/src/ollama.rs` direct HTTP client | Retain as legacy-server transport. New embedded callers use `llama-harness-ollama`, a direct loopback `ModelProvider`. |
| LiteLLM sidecar/runtime management | Retain now. It is transport/runtime orchestration, not core agent-loop behavior. |
| `server/src/app_policy.rs` catalog and app authorization | Retain now; map resolved agent/tool policy into `AgentDefinition`, `PolicyEngine`, and `ApprovalHandler` later. |
| App-submitted tool continuation | Deprecate for embedded callers over time; embedded applications execute their own `Tool` implementations directly. |
| JSON configuration and JSONL run/audit persistence | Retain as legacy host persistence. The core deliberately has no persistence, while the optional `llama-harness-observability` crate persists redacted structured events in project-local SQLite. |

Adopters can start with a scripted mock and an application-owned read-only tool, then select the direct Ollama provider and attach the SQLite event sink. Legacy-server adapters, a developer console, and UI approval surfaces remain separate migration work; the embedded runtime does not depend on them.
