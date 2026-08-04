# Migration from the legacy service

The existing Axum server remains supported and independently testable during this migration. This branch does not route production traffic through the core library yet.

| Existing service responsibility | Migration status |
| --- | --- |
| Axum routes, SSE, bearer tokens, pairing, and local admin UI | Retain as legacy transport/UI. A later adapter may call the embedded core. |
| `server/src/ollama.rs` direct HTTP client | Retain now; migrate its request/response mapping into a `ModelProvider` implementation later. |
| LiteLLM sidecar/runtime management | Retain now. It is transport/runtime orchestration, not core agent-loop behavior. |
| `server/src/app_policy.rs` catalog and app authorization | Retain now; map resolved agent/tool policy into `AgentDefinition`, `PolicyEngine`, and `ApprovalHandler` later. |
| App-submitted tool continuation | Deprecate for embedded callers over time; embedded applications execute their own `Tool` implementations directly. |
| JSON configuration and JSONL run/audit persistence | Retain as legacy host persistence. The core deliberately has no persistence. |

Adopters can start with a scripted mock and an application-owned read-only tool. Direct Ollama, legacy-server adapters, streaming, persistence, and UI approval surfaces are intentionally not supplied by this branch.
