# Migration from the legacy service

The existing Axum server remains isolated and independently testable during this migration. The embedded runtime and developer console do not start, call, or depend on it.

| Existing service responsibility | Migration status |
| --- | --- |
| Axum routes, SSE, bearer tokens, pairing, and local admin UI | Retain only as legacy transport. The daemon-backed admin UI and its desktop wrapper were removed; the replacement console has no HTTP daemon dependency. A later adapter may call the embedded core. |
| `server/src/ollama.rs` direct HTTP client | Retain as legacy-server transport. New embedded callers use `llama-harness-ollama`, a direct loopback `ModelProvider`. |
| LiteLLM sidecar/runtime management | Retain now. It is transport/runtime orchestration, not core agent-loop behavior. |
| `server/src/app_policy.rs` catalog and app authorization | Retain now; map resolved agent/tool policy into `AgentDefinition`, `PolicyEngine`, and `ApprovalHandler` later. |
| App-submitted tool continuation | Deprecate for embedded callers over time; embedded applications execute their own `Tool` implementations directly. |
| JSON configuration and JSONL run/audit persistence | Retain as legacy host persistence. The core deliberately has no persistence, while the optional `llama-harness-observability` crate persists redacted structured events in project-local SQLite. |

Adopters can start with a scripted mock and an application-owned read-only tool, then select the direct Ollama provider and attach the SQLite event sink. The optional `apps/harness-console` can inspect a selected project’s read-only trace database, saved evaluation reports, and loopback Ollama inventory. It displays redacted structured events only and constrains Harness CLI actions to project-relative eval/replay commands. Legacy-server adapters and UI approval surfaces remain separate migration work; the embedded runtime does not depend on them.
