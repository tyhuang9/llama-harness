# Completing the embedded-runtime migration

The daemon-backed Axum service, HTTP/SSE TypeScript client, LiteLLM sidecar scripts, legacy JSON configuration, seeded admin dashboard, and desktop wrapper have been retired. The maintained architecture is the embedded Rust `AgentRunner`, direct loopback Ollama provider, optional redacted SQLite trace sink, deterministic evaluation contracts, concrete local task-agent example, and optional Developer Console.

## What changed

| Retired responsibility | Maintained replacement |
| --- | --- |
| HTTP routes, SSE streaming, bearer tokens, pairing, and daemon lifecycle | An application embeds `AgentRunner` and owns its lifecycle, tools, policy, approvals, and UI. |
| Server-owned Ollama proxy | `llama-harness-ollama` talks directly to a loopback-only local Ollama instance. |
| LiteLLM runtime and remote-provider configuration | No replacement: this rework is intentionally local-Ollama only. |
| Server catalog/app authorization | A project-owned agent manifest plus application `Tool`, `PolicyEngine`, and `ApprovalHandler` implementations. |
| HTTP tool-continuation requests | Direct application-owned `Tool` implementations registered in the embedded process. |
| JSONL run/audit persistence | Optional project-local SQLite causal events through `llama-harness-observability`, with raw payloads disabled by default. |
| Admin dashboard / desktop wrapper | Optional `apps/harness-console` Tauri console for a selected local workspace. |

## Data and configuration

There is no automatic data migration. The retired JSON configuration, app catalogs, provider credentials, pairing/bearer tokens, JSONL run history, and audit records are not consumed by the new runtime and must not be copied into the trace database. SQLite traces are new, redacted causal-event evidence; they do not reconstruct old requests, responses, fixture state, or hidden reasoning.

To adopt the new architecture, define a project-owned agent manifest, construct application tools and policy/approval handlers, select an already-installed local Ollama model, and optionally attach a project-local SQLite event sink. Use versioned suites and the Promptfoo adapter only for development evaluation. The Developer Console may then inspect that project’s redacted traces, serialized agent definitions, evaluation artifacts, fixed generated Promptfoo files, and local Ollama inventory.
