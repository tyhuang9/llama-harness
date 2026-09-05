# Completing the embedded-runtime migration

The daemon-backed Axum service, HTTP/SSE TypeScript client, LiteLLM sidecar scripts, legacy JSON configuration, seeded admin dashboard, and desktop wrapper have been retired. The maintained architecture is the embedded Rust `AgentRunner`, direct loopback Ollama provider, optional redacted SQLite trace sink, deterministic evaluation contracts, concrete local task-agent example, optional Developer Console, and managed child-sidecar SDKs for Node and Python.

## What changed

| Retired responsibility | Maintained replacement |
| --- | --- |
| HTTP routes, SSE streaming, bearer tokens, pairing, and daemon lifecycle | An application embeds `AgentRunner` and owns its lifecycle, tools, policy, approvals, and UI. Node/Python use a private child sidecar, never a shared service. |
| Server-owned Ollama proxy | `llama-harness-ollama` talks directly to a loopback-only local Ollama instance. |
| LiteLLM runtime and remote-provider configuration | No replacement: this rework is intentionally local-Ollama only. |
| Server catalog/app authorization | A project-owned agent manifest plus application `Tool`, `PolicyEngine`, and `ApprovalHandler` implementations. |
| HTTP tool-continuation requests | Direct application-owned Rust `Tool` implementations or correlated Node/Python child-sidecar callbacks, always validated by the canonical runner. |
| JSONL run/audit persistence | Optional project-local SQLite causal events through `llama-harness-observability`, with raw payloads disabled by default. |
| Admin dashboard / desktop wrapper | Optional `apps/harness-console` Tauri console for a selected local workspace. |

## Data and configuration

There is no automatic data migration. The retired JSON configuration, app catalogs, provider credentials, pairing/bearer tokens, JSONL run history, and audit records are not consumed by the new runtime and must not be copied into the trace database. SQLite traces are new, redacted causal-event evidence; they do not reconstruct old requests, responses, fixture state, or hidden reasoning.

To adopt the new architecture, define a project-owned agent manifest, construct application tools and policy/approval handlers, select an already-installed local Ollama model, and optionally attach a project-local SQLite event sink. Use versioned suites and the Promptfoo adapter only for development evaluation. The Developer Console may then inspect that project’s redacted traces, serialized agent definitions, evaluation artifacts, fixed generated Promptfoo files, and local Ollama inventory.

## 0.1 to 0.2 package migration

Update every first-party Rust dependency constraint to `0.2.0`; the published
Rust crates remain a coordinated eight-crate set. Node applications install
`@llama-harness/sdk@0.2.0` with the reviewed matching platform runtime package.
Python applications install the matching `llama-harness==0.2.0` platform wheel.
Do not mix an SDK from one release with a runtime artifact from another. The
startup hello reports both SDK and runtime identities, and release validation
rejects a mismatch before artifacts can be uploaded.

Protocol v1 remains the wire major. The 0.2 SDKs offer v1.1 and transparently
fall back to v1.0 when the child advertises only that supported minor. A major
version rejection, malformed hello, or post-handshake protocol drift is a
startup/run failure; treat it as an incompatible installed artifact, not as a
signal to retry against a different network endpoint.

## Migration rollback

Keep the previous lockfile or package resolution long enough to restore a
known-good local application build, but do not relabel, overwrite, or replace
published packages. If a 0.2 artifact fails validation or deployment, stop its
dependent uploads, retain the checksums and release manifest, and ship a
coordinated forward version after diagnosis. A registry yank or deprecation can
discourage new resolutions but does not repair existing lockfiles or downloaded
artifacts. Reverting only a runtime binary while retaining a 0.2 SDK is not a
safe rollback unless the validated hello and protocol compatibility contract
still match.
