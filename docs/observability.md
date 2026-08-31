# Local trace observability

Every `AgentRunner` run has a generated internal execution identity, a public `run_id`/`trace_id` correlation pair, monotonically increasing event sequence, and timestamp. The internal identity remains distinct even when an application deliberately reuses a public run ID, so local SQLite persistence cannot collide on sequence numbers. Enable the facade's `observability` feature and attach `SqliteEventSink` from `llama_harness::observability` as the core runner's `EventSink` to persist that causal event stream locally. The sink uses a project-selected SQLite database; it does not start a service or require the developer console to remain open.

## Stored events

The runtime records structured facts such as run start and completion, model requests and responses, tool rejection or completion, policy decisions, approval requests, retries, and terminal status. Events are ordered by `(run_id, sequence)` and inserts are idempotent, so a safe retry of a persistence write does not duplicate an event. A run is constrained to one trace ID to preserve causal queries.

The store supports run lookup, filterable summaries, paged event reads, redacted JSON export, retention by age or recent-run count, explicit run deletion, and SQLite compaction. Export refuses oversized traces instead of silently producing a partial file; callers can inspect large traces through the paged API.

## Redaction and raw data

All structured events are converted to JSON and redacted before they are written. The default redaction covers common sensitive field names including authorization, cookies, passwords, secrets, tokens, and API keys. Applications may supply additional key fragments and literal values through `RedactionConfig`.

Raw request and response payloads are disabled by default. When an application explicitly enables `TraceStoreConfig::persist_raw_payloads`, it must pass the raw value through `append_with_raw` or `append_batch`; that value is redacted and size-bounded before storage. The store never captures or displays hidden model chain-of-thought. Persist only user-visible or application-approved diagnostics.

## Operational notes

SQLite uses WAL mode and a configurable busy timeout. One sink serializes its own writes with a mutex, making cloneable sinks safe to use from concurrent local runs without holding a lock across model calls, tool execution, or approvals. Call `append` when persistence errors must be handled by the caller. `EventSink::emit` has no error return by design, so it records the latest write failure for inspection through `last_emit_error`.

OpenTelemetry is intentionally not included in the initial crate: it remains an optional future export adapter, not a reason to make telemetry infrastructure a runtime dependency.
