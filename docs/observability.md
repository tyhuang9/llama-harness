# Local trace observability

Every `AgentRunner` run has a generated internal execution identity, a public `run_id`/`trace_id` correlation pair, monotonically increasing event sequence, and timestamp. The internal identity remains distinct even when an application deliberately reuses a public run ID, so local SQLite persistence cannot collide on sequence numbers. Enable the facade's `observability` feature and attach `SqliteEventSink` from `llama_harness::observability` as the core runner's `EventSink` to persist that causal event stream locally. The sink uses a project-selected SQLite database; it does not start a service or require the developer console to remain open.

## Stored events

The runtime records structured facts such as run start and completion, model requests and responses, tool rejection or completion, policy decisions, approval requests, retries, and terminal status. Events are ordered and conflict-identified by `(execution_id, sequence)`; SQLite accepts a new event only at the next contiguous sequence while accepting an exact idempotent retry. Conflict errors identify that execution ID unambiguously while retaining the application-visible run ID and sequence for compatibility. A run is constrained to one trace ID to preserve causal queries.

The store supports run lookup, filterable summaries, paged event reads, redacted JSON export, retention by age or recent-run count, explicit run deletion, and SQLite compaction. Export refuses oversized traces instead of silently producing a partial file; callers can inspect large traces through the paged API. Public multi-event producers must pass one explicit execution ID to `EventRecord::new` and use contiguous sequences; this semver-major constructor change prevents each record from silently becoming a separate trace.

## Redaction and raw data

All structured events are converted to JSON and redacted before they are written. The default redaction covers common sensitive field names including authorization, cookies, passwords, secrets, tokens, API keys, and program artifacts. Matching is case-insensitive and token-aware across separators, camel case, and acronyms, so names such as `accessToken`, `openaiApiKey`, and `programAST` are redacted without broad substring matching. Applications may supply additional key fragments and literal values through `RedactionConfig`.

Raw request and response payloads are disabled by default. When an application explicitly enables `TraceStoreConfig::persist_raw_payloads`, it must pass the raw value through `append_with_raw` or `append_batch`; that value is redacted and size-bounded before storage. The store never captures or displays hidden model chain-of-thought. Persist only user-visible or application-approved diagnostics.

## Operational notes

SQLite uses WAL mode and a configurable busy timeout. One sink serializes its own writes with a mutex, making cloneable sinks safe to use from concurrent local runs without holding a lock across model calls, tool execution, or approvals. Call `append` when persistence errors must be handled by the caller. `EventSink::emit` has no error return by design, so it records the latest write failure for inspection through `last_emit_error`.

For operations, read `persistence_health()` rather than polling diagnostic text: `persist_failures_total` is monotonic, the latest failure timestamp/category and last success timestamp are value-free, and `healthy` becomes true after a successful persistence attempt. Hosts can install a value-free `PersistenceFailureHandler` for alert integration; handler panics are contained. The callback runs on the failed emit path, so it should only enqueue into a bounded, nonblocking host queue and must not perform I/O, wait for a consumer, or re-enter the sink. Alert on a nonzero failure-rate increase, sustained unhealthy state, SQLite busy failures, and migration/open failures. Dashboard terminal lifecycle outcomes, admission limits, VM fuel, scheduling slices, tool yields, partial failures, and duration together with SQLite contention and migration outcomes; none of these fields include model, tool argument, result, raw payload, SQL, or secret content.

Age retention deletes complete execution groups only when the group's latest event is older than the cutoff, so it never leaves a partial trace. Incomplete executions are retained intact by age policy until explicit deletion or a `max_runs` group eviction; count retention also deletes whole execution groups.

The current SQLite schema is v3. Opening a v1 database first performs the v1-to-v2 replacement-table migration, then the v2-to-v3 replacement-table execution-ID rewrite; each phase copies the full table, replaces it, and rebuilds indexes inside its own transaction. Plan temporary disk space comparable to the table and indexes, expect a write lock, and rely on SQLite rollback if a migration statement fails. Back up and rehearse a representative large database before upgrade. The read-only developer console and CLI `inspect run` never migrate or create a database: they report that a writer must first open a v1 or v2 database and complete the v3 upgrade.

OpenTelemetry is intentionally not included in the initial crate: it remains an optional future export adapter, not a reason to make telemetry infrastructure a runtime dependency.
