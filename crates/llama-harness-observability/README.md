# llama-harness-observability

`llama-harness-observability` provides redacted local SQLite trace persistence
for [`llama-harness`](https://crates.io/crates/llama-harness). It is designed
for application-owned event storage and applies redaction before persistence.

```toml
[dependencies]
llama-harness = { version = "0.2.0", features = ["observability"] }
```

The integration is available as `llama_harness::observability` through the
facade, or as `llama-harness-observability` for lower-level consumers. Hosts
remain responsible for database access controls, retention, and deletion.

Tool discovery events persist only bounded scalar statistics and stable
selection/outcome enums. The additive event fields deserialize with a stable
`legacy_unclassified` category when reading traces written by earlier releases; no query, tool
identifier, schema, fingerprint, or discovery-cache state is persisted.

SQLite event identity and ordering are `(execution_id, sequence)`. Repeating an
identical event is idempotent, while a conflicting sequence error identifies the
execution ID as well as the application-visible run ID.

Guarded speculation deliberately does not extend the canonical event or result
contracts. Candidate arguments, results, raw errors, readiness streaks, modes,
and counters are never persisted by this crate. Trusted hosts may inspect the
runner's pull-only readiness and metrics APIs separately, but should not place
those privacy-sensitive diagnostics in raw trace payloads.

The minimum supported Rust version is 1.88. This crate is licensed under the
MIT License; see `LICENSE`.
