# llama-harness-observability

`llama-harness-observability` provides redacted local SQLite trace persistence
for [`llama-harness`](https://crates.io/crates/llama-harness). It is designed
for application-owned event storage and applies redaction before persistence.

```toml
[dependencies]
llama-harness = { version = "0.1.0", features = ["observability"] }
```

The integration is available as `llama_harness::observability` through the
facade, or as `llama-harness-observability` for lower-level consumers. Hosts
remain responsible for database access controls, retention, and deletion.

Tool discovery events persist only bounded scalar statistics and stable
selection/outcome enums. The additive event fields deserialize with conservative
defaults when reading traces written by earlier releases; no query, tool
identifier, schema, fingerprint, or discovery-cache state is persisted.

The minimum supported Rust version is 1.88. This crate is licensed under the
MIT License; see `LICENSE`.
