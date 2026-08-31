# llama-harness-core

`llama-harness-core` is the provider-neutral runtime behind
[`llama-harness`](https://crates.io/crates/llama-harness). It owns bounded model
and tool sequencing, policy and approval hooks, schema validation, cancellation,
resource limits, and ordered run events.

Most applications should depend on the `llama-harness` facade. Depend on this
crate directly only when implementing a lower-level integration.

```toml
[dependencies]
llama-harness-core = "0.1.0"
```

The runtime does not provide universal shell, filesystem, database, or network
tools. Applications own their tools and side effects and should make mutation
tools idempotent where practical.

Large catalogs can mark individual registrations as deferred with
`ToolRegistry::register_with_discovery`. Selection is deterministic and bounded,
and every selected invocation still crosses the same core broker validation,
policy, approval, cancellation, and resource-limit boundary. Existing
`register` calls remain hot and preserve small-catalog behavior.

The minimum supported Rust version is 1.88. This crate is licensed under the
MIT License; see `LICENSE`.
