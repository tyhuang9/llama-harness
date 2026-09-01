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

Each completed caller-scope selection emits one metadata-only discovery event.
Mandatory count or serialized-schema budget overflow becomes a zero-effect
`LimitReached` run outcome; invalid requests and internal discovery errors
remain errors. Queries, tool metadata, fingerprints, and cache state are never
included in discovery events.

Guarded speculation is an explicit, Direct-only optimization for strongly
attested local private reads. It starts per tool in Shadow, requires at least
1,000 exact same-runner observations plus explicit host activation, and keeps
candidate diagnostics behind pull-only runner APIs. An Active candidate still
crosses the broker with a dedicated speculative policy decision and is reused
only at an exact typed commit. Writes, remote or egress-capable tools, and
imported MCP tools remain ineligible. No speculative arguments, results, raw
errors, readiness state, or counters are added to `RunEvent` or `RunResult`.
See the repository's [guarded speculation guide](../../docs/speculative-tool-calling.md).

The minimum supported Rust version is 1.88. This crate is licensed under the
MIT License; see `LICENSE`.
