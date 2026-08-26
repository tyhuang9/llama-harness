# llama-harness-evals

`llama-harness-evals` provides deterministic evaluation and replay contracts
for [`llama-harness`](https://crates.io/crates/llama-harness). It supports
repeatable suites, assertions, execution, and result reporting without changing
the application-owned runtime boundary.

```toml
[dependencies]
llama-harness = { version = "0.1.0", features = ["evals"] }
```

The integration is available as `llama_harness::evals` through the facade, or
as `llama-harness-evals` for lower-level consumers.

The minimum supported Rust version is 1.88. This crate is licensed under the
MIT License; see `LICENSE`.
