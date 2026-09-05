# llama-harness-evals

`llama-harness-evals` provides deterministic evaluation and replay contracts
for [`llama-harness`](https://crates.io/crates/llama-harness). It supports
repeatable suites, assertions, execution, and result reporting without changing
the application-owned runtime boundary.

```toml
[dependencies]
llama-harness = { version = "0.2.0", features = ["evals"] }
```

The integration is available as `llama_harness::evals` through the facade, or
as `llama-harness-evals` for lower-level consumers.

Guarded-speculation evaluations keep forced mode selection in application-owned
fixture data. A trusted executor can read pull-only runner readiness and metrics
to compare Disabled, Shadow, exact Active commit, and safe fallback while the
normalized report continues to rank correctness and zero unintended effects as
hard gates. Evaluation does not add speculative fields to canonical run events
or reconstruct candidates from traces.

The minimum supported Rust version is 1.88. This crate is licensed under the
MIT License; see `LICENSE`.
