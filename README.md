# llama-harness

`llama-harness` is a Rust-native framework for controlled local agentic
workflows. One canonical `AgentRunner` owns model/tool looping, validation,
policy, approvals, limits, cancellation, and causal events.

**[Read the API documentation →](https://tyhuang9.github.io/llama-harness/)**

The documentation site is a Stripe-inspired, searchable API reference for the
core runner, tool controls, sidecar SDKs, Tauri integration, and release path.

Rust and Tauri applications embed that engine directly through the
`llama-harness` facade. The Rust 0.1 release is independent of the deferred
Node/TypeScript and Python sidecar distribution. It is not a daemon, HTTP
listener, hosted service, control plane, model downloader, or universal shell.

## Packages

- `llama-harness`: intentional public Rust facade with opt-in `ollama`,
  `observability`, `evals`, and `tauri` features.
- `llama-harness-core`: provider-neutral canonical engine.
- `llama-harness-ollama`: direct loopback-only Ollama provider; no model pulls.
- `llama-harness-observability`: optional redacted local SQLite events.
- `llama-harness-evals`: deterministic versioned evaluation contracts.
- `llama-harness-protocol` and `llama-harness-runtime`: deferred, non-published
  child-sidecar work for future non-Rust SDK distribution.
- `llama-harness-tauri`: embedded event, approval, cancellation, and trace-path
  helpers for a Tauri Rust backend.
- `sdks/typescript` and `sdks/python`: managed child-sidecar SDKs.

The CLI, Promptfoo adapter, scripted test runtime, examples, and developer
console are development/local tools and are not published framework packages.

## Rust quick start

The minimum supported Rust version is 1.88. Default features are empty; enable
only the named integration modules your application needs.

```toml
[dependencies]
llama-harness = { version = "0.1.0", features = ["ollama"] }
```

```rust
use std::sync::Arc;
use llama_harness::{mock::{final_response, MockModelProvider}, AgentDefinition, AgentRunner, RunRequest};

# async fn example() -> Result<(), llama_harness::HarnessError> {
let runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response("done")]))).build();
let result = runner.run(RunRequest::new(AgentDefinition::new("example", "Example", "1", "mock-model"), "Reply with done")).await?;
assert_eq!(result.final_output.as_deref(), Some("done"));
# Ok(()) }
```

For a runnable task-agent reference:

```bash
cargo run -p local-task-agent -- --trace-db local-task-agent-traces.sqlite
cargo run -p llama-harness-cli -- eval validate evals/local-task-agent/suite.yaml
```

Ollama is opt-in, must already run locally, and is limited to loopback URLs.

## SDK and Tauri quick start

The TypeScript and Python SDKs remain workspace prototypes; see their guides for
managed child-sidecar integration details.

Tauri hosts should use the embedded Rust facade and optional `tauri` feature;
the frontend receives structured events and opaque one-time approval IDs, never
direct tool capability. See the linked guides below.

## Guides

- [API documentation](https://tyhuang9.github.io/llama-harness/) (GitHub Pages)

- [Architecture](docs/architecture.md) and [embedding](docs/embedding.md)
- [Protocol compatibility](protocol/compatibility/v1.md) and [SDK architecture](docs/sdk-architecture.md)
- [TypeScript SDK](docs/typescript-sdk.md) and [Python SDK](docs/python-sdk.md)
- [Tauri integration](docs/tauri.md) and [Note integration](docs/integrating-note.md)
- [Tools and policies](docs/tools-and-policies.md), [observability](docs/observability.md), and [security](docs/security.md)
- [Distribution](docs/distribution.md), [releasing](docs/releasing.md), and [migration](docs/migration.md)

## Verification

```bash
cargo fmt --check --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --all-features --no-deps
cargo run -p xtask -- protocol-check
cargo run -p xtask -- release-check

npm --prefix sdks/typescript run test
npm --prefix sdks/typescript run pack:dry-run
python -m build sdks/python
python scripts/inspect_python_packages.py
```

Rust publication is a separate, manual, review-gated operation. Normal CI does
not publish crates, binaries, models, or hosted-service artifacts.
