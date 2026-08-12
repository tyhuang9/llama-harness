# llama-harness

`llama-harness` is a Rust-native framework for controlled local agentic
workflows. One canonical `AgentRunner` owns model/tool looping, validation,
policy, approvals, limits, cancellation, and causal events.

Rust and Tauri applications embed that engine directly. Node/TypeScript and
Python applications may use a private `llama-harness-runtime` JSONL child
process that wraps the same engine. It is not a daemon, HTTP listener, hosted
service, control plane, model downloader, or universal shell.

## Packages

- `llama-harness`: intentional public Rust facade with opt-in `ollama`,
  `observability`, `evals`, `protocol`, and `tauri` features.
- `llama-harness-core`: provider-neutral canonical engine.
- `llama-harness-ollama`: direct loopback-only Ollama provider; no model pulls.
- `llama-harness-observability`: optional redacted local SQLite events.
- `llama-harness-evals`: deterministic versioned evaluation contracts.
- `llama-harness-protocol` and `llama-harness-runtime`: versioned child-sidecar
  contract and runtime for non-Rust SDKs.
- `llama-harness-tauri`: embedded event, approval, cancellation, and trace-path
  helpers for a Tauri Rust backend.
- `sdks/typescript` and `sdks/python`: managed child-sidecar SDKs.

The CLI, Promptfoo adapter, scripted test runtime, examples, and developer
console are development/local tools and are not published framework packages.

## Rust quick start

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

The TypeScript and Python SDKs start a process only in `start()`, route host
tools/policy/approval callbacks over correlated JSONL messages, expose ordered
events, and fail closed on missing approval or callback errors. Give them an
explicit `runtimePath`, `LLAMA_HARNESS_RUNTIME_PATH`, or the matching
package-owned platform runtime. They never download a binary or search arbitrary
`PATH` entries.

Tauri hosts should use the embedded Rust facade and optional `tauri` feature;
the frontend receives structured events and opaque one-time approval IDs, never
direct tool capability. See the linked guides below.

## Guides

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

The manual-only release workflow builds supported runtime artifacts, staged
platform SDK packages, checksums, and a machine-readable manifest. Normal CI
does not publish Rust, npm, Python, binary, model, or hosted-service artifacts.
