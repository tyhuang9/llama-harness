# llama-harness

`llama-harness` is an embedded Rust agent runtime for local applications. It is not a daemon or hosted control plane: the application owns its tools, policies, approvals, state, and lifecycle. Optional local tooling provides direct loopback Ollama access, redacted SQLite traces, deterministic evaluations, a CLI, and a desktop developer console.

## What is in the rework

- `llama-harness-core`: bounded, provider-neutral agent runner with schema-validated tools, policy/approval hooks, cancellation, limits, and structured causal events.
- `llama-harness-ollama`: direct local Ollama provider. It accepts loopback URLs only and defaults to `http://127.0.0.1:11434`.
- `llama-harness-observability`: opt-in local SQLite trace persistence. Structured events are redacted; raw persistence is disabled by default.
- `llama-harness-evals`: versioned YAML/JSON suites, deterministic assertions, report artifacts, and replay contracts.
- `llama-harness-cli`: local inspection and validation commands. It never fabricates an application adapter.
- `examples/local-task-agent`: deterministic, runnable reference integration with explicit task tools and policy approval.
- `apps/harness-console`: optional Tauri/React desktop console for one local project workspace.

The retired daemon-backed service, HTTP/SSE TypeScript client, LiteLLM scripts, JSON configuration, admin dashboard, and desktop wrapper have been removed. The repository now contains only the embedded-runtime architecture and optional local developer tooling.

## Quick start

Run the embedded reference without a daemon, GPU, or network:

```bash
cargo run -p local-task-agent -- --trace-db local-task-agent-traces.sqlite
cargo test --workspace
```

Validate an evaluation suite or inspect redacted persisted traces:

```bash
cargo run -p llama-harness-cli -- eval validate evals/local-task-agent/suite.yaml
cargo run -p llama-harness-cli -- inspect run <run-id> --db local-task-agent-traces.sqlite --export-json
cargo run -p llama-harness-cli -- models list
```

The local task-agent can use a real Ollama model only when explicitly requested:

```powershell
$env:LLAMA_HARNESS_TEST_OLLAMA='1'
cargo test -p local-task-agent real_ollama_task_agent_smoke_is_opt_in -- --exact --nocapture
```

## Developer console

The console is optional and only reads a selected project workspace, an existing SQLite trace file, optional evaluation-report JSON, and a loopback Ollama instance. It does not start or call the legacy HTTP service.

```bash
npm --prefix apps/harness-console install
npm run dev
```

Then select:

- an absolute project root;
- an existing SQLite trace database;
- an optional existing evaluation results file or directory; and
- a loopback Ollama URL such as `http://127.0.0.1:11434`.

The console displays only structured redacted trace events. It can construct and run only project-relative `llama-harness-cli` evaluation and replay commands without invoking a shell. A standalone `eval run` or replay command will report that the embedding application owns the required adapter; that limitation is intentional.

Useful commands:

```bash
npm run console:test
npm run console:build
```

## Architecture and integration

An embedding application supplies a `ModelProvider`, its own `Tool` implementations, `PolicyEngine`, `ApprovalHandler`, and optionally an `EventSink`. The runner validates tool arguments before execution, records causal events, and stops deterministically on final output, cancellation, timeouts, limits, or errors. Tools are never implicitly retried.

Read the detailed guides before embedding:

- [architecture](docs/architecture.md)
- [embedding](docs/embedding.md)
- [tools and policies](docs/tools-and-policies.md)
- [observability](docs/observability.md)
- [evaluations](docs/evaluations.md)
- [Promptfoo integration](docs/promptfoo-integration.md)
- [developer console](docs/developer-console.md)
- [migration map](docs/migration.md)

## Repository structure

```text
llama-harness/
  crates/
    llama-harness-core/
    llama-harness-ollama/
    llama-harness-observability/
    llama-harness-evals/
    llama-harness-cli/
  examples/local-task-agent/
  apps/harness-console/       Optional local Tauri developer console
  evals/local-task-agent/
  tools/promptfoo/            Pinned development-only Promptfoo dependency
  docs/
```

## Verification

```bash
cargo fmt --check --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --manifest-path apps/harness-console/src-tauri/Cargo.toml
npm --prefix apps/harness-console run test
npm --prefix apps/harness-console run build
```

## Current boundaries

- No hosted accounts, multi-user tenancy, cloud sync, model marketplace, or generic remote model gateway is part of the rework.
- Ollama must already be installed and running locally for live model discovery or inference.
- The console stores only its own local workspace preferences. It cannot enable raw trace capture, change retention in an existing database, or retrieve chain-of-thought.
- The retired HTTP/SSE, TypeScript client, LiteLLM, and JSON-configuration workflows no longer exist in this repository. Historical configuration, pairing tokens, JSONL run history, and audit data are not migrated into the redacted SQLite trace store.
