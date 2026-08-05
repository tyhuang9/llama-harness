# Embedding Llama Harness

Llama Harness runs in the consuming Rust or Tauri application process. The application owns its tools, data, policy, approval callback, and UI; the embedded runtime owns the bounded model/tool loop. Ollama is the optional shared local inference process at `http://127.0.0.1:11434`. No Harness daemon is started or required.

The runnable reference is [`examples/local-task-agent`](../examples/local-task-agent). It keeps an in-memory task store in the application, registers only `list_tasks`, `create_task`, and `update_task`, requires approval for mutations, and writes redacted structured events to a project-selected SQLite database.

Run its deterministic mock path:

```bash
cargo run -p local-task-agent -- --trace-db local-task-agent-traces.sqlite
cargo run -p llama-harness-cli -- inspect run <run-id> --db local-task-agent-traces.sqlite --export-json
```

The example has an optional live Ollama route; it checks health and installed models and never pulls a model:

```bash
cargo run -p local-task-agent -- \
  --provider ollama --model qwen3:8b \
  --trace-db local-task-agent-traces.sqlite
```

## Application setup

```rust
use std::sync::Arc;
use llama_harness_core::{AgentRunner, EventSink, ModelProvider};
use llama_harness_observability::{SqliteEventSink, TraceStoreConfig};

let provider: Arc<dyn ModelProvider> = Arc::new(
    llama_harness_ollama::OllamaProvider::new()?
);
let traces = Arc::new(SqliteEventSink::open("traces.sqlite", TraceStoreConfig::default())?);

// The application builds its own ToolRegistry, PolicyEngine, and ApprovalHandler.
let runner = AgentRunner::builder(provider)
    .tools(application_tools)
    .policy(application_policy)
    .approvals(application_approval_handler)
    .event_sink(traces as Arc<dyn EventSink>)
    .build();
```

Call `runner.run(RunRequest { ... }).await` from an application command or function. A `RunRequest` contains an `AgentDefinition`, user input, application context, history, metadata, optional model override, and cancellation token. Keep secrets out of the agent definition and context unless the host has an explicit redaction/persistence policy for them.

## Safety boundary

Tools are application implementations of the typed `Tool` trait. The runner validates model-proposed JSON against each registered tool schema, checks the agent allowlist, applies the application policy, and invokes approval where required. It never exposes a universal shell, filesystem, or database tool.

Use proposal/commit tools for sensitive mutations when appropriate: an application can let a model generate a proposal, validate and preview it, collect user approval, then expose a narrowly scoped commit tool. Cancellation and timeouts stop future runner work but cannot roll back an external side effect already started by a tool; application tools must remain idempotent or use application-level idempotency keys where needed.

## Deterministic evaluation

Implement `llama_harness_evals::EvalExecutor` in the application to build a fresh fixture sandbox and real runtime for every case. The local-task-agent example provides `TaskAgentEvalExecutor` and `evals/local-task-agent/suite.yaml`; its mock-provider evaluation tests need no network, GPU, Promptfoo, or Ollama.

```bash
cargo test -p local-task-agent
cargo run -p llama-harness-cli -- eval validate evals/local-task-agent/suite.yaml
```

The optional real example smoke test is enabled only when requested:

```bash
LLAMA_HARNESS_TEST_OLLAMA=1 cargo test -p local-task-agent real_ollama_task_agent_smoke_is_opt_in -- --exact --nocapture
```
