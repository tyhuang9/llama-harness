# llama-harness

`llama-harness` is the supported Rust entry point for embedding a bounded,
application-owned agent loop. Your application owns its tools, data, policy,
approvals, and UI; the harness owns model/tool sequencing, schema validation,
limits, cancellation, and ordered events.

The minimum supported Rust version is 1.88.

## Install

The base crate is provider-neutral and has no default features:

```toml
[dependencies]
llama-harness = "0.2.0"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Enable only the integrations your application uses:

```toml
llama-harness = { version = "0.2.0", features = ["ollama", "observability"] }
```

| Feature | Public module | Purpose |
| --- | --- | --- |
| `ollama` | `llama_harness::ollama` | Direct loopback-only Ollama provider |
| `observability` | `llama_harness::observability` | Redacted local SQLite event storage |
| `evals` | `llama_harness::evals` | Deterministic evaluation and replay contracts |
| `tauri` | `llama_harness::tauri` | Tauri event, approval, cancellation, and path helpers |
| `programmatic` | `llama_harness::programmatic` | Explicitly opted-in deterministic program sandbox contracts |
| `mcp` | `llama_harness::mcp` | Transport-neutral MCP tool catalog adapter |

## Run with Ollama

Ollama must already be running and the requested model must already be
installed. The provider does not pull models and rejects non-loopback URLs.

```rust,no_run
use std::sync::Arc;
use llama_harness::{
    ollama::OllamaProvider, AgentDefinition, AgentRunner, RunRequest,
};

#[tokio::main]
async fn main() -> Result<(), llama_harness::HarnessError> {
    let provider = Arc::new(OllamaProvider::new()?);
    let runner = AgentRunner::builder(provider).build();
    let agent = AgentDefinition::new("assistant", "Assistant", "1", "qwen3:8b");
    let result = runner.run(RunRequest::new(agent, "Say hello")).await?;
    println!("{}", result.final_output.unwrap_or_default());
    Ok(())
}
```

## Application-owned providers and tools

The facade re-exports the async-trait macro, cancellation token, and JSON types
needed to implement the extension contracts without depending directly on
`llama-harness-core`.

```rust,no_run
use llama_harness::{
    async_trait, serde_json::json, CancellationToken, HarnessError, JsonValue,
    ModelCapabilities, ModelInfo, ModelProvider, ModelRequest, ModelResponse,
    ProviderHealth, Tool, ToolDefinition, ToolResult, ToolRisk,
};

struct Provider;

#[async_trait]
impl ModelProvider for Provider {
    fn id(&self) -> &str { "application-provider" }
    fn capabilities(&self) -> ModelCapabilities { ModelCapabilities::default() }
    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth::healthy())
    }
    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(vec![ModelInfo::new("application-model")])
    }
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError> {
        Ok(ModelResponse::new(request.model).with_final_output("done"))
    }
}

struct ReadStatus(ToolDefinition);

#[async_trait]
impl Tool for ReadStatus {
    fn definition(&self) -> &ToolDefinition { &self.0 }
    async fn execute(
        &self,
        _arguments: JsonValue,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        if cancellation.is_cancelled() { return Err(HarnessError::Cancelled); }
        Ok(ToolResult::success(json!({"status": "ready"})))
    }
}

let _tool = ReadStatus(
    ToolDefinition::new(
        "status.read",
        "Read status",
        "Read application status",
        json!({"type":"object","additionalProperties":false}),
    )
    .with_risk(ToolRisk::Low)
    .with_idempotent(true)
    .with_read_only(true),
);
```

## Runtime behavior and safety

- `AgentRunner::run` returns `Err` when a run cannot be started or the runner
  itself fails. A started run returns a `RunResult` with its terminal status and
  recorded errors.
- The default policy allows declared read-only tools and denies state-changing
  tools. Applications must deliberately supply mutation policy and approval.
- Cancellation and deadlines prevent future harness work but cannot undo an
  external side effect that a tool has already started. Mutation tools should
  be idempotent or use application-level idempotency keys.
- `programmatic` is disabled by default. It requires the facade feature, an
  explicit `ProgrammaticHostConfig` on the runner, and a conforming provider;
  `Adaptive` never selects it. The sandbox has no ambient authority, but it is
  same-process code and every yielded tool request still goes through the core
  broker, policy, approval, and effect-ledger gates.
- The harness exposes no universal shell, filesystem, database, or network tool.

See the [embedding guide](https://github.com/tyhuang9/llama-harness/blob/main/docs/embedding.md)
and [security guide](https://github.com/tyhuang9/llama-harness/blob/main/docs/security.md)
for complete integration guidance.

This crate is licensed under the MIT License; see `LICENSE`.
