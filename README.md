<div align="center">

<img src="https://tyhuang9.github.io/llama-harness/assets/llama-harness-logo.png" alt="llama-harness mascot: a white llama wearing a burgundy harness" width="168" />

# llama-harness

**Run local. Connect any model.**

A Rust library for agent workflows that stay under your application's control.

[Documentation](https://tyhuang9.github.io/llama-harness/) ·
[Rust API](https://docs.rs/llama-harness) ·
[Examples](examples/local-task-agent) ·
[License](LICENSE)

</div>

`llama-harness` runs the model-and-tool loop inside your Rust application. It
handles validation, limits, cancellation, approvals, and ordered events while
your application keeps ownership of its tools, data, policy, and UI.

## What you get

- One bounded `AgentRunner` for model calls and tool execution.
- Application-defined providers, tools, policy, approvals, and event sinks.
- Fail-closed defaults for state-changing tools and callback failures.
- Optional Ollama, SQLite observability, evaluation, and Tauri integrations.

## Install

The supported facade has no default features. Enable only what your application
uses:

```toml
[dependencies]
llama-harness = { version = "0.1.0", features = ["ollama"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

If crates.io indexing is still in progress, pin the reviewed `0.1.0` source:

```toml
llama-harness = { git = "https://github.com/tyhuang9/llama-harness", rev = "d9f7a84a579a36cd1987c5eeeb30764be70aa8ce", features = ["ollama"] }
```

Minimum supported Rust version: **1.88**.

| Feature | Adds |
| --- | --- |
| `ollama` | A loopback-only provider for an existing local Ollama service |
| `observability` | Redacted, local SQLite run and event storage |
| `evals` | Deterministic evaluation and replay contracts |
| `tauri` | Tauri event, approval, cancellation, and trace helpers |

## Quick start with Ollama

Ollama and the selected model must already be installed and running locally.

```rust
use std::sync::Arc;

use llama_harness::{
    ollama::OllamaProvider, AgentDefinition, AgentRunner, RunRequest,
};

#[tokio::main]
async fn main() -> Result<(), llama_harness::HarnessError> {
    let provider = Arc::new(OllamaProvider::new()?);
    let runner = AgentRunner::builder(provider).build();
    let agent = AgentDefinition::new("assistant", "Assistant", "1", "qwen3:8b");

    let result = runner
        .run(RunRequest::new(agent, "Explain why the sky is blue."))
        .await?;

    println!("{}", result.final_output.unwrap_or_default());
    Ok(())
}
```

## Your application stays in charge

The harness does not become your control plane. Your application decides which
tools exist, what data they can access, which calls need approval, how events
are displayed, and when a run should stop. The library supplies the reusable
loop and enforces the boundaries you configure.

## Documentation

- [Start embedding the runner](docs/embedding.md)
- [Define tools, policy, and approvals](docs/tools-and-policies.md)
- [Add local observability](docs/observability.md)
- [Build deterministic evaluations](docs/evaluations.md)
- [Integrate with Tauri](docs/tauri.md)
- [Review the security model](docs/security.md)
- [Understand the architecture](docs/architecture.md)

The complete end-user documentation is published with
[GitHub Pages](https://tyhuang9.github.io/llama-harness/).

## Project scope

The `0.1` release supports Rust applications through the `llama-harness`
facade. TypeScript, Python, and sidecar distribution remain future work and do
not affect the Rust crate.

## Logo and brand

The white llama and burgundy harness are the official project mark. The
[brand board](docs/assets/llama-harness-brand-board.png) records the source
artwork, wordmark treatment, and color palette; the reusable transparent
[mascot asset](docs/assets/llama-harness-logo.png) is sized for documentation
and project surfaces.
