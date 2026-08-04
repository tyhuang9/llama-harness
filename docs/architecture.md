# Embedded core architecture

`llama-harness-core` is a Rust library embedded by the application that owns a run. It is not a daemon, HTTP API, or UI. The application supplies a `ModelProvider`, its own `Tool` implementations, a policy engine, an approval callback, and (optionally) an event sink such as the local SQLite store in `llama-harness-observability`.

The controlled `AgentRunner` validates an `AgentDefinition` and `RunRequest`, requests a model response, validates every requested tool call before execution, applies policy and approval, then returns tool results to the model. It stops on a final response, configured limits, cancellation, timeout, or an error. Tools are never retried implicitly.

The core library keeps only the data needed while a run is active. Persistence is deliberately an optional dependency: `llama-harness-observability` provides a local, redacted SQLite `EventSink`, while hosts decide whether they want to attach it. Raw payloads are off by default and must be explicitly enabled.

`ModelProvider` is provider-neutral: it exposes health, inventory, capabilities, and bounded cancellable completion. The scripted mock in `llama_harness_core::mock` keeps tests independent of a network, GPU, or Ollama installation. `llama-harness-ollama` supplies the direct loopback Ollama provider, defaulting to `http://127.0.0.1:11434`; it does not require LiteLLM or any Harness daemon.

Tool argument schemas use JSON Schema validation. `ToolDefinition` also declares risk, idempotency, and read-only metadata, but applications remain responsible for the actual side effects and authorization behind a tool.
