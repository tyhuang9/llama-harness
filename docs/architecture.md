# Embedded core architecture

`llama-harness-core` is a Rust library embedded by the application that owns a run. It is not a daemon, HTTP API, database, or UI. The application supplies a `ModelProvider`, its own `Tool` implementations, a policy engine, an approval callback, and (optionally) an event sink.

The controlled `AgentRunner` validates an `AgentDefinition` and `RunRequest`, requests a model response, validates every requested tool call before execution, applies policy and approval, then returns tool results to the model. It stops on a final response, configured limits, cancellation, timeout, or an error. Tools are never retried implicitly.

The library keeps only the data needed while a run is active. It does not persist prompts, raw model responses, secrets, or tool output. Hosts decide whether and how to persist their own safe summaries.

`ModelProvider` is provider-neutral: it exposes health, inventory, capabilities, and bounded cancellable completion. The first implementation is a scripted mock in `llama_harness_core::mock`; a direct Ollama provider belongs in a later increment. The mock keeps tests independent of a network, GPU, or Ollama installation.

Tool argument schemas use JSON Schema validation. `ToolDefinition` also declares risk, idempotency, and read-only metadata, but applications remain responsible for the actual side effects and authorization behind a tool.
