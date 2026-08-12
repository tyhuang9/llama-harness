# Architecture

`llama-harness-core` is the canonical Rust engine. A Rust or Tauri host embeds it directly; Node and Python hosts start `llama-harness-runtime` only as a private JSONL child process. Neither mode starts a daemon, HTTP listener, or shared control plane.

The host supplies a `ModelProvider`, its own `Tool` implementations, a policy engine, an approval callback, and (optionally) an event sink such as the local SQLite store in `llama-harness-observability`. The sidecar is a thin process boundary around the same `AgentRunner`; it never reimplements model/tool looping or owns application authorization.

The controlled `AgentRunner` validates an `AgentDefinition` and `RunRequest`, requests a model response, validates every requested tool call before execution, applies policy and approval, then returns tool results to the model. It stops on a final response, configured limits, cancellation, timeout, or an error. Tools are never retried implicitly.

The core library keeps only the data needed while a run is active. Persistence is deliberately an optional dependency: `llama-harness-observability` provides a local, redacted SQLite `EventSink`, while hosts decide whether they want to attach it. Raw payloads are off by default and must be explicitly enabled.

`ModelProvider` is provider-neutral: it exposes health, inventory, capabilities, and bounded cancellable completion. The scripted mock in `llama_harness_core::mock` keeps tests independent of a network, GPU, or Ollama installation. `llama-harness-ollama` supplies the direct loopback Ollama provider, defaulting to `http://127.0.0.1:11434`; it does not require LiteLLM or any Harness daemon.

Tool argument schemas use JSON Schema validation. `ToolDefinition` also declares risk, idempotency, and read-only metadata, but applications remain responsible for the actual side effects and authorization behind a tool.

## Runtime choices

| Host | Runtime choice | Tool and approval owner |
| --- | --- | --- |
| Rust | Embedded `AgentRunner` | Rust application |
| Tauri | Embedded `AgentRunner` with optional `llama-harness-tauri` helpers | Rust backend; webview only renders events/returns opaque approvals |
| Node/TypeScript | Private `llama-harness-runtime` child over stdin/stdout JSONL | Node host callback |
| Python | Private `llama-harness-runtime` child over stdin/stdout JSONL | Python host callback |

The child protocol has a mandatory hello, version negotiation, bounded frames,
request/callback correlation, one stdout writer, and cooperative cancellation.
Stderr is diagnostics only. The production child supports the existing
loopback-only Ollama provider; the scripted provider exists only in a
non-publishable test binary.
