# llama-harness-tauri

Embedded helpers for forwarding canonical llama-harness events to a Tauri
frontend, routing user approval decisions, registering cancellable runs, and
deriving a safe per-application trace file path. This crate does not use the
sidecar runtime or start a listener.

```toml
[dependencies]
llama-harness = { version = "0.1.0", features = ["tauri"] }
```

The integration is available as `llama_harness::tauri` through the facade, or
as `llama-harness-tauri` for lower-level consumers. Applications remain
responsible for command authorization and for exposing only the events and
paths their frontend needs.

The minimum supported Rust version is 1.88. This crate is licensed under the
MIT License; see `LICENSE`.
