# llama-harness-tauri

Embedded helpers for forwarding canonical llama-harness events to a Tauri
frontend, routing user approval decisions, registering cancellable runs, and
deriving a safe per-application trace file path. This crate does not use the
sidecar runtime or start a listener.
