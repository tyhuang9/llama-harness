# Tauri integration

Use `llama-harness` inside the Rust backend of a Tauri application. The
embedded `AgentRunner` remains the only execution engine: providers, tools,
policy, approvals, and trace storage are owned by Rust. The webview is an
untrusted presentation surface, not a tool host.

Enable the helper feature in a consuming application:

```toml
llama-harness = { version = "0.1", features = ["ollama", "observability", "tauri"] }
```

`llama_harness::tauri` provides `TauriEventSink` for structured run events,
`ApprovalRouter` for one-time approval IDs, `RunRegistry` for cooperative
cancellation, `FanoutEventSink` for event forwarding plus local SQLite, and
`trace_database_path` for a contained `.sqlite` filename beneath the app data
directory. See `examples/tauri-agent-host` for a minimal composition helper.

Implement narrow Tauri commands in the application: one to cancel a known run,
and one to return `{ approvalId, granted, reason }`. Do not expose arbitrary
tool invocation, raw trace reads, database paths, or model endpoint selection
to the frontend. The frontend must treat each approval ID as opaque and cannot
reuse it.

On window close or application shutdown, cancel each registered run and call
`ApprovalRouter::cancel_run`/`cancel_all`; cancellation is cooperative and does
not undo a tool side effect already begun. SQLite is optional and retains its
existing redaction policy; never persist chain-of-thought or make raw payloads
the default.
