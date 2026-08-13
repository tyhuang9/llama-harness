# Integrating llama-harness into Note

This guide is intentionally repository-agnostic: it does not modify the Note
application. Integrate from Note's Rust/Tauri backend, not its webview.

During pre-release work, pin a reviewed Git revision:

```toml
llama-harness = { git = "https://github.com/tyhuang9/llama-harness", rev = "<reviewed-commit>", features = ["ollama", "observability", "tauri"] }
```

After registry publication, replace that entry with the reviewed crates.io
release and preserve the same explicit feature list. Construct Note-owned Rust
tools for note search, retrieval, and writes; keep their arguments schemas
narrow, mark write tools non-read-only, apply Note's authorization policy, and
route any user confirmation through `ApprovalRouter`.

The Note frontend may render `TauriEventSink` events and respond with opaque
approval IDs, but must not receive provider credentials, arbitrary filesystem
paths, tool capabilities, or a raw trace-export API. Use a fixed app-data trace
filename through `trace_database_path`, attach a redacted `SqliteEventSink`
through `FanoutEventSink` if persistence is needed, and cancel registered runs
when a Note workspace closes. Construct the emitters with `TauriTargetEmitter`
for the `main` window only: widgets, previews, and auxiliary windows must never
receive run or approval payloads. No local HTTP daemon is needed or supported
for this embedded path. See `note-embedding-dependencies.md` for reproducible
dependency evidence and the accepted current networking cost.
