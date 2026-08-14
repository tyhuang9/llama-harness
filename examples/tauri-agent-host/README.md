# Embedded Tauri agent host

Use `EmbeddedAgentHost` while constructing an in-process `AgentRunner`. The
model provider, tools, policy, and any database path stay in Rust. The webview
only receives structured events and opaque approval IDs; it never obtains tool
capabilities or raw traces.

`EmbeddedAgentHost` targets only the `main` window. Do not reuse its approval
or run-event channel for widgets, previews, or auxiliary windows.
