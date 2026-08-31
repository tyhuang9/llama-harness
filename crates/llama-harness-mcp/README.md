# llama-harness-mcp

Optional, transport-neutral MCP catalog adapter for `llama-harness`.

The crate deliberately does not implement JSON-RPC or select a network client. Hosts supply an
`McpTransport`; validated remote tools are registered as normal deferred core tools.
