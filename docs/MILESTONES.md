# Milestones

## Embedded runtime rework

### Completed

- Embedded core runtime: bounded agent loop, model abstraction, tool schema validation, policy/approval hooks, cancellation, limits, and causal event records.
- Direct local Ollama provider: loopback-only URL validation, health, model inventory, chat, bounded streaming, and tool calls.
- Local observability: redacted SQLite event store, read-only inspection, query/export/retention support, and WAL-backed writes for application-owned stores.
- Evaluation and CLI foundation: strict YAML/JSON suites, deterministic assertions, report/replay artifacts, and local CLI validation/inspection.
- Local task-agent example: mock-first task tools, explicit approval, trace persistence, evaluation suite, and opt-in real Ollama smoke test.
- Developer console: optional Tauri/React console with real project-path validation, read-only trace inspection, direct loopback Ollama discovery, evaluation artifact inspection, constrained CLI launch previews, accessible empty/error/loading states, and no seeded data.
- Project agent catalog and Promptfoo evaluation: validated project-owned manifests, a development-only concrete local-task-agent adapter, redacted trace-linked normalized reports, and fixed generated artifact inspection.
- Legacy retirement: removed the daemon-backed server, HTTP/SSE TypeScript client, LiteLLM scripts, legacy JSON configuration, and obsolete development commands.

### Migration work remaining

- Add installer/signing and cross-platform packaging verification for the developer console.

Historical daemon/dashboard milestones are preserved in Git history. They are not current architecture guidance.
