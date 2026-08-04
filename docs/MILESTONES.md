# Milestones

## Embedded runtime rework

### Completed

- Embedded core runtime: bounded agent loop, model abstraction, tool schema validation, policy/approval hooks, cancellation, limits, and causal event records.
- Direct local Ollama provider: loopback-only URL validation, health, model inventory, chat, bounded streaming, and tool calls.
- Local observability: redacted SQLite event store, read-only inspection, query/export/retention support, and WAL-backed writes for application-owned stores.
- Evaluation and CLI foundation: strict YAML/JSON suites, deterministic assertions, report/replay artifacts, and local CLI validation/inspection.
- Local task-agent example: mock-first task tools, explicit approval, trace persistence, evaluation suite, and opt-in real Ollama smoke test.
- Developer console: optional Tauri/React console with real project-path validation, read-only trace inspection, direct loopback Ollama discovery, evaluation artifact inspection, constrained CLI launch previews, accessible empty/error/loading states, and no seeded data.

### Migration work remaining

- Decide whether the retained legacy server should receive a narrow adapter to the embedded runtime or be removed in a dedicated migration branch.
- Classify or remove the legacy TypeScript client, LiteLLM runtime scripts, and JSON configuration once consumers have migrated.
- Add installer/signing and cross-platform packaging verification for the developer console.

Historical daemon/dashboard milestones are preserved in Git history. They are not current architecture guidance.
