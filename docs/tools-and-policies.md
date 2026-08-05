# Tools and policies

Llama Harness executes application-owned tools. The runtime does not provide an unrestricted shell, filesystem, database, or application-function tool.

## Define and register a tool

Implement `Tool` in the embedding application and expose a `ToolDefinition` with a stable ID, description, JSON Schema argument contract, risk level, idempotency flag, and read-only flag. Register it in the application's `ToolRegistry` before building its `AgentRunner`.

```rust
let mut tools = ToolRegistry::default();
tools.register(Arc::new(MyApplicationTool::new(state)))?;
let runner = AgentRunner::builder(provider).tools(tools).build();
```

The runner validates model-proposed arguments against the registered schema before policy is evaluated or a tool executes. Unknown, disallowed, malformed, and invalid-schema calls become structured rejection events rather than application side effects.

## Agent allowlists

Each `AgentDefinition` lists `tool_allowlist`. The provider receives only matching registered tool definitions, and the runner rejects any requested tool outside that list.

Project-owned manifests can make definitions inspectable without making them executable on their own:

```yaml
version: 1
agents:
  - id: local-task-agent
    name: Local Task Agent
    version: "1"
    default_model: ollama:qwen3
    tool_allowlist: [list_tasks, update_task]
```

Validate or inspect a manifest with the local CLI:

```bash
llama-harness agents validate llama-harness.agents.yaml
llama-harness agents list llama-harness.agents.yaml
llama-harness agents inspect llama-harness.agents.yaml local-task-agent --json
```

The manifest is developer-visible metadata. Do not put credentials, secret prompt material, or application data in it. Loading a manifest does not register application tools, resolve a provider, or grant permissions.

## Policy and approval

The application supplies a `PolicyEngine` returning `Allow`, `Deny`, or `RequireApproval`, plus an async `ApprovalHandler`. The runtime records those decisions in ordered events. The safe default permits read-only tools and denies state-changing tools unless the embedding application explicitly supplies another policy.

Use approvals for high-risk or state-changing actions. The application controls how approval is presented and may cancel while an approval is pending; the runtime does not assume a UI.

## Proposal and commit

For consequential changes, prefer a two-step application design:

```text
model proposes a change → application validates/previews it → user approves → application commits it
```

The local task-agent example demonstrates the smaller variant: task mutations require its application approval callback, while task listing is allowed as read-only. Real applications should make their commit boundary explicit and use their own authorization and transactional guarantees.
