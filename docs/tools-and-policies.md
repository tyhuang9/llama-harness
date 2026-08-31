# Tools and policies

Llama Harness executes application-owned tools. The runtime does not provide an unrestricted shell, filesystem, database, or application-function tool.

## Define and register a tool

Implement `Tool` in the embedding application and expose a `ToolDefinition` with a stable ID, description, JSON Schema argument contract, risk level, idempotency flag, and read-only flag. Register it in the application's `ToolRegistry` before building its `AgentRunner`.

```rust
let mut tools = ToolRegistry::default();
tools.register(Arc::new(MyApplicationTool::new(state)))?;
let runner = AgentRunner::builder(provider).tools(tools).build();
```

## Deferred tool discovery

Tools registered with `ToolRegistry::register` remain **hot** for compatibility. A host can opt large-catalog entries into deferred discovery without changing the tool definition or weakening the execution boundary:

```rust
use std::sync::Arc;
use llama_harness::{ToolDiscoveryMetadata, ToolRegistry};

# fn example(tool: Arc<dyn llama_harness::Tool>) -> Result<(), llama_harness::HarnessError> {
let mut tools = ToolRegistry::default();
tools.register_with_discovery(
    tool,
    ToolDiscoveryMetadata::deferred()
        .with_namespace("weather")
        .with_aliases(["forecast", "temperature"]),
)?;
# Ok(())
# }
```

The runner exposes the complete caller-compatible catalog when it fits. Otherwise it keeps hot tools and selects a bounded deferred subset by exact ID, name, namespace, alias, then deterministic lexical relevance. The selected `ToolScope` is immutable for the run and is enforced again by the core broker, so provider-supplied IDs cannot bypass discovery, allowlists, caller permissions, schemas, policy, approvals, cancellation, or limits.

Only tool ID/name, namespace, and aliases enter the cached index. Descriptions, schemas, prompts, provider data, and request data are never indexed or cached as queries/results. Discovery events contain counts and booleans only. The default host budget is 64 tools and 128 KiB of exactly serialized tool-definition array data; use `AgentRunner::builder(...).discovery_limits(...)` to lower it. Effective count and byte budgets are the minimum of host and provider limits. Explicit provider capacity of zero follows the normal no-tool model path, while a nonzero budget too small for mandatory hot or exact matches fails before a model or tool call.

Discovery evaluation should record cold and warm catalog selection latency without using wall-clock gates in unit tests. The current release target is cold P95 below 50 ms and warm P95 below 5 ms for a 1,000-tool local catalog on supported release hardware; correctness, deterministic selection, and zero unintended effects remain hard gates.

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
