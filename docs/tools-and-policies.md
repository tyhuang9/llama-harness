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

The runner exposes the complete caller-compatible catalog when it fits. Otherwise it keeps hot tools and selects a bounded deferred subset by exact ID, name, namespace, alias, then deterministic lexical relevance. Lexical ranking uses standard BM25 IDF `ln(1 + (N - df + 0.5) / (df + 0.5))` over only the allowlisted, caller-compatible corpus, with `k1 = 1.2` and `b = 0.75`. Scores use nine decimal fixed-point digits. The integer logarithm range-reduces its positive rational input to `[1, 2)`, evaluates 20 terms of the `atanh` series (whose omitted tail is below `2e-19` on that interval), and rounds every integer division half up; arithmetic saturates only at integer-capacity boundaries. Term frequency saturates, document length is normalized, and equal integer scores use tool ID as a stable tie-breaker. Document frequency, total document length, length normalization, term-frequency saturation, and immutable term postings are computed once when the caller-scoped index is built. A query computes IDF once per distinct bounded query term, combines only matching postings, and maintains only the top `K` candidates needed for confidence and expansion rather than sorting every positive match. With vocabulary size `V`, at most `Q = 64` query terms, `M` matching postings, `D` documents, and bounded retained `K`, query work is `O(Q log V + M + D * K)` after index construction; the logarithm series runs at most once per distinct query term found in the scoped vocabulary. The selected `ToolScope` is immutable for the run and is enforced again by the core broker, so provider-supplied IDs cannot bypass discovery, allowlists, caller permissions, schemas, policy, approvals, cancellation, or limits.

Only tool ID/name, namespace, and aliases enter the bounded metadata index. Descriptions, schemas, prompts, provider data, and request data are never indexed or cached as queries/results. Each lazy index entry is keyed by caller plus canonical authorized IDs and their immutable registration versions, so another caller cannot warm it, an unrelated registration cannot make it cold, and unallowlisted tools are not traversed during its construction. Successful registration changes the key only for scopes whose authorized catalog actually changes; rejected registration changes neither scope-cache nor catalog-fingerprint state. Exact immutable serialized definition and standard function-tool fragments are created once at successful registration. A bounded, version-keyed prepared-catalog cache assembles those fragments once per ordered selected definition set and attaches the same immutable `Arc` to every `ModelRequest` that reuses that scope; providers such as Ollama embed its validated raw JSON directly, while the legacy `tools` vector remains byte-compatible. The lexical-index cache retains at most 16 entries and a conservative 16 MiB of owned keys, terms, metadata, and index structures. The prepared-catalog cache retains at most 32 entries and a conservative 8 MiB including keys, cloned definitions and their JSON values, serialized definition arrays, and provider fragments. Both use byte-weighted LRU eviction and retain full canonical ID/version keys rather than hash-only identities. An individual entry above its cache ceiling is returned for the current run but not cached. Cache build, byte, and eviction counters are test-internal and never expose identifiers, fingerprints, queries, or cache timing in public events. Concurrent identical cold misses may duplicate build CPU; single-flight coordination is intentionally deferred because independent waiter cancellation and deadlines require a more complex lifecycle. Publication is rechecked under the cache write lock, so at most one duplicate result is retained and count/byte ceilings still hold.

Discovered scopes use canonical tool-ID order independent of allowlist permutation, while full-fit catalogs preserve legacy allowlist order. Every completed caller-scope selection emits exactly one `ToolDiscoveryCompleted` event, including empty catalogs, provider no-capacity, full-fit, hot-only, exact, lexical, no-match, and mandatory-budget outcomes. The event contains only stable outcome/selection enums; scope-local candidate, selected, deferred, and expansion counts; effective count and byte budgets; the exact serialized definition-array byte count (a size scalar, never content); whether the full catalog exceeded the budget; and selection elapsed milliseconds. It never contains the query, identifiers, names, namespaces, aliases, descriptions, schemas, catalog fingerprint, cache state, raw errors, model output, or cross-caller data. Event absence means validation, preflight, cancellation, or timeout prevented that selection from completing; immutable scope reuse during repair, recovery, fallback, and final synthesis emits no duplicate. Cache state remains internal.

The default host budget is 64 tools and 128 KiB of exactly serialized tool-definition array data; use `AgentRunner::builder(...).discovery_limits(...)` to lower it. Effective count and byte budgets are the minimum of host and provider limits. Explicit provider capacity of zero follows the normal no-tool model path. When mandatory hot or exact matches exceed a nonzero count or byte budget, discovery completes with a value-free limit event and the validated run returns `RunStatus::LimitReached` with zero strategy/model/tool/approval/effect usage, followed by the normal usage and completion events. Invalid requests and internal discovery failures remain errors; cancellation or timeout before completed selection uses its existing terminal path without a discovery-completed event.

`ToolRegistry::register` preserves the base API's legacy Hot behavior: any nonempty trimmed ID and unrestricted display name remain valid, and the bounded tokenizer limits work if a mixed catalog later requires discovery. Explicit `register_with_discovery` registrations use the stricter indexed contract: tool IDs are canonical lowercase ASCII with at most 256 bytes, names are canonical printable ASCII with at most 256 bytes, namespaces and aliases are stable lowercase ASCII identifiers with at most 128 bytes each, normalized aliases must be unique, and each tool can have at most 32 aliases and 4 KiB of aggregate serialized safe discovery metadata. Deferred entries additionally admit at most 256 lexical term occurrences and 1,024 lexical bytes across ID, name, namespace, and aliases. Oversized Deferred metadata is rejected before registry or cache state changes. Hot entries retain legacy registration compatibility; if a mixed oversized catalog later needs an index, its lexical contribution is deterministically truncated to the same per-entry limits without changing the full-fit provider request. Invalid registration leaves the previous catalog generation, fingerprint, serialized fragments, and warmed scope caches unchanged.

Discovery evaluation should record cold and warm catalog selection latency without using wall-clock gates in unit tests. Traversals, exact-match scans, cache-key assembly, ranking phases, and selected-scope materialization have bounded cancellation/deadline checkpoints. The standard-library sorts used for stable cache keys and canonical selected order remain low-risk noninterruptible atomic steps between checkpoints; their inputs are authorized and bounded. The current release target is cold P95 below 50 ms and warm P95 below 5 ms for a 1,000-tool local catalog on supported release hardware; correctness, deterministic selection, and zero unintended effects remain hard gates.

Run the ignored, non-gating release evaluation with:

```bash
cargo test -p llama-harness-core --release --all-features --locked discovery_release_microbenchmark -- --ignored --nocapture
```

It reports cold and warm median/P95 latency plus target status for 30, 100, and 1,000 normal tools and a 1,000-tool, 64-term adversarial query. Record the command output and machine/Rust details with evaluation results; do not turn those wall-clock targets into CI assertions.

The runner validates model-proposed arguments against the registered schema before policy is evaluated or a tool executes. Unknown, disallowed, malformed, and invalid-schema calls become structured rejection events rather than application side effects. Rejected malformed calls do not enter the canonical `RunResult` tool-call transcript: retaining only a canonical parsed argument record avoids representing unparseable model text as an executed or replayable call.

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
