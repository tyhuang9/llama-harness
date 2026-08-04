# Deterministic evaluations and replay

`llama-harness-evals` defines a provider-neutral YAML or JSON evaluation suite and evaluates it through an application-supplied `EvalExecutor`. The executor owns fixture construction, sandboxed application tools, policies, approvals, and the `AgentRunner` instance. This keeps evaluations on the same runtime path as an application while avoiding a generic shell, filesystem, or database tool.

## Suite format

Suites use version `1`, a stable suite and agent ID, one or more model IDs, defaults, and explicit cases. Each case can provide an isolated fixture, input, history/context, agent or prompt-version/override, tags, and deterministic expectations.

```yaml
version: 1
id: task-agent-core
name: Task Agent Core Regression Suite
agent: task-agent
agent_version: "2"
models: [ollama:qwen3:8b]
defaults:
  repeat: 1
  max_latency_ms: 10000
cases:
  - id: explicit-completion
    fixture:
      id: medication-incomplete
      data: {tasks: [{id: task-123, status: incomplete}]}
    input: Complete the evening medication task.
    expected:
      status: completed
      required_tools: [update_task]
      forbidden_tools: [create_task]
      expected_tool_arguments:
        - tool_id: update_task
          arguments_subset: {id: task-123, status: completed}
      final_state_subset:
        tasks: [{id: task-123, status: completed}]
      max_model_calls: 3
      max_tool_calls: 2
```

Supported deterministic assertions include terminal status, exact or contained final output, a JSON-output subset, required/forbidden tools, tool sequence and argument subsets, final-state and unresolved-item subsets, approved tools, maximum model/tool calls, latency, cancellation, and expected error metadata. Array subsets are order-independent. Model-graded assertions are intentionally not part of this first implementation.

## Running a suite

Validate a local suite without needing Ollama, a GPU, or an application fixture adapter:

```bash
cargo run -p llama-harness-cli -- eval validate path/to/suite.yaml
```

An embedding app executes its real controlled loop by implementing `EvalExecutor` and calling:

```rust
let report = llama_harness_evals::evaluate_suite(
    &suite,
    &application_executor,
    &model_overrides,
    repeat_override,
).await?;
```

Each call receives an owned fixture clone. Applications should create fresh sandbox state from it, register only the tools allowed for the case, run the real `AgentRunner`, return a final state snapshot, and clean up their sandbox. The standalone `llama-harness eval run` command validates input but deliberately returns a clear error until an embedding application or example adapter is configured; it never fabricates application-owned tools.

Reports are serializable normalized JSON. Browse a saved report locally with:

```bash
llama-harness eval results .llama-harness/results/report.json
llama-harness eval results .llama-harness/results/report.json --failed --case explicit-completion
```

## Regression replay

`export_regression_case` creates a self-contained regression artifact from an explicit suite case, model selection, agent/prompt metadata, and the linked trace ID. The trace ID is evidence only. Replay calls `replay_regression` through the same application executor; it does not reconstruct fixture data, user input, raw requests, or hidden reasoning from a SQLite trace.

The CLI can inspect a saved artifact with `llama-harness replay regression.json --json`. Executing it requires the embedding application's adapter for the same reason that `eval run` does.

## Trace and model inspection

```bash
llama-harness inspect run <run-id> --db traces.sqlite
llama-harness inspect run <run-id> --db traces.sqlite --export-json
llama-harness models health
llama-harness models list
```

Trace inspection reads the optional local SQLite store. Exports contain only persisted, redacted values; raw payloads remain unavailable when trace storage had its default raw-data setting. No evaluation artifact or report collects hidden chain-of-thought.

## Promptfoo

Promptfoo generation and execution are deliberately deferred. A future wrapper will consume the same validated suite and invoke the application executor, then write visible generated files under `.llama-harness/`. Promptfoo is not a dependency of the core runtime or this deterministic test path.
