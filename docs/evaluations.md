# Deterministic evaluations and replay

The optional `llama_harness::evals` module exposes provider-neutral YAML or JSON
evaluation suites and evaluates them through an application-supplied
`EvalExecutor`. The executor owns fixture construction, sandboxed application
tools, policies, approvals, and the `AgentRunner` instance. This keeps
evaluations on the same runtime path as an application while avoiding a generic
shell, filesystem, or database tool.

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
let report = llama_harness::evals::evaluate_suite(
    &suite,
    &application_executor,
    &model_overrides,
    repeat_override,
).await?;
```

Each call receives an owned fixture clone. Applications should create fresh sandbox state from it, register only the tools allowed for the case, run the real `AgentRunner`, return a final state snapshot, and clean up their sandbox. The standalone `llama-harness eval run` command validates input but deliberately returns a clear error until an embedding application or example adapter is configured; it never fabricates application-owned tools.

## Forced strategy matrix

Run Programmatic only as an explicitly forced strategy with the facade feature,
host `ProgrammaticHostConfig`, and a provider that advertises strict AST V1
conformance, nonzero program bytes, and at least two model calls. Keep the
existing Direct and Adaptive baselines in the same suite; Adaptive must never
select Programmatic.

| Forced case | Required observation |
| --- | --- |
| Direct baseline | Completes through the existing sequential broker path. |
| Adaptive baseline | Executes through the real `AgentRunner`, records its actual Direct or declarative selection, and never selects Programmatic. |
| Programmatic, conforming provider | Records forced Programmatic selection, bounded lifecycle metadata, broker-audited calls, and a final result. |
| Missing host feature/configuration or false/zero provider capability | Fails closed before a program tool call; it is not a Direct fallback. |
| First invalid program, corrected program valid | Uses at most one pre-dispatch repair and records the repair lifecycle. |
| Invalid program after the one repair, no dispatched effect | Continues the same run through a fresh Direct scope and records `invalid_program`; IDs, event sequence, deadline, budgets, and broker state remain continuous. Do not count it as Programmatic success. |
| Any failure after dispatch, including cancellation, deadline, invalid output, or resource limit | Ends terminally with the effect uncertain; no repair, restart, replay, fallback, or speculation. |
| Read-only parallel-safe fan-out | Preserves source order, validates and reserves the entire bounded worst-case canonical transcript envelope before policy, approval, or dispatch, and respects the effective cap of eight. |
| Mutation in fan-out or mixed read/write batch | Rejects the batch before dispatch; state-changing calls remain serial. |

For every Programmatic case, assert the existing tool-sequence, exact canonical
argument, policy/approval, cancellation, limit, and final-state expectations.
The repository acceptance matrix executes those cases through a fresh real
`AgentRunner` with deterministic provider, tool, policy, and approval fixtures;
it never constructs a `RunResult` or safety metric by hand. Include repair,
pre-dispatch fallback, capability downgrade, and an Adaptive-never-Programmatic
case alongside Direct, declarative-plan, and Programmatic comparisons.
Add privacy canaries that prove lifecycle events, SQLite export, errors, and
debug formatting omit program source, AST, bytecode, constants, locals,
arguments, and tool results. Evaluation artifacts retain only the explicit
fixture, normalized outcome, and permitted trace reference; they never recover
raw program payloads or hidden reasoning.

Sandbox failures use a stable value-free public mapping: sandbox resource limits
become `resource_limit`; resume or deterministic execution failures become
`tool_error`; malformed programs, invalid sandbox limits, and verification
failures become `invalid_output`. Direct and Adaptive retain their existing
broker and error contracts; only a forced, explicitly configured Programmatic
request reaches this mapping. A pre-effect invalid Programmatic generation may
continue through Direct under the same logical run; capability failures do not.

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
llama-harness inspect run --execution-id <execution-id> --db traces.sqlite --export-json
llama-harness models health
llama-harness models list
```

Trace inspection reads the optional local SQLite store. Every event emitter
creates one opaque execution ID, independently of an application-provided run
ID. Listing, ordering, retention, event reads, and exports operate by that
execution ID and sequence; the public run-ID form remains compatible only when
it identifies a single execution, otherwise the CLI asks for `--execution-id`.
Exports contain only persisted, redacted values; raw payloads remain unavailable
when trace storage had its default raw-data setting. No evaluation artifact or
report collects hidden chain-of-thought.

## Promptfoo

The optional, development-only Promptfoo wrapper is documented in [Promptfoo integration](promptfoo-integration.md). It generates a visible custom provider, config, raw output, redacted local trace database, and normalized report under `.llama-harness/`. Promptfoo is not a dependency of the core runtime or deterministic application-evaluation path.
