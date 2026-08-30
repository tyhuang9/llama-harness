# Adaptive tool calling

`AgentRunner::run` uses `RunStrategy::Adaptive`. Hosts can use
`run_with_strategy` to force `Direct` or `DeclarativePlan` for evaluation and
debugging. `Programmatic` is reserved for the separately sandboxed runtime and
fails closed in core.

Adaptive planning is capability gated. A provider must advertise structured
plan support and sufficient plan, catalog, and model-call limits. Otherwise the
run emits fallback metadata and uses the existing direct reactive loop. A
forced declarative run never silently changes its selected strategy.

## Broker safety boundary

Every concrete invocation, including a declarative-plan node, passes through
the same core broker. The broker enforces total and repeated-call limits, caller
permissions, the agent allowlist, argument byte and depth limits, input and
output schemas, policy, approval, cancellation, deadlines, result limits, and
transcript-safe recording.

A plan is structurally validated and preflighted before a tool is dispatched.
Result-bound nodes defer signature-dependent checks until dependency outputs
have produced their final arguments. The broker then computes the canonical
signature and performs schema validation, repeat accounting, effect-ledger
lookup, policy, and any explicit or policy-required approval against exactly
the arguments that will be executed. Approval of placeholder arguments never
authorizes resolved arguments.

State-changing calls have an effect-ledger state independent of success:

- `Dispatched` is recorded immediately before invocation.
- `Completed` contains the validated successful result for the exact canonical
  tool-and-arguments signature.
- `Uncertain` records any failure after dispatch, including a failure result,
  invalid output, timeout, cancellation, or result resource-limit failure.

Only `Completed` results may be reused. `Dispatched` and `Uncertain` entries
prohibit implicit replay. Direct execution uses the same ledger boundary but
does not enable automatic reuse.

## Scheduling and recovery

The scheduler executes deterministic dependency waves with a hard parallelism
cap of eight, intersected with provider limits. A node is parallel eligible
only when its tool is both read-only and parallel-safe. Mutations, serial nodes,
approval barriers, commit boundaries, and nodes sharing a concurrency key are
serialized.

Invalid planner output receives one repair attempt. Adaptive execution can fall
back to Direct only before tool execution begins. Once execution begins, one
separate recovery plan may be requested using the deterministically ordered set
of recorded completed-node results, and only when the effect ledger proves
recovery safe. Cancellation, deadlines, resource limits, and transcript limits
are terminal and never trigger recovery. Exact completed mutations are reused;
uncertain mutations stop the run.

Planner responses currently use a strict core-validated JSON envelope. Native
provider structured-response constraints can be added as a capability-specific
optimization later; core validation remains authoritative.

Strategy, plan-node, fallback, timing, and usage events contain metadata only.
Provider raw deltas are not persisted by this execution path, and runtimes may
continue filtering unknown additive core events when projecting older protocol
versions.
