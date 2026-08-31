# Adaptive tool calling

`AgentRunner::run` uses `RunStrategy::Adaptive`. Hosts can use
`run_with_strategy` to force `Direct` or `DeclarativePlan` for evaluation and
debugging. `Programmatic` is separately sandboxed, forced-only, and unavailable
unless the facade/core feature and explicit host configuration are both present.

Adaptive planning is capability gated. A provider must advertise structured
plan support and sufficient plan, catalog, and model-call limits. Otherwise the
run emits fallback metadata and uses the existing direct reactive loop. A
forced declarative run never silently changes its selected strategy.

## Programmatic tool calling

Programmatic execution is an explicit same-process opt-in, not an Adaptive
candidate. Compile the `programmatic` facade feature, configure
`AgentRunnerBuilder::programmatic(ProgrammaticHostConfig { .. })`, and force
`RunStrategy::Programmatic` only in a controlled rollout or evaluation. Omitting
either the feature or builder configuration disables new programmatic runs;
`Adaptive` continues to select only its existing Direct or declarative paths.

The provider must advertise tool support, `StrictJsonAstV1` programmatic
conformance, a nonzero `max_program_bytes` capability, and at least two model
calls. A missing, false, or zero capability fails closed before any program
tool call; it is not silently treated as Direct. The sandbox receives a strict
versioned AST and privately verified bytecode, has no ambient filesystem,
network, process, provider, registry, policy, or tool capability, and only
yields inert owned-data batches back to the host.

The host still owns the complete authority boundary. Every yielded call uses
the same broker as Direct and declarative execution, including frozen caller
and tool scope, canonical arguments, schema validation, policy, approval,
effect ledger, deadlines, cancellation, transcript limits, and audited events.
The effective bound for each resource is the minimum applicable library hard
cap, host sandbox setting, Agent limit, and provider capability. Program runs
require a finite deadline and VM admission; read-only, parallel-safe fan-out is
bounded at eight and additionally limited by the host, provider, Agent, and
broker caps. Mutations are always serialized.

One correction prompt is permitted only before an effect is dispatched. If the
corrected program is still invalid and no effect was issued, the runner starts
a fresh Direct-scope sequential fallback and records the `invalid_program`
fallback metadata; a forced run therefore never claims that Programmatic
succeeded. Cancellation, deadline, resource, invalid-output, tool, resume, or
result failures after dispatch leave the effect uncertain and terminal. They
never repair, restart, replay, speculate, or fall back.

To roll out safely, leave the feature disabled by default, gate the builder
configuration in the host deployment, and force Programmatic only in an
evaluation matrix with conforming providers. Removing that configuration or
feature stops future programmatic runs, but cannot undo an external effect
already dispatched; rely on application-level idempotency and the broker's
uncertain-effect boundary for recovery. Program lifecycle telemetry and SQLite
kind hooks are metadata-only; source, AST, bytecode, constants, local values,
arguments, results, raw errors, and model identifiers are not recorded there.

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

Before copying dependency output into bound arguments, the scheduler accounts
for the template plus every referenced value without cloning them. The copy
budget is the lower of the run's argument limit and the declarative-plan hard
cap. An amplified binding set is rejected before replacement or tool
invocation.

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
serialized. Concurrency keys are also enforced by runner-wide asynchronous
permits, so Direct and declarative calls in different runs on the same runner
cannot overlap a shared external resource. Permit-map locks are never held
while a tool or permit wait is awaited.

Plan outputs have one authoritative `Arc`-shared representation across the
completed-result map, transcript staging, and effect ledger. The scheduler
charges each retained call/result incrementally against the remaining
transcript budget and a 16 MiB hard cap. Before dispatching a wave, it reserves
for the maximum permitted output of every invocation in that wave; if no next
invocation can fit, the run terminates before dispatch.

Before the first mutation, the scheduler prioritizes ready read-only nodes so
bound arguments can resolve, then exactly validates and authorizes every
remaining mutation or explicit approval barrier as one gate. Declarative plans
with a bound node that transitively depends on a mutation are rejected because
their exact downstream authorization cannot be established before the earlier
effect. Adaptive mode may repair or fall back before effects; forced planning
fails closed.

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

Planning, repair, provider retry, and execution recovery are optional phases.
They run only when at least one model call remains reserved for direct fallback
or final synthesis. In particular, a two-call budget can spend one call on the
initial planner and one on final synthesis; it cannot spend the second call on
repair or a provider retry. Forced declarative execution skips optional repair
under the same boundary and returns an explicit failed result for invalid
planner output instead of executing a plan that cannot be finalized.

## Telemetry contract

Strategy, plan-node, fallback, timing, lifecycle, and usage events contain
metadata only. Plan call and node event identifiers are runner-generated
attempt/ordinal IDs; model-provided node identifiers are never projected into
those telemetry fields. `PlanLifecycle` identifies the `planning`, `repair`,
`validation`, `preflight`, or `recovery` phase, its explicit attempt, and a
stable value-free outcome. `PlanNodeStarted` and `PlanNodeCompleted` carry the
execution-plan attempt directly, so consumers do not parse opaque node IDs.

`PlanNodeCompleted` reports a stable outcome: `succeeded`, `failed`,
`cancelled`, `timed_out`, `rejected`, `limit_reached`, or `reused`. Its duration
is measured for that node's own broker/preflight operation. Parallel-wave
completion events may still be emitted in deterministic node order after all
wave futures settle; the duration is per invocation rather than wave join time.
No error message, argument, result, model-provided node ID, or model output is
placed in these fields.

`StrategyUsage` keeps total provider and admitted tool-call counts and also
exposes phase and disposition counters. The following equalities are runtime
invariants:

```text
model_calls = planning_model_calls + repair_model_calls
            + recovery_model_calls + final_synthesis_model_calls
            + reactive_model_calls

tool_calls = tool_issued + tool_reused + tool_rejected
           + tool_pre_dispatch_aborted

tool_issued = tool_completed + tool_failed + tool_cancelled
```

`tool_calls` means proposals admitted under the broker's total-attempt limit;
it does not mean validated or executed calls. `tool_issued` means the execution
boundary was entered, including a keyed-permit wait. Calls stopped before that
boundary are rejected or pre-dispatch-aborted. This makes rejection,
cancellation, reuse, and partial-plan accounting comparable across Direct and
declarative strategies.

`final_synthesis_model_calls` counts the first provider call made after a
successful plan (including a successful recovery plan) to turn recorded tool
results into the final answer. If that response requests more tools, later
calls are reactive. Direct selection and pre-effect fallback never charge final
synthesis. A recovery lifecycle is marked `succeeded` only after the recovered
plan executes successfully; parsing a replacement plan is not execution
success.

Recommended evaluation metrics derived from these counters include planner
repair and recovery rates, rejected/attempted rate, reused/attempted rate,
issued success rate, issued cancellation rate, and per-outcome node latency.
Dashboards and SQLite indexes remain host concerns rather than core scheduler
behavior.

Provider raw deltas are not persisted by this execution path. The local SQLite
store records `PlanLifecycle` as `plan.lifecycle`; existing event kinds remain
unchanged. Protocol runtimes intentionally filter additive core-only strategy
and plan events when projecting the current wire protocol. Because the core
event emitter allocates sequence numbers before that projection, wire consumers
may observe sequence gaps; gaps are filtering artifacts, not lost or reordered
wire events. Protocol and SDK contracts remain unchanged.
