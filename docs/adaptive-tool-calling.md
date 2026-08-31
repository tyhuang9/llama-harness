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
require a finite deadline and Programmatic-run admission. A slot is acquired
nonblocking before a candidate program is requested, parsed, and compiled.
If all slots are held, the candidate immediately ends as `LimitReached` with
no model or tool work. An admitted slot remains held while its `Execution`
retains state across broker, policy, approval, or tool waits, and is released
only after final synthesis and output validation make the whole Programmatic
run terminal. `max_active_vms` therefore bounds concurrent Programmatic-run
state—program and model buffers, VM state, canonical tool transcript, and
final synthesis buffers—not only sandbox VM live bytes; it is not a per-slice
compute throttle. For conservative host capacity planning, multiply the slot
count by the sum of the effective program-byte cap, model-response cap,
sandbox live-byte cap, transcript cap, and the effective bounded batch count
times the tool-result cap. This is an upper envelope rather than an exact heap
measurement because those bounded buffers can overlap. Read-only, parallel-safe fan-out is
bounded at eight and additionally limited by the host, provider, Agent, and
broker caps. Mutations are always serialized.

The runner yields to Tokio after every fuel slice and rechecks cancellation and
the absolute deadline before it starts the next slice. This keeps slice-fuel
fairness explicit even for a program that has no tool, policy, or provider
await points. The sandbox verifier's allocation complexity remains bounded by
the effective bytecode, local, control-stack, and operand-stack limits. Operand
stack reuse across independent resumable evaluation states is a separately
profiled follow-up; the current VM retains its bounded stack with each active
continuation to preserve isolation and accounting.

One correction prompt is permitted only before an effect is dispatched and
only while more than one model call remains: one call stays reserved for final
synthesis. A two-call budget therefore spends its second call on the approved
zero-effect Direct fallback after an invalid initial program; a three-call
budget can repair once and still synthesize the verified program's answer. If
the corrected program is still invalid and no effect was issued, the runner
enters a fresh Direct scope as a continuation of the same logical run: it
preserves the public run and trace IDs, event sequence, deadline, cumulative
budgets, and broker state while recording `invalid_program` fallback metadata.
A forced run therefore never claims that Programmatic succeeded. Program
generation, repair, and final synthesis deliberately use the bounded completion
path even when a provider advertises streaming; streaming is not a programmatic
runtime contract yet. Cancellation, deadline, resource, invalid-output, tool,
resume, or result failures after dispatch leave the effect uncertain and
terminal. They never repair, restart, replay, speculate, or fall back.

To roll out safely, leave the feature disabled by default, gate the builder
configuration in the host deployment, and force Programmatic only in an
evaluation matrix with conforming providers. Removing that configuration or
feature stops future programmatic runs, but cannot undo an external effect
already dispatched; rely on application-level idempotency and the broker's
uncertain-effect boundary for recovery. Program lifecycle telemetry and SQLite
kind hooks are metadata-only; source, AST, bytecode, constants, local values,
arguments, results, raw errors, and model identifiers are not recorded there.
`ProgramExecutionCompleted` is a nonterminal VM fact. A program attempt emits
its only terminal lifecycle outcome only after final synthesis and output
validation succeed; synthesis errors instead emit one final `failed`,
`cancelled`, `timed_out`, or `limit_reached` outcome.

`ProgramExecutionCompleted` also reports value-free VM telemetry: fuel,
scheduling slices entered by the host loop, tool-yield batches, branches,
bounded loop iterations, fan-out batches, partial failures, peak accounted
bytes, and VM duration. These counters are persisted as local metadata and
available to trace-store filtering/export only; the runtime wire contract does
not project this Programmatic-only event yet. Projecting it is deferred to the
compatibility/protocol milestone; the current runtime filters core-only events,
so its wire sequence can have intentional filtering gaps without losing or
reordering projected events.

### Programmatic event-contract matrix

| Case | Terminal result | `ProgramExecutionCompleted` |
| --- | --- | --- |
| Verified program and final synthesis succeed | Program lifecycle `succeeded`; one run `completed` event | Present before final synthesis |
| First program is repaired, then verified and synthesized | Repair is counted; successful attempt is `succeeded` | Present for the verified attempt |
| Invalid program falls back before effects | Same-run Direct continuation, never Programmatic success | Absent |
| Provider failure before VM completion | Failed terminal result | Absent |
| Final synthesis failure after VM completion | Failed terminal lifecycle, no second success | Present |
| Policy or approval rejection | Broker rejection; terminal result follows the program outcome | Present only if the VM subsequently completes |
| Partial tool failure returned to the VM | Recorded as partial failure; terminal result follows the program outcome | Present if the VM completes |
| Cancellation or deadline | Cancelled/timed-out terminal result | Present only when VM completion occurred first |
| VM fuel, admission, or transcript limit | Limit-reached terminal result | Absent unless completion preceded a later transcript/synthesis limit |

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
unchanged. Protocol runtimes intentionally filter additive core-only strategy,
plan, and Programmatic events until the compatibility/protocol milestone
projects them. Because the core event emitter allocates sequence numbers before
that projection, wire consumers may observe sequence gaps; gaps are filtering
artifacts, not lost or reordered wire events. Protocol and SDK contracts remain
unchanged.
