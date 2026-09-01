# Guarded speculative tool calling

Guarded speculation can overlap one eligible local read with the tail of a
streamed Direct model response. It is an optional optimization inside the
existing Direct path, not a fourth `RunStrategy`, a second broker, or permission
to predict side effects. Omitting `AgentRunnerBuilder::speculation` preserves the
ordinary non-streaming completion path.

## Execution boundary

The runner considers only the finalized tool call at provider index `0`. A
partial argument fragment is never executable. After the provider marks that
indexed call final, the core assembler has fixed its call ID, registered tool
ID, schema-valid typed arguments, and canonical JSON. A provider that cannot
surface a useful final-call boundary may still stream normally, but it gains no
speculative overlap.

The candidate future remains pinned inside the reactive turn and is polled
alongside the provider tail. This works for push streams, pull-driven streams,
and bounded-channel streams without spawning a detached task. If the provider
finishes first, the turn continues polling the candidate through its bounded
deadline and cooperative cleanup. If the candidate finishes first, the runner
continues consuming and validating the authoritative tail. Terminal provider
failure cancels and drains a pending candidate before returning and is never
retried.

Speculation applies only to Direct execution, including Adaptive after Adaptive
has selected its Direct path. Declarative plans and Programmatic execution do
not speculate. Later calls in a multi-call response execute sequentially in
authoritative index order through the ordinary broker.

The runner permits at most one candidate globally. It tries that slot without
waiting. If the tool has a concurrency key, the keyed permit is also tried
without waiting. A busy slot or key immediately preserves sequential Direct
execution; speculation never adds a queue ahead of authoritative work.

## Three independent opt-ins

No tool becomes eligible from read-only metadata alone. All three authorities
must opt in:

1. The host configures bounded shadow-first speculation on the runner.
2. The tool allows `ToolCaller::Speculative` and sets its dedicated
   `SpeculationPolicy` to `Enabled` while satisfying every static safety gate.
3. The host policy independently returns `Allow` from `decide_speculative` for
   the exact typed candidate.

An ordinary policy allow, an approval grant, or `AllowAllPolicy` does not satisfy
the third gate. The default speculative policy decision is deny.

`SpeculationPolicy::Enabled` is a strong host attestation. For identical
canonical arguments and the bound run context, Direct and Speculative calls
must have caller-invariant successful-result semantics. Authorization and the
meaningful freshness of a successful result must remain stable for the entire
bounded candidate window. Leave volatile, ACL-sensitive, identity-sensitive,
caller-sensitive, or otherwise freshness-sensitive reads disabled.

Eligible definitions must also be read-only, idempotent, parallel-safe,
guaranteed safe to cancel, guaranteed side-effect-free merely to issue,
confined to a local private execution location, and prohibited from network
egress. Unknown declarations fail closed. Writes, remote tools, tools with
egress, and imported MCP tools are ineligible.

## Shadow-first activation

Each eligible tool starts in Shadow. Shadow mode validates a candidate and
compares it with the authoritative finalized index-0 call, but does not invoke
the tool or call speculative policy. The same runner must observe at least
`MIN_SPECULATION_SHADOW_OBSERVATIONS` consecutive exact matches for that tool.
The host must then explicitly call `activate_speculation`; reaching the
threshold alone never enters Active.

Readiness and counters are available only through the runner's pull APIs. State
is per tool and per runner. A mismatch, terminal stream failure, failed or
discarded issued candidate, cancellation, or explicit host action returns only
that tool to Shadow and clears its readiness streak. There is no automatic
fleet-wide activation.

## Exact commit and fallback

Active mode issues a bounded candidate through the same core broker and the
dedicated speculative policy decision. The candidate result is reusable only
when the authoritative first call exactly matches the candidate's call ID,
registered tool identity, and typed canonical arguments, and the bound tool,
catalog generation, allowlist, caller permissions, safety metadata, output
schema, size limits, deadline, and policy remain valid at commit.

An exact candidate is committed once into the canonical Direct transcript and
accounting. It is not invoked again. A stale binding, invalid result, occupied
no-wait permit, deadline, or failed gate discards or skips the candidate and
continues with sequential Direct execution when doing so is safe. Invocation
mismatch remains a defensive internal breaker: a conforming provider cannot
change a call after the finalized index-0 capture boundary. Candidate work
never authorizes a write, remote call, network egress, or retry.

The atomic commit check is the candidate publication boundary. After it
succeeds, the broker releases the speculative runner slot and concurrency-key
permit before emitting the canonical `ToolCompleted` event. A slow synchronous
event sink therefore cannot keep speculative permits occupied or make the
already-authorized cached result stale. `ToolCompleted` still precedes the
result's transcript insertion in normal Direct event order; sink latency is
ordinary publication latency, not speculative execution or publication-wait
latency. The pull-only `publication_wait_ms` metric instead measures from any
speculative tool result becoming ready, including a failed or invalid result,
to its terminal settlement, whether commit, discard, or cancellation wins.

Once a streamed item has been accepted, a terminal stream failure is terminal
for that model turn. The runner does not retry or replay the stream: a retry
could duplicate an effect whose issue state is uncertain. A candidate is
cancelled or discarded, its per-tool breaker trips, and the public failure stays
value-free.

## Privacy and observability

Candidate existence is privacy-sensitive. Shadow and Active diagnostics do not
add `RunEvent` variants, `RunResult` fields, SQLite columns, sidecar protocol
fields, SDK projections, or runtime messages. In particular, canonical
observability never exposes candidate arguments, results, raw errors,
readiness streaks, activation state, or speculative counters. Enabling raw
SQLite payload persistence does not synthesize speculative data; hosts must not
copy pull-only diagnostics into raw payloads.

Applications may poll `speculation_readiness` and `speculation_metrics` in the
same trusted process for rollout decisions. Treat those values as private
operational metadata, do not project them across the child protocol, and do not
use them as authorization.

The metrics snapshot keeps `issued == in_flight + committed + discarded +
cancelled`. `issued` increments only at the actual broker dispatch boundary.
Value-free pre-issue counters distinguish validation, policy, unexpected
failure, invalidation, cancellation/deadline, global-slot saturation, and
concurrency-key saturation. `oldest_in_flight_ms` supports stuck-work alerts.
Execution duration and speculative-result-ready-to-terminal-settlement wait use
cumulative fixed buckets at 1, 5, 10, 25, 50, 100, 250, 500, 1,000, 2,500, and
5,000 ms, plus an overflow count; each also reports count, saturating sum, and
maximum. These bounded aggregates are sufficient to derive P50/P95 without
retaining per-call timestamps or values.

## Evaluation and rollout

Use fresh runners and deterministic streaming providers to force a Disabled,
Shadow, Active exact-commit, mismatch or discard fallback, and saturated-slot
matrix. Include ineligible write, remote, egress-capable, and MCP definitions.
Judge correctness and safety before latency: completed task/final-state
correctness and zero unauthorized, duplicate, or unintended effects are hard
gates. Compare latency only among candidates that pass those gates.

Shadow evidence is exact and per tool. Record at least 1,000 observations before
explicit activation, reset evidence after every breaker incident, and retain a
Disabled baseline. Release measurements should report disabled, shadow, exact
Active, mismatch/discard fallback, and saturated fallback separately. They are
machine observations, not CI wall-clock thresholds.

Start Active rollout per tool with at least 100 Active considerations before
applying rate alerts. Alert when global-slot plus key saturation exceeds 10%,
the sum of pre-issue skip counters exceeds 20%, or committed divided by issued
falls below 95%. Alert immediately if `in_flight` exceeds one or
`oldest_in_flight_ms` exceeds the configured candidate duration plus the 250 ms
cooperative cleanup allowance. Any new breaker-incident delta is an immediate
signal to return that tool to Shadow; do not wait for a rate window.

Promotion requires matched Active and Disabled cohorts with at least 1,000
correct observations each, identical task/final-state correctness, and zero
unauthorized, duplicate, or unintended effects. Require no P95 latency
regression and an improvement of at least 10% or 25 ms before treating the
optimization as beneficial. These are rollout criteria, not authorization.
Rollback is the explicit per-tool `return_speculation_to_shadow` kill switch;
remove runner speculation configuration in the next deployment if the host
must disable all candidate observation. Keep the ordinary Direct path and
canonical policy/approval boundary enabled during either rollback.

The deterministic evaluation framework keeps mode selection in application
fixture data and obtains private counters from the trusted runner. It does not
extend public events or reconstruct candidates from traces. See
[Deterministic evaluations and replay](evaluations.md) and
[Local trace observability](observability.md).
