# 0.2 release-readiness assessment — September 5, 2026

The current implementation has demonstrated real, audited task-tool round trips
with the installed Gemma model. It is ready for review of this validation PR.
The evidence does not establish general model compatibility or production
readiness for advanced strategies. This assessment does not authorize a merge,
tag, release, publication, or strategy activation.

| Surface | Verified evidence | Remaining condition |
| --- | --- | --- |
| Current main integration | PRs #28–#35 merged; exact main `8e52758` continuous verification and docs workflows passed | Verify the validation PR's own head checks before merge |
| Rust 0.2 contracts and packaging | Canonical `release-check` passed: 492 tests, two ignored, docs, archive validation and extracted consumers | Explicit approval for a specific PR merge; separate release approval |
| Live evaluation infrastructure | 20 deterministic example tests; strict state, dispatch, approval, output and audit assertions; missing prerequisites exit nonzero | Keep live inference opt-in and outside ordinary CI |
| Gemma Direct | 30/30 strict live passes across ten cases, including approval denial and recovery | More task diversity, larger fixtures, production failure injection and larger cohorts |
| Gemma Adaptive | 30/30 strict passes; all 30 actually executed Direct | No inference of DAG, Programmatic, or speed improvement |
| LFM under the measured profile | 0/30 Direct and 0/30 Adaptive; all stores safe, zero tool proposals | Qualify a prompt/catalog/sampling profile in a new matched cohort before deployment |
| Ollama adapter | Real Gemma dependent and multiple-call round trips; IPv6 loopback defect fixed and regression-tested; LFM identical-payload native replay reproduced its failure | Broader provider/model and streaming workload evidence remains separate |
| DAG and Programmatic | Four explicit admission checks rejected unsupported configurations before any model/tool calls | Conforming provider, configured sandbox where required, workload evidence, then specific promotion approval |
| Speculation | Disabled throughout; no writes speculated | Existing per-tool Shadow and activation gates plus matched evidence, with explicit approval |
| SDK and protocol compatibility | No SDK/protocol changes in this PR; current-main SDK CI passed; canonical protocol contract checks passed | Production embedding, services and deployment were not exercised by these local fixtures |

The [results and retained artifacts](live-ollama-2026-09-05.md) establish a
limited functional result: Gemma, Ollama 0.33.3, prompt v3, a four-tool synthetic
catalog, fixed generation limits, and a local in-memory store. The instructions
explicitly name required tools and the output protocol; this is a guided tool
integration test, not a benchmark of open-ended planning. Three repetitions
per case do not establish reliability rates or tail-latency bounds.

Unauthorized, duplicate, and unintended writes were zero across the 120 matched
samples. That includes 60 LFM failures that made no tool requests: absence of
effects in those trials does not validate LFM approval-denial or recovery
competence. The same ten cases do establish those behaviors for Gemma.

The [production-like benchmark plan](strategy-benchmark-plan.md) is prepared
with exact workload sizes, correctness gates, matched settings, balancing,
measurement and rollback conditions. It has not been run. Keep the Adaptive
Programmatic allowlist empty and speculation Disabled. No benchmark or package
publication is justified by these timings alone.

The PR is opt-in evaluation work plus a narrow loopback URL fix. Rollback is to
stop invoking the evaluation binary and, if necessary, revert the IPv6 parsing
change. No user data migration, service reconfiguration, persistent task-store
write, or deployment is part of this change.
