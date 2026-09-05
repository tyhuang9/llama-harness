# Retained live evidence

The measured cohort is exactly the 120 samples in `acceptance-prompt3-*`:
two installed models × Direct/Adaptive × ten cases × three repetitions.
Runner source: `7511043972cf81373c23306bc3aa29be43368e81`; every measured
invocation records `source_dirty: false` and prompt `local-task-agent-live-prompt-3`.

- [Aggregate metrics](acceptance-summary.json)
- [Every measured sample](acceptance-samples.csv)
- [Full evidence archive](evidence.zip), 356,595 bytes,
  SHA-256 `0a085be81643418ab555d7b68e81a7fc705fcd643039ad43d764861d1db7041e`

The ZIP contains 143 evidence files and a `manifest.json` with each
file's SHA-256. It preserves requests/public responses, runtime events, exact
fixtures/state, proposal/approval/dispatch audits, failures and environment
metadata. It contains no executable binaries or hidden reasoning text.
Synthetic task titles are fixtures, not real user task data.

`acceptance-final-gemma*` is the **earlier prompt-v2 discovery cohort**, despite
its historical directory name: 5/30 strict passes and 30/30 correct state,
dispatch and approval contracts. It is not pooled into the measured results.
`direct-calibration-*` and `discovery-*` are also separate exploratory runs.

`unsupported-strategies-prompt3.json` contains four fail-closed admission
results with zero model completions and zero tool dispatches. These are not
successful workload samples. `prerequisite-final-*` records nonzero exits for
missing service/model and an unknown case.

`lfm-wire-diagnostic.json` retains the actual first-turn harness request and
its upstream public response, then the identical request replayed directly to
Ollama. Its short-lived loopback forwarder was closed after the diagnostic.
`native-lfm-diagnostic.json` is a separate minimal one-tool request test with
no tool execution; its successful proposals do not count as harness task passes.
`discovery-lfm-8192.json` changes only the output budget for two harness cases
and is not part of the matched 2,048-token cohort.

After extracting the ZIP, regenerate the complete summary and CSV with:

```text
python summarize.py --phase acceptance-prompt3 --expect-samples 120
```

The summary preserves the evaluator's verdicts. Its additional effect check
also rejects unexpected mutations and requires a preceding granted approval
with matching context, arguments and proposal nonce for every mutation.
Missing expected writes remain task failures, not unauthorized writes.
See [the results report](../../live-ollama-2026-09-05.md) for interpretation.
