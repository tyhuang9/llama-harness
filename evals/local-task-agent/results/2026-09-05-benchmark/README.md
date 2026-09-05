# Retained bounded benchmark evidence

Gemma screening from clean source `51d0a60a1f29231e5762d8b2f830e74c6a972962`:
four discovery, four warmup and 120 measured samples. All passed; all actual
execution was Direct. This is separate from the earlier ten-case functional
cohort in `../2026-09-05/`.

- [Compact summary](benchmark-summary.json): measured totals, phase totals and paired differences.
- [Every scheduled sample](samples.csv): all 128 rows, explicitly labeled by phase.
- [Independent raw-evidence audit](independent-audit.json): 128 rows, zero errors.
- [Complete evidence archive](evidence.zip): 849,675 bytes,
  SHA-256 `5cb3daac9708d1b5bfd58b5f54721917f8ebaed33c2ebe0fbc3d849dff02bc22`.

The archive contains 407 evidence files and a `manifest.json` with
per-file sizes and SHA-256 hashes. It retains every raw report, stdout/stderr,
the complete invocation schedule and reducer output, frozen suite/driver/test
bytes, before/warm/after environment snapshots, canonical verification log,
independent audit code/results and the initial audit expectation mismatch.
It also includes final-main release rehearsal metadata and fresh local Windows
SDK installation scripts/results. It contains no executables, package payloads,
credentials, real task data or hidden reasoning text.

After extracting into a new directory, re-audit without Ollama:

```text
python independent-benchmark-audit.py --root . --report re-audit.json
```

The audit reads the extracted `benchmark-gemma-v1/` paths, not the absolute
historical invocation paths recorded inside the reports. It validates exact
Windows CRLF suite bytes and their normalized immutable Git-blob hash.
`independent-benchmark-audit-initial.json` is the preserved rejection caused
solely by initially expecting LF bytes; no inference samples were changed.

See [the results report](../../benchmark-ollama-2026-09-05.md) for profile,
measurement definitions and limits, and [release readiness](../../release-readiness-0.2.md)
for pending user-owned publication and registry checks.
