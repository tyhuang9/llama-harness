# Security boundaries

Llama Harness is a local integration framework, not a sandbox. The application
owns every provider credential, tool capability, authorization rule, approval
surface, and storage path.

- The embedded runner validates tool schemas and allowlists before policy,
  approval, and tool execution. Model output never gains a shell, filesystem,
  network, or database capability unless the host intentionally registers one.
- The runtime sidecar uses private stdin/stdout JSONL, no TCP listener, bounded
  protocol frames and queues, generated correlation IDs, a mandatory handshake,
  and fail-closed callback handling. Do not treat its stdin peer as trusted.
- SDK runtime discovery never searches arbitrary PATH entries or downloads a
  binary. Pin platform package versions and verify the release manifest/checksum.
- Ollama endpoints must be loopback; redirects are disabled. The runtime never
  pulls a model. Live Ollama is opt-in operational testing, not a test fixture.
- `read_only` is host-declared metadata, not proof that a tool has no side
  effects. Keep schemas narrow, require explicit policy for mutations, and use
  application idempotency keys or proposal/commit flows where appropriate.
- Guarded speculation requires independent runner, tool, and dedicated policy
  opt-ins. Enable it only for caller-invariant, authorization-stable,
  freshness-stable local private reads with guaranteed issue and cancellation
  safety and no egress. Writes, remote tools, and imported MCP tools are
  ineligible. One no-wait global slot and any no-wait concurrency-key permit
  prevent speculative queue growth.
- A finalized index-0 call is the earliest executable candidate. Exact typed
  identity and arguments are revalidated at commit. A mismatch, stale binding,
  candidate failure, cancellation, or terminal stream failure trips that
  tool's breaker and preserves safe sequential behavior; a stream is never
  retried after an accepted item.
- SQLite events are structured causal metadata. Raw payload storage is opt-in;
  do not persist chain-of-thought, secrets, or authorization tokens. Restrict
  database paths to application-controlled directories.
- Speculative candidate existence, arguments, results, raw errors, readiness,
  and counters are privacy-sensitive pull-only runner state. They are absent
  from canonical events, results, SQLite, protocol messages, and SDK
  projections. Do not copy them into raw trace payloads.
- Cancellation, deadlines, and process shutdown are cooperative. They prevent
  future runner work but cannot roll back an external effect that has begun.
- Tauri approvals and run events are sensitive host-to-window messages. Target
  them to the main application window with `TauriTargetEmitter`; never
  broadcast them to widgets, previews, or auxiliary windows. `TauriEmitter` is
  retained only for non-sensitive compatibility broadcasts.

Report a suspected vulnerability privately to the repository maintainers with a
minimal reproduction, affected revision, impact, and suggested mitigation. Do
not publish credentials, private traces, or exploitable details in an issue.
