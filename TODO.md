# TODO

## Active

- [ ] Decide the end state for the isolated legacy server, TypeScript client, LiteLLM scripts, and legacy configuration.
  - Priority: High
  - Area: migration
  - Notes: The reworked embedded runtime and local console do not depend on these files. Remove or adapt them in a dedicated, separately reviewable branch.

- [ ] Add installer, signing, and Windows/macOS/Linux packaging checks for `apps/harness-console`.
  - Priority: Medium
  - Area: developer console
  - Notes: The Tauri crate and frontend build locally; release packaging has not been verified.

- [ ] Add console integration coverage against a real temporary SQLite trace store and report-artifact directory.
  - Priority: Medium
  - Area: developer console
  - Notes: Unit tests cover path/loopback/command constraints and React state. A cross-process Tauri integration harness remains future work.

- [ ] Upgrade the console Vite/esbuild dependency chain after evaluating the breaking changes.
  - Priority: Low
  - Area: developer console
  - Notes: `npm audit` currently reports transitive development-tool vulnerabilities. Do not use a forced upgrade without testing the Tauri/Vite integration.

## Completed

- [x] Embedded core, direct local Ollama, SQLite traces, evaluation contracts, CLI, and local task-agent reference.
- [x] Replace the daemon-backed seeded admin dashboard with the optional project-oriented developer console.
