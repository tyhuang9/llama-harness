# TODO

## Active

- [ ] Decide whether the local API token setting should be enforced by the server.
  - Priority: Medium
  - Area: server
  - Notes: `api_token` is persisted in settings for future local auth, but complex auth is out of scope for the MVP.

- [ ] Add packaging guidance for running the server and UI together.
  - Priority: Medium
  - Area: infra
  - Notes: Current setup uses separate Rust and Vite dev commands.

- [ ] Upgrade Vite/esbuild dev tooling when the project is ready for the breaking update.
  - Priority: Low
  - Area: admin-ui
  - Notes: `npm audit` reports dev-server advisories through Vite 5; `npm audit --omit=dev` reports 0 production vulnerabilities.

## In Progress

- [ ] Initial MVP branch.
  - Branch: feature/initial-mvp
  - Owner: Codex
  - Notes: Builds the server, admin UI, TypeScript client, and MVP docs.

## Completed

- [x] ~~Create initial project structure.~~
  - Completed in: pending local commit on `feature/initial-mvp`
  - Notes: Added `server/`, `admin-ui/`, `clients/ts/`, and `docs/`.

- [x] ~~Implement Ollama-only MVP API surface.~~
  - Completed in: pending local commit on `feature/initial-mvp`
  - Notes: Added health, models, chat, streaming chat, settings, runs, and tools placeholder endpoints.

- [x] ~~Create minimal local admin UI.~~
  - Completed in: pending local commit on `feature/initial-mvp`
  - Notes: Added dashboard, models, runs, tools, and settings views.

- [x] ~~Create TypeScript client SDK skeleton.~~
  - Completed in: pending local commit on `feature/initial-mvp`
  - Notes: Added health, listModels, chat, streamChat, runs, settings, and updateSettings helpers.

- [x] ~~Verify the MVP against a live Ollama model on the target machine.~~
  - Completed in: pending local commit on `feature/initial-mvp`
  - Notes: Verified `/health`, `/api/settings`, `/api/runs`, and `/api/chat` with local `qwen2.5:7b`.

## Backlog / Future

- [ ] Add a production static-file mode for serving the built admin UI from the Rust server.
  - Priority: Low
  - Notes: Useful later if the harness should run as one local process.

- [ ] Add a richer local tool registry after the API contract settles.
  - Priority: Low
  - Notes: Tool calling is intentionally only a placeholder in the MVP.

- [ ] Add integration tests with a fake Ollama server.
  - Priority: Medium
  - Notes: Would make model and chat API behavior testable without requiring local Ollama.
