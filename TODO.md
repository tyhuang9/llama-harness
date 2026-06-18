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

None.

## Completed

- [x] ~~Implement app-agent policy model.~~
  - Completed in: `84494e3`
  - Notes: Added top-level app/agent/tool/model catalog files, app capability resolution, app-policy run endpoints, backend audit JSONL, admin UI Apps/Tools/Audit updates, and TypeScript client helpers.

- [x] ~~Create initial project structure.~~
  - Completed in: `8336012`
  - Notes: Added `server/`, `admin-ui/`, `clients/ts/`, and `docs/`.

- [x] ~~Implement Ollama-only MVP API surface.~~
  - Completed in: `8336012`
  - Notes: Added health, models, chat, streaming chat, settings, runs, and tools placeholder endpoints.

- [x] ~~Create minimal local admin UI.~~
  - Completed in: `8336012`
  - Notes: Added dashboard, models, runs, tools, and settings views.

- [x] ~~Create TypeScript client SDK skeleton.~~
  - Completed in: `8336012`
  - Notes: Added health, listModels, chat, streamChat, runs, settings, and updateSettings helpers.

- [x] ~~Verify the MVP against a live Ollama model on the target machine.~~
  - Completed in: `8336012`
  - Notes: Verified `/health`, `/api/settings`, `/api/runs`, and `/api/chat` with local `qwen2.5:7b`.

- [x] ~~Complete initial MVP branch.~~
  - Completed in: `8336012`
  - Notes: Built the server, admin UI, TypeScript client, and MVP docs on `feature/initial-mvp`.

- [x] ~~Merge initial MVP into main.~~
  - Completed in: local `main` branch at `41652ac`
  - Notes: Created local `main` from the completed initial MVP branch because the repo had no existing main commit.

- [x] ~~Add global instructions and admin UI refresh.~~
  - Completed in: `33050fa`
  - Notes: Added global/request-specific LLM instructions and a softer dark admin UI.

## Backlog / Future

- [ ] Add a production static-file mode for serving the built admin UI from the Rust server.
  - Priority: Low
  - Notes: Useful later if the harness should run as one local process.

- [ ] Add a richer local tool registry after the API contract settles.
  - Priority: Low
  - Notes: Tool records are now first-class visible capabilities, but real local tool execution is still intentionally out of scope.

- [ ] Add integration tests with a fake Ollama server.
  - Priority: Medium
  - Notes: Would make model and chat API behavior testable without requiring local Ollama.
