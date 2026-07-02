# Milestones

## Milestone 0 - Project Setup

### Status
Completed

### Summary
Created the repo structure, Rust server skeleton, React admin UI skeleton, TypeScript client SDK skeleton, and root project docs.

### Files Changed
- `server/`
- `admin-ui/`
- `clients/ts/`
- `docs/MILESTONES.md`
- `README.md`
- `TODO.md`
- `config.json`
- `.gitignore`

### Implementation Notes
The repo was empty except for `.git`. `origin` exists, but `origin/main` was not available after fetch, so the work started from the empty local state on `feature/initial-mvp`. The MVP uses JSON files and in-memory state only; no database was added.

### Manual Test Steps
1. Run `cargo check` in `server/`.
2. Run `cargo test` in `server/`.
3. Run `npm install` and `npm run build` in `admin-ui/`.
4. Run `npm install` and `npm run build` in `clients/ts/`.

### Known Issues / Follow-ups
- No remote PR exists yet.
- The server and UI are run as separate local processes.

### Commit / PR
Local implementation commit: `8336012`. PR not created yet.

## Milestone 1 - Ollama Connection

### Status
Completed

### Summary
Added health and model endpoints that connect to a local Ollama instance, defaulting to `http://localhost:11434`.

### Files Changed
- `server/src/ollama.rs`
- `server/src/routes.rs`
- `server/src/config.rs`
- `admin-ui/src/App.tsx`
- `admin-ui/src/api.ts`
- `README.md`

### Implementation Notes
The server calls Ollama `/api/tags` for reachability and model listing. Health remains available even if Ollama is not reachable.

### Manual Test Steps
1. Start Ollama locally.
2. Run `cd server && cargo run`.
3. Open `http://127.0.0.1:8787/health`.
4. Open `http://127.0.0.1:8787/api/models`.

### Known Issues / Follow-ups
- Verified locally with Ollama reporting one installed model, `qwen2.5:7b`.

### Commit / PR
Local implementation commit: `8336012`. PR not created yet.

## Milestone 2 - Basic Chat API

### Status
Completed

### Summary
Added non-streaming chat and streaming SSE chat endpoints for external local apps.

### Files Changed
- `server/src/ollama.rs`
- `server/src/routes.rs`
- `server/src/runs.rs`
- `admin-ui/src/App.tsx`
- `clients/ts/src/index.ts`
- `README.md`

### Implementation Notes
Chat requests can pass `messages` or a simple `prompt`. Each request can override the model, or it can use the configured default model. Run metadata is recorded for successful and failed calls.

### Manual Test Steps
1. Configure a valid default model or pass `model` in the request.
2. Run `curl -X POST http://127.0.0.1:8787/api/chat -H 'content-type: application/json' -d '{"prompt":"Say ok.","model":"MODEL_NAME"}'`.
3. Run `curl -N -X POST http://127.0.0.1:8787/api/chat/stream -H 'content-type: application/json' -d '{"prompt":"Say ok.","model":"MODEL_NAME"}'`.

### Known Issues / Follow-ups
- Streaming is an MVP SSE bridge and may need more robust client cancellation metadata later.
- Non-streaming chat was smoke-tested with local `qwen2.5:7b`; streaming should still be tested manually with a longer prompt.

### Commit / PR
Local implementation commit: `8336012`. PR not created yet.

## Milestone 3 - Settings

### Status
Completed

### Summary
Added `config.json`, settings read/update endpoints, and admin UI settings controls.

### Files Changed
- `config.json`
- `server/src/config.rs`
- `server/src/routes.rs`
- `admin-ui/src/App.tsx`
- `admin-ui/src/api.ts`
- `README.md`

### Implementation Notes
Settings are persisted as pretty JSON. The server discovers `config.json` from the current working directory, `../config.json`, or `LLAMA_HARNESS_CONFIG`.

### Manual Test Steps
1. Run `curl http://127.0.0.1:8787/api/settings`.
2. Run `curl -X PUT http://127.0.0.1:8787/api/settings -H 'content-type: application/json' -d '{"default_model":"MODEL_NAME"}'`.
3. Verify `config.json` updates.

### Known Issues / Follow-ups
- The API token setting is persisted but not enforced.

### Commit / PR
Local implementation commit: `8336012`. PR not created yet.

## Milestone 4 - Lightweight Runs/Audit

### Status
Completed

### Summary
Added recent run tracking in memory and optional append-only JSONL persistence.

### Files Changed
- `server/src/runs.rs`
- `server/src/routes.rs`
- `config.json`
- `admin-ui/src/App.tsx`
- `README.md`
- `TODO.md`

### Implementation Notes
The server keeps the most recent 100 runs in memory and appends to `runs.jsonl` when logging is enabled. No database is used.

### Manual Test Steps
1. Send a successful or failed chat request.
2. Run `curl http://127.0.0.1:8787/api/runs`.
3. Verify recent run metadata appears in the admin UI Runs page.

### Known Issues / Follow-ups
- Add fake-Ollama integration tests for repeatable run-history checks.
- Live run metadata was verified through `/api/runs` after a successful local chat smoke test.

### Commit / PR
Local implementation commit: `8336012`. PR not created yet.

## Milestone 5 - TypeScript Client

### Status
Completed

### Summary
Added a small TypeScript SDK for external apps that call llama-harness.

### Files Changed
- `clients/ts/package.json`
- `clients/ts/tsconfig.json`
- `clients/ts/README.md`
- `clients/ts/src/index.ts`
- `README.md`

### Implementation Notes
The SDK exposes `health`, `listModels`, `chat`, `streamChat`, `runs`, `settings`, and `updateSettings`. Streaming uses a POST request and parses SSE chunks from `fetch`.

### Manual Test Steps
1. Run `cd clients/ts && npm install`.
2. Run `npm run build`.
3. Import `LlamaHarnessClient` from `clients/ts/src/index.ts` or the built `dist` output.

### Known Issues / Follow-ups
- Package publishing metadata is intentionally minimal.

### Commit / PR
Local implementation commit: `8336012`. PR not created yet.

## Milestone 6 - Global Instructions and UI Refresh

### Status
Completed

### Summary
Adds persisted global instruction settings, request-specific chat instructions, and a softer dark admin UI with a dedicated Instructions page.

### Files Changed
- `config.json`
- `server/src/config.rs`
- `server/src/routes.rs`
- `admin-ui/src/App.tsx`
- `admin-ui/src/App.css`
- `admin-ui/src/api.ts`
- `clients/ts/src/index.ts`
- `clients/ts/README.md`
- `README.md`
- `TODO.md`

### Implementation Notes
Global instructions are stored in `config.json` under `instructions`. When enabled, the server prepends them as a system message before forwarding chat requests to Ollama. Per-request `instructions` can also be supplied through the API and TypeScript client.

### Manual Test Steps
1. Run `cargo test` in `server/`.
2. Run `npm run build` in `admin-ui/`.
3. Run `npm run build` in `clients/ts/`.
4. Update `/api/settings` with enabled instructions and verify `/api/settings` returns the saved values.

### Known Issues / Follow-ups
- Addressed after the MVP: app-owned tools can now be requested through runs and executed by the registered app.

### Commit / PR
Local implementation commit: `33050fa`. PR not created yet.

## Milestone 7 - App-Agent Policy Model

### Status
Completed

### Summary
Added first-class local Models, Agents, Apps, Tools, Runs, and Audit concepts so external apps can connect with an app id and let llama-harness resolve the assigned agent, tools, model, and run policy.

### Files Changed
- `server/src/app_policy.rs`
- `server/src/config.rs`
- `server/src/routes.rs`
- `server/src/runs.rs`
- `config/models.json`
- `config/agents.json`
- `config/apps.json`
- `config/tools.json`
- `admin-ui/src/App.tsx`
- `admin-ui/src/api.ts`
- `clients/ts/src/index.ts`
- `README.md`
- `.gitignore`

### Implementation Notes
Agents are reusable top-level records and are not nested under apps. Apps define `defaultAgentId`, `allowedAgentIds`, optional `allowedToolIds`, and enabled state. `/apps/:appId/capabilities` resolves the default agent and visible capabilities for external apps such as Note. `/runs` and `/runs/stream` create app-policy runs and append lightweight run/audit JSONL records. Existing `/api/*` routes remain available for the admin UI and client SDK. No database was added.

### Manual Test Steps
1. Run `cargo fmt --check --manifest-path server/Cargo.toml`.
2. Run `cargo test --manifest-path server/Cargo.toml`.
3. Run `npm --prefix admin-ui run build`.
4. Run `npm --prefix clients/ts run build`.
5. Start a temporary server and verify `GET /health`, `GET /models`, `GET /apps`, `GET /apps/note/capabilities`, `POST /runs`, `GET /runs`, and `GET /audit`.

### Known Issues / Follow-ups
- Addressed after this milestone: enabled app-owned tools can now produce structured run tool requests and app-submitted results.
- If no default or local Ollama model is available, app runs return a model-selection error while capabilities show warnings.

### Commit / PR
Local implementation commit: `84494e3`. PR not created yet.
