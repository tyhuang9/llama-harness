import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { tmpdir } from "node:os";
import test from "node:test";

import { defineTool, HarnessClient, RuntimeProtocolError } from "./index.js";

const here = dirname(fileURLToPath(import.meta.url));
const workspaceRoot = resolve(process.cwd(), "../../../../");
const scriptedRuntime = join(workspaceRoot, "target", "debug", process.platform === "win32" ? "llama-harness-scripted-runtime.exe" : "llama-harness-scripted-runtime");

test("routes a typed host tool callback and ordered runtime event", async () => {
  const client = await HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: process.execPath, runtimeArgs: [join(here, "fake-runtime.js")] });
  try {
    const tool = defineTool({ id: "notes.search", name: "Search", description: "Search notes", argumentsSchema: {}, risk: "low", idempotent: true, readOnly: true, execute: async (toolArguments) => ({ found: (toolArguments as { query: string }).query }) });
    const run = await client.run({ agent: { id: "agent", name: "Agent", version: "1", defaultModel: "mock", toolAllowlist: ["notes.search"] }, input: "find", tools: [tool], strategy: "adaptive" });
    const events = [];
    for await (const event of run.events()) events.push(event);
    const result = await run.result();
    assert.equal(result.status, "completed");
    assert.match(result.finalOutput ?? "", /harness/);
    assert.equal(events.length, 1);
    assert.equal(events[0].sequence, 2);
  } finally { await client.close(); }
});

test("keeps provider health and model inventory outside agent runs", async () => {
  const client = await HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: process.execPath, runtimeArgs: [join(here, "fake-runtime.js")] });
  try {
    assert.deepEqual(await client.health(), { healthy: true, detail: "fake runtime" });
    assert.deepEqual(await client.listModels(), [{ id: "mock", capabilities: { supportsTools: true, supportsStreaming: false, supportsStructuredOutput: true } }]);
  } finally { await client.close(); }
});

test("negotiates 1.1, derives hello identity from the package version, and serializes advanced strategy metadata", async () => {
  const client = await HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: process.execPath, runtimeArgs: [join(here, "fake-runtime.js"), "modern"] });
  try {
    assert.equal(client.negotiatedProtocolVersion, "1.1");
    const tool = defineTool({ id: "notes.search", name: "Search", description: "Search", argumentsSchema: {}, risk: "low", idempotent: true, readOnly: true, outputSchema: { type: "object" }, parallelSafe: true, concurrencyKey: "notes", cancellationSafety: "guaranteed", expectedLatencyMs: 5, allowedCallers: ["programmatic"], speculationPolicy: "disabled", issueSafety: "guaranteed", executionLocation: "local_private", networkEgress: "prohibited", execute: () => ({}) });
    const run = await client.run({ agent: { id: "agent", name: "Agent", version: "1", defaultModel: "mock" }, input: "test", tools: [tool], strategy: "programmatic" });
    const events = [];
    for await (const event of run.events()) events.push(event);
    assert.equal(events[0].event.type, "strategy_usage");
  } finally { await client.close(); }
});

test("falls back to 1.0 and rejects forced advanced strategies before start_run", async () => {
  const client = await HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: process.execPath, runtimeArgs: [join(here, "fake-runtime.js"), "legacy"] });
  try {
    assert.equal(client.negotiatedProtocolVersion, "1.0");
    await assert.rejects(client.run({ agent: { id: "agent", name: "Agent", version: "1", defaultModel: "mock" }, input: "test", strategy: "declarative_plan" }), RuntimeProtocolError);
    const direct = await client.run({ agent: { id: "agent", name: "Agent", version: "1", defaultModel: "mock" }, input: "test", strategy: "direct" });
    await direct.result();
  } finally { await client.close(); }
});

test("fails incompatible majors, envelope drift, and structured protocol errors", async () => {
  await assert.rejects(HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: process.execPath, runtimeArgs: [join(here, "fake-runtime.js"), "incompatible"] }), RuntimeProtocolError);
  await assert.rejects(HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: process.execPath, runtimeArgs: [join(here, "fake-runtime.js"), "version_mismatch"] }), /Runtime version mismatch/);
  const drift = await HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: process.execPath, runtimeArgs: [join(here, "fake-runtime.js"), "drift"] });
  try {
    const run = await drift.run({ agent: { id: "agent", name: "Agent", version: "1", defaultModel: "mock" }, input: "test" });
    await assert.rejects(run.result(), /version drift/);
  } finally { await drift.close(); }
  const failure = await HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: process.execPath, runtimeArgs: [join(here, "fake-runtime.js"), "protocol_error"] });
  try {
    await assert.rejects(failure.run({ agent: { id: "agent", name: "Agent", version: "1", defaultModel: "mock" }, input: "test" }), (error: unknown) => error instanceof RuntimeProtocolError && error.code === "invalid_state");
  } finally { await failure.close(); }
});

test("terminates the child when the handshake request is rejected", async () => {
  const directory = mkdtempSync(join(tmpdir(), "llama-harness-sdk-hello-"));
  const marker = join(directory, "pid");
  try {
    const startedAt = Date.now();
    await assert.rejects(HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: process.execPath, runtimeArgs: [join(here, "fake-runtime.js"), "hello_error_unresponsive", marker] }), /hello rejected/);
    assert.ok(Date.now() - startedAt < 1_000, "failed handshake cleanup must be bounded");
    const childPid = Number(readFileSync(marker, "utf8"));
    let childIsRunning = true;
    try { process.kill(childPid, 0); } catch { childIsRunning = false; }
    assert.equal(childIsRunning, false, "failed handshake child must be terminated");
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test("normalizes malformed callback safety metadata conservatively", async () => {
  const client = await HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: process.execPath, runtimeArgs: [join(here, "fake-runtime.js"), "malformed_metadata"] });
  try {
    let observed: unknown;
    const run = await client.run({ agent: { id: "agent", name: "Agent", version: "1", defaultModel: "mock" }, input: "test", policy: (request) => { observed = request.tool; return { outcome: "deny", reason: "fixture" }; } });
    await run.result();
    assert.deepEqual(observed, { id: "notes.search", name: "Search", description: "Search", argumentsSchema: {}, risk: "high", idempotent: false, readOnly: false, outputSchema: undefined, parallelSafe: false, concurrencyKey: undefined, cancellationSafety: "unknown", expectedLatencyMs: undefined, allowedCallers: [], speculationPolicy: "disabled", issueSafety: "unknown", executionLocation: "unknown", networkEgress: "unknown" });
  } finally { await client.close(); }
});

test("performs a handshake with the workspace-built Rust runtime", { skip: !existsSync(resolve(process.cwd(), "../../../../target/debug/llama-harness-runtime.exe")) }, async () => {
  const client = await HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: resolve(process.cwd(), "../../../../target/debug/llama-harness-runtime.exe") });
  await client.close();
});

test("completes host callbacks through the workspace-built scripted Rust sidecar", { skip: !existsSync(scriptedRuntime) }, async () => {
  let policyCalls = 0;
  let approvalCalls = 0;
  let toolCalls = 0;
  const client = await HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: scriptedRuntime });
  try {
    const search = defineTool({
      id: "notes.search", name: "Search notes", description: "Search locally owned notes",
      argumentsSchema: { type: "object", required: ["query"], properties: { query: { type: "string" } } },
      risk: "medium", idempotent: true, readOnly: false,
      execute: async (toolArguments) => { toolCalls += 1; return { found: (toolArguments as { query: string }).query }; },
    });
    const run = await client.run({
      agent: { id: "scripted", name: "Scripted", version: "1", defaultModel: "mock", toolAllowlist: [search.id] },
      input: "find harness", tools: [search],
      policy: async () => { policyCalls += 1; return { outcome: "require_approval", reason: "test approval routing" }; },
      approve: async () => { approvalCalls += 1; return { granted: true, reason: "approved by test host" }; },
    });
    const events = [];
    for await (const event of run.events()) events.push(event);
    const result = await run.result();
    assert.equal(result.status, "completed");
    assert.match(result.finalOutput ?? "", /host tool callback/);
    assert.equal(policyCalls, 1);
    assert.equal(approvalCalls, 1);
    assert.equal(toolCalls, 1);
    assert.ok(events.length >= 4);
    assert.deepEqual(events.map((event) => event.sequence), [...events].map((event) => event.sequence).sort((a, b) => a - b));
  } finally { await client.close(); }
});
