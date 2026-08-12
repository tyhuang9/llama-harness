import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import test from "node:test";

import { defineTool, HarnessClient } from "./index.js";

const here = dirname(fileURLToPath(import.meta.url));
const workspaceRoot = resolve(process.cwd(), "../../../../");
const scriptedRuntime = join(workspaceRoot, "target", "debug", process.platform === "win32" ? "llama-harness-scripted-runtime.exe" : "llama-harness-scripted-runtime");

test("routes a typed host tool callback and ordered runtime event", async () => {
  const client = await HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: process.execPath, runtimeArgs: [join(here, "fake-runtime.js")] });
  try {
    const tool = defineTool({ id: "notes.search", name: "Search", description: "Search notes", argumentsSchema: {}, risk: "low", idempotent: true, readOnly: true, execute: async (toolArguments) => ({ found: (toolArguments as { query: string }).query }) });
    const run = await client.run({ agent: { id: "agent", name: "Agent", version: "1", defaultModel: "mock", toolAllowlist: ["notes.search"] }, input: "find", tools: [tool] });
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
