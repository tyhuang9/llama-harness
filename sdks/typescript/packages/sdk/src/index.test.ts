import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import test from "node:test";

import { defineTool, HarnessClient } from "./index.js";

const here = dirname(fileURLToPath(import.meta.url));

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

test("performs a handshake with the workspace-built Rust runtime", { skip: !existsSync(resolve(process.cwd(), "../../../../target/debug/llama-harness-runtime.exe")) }, async () => {
  const client = await HarnessClient.start({ provider: { kind: "ollama" }, runtimePath: resolve(process.cwd(), "../../../../target/debug/llama-harness-runtime.exe") });
  await client.close();
});
