import { createInterface } from "node:readline";

function write(envelope: object): void { process.stdout.write(`${JSON.stringify(envelope)}\n`); }

createInterface({ input: process.stdin }).on("line", (line) => {
  const message = JSON.parse(line) as { request_id: string; type: string; payload: Record<string, unknown>; };
  if (message.type === "client_hello") {
    write({ protocol_version: "1.0", request_id: message.request_id, type: "runtime_hello", payload: { runtime_version: "test", capabilities: { supports_output_deltas: false, supports_structured_output: true, supports_trace_persistence: false, concurrent_runs: 1, max_pending_callbacks: 1, max_queue_depth: 8 }, providers: ["ollama"] } });
  } else if (message.type === "start_run") {
    write({ protocol_version: "1.0", request_id: message.request_id, run_id: "test-run", type: "command_acknowledged", payload: { command: "start_run" } });
    write({ protocol_version: "1.0", request_id: "tool-request", run_id: "test-run", type: "tool_execution_requested", payload: { run_sequence: 1, callback_id: "tool-1", trace_id: "test-trace", call_id: "call-1", tool: { id: "notes.search", name: "Search", description: "Search notes", arguments_schema: {}, risk: "low", idempotent: true, read_only: true }, arguments: { query: "harness" } } });
  } else if (message.type === "tool_result") {
    const result = (message.payload.result as { ok: boolean; output: unknown });
    write({ protocol_version: "1.0", request_id: "event-1", run_id: "test-run", type: "run_event", payload: { trace_id: "test-trace", sequence: 2, timestamp_ms: 1, event: { type: "tool_completed", call_id: "call-1", tool_id: "notes.search", ok: result.ok } } });
    write({ protocol_version: "1.0", request_id: "complete-1", run_id: "test-run", type: "run_completed", payload: { run_sequence: 3, result: { status: "completed", final_output: JSON.stringify(result.output), model: "mock", trace_id: "test-trace", duration_ms: 1 } } });
  } else if (message.type === "shutdown") {
    write({ protocol_version: "1.0", request_id: message.request_id, type: "command_acknowledged", payload: { command: "shutdown" } });
  }
});
