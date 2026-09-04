import { createInterface } from "node:readline";
import { writeFileSync } from "node:fs";

function write(envelope: object): void { process.stdout.write(`${JSON.stringify(envelope)}\n`); }
const mode = process.argv[2] ?? "legacy";
const selectedVersion = mode === "modern" || mode === "drift" || mode === "protocol_error" || mode === "version_mismatch" || mode === "malformed_metadata" || mode === "hello_error" || mode === "hello_error_unresponsive" ? "1.1" : mode === "incompatible" ? "2.0" : "1.0";
const shutdownMarker = process.argv[3];
if (mode === "hello_error_unresponsive" && shutdownMarker) writeFileSync(shutdownMarker, String(process.pid), "utf8");

createInterface({ input: process.stdin }).on("line", (line) => {
  const message = JSON.parse(line) as { protocol_version: string; request_id: string; type: string; payload: Record<string, unknown>; };
  if (message.type === "client_hello") {
    if (mode === "hello_error" || mode === "hello_error_unresponsive") { write({ protocol_version: selectedVersion, request_id: message.request_id, type: "protocol_error", payload: { code: "invalid_state", message: "hello rejected", retryable: false } }); return; }
    if ((message.payload.sdk as { version?: string } | undefined)?.version !== "0.2.0") { write({ protocol_version: selectedVersion, request_id: message.request_id, type: "protocol_error", payload: { code: "invalid_message", message: "SDK identity version mismatch", retryable: false } }); return; }
    write({ protocol_version: selectedVersion, request_id: message.request_id, type: "runtime_hello", payload: { runtime_version: mode === "version_mismatch" ? "0.1.0" : "0.2.0", capabilities: { supports_output_deltas: false, supports_structured_output: true, supports_trace_persistence: false, concurrent_runs: 1, max_pending_callbacks: 1, max_queue_depth: 8 }, providers: ["ollama"] } });
  } else if (message.protocol_version !== selectedVersion) {
    write({ protocol_version: selectedVersion, request_id: message.request_id, type: "protocol_error", payload: { code: "incompatible_version", message: "client version drift", retryable: false } });
  } else if (message.type === "get_provider_health") {
    write({ protocol_version: selectedVersion, request_id: message.request_id, type: "provider_health", payload: { healthy: true, detail: "fake runtime" } });
  } else if (message.type === "get_model_inventory") {
    write({ protocol_version: selectedVersion, request_id: message.request_id, type: "model_inventory", payload: { models: [{ id: "mock", capabilities: { supports_tools: true, supports_streaming: false, supports_structured_output: true } }] } });
  } else if (message.type === "start_run") {
    const request = message.payload.request as Record<string, unknown>;
    const firstTool = Array.isArray(request?.tools) ? request.tools[0] as Record<string, unknown> | undefined : undefined;
    if (mode === "modern" && (request?.strategy !== "programmatic" || !firstTool || firstTool.output_schema === undefined || firstTool.parallel_safe !== true || firstTool.execution_location !== "local_private")) { write({ protocol_version: selectedVersion, request_id: message.request_id, type: "protocol_error", payload: { code: "invalid_message", message: "1.1 metadata was not serialized", retryable: false } }); return; }
    if (mode === "legacy" && ((firstTool && ("output_schema" in firstTool || "parallel_safe" in firstTool)) || "strategy" in request)) { write({ protocol_version: selectedVersion, request_id: message.request_id, type: "protocol_error", payload: { code: "invalid_message", message: "legacy request included 1.1 metadata", retryable: false } }); return; }
    if (mode === "protocol_error") { write({ protocol_version: selectedVersion, request_id: message.request_id, type: "protocol_error", payload: { code: "invalid_state", message: "test protocol failure", retryable: false } }); return; }
    write({ protocol_version: selectedVersion, request_id: message.request_id, run_id: "test-run", type: "command_acknowledged", payload: { command: "start_run" } });
    if (mode === "malformed_metadata") { write({ protocol_version: selectedVersion, request_id: "policy-request", run_id: "test-run", type: "policy_decision_requested", payload: { callback_id: "policy-1", trace_id: "test-trace", call_id: "call-1", tool: { id: "notes.search", name: "Search", description: "Search", arguments_schema: {}, risk: "invalid", idempotent: "true", read_only: "true", parallel_safe: "true", cancellation_safety: "invalid", allowed_callers: ["invalid"], speculation_policy: "invalid", issue_safety: "invalid", execution_location: "invalid", network_egress: "invalid" }, arguments: {} } }); return; }
    if (mode === "drift") { setTimeout(() => write({ protocol_version: "1.0", request_id: "drift", run_id: "test-run", type: "run_event", payload: { trace_id: "test-trace", sequence: 1, timestamp_ms: 1, event: { type: "model_responded", call_number: 1 } } }), 20); return; }
    if (mode === "modern") { write({ protocol_version: selectedVersion, request_id: "event-1", run_id: "test-run", type: "run_event", payload: { trace_id: "test-trace", sequence: 1, timestamp_ms: 1, event: { type: "strategy_usage", strategy: "programmatic", model_calls: 1, planning_model_calls: 0, repair_model_calls: 0, recovery_model_calls: 0, final_synthesis_model_calls: 1, reactive_model_calls: 0, tool_calls: 0, tool_issued: 0, tool_reused: 0, tool_rejected: 0, tool_pre_dispatch_aborted: 0, tool_completed: 0, tool_failed: 0, tool_cancelled: 0, duration_ms: 1 } } }); write({ protocol_version: selectedVersion, request_id: "complete-1", run_id: "test-run", type: "run_completed", payload: { run_sequence: 2, result: { status: "completed", final_output: "modern", model: "mock", trace_id: "test-trace", duration_ms: 1 } } }); return; }
    write({ protocol_version: selectedVersion, request_id: "tool-request", run_id: "test-run", type: "tool_execution_requested", payload: { run_sequence: 1, callback_id: "tool-1", trace_id: "test-trace", call_id: "call-1", tool: { id: "notes.search", name: "Search", description: "Search notes", arguments_schema: {}, risk: "low", idempotent: true, read_only: true }, arguments: { query: "harness" } } });
  } else if (message.type === "tool_result") {
    const result = (message.payload.result as { ok: boolean; output: unknown });
    write({ protocol_version: selectedVersion, request_id: "event-1", run_id: "test-run", type: "run_event", payload: { trace_id: "test-trace", sequence: 2, timestamp_ms: 1, event: { type: "tool_completed", call_id: "call-1", tool_id: "notes.search", ok: result.ok } } });
    write({ protocol_version: selectedVersion, request_id: "complete-1", run_id: "test-run", type: "run_completed", payload: { run_sequence: 3, result: { status: "completed", final_output: JSON.stringify(result.output), model: "mock", trace_id: "test-trace", duration_ms: 1 } } });
  } else if (message.type === "policy_decision" && mode === "malformed_metadata") {
    write({ protocol_version: selectedVersion, request_id: "complete-1", run_id: "test-run", type: "run_completed", payload: { run_sequence: 2, result: { status: "completed", final_output: "denied", model: "mock", trace_id: "test-trace", duration_ms: 1 } } });
  } else if (message.type === "shutdown") {
    if (mode === "hello_error_unresponsive") return;
    if (shutdownMarker) writeFileSync(shutdownMarker, "closed\n", "utf8");
    write({ protocol_version: selectedVersion, request_id: message.request_id, type: "command_acknowledged", payload: { command: "shutdown" } });
  }
});
