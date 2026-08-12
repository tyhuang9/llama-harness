import { createRequire } from "node:module";
import { basename, join } from "node:path";
import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { randomUUID } from "node:crypto";

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export interface ToolDefinition {
  id: string;
  name: string;
  description: string;
  argumentsSchema: Json;
  risk: "low" | "medium" | "high";
  idempotent: boolean;
  readOnly: boolean;
}

export interface HarnessTool<TArguments extends Json = Json> extends ToolDefinition {
  execute(toolArguments: TArguments, context: ToolCallbackContext): Promise<Json> | Json;
}

export interface ToolCallbackContext {
  runId: string;
  traceId: string;
  callId: string;
  deadlineMs?: number;
}

export interface PolicyRequest {
  runId: string;
  traceId: string;
  callId: string;
  tool: ToolDefinition;
  arguments: Json;
  deadlineMs?: number;
}

export type PolicyDecision =
  | { outcome: "allow"; reason: string }
  | { outcome: "deny"; reason: string }
  | { outcome: "require_approval"; reason: string };

export interface ApprovalRequest extends PolicyRequest {}
export type ApprovalDecision = { granted: boolean; reason: string };

export interface AgentDefinition {
  id: string;
  name: string;
  version: string;
  instructions?: string;
  defaultModel: string;
  toolAllowlist?: string[];
  limits?: Partial<AgentLimits>;
  generation?: GenerationOptions;
  outputSchema?: Json;
  metadata?: Record<string, Json>;
}

export interface AgentLimits {
  maxModelCalls: number;
  maxToolCalls: number;
  maxIdenticalToolCalls: number;
  maxRunDurationMs?: number;
  maxModelCallDurationMs?: number;
  maxOutputRepairs: number;
  maxProviderRetries: number;
  maxInputBytes: number;
  maxRequestPayloadBytes: number;
  maxModelResponseBytes: number;
  maxToolArgumentsBytes: number;
  maxToolResultBytes: number;
  maxTranscriptBytes: number;
  maxJsonDepth: number;
}

export interface GenerationOptions { temperature?: number; topP?: number; maxOutputTokens?: number; }
export interface OllamaProvider { kind: "ollama"; baseUrl?: string; }
export interface RunOptions {
  agent: AgentDefinition;
  input: string;
  tools?: HarnessTool[];
  policy?: (request: PolicyRequest) => Promise<PolicyDecision> | PolicyDecision;
  approve?: (request: ApprovalRequest) => Promise<ApprovalDecision> | ApprovalDecision;
  applicationContext?: Record<string, Json>;
  metadata?: Record<string, Json>;
  evaluation?: Record<string, Json>;
  model?: string;
  generation?: GenerationOptions;
}

export interface RunEvent { type: string; sequence: number; timestampMs: number; traceId: string; event: Record<string, Json>; }
export interface RunResult { status: "completed" | "failed" | "cancelled" | "limit_reached"; finalOutput?: string; model: string; traceId: string; durationMs: number; [key: string]: Json | undefined; }
export interface ProviderHealth { healthy: boolean; detail?: string; }
export interface ProviderModel { id: string; capabilities: { supportsTools: boolean; supportsStreaming: boolean; supportsStructuredOutput: boolean; }; }
export interface HarnessClientOptions { provider: OllamaProvider; runtimePath?: string; runtimeArgs?: string[]; onStderr?: (line: string) => void; }

export class HarnessError extends Error {}
export class RuntimeUnavailableError extends HarnessError {}
export class RuntimeProtocolError extends HarnessError {}
export class RuntimeExitedError extends HarnessError {}
export class RunCancelledError extends HarnessError {}

export function defineTool<TArguments extends Json = Json>(definition: HarnessTool<TArguments>): HarnessTool<TArguments> {
  validateTool(definition);
  return definition;
}

export class HarnessClient {
  private readonly process: ChildProcessWithoutNullStreams;
  private readonly options: HarnessClientOptions;
  private readonly pending = new Map<string, Deferred<Envelope>>();
  private readonly runs = new Map<string, ClientRun>();
  private readonly buffered = new Map<string, Envelope[]>();
  private closed = false;

  private constructor(process: ChildProcessWithoutNullStreams, options: HarnessClientOptions) {
    this.process = process;
    this.options = options;
    this.attachListeners();
  }

  static async start(options: HarnessClientOptions): Promise<HarnessClient> {
    const runtimePath = resolveRuntimePath(options.runtimePath);
    const process = spawn(runtimePath, options.runtimeArgs ?? [], { stdio: ["pipe", "pipe", "pipe"], windowsHide: true });
    const client = new HarnessClient(process, options);
    await once(process, "spawn").catch((error) => { throw new RuntimeUnavailableError(`Could not start ${runtimePath}: ${String(error)}`); });
    const hello = client.request({ type: "client_hello", payload: { sdk: { name: "@llama-harness/sdk", version: "0.1.0" }, capabilities: ["async_callbacks"] } });
    const response = await hello;
    if (response.type !== "runtime_hello") throw new RuntimeProtocolError(`Expected runtime_hello, received ${response.type}`);
    return client;
  }

  async run(options: RunOptions): Promise<HarnessRun> {
    const tools = options.tools ?? [];
    const payload = {
      provider: toWireProvider(this.options.provider),
      agent: toWireAgent(options.agent),
      input: options.input,
      application_context: options.applicationContext ?? {},
      metadata: options.metadata ?? {},
      evaluation: options.evaluation ?? {},
      overrides: { model: options.model, generation: toWireGeneration(options.generation ?? {}) },
      tools: tools.map(toWireTool),
    };
    const acknowledgement = await this.request({ type: "start_run", payload: { request: payload } });
    if (acknowledgement.type !== "command_acknowledged" || !acknowledgement.run_id) throw new RuntimeProtocolError("Runtime did not acknowledge start_run with a run ID");
    const run = new ClientRun(this, acknowledgement.run_id, tools, options.policy, options.approve);
    this.runs.set(run.id, run);
    for (const event of this.buffered.get(run.id) ?? []) this.handleRunMessage(run, event);
    this.buffered.delete(run.id);
    return new HarnessRun(run);
  }

  async health(): Promise<ProviderHealth> {
    const response = await this.request({ type: "get_provider_health", payload: { provider: toWireProvider(this.options.provider) } });
    if (response.type !== "provider_health") throw new RuntimeProtocolError(`Expected provider_health, received ${response.type}`);
    return { healthy: Boolean(response.payload.healthy), detail: typeof response.payload.detail === "string" ? response.payload.detail : undefined };
  }

  async listModels(): Promise<ProviderModel[]> {
    const response = await this.request({ type: "get_model_inventory", payload: { provider: toWireProvider(this.options.provider) } });
    if (response.type !== "model_inventory") throw new RuntimeProtocolError(`Expected model_inventory, received ${response.type}`);
    const models = response.payload.models;
    if (!Array.isArray(models)) throw new RuntimeProtocolError("Model inventory did not contain models");
    return models.map((value) => {
      const model = value as Record<string, Json>;
      const capabilities = model.capabilities as Record<string, Json>;
      return { id: String(model.id), capabilities: { supportsTools: Boolean(capabilities?.supports_tools), supportsStreaming: Boolean(capabilities?.supports_streaming), supportsStructuredOutput: Boolean(capabilities?.supports_structured_output) } };
    });
  }

  async close(): Promise<void> {
    if (this.closed) return;
    try { await this.request({ type: "shutdown", payload: {} }); } catch { this.process.kill(); }
    finally { this.closed = true; this.process.stdin.end(); }
  }

  async [Symbol.asyncDispose](): Promise<void> { await this.close(); }

  async sendForRun(runId: string, type: string, payload: unknown): Promise<void> { await this.send({ run_id: runId, type, payload }); }

  private attachListeners(): void {
    createInterface({ input: this.process.stdout }).on("line", (line) => {
      try { this.receive(parseEnvelope(line)); } catch (error) { this.failAll(new RuntimeProtocolError(`Runtime stdout corruption: ${String(error)}`)); this.process.kill(); }
    });
    createInterface({ input: this.process.stderr }).on("line", (line) => this.options.onStderr?.(line));
    this.process.once("exit", (code, signal) => { if (!this.closed) this.failAll(new RuntimeExitedError(`Runtime exited (${code ?? signal ?? "unknown"})`)); });
    this.process.once("error", (error) => this.failAll(new RuntimeExitedError(`Runtime process error: ${error.message}`)));
  }

  private receive(envelope: Envelope): void {
    const pending = this.pending.get(envelope.request_id);
    if (pending && ["runtime_hello", "command_acknowledged", "protocol_error", "pong", "provider_health", "model_inventory"].includes(envelope.type)) { this.pending.delete(envelope.request_id); pending.resolve(envelope); return; }
    if (!envelope.run_id) return;
    const run = this.runs.get(envelope.run_id);
    if (!run) { const buffered = this.buffered.get(envelope.run_id) ?? []; buffered.push(envelope); this.buffered.set(envelope.run_id, buffered); return; }
    this.handleRunMessage(run, envelope);
  }

  private handleRunMessage(run: ClientRun, envelope: Envelope): void {
    if (envelope.type === "run_event") { run.emitEvent(envelope.payload); return; }
    if (envelope.type === "tool_execution_requested") { void run.handleTool(envelope.payload); return; }
    if (envelope.type === "policy_decision_requested") { void run.handlePolicy(envelope.payload); return; }
    if (envelope.type === "approval_requested") { void run.handleApproval(envelope.payload); return; }
    if (envelope.type === "run_completed") { run.complete(fromWireResult(envelope.payload.result as Record<string, Json>)); this.runs.delete(run.id); return; }
    if (envelope.type === "run_cancelled") { run.fail(new RunCancelledError(String(envelope.payload.reason ?? "Run cancelled"))); this.runs.delete(run.id); return; }
    if (envelope.type === "run_failed") { run.fail(new HarnessError(String((envelope.payload.error as Record<string, Json>).message ?? "Run failed"))); this.runs.delete(run.id); }
  }

  private request(message: OutgoingMessage): Promise<Envelope> { const requestId = randomUUID(); const deferred = new Deferred<Envelope>(); this.pending.set(requestId, deferred); void this.send({ ...message, request_id: requestId }).catch((error) => { this.pending.delete(requestId); deferred.reject(error); }); return deferred.promise; }
  private async send(message: OutgoingMessage & { request_id?: string }): Promise<void> { if (this.closed) throw new RuntimeExitedError("Harness client is closed"); const envelope = { protocol_version: "1.0", request_id: message.request_id ?? randomUUID(), run_id: message.run_id, type: message.type, payload: message.payload }; const line = `${JSON.stringify(envelope)}\n`; if (!this.process.stdin.write(line)) await once(this.process.stdin, "drain"); }
  private failAll(error: Error): void { for (const pending of this.pending.values()) pending.reject(error); this.pending.clear(); for (const run of this.runs.values()) run.fail(error); this.runs.clear(); }
}

export class HarnessRun {
  constructor(private readonly inner: ClientRun) {}
  get id(): string { return this.inner.id; }
  events(): AsyncIterable<RunEvent> { return this.inner.events(); }
  result(): Promise<RunResult> { return this.inner.result; }
  cancel(reason = "cancelled by SDK host"): Promise<void> { return this.inner.cancel(reason); }
}

class ClientRun {
  readonly result: Promise<RunResult>;
  private readonly deferred = new Deferred<RunResult>();
  private readonly queue = new AsyncQueue<RunEvent>();
  private readonly toolMap = new Map<string, HarnessTool>();
  constructor(private readonly client: HarnessClient, readonly id: string, tools: HarnessTool[], private readonly policy?: RunOptions["policy"], private readonly approve?: RunOptions["approve"]) { this.result = this.deferred.promise; for (const tool of tools) this.toolMap.set(tool.id, tool); }
  events(): AsyncIterable<RunEvent> { return this.queue; }
  async cancel(reason: string): Promise<void> { await this.client.sendForRun(this.id, "cancel_run", { reason }); }
  emitEvent(payload: Record<string, Json>): void { this.queue.push({ type: String((payload.event as Record<string, Json>).type ?? "unknown"), sequence: Number(payload.sequence), timestampMs: Number(payload.timestamp_ms), traceId: String(payload.trace_id), event: payload.event as Record<string, Json> }); }
  complete(result: RunResult): void { this.queue.close(); this.deferred.resolve(result); }
  fail(error: Error): void { this.queue.fail(error); this.deferred.reject(error); }
  async handleTool(payload: Record<string, Json>): Promise<void> { const callbackId = String(payload.callback_id); const tool = this.toolMap.get(String((payload.tool as Record<string, Json>).id)); let result: Json; try { if (!tool) throw new HarnessError(`No host tool registered for ${String((payload.tool as Record<string, Json>).id)}`); result = await tool.execute(payload.arguments as Json, { runId: this.id, traceId: String(payload.trace_id), callId: String(payload.call_id), deadlineMs: optionalNumber(payload.deadline_ms) }); await this.client.sendForRun(this.id, "tool_result", { callback_id: callbackId, result: { ok: true, output: result } }); } catch (error) { await this.client.sendForRun(this.id, "tool_result", { callback_id: callbackId, result: { ok: false, output: null, error: String(error) } }); } }
  async handlePolicy(payload: Record<string, Json>): Promise<void> { const tool = fromWireTool(payload.tool as Record<string, Json>); const request = callbackRequest(this.id, payload, tool); try { const decision = this.policy ? await this.policy(request) : tool.readOnly ? { outcome: "allow", reason: "read-only tool allowed by SDK default policy" } : { outcome: "deny", reason: "state-changing tool requires an explicit policy" }; await this.client.sendForRun(this.id, "policy_decision", { callback_id: payload.callback_id, decision }); } catch (error) { await this.client.sendForRun(this.id, "policy_decision", { callback_id: payload.callback_id, decision: { outcome: "deny", reason: `SDK policy handler failed: ${String(error)}` } }); } }
  async handleApproval(payload: Record<string, Json>): Promise<void> { const tool = fromWireTool(payload.tool as Record<string, Json>); const request = callbackRequest(this.id, payload, tool); try { const decision = this.approve ? await this.approve(request) : { granted: false, reason: "no SDK approval handler configured" }; await this.client.sendForRun(this.id, "approval_decision", { callback_id: payload.callback_id, ...decision }); } catch (error) { await this.client.sendForRun(this.id, "approval_decision", { callback_id: payload.callback_id, granted: false, reason: `SDK approval handler failed: ${String(error)}` }); } }
}

type Envelope = { protocol_version: string; request_id: string; run_id?: string; type: string; payload: Record<string, Json>; };
type OutgoingMessage = { request_id?: string; run_id?: string; type: string; payload: unknown; };
class Deferred<T> { promise: Promise<T>; resolve!: (value: T) => void; reject!: (reason?: unknown) => void; constructor() { this.promise = new Promise<T>((resolve, reject) => { this.resolve = resolve; this.reject = reject; }); } }
class AsyncQueue<T> implements AsyncIterable<T> { private values: T[] = []; private waiters: Array<{ resolve: (value: IteratorResult<T>) => void; reject: (error: unknown) => void }> = []; private closed = false; private error?: Error; push(value: T): void { const waiter = this.waiters.shift(); if (waiter) waiter.resolve({ value, done: false }); else this.values.push(value); } close(): void { this.closed = true; for (const waiter of this.waiters.splice(0)) waiter.resolve({ value: undefined as never, done: true }); } fail(error: Error): void { this.error = error; for (const waiter of this.waiters.splice(0)) waiter.reject(error); } async next(): Promise<IteratorResult<T>> { if (this.values.length) return { value: this.values.shift()!, done: false }; if (this.error) throw this.error; if (this.closed) return { value: undefined as never, done: true }; return new Promise((resolve, reject) => this.waiters.push({ resolve, reject })); } [Symbol.asyncIterator](): AsyncIterator<T> { return this; } }

export function resolveRuntimePath(explicitPath?: string): string { if (explicitPath) return explicitPath; if (process.env.LLAMA_HARNESS_RUNTIME_PATH) return process.env.LLAMA_HARNESS_RUNTIME_PATH; const target = `${process.platform}-${process.arch}`; const packageName = `@llama-harness/runtime-${target}`; try { const require = createRequire(import.meta.url); const manifest = require.resolve(`${packageName}/package.json`); return join(manifest, "..", "bin", process.platform === "win32" ? "llama-harness-runtime.exe" : "llama-harness-runtime"); } catch { throw new RuntimeUnavailableError(`No packaged runtime for ${target}. Install ${packageName} or set LLAMA_HARNESS_RUNTIME_PATH explicitly.`); } }
function parseEnvelope(line: string): Envelope { const value = JSON.parse(line) as Envelope; if (value.protocol_version.split(".")[0] !== "1" || !value.request_id || !value.type || typeof value.payload !== "object") throw new RuntimeProtocolError("Malformed protocol envelope"); return value; }
function once(target: NodeJS.EventEmitter, event: string): Promise<unknown> { return new Promise((resolve, reject) => { target.once(event, resolve); target.once("error", reject); }); }
function optionalNumber(value: Json | undefined): number | undefined { return typeof value === "number" ? value : undefined; }
function callbackRequest(runId: string, payload: Record<string, Json>, tool: ToolDefinition): PolicyRequest { return { runId, traceId: String(payload.trace_id), callId: String(payload.call_id), tool, arguments: payload.arguments as Json, deadlineMs: optionalNumber(payload.deadline_ms) }; }
function validateTool(tool: HarnessTool): void { if (!tool.id.trim() || !tool.name.trim() || !tool.description.trim()) throw new HarnessError("Tool id, name, and description are required"); }
function toWireTool(tool: ToolDefinition): Record<string, unknown> { return { id: tool.id, name: tool.name, description: tool.description, arguments_schema: tool.argumentsSchema, risk: tool.risk, idempotent: tool.idempotent, read_only: tool.readOnly }; }
function toWireProvider(provider: OllamaProvider): Record<string, string> { return { kind: "ollama", base_url: provider.baseUrl ?? "http://127.0.0.1:11434" }; }
function fromWireTool(tool: Record<string, Json>): ToolDefinition { return { id: String(tool.id), name: String(tool.name), description: String(tool.description), argumentsSchema: tool.arguments_schema ?? {}, risk: String(tool.risk) as ToolDefinition["risk"], idempotent: Boolean(tool.idempotent), readOnly: Boolean(tool.read_only) }; }
function toWireGeneration(generation: GenerationOptions): Record<string, unknown> { return compact({ temperature: generation.temperature, top_p: generation.topP, max_output_tokens: generation.maxOutputTokens }); }
function toWireAgent(agent: AgentDefinition): Record<string, unknown> { const defaults: AgentLimits = { maxModelCalls: 8, maxToolCalls: 16, maxIdenticalToolCalls: 2, maxOutputRepairs: 1, maxProviderRetries: 2, maxInputBytes: 65536, maxRequestPayloadBytes: 262144, maxModelResponseBytes: 1048576, maxToolArgumentsBytes: 65536, maxToolResultBytes: 1048576, maxTranscriptBytes: 4194304, maxJsonDepth: 64 }; const limits = { ...defaults, ...agent.limits }; return compact({ id: agent.id, name: agent.name, version: agent.version, system_instructions: agent.instructions ?? "", default_model: agent.defaultModel, tool_allowlist: agent.toolAllowlist ?? [], limits: compact({ max_model_calls: limits.maxModelCalls, max_tool_calls: limits.maxToolCalls, max_identical_tool_calls: limits.maxIdenticalToolCalls, max_run_duration_ms: limits.maxRunDurationMs, max_model_call_duration_ms: limits.maxModelCallDurationMs, max_output_repairs: limits.maxOutputRepairs, max_provider_retries: limits.maxProviderRetries, max_input_bytes: limits.maxInputBytes, max_request_payload_bytes: limits.maxRequestPayloadBytes, max_model_response_bytes: limits.maxModelResponseBytes, max_tool_arguments_bytes: limits.maxToolArgumentsBytes, max_tool_result_bytes: limits.maxToolResultBytes, max_transcript_bytes: limits.maxTranscriptBytes, max_json_depth: limits.maxJsonDepth }), generation: toWireGeneration(agent.generation ?? {}), output_schema: agent.outputSchema, metadata: agent.metadata ?? {} }); }
function compact(value: Record<string, unknown>): Record<string, unknown> { return Object.fromEntries(Object.entries(value).filter(([, entry]) => entry !== undefined)); }
function fromWireResult(result: Record<string, Json>): RunResult { return { ...result, status: String(result.status) as RunResult["status"], finalOutput: typeof result.final_output === "string" ? result.final_output : undefined, model: String(result.model), traceId: String(result.trace_id), durationMs: Number(result.duration_ms) }; }
