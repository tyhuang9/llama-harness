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
  /** 1.1 safety metadata. These defaults intentionally make no extra safety claims. */
  outputSchema?: Json;
  parallelSafe?: boolean;
  concurrencyKey?: string;
  cancellationSafety?: CancellationSafety;
  expectedLatencyMs?: number;
  allowedCallers?: ToolCaller[];
  speculationPolicy?: SpeculationPolicy;
  issueSafety?: IssueSafety;
  executionLocation?: ExecutionLocation;
  networkEgress?: NetworkEgress;
}

export type RunStrategy = "adaptive" | "direct" | "declarative_plan" | "programmatic";
export type CancellationSafety = "unknown" | "cooperative" | "guaranteed";
export type ToolCaller = "direct" | "declarative_plan" | "programmatic" | "speculative";
export type SpeculationPolicy = "disabled" | "enabled";
export type IssueSafety = "unknown" | "guaranteed";
export type ExecutionLocation = "unknown" | "local_private" | "remote";
export type NetworkEgress = "unknown" | "prohibited" | "permitted";

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
  strategy?: RunStrategy;
}

export type RunEventPayload =
  | { type: "started"; trace_id: string }
  | { type: "model_requested"; call_number: number; model: string }
  | { type: "model_retrying"; next_call_number: number; reason: string }
  | { type: "model_responded"; call_number: number }
  | { type: "tool_discovery_completed"; caller: ToolCaller; outcome: string; selection: string; candidate_count: number; selected_count: number; deferred_candidate_count: number; effective_tool_count_budget: number; effective_schema_byte_budget: number; selected_schema_bytes: number; expansion_count: number; expansion_limit: number; catalog_exceeded_budget: boolean; duration_ms: number }
  | { type: "strategy_selected"; requested: RunStrategy; selected: RunStrategy; reason: string }
  | { type: "strategy_fallback"; from: RunStrategy; to: RunStrategy; reason: string }
  | { type: "plan_lifecycle"; phase: string; attempt: number; outcome: string }
  | { type: "plan_validated"; attempt: number; node_count: number }
  | { type: "program_lifecycle"; attempt: number; outcome: string }
  | { type: "program_validated"; attempt: number; statement_count: number; instruction_count: number }
  | { type: "program_execution_completed"; attempt: number; fuel_used: number; scheduling_slices: number; tool_yields: number; branches: number; loop_iterations: number; fanout_batches: number; partial_failures: number; peak_accounted_bytes: number; duration_ms: number }
  | { type: "plan_node_started"; node_id: string; tool_id: string; attempt: number; wave: number }
  | { type: "plan_node_completed"; node_id: string; tool_id: string; attempt: number; wave: number; ok: boolean; outcome: string; duration_ms: number }
  | { type: "tool_effect_reused"; call_id: string; tool_id: string }
  | { type: "strategy_usage"; strategy: RunStrategy; model_calls: number; planning_model_calls: number; repair_model_calls: number; recovery_model_calls: number; final_synthesis_model_calls: number; reactive_model_calls: number; tool_calls: number; tool_issued: number; tool_reused: number; tool_rejected: number; tool_pre_dispatch_aborted: number; tool_completed: number; tool_failed: number; tool_cancelled: number; duration_ms: number }
  | { type: "tool_rejected"; call_id: string; tool_id: string; reason: string }
  | { type: "policy_decided"; call_id: string; decision: Record<string, Json> }
  | { type: "approval_requested"; call_id: string; tool_id: string }
  | { type: "tool_completed"; call_id: string; tool_id: string; ok: boolean }
  | { type: "completed"; status: string };
export interface RunEvent { type: RunEventPayload["type"] | "unknown"; sequence: number; timestampMs: number; traceId: string; event: RunEventPayload; }
export interface RunResult { status: "completed" | "failed" | "cancelled" | "limit_reached"; finalOutput?: string; model: string; traceId: string; durationMs: number; [key: string]: Json | undefined; }
export interface ProviderHealth { healthy: boolean; detail?: string; }
export type ProgrammaticConformance = "strict_json_ast_v1";
export interface ProviderModelCapabilities { supportsTools: boolean; supportsStreaming: boolean; supportsStructuredOutput: boolean; supportsStrictToolSchemas?: boolean; supportsStreamingToolArguments?: boolean; supportsParallelToolCalls?: boolean; supportsStructuredPlans?: boolean; supportsProgrammaticCalling?: boolean; programmaticConformance?: ProgrammaticConformance; limits?: ProviderModelLimits; }
export interface ProviderModelLimits { maxTools?: number; maxToolSchemaBytes?: number; maxParallelToolCalls?: number; maxStreamedArgumentBytes?: number; maxStreamedToolCalls?: number; maxPlanBytes?: number; maxPlanNodes?: number; maxProgramBytes?: number; }
export interface ProviderModel { id: string; capabilities: ProviderModelCapabilities; }
export interface HarnessClientOptions { provider: OllamaProvider; runtimePath?: string; runtimeArgs?: string[]; onStderr?: (line: string) => void; }

export class HarnessError extends Error {}
export class RuntimeUnavailableError extends HarnessError {}
export class RuntimeProtocolError extends HarnessError {
  constructor(message: string, readonly code?: string, readonly retryable?: boolean) { super(message); this.name = "RuntimeProtocolError"; }
}
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
  private _negotiatedProtocolVersion?: ProtocolVersion;
  private terminalError?: Error;
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
    try {
      const response = await client.request({ type: "client_hello", payload: { sdk: { name: "@llama-harness/sdk", version: SDK_VERSION }, capabilities: ["async_callbacks"] } });
      if (response.type !== "runtime_hello") throw new RuntimeProtocolError(`Expected runtime_hello, received ${response.type}`);
      client._negotiatedProtocolVersion = selectedProtocolVersion(response.protocol_version);
      const runtimeVersion = response.payload.runtime_version;
      if (runtimeVersion !== SDK_VERSION) throw new RuntimeProtocolError(`Runtime version mismatch: SDK ${SDK_VERSION}, runtime ${String(runtimeVersion)}`);
      return client;
    }
    catch (error) { await client.abortStartup(); throw error; }
  }

  /** Exact protocol version selected by the runtime hello envelope. */
  get negotiatedProtocolVersion(): string | undefined { return this._negotiatedProtocolVersion; }

  async run(options: RunOptions): Promise<HarnessRun> {
    if (this.terminalError) throw this.terminalError;
    const strategy = options.strategy;
    if (strategy !== undefined && this._negotiatedProtocolVersion !== "1.1") {
      throw new RuntimeProtocolError(`explicit strategy ${strategy} requires negotiated protocol version 1.1`);
    }
    const tools = options.tools ?? [];
    const payload = {
      provider: toWireProvider(this.options.provider),
      agent: toWireAgent(options.agent),
      input: options.input,
      application_context: options.applicationContext ?? {},
      metadata: options.metadata ?? {},
      evaluation: options.evaluation ?? {},
      overrides: { model: options.model, generation: toWireGeneration(options.generation ?? {}) },
      tools: tools.map((tool) => toWireTool(tool, this._negotiatedProtocolVersion === "1.1")),
      ...(this._negotiatedProtocolVersion === "1.1" && strategy ? { strategy } : {}),
    };
    const acknowledgement = await this.request({ type: "start_run", payload: { request: payload } });
    if (this.terminalError) throw this.terminalError;
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
      const limits = capabilities?.limits as Record<string, Json> | undefined;
      const providerModel: ProviderModel = { id: String(model.id), capabilities: { supportsTools: Boolean(capabilities?.supports_tools), supportsStreaming: Boolean(capabilities?.supports_streaming), supportsStructuredOutput: Boolean(capabilities?.supports_structured_output) } };
      const strictToolSchemas = optionalBoolean(capabilities?.supports_strict_tool_schemas); if (strictToolSchemas !== undefined) providerModel.capabilities.supportsStrictToolSchemas = strictToolSchemas;
      const streamingToolArguments = optionalBoolean(capabilities?.supports_streaming_tool_arguments); if (streamingToolArguments !== undefined) providerModel.capabilities.supportsStreamingToolArguments = streamingToolArguments;
      const parallelToolCalls = optionalBoolean(capabilities?.supports_parallel_tool_calls); if (parallelToolCalls !== undefined) providerModel.capabilities.supportsParallelToolCalls = parallelToolCalls;
      const structuredPlans = optionalBoolean(capabilities?.supports_structured_plans); if (structuredPlans !== undefined) providerModel.capabilities.supportsStructuredPlans = structuredPlans;
      const programmaticCalling = optionalBoolean(capabilities?.supports_programmatic_calling); if (programmaticCalling !== undefined) providerModel.capabilities.supportsProgrammaticCalling = programmaticCalling;
      if (capabilities?.programmatic_conformance === "strict_json_ast_v1") providerModel.capabilities.programmaticConformance = "strict_json_ast_v1";
      if (limits) providerModel.capabilities.limits = compact({ maxTools: optionalNumber(limits.max_tools), maxToolSchemaBytes: optionalNumber(limits.max_tool_schema_bytes), maxParallelToolCalls: optionalNumber(limits.max_parallel_tool_calls), maxStreamedArgumentBytes: optionalNumber(limits.max_streamed_argument_bytes), maxStreamedToolCalls: optionalNumber(limits.max_streamed_tool_calls), maxPlanBytes: optionalNumber(limits.max_plan_bytes), maxPlanNodes: optionalNumber(limits.max_plan_nodes), maxProgramBytes: optionalNumber(limits.max_program_bytes) }) as ProviderModelLimits;
      return providerModel;
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

  private async abortStartup(): Promise<void> {
    this.closed = true;
    this.process.stdin.destroy();
    if (this.process.exitCode !== null || this.process.signalCode !== null) return;
    let exited = false;
    const exit = once(this.process, "exit").then(() => { exited = true; }, () => { exited = true; });
    this.process.kill();
    await Promise.race([exit, delay(250)]);
    if (!exited && this.process.exitCode === null && this.process.signalCode === null) {
      this.process.kill("SIGKILL");
      await Promise.race([exit, delay(250)]);
    }
  }

  private receive(envelope: Envelope): void {
    if (this._negotiatedProtocolVersion && envelope.protocol_version !== this._negotiatedProtocolVersion) {
      this.terminalError = new RuntimeProtocolError(`Runtime protocol version drift: expected ${this._negotiatedProtocolVersion}, received ${envelope.protocol_version}`);
      this.failAll(this.terminalError);
      this.process.kill();
      return;
    }
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
    if (envelope.type === "protocol_error") { run.fail(protocolError(envelope.payload)); this.runs.delete(run.id); }
  }

  private request(message: OutgoingMessage): Promise<Envelope> { const requestId = randomUUID(); const deferred = new Deferred<Envelope>(); this.pending.set(requestId, deferred); void this.send({ ...message, request_id: requestId }).catch((error) => { this.pending.delete(requestId); deferred.reject(error); }); return deferred.promise.then((response) => { if (response.type === "protocol_error") throw protocolError(response.payload); return response; }); }
  private async send(message: OutgoingMessage & { request_id?: string }): Promise<void> { if (this.closed) throw new RuntimeExitedError("Harness client is closed"); const envelope = { protocol_version: this._negotiatedProtocolVersion ?? OFFERED_PROTOCOL_VERSION, request_id: message.request_id ?? randomUUID(), run_id: message.run_id, type: message.type, payload: message.payload }; const line = `${JSON.stringify(envelope)}\n`; if (!this.process.stdin.write(line)) await once(this.process.stdin, "drain"); }
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
  emitEvent(payload: Record<string, Json>): void { const event = payload.event as unknown as RunEventPayload; this.queue.push({ type: typeof event.type === "string" ? event.type : "unknown", sequence: Number(payload.sequence), timestampMs: Number(payload.timestamp_ms), traceId: String(payload.trace_id), event }); }
  complete(result: RunResult): void { this.queue.close(); this.deferred.resolve(result); }
  fail(error: Error): void { this.queue.fail(error); this.deferred.reject(error); }
  async handleTool(payload: Record<string, Json>): Promise<void> { const callbackId = String(payload.callback_id); const tool = this.toolMap.get(String((payload.tool as Record<string, Json>).id)); let result: Json; try { if (!tool) throw new HarnessError(`No host tool registered for ${String((payload.tool as Record<string, Json>).id)}`); result = await tool.execute(payload.arguments as Json, { runId: this.id, traceId: String(payload.trace_id), callId: String(payload.call_id), deadlineMs: optionalNumber(payload.deadline_ms) }); await this.client.sendForRun(this.id, "tool_result", { callback_id: callbackId, result: { ok: true, output: result } }); } catch (error) { await this.client.sendForRun(this.id, "tool_result", { callback_id: callbackId, result: { ok: false, output: null, error: String(error) } }); } }
  async handlePolicy(payload: Record<string, Json>): Promise<void> { const tool = fromWireTool(payload.tool as Record<string, Json>); const request = callbackRequest(this.id, payload, tool); try { const decision = this.policy ? await this.policy(request) : tool.readOnly ? { outcome: "allow", reason: "read-only tool allowed by SDK default policy" } : { outcome: "deny", reason: "state-changing tool requires an explicit policy" }; await this.client.sendForRun(this.id, "policy_decision", { callback_id: payload.callback_id, decision }); } catch (error) { await this.client.sendForRun(this.id, "policy_decision", { callback_id: payload.callback_id, decision: { outcome: "deny", reason: `SDK policy handler failed: ${String(error)}` } }); } }
  async handleApproval(payload: Record<string, Json>): Promise<void> { const tool = fromWireTool(payload.tool as Record<string, Json>); const request = callbackRequest(this.id, payload, tool); try { const decision = this.approve ? await this.approve(request) : { granted: false, reason: "no SDK approval handler configured" }; await this.client.sendForRun(this.id, "approval_decision", { callback_id: payload.callback_id, ...decision }); } catch (error) { await this.client.sendForRun(this.id, "approval_decision", { callback_id: payload.callback_id, granted: false, reason: `SDK approval handler failed: ${String(error)}` }); } }
}

const SDK_VERSION = readSdkVersion();
const OFFERED_PROTOCOL_VERSION = "1.1";
type ProtocolVersion = "1.0" | "1.1";
type Envelope = { protocol_version: string; request_id: string; run_id?: string; type: string; payload: Record<string, Json>; };
type OutgoingMessage = { request_id?: string; run_id?: string; type: string; payload: unknown; };
class Deferred<T> { promise: Promise<T>; resolve!: (value: T) => void; reject!: (reason?: unknown) => void; constructor() { this.promise = new Promise<T>((resolve, reject) => { this.resolve = resolve; this.reject = reject; }); } }
class AsyncQueue<T> implements AsyncIterable<T> { private values: T[] = []; private waiters: Array<{ resolve: (value: IteratorResult<T>) => void; reject: (error: unknown) => void }> = []; private closed = false; private error?: Error; push(value: T): void { const waiter = this.waiters.shift(); if (waiter) waiter.resolve({ value, done: false }); else this.values.push(value); } close(): void { this.closed = true; for (const waiter of this.waiters.splice(0)) waiter.resolve({ value: undefined as never, done: true }); } fail(error: Error): void { this.error = error; for (const waiter of this.waiters.splice(0)) waiter.reject(error); } async next(): Promise<IteratorResult<T>> { if (this.values.length) return { value: this.values.shift()!, done: false }; if (this.error) throw this.error; if (this.closed) return { value: undefined as never, done: true }; return new Promise((resolve, reject) => this.waiters.push({ resolve, reject })); } [Symbol.asyncIterator](): AsyncIterator<T> { return this; } }

export function resolveRuntimePath(explicitPath?: string): string { if (explicitPath) return explicitPath; if (process.env.LLAMA_HARNESS_RUNTIME_PATH) return process.env.LLAMA_HARNESS_RUNTIME_PATH; const target = `${process.platform}-${process.arch}`; const packageName = `@llama-harness/runtime-${target}`; try { const require = createRequire(import.meta.url); const manifest = require.resolve(`${packageName}/package.json`); return join(manifest, "..", "bin", process.platform === "win32" ? "llama-harness-runtime.exe" : "llama-harness-runtime"); } catch { throw new RuntimeUnavailableError(`No packaged runtime for ${target}. Install ${packageName} or set LLAMA_HARNESS_RUNTIME_PATH explicitly.`); } }
function parseEnvelope(line: string): Envelope { const value = JSON.parse(line) as Envelope; if (typeof value.protocol_version !== "string" || value.protocol_version.split(".")[0] !== "1" || !value.request_id || !value.type || !value.payload || typeof value.payload !== "object" || Array.isArray(value.payload)) throw new RuntimeProtocolError("Malformed protocol envelope"); return value; }
function selectedProtocolVersion(version: string): ProtocolVersion { if (version === "1.0" || version === "1.1") return version; throw new RuntimeProtocolError(`Unsupported runtime protocol version ${version}`, "incompatible_version", false); }
function protocolError(payload: Record<string, Json>): RuntimeProtocolError { return new RuntimeProtocolError(String(payload.message ?? "Runtime protocol error"), typeof payload.code === "string" ? payload.code : undefined, optionalBoolean(payload.retryable)); }
function once(target: NodeJS.EventEmitter, event: string): Promise<unknown> { return new Promise((resolve, reject) => { target.once(event, resolve); target.once("error", reject); }); }
function delay(milliseconds: number): Promise<void> { return new Promise((resolve) => { const timer = setTimeout(resolve, milliseconds); timer.unref(); }); }
function readSdkVersion(): string { const manifest = createRequire(import.meta.url)("../package.json") as { version?: unknown }; if (typeof manifest.version !== "string" || !manifest.version) throw new RuntimeProtocolError("SDK package metadata does not contain a version"); return manifest.version; }
function optionalNumber(value: Json | undefined): number | undefined { return typeof value === "number" ? value : undefined; }
function optionalBoolean(value: Json | undefined): boolean | undefined { return typeof value === "boolean" ? value : undefined; }
function enumOr<T extends string>(value: Json | undefined, allowed: readonly T[], fallback: T): T { return typeof value === "string" && allowed.includes(value as T) ? value as T : fallback; }
function wireCallers(value: Json | undefined): ToolCaller[] { return Array.isArray(value) ? value.filter((entry): entry is ToolCaller => typeof entry === "string" && (["direct", "declarative_plan", "programmatic", "speculative"] as const).includes(entry as ToolCaller)) : ["direct"]; }
function callbackRequest(runId: string, payload: Record<string, Json>, tool: ToolDefinition): PolicyRequest { return { runId, traceId: String(payload.trace_id), callId: String(payload.call_id), tool, arguments: payload.arguments as Json, deadlineMs: optionalNumber(payload.deadline_ms) }; }
function validateTool(tool: HarnessTool): void { if (!tool.id.trim() || !tool.name.trim() || !tool.description.trim()) throw new HarnessError("Tool id, name, and description are required"); }
function toWireTool(tool: ToolDefinition, includeV11Metadata: boolean): Record<string, unknown> {
  const base = { id: tool.id, name: tool.name, description: tool.description, arguments_schema: tool.argumentsSchema, risk: tool.risk, idempotent: tool.idempotent, read_only: tool.readOnly };
  if (!includeV11Metadata) return base;
  return compact({ ...base, output_schema: tool.outputSchema, parallel_safe: tool.parallelSafe ?? false, concurrency_key: tool.concurrencyKey, cancellation_safety: tool.cancellationSafety ?? "unknown", expected_latency_ms: tool.expectedLatencyMs, allowed_callers: tool.allowedCallers ?? ["direct"], speculation_policy: tool.speculationPolicy ?? "disabled", issue_safety: tool.issueSafety ?? "unknown", execution_location: tool.executionLocation ?? "unknown", network_egress: tool.networkEgress ?? "unknown" });
}
function toWireProvider(provider: OllamaProvider): Record<string, string> { return { kind: "ollama", base_url: provider.baseUrl ?? "http://127.0.0.1:11434" }; }
function fromWireTool(tool: Record<string, Json>): ToolDefinition { return { id: String(tool.id), name: String(tool.name), description: String(tool.description), argumentsSchema: tool.arguments_schema ?? {}, risk: enumOr(tool.risk, ["low", "medium", "high"], "high"), idempotent: optionalBoolean(tool.idempotent) ?? false, readOnly: optionalBoolean(tool.read_only) ?? false, outputSchema: tool.output_schema, parallelSafe: optionalBoolean(tool.parallel_safe) ?? false, concurrencyKey: typeof tool.concurrency_key === "string" ? tool.concurrency_key : undefined, cancellationSafety: enumOr<CancellationSafety>(tool.cancellation_safety, ["unknown", "cooperative", "guaranteed"], "unknown"), expectedLatencyMs: optionalNumber(tool.expected_latency_ms), allowedCallers: wireCallers(tool.allowed_callers), speculationPolicy: enumOr<SpeculationPolicy>(tool.speculation_policy, ["disabled", "enabled"], "disabled"), issueSafety: enumOr<IssueSafety>(tool.issue_safety, ["unknown", "guaranteed"], "unknown"), executionLocation: enumOr<ExecutionLocation>(tool.execution_location, ["unknown", "local_private", "remote"], "unknown"), networkEgress: enumOr<NetworkEgress>(tool.network_egress, ["unknown", "prohibited", "permitted"], "unknown") }; }
function toWireGeneration(generation: GenerationOptions): Record<string, unknown> { return compact({ temperature: generation.temperature, top_p: generation.topP, max_output_tokens: generation.maxOutputTokens }); }
function toWireAgent(agent: AgentDefinition): Record<string, unknown> { const defaults: AgentLimits = { maxModelCalls: 8, maxToolCalls: 16, maxIdenticalToolCalls: 2, maxOutputRepairs: 1, maxProviderRetries: 2, maxInputBytes: 65536, maxRequestPayloadBytes: 262144, maxModelResponseBytes: 1048576, maxToolArgumentsBytes: 65536, maxToolResultBytes: 1048576, maxTranscriptBytes: 4194304, maxJsonDepth: 64 }; const limits = { ...defaults, ...agent.limits }; return compact({ id: agent.id, name: agent.name, version: agent.version, system_instructions: agent.instructions ?? "", default_model: agent.defaultModel, tool_allowlist: agent.toolAllowlist ?? [], limits: compact({ max_model_calls: limits.maxModelCalls, max_tool_calls: limits.maxToolCalls, max_identical_tool_calls: limits.maxIdenticalToolCalls, max_run_duration_ms: limits.maxRunDurationMs, max_model_call_duration_ms: limits.maxModelCallDurationMs, max_output_repairs: limits.maxOutputRepairs, max_provider_retries: limits.maxProviderRetries, max_input_bytes: limits.maxInputBytes, max_request_payload_bytes: limits.maxRequestPayloadBytes, max_model_response_bytes: limits.maxModelResponseBytes, max_tool_arguments_bytes: limits.maxToolArgumentsBytes, max_tool_result_bytes: limits.maxToolResultBytes, max_transcript_bytes: limits.maxTranscriptBytes, max_json_depth: limits.maxJsonDepth }), generation: toWireGeneration(agent.generation ?? {}), output_schema: agent.outputSchema, metadata: agent.metadata ?? {} }); }
function compact(value: Record<string, unknown>): Record<string, unknown> { return Object.fromEntries(Object.entries(value).filter(([, entry]) => entry !== undefined)); }
function fromWireResult(result: Record<string, Json>): RunResult { return { ...result, status: String(result.status) as RunResult["status"], finalOutput: typeof result.final_output === "string" ? result.final_output : undefined, model: String(result.model), traceId: String(result.trace_id), durationMs: Number(result.duration_ms) }; }
