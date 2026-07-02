export type GenerationSettings = {
  temperature: number;
  top_p: number;
  max_tokens: number;
};

export type InstructionSettings = {
  enabled: boolean;
  system_prompt: string;
  tool_context: string;
};

export type LiteLlmSettings = {
  enabled: boolean;
  base_url: string;
  api_key: string | null;
  default_model: string | null;
  timeout_ms: number;
  managed_config_path: string | null;
  allow_unconfigured_models: boolean;
};

export type LiteLlmProviderConfig = {
  id: string;
  enabled: boolean;
  provider_type: string;
  display_name: string;
  api_key_env_var: string;
  api_key: string | null;
  api_base: string | null;
};

export type ModelRoute = {
  id: string;
  enabled: boolean;
  display_name: string;
  provider: string;
  provider_family: "openai" | "anthropic" | "openrouter" | "gemini" | "custom" | string;
  model_alias: string;
  litellm_model: string;
  api_key_env_var: string;
  api_base: string | null;
  notes: string | null;
};

export type AgentPermissions = {
  browser: boolean;
  file_read: boolean;
  file_write: boolean;
  terminal: boolean;
};

export type AgentRecord = {
  id: string;
  name: string;
  role: string;
  description: string;
  system_prompt: string;
  default_model_id: string | null;
  default_provider_id: string;
  default_model: string;
  default_environment: string;
  autonomy: "observe" | "ask" | "low-risk" | "autonomous" | string;
  permissions: AgentPermissions;
  allowed_tool_ids: string[];
  temperature: number | null;
  max_tokens: number | null;
  enabled: boolean;
  status: "active" | "paused" | "draft";
  tasks_run: number;
  updated_at: string;
};

export type AgentCreateRequest = Partial<AgentRecord>;

export type AgentPatch = Partial<
  Pick<
    AgentRecord,
    | "name"
    | "description"
    | "system_prompt"
    | "default_model_id"
    | "default_provider_id"
    | "default_model"
    | "allowed_tool_ids"
    | "temperature"
    | "max_tokens"
    | "enabled"
    | "status"
  >
>;

export type ChatMessage = {
  role: "system" | "user" | "assistant" | string;
  content: string | Record<string, unknown> | unknown[];
};

export type ChatRequest = {
  provider?: string;
  model?: string;
  source_app?: string;
  messages?: ChatMessage[];
  prompt?: string;
  instructions?: string;
  generation?: GenerationSettings;
  tools?: unknown;
  tool_choice?: unknown;
  metadata?: unknown;
};

export type TokenUsage = {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
};

export type ChatResponse = {
  run_id: string;
  provider: string;
  model: string;
  message: ChatMessage;
  usage?: TokenUsage | null;
  started_at: string;
  ended_at: string;
  duration_ms: number;
};

export type ToolCall = {
  id: string;
  type: "function" | string;
  function: {
    name: string;
    arguments: string | Record<string, unknown>;
  };
};

export type AgentChatRequest = {
  messages?: ChatMessage[];
  prompt?: string;
  instructions?: string;
  app_context?: unknown;
  generation?: GenerationSettings;
  tools?: unknown;
  tool_choice?: unknown;
  source_app?: string;
  metadata?: unknown;
};

export type AgentChatResponse = {
  run_id: string;
  agent_id: string;
  provider: string;
  model: string;
  message: ChatMessage;
  tool_calls?: ToolCall[] | unknown[] | null;
  usage?: TokenUsage | null;
  started_at: string;
  ended_at: string;
  duration_ms: number;
};

export type HealthResponse = {
  service: string;
  running: boolean;
  ollama_reachable: boolean;
  ollama_endpoint: string;
  default_provider: string;
  default_model: string | null;
  model_count: number | null;
  started_at: string;
  checked_at: string;
  uptime_seconds: number;
};

export type SetupStatus = {
  litellm_enabled: boolean;
  litellm_ready: boolean;
  usable_provider_count: number;
  usable_model_count: number;
  active_agent_count: number;
  ready: boolean;
  next_step: "start_litellm" | "add_provider" | "select_model" | "create_agent" | "ready";
  missing_steps: Array<"start_litellm" | "add_provider" | "select_model" | "create_agent" | "ready">;
};

export type ModelDetails = {
  family?: string | null;
  families?: string[] | null;
  format?: string | null;
  parameter_size?: string | null;
  quantization_level?: string | null;
};

export type OllamaModel = {
  name: string;
  model?: string | null;
  modified_at?: string | null;
  size?: number | null;
  digest?: string | null;
  details?: ModelDetails | null;
};

export type ModelsResponse = {
  default_model: string | null;
  models: OllamaModel[];
};

export type RunRecord = {
  id: string;
  app_id?: string | null;
  agent_id?: string | null;
  model_id?: string | null;
  resolved_tool_ids?: string[];
  provider: string;
  model: string;
  source_app: string | null;
  prompt_summary: string;
  response_summary: string | null;
  status: "completed" | "requires_action" | "failed";
  started_at: string;
  ended_at: string;
  duration_ms: number;
  error: string | null;
  usage?: TokenUsage | null;
};

export type RunToolRequest = {
  id: string;
  toolId: string;
  name: string;
  arguments: unknown;
  riskLevel: "low" | "medium" | "high" | string;
  displayName: string;
};

export type AppRecord = {
  id: string;
  name: string;
  description: string | null;
  defaultAgentId: string;
  allowedAgentIds: string[];
  allowedToolIds: string[] | null;
  enabled: boolean;
};

export type AppCapabilities = {
  appId: string;
  appName: string;
  defaultAgent: {
    id: string;
    name: string;
    description: string;
  };
  allowedAgents: Array<{
    id: string;
    name: string;
    description: string;
  }>;
  tools: Array<{
    id: string;
    name: string;
    description: string;
    riskLevel: string;
    enabled: boolean;
  }>;
  model: {
    id: string;
    name: string;
    provider: string;
    modelName: string;
    status: string;
  };
  warnings?: string[];
};

export type PairingStatus = "pending" | "approved" | "denied" | "expired";
export type AppTokenKind = "pairing" | "service";

export type AppPairingSummary = {
  id: string;
  appId: string;
  appName: string;
  requestedScopes: string[];
  origin: string | null;
  redirectUri: string | null;
  userCode: string;
  status: PairingStatus;
  createdAt: string;
  expiresAt: string;
  approvedAt: string | null;
  deniedAt: string | null;
  deliveredAt: string | null;
};

export type AppTokenSummary = {
  id: string;
  appId: string;
  name: string;
  scopes: string[];
  origin: string | null;
  kind: AppTokenKind;
  createdAt: string;
  lastUsedAt: string | null;
  expiresAt: string | null;
  revokedAt: string | null;
};

export type ConnectionsResponse = {
  pairings: AppPairingSummary[];
  tokens: AppTokenSummary[];
};

export type PairingStartRequest = {
  appId: string;
  appName?: string;
  requestedScopes?: string[];
  origin?: string;
  redirectUri?: string;
};

export type PairingStartResponse = {
  pairingId: string;
  pairingSecret: string;
  userCode: string;
  verificationUri: string;
  expiresAt: string;
};

export type PairingExchangeResponse = {
  status: "approved";
  appId: string;
  tokenId: string;
  token: string;
  scopes: string[];
};

export type ServiceTokenCreateRequest = {
  name?: string;
  scopes?: string[];
  origin?: string;
  expiresAt?: string;
};

export type IssuedAppTokenResponse = {
  token: string;
  record: AppTokenSummary;
};

export type ToolRecord = {
  id: string;
  name: string;
  description: string;
  riskLevel: "low" | "medium" | "high" | string;
  enabled: boolean;
  inputSchema: unknown | null;
  outputSchema: unknown | null;
};

export type AuditRecord = {
  id: string;
  event: string;
  level: "info" | "warn" | "denied" | "error";
  message: string;
  app_id?: string | null;
  agent_id?: string | null;
  run_id?: string | null;
  metadata?: unknown | null;
  created_at: string;
};

export type RunCreateRequest = {
  appId: string;
  agentId?: string | null;
  input?: string;
  messages?: ChatMessage[];
  instructions?: string;
  context?: unknown;
  generation?: GenerationSettings;
  metadata?: unknown;
};

export type RunCreateResponse = {
  runId: string;
  status: "completed" | "requires_action" | "failed";
  appId: string;
  agentId: string;
  modelId: string;
  output?: string;
  toolRequests: RunToolRequest[];
  durationMs: number;
};

export type RunToolResult = {
  toolCallId: string;
  toolId?: string;
  result?: unknown;
  error?: string;
};

export type RunToolResultsRequest = {
  appId: string;
  toolResults: RunToolResult[];
};

export type Settings = {
  ollama_endpoint: string;
  default_provider: string;
  default_model: string | null;
  generation: GenerationSettings;
  instructions: InstructionSettings;
  logging_enabled: boolean;
  api_token: string | null;
  theme: string;
  litellm: LiteLlmSettings;
  litellm_providers: LiteLlmProviderConfig[];
  model_routes: ModelRoute[];
  agents: AgentRecord[];
};

export type ProviderStatus = {
  id: string;
  name: string;
  kind: string;
  enabled: boolean;
  healthy: boolean;
  provider_type?: string | null;
  api_key_configured?: boolean | null;
  api_key_env_var?: string | null;
  base_url?: string;
};

export type LiteLlmProviderTestRequest = {
  provider_id?: string;
  model: string;
  message?: string;
};

export type LiteLlmProviderTestResponse = {
  ok: boolean;
  content: string;
  usage?: TokenUsage | null;
};

export type GenerateLiteLlmConfigResponse = {
  path: string;
  routes_written: number;
  providers_written: number;
  entries_written: number;
};

export type LiteLlmServiceStartResponse = {
  status: string;
  base_url: string;
  config_path: string;
  command: string;
  pid: number | null;
};

export type ProviderModelOption = {
  name: string;
  litellm_model: string;
  source: string;
};

export type ProviderModelsResponse = {
  provider_id: string;
  provider_type: string;
  models: ProviderModelOption[];
};

export type StreamEvent = {
  event: string;
  data: string;
};

export type LlamaHarnessClientOptions = {
  baseUrl?: string;
  token?: string;
  fetchImpl?: typeof fetch;
};

export class LlamaHarnessClient {
  private readonly baseUrl: string;
  private readonly token?: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: LlamaHarnessClientOptions = {}) {
    this.baseUrl = (options.baseUrl || "http://127.0.0.1:8787").replace(/\/$/, "");
    this.token = options.token;
    this.fetchImpl = options.fetchImpl || fetch;
  }

  health(): Promise<HealthResponse> {
    return this.request("/health");
  }

  listModels(): Promise<ModelsResponse> {
    return this.request("/api/models");
  }

  listProviders(): Promise<ProviderStatus[]> {
    return this.request("/api/providers");
  }

  setupStatus(): Promise<SetupStatus> {
    return this.request("/api/setup/status");
  }

  listAgents(): Promise<AgentRecord[]> {
    return this.request("/api/agents");
  }

  listApps(): Promise<AppRecord[]> {
    return this.request("/api/apps");
  }

  appCapabilities(appId: string): Promise<AppCapabilities> {
    return this.request(`/api/apps/${encodeURIComponent(appId)}/capabilities`);
  }

  startPairing(request: PairingStartRequest): Promise<PairingStartResponse> {
    return this.request("/api/pairing/start", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  exchangePairing(pairingId: string, pairingSecret: string): Promise<PairingExchangeResponse> {
    return this.request(`/api/pairing/${encodeURIComponent(pairingId)}/exchange`, {
      method: "POST",
      body: JSON.stringify({ pairingSecret }),
    });
  }

  listConnections(): Promise<ConnectionsResponse> {
    return this.request("/api/admin/connections");
  }

  approvePairing(pairingId: string, scopes?: string[]): Promise<AppPairingSummary> {
    return this.request(`/api/admin/pairing/${encodeURIComponent(pairingId)}/approve`, {
      method: "POST",
      body: JSON.stringify({ scopes }),
    });
  }

  denyPairing(pairingId: string): Promise<AppPairingSummary> {
    return this.request(`/api/admin/pairing/${encodeURIComponent(pairingId)}/deny`, {
      method: "POST",
    });
  }

  createServiceToken(appId: string, request: ServiceTokenCreateRequest): Promise<IssuedAppTokenResponse> {
    return this.request(`/api/admin/apps/${encodeURIComponent(appId)}/tokens`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  revokeAppToken(appId: string, tokenId: string): Promise<AppTokenSummary> {
    return this.request(`/api/admin/apps/${encodeURIComponent(appId)}/tokens/${encodeURIComponent(tokenId)}/revoke`, {
      method: "POST",
    });
  }

  listTools(): Promise<ToolRecord[]> {
    return this.request("/api/tools");
  }

  createAgent(agent: AgentCreateRequest): Promise<AgentRecord> {
    return this.request("/api/agents", {
      method: "POST",
      body: JSON.stringify(agent),
    });
  }

  getAgent(agentId: string): Promise<AgentRecord> {
    return this.request(`/api/agents/${encodeURIComponent(agentId)}`);
  }

  patchAgent(agentId: string, patch: AgentPatch): Promise<AgentRecord> {
    return this.request(`/api/agents/${encodeURIComponent(agentId)}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
  }

  chatWithAgent(agentId: string, request: AgentChatRequest): Promise<AgentChatResponse> {
    return this.request(`/api/agents/${encodeURIComponent(agentId)}/chat`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  listProviderModels(providerId: string): Promise<ProviderModelsResponse> {
    return this.request(`/api/providers/${encodeURIComponent(providerId)}/models`);
  }

  chat(request: ChatRequest): Promise<ChatResponse> {
    return this.request("/api/chat", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  runs(limit = 50): Promise<{ runs: RunRecord[] }> {
    return this.request(`/api/runs?limit=${limit}`);
  }

  run(request: RunCreateRequest): Promise<RunCreateResponse> {
    return this.request("/api/runs", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  submitRunToolResults(runId: string, request: RunToolResultsRequest): Promise<RunCreateResponse> {
    return this.request(`/api/runs/${encodeURIComponent(runId)}/tool-results`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  listAudit(limit = 50): Promise<{ audit: AuditRecord[] }> {
    return this.request(`/api/audit?limit=${limit}`);
  }

  settings(): Promise<Settings> {
    return this.request("/api/settings");
  }

  updateSettings(settings: Partial<Settings>): Promise<Settings> {
    return this.request("/api/settings", {
      method: "PUT",
      body: JSON.stringify(settings),
    });
  }

  testLiteLLMProvider(request: LiteLlmProviderTestRequest): Promise<LiteLlmProviderTestResponse> {
    return this.request("/api/providers/litellm/test", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  generateLiteLLMConfig(outputPath?: string | null): Promise<GenerateLiteLlmConfigResponse> {
    return this.request("/api/litellm/config/generate", {
      method: "POST",
      body: JSON.stringify({ output_path: outputPath || undefined }),
    });
  }

  startLiteLLMService(): Promise<LiteLlmServiceStartResponse> {
    return this.request("/api/litellm/service/start", {
      method: "POST",
    });
  }

  async *streamChat(request: ChatRequest): AsyncGenerator<StreamEvent> {
    const response = await this.fetchImpl(`${this.baseUrl}/api/chat/stream`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(await errorMessage(response));
    }
    if (!response.body) {
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const chunk = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseSseChunk(chunk);
        if (event) {
          yield event;
        }
        boundary = buffer.indexOf("\n\n");
      }
    }

    const trailing = parseSseChunk(buffer.trim());
    if (trailing) {
      yield trailing;
    }
  }

  async *streamRun(request: RunCreateRequest): AsyncGenerator<StreamEvent> {
    const response = await this.fetchImpl(`${this.baseUrl}/api/runs/stream`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(await errorMessage(response));
    }
    if (!response.body) {
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const chunk = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseSseChunk(chunk);
        if (event) {
          yield event;
        }
        boundary = buffer.indexOf("\n\n");
      }
    }

    const trailing = parseSseChunk(buffer.trim());
    if (trailing) {
      yield trailing;
    }
  }

  private request<T>(path: string, init?: RequestInit): Promise<T> {
    return this.fetchImpl(`${this.baseUrl}${path}`, {
      headers: this.headers(init?.headers),
      ...init,
    }).then(async (response) => {
      if (!response.ok) {
        throw new Error(await errorMessage(response));
      }
      return response.json() as Promise<T>;
    });
  }

  private headers(extra?: HeadersInit): HeadersInit {
    return {
      "content-type": "application/json",
      ...(this.token ? { authorization: `Bearer ${this.token}` } : {}),
      ...(extra || {}),
    };
  }
}

async function errorMessage(response: Response): Promise<string> {
  const body = await response.json().catch(() => null);
  return body?.error || `${response.status} ${response.statusText}`;
}

function parseSseChunk(chunk: string): StreamEvent | null {
  if (!chunk) {
    return null;
  }

  let event = "message";
  const data: string[] = [];

  for (const line of chunk.split(/\r?\n/)) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      data.push(line.slice("data:".length).trimStart());
    }
  }

  if (!data.length) {
    return null;
  }

  return {
    event,
    data: data.join("\n"),
  };
}
