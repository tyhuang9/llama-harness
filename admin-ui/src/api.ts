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

export type AppRecord = {
  id: string;
  name: string;
  description: string | null;
  defaultAgentId: string;
  allowedAgentIds: string[];
  allowedToolIds: string[] | null;
  enabled: boolean;
};

export type AppPatch = Partial<
  Pick<AppRecord, "name" | "description" | "defaultAgentId" | "allowedAgentIds" | "allowedToolIds" | "enabled">
>;

export type ToolRecord = {
  id: string;
  name: string;
  description: string;
  riskLevel: "low" | "medium" | "high" | string;
  enabled: boolean;
  inputSchema: unknown | null;
  outputSchema: unknown | null;
};

export type ToolPatch = Partial<Pick<ToolRecord, "name" | "description" | "riskLevel" | "enabled" | "inputSchema" | "outputSchema">>;

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

export type Health = {
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

export type OllamaModel = {
  name: string;
  model?: string | null;
  modified_at?: string | null;
  size?: number | null;
  digest?: string | null;
  details?: {
    family?: string | null;
    parameter_size?: string | null;
    quantization_level?: string | null;
  } | null;
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
  status: "completed" | "failed";
  started_at: string;
  ended_at: string;
  duration_ms: number;
  error: string | null;
  usage?: TokenUsage | null;
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
  messages?: Array<{ role: string; content: string | unknown }>;
  instructions?: string;
  context?: unknown;
  generation?: GenerationSettings;
  metadata?: unknown;
};

export type RunCreateResponse = {
  runId: string;
  status: "completed" | "failed";
  appId: string;
  agentId: string;
  modelId: string;
  output: string;
  durationMs: number;
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
  message: {
    role: string;
    content: string;
  };
  usage?: TokenUsage | null;
  started_at: string;
  ended_at: string;
  duration_ms: number;
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

export type LiteLlmTestResponse = {
  ok: boolean;
  content: string;
  usage?: TokenUsage | null;
};

export type LiteLlmProviderTestDraft = {
  provider: LiteLlmProviderConfig;
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

export type ApplyLiteLlmProvidersResponse = {
  settings: Settings;
  provider_statuses: ProviderStatus[];
  env_file_path: string;
  config_path: string;
  litellm_ready: boolean;
  warning?: string | null;
};

const API_BASE_KEY = "llama-harness-api-base";

export function getApiBase(): string {
  return localStorage.getItem(API_BASE_KEY) || import.meta.env.VITE_HARNESS_API_URL || "http://127.0.0.1:8787";
}

export function setApiBase(value: string): void {
  localStorage.setItem(API_BASE_KEY, value.replace(/\/$/, ""));
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBase()}${path}`, {
    headers: {
      "content-type": "application/json",
      ...(init?.headers || {}),
    },
    ...init,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),
  setupStatus: () => request<SetupStatus>("/api/setup/status"),
  agents: () => request<AgentRecord[]>("/api/agents"),
  agent: (agentId: string) => request<AgentRecord>(`/api/agents/${encodeURIComponent(agentId)}`),
  createAgent: (agent: AgentCreateRequest) =>
    request<AgentRecord>("/api/agents", {
      method: "POST",
      body: JSON.stringify(agent),
    }),
  patchAgent: (agentId: string, patch: AgentPatch) =>
    request<AgentRecord>(`/api/agents/${encodeURIComponent(agentId)}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),
  apps: () => request<AppRecord[]>("/api/apps"),
  app: (appId: string) => request<AppRecord>(`/api/apps/${encodeURIComponent(appId)}`),
  patchApp: (appId: string, patch: AppPatch) =>
    request<AppRecord>(`/api/apps/${encodeURIComponent(appId)}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),
  appCapabilities: (appId: string) => request<AppCapabilities>(`/api/apps/${encodeURIComponent(appId)}/capabilities`),
  tools: () => request<ToolRecord[]>("/api/tools"),
  patchTool: (toolId: string, patch: ToolPatch) =>
    request<ToolRecord>(`/api/tools/${encodeURIComponent(toolId)}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),
  providers: () => request<ProviderStatus[]>("/api/providers"),
  providerModels: (providerId: string) => request<ProviderModelsResponse>(`/api/providers/${encodeURIComponent(providerId)}/models`),
  models: () => request<ModelsResponse>("/api/models"),
  settings: () => request<Settings>("/api/settings"),
  updateSettings: (settings: Partial<Settings>) =>
    request<Settings>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(settings),
    }),
  setDefaultModel: (model: string) =>
    request<Settings>("/api/models/default", {
      method: "POST",
      body: JSON.stringify({ model }),
    }),
  testModel: (model: string | null, prompt: string) =>
    request<ChatResponse>("/api/models/test", {
      method: "POST",
      body: JSON.stringify({
        model: model || undefined,
        prompt,
        source_app: "admin-ui",
      }),
    }),
  testLiteLLMProvider: (model: string, message: string, providerId?: string | null, draft?: LiteLlmProviderTestDraft) =>
    request<LiteLlmTestResponse>("/api/providers/litellm/test", {
      method: "POST",
      body: JSON.stringify({ model, message, provider_id: providerId || undefined, draft_provider: draft?.provider }),
    }),
  applyLiteLLMProviders: (providers: LiteLlmProviderConfig[]) =>
    request<ApplyLiteLlmProvidersResponse>("/api/litellm/providers/apply", {
      method: "POST",
      body: JSON.stringify({ providers }),
    }),
  generateLiteLLMConfig: (outputPath?: string | null) =>
    request<GenerateLiteLlmConfigResponse>("/api/litellm/config/generate", {
      method: "POST",
      body: JSON.stringify({ output_path: outputPath || undefined }),
    }),
  startLiteLLMService: () =>
    request<LiteLlmServiceStartResponse>("/api/litellm/service/start", {
      method: "POST",
    }),
  chat: (prompt: string, model?: string, instructions?: string) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        model,
        prompt,
        instructions,
        source_app: "admin-ui",
      }),
    }),
  runs: (limit = 50) => request<{ runs: RunRecord[] }>(`/api/runs?limit=${limit}`),
  createRun: (payload: RunCreateRequest) =>
    request<RunCreateResponse>("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  audit: (limit = 50) => request<{ audit: AuditRecord[] }>(`/api/audit?limit=${limit}`),
};
