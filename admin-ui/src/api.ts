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

export type Settings = {
  ollama_endpoint: string;
  default_model: string | null;
  generation: GenerationSettings;
  instructions: InstructionSettings;
  logging_enabled: boolean;
  api_token: string | null;
  theme: string;
};

export type Health = {
  service: string;
  running: boolean;
  ollama_reachable: boolean;
  ollama_endpoint: string;
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
  model: string;
  source_app: string | null;
  prompt_summary: string;
  response_summary: string | null;
  status: "completed" | "failed";
  started_at: string;
  ended_at: string;
  duration_ms: number;
  error: string | null;
};

export type ChatResponse = {
  run_id: string;
  model: string;
  message: {
    role: string;
    content: string;
  };
  started_at: string;
  ended_at: string;
  duration_ms: number;
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
  tools: () => request<unknown>("/api/tools"),
};
