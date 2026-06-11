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

export type ChatMessage = {
  role: "system" | "user" | "assistant" | string;
  content: string;
};

export type ChatRequest = {
  model?: string;
  source_app?: string;
  messages?: ChatMessage[];
  prompt?: string;
  instructions?: string;
  generation?: GenerationSettings;
};

export type ChatResponse = {
  run_id: string;
  model: string;
  message: ChatMessage;
  started_at: string;
  ended_at: string;
  duration_ms: number;
};

export type HealthResponse = {
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

export type Settings = {
  ollama_endpoint: string;
  default_model: string | null;
  generation: GenerationSettings;
  instructions: InstructionSettings;
  logging_enabled: boolean;
  api_token: string | null;
  theme: string;
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

  chat(request: ChatRequest): Promise<ChatResponse> {
    return this.request("/api/chat", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  runs(limit = 50): Promise<{ runs: RunRecord[] }> {
    return this.request(`/api/runs?limit=${limit}`);
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
