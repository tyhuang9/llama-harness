use crate::{
    config::{
        self, AppConfig, GenerationSettings, InstructionSettings, LiteLlmProviderConfig,
        LiteLlmSettings, ModelRoute, REDACTED_SECRET,
    },
    litellm::{generate_litellm_config, litellm_model_for_provider},
    ollama::OllamaModel,
    providers::{
        ChatMessage, ModelProvider, ProviderChatRequest, ProviderRegistry, ProviderStreamEvent,
        TokenUsage,
    },
    runs::{self, RunRecord, RunStatus},
};
use axum::{
    extract::{Path as AxumPath, Query, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    convert::Infallible,
    env,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::process::Command;
use tokio::sync::RwLock;
use uuid::Uuid;

const MAX_RECENT_RUNS: usize = 100;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<RwLock<AppConfig>>,
    pub config_path: PathBuf,
    pub runs_path: PathBuf,
    pub runs: Arc<RwLock<VecDeque<RunRecord>>>,
    pub providers: ProviderRegistry,
    pub started_at: DateTime<Utc>,
}

#[derive(Serialize)]
struct ApiError {
    error: String,
}

type ApiResult<T> = Result<T, (StatusCode, Json<ApiError>)>;

#[derive(Serialize)]
struct HealthResponse {
    service: String,
    running: bool,
    ollama_reachable: bool,
    ollama_endpoint: String,
    default_provider: String,
    default_model: Option<String>,
    model_count: Option<usize>,
    started_at: DateTime<Utc>,
    checked_at: DateTime<Utc>,
    uptime_seconds: u64,
}

#[derive(Serialize)]
struct ModelsResponse {
    default_model: Option<String>,
    models: Vec<OllamaModel>,
}

#[derive(Deserialize)]
struct SetDefaultModelRequest {
    model: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    provider: Option<String>,
    model: Option<String>,
    source_app: Option<String>,
    messages: Option<Vec<ChatMessage>>,
    prompt: Option<String>,
    instructions: Option<String>,
    generation: Option<GenerationSettings>,
    tools: Option<serde_json::Value>,
    tool_choice: Option<serde_json::Value>,
    stream: Option<bool>,
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ChatResponse {
    run_id: String,
    provider: String,
    model: String,
    message: ChatMessage,
    usage: Option<TokenUsage>,
    started_at: DateTime<Utc>,
    ended_at: DateTime<Utc>,
    duration_ms: u64,
}

#[derive(Serialize)]
struct ProviderStatus {
    id: String,
    name: String,
    kind: String,
    enabled: bool,
    healthy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_key_configured: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_key_env_var: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    base_url: Option<String>,
}

#[derive(Deserialize)]
struct LiteLlmProviderTestRequest {
    provider_id: Option<String>,
    model: String,
    message: Option<String>,
    draft_provider: Option<LiteLlmProviderConfig>,
}

#[derive(Serialize)]
struct LiteLlmProviderTestResponse {
    ok: bool,
    content: String,
    usage: Option<TokenUsage>,
}

#[derive(Deserialize)]
struct GenerateLiteLlmConfigRequest {
    output_path: Option<String>,
}

#[derive(Serialize)]
struct GenerateLiteLlmConfigResponse {
    path: String,
    routes_written: usize,
    providers_written: usize,
    entries_written: usize,
}

#[derive(Serialize)]
struct LiteLlmServiceStartResponse {
    status: String,
    base_url: String,
    config_path: String,
    command: String,
    pid: Option<u32>,
}

#[derive(Serialize)]
struct ProviderModelsResponse {
    provider_id: String,
    provider_type: String,
    models: Vec<ProviderModelOption>,
}

#[derive(Serialize)]
struct ProviderModelOption {
    name: String,
    litellm_model: String,
    source: String,
}

#[derive(Deserialize)]
struct RunsQuery {
    limit: Option<usize>,
}

#[derive(Serialize)]
struct RunsResponse {
    runs: Vec<RunRecord>,
}

#[derive(Deserialize)]
struct SettingsPatch {
    ollama_endpoint: Option<String>,
    default_provider: Option<String>,
    default_model: Option<String>,
    generation: Option<GenerationSettings>,
    instructions: Option<InstructionSettings>,
    logging_enabled: Option<bool>,
    api_token: Option<String>,
    theme: Option<String>,
    litellm: Option<LiteLlmSettings>,
    litellm_providers: Option<Vec<LiteLlmProviderConfig>>,
    model_routes: Option<Vec<ModelRoute>>,
}

#[derive(Serialize)]
struct ToolsResponse {
    enabled: bool,
    summary: &'static str,
    planned_registry_shape: serde_json::Value,
}

#[derive(Serialize)]
struct StreamEvent<'a> {
    run_id: &'a str,
    content: &'a str,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/api/health", get(health))
        .route("/api/models", get(list_models))
        .route("/api/models/default", post(set_default_model))
        .route("/api/models/test", post(test_model))
        .route("/api/providers", get(list_providers))
        .route(
            "/api/providers/:provider_id/models",
            get(list_provider_models),
        )
        .route("/api/providers/litellm/test", post(test_litellm_provider))
        .route(
            "/api/litellm/config/generate",
            post(generate_litellm_config_endpoint),
        )
        .route("/api/litellm/service/start", post(start_litellm_service))
        .route("/api/chat", post(chat))
        .route("/api/chat/stream", post(stream_chat))
        .route("/api/runs", get(list_runs))
        .route("/api/settings", get(get_settings).put(update_settings))
        .route("/api/tools", get(tools_placeholder))
        .with_state(state)
}

async fn root() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "service": "llama-harness",
        "api": "/api",
        "health": "/health"
    }))
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let started_at = state.started_at;
    let checked_at = Utc::now();
    let config = state.config.read().await.clone();
    let models = state.providers.list_ollama_models(&config).await;

    Json(HealthResponse {
        service: "llama-harness".to_string(),
        running: true,
        ollama_reachable: models.is_ok(),
        ollama_endpoint: config.ollama_endpoint,
        default_provider: config.default_provider,
        default_model: config.default_model,
        model_count: models.ok().map(|models| models.len()),
        started_at,
        checked_at,
        uptime_seconds: (checked_at - started_at).num_seconds().max(0) as u64,
    })
}

async fn list_models(State(state): State<AppState>) -> ApiResult<Json<ModelsResponse>> {
    let config = state.config.read().await.clone();
    let models = state
        .providers
        .list_ollama_models(&config)
        .await
        .map_err(|err| api_error(StatusCode::BAD_GATEWAY, err.to_string()))?;

    Ok(Json(ModelsResponse {
        default_model: config.default_model,
        models,
    }))
}

async fn set_default_model(
    State(state): State<AppState>,
    Json(payload): Json<SetDefaultModelRequest>,
) -> ApiResult<Json<AppConfig>> {
    let model = payload.model.trim();
    if model.is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "model is required"));
    }

    let config = {
        let mut config = state.config.write().await;
        config.default_model = Some(model.to_string());
        config::save_config(&state.config_path, &config)
            .await
            .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
        config.redacted_for_response()
    };

    Ok(Json(config))
}

async fn test_model(
    State(state): State<AppState>,
    Json(payload): Json<ChatRequest>,
) -> ApiResult<Json<ChatResponse>> {
    let mut payload = payload;
    if payload.prompt.is_none() && payload.messages.as_ref().is_none_or(Vec::is_empty) {
        payload.prompt = Some("Reply with a short local model health check.".to_string());
    }
    if payload.source_app.is_none() {
        payload.source_app = Some("admin-ui".to_string());
    }
    chat_with_payload(state, payload).await
}

async fn list_providers(State(state): State<AppState>) -> Json<Vec<ProviderStatus>> {
    let config = state.config.read().await.clone();
    let ollama_healthy = state.providers.list_ollama_models(&config).await.is_ok();
    let litellm_healthy = state.providers.litellm_healthy(&config).await;

    let mut providers = vec![
        ProviderStatus {
            id: "ollama".to_string(),
            name: "Ollama".to_string(),
            kind: "local".to_string(),
            enabled: true,
            healthy: ollama_healthy,
            provider_type: Some("ollama".to_string()),
            api_key_configured: None,
            api_key_env_var: None,
            base_url: None,
        },
        ProviderStatus {
            id: "litellm".to_string(),
            name: "LiteLLM".to_string(),
            kind: "gateway".to_string(),
            enabled: config.litellm.enabled,
            healthy: litellm_healthy,
            provider_type: None,
            api_key_configured: None,
            api_key_env_var: None,
            base_url: Some(config.litellm.base_url.clone()),
        },
    ];

    providers.extend(config.litellm_providers.iter().map(|provider| {
        ProviderStatus {
            id: provider.id.clone(),
            name: provider_display_name(provider),
            kind: "litellm_provider".to_string(),
            enabled: provider.enabled,
            healthy: provider.enabled && litellm_healthy,
            provider_type: Some(normalize_provider_type(&provider.provider_type)),
            api_key_configured: Some(provider_api_key_configured(provider)),
            api_key_env_var: Some(provider.api_key_env_var.clone()),
            base_url: provider
                .api_base
                .clone()
                .or_else(|| Some(config.litellm.base_url.clone())),
        }
    }));

    Json(providers)
}

async fn list_provider_models(
    State(state): State<AppState>,
    AxumPath(provider_id): AxumPath<String>,
) -> ApiResult<Json<ProviderModelsResponse>> {
    let config = state.config.read().await.clone();
    let provider_id = provider_id.trim();

    if provider_id.eq_ignore_ascii_case("ollama") {
        let models = state
            .providers
            .list_ollama_models(&config)
            .await
            .map_err(|err| api_error(StatusCode::BAD_GATEWAY, err.to_string()))?
            .into_iter()
            .map(|model| ProviderModelOption {
                litellm_model: model.name.clone(),
                name: model.name,
                source: "ollama".to_string(),
            })
            .collect();

        return Ok(Json(ProviderModelsResponse {
            provider_id: "ollama".to_string(),
            provider_type: "ollama".to_string(),
            models,
        }));
    }

    let provider = find_litellm_provider(&config, provider_id).ok_or_else(|| {
        api_error(
            StatusCode::NOT_FOUND,
            format!("unknown LiteLLM provider: {provider_id}"),
        )
    })?;
    let provider_type = normalize_provider_type(&provider.provider_type);
    let models = suggested_provider_models(&provider)
        .into_iter()
        .map(|model| ProviderModelOption {
            litellm_model: litellm_model_for_provider(&provider, &model),
            name: model,
            source: "catalog".to_string(),
        })
        .collect();

    Ok(Json(ProviderModelsResponse {
        provider_id: provider.id,
        provider_type,
        models,
    }))
}

async fn test_litellm_provider(
    State(state): State<AppState>,
    Json(payload): Json<LiteLlmProviderTestRequest>,
) -> ApiResult<Json<LiteLlmProviderTestResponse>> {
    let requested_model = payload.model.trim();
    if requested_model.is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "model is required"));
    }

    let mut config = state.config.read().await.clone();
    let provider_id = payload
        .provider_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("litellm");
    let draft_provider = match payload.draft_provider {
        Some(provider) => Some(normalize_litellm_provider_config(
            provider,
            &config.litellm_providers,
            &[],
        )?),
        None => None,
    };
    if let Some(provider) = draft_provider {
        let mut replaced = false;
        config.litellm_providers = config
            .litellm_providers
            .into_iter()
            .map(|existing| {
                if existing.id.eq_ignore_ascii_case(&provider.id) {
                    replaced = true;
                    provider.clone()
                } else {
                    existing
                }
            })
            .collect();
        if !replaced {
            config.litellm_providers.push(provider);
        }
    }
    let model = if provider_id == "litellm" {
        requested_model.to_string()
    } else {
        let configured_provider = find_litellm_provider(&config, provider_id).ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                format!("unknown LiteLLM provider: {provider_id}"),
            )
        })?;
        litellm_model_for_provider(&configured_provider, requested_model)
    };
    let provider = state
        .providers
        .get("litellm", &config)
        .ok_or_else(|| api_error(StatusCode::BAD_REQUEST, "unknown model provider: litellm"))?;
    let request = ProviderChatRequest {
        provider: provider_id.to_string(),
        model: model.clone(),
        messages: vec![ChatMessage::text(
            "user",
            payload
                .message
                .unwrap_or_else(|| "Say hello from llama-harness.".to_string()),
        )],
        temperature: Some(config.generation.temperature),
        top_p: Some(config.generation.top_p),
        max_tokens: Some(config.generation.max_tokens),
        tools: None,
        tool_choice: None,
        stream: false,
        metadata: Some(
            serde_json::json!({ "source_app": "admin-ui", "operation": "provider.test" }),
        ),
    };

    match provider.chat_completion(request).await {
        Ok(response) => {
            audit_hook(
                "provider.tested",
                serde_json::json!({ "provider": provider_id, "model": model }),
            );
            Ok(Json(LiteLlmProviderTestResponse {
                ok: true,
                content: response.content,
                usage: response.usage,
            }))
        }
        Err(err) => {
            let error = err.to_string();
            audit_hook(
                "provider.test_failed",
                serde_json::json!({ "provider": provider_id, "model": model }),
            );
            Err(api_error(StatusCode::BAD_GATEWAY, error))
        }
    }
}

async fn generate_litellm_config_endpoint(
    State(state): State<AppState>,
    Json(payload): Json<GenerateLiteLlmConfigRequest>,
) -> ApiResult<Json<GenerateLiteLlmConfigResponse>> {
    let config = state.config.read().await.clone();
    let output_path =
        resolve_litellm_config_path(&state.config_path, &config, payload.output_path.as_deref());
    let generation = generate_litellm_config(
        &config.litellm_providers,
        &config.model_routes,
        &output_path,
        &config.ollama_endpoint,
    )
    .await
    .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    audit_hook(
        "litellm.config.generated",
        serde_json::json!({
            "path": output_path.display().to_string(),
            "routes_written": generation.routes_written,
            "providers_written": generation.providers_written
        }),
    );

    Ok(Json(GenerateLiteLlmConfigResponse {
        path: output_path.display().to_string(),
        routes_written: generation.routes_written,
        providers_written: generation.providers_written,
        entries_written: generation.entries_written,
    }))
}

async fn start_litellm_service(
    State(state): State<AppState>,
) -> ApiResult<Json<LiteLlmServiceStartResponse>> {
    let config = state.config.read().await.clone();
    let output_path = resolve_litellm_config_path(&state.config_path, &config, None);

    if state.providers.litellm_healthy(&config).await {
        return Ok(Json(LiteLlmServiceStartResponse {
            status: "already_running".to_string(),
            base_url: config.litellm.base_url.clone(),
            config_path: output_path.display().to_string(),
            command: litellm_start_command_summary(&config, &output_path),
            pid: None,
        }));
    }

    generate_litellm_config(
        &config.litellm_providers,
        &config.model_routes,
        &output_path,
        &config.ollama_endpoint,
    )
    .await
    .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    let command =
        env::var("LLAMA_HARNESS_LITELLM_COMMAND").unwrap_or_else(|_| "litellm".to_string());
    let (host, port) = litellm_host_port(&config.litellm.base_url);
    let mut child = Command::new(&command)
        .arg("--config")
        .arg(&output_path)
        .arg("--host")
        .arg(&host)
        .arg("--port")
        .arg(&port)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .envs(litellm_child_env(&config))
        .spawn()
        .map_err(|err| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to start LiteLLM using '{command}': {err}"),
            )
        })?;
    let pid = child.id();
    tokio::spawn(async move {
        if let Err(err) = child.wait().await {
            tracing::warn!(error = %err, "failed while waiting for LiteLLM process");
        }
    });

    audit_hook(
        "litellm.service.started",
        serde_json::json!({
            "path": output_path.display().to_string(),
            "base_url": config.litellm.base_url,
            "pid": pid
        }),
    );

    Ok(Json(LiteLlmServiceStartResponse {
        status: "started".to_string(),
        base_url: config.litellm.base_url.clone(),
        config_path: output_path.display().to_string(),
        command: litellm_start_command_summary(&config, &output_path),
        pid,
    }))
}

async fn chat(
    State(state): State<AppState>,
    Json(payload): Json<ChatRequest>,
) -> ApiResult<Json<ChatResponse>> {
    chat_with_payload(state, payload).await
}

async fn stream_chat(
    State(state): State<AppState>,
    Json(payload): Json<ChatRequest>,
) -> ApiResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    let resolved = resolve_chat(&state, payload).await?;
    let provider_request = resolved.provider_request(true);
    let run_id = Uuid::new_v4().to_string();
    let started_at = Utc::now();
    let started_timer = Instant::now();
    let prompt_summary = summarize_messages(&resolved.messages);

    let provider_stream = match resolved
        .provider
        .stream_chat_completion(provider_request)
        .await
    {
        Ok(provider_stream) => provider_stream,
        Err(err) => {
            let error = err.to_string();
            let ended_at = Utc::now();
            let record = RunRecord {
                id: run_id,
                provider: resolved.provider_id,
                model: resolved.model,
                source_app: resolved.source_app,
                prompt_summary,
                response_summary: None,
                status: RunStatus::Failed,
                started_at,
                ended_at,
                duration_ms: duration_ms(started_timer.elapsed()),
                error: Some(error.clone()),
                usage: None,
            };
            record_run(&state, record).await;
            return Err(api_error(StatusCode::BAD_GATEWAY, error));
        }
    };

    let stream_state = state.clone();
    let stream = async_stream::stream! {
        let mut provider_stream = provider_stream;
        let mut response_summary = String::new();
        let mut error: Option<String> = None;
        let mut usage: Option<TokenUsage> = None;

        while let Some(event) = provider_stream.next().await {
            match event {
                ProviderStreamEvent::Token { content } => {
                    response_summary.push_str(&content);
                    if let Ok(event) = Event::default()
                        .event("token")
                        .json_data(StreamEvent {
                            run_id: &run_id,
                            content: &content,
                        }) {
                        yield Ok(event);
                    }
                }
                ProviderStreamEvent::Done { usage: event_usage, raw } => {
                    usage = event_usage;
                    if let Ok(event) = Event::default()
                        .event("done")
                        .json_data(serde_json::json!({ "run_id": run_id, "usage": usage, "raw": raw })) {
                        yield Ok(event);
                    }
                }
                ProviderStreamEvent::Error { error: message } => {
                    error = Some(message.clone());
                    if let Ok(event) = Event::default()
                        .event("error")
                        .json_data(serde_json::json!({ "run_id": run_id, "error": message })) {
                        yield Ok(event);
                    }
                    break;
                }
                ProviderStreamEvent::Raw { data } => {
                    yield Ok(Event::default().event("raw").data(data));
                }
            }
        }

        let ended_at = Utc::now();
        let record = RunRecord {
            id: run_id,
            provider: resolved.provider_id,
            model: resolved.model,
            source_app: resolved.source_app,
            prompt_summary,
            response_summary: Some(truncate(&response_summary, 500)),
            status: if error.is_some() { RunStatus::Failed } else { RunStatus::Completed },
            started_at,
            ended_at,
            duration_ms: duration_ms(started_timer.elapsed()),
            error,
            usage,
        };
        record_run(&stream_state, record).await;
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

async fn list_runs(
    State(state): State<AppState>,
    Query(query): Query<RunsQuery>,
) -> Json<RunsResponse> {
    let limit = query.limit.unwrap_or(50).min(MAX_RECENT_RUNS);
    let runs = state
        .runs
        .read()
        .await
        .iter()
        .take(limit)
        .cloned()
        .collect();
    Json(RunsResponse { runs })
}

async fn get_settings(State(state): State<AppState>) -> Json<AppConfig> {
    Json(state.config.read().await.redacted_for_response())
}

async fn update_settings(
    State(state): State<AppState>,
    Json(patch): Json<SettingsPatch>,
) -> ApiResult<Json<AppConfig>> {
    let config = {
        let mut config = state.config.write().await;

        if let Some(endpoint) = patch.ollama_endpoint {
            let endpoint = endpoint.trim();
            if endpoint.is_empty() {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    "ollama_endpoint cannot be empty",
                ));
            }
            config.ollama_endpoint = endpoint.to_string();
        }
        if let Some(provider) = patch.default_provider {
            let provider = provider.trim().to_ascii_lowercase();
            if provider.is_empty() {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    "default_provider cannot be empty",
                ));
            }
            config.default_provider = provider;
        }
        if let Some(model) = patch.default_model {
            config.default_model = if model.trim().is_empty() {
                None
            } else {
                Some(model.trim().to_string())
            };
        }
        if let Some(generation) = patch.generation {
            config.generation = generation;
        }
        if let Some(instructions) = patch.instructions {
            config.instructions = instructions;
        }
        if let Some(logging_enabled) = patch.logging_enabled {
            config.logging_enabled = logging_enabled;
        }
        if let Some(api_token) = patch.api_token {
            config.api_token = if api_token.trim().is_empty() {
                None
            } else {
                Some(api_token)
            };
        }
        if let Some(theme) = patch.theme {
            config.theme = theme;
        }
        if let Some(litellm) = patch.litellm {
            let existing_api_key = config.litellm.api_key.clone();
            let mut litellm = litellm;
            litellm.api_key = match litellm.api_key {
                Some(value) if value == REDACTED_SECRET => existing_api_key,
                Some(value) if value.trim().is_empty() => None,
                Some(value) => Some(value),
                None => None,
            };
            config.litellm = litellm;
        }
        if let Some(litellm_providers) = patch.litellm_providers {
            let existing_providers = config.litellm_providers.clone();
            config.litellm_providers =
                normalize_litellm_provider_configs(litellm_providers, &existing_providers)?;
        }
        if let Some(model_routes) = patch.model_routes {
            config.model_routes = model_routes;
        }

        config::save_config(&state.config_path, &config)
            .await
            .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
        config.redacted_for_response()
    };

    Ok(Json(config))
}

async fn tools_placeholder() -> Json<ToolsResponse> {
    Json(ToolsResponse {
        enabled: false,
        summary: "Tool calling is intentionally out of scope for the MVP. This endpoint documents the future local-only registry shape.",
        planned_registry_shape: serde_json::json!({
            "tools": [
                {
                    "name": "string",
                    "description": "string",
                    "input_schema": {},
                    "local_only": true,
                    "enabled": false
                }
            ]
        }),
    })
}

async fn chat_with_payload(state: AppState, payload: ChatRequest) -> ApiResult<Json<ChatResponse>> {
    let resolved = resolve_chat(&state, payload).await?;
    let provider_request = resolved.provider_request(false);
    let run_id = Uuid::new_v4().to_string();
    let started_at = Utc::now();
    let started_timer = Instant::now();
    let prompt_summary = summarize_messages(&resolved.messages);

    let result = resolved.provider.chat_completion(provider_request).await;

    let ended_at = Utc::now();
    let duration_ms = duration_ms(started_timer.elapsed());

    match result {
        Ok(response) => {
            let message = ChatMessage::text("assistant", response.content);
            let model = response.model.unwrap_or_else(|| resolved.model.clone());
            let record = RunRecord {
                id: run_id.clone(),
                provider: resolved.provider_id.clone(),
                model: model.clone(),
                source_app: resolved.source_app,
                prompt_summary,
                response_summary: Some(truncate(&message.content_text(), 500)),
                status: RunStatus::Completed,
                started_at,
                ended_at,
                duration_ms,
                error: None,
                usage: response.usage.clone(),
            };
            record_run(&state, record).await;

            Ok(Json(ChatResponse {
                run_id,
                provider: resolved.provider_id,
                model,
                message,
                usage: response.usage,
                started_at,
                ended_at,
                duration_ms,
            }))
        }
        Err(err) => {
            let error = err.to_string();
            let record = RunRecord {
                id: run_id,
                provider: resolved.provider_id,
                model: resolved.model,
                source_app: resolved.source_app,
                prompt_summary,
                response_summary: None,
                status: RunStatus::Failed,
                started_at,
                ended_at,
                duration_ms,
                error: Some(error.clone()),
                usage: None,
            };
            record_run(&state, record).await;
            Err(api_error(StatusCode::BAD_GATEWAY, error))
        }
    }
}

#[derive(Clone)]
struct ResolvedChat {
    provider_id: String,
    provider: Arc<dyn ModelProvider>,
    model: String,
    source_app: Option<String>,
    messages: Vec<ChatMessage>,
    generation: GenerationSettings,
    tools: Option<serde_json::Value>,
    tool_choice: Option<serde_json::Value>,
    stream_requested: bool,
    metadata: Option<serde_json::Value>,
}

impl ResolvedChat {
    fn provider_request(&self, stream: bool) -> ProviderChatRequest {
        ProviderChatRequest {
            provider: self.provider_id.clone(),
            model: self.model.clone(),
            messages: self.messages.clone(),
            temperature: Some(self.generation.temperature),
            top_p: Some(self.generation.top_p),
            max_tokens: Some(self.generation.max_tokens),
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            stream: stream || self.stream_requested,
            metadata: self.metadata.clone(),
        }
    }
}

async fn resolve_chat(state: &AppState, payload: ChatRequest) -> ApiResult<ResolvedChat> {
    let config = state.config.read().await.clone();
    let requested_provider_id = payload
        .provider
        .filter(|provider| !provider.trim().is_empty())
        .unwrap_or_else(|| config.default_provider.clone())
        .trim()
        .to_ascii_lowercase();
    let messages = normalize_messages(payload.messages, payload.prompt)?;
    let messages = apply_instructions(&config.instructions, payload.instructions, messages);
    let model = payload
        .model
        .filter(|model| !model.trim().is_empty())
        .or_else(|| config.default_model_for_provider(&requested_provider_id))
        .ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "model is required because no default model is configured for provider {requested_provider_id}",
                ),
            )
        })?;
    let (provider_registry_id, model) = if requested_provider_id == "ollama"
        || requested_provider_id == "litellm"
    {
        (requested_provider_id.clone(), model)
    } else if let Some(litellm_provider) = find_litellm_provider(&config, &requested_provider_id) {
        (
            "litellm".to_string(),
            litellm_model_for_provider(&litellm_provider, &model),
        )
    } else {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!("unknown model provider: {requested_provider_id}"),
        ));
    };
    let provider = state
        .providers
        .get(&provider_registry_id, &config)
        .ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                format!("unknown model provider: {requested_provider_id}"),
            )
        })?;

    Ok(ResolvedChat {
        provider_id: requested_provider_id,
        provider,
        model,
        source_app: payload.source_app,
        messages,
        generation: payload.generation.unwrap_or(config.generation),
        tools: payload.tools,
        tool_choice: payload.tool_choice,
        stream_requested: payload.stream.unwrap_or(false),
        metadata: payload.metadata,
    })
}

fn normalize_messages(
    messages: Option<Vec<ChatMessage>>,
    prompt: Option<String>,
) -> ApiResult<Vec<ChatMessage>> {
    if let Some(messages) = messages {
        if !messages.is_empty() {
            return Ok(messages);
        }
    }

    if let Some(prompt) = prompt {
        let prompt = prompt.trim();
        if !prompt.is_empty() {
            return Ok(vec![ChatMessage::text("user", prompt)]);
        }
    }

    Err(api_error(
        StatusCode::BAD_REQUEST,
        "messages or prompt is required",
    ))
}

fn apply_instructions(
    settings: &InstructionSettings,
    request_instructions: Option<String>,
    mut messages: Vec<ChatMessage>,
) -> Vec<ChatMessage> {
    let mut parts = Vec::new();

    if settings.enabled {
        push_trimmed(&mut parts, &settings.system_prompt);

        let tool_context = settings.tool_context.trim();
        if !tool_context.is_empty() {
            parts.push(format!(
                "Available tools and tool instructions:\n{tool_context}"
            ));
        }
    }

    if let Some(instructions) = request_instructions {
        let instructions = instructions.trim();
        if !instructions.is_empty() {
            parts.push(format!("Request-specific instructions:\n{instructions}"));
        }
    }

    if !parts.is_empty() {
        messages.insert(0, ChatMessage::text("system", parts.join("\n\n")));
    }

    messages
}

fn push_trimmed(parts: &mut Vec<String>, value: &str) {
    let value = value.trim();
    if !value.is_empty() {
        parts.push(value.to_string());
    }
}

async fn record_run(state: &AppState, record: RunRecord) {
    let logging_enabled = state.config.read().await.logging_enabled;
    {
        let mut history = state.runs.write().await;
        runs::push_recent(&mut history, record.clone(), MAX_RECENT_RUNS);
    }

    if logging_enabled {
        if let Err(err) = runs::append_jsonl(&state.runs_path, &record).await {
            tracing::warn!(error = %err, "failed to append run log");
        }
    }
}

fn resolve_litellm_config_path(
    config_path: &Path,
    config: &AppConfig,
    requested_path: Option<&str>,
) -> PathBuf {
    let raw_path = requested_path
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| config.litellm.managed_config_path.clone())
        .unwrap_or_else(|| "litellm.config.yaml".to_string());
    let path = PathBuf::from(raw_path);
    if path.is_absolute() {
        return path;
    }

    config_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join(path)
}

fn litellm_start_command_summary(config: &AppConfig, config_path: &Path) -> String {
    let command =
        env::var("LLAMA_HARNESS_LITELLM_COMMAND").unwrap_or_else(|_| "litellm".to_string());
    let (host, port) = litellm_host_port(&config.litellm.base_url);
    format!(
        "{} --config {} --host {} --port {}",
        command,
        config_path.display(),
        host,
        port
    )
}

fn litellm_child_env(config: &AppConfig) -> Vec<(&'static str, String)> {
    config
        .litellm
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != REDACTED_SECRET)
        .map(|value| vec![("LITELLM_MASTER_KEY", value.to_string())])
        .unwrap_or_default()
}

fn litellm_host_port(base_url: &str) -> (String, String) {
    let trimmed = base_url.trim().trim_end_matches('/');
    let without_scheme = trimmed
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(trimmed);
    let authority = without_scheme
        .split('/')
        .next()
        .unwrap_or(without_scheme)
        .rsplit('@')
        .next()
        .unwrap_or(without_scheme);

    if let Some(rest) = authority.strip_prefix('[') {
        if let Some((host, after_host)) = rest.split_once(']') {
            let port = after_host.strip_prefix(':').unwrap_or("4000");
            return (host.to_string(), port.to_string());
        }
    }

    let (host, port) = authority
        .rsplit_once(':')
        .map(|(host, port)| (host, port))
        .unwrap_or((authority, "4000"));
    let host = if host.is_empty() { "127.0.0.1" } else { host };
    let port = if port.is_empty() { "4000" } else { port };
    (host.to_string(), port.to_string())
}

fn audit_hook(event: &str, metadata: serde_json::Value) {
    tracing::debug!(event, metadata = %metadata, "audit hook");
}

fn summarize_messages(messages: &[ChatMessage]) -> String {
    let joined = messages
        .iter()
        .map(|message| {
            if message.role == "system" {
                "system: [instructions applied]".to_string()
            } else {
                format!(
                    "{}: {}",
                    message.role,
                    message.content_text().replace('\n', " ")
                )
            }
        })
        .collect::<Vec<_>>()
        .join(" | ");
    truncate(&joined, 240)
}

fn truncate(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn find_litellm_provider(config: &AppConfig, provider_id: &str) -> Option<LiteLlmProviderConfig> {
    config
        .litellm_providers
        .iter()
        .find(|provider| provider.id.eq_ignore_ascii_case(provider_id.trim()))
        .cloned()
}

fn normalize_litellm_provider_configs(
    providers: Vec<LiteLlmProviderConfig>,
    existing_providers: &[LiteLlmProviderConfig],
) -> ApiResult<Vec<LiteLlmProviderConfig>> {
    let mut normalized = Vec::with_capacity(providers.len());
    let mut used_names: Vec<String> = Vec::new();
    let mut used_ids: Vec<String> = Vec::new();

    for provider in providers {
        let provider = normalize_litellm_provider_config(provider, existing_providers, &used_ids)?;
        let name_key = provider.display_name.trim().to_ascii_lowercase();
        if name_key.is_empty() {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "provider name is required",
            ));
        }
        if used_names.iter().any(|name| name == &name_key) {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!("provider name '{}' is already used", provider.display_name),
            ));
        }
        if provider.provider_type.trim().is_empty() {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!("provider type is required for '{}'", provider.display_name),
            ));
        }
        if provider.provider_type != "ollama" && provider.api_key_env_var.trim().is_empty() {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "API Key Environment Variable Name is required for '{}'",
                    provider.display_name
                ),
            ));
        }

        used_names.push(name_key);
        used_ids.push(provider.id.clone());
        normalized.push(provider);
    }

    Ok(normalized)
}

fn normalize_litellm_provider_config(
    mut provider: LiteLlmProviderConfig,
    existing_providers: &[LiteLlmProviderConfig],
    used_ids: &[String],
) -> ApiResult<LiteLlmProviderConfig> {
    provider.provider_type = normalize_provider_type(&provider.provider_type);

    provider.id = normalize_provider_id(&provider.id);
    if provider.id.is_empty() || provider.id.starts_with("draft_") {
        provider.id = provider_id_from_display_name(&provider.display_name, used_ids);
    }

    provider.display_name = provider.display_name.trim().to_string();

    provider.api_key_env_var = provider.api_key_env_var.trim().to_string();
    if provider.api_key_env_var.is_empty() && provider.provider_type != "ollama" {
        provider.api_key_env_var = default_api_key_env_var(&provider.provider_type).to_string();
    }

    let existing_api_key = existing_providers
        .iter()
        .find(|existing| existing.id == provider.id)
        .and_then(|existing| existing.api_key.clone());
    provider.api_key = match provider.api_key {
        Some(value) if value == REDACTED_SECRET => existing_api_key,
        Some(value) if value.trim().is_empty() => None,
        Some(value) => Some(value),
        None => None,
    };

    provider.api_base = provider
        .api_base
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    Ok(provider)
}

fn normalize_provider_id(value: &str) -> String {
    value
        .trim()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

fn provider_id_from_display_name(display_name: &str, used_ids: &[String]) -> String {
    let base = normalize_provider_id(display_name);
    let base = if base.is_empty() {
        format!("provider_{}", Uuid::new_v4().simple())
    } else {
        base
    };

    if !used_ids.iter().any(|id| id == &base) {
        return base;
    }

    for suffix in 2.. {
        let candidate = format!("{base}_{suffix}");
        if !used_ids.iter().any(|id| id == &candidate) {
            return candidate;
        }
    }

    unreachable!("unbounded suffix search should return");
}

fn normalize_provider_type(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

fn provider_display_name(provider: &LiteLlmProviderConfig) -> String {
    let name = provider.display_name.trim();
    if !name.is_empty() {
        return name.to_string();
    }

    provider
        .provider_type
        .split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => format!("{}{}", first.to_ascii_uppercase(), chars.as_str()),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn provider_api_key_configured(provider: &LiteLlmProviderConfig) -> bool {
    provider
        .api_key
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty())
        || (!provider.api_key_env_var.trim().is_empty()
            && env::var(&provider.api_key_env_var)
                .ok()
                .is_some_and(|value| !value.trim().is_empty()))
}

fn default_api_key_env_var(provider_type: &str) -> &'static str {
    match provider_type {
        "openai" => "OPENAI_API_KEY",
        "anthropic" => "ANTHROPIC_API_KEY",
        "gemini" => "GEMINI_API_KEY",
        "openrouter" => "OPENROUTER_API_KEY",
        "cohere" => "COHERE_API_KEY",
        "mistral" => "MISTRAL_API_KEY",
        "groq" => "GROQ_API_KEY",
        "deepseek" => "DEEPSEEK_API_KEY",
        "xai" => "XAI_API_KEY",
        "perplexity" => "PERPLEXITY_API_KEY",
        "together_ai" => "TOGETHER_API_KEY",
        "fireworks_ai" => "FIREWORKS_API_KEY",
        "huggingface" => "HUGGINGFACE_API_KEY",
        "replicate" => "REPLICATE_API_TOKEN",
        _ => "PROVIDER_API_KEY",
    }
}

fn suggested_provider_models(provider: &LiteLlmProviderConfig) -> Vec<String> {
    match normalize_provider_type(&provider.provider_type).as_str() {
        "openai" => [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "o3-mini",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        "anthropic" => [
            "claude-sonnet-4-0",
            "claude-opus-4-0",
            "claude-3-5-haiku-latest",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        "gemini" => ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        "openrouter" => [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "anthropic/claude-sonnet-4-0",
            "google/gemini-2.5-pro",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        "ollama" => ["llama3.2", "qwen2.5:7b", "mistral"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

fn api_error(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<ApiError>) {
    (
        status,
        Json(ApiError {
            error: message.into(),
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_global_and_request_instructions_as_system_message() {
        let settings = InstructionSettings {
            enabled: true,
            system_prompt: "You are a local automation assistant.".to_string(),
            tool_context: "read_file: inspect a local file".to_string(),
        };
        let messages = apply_instructions(
            &settings,
            Some("Prefer concise answers.".to_string()),
            vec![ChatMessage::text("user", "Summarize this note.")],
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert!(messages[0]
            .content_text()
            .contains("You are a local automation assistant."));
        assert!(messages[0]
            .content_text()
            .contains("Available tools and tool instructions"));
        assert!(messages[0]
            .content_text()
            .contains("Prefer concise answers."));
        assert_eq!(messages[1].role, "user");
    }

    #[test]
    fn request_instructions_apply_even_when_global_settings_are_disabled() {
        let settings = InstructionSettings {
            enabled: false,
            system_prompt: "Ignored".to_string(),
            tool_context: "Ignored".to_string(),
        };
        let messages = apply_instructions(
            &settings,
            Some("Use bullet points.".to_string()),
            vec![ChatMessage::text("user", "List tasks.")],
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert!(messages[0].content_text().contains("Use bullet points."));
        assert!(!messages[0].content_text().contains("Ignored"));
    }

    #[test]
    fn run_summary_masks_instruction_text() {
        let summary = summarize_messages(&[
            ChatMessage::text("system", "Do not leak this instruction."),
            ChatMessage::text("user", "Hello"),
        ]);

        assert_eq!(summary, "system: [instructions applied] | user: Hello");
    }

    #[test]
    fn provider_normalization_requires_unique_display_names() {
        let providers = vec![
            LiteLlmProviderConfig {
                id: "openai_a".to_string(),
                enabled: true,
                provider_type: "openai".to_string(),
                display_name: "OpenAI".to_string(),
                api_key_env_var: "OPENAI_API_KEY".to_string(),
                api_key: None,
                api_base: None,
            },
            LiteLlmProviderConfig {
                id: "openai_b".to_string(),
                enabled: true,
                provider_type: "openai".to_string(),
                display_name: "openai".to_string(),
                api_key_env_var: "OPENAI_WORK_API_KEY".to_string(),
                api_key: None,
                api_base: None,
            },
        ];

        assert!(normalize_litellm_provider_configs(providers, &[]).is_err());
    }

    #[test]
    fn provider_normalization_generates_hidden_id_from_name() {
        let providers = vec![LiteLlmProviderConfig {
            id: "draft_provider".to_string(),
            enabled: true,
            provider_type: "openai".to_string(),
            display_name: "OpenAI Work".to_string(),
            api_key_env_var: String::new(),
            api_key: None,
            api_base: None,
        }];

        let normalized = match normalize_litellm_provider_configs(providers, &[]) {
            Ok(providers) => providers,
            Err(_) => panic!("provider should normalize"),
        };

        assert_eq!(normalized[0].id, "openai_work");
        assert_eq!(normalized[0].api_key_env_var, "OPENAI_API_KEY");
    }
}
