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
    secrets::{self, LITELLM_MASTER_KEY_ENV},
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
    collections::{HashMap, HashSet, VecDeque},
    convert::Infallible,
    env,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::net::TcpStream;
use tokio::process::Command;
use tokio::sync::RwLock;
use tokio::time::sleep;
use uuid::Uuid;

const MAX_RECENT_RUNS: usize = 100;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<RwLock<AppConfig>>,
    pub config_path: PathBuf,
    pub runs_path: PathBuf,
    pub runs: Arc<RwLock<VecDeque<RunRecord>>>,
    pub providers: ProviderRegistry,
    pub litellm_process: Arc<RwLock<Option<ManagedLiteLlmProcess>>>,
    pub started_at: DateTime<Utc>,
}

pub struct ManagedLiteLlmProcess {
    child: tokio::process::Child,
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
struct ApplyLiteLlmProvidersRequest {
    providers: Vec<LiteLlmProviderConfig>,
}

#[derive(Serialize)]
struct ApplyLiteLlmProvidersResponse {
    settings: AppConfig,
    provider_statuses: Vec<ProviderStatus>,
    env_file_path: String,
    config_path: String,
    litellm_ready: bool,
    warning: Option<String>,
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
            "/api/litellm/providers/apply",
            post(apply_litellm_providers),
        )
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
    let app_env = load_app_env_or_default().await;
    let config = hydrate_config_secrets(config, &app_env);
    Json(provider_statuses_for_config(&state, &config, &app_env).await)
}

async fn provider_statuses_for_config(
    state: &AppState,
    config: &AppConfig,
    app_env: &HashMap<String, String>,
) -> Vec<ProviderStatus> {
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
            api_key_configured: Some(provider_api_key_configured(provider, app_env)),
            api_key_env_var: Some(provider.api_key_env_var.clone()),
            base_url: provider
                .api_base
                .clone()
                .or_else(|| Some(config.litellm.base_url.clone())),
        }
    }));

    providers
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
    let has_draft_provider = payload.draft_provider.is_some();
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
    let mut app_env = load_app_env().await?;
    if has_draft_provider {
        overlay_inline_provider_keys(&mut app_env, &config.litellm_providers);
        ensure_litellm_master_key_for_runtime(&mut app_env);
        config.litellm.enabled = true;
        let output_path = draft_litellm_config_path();
        generate_litellm_config(
            &config.litellm_providers,
            &config.model_routes,
            &output_path,
            &config.ollama_endpoint,
        )
        .await
        .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
        let hydrated_config = hydrate_config_secrets(config.clone(), &app_env);
        let (ready, warning) =
            ensure_litellm_runtime_after_apply(&state, &hydrated_config, &output_path, &app_env)
                .await?;
        if !ready {
            return Err(api_error(
                StatusCode::BAD_GATEWAY,
                warning.unwrap_or_else(|| {
                    "LiteLLM did not become ready for the draft provider test.".to_string()
                }),
            ));
        }
        config = hydrated_config;
    } else {
        config = hydrate_config_secrets(config, &app_env);
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

async fn apply_litellm_providers(
    State(state): State<AppState>,
    Json(payload): Json<ApplyLiteLlmProvidersRequest>,
) -> ApiResult<Json<ApplyLiteLlmProvidersResponse>> {
    let env_path = secrets::env_file_path();
    let mut app_env = load_app_env().await?;

    let (config, output_path) = {
        let mut config = state.config.write().await;
        let existing_providers = config.litellm_providers.clone();
        let normalized =
            normalize_litellm_provider_configs(payload.providers, &existing_providers)?;
        let mut updates = secret_updates_for_providers(&normalized, &existing_providers, &app_env)?;

        match config.litellm.api_key.as_deref().map(str::trim) {
            Some(value) if !value.is_empty() && value != REDACTED_SECRET => {
                updates.insert(LITELLM_MASTER_KEY_ENV.to_string(), Some(value.to_string()));
            }
            _ if !secrets::env_value_configured(&app_env, LITELLM_MASTER_KEY_ENV) => {
                updates.insert(
                    LITELLM_MASTER_KEY_ENV.to_string(),
                    Some(format!("sk-lh-{}", Uuid::new_v4().simple())),
                );
            }
            _ => {}
        }

        if !updates.is_empty() {
            secrets::write_env_updates(&env_path, &updates)
                .await
                .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
            app_env = load_app_env().await?;
        }

        ensure_required_provider_keys(&normalized, &app_env)?;

        config.litellm.enabled = true;
        if config.litellm.managed_config_path.is_none() {
            config.litellm.managed_config_path = Some("litellm.config.yaml".to_string());
        }
        config.litellm.api_key = None;
        config.litellm_providers = strip_provider_api_keys(normalized);

        let output_path = resolve_litellm_config_path(&state.config_path, &config, None);
        config::save_config(&state.config_path, &config)
            .await
            .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
        (config.clone(), output_path)
    };

    let generation = generate_litellm_config(
        &config.litellm_providers,
        &config.model_routes,
        &output_path,
        &config.ollama_endpoint,
    )
    .await
    .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    audit_hook(
        "litellm.providers.applied",
        serde_json::json!({
            "providers_written": generation.providers_written,
            "entries_written": generation.entries_written,
            "env_path": env_path.display().to_string(),
            "config_path": output_path.display().to_string()
        }),
    );

    let hydrated_config = hydrate_config_secrets(config.clone(), &app_env);
    let (litellm_ready, warning) =
        ensure_litellm_runtime_after_apply(&state, &hydrated_config, &output_path, &app_env)
            .await?;
    let provider_statuses = provider_statuses_for_config(&state, &hydrated_config, &app_env).await;

    Ok(Json(ApplyLiteLlmProvidersResponse {
        settings: redacted_config_for_response(config, &app_env),
        provider_statuses,
        env_file_path: env_path.display().to_string(),
        config_path: output_path.display().to_string(),
        litellm_ready,
        warning,
    }))
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
    let app_env = load_app_env().await?;
    let config = hydrate_config_secrets(state.config.read().await.clone(), &app_env);
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

    if managed_litellm_is_running(&state).await {
        stop_managed_litellm(&state).await?;
    } else if litellm_port_open(&config.litellm.base_url).await {
        return Err(api_error(
            StatusCode::CONFLICT,
            "a process is already using the LiteLLM port; Llama Harness will not stop an external process",
        ));
    }

    let pid = start_managed_litellm(&state, &config, &output_path, &app_env).await?;

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
        pid: Some(pid),
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
    let config = state.config.read().await.clone();
    let app_env = load_app_env_or_default().await;
    Json(redacted_config_for_response(config, &app_env))
}

async fn update_settings(
    State(state): State<AppState>,
    Json(patch): Json<SettingsPatch>,
) -> ApiResult<Json<AppConfig>> {
    let env_path = secrets::env_file_path();
    let mut app_env = load_app_env().await?;
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
            let existing_litellm_api_key = config.litellm.api_key.clone();
            let mut litellm = litellm;
            match litellm.api_key.as_deref().map(str::trim) {
                Some(value) if value == REDACTED_SECRET => {
                    if !secrets::env_value_configured(&app_env, LITELLM_MASTER_KEY_ENV) {
                        if let Some(existing_secret) = existing_litellm_api_key
                            .as_deref()
                            .map(str::trim)
                            .filter(|value| !value.is_empty() && *value != REDACTED_SECRET)
                        {
                            let mut updates = HashMap::new();
                            updates.insert(
                                LITELLM_MASTER_KEY_ENV.to_string(),
                                Some(existing_secret.to_string()),
                            );
                            secrets::write_env_updates(&env_path, &updates)
                                .await
                                .map_err(|err| {
                                    api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
                                })?;
                            app_env = load_app_env().await?;
                        }
                    }
                }
                Some(value) if !value.is_empty() => {
                    let mut updates = HashMap::new();
                    updates.insert(LITELLM_MASTER_KEY_ENV.to_string(), Some(value.to_string()));
                    secrets::write_env_updates(&env_path, &updates)
                        .await
                        .map_err(|err| {
                            api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
                        })?;
                    app_env = load_app_env().await?;
                }
                _ => {}
            }
            litellm.api_key = None;
            config.litellm = litellm;
        }
        if let Some(litellm_providers) = patch.litellm_providers {
            let existing_providers = config.litellm_providers.clone();
            let normalized =
                normalize_litellm_provider_configs(litellm_providers, &existing_providers)?;
            let updates = secret_updates_for_providers(&normalized, &existing_providers, &app_env)?;
            if !updates.is_empty() {
                secrets::write_env_updates(&env_path, &updates)
                    .await
                    .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
                app_env = load_app_env().await?;
            }
            config.litellm_providers = strip_provider_api_keys(normalized);
        }
        if let Some(model_routes) = patch.model_routes {
            config.model_routes = model_routes;
        }

        config::save_config(&state.config_path, &config)
            .await
            .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
        redacted_config_for_response(config.clone(), &app_env)
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
    let app_env = load_app_env().await?;
    let config = hydrate_config_secrets(state.config.read().await.clone(), &app_env);
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

fn litellm_child_env(
    config: &AppConfig,
    app_env: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut child_env = app_env.clone();
    if let Some(api_key) = config
        .litellm
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != REDACTED_SECRET)
    {
        child_env.insert(LITELLM_MASTER_KEY_ENV.to_string(), api_key.to_string());
    }
    child_env
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

async fn load_app_env() -> ApiResult<HashMap<String, String>> {
    secrets::load_env_file(&secrets::env_file_path())
        .await
        .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))
}

async fn load_app_env_or_default() -> HashMap<String, String> {
    match secrets::load_env_file(&secrets::env_file_path()).await {
        Ok(vars) => vars,
        Err(err) => {
            tracing::warn!(error = %err, "failed to load app env file");
            HashMap::new()
        }
    }
}

fn hydrate_config_secrets(mut config: AppConfig, app_env: &HashMap<String, String>) -> AppConfig {
    if config
        .litellm
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != REDACTED_SECRET)
        .is_none()
    {
        config.litellm.api_key = app_env
            .get(LITELLM_MASTER_KEY_ENV)
            .filter(|value| !value.trim().is_empty())
            .cloned();
    }

    for provider in &mut config.litellm_providers {
        if provider
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty() && *value != REDACTED_SECRET)
            .is_some()
        {
            continue;
        }
        provider.api_key = app_env
            .get(provider.api_key_env_var.trim())
            .filter(|value| !value.trim().is_empty())
            .cloned();
    }

    config
}

fn redacted_config_for_response(
    mut config: AppConfig,
    app_env: &HashMap<String, String>,
) -> AppConfig {
    if config.litellm.api_key.is_some()
        || secrets::env_value_configured(app_env, LITELLM_MASTER_KEY_ENV)
    {
        config.litellm.api_key = Some(REDACTED_SECRET.to_string());
    }

    for provider in &mut config.litellm_providers {
        let env_configured = secrets::env_value_configured(app_env, &provider.api_key_env_var);
        if provider.api_key.is_some() || env_configured {
            provider.api_key = Some(REDACTED_SECRET.to_string());
        }
    }

    config.redacted_for_response()
}

fn secret_updates_for_providers(
    providers: &[LiteLlmProviderConfig],
    existing_providers: &[LiteLlmProviderConfig],
    app_env: &HashMap<String, String>,
) -> ApiResult<HashMap<String, Option<String>>> {
    let mut updates = HashMap::new();
    let existing_keys = existing_providers
        .iter()
        .map(|provider| provider.api_key_env_var.trim())
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .collect::<HashSet<_>>();
    let next_keys = providers
        .iter()
        .map(|provider| provider.api_key_env_var.trim())
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .collect::<HashSet<_>>();

    for key in existing_keys.difference(&next_keys) {
        updates.insert(key.clone(), None);
    }

    for provider in providers {
        let key = provider.api_key_env_var.trim();
        if key.is_empty() {
            continue;
        }
        match provider.api_key.as_deref().map(str::trim) {
            Some(value) if value == REDACTED_SECRET => {
                if !secrets::env_value_configured(app_env, key) {
                    if let Some(existing_secret) = existing_providers
                        .iter()
                        .find(|existing| existing.id == provider.id)
                        .and_then(|existing| existing.api_key.as_deref())
                        .map(str::trim)
                        .filter(|value| !value.is_empty() && *value != REDACTED_SECRET)
                    {
                        updates.insert(key.to_string(), Some(existing_secret.to_string()));
                    } else {
                        return Err(api_error(
                            StatusCode::BAD_REQUEST,
                            format!(
                                "API key for '{}' is marked configured, but no app-data secret exists.",
                                provider_display_name(provider)
                            ),
                        ));
                    }
                }
            }
            Some(value) if !value.is_empty() => {
                updates.insert(key.to_string(), Some(value.to_string()));
            }
            _ => {}
        }
    }

    Ok(updates)
}

fn ensure_required_provider_keys(
    providers: &[LiteLlmProviderConfig],
    app_env: &HashMap<String, String>,
) -> ApiResult<()> {
    for provider in providers {
        if !provider.enabled || normalize_provider_type(&provider.provider_type) == "ollama" {
            continue;
        }
        if !secrets::env_value_configured(app_env, &provider.api_key_env_var) {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "API key is required for '{}'.",
                    provider_display_name(provider)
                ),
            ));
        }
    }
    Ok(())
}

fn strip_provider_api_keys(providers: Vec<LiteLlmProviderConfig>) -> Vec<LiteLlmProviderConfig> {
    providers
        .into_iter()
        .map(|mut provider| {
            provider.api_key = None;
            provider
        })
        .collect()
}

fn overlay_inline_provider_keys(
    app_env: &mut HashMap<String, String>,
    providers: &[LiteLlmProviderConfig],
) {
    for provider in providers {
        let key = provider.api_key_env_var.trim();
        if key.is_empty() {
            continue;
        }
        if let Some(value) = provider
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty() && *value != REDACTED_SECRET)
        {
            app_env.insert(key.to_string(), value.to_string());
        }
    }
}

fn ensure_litellm_master_key_for_runtime(app_env: &mut HashMap<String, String>) {
    if !secrets::env_value_configured(app_env, LITELLM_MASTER_KEY_ENV) {
        app_env.insert(
            LITELLM_MASTER_KEY_ENV.to_string(),
            format!("sk-lh-{}", Uuid::new_v4().simple()),
        );
    }
}

fn draft_litellm_config_path() -> PathBuf {
    secrets::app_data_dir().join("litellm.draft.config.yaml")
}

async fn ensure_litellm_runtime_after_apply(
    state: &AppState,
    config: &AppConfig,
    output_path: &Path,
    app_env: &HashMap<String, String>,
) -> ApiResult<(bool, Option<String>)> {
    if managed_litellm_is_running(state).await {
        stop_managed_litellm(state).await?;
        let _pid = start_managed_litellm(state, config, output_path, app_env).await?;
        let ready = wait_for_litellm_ready(state, config).await;
        return Ok((
            ready,
            (!ready).then(|| {
                "LiteLLM was restarted, but it did not become ready before the health check timed out.".to_string()
            }),
        ));
    }

    if litellm_port_open(&config.litellm.base_url).await {
        let ready = wait_for_litellm_ready(state, config).await;
        return Ok((
            ready,
            Some(if ready {
                "LiteLLM is already running outside Llama Harness. Saved keys and config are ready, but restart that proxy if new providers do not appear.".to_string()
            } else {
                "A process is already using the LiteLLM port, and it did not respond as a ready LiteLLM proxy. Llama Harness did not stop it.".to_string()
            }),
        ));
    }

    let _pid = start_managed_litellm(state, config, output_path, app_env).await?;
    let ready = wait_for_litellm_ready(state, config).await;
    Ok((
        ready,
        (!ready).then(|| {
            "LiteLLM started, but it did not become ready before the health check timed out."
                .to_string()
        }),
    ))
}

async fn managed_litellm_is_running(state: &AppState) -> bool {
    let mut process = state.litellm_process.write().await;
    let should_clear = match process.as_mut() {
        Some(managed) => match managed.child.try_wait() {
            Ok(Some(_status)) => true,
            Ok(None) => false,
            Err(err) => {
                tracing::warn!(error = %err, "failed to inspect managed LiteLLM process");
                true
            }
        },
        None => return false,
    };

    if should_clear {
        *process = None;
        false
    } else {
        true
    }
}

async fn stop_managed_litellm(state: &AppState) -> ApiResult<()> {
    let mut managed = {
        let mut process = state.litellm_process.write().await;
        process.take()
    };

    if let Some(mut process) = managed.take() {
        process
            .child
            .kill()
            .await
            .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
        let _ = process.child.wait().await;
    }

    Ok(())
}

async fn start_managed_litellm(
    state: &AppState,
    config: &AppConfig,
    output_path: &Path,
    app_env: &HashMap<String, String>,
) -> ApiResult<u32> {
    let command =
        env::var("LLAMA_HARNESS_LITELLM_COMMAND").unwrap_or_else(|_| "litellm".to_string());
    let (host, port) = litellm_host_port(&config.litellm.base_url);
    let child = Command::new(&command)
        .arg("--config")
        .arg(output_path)
        .arg("--host")
        .arg(&host)
        .arg("--port")
        .arg(&port)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .envs(litellm_child_env(config, app_env))
        .spawn()
        .map_err(|err| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to start LiteLLM using '{command}': {err}"),
            )
        })?;
    let pid = child.id().unwrap_or(0);
    let mut process = state.litellm_process.write().await;
    *process = Some(ManagedLiteLlmProcess { child });
    Ok(pid)
}

async fn wait_for_litellm_ready(state: &AppState, config: &AppConfig) -> bool {
    for _ in 0..20 {
        if state.providers.litellm_healthy(config).await {
            return true;
        }
        sleep(Duration::from_millis(250)).await;
    }
    false
}

async fn litellm_port_open(base_url: &str) -> bool {
    let (host, port) = litellm_host_port(base_url);
    let Ok(port) = port.parse::<u16>() else {
        return false;
    };
    TcpStream::connect((host.as_str(), port)).await.is_ok()
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
    let mut used_env_vars: Vec<String> = Vec::new();

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
        if used_ids.iter().any(|id| id == &provider.id) {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!("provider id '{}' is already used", provider.id),
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
        let env_key = provider.api_key_env_var.trim();
        if !env_key.is_empty() {
            if used_env_vars.iter().any(|key| key == env_key) {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    format!("provider API key environment variable '{env_key}' is already used"),
                ));
            }
            used_env_vars.push(env_key.to_string());
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

    provider.api_key_env_var = provider_api_key_env_var(&provider, existing_providers);

    provider.api_key = match provider.api_key {
        Some(value) if value == REDACTED_SECRET => Some(REDACTED_SECRET.to_string()),
        Some(value) if value.trim().is_empty() => None,
        Some(value) => Some(value.trim().to_string()),
        None => None,
    };

    provider.api_base = provider
        .api_base
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    Ok(provider)
}

fn provider_api_key_env_var(
    provider: &LiteLlmProviderConfig,
    existing_providers: &[LiteLlmProviderConfig],
) -> String {
    if provider.provider_type == "ollama" {
        return String::new();
    }

    if let Some(existing_key) = existing_providers
        .iter()
        .find(|existing| normalize_provider_id(&existing.id) == provider.id)
        .map(|existing| existing.api_key_env_var.trim())
        .filter(|key| !key.is_empty())
    {
        return existing_key.to_string();
    }

    api_key_env_var_from_provider_id(&provider.id)
}

fn api_key_env_var_from_provider_id(provider_id: &str) -> String {
    let slug = provider_id
        .trim()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string();
    let slug = if slug.is_empty() {
        "PROVIDER".to_string()
    } else {
        slug
    };
    format!("{slug}_API_KEY")
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

fn provider_api_key_configured(
    provider: &LiteLlmProviderConfig,
    app_env: &HashMap<String, String>,
) -> bool {
    provider
        .api_key
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty())
        || secrets::env_value_configured(app_env, &provider.api_key_env_var)
        || (!provider.api_key_env_var.trim().is_empty()
            && env::var(&provider.api_key_env_var)
                .ok()
                .is_some_and(|value| !value.trim().is_empty()))
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
        assert_eq!(normalized[0].api_key_env_var, "OPENAI_WORK_API_KEY");
    }

    #[test]
    fn provider_normalization_allows_duplicate_enabled_types_with_unique_identity() {
        let providers = vec![
            LiteLlmProviderConfig {
                id: "openai_work".to_string(),
                enabled: true,
                provider_type: "openai".to_string(),
                display_name: "OpenAI Work".to_string(),
                api_key_env_var: "OPENAI_API_KEY".to_string(),
                api_key: Some("sk-work".to_string()),
                api_base: None,
            },
            LiteLlmProviderConfig {
                id: "openai_personal".to_string(),
                enabled: true,
                provider_type: "openai".to_string(),
                display_name: "OpenAI Personal".to_string(),
                api_key_env_var: "OPENAI_PERSONAL_API_KEY".to_string(),
                api_key: Some("sk-personal".to_string()),
                api_base: None,
            },
        ];

        let normalized = match normalize_litellm_provider_configs(providers, &[]) {
            Ok(providers) => providers,
            Err(_) => panic!("providers with unique names and ids should normalize"),
        };

        assert_eq!(normalized.len(), 2);
        assert_eq!(normalized[0].api_key_env_var, "OPENAI_WORK_API_KEY");
        assert_eq!(normalized[1].api_key_env_var, "OPENAI_PERSONAL_API_KEY");
    }

    #[test]
    fn provider_normalization_rejects_duplicate_ids() {
        let providers = vec![
            LiteLlmProviderConfig {
                id: "openai".to_string(),
                enabled: true,
                provider_type: "openai".to_string(),
                display_name: "OpenAI Work".to_string(),
                api_key_env_var: String::new(),
                api_key: Some("sk-work".to_string()),
                api_base: None,
            },
            LiteLlmProviderConfig {
                id: "openai".to_string(),
                enabled: true,
                provider_type: "openai".to_string(),
                display_name: "OpenAI Personal".to_string(),
                api_key_env_var: String::new(),
                api_key: Some("sk-personal".to_string()),
                api_base: None,
            },
        ];

        assert!(normalize_litellm_provider_configs(providers, &[]).is_err());
    }

    #[test]
    fn provider_normalization_rejects_duplicate_env_vars() {
        let existing = vec![
            LiteLlmProviderConfig {
                id: "openai_work".to_string(),
                enabled: true,
                provider_type: "openai".to_string(),
                display_name: "OpenAI Work".to_string(),
                api_key_env_var: "OPENAI_API_KEY".to_string(),
                api_key: None,
                api_base: None,
            },
            LiteLlmProviderConfig {
                id: "openai_personal".to_string(),
                enabled: true,
                provider_type: "openai".to_string(),
                display_name: "OpenAI Personal".to_string(),
                api_key_env_var: "OPENAI_API_KEY".to_string(),
                api_key: None,
                api_base: None,
            },
        ];

        assert!(normalize_litellm_provider_configs(existing.clone(), &existing).is_err());
    }

    #[test]
    fn provider_rename_preserves_existing_env_var() {
        let existing = vec![LiteLlmProviderConfig {
            id: "openai_main".to_string(),
            enabled: true,
            provider_type: "openai".to_string(),
            display_name: "OpenAI".to_string(),
            api_key_env_var: "OPENAI_API_KEY".to_string(),
            api_key: None,
            api_base: None,
        }];
        let providers = vec![LiteLlmProviderConfig {
            display_name: "Renamed OpenAI".to_string(),
            ..existing[0].clone()
        }];

        let normalized = match normalize_litellm_provider_configs(providers, &existing) {
            Ok(providers) => providers,
            Err(_) => panic!("rename should preserve provider identity"),
        };

        assert_eq!(normalized[0].id, "openai_main");
        assert_eq!(normalized[0].api_key_env_var, "OPENAI_API_KEY");
    }

    #[test]
    fn redacted_response_marks_app_env_provider_key_configured() {
        let mut config = AppConfig::default();
        config.litellm_providers.push(LiteLlmProviderConfig {
            id: "openai_main".to_string(),
            enabled: true,
            provider_type: "openai".to_string(),
            display_name: "OpenAI".to_string(),
            api_key_env_var: "OPENAI_API_KEY".to_string(),
            api_key: None,
            api_base: None,
        });
        let mut app_env = HashMap::new();
        app_env.insert("OPENAI_API_KEY".to_string(), "sk-secret".to_string());

        let redacted = redacted_config_for_response(config, &app_env);

        assert_eq!(
            redacted.litellm_providers[0].api_key.as_deref(),
            Some(REDACTED_SECRET)
        );
    }

    #[test]
    fn configured_secret_preserves_existing_raw_value_for_env_migration() {
        let existing = vec![LiteLlmProviderConfig {
            id: "openai_main".to_string(),
            enabled: true,
            provider_type: "openai".to_string(),
            display_name: "OpenAI".to_string(),
            api_key_env_var: "OPENAI_API_KEY".to_string(),
            api_key: Some("sk-existing".to_string()),
            api_base: None,
        }];
        let providers = vec![LiteLlmProviderConfig {
            api_key: Some(REDACTED_SECRET.to_string()),
            ..existing[0].clone()
        }];
        let updates = match secret_updates_for_providers(&providers, &existing, &HashMap::new()) {
            Ok(updates) => updates,
            Err(_) => panic!("configured provider should migrate existing secret"),
        };

        assert_eq!(
            updates
                .get("OPENAI_API_KEY")
                .and_then(|value| value.as_deref()),
            Some("sk-existing")
        );
    }

    #[test]
    fn deleted_provider_schedules_secret_removal() {
        let existing = vec![LiteLlmProviderConfig {
            id: "openai_main".to_string(),
            enabled: true,
            provider_type: "openai".to_string(),
            display_name: "OpenAI".to_string(),
            api_key_env_var: "OPENAI_API_KEY".to_string(),
            api_key: None,
            api_base: None,
        }];

        let updates = match secret_updates_for_providers(&[], &existing, &HashMap::new()) {
            Ok(updates) => updates,
            Err(_) => panic!("deleted provider should produce secret updates"),
        };

        assert!(matches!(updates.get("OPENAI_API_KEY"), Some(None)));
    }

    #[test]
    fn secret_removal_preserves_keys_still_referenced_by_remaining_provider() {
        let existing = vec![
            LiteLlmProviderConfig {
                id: "openai_work".to_string(),
                enabled: true,
                provider_type: "openai".to_string(),
                display_name: "OpenAI Work".to_string(),
                api_key_env_var: "OPENAI_API_KEY".to_string(),
                api_key: None,
                api_base: None,
            },
            LiteLlmProviderConfig {
                id: "openai_personal".to_string(),
                enabled: true,
                provider_type: "openai".to_string(),
                display_name: "OpenAI Personal".to_string(),
                api_key_env_var: "OPENAI_API_KEY".to_string(),
                api_key: None,
                api_base: None,
            },
        ];
        let remaining = vec![existing[1].clone()];

        let updates = match secret_updates_for_providers(&remaining, &existing, &HashMap::new()) {
            Ok(updates) => updates,
            Err(_) => panic!("remaining provider should preserve shared secret key"),
        };

        assert!(!updates.contains_key("OPENAI_API_KEY"));
    }

    #[test]
    fn strip_provider_api_keys_removes_raw_values_before_config_save() {
        let providers = vec![LiteLlmProviderConfig {
            id: "openai_main".to_string(),
            enabled: true,
            provider_type: "openai".to_string(),
            display_name: "OpenAI".to_string(),
            api_key_env_var: "OPENAI_API_KEY".to_string(),
            api_key: Some("sk-secret".to_string()),
            api_base: None,
        }];

        let stripped = strip_provider_api_keys(providers);

        assert!(stripped[0].api_key.is_none());
    }

    #[test]
    fn litellm_child_env_includes_app_data_values() {
        let config = AppConfig::default();
        let mut app_env = HashMap::new();
        app_env.insert("OPENAI_API_KEY".to_string(), "sk-secret".to_string());
        app_env.insert(LITELLM_MASTER_KEY_ENV.to_string(), "sk-master".to_string());

        let child_env = litellm_child_env(&config, &app_env);

        assert_eq!(
            child_env.get("OPENAI_API_KEY").map(String::as_str),
            Some("sk-secret")
        );
        assert_eq!(
            child_env.get(LITELLM_MASTER_KEY_ENV).map(String::as_str),
            Some("sk-master")
        );
    }

    #[test]
    fn draft_provider_inline_key_overlays_runtime_env_only() {
        let providers = vec![LiteLlmProviderConfig {
            id: "openai_draft".to_string(),
            enabled: true,
            provider_type: "openai".to_string(),
            display_name: "OpenAI Draft".to_string(),
            api_key_env_var: "OPENAI_API_KEY".to_string(),
            api_key: Some("sk-draft".to_string()),
            api_base: None,
        }];
        let mut app_env = HashMap::new();

        overlay_inline_provider_keys(&mut app_env, &providers);

        assert_eq!(
            app_env.get("OPENAI_API_KEY").map(String::as_str),
            Some("sk-draft")
        );
        assert_eq!(providers[0].api_key.as_deref(), Some("sk-draft"));
    }
}
