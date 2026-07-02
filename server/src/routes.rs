use crate::{
    app_policy::{
        self, AppCapabilitiesResponse, AppPolicyError, AppPolicyErrorKind, ClientAppConfig,
        DomainCatalog, ToolConfig,
    },
    config::{
        self, AgentConfig, AgentPermissions, AppConfig, GenerationSettings, InstructionSettings,
        LiteLlmProviderConfig, LiteLlmSettings, ModelRoute, REDACTED_SECRET,
    },
    litellm::{generate_litellm_config, litellm_model_for_provider},
    litellm_runtime::{litellm_start_command_summary, LiteLlmRuntimeManager},
    ollama::OllamaModel,
    providers::{
        ChatMessage, MessageContent, ModelProvider, ProviderChatRequest, ProviderChatResponse,
        ProviderRegistry, ProviderStreamEvent, TokenUsage,
    },
    runs::{self, AuditLevel, AuditRecord, RunRecord, RunStatus},
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
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use uuid::Uuid;

const MAX_RECENT_RUNS: usize = 100;
const MAX_RECENT_AUDIT: usize = 100;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<RwLock<AppConfig>>,
    pub config_path: PathBuf,
    pub catalog: Arc<RwLock<DomainCatalog>>,
    pub catalog_dir: PathBuf,
    pub runs_path: PathBuf,
    pub runs: Arc<RwLock<VecDeque<RunRecord>>>,
    pub(crate) pending_runs: Arc<RwLock<HashMap<String, PendingRun>>>,
    pub audit_path: PathBuf,
    pub audit: Arc<RwLock<VecDeque<AuditRecord>>>,
    pub providers: ProviderRegistry,
    pub litellm_runtime: LiteLlmRuntimeManager,
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

#[derive(Serialize)]
struct SetupStatusResponse {
    litellm_enabled: bool,
    litellm_ready: bool,
    usable_provider_count: usize,
    usable_model_count: usize,
    active_agent_count: usize,
    ready: bool,
    next_step: SetupStep,
    missing_steps: Vec<SetupStep>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum SetupStep {
    StartLitellm,
    AddProvider,
    SelectModel,
    CreateAgent,
    Ready,
}

#[derive(Deserialize)]
struct AgentCreateRequest {
    id: Option<String>,
    name: Option<String>,
    role: Option<String>,
    description: Option<String>,
    system_prompt: Option<String>,
    default_model_id: Option<String>,
    default_provider_id: Option<String>,
    default_model: Option<String>,
    default_environment: Option<String>,
    autonomy: Option<String>,
    permissions: Option<AgentPermissions>,
    allowed_tool_ids: Option<Vec<String>>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    enabled: Option<bool>,
    status: Option<String>,
}

#[derive(Deserialize)]
struct AgentPatch {
    name: Option<String>,
    description: Option<String>,
    system_prompt: Option<String>,
    default_model_id: Option<String>,
    default_provider_id: Option<String>,
    default_model: Option<String>,
    allowed_tool_ids: Option<Vec<String>>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    enabled: Option<bool>,
    status: Option<String>,
}

#[derive(Deserialize)]
struct AgentChatRequest {
    messages: Option<Vec<ChatMessage>>,
    prompt: Option<String>,
    instructions: Option<String>,
    app_context: Option<serde_json::Value>,
    generation: Option<GenerationSettings>,
    tools: Option<serde_json::Value>,
    tool_choice: Option<serde_json::Value>,
    source_app: Option<String>,
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct AgentChatResponse {
    run_id: String,
    agent_id: String,
    provider: String,
    model: String,
    message: ChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<serde_json::Value>,
    usage: Option<TokenUsage>,
    started_at: DateTime<Utc>,
    ended_at: DateTime<Utc>,
    duration_ms: u64,
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
struct AuditQuery {
    limit: Option<usize>,
}

#[derive(Serialize)]
struct AuditResponse {
    audit: Vec<AuditRecord>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppPatch {
    name: Option<String>,
    description: Option<Option<String>>,
    default_agent_id: Option<String>,
    allowed_agent_ids: Option<Vec<String>>,
    allowed_tool_ids: Option<Option<Vec<String>>>,
    enabled: Option<bool>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolPatch {
    name: Option<String>,
    description: Option<String>,
    risk_level: Option<String>,
    enabled: Option<bool>,
    input_schema: Option<Option<serde_json::Value>>,
    output_schema: Option<Option<serde_json::Value>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunCreateRequest {
    app_id: String,
    agent_id: Option<String>,
    input: Option<String>,
    messages: Option<Vec<ChatMessage>>,
    instructions: Option<String>,
    context: Option<serde_json::Value>,
    generation: Option<GenerationSettings>,
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunCreateResponse {
    run_id: String,
    status: RunStatus,
    app_id: String,
    agent_id: String,
    model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_requests: Vec<RunToolRequest>,
    duration_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RunToolRequest {
    id: String,
    tool_id: String,
    name: String,
    arguments: serde_json::Value,
    risk_level: String,
    display_name: String,
}

#[derive(Clone, Debug)]
pub(crate) struct PendingRun {
    app_id: String,
    agent_id: String,
    model_id: String,
    provider_id: String,
    provider_model: String,
    resolved_model_name: String,
    resolved_tool_ids: Vec<String>,
    tools: Vec<ToolConfig>,
    messages: Vec<ChatMessage>,
    provider_tool_calls: serde_json::Value,
    tool_requests: Vec<RunToolRequest>,
    generation: GenerationSettings,
    metadata: Option<serde_json::Value>,
    started_at: DateTime<Utc>,
    prompt_summary: String,
    usage: Option<TokenUsage>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunToolResultsRequest {
    app_id: String,
    tool_results: Vec<RunToolResultInput>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunToolResultInput {
    tool_call_id: String,
    tool_id: Option<String>,
    result: Option<serde_json::Value>,
    error: Option<String>,
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
struct StreamEvent<'a> {
    run_id: &'a str,
    content: &'a str,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/models", get(list_models))
        .route("/agents", get(list_agents))
        .route("/apps", get(list_apps))
        .route("/apps/:app_id", get(get_app).patch(patch_app))
        .route("/apps/:app_id/capabilities", get(app_capabilities))
        .route("/tools", get(list_tools))
        .route("/tools/:tool_id", get(get_tool).patch(patch_tool))
        .route("/runs", get(list_runs).post(create_run))
        .route("/runs/:run_id/tool-results", post(submit_run_tool_results))
        .route("/runs/stream", post(stream_run))
        .route("/audit", get(list_audit))
        .route("/api/health", get(health))
        .route("/api/models", get(list_models))
        .route("/api/models/default", post(set_default_model))
        .route("/api/models/test", post(test_model))
        .route("/api/setup/status", get(setup_status))
        .route("/api/agents", get(list_agents).post(create_agent))
        .route("/api/agents/:agent_id", get(get_agent).patch(patch_agent))
        .route("/api/agents/:agent_id/chat", post(agent_chat))
        .route("/api/apps", get(list_apps))
        .route("/api/apps/:app_id", get(get_app).patch(patch_app))
        .route("/api/apps/:app_id/capabilities", get(app_capabilities))
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
        .route("/api/chat/completions", post(chat_completions))
        .route("/api/chat", post(chat))
        .route("/api/chat/stream", post(stream_chat))
        .route("/api/runs", get(list_runs).post(create_run))
        .route(
            "/api/runs/:run_id/tool-results",
            post(submit_run_tool_results),
        )
        .route("/api/runs/stream", post(stream_run))
        .route("/api/audit", get(list_audit))
        .route("/api/settings", get(get_settings).put(update_settings))
        .route("/api/tools", get(list_tools))
        .route("/api/tools/:tool_id", get(get_tool).patch(patch_tool))
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

async fn setup_status(State(state): State<AppState>) -> Json<SetupStatusResponse> {
    let app_env = load_app_env_or_default().await;
    let config = state.config.read().await.clone();
    let hydrated_config = hydrate_config_secrets(config.clone(), &app_env);
    let ollama_models = state
        .providers
        .list_ollama_models(&hydrated_config)
        .await
        .ok();
    let litellm_ready = state
        .litellm_runtime
        .readiness_healthy(&hydrated_config)
        .await;

    let mut usable_provider_count = 0;
    let mut usable_model_count = 0;
    if let Some(models) = &ollama_models {
        if !models.is_empty() {
            usable_provider_count += 1;
            usable_model_count += models.len();
        }
    }

    if hydrated_config.litellm.enabled && litellm_ready {
        for provider in hydrated_config
            .litellm_providers
            .iter()
            .filter(|provider| provider.enabled)
        {
            usable_provider_count += 1;
            usable_model_count += suggested_provider_models(provider).len().max(1);
        }
    }

    let active_agent_count = state
        .catalog
        .read()
        .await
        .agents
        .iter()
        .filter(|agent| app_policy::agent_enabled(agent))
        .count();
    let mut missing_steps = Vec::new();
    if config.litellm.enabled && !litellm_ready {
        missing_steps.push(SetupStep::StartLitellm);
    }
    if usable_provider_count == 0 {
        missing_steps.push(SetupStep::AddProvider);
    }
    if usable_model_count == 0 {
        missing_steps.push(SetupStep::SelectModel);
    }
    if active_agent_count == 0 {
        missing_steps.push(SetupStep::CreateAgent);
    }
    let next_step = missing_steps.first().copied().unwrap_or(SetupStep::Ready);

    Json(SetupStatusResponse {
        litellm_enabled: config.litellm.enabled,
        litellm_ready,
        usable_provider_count,
        usable_model_count,
        active_agent_count,
        ready: missing_steps.is_empty(),
        next_step,
        missing_steps,
    })
}

async fn list_agents(State(state): State<AppState>) -> Json<Vec<AgentConfig>> {
    Json(state.catalog.read().await.agents.clone())
}

async fn get_agent(
    State(state): State<AppState>,
    AxumPath(agent_id): AxumPath<String>,
) -> ApiResult<Json<AgentConfig>> {
    let catalog = state.catalog.read().await;
    let agent = app_policy::find_agent(&catalog, &agent_id)
        .cloned()
        .ok_or_else(|| api_error(StatusCode::NOT_FOUND, format!("unknown agent: {agent_id}")))?;
    Ok(Json(agent))
}

async fn create_agent(
    State(state): State<AppState>,
    Json(payload): Json<AgentCreateRequest>,
) -> ApiResult<Json<AgentConfig>> {
    let config = state.config.read().await.clone();
    let catalog_snapshot = state.catalog.read().await.clone();
    let existing_ids = catalog_snapshot
        .agents
        .iter()
        .map(|agent| agent.id.clone())
        .collect::<Vec<_>>();
    let mut agent = agent_from_create(payload, &existing_ids)?;
    validate_agent_defaults(&state, &config, &agent).await?;

    let saved = {
        let mut catalog = state.catalog.write().await;
        if catalog
            .agents
            .iter()
            .any(|existing| existing.id.eq_ignore_ascii_case(&agent.id))
        {
            return Err(api_error(
                StatusCode::CONFLICT,
                format!("agent id '{}' is already used", agent.id),
            ));
        }
        agent.updated_at = timestamp_now();
        catalog.agents.push(agent.clone());
        app_policy::save_agents(&state.catalog_dir, &catalog.agents)
            .await
            .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
        agent
    };

    Ok(Json(saved))
}

async fn patch_agent(
    State(state): State<AppState>,
    AxumPath(agent_id): AxumPath<String>,
    Json(value): Json<serde_json::Value>,
) -> ApiResult<Json<AgentConfig>> {
    reject_protected_agent_patch_fields(&value)?;
    let patch: AgentPatch = serde_json::from_value(value)
        .map_err(|err| api_error(StatusCode::BAD_REQUEST, err.to_string()))?;

    let config = state.config.read().await.clone();
    let catalog_snapshot = state.catalog.read().await.clone();
    let current = app_policy::find_agent(&catalog_snapshot, &agent_id)
        .cloned()
        .ok_or_else(|| api_error(StatusCode::NOT_FOUND, format!("unknown agent: {agent_id}")))?;
    let mut agent = current;
    apply_agent_patch(&mut agent, patch)?;
    validate_agent_defaults(&state, &config, &agent).await?;

    let saved = {
        let mut catalog = state.catalog.write().await;
        let existing = app_policy::find_agent_mut(&mut catalog, &agent_id).ok_or_else(|| {
            api_error(StatusCode::NOT_FOUND, format!("unknown agent: {agent_id}"))
        })?;
        agent.updated_at = timestamp_now();
        *existing = agent.clone();
        app_policy::save_agents(&state.catalog_dir, &catalog.agents)
            .await
            .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
        agent
    };

    Ok(Json(saved))
}

async fn list_apps(State(state): State<AppState>) -> Json<Vec<ClientAppConfig>> {
    Json(state.catalog.read().await.apps.clone())
}

async fn get_app(
    State(state): State<AppState>,
    AxumPath(app_id): AxumPath<String>,
) -> ApiResult<Json<ClientAppConfig>> {
    let catalog = state.catalog.read().await;
    let app = app_policy::find_app(&catalog, &app_id)
        .cloned()
        .ok_or_else(|| api_error(StatusCode::NOT_FOUND, format!("unknown app: {app_id}")))?;
    Ok(Json(app))
}

async fn patch_app(
    State(state): State<AppState>,
    AxumPath(app_id): AxumPath<String>,
    Json(patch): Json<AppPatch>,
) -> ApiResult<Json<ClientAppConfig>> {
    let saved = {
        let mut catalog = state.catalog.write().await;
        let app = app_policy::find_app_mut(&mut catalog, &app_id)
            .ok_or_else(|| api_error(StatusCode::NOT_FOUND, format!("unknown app: {app_id}")))?;

        if let Some(name) = patch.name {
            app.name = trim_or_default(name, "Untitled app");
        }
        if let Some(description) = patch.description {
            app.description = description.map(|value| value.trim().to_string());
        }
        if let Some(default_agent_id) = patch.default_agent_id {
            let default_agent_id = default_agent_id.trim();
            if default_agent_id.is_empty() {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    "defaultAgentId is required",
                ));
            }
            app.default_agent_id = default_agent_id.to_string();
            if !app
                .allowed_agent_ids
                .iter()
                .any(|agent_id| agent_id.eq_ignore_ascii_case(default_agent_id))
            {
                app.allowed_agent_ids.push(default_agent_id.to_string());
            }
        }
        if let Some(allowed_agent_ids) = patch.allowed_agent_ids {
            app.allowed_agent_ids = normalize_id_list(allowed_agent_ids);
            if !app
                .allowed_agent_ids
                .iter()
                .any(|agent_id| agent_id.eq_ignore_ascii_case(&app.default_agent_id))
            {
                app.allowed_agent_ids.push(app.default_agent_id.clone());
            }
        }
        if let Some(allowed_tool_ids) = patch.allowed_tool_ids {
            app.allowed_tool_ids = allowed_tool_ids.map(normalize_id_list);
        }
        if let Some(enabled) = patch.enabled {
            app.enabled = enabled;
        }

        let app = app.clone();
        app_policy::save_apps(&state.catalog_dir, &catalog.apps)
            .await
            .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
        app
    };

    Ok(Json(saved))
}

async fn app_capabilities(
    State(state): State<AppState>,
    AxumPath(app_id): AxumPath<String>,
) -> ApiResult<Json<AppCapabilitiesResponse>> {
    let (config, catalog) = {
        (
            state.config.read().await.clone(),
            state.catalog.read().await.clone(),
        )
    };
    let fallback_model = first_ollama_model_name(&state, &config).await;
    let resolved =
        app_policy::resolve_app_policy(&catalog, &config, &app_id, None, fallback_model.as_deref())
            .map_err(app_policy_api_error)?;
    record_audit(
        &state,
        audit_record(
            "app.capabilities_resolved",
            AuditLevel::Info,
            format!("Resolved capabilities for app '{}'", resolved.app.id),
            Some(resolved.app.id.clone()),
            Some(resolved.agent.id.clone()),
            None,
            Some(serde_json::json!({
                "tool_ids": resolved.tools.iter().map(|tool| tool.id.clone()).collect::<Vec<_>>(),
                "model_id": resolved.model.id,
                "warnings": resolved.warnings,
            })),
        ),
    )
    .await;

    Ok(Json(AppCapabilitiesResponse::from_policy(&resolved)))
}

async fn list_tools(State(state): State<AppState>) -> Json<Vec<ToolConfig>> {
    Json(state.catalog.read().await.tools.clone())
}

async fn get_tool(
    State(state): State<AppState>,
    AxumPath(tool_id): AxumPath<String>,
) -> ApiResult<Json<ToolConfig>> {
    let catalog = state.catalog.read().await;
    let tool = catalog
        .tools
        .iter()
        .find(|tool| tool.id.eq_ignore_ascii_case(tool_id.trim()))
        .cloned()
        .ok_or_else(|| api_error(StatusCode::NOT_FOUND, format!("unknown tool: {tool_id}")))?;
    Ok(Json(tool))
}

async fn patch_tool(
    State(state): State<AppState>,
    AxumPath(tool_id): AxumPath<String>,
    Json(patch): Json<ToolPatch>,
) -> ApiResult<Json<ToolConfig>> {
    let saved = {
        let mut catalog = state.catalog.write().await;
        let tool = catalog
            .tools
            .iter_mut()
            .find(|tool| tool.id.eq_ignore_ascii_case(tool_id.trim()))
            .ok_or_else(|| api_error(StatusCode::NOT_FOUND, format!("unknown tool: {tool_id}")))?;

        if let Some(name) = patch.name {
            tool.name = trim_or_default(name, "Untitled tool");
        }
        if let Some(description) = patch.description {
            tool.description = description.trim().to_string();
        }
        if let Some(risk_level) = patch.risk_level {
            let risk_level = risk_level.trim().to_ascii_lowercase();
            if !["low", "medium", "high"].contains(&risk_level.as_str()) {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    format!("invalid riskLevel '{risk_level}'"),
                ));
            }
            tool.risk_level = risk_level;
        }
        if let Some(enabled) = patch.enabled {
            tool.enabled = enabled;
        }
        if let Some(input_schema) = patch.input_schema {
            tool.input_schema = input_schema;
        }
        if let Some(output_schema) = patch.output_schema {
            tool.output_schema = output_schema;
        }

        let tool = tool.clone();
        app_policy::save_tools(&state.catalog_dir, &catalog.tools)
            .await
            .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
        tool
    };

    Ok(Json(saved))
}

async fn agent_chat(
    State(state): State<AppState>,
    AxumPath(agent_id): AxumPath<String>,
    Json(payload): Json<AgentChatRequest>,
) -> ApiResult<Json<AgentChatResponse>> {
    let app_env = load_app_env().await?;
    let config = hydrate_config_secrets(state.config.read().await.clone(), &app_env);
    let catalog = state.catalog.read().await.clone();
    let agent = app_policy::find_agent(&catalog, &agent_id)
        .cloned()
        .ok_or_else(|| api_error(StatusCode::NOT_FOUND, format!("unknown agent: {agent_id}")))?;
    if !app_policy::agent_enabled(&agent) {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!("agent '{}' is not active", agent.id),
        ));
    }
    validate_agent_defaults(&state, &config, &agent).await?;

    let source_app = payload
        .source_app
        .unwrap_or_else(|| "agent-api".to_string());
    let messages = normalize_messages(payload.messages, payload.prompt)?;
    let messages = apply_agent_context(
        &config.instructions,
        &agent,
        payload.instructions,
        payload.app_context,
        messages,
    );
    let tools = payload.tools.or_else(|| {
        app_policy::resolve_app_policy(&catalog, &config, &source_app, Some(&agent.id), None)
            .ok()
            .and_then(|policy| provider_tools_from_policy(&policy.tools))
    });
    let tool_choice = payload
        .tool_choice
        .or_else(|| tools.as_ref().map(|_| serde_json::json!("auto")));
    let metadata = Some(agent_chat_metadata(
        payload.metadata,
        &source_app,
        &agent.id,
    ));

    let resolved = resolve_chat(
        &state,
        ChatRequest {
            provider: Some(agent.default_provider_id.clone()),
            model: Some(agent.default_model.clone()),
            source_app: Some(source_app.clone()),
            messages: Some(messages),
            prompt: None,
            instructions: None,
            generation: payload.generation,
            tools: tools.clone(),
            tool_choice,
            stream: Some(false),
            metadata,
        },
    )
    .await?;

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
                app_id: Some(source_app.clone()),
                agent_id: Some(agent.id.clone()),
                model_id: app_policy::model_config_for_agent(&catalog, &agent)
                    .map(|model| model.id.clone()),
                resolved_tool_ids: tool_ids_from_provider_tools(&tools),
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
            increment_agent_tasks_run(&state, &agent.id).await?;

            Ok(Json(AgentChatResponse {
                run_id,
                agent_id: agent.id,
                provider: resolved.provider_id,
                model,
                message,
                tool_calls: response.tool_calls,
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
                app_id: Some(source_app.clone()),
                agent_id: Some(agent.id.clone()),
                model_id: app_policy::model_config_for_agent(&catalog, &agent)
                    .map(|model| model.id.clone()),
                resolved_tool_ids: tool_ids_from_provider_tools(&tools),
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
    let draft_provider = match payload.draft_provider {
        Some(provider) => Some(normalize_litellm_provider_config(
            provider,
            &config.litellm_providers,
            &[],
        )?),
        None => None,
    };
    let has_draft_provider = draft_provider.is_some();
    if let Some(provider) = draft_provider.clone() {
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
    let configured_provider = if provider_id == "litellm" {
        None
    } else {
        Some(find_litellm_provider(&config, provider_id).ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                format!("unknown LiteLLM provider: {provider_id}"),
            )
        })?)
    };
    if let Some(provider) = &configured_provider {
        if normalize_provider_type(&provider.provider_type) == "ollama" {
            return test_ollama_provider_connection(
                &state,
                &config,
                provider_id,
                provider,
                requested_model,
            )
            .await;
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
        let configured_provider = configured_provider
            .as_ref()
            .expect("non-litellm provider was resolved before runtime setup");
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

async fn test_ollama_provider_connection(
    state: &AppState,
    config: &AppConfig,
    provider_id: &str,
    provider: &LiteLlmProviderConfig,
    requested_model: &str,
) -> ApiResult<Json<LiteLlmProviderTestResponse>> {
    let endpoint = provider
        .api_base
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(config.ollama_endpoint.as_str());
    let mut ollama_config = config.clone();
    ollama_config.ollama_endpoint = endpoint.to_string();
    let models = state
        .providers
        .list_ollama_models(&ollama_config)
        .await
        .map_err(|err| api_error(StatusCode::BAD_GATEWAY, err.to_string()))?;

    if models.is_empty() {
        return Err(api_error(
            StatusCode::BAD_GATEWAY,
            format!("Ollama is reachable at {endpoint}, but no models are installed."),
        ));
    }
    if !models.iter().any(|model| {
        model.name == requested_model || model.model.as_deref() == Some(requested_model)
    }) {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!(
                "Ollama is reachable at {endpoint}, but model '{requested_model}' was not found."
            ),
        ));
    }

    audit_hook(
        "provider.tested",
        serde_json::json!({ "provider": provider_id, "model": requested_model, "mode": "ollama_tags" }),
    );
    Ok(Json(LiteLlmProviderTestResponse {
        ok: true,
        content: format!(
            "Ollama reachable at {endpoint}. Model '{requested_model}' is available ({} models).",
            models.len()
        ),
        usage: None,
    }))
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
    let mut app_env = load_app_env().await?;
    ensure_litellm_master_key_for_runtime(&mut app_env);
    let config = hydrate_config_secrets(state.config.read().await.clone(), &app_env);
    let output_path = resolve_litellm_config_path(&state.config_path, &config, None);

    if state.litellm_runtime.readiness_healthy(&config).await {
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

    let start = state
        .litellm_runtime
        .ensure_started(&config, &output_path, &app_env)
        .await
        .map_err(runtime_start_api_error)?;

    audit_hook(
        "litellm.service.started",
        serde_json::json!({
            "path": output_path.display().to_string(),
            "base_url": config.litellm.base_url,
            "pid": start.as_ref().map(|start| start.pid)
        }),
    );

    Ok(Json(LiteLlmServiceStartResponse {
        status: if start.is_some() {
            "started".to_string()
        } else {
            "already_running".to_string()
        },
        base_url: config.litellm.base_url.clone(),
        config_path: output_path.display().to_string(),
        command: start
            .as_ref()
            .map(|start| start.command.clone())
            .unwrap_or_else(|| litellm_start_command_summary(&config, &output_path)),
        pid: start.map(|start| start.pid),
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> ApiResult<Json<serde_json::Value>> {
    let mut app_env = load_app_env().await?;
    ensure_litellm_master_key_for_runtime(&mut app_env);
    let config = hydrate_config_secrets(state.config.read().await.clone(), &app_env);
    if !config.litellm.enabled {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "LiteLLM is disabled in settings.",
        ));
    }

    let output_path = resolve_litellm_config_path(&state.config_path, &config, None);
    generate_litellm_config(
        &config.litellm_providers,
        &config.model_routes,
        &output_path,
        &config.ollama_endpoint,
    )
    .await
    .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    state
        .litellm_runtime
        .ensure_started(&config, &output_path, &app_env)
        .await
        .map_err(runtime_start_api_error)?;

    let response = state
        .litellm_runtime
        .forward_chat_completions(&config, body)
        .await
        .map_err(|err| api_error(StatusCode::BAD_GATEWAY, err.to_string()))?;

    Ok(Json(response))
}

async fn chat(
    State(state): State<AppState>,
    Json(payload): Json<ChatRequest>,
) -> ApiResult<Json<ChatResponse>> {
    chat_with_payload(state, payload).await
}

async fn create_run(
    State(state): State<AppState>,
    Json(payload): Json<RunCreateRequest>,
) -> ApiResult<Json<RunCreateResponse>> {
    let app_env = load_app_env().await?;
    let config = hydrate_config_secrets(state.config.read().await.clone(), &app_env);
    let catalog = state.catalog.read().await.clone();
    let fallback_model = first_ollama_model_name(&state, &config).await;
    let policy = match app_policy::resolve_app_policy(
        &catalog,
        &config,
        &payload.app_id,
        payload.agent_id.as_deref(),
        fallback_model.as_deref(),
    ) {
        Ok(policy) => policy,
        Err(err) => {
            record_audit(
                &state,
                audit_record(
                    "app.run_denied",
                    audit_level_for_policy_error(&err),
                    err.message.clone(),
                    Some(payload.app_id.clone()),
                    payload.agent_id.clone(),
                    None,
                    None,
                ),
            )
            .await;
            return Err(app_policy_api_error(err));
        }
    };

    if policy.provider_model.trim().is_empty() {
        let message = format!("agent '{}' has no resolved model", policy.agent.id);
        record_audit(
            &state,
            audit_record(
                "app.run_misconfigured",
                AuditLevel::Error,
                message.clone(),
                Some(policy.app.id.clone()),
                Some(policy.agent.id.clone()),
                None,
                Some(serde_json::json!({ "model_id": policy.model.id })),
            ),
        )
        .await;
        return Err(api_error(StatusCode::BAD_REQUEST, message));
    }

    let mut messages = normalize_messages(payload.messages, payload.input)?;
    messages = apply_agent_context(
        &config.instructions,
        &policy.agent,
        payload.instructions,
        payload.context,
        messages,
    );
    let tools = provider_tools_from_policy(&policy.tools);
    let tool_choice = tools.as_ref().map(|_| serde_json::json!("auto"));
    let metadata = Some(run_metadata(
        payload.metadata,
        &policy.app.id,
        &policy.agent.id,
        &policy.model.id,
    ));
    let generation = payload.generation.unwrap_or(policy.generation.clone());
    let (provider_id, model, provider) =
        provider_for_request(&state, &config, &policy.provider_id, &policy.provider_model)?;

    let provider_request = ProviderChatRequest {
        provider: provider_id.clone(),
        model: model.clone(),
        messages: messages.clone(),
        temperature: Some(generation.temperature),
        top_p: Some(generation.top_p),
        max_tokens: Some(generation.max_tokens),
        tools: tools.clone(),
        tool_choice,
        stream: false,
        metadata: metadata.clone(),
    };
    let run_id = Uuid::new_v4().to_string();
    let started_at = Utc::now();
    let started_timer = Instant::now();
    let prompt_summary = summarize_messages(&messages);
    let result = provider.chat_completion(provider_request).await;
    let ended_at = Utc::now();
    let duration_ms = duration_ms(started_timer.elapsed());

    match result {
        Ok(response) => {
            let pending = PendingRun {
                app_id: policy.app.id.clone(),
                agent_id: policy.agent.id.clone(),
                model_id: policy.model.id.clone(),
                provider_id: provider_id.clone(),
                provider_model: model.clone(),
                resolved_model_name: response.model.clone().unwrap_or(model.clone()),
                resolved_tool_ids: policy.tools.iter().map(|tool| tool.id.clone()).collect(),
                tools: policy.tools.clone(),
                messages,
                provider_tool_calls: serde_json::Value::Array(Vec::new()),
                tool_requests: Vec::new(),
                generation,
                metadata,
                started_at,
                prompt_summary,
                usage: response.usage.clone(),
            };

            finish_app_run_from_provider_response(
                &state,
                run_id,
                pending,
                response,
                duration_ms,
                Some(serde_json::json!({ "warnings": policy.warnings })),
            )
            .await
        }
        Err(err) => {
            let error = err.to_string();
            let record = RunRecord {
                id: run_id.clone(),
                app_id: Some(policy.app.id.clone()),
                agent_id: Some(policy.agent.id.clone()),
                model_id: Some(policy.model.id.clone()),
                resolved_tool_ids: policy.tools.iter().map(|tool| tool.id.clone()).collect(),
                provider: provider_id,
                model,
                source_app: Some(policy.app.id.clone()),
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
            record_audit(
                &state,
                audit_record(
                    "app.run_failed",
                    AuditLevel::Error,
                    error.clone(),
                    Some(policy.app.id),
                    Some(policy.agent.id),
                    Some(run_id),
                    Some(serde_json::json!({ "model_id": policy.model.id })),
                ),
            )
            .await;
            Err(api_error(StatusCode::BAD_GATEWAY, error))
        }
    }
}

async fn finish_app_run_from_provider_response(
    state: &AppState,
    run_id: String,
    mut pending: PendingRun,
    response: ProviderChatResponse,
    duration_ms: u64,
    audit_extra: Option<serde_json::Value>,
) -> ApiResult<Json<RunCreateResponse>> {
    let response_model = response
        .model
        .clone()
        .unwrap_or_else(|| pending.provider_model.clone());
    pending.resolved_model_name = response_model.clone();
    pending.usage = response.usage.clone().or(pending.usage.clone());

    if let Some(provider_tool_calls) = response.tool_calls.clone() {
        let (normalized_tool_calls, tool_requests) =
            normalize_provider_tool_calls(&provider_tool_calls, &pending.tools).map_err(|err| {
                api_error(
                    StatusCode::BAD_GATEWAY,
                    format!("invalid provider tool call: {err}"),
                )
            })?;

        if !tool_requests.is_empty() {
            pending.provider_tool_calls = normalized_tool_calls;
            pending.tool_requests = tool_requests.clone();

            {
                let mut pending_runs = state.pending_runs.write().await;
                pending_runs.insert(run_id.clone(), pending.clone());
            }

            let output = nonempty_string(response.content);
            let record = RunRecord {
                id: run_id.clone(),
                app_id: Some(pending.app_id.clone()),
                agent_id: Some(pending.agent_id.clone()),
                model_id: Some(pending.model_id.clone()),
                resolved_tool_ids: pending.resolved_tool_ids.clone(),
                provider: pending.provider_id.clone(),
                model: response_model,
                source_app: Some(pending.app_id.clone()),
                prompt_summary: pending.prompt_summary.clone(),
                response_summary: output
                    .as_deref()
                    .map(|content| truncate(content, 500))
                    .or_else(|| Some(format!("Requested {} tool(s)", tool_requests.len()))),
                status: RunStatus::RequiresAction,
                started_at: pending.started_at,
                ended_at: Utc::now(),
                duration_ms,
                error: None,
                usage: pending.usage.clone(),
            };
            record_run(state, record).await;
            record_audit(
                state,
                audit_record(
                    "tool.requested",
                    AuditLevel::Info,
                    format!(
                        "Run requested {} tool(s) from app '{}'",
                        tool_requests.len(),
                        pending.app_id
                    ),
                    Some(pending.app_id.clone()),
                    Some(pending.agent_id.clone()),
                    Some(run_id.clone()),
                    Some(serde_json::json!({
                        "tool_requests": tool_requests.clone(),
                        "tool_ids": pending.resolved_tool_ids.clone(),
                    })),
                ),
            )
            .await;

            return Ok(Json(RunCreateResponse {
                run_id,
                status: RunStatus::RequiresAction,
                app_id: pending.app_id,
                agent_id: pending.agent_id,
                model_id: pending.model_id,
                output,
                tool_requests,
                duration_ms,
            }));
        }
    }

    let output = response.content;
    let record = RunRecord {
        id: run_id.clone(),
        app_id: Some(pending.app_id.clone()),
        agent_id: Some(pending.agent_id.clone()),
        model_id: Some(pending.model_id.clone()),
        resolved_tool_ids: pending.resolved_tool_ids.clone(),
        provider: pending.provider_id.clone(),
        model: response_model,
        source_app: Some(pending.app_id.clone()),
        prompt_summary: pending.prompt_summary.clone(),
        response_summary: Some(truncate(&output, 500)),
        status: RunStatus::Completed,
        started_at: pending.started_at,
        ended_at: Utc::now(),
        duration_ms,
        error: None,
        usage: response.usage,
    };
    record_run(state, record).await;
    increment_agent_tasks_run(state, &pending.agent_id).await?;

    let mut metadata = serde_json::json!({
        "model_id": pending.model_id.clone(),
        "tool_ids": pending.resolved_tool_ids.clone(),
    });
    merge_json_object(&mut metadata, audit_extra);
    record_audit(
        state,
        audit_record(
            "app.run_completed",
            AuditLevel::Info,
            format!("Completed run for app '{}'", pending.app_id),
            Some(pending.app_id.clone()),
            Some(pending.agent_id.clone()),
            Some(run_id.clone()),
            Some(metadata),
        ),
    )
    .await;

    Ok(Json(RunCreateResponse {
        run_id,
        status: RunStatus::Completed,
        app_id: pending.app_id,
        agent_id: pending.agent_id,
        model_id: pending.model_id,
        output: Some(output),
        tool_requests: Vec::new(),
        duration_ms,
    }))
}

async fn submit_run_tool_results(
    State(state): State<AppState>,
    AxumPath(run_id): AxumPath<String>,
    Json(payload): Json<RunToolResultsRequest>,
) -> ApiResult<Json<RunCreateResponse>> {
    let pending = {
        let pending_runs = state.pending_runs.read().await;
        pending_runs.get(&run_id).cloned()
    }
    .ok_or_else(|| api_error(StatusCode::NOT_FOUND, "run is not waiting for tool results"))?;

    if payload.app_id != pending.app_id {
        record_audit(
            &state,
            audit_record(
                "tool.result_denied",
                AuditLevel::Denied,
                format!(
                    "Rejected tool results for app '{}' on run owned by '{}'",
                    payload.app_id, pending.app_id
                ),
                Some(payload.app_id),
                Some(pending.agent_id.clone()),
                Some(run_id),
                None,
            ),
        )
        .await;
        return Err(api_error(
            StatusCode::FORBIDDEN,
            "tool results were submitted by the wrong app",
        ));
    }

    let tool_messages = tool_result_messages_for_pending(&pending, payload.tool_results)?;

    {
        let mut pending_runs = state.pending_runs.write().await;
        pending_runs.remove(&run_id);
    }

    record_audit(
        &state,
        audit_record(
            "tool.result_received",
            AuditLevel::Info,
            format!(
                "Received {} tool result(s) from app '{}'",
                tool_messages.len(),
                pending.app_id
            ),
            Some(pending.app_id.clone()),
            Some(pending.agent_id.clone()),
            Some(run_id.clone()),
            Some(serde_json::json!({
                "tool_call_ids": pending.tool_requests.iter().map(|request| request.id.clone()).collect::<Vec<_>>(),
                "tool_ids": pending.tool_requests.iter().map(|request| request.tool_id.clone()).collect::<Vec<_>>(),
            })),
        ),
    )
    .await;

    let app_env = load_app_env().await?;
    let config = hydrate_config_secrets(state.config.read().await.clone(), &app_env);
    let (provider_id, model, provider) = provider_for_request(
        &state,
        &config,
        &pending.provider_id,
        &pending.provider_model,
    )?;
    let tools = provider_tools_from_policy(&pending.tools);
    let tool_choice = tools.as_ref().map(|_| serde_json::json!("auto"));
    let mut messages = pending.messages.clone();
    messages.push(ChatMessage {
        role: "assistant".to_string(),
        content: MessageContent::Text(String::new()),
        name: None,
        tool_call_id: None,
        tool_calls: Some(pending.provider_tool_calls.clone()),
    });
    messages.extend(tool_messages);

    let provider_request = ProviderChatRequest {
        provider: provider_id.clone(),
        model: model.clone(),
        messages: messages.clone(),
        temperature: Some(pending.generation.temperature),
        top_p: Some(pending.generation.top_p),
        max_tokens: Some(pending.generation.max_tokens),
        tools,
        tool_choice,
        stream: false,
        metadata: pending.metadata.clone(),
    };

    let result = provider.chat_completion(provider_request).await;
    let ended_at = Utc::now();
    let duration_ms = duration_ms_between(pending.started_at, ended_at);

    match result {
        Ok(response) => {
            let next_pending = PendingRun {
                app_id: pending.app_id,
                agent_id: pending.agent_id,
                model_id: pending.model_id,
                provider_id,
                provider_model: model,
                resolved_model_name: response
                    .model
                    .clone()
                    .unwrap_or(pending.resolved_model_name),
                resolved_tool_ids: pending.resolved_tool_ids,
                tools: pending.tools,
                messages,
                provider_tool_calls: serde_json::Value::Array(Vec::new()),
                tool_requests: Vec::new(),
                generation: pending.generation,
                metadata: pending.metadata,
                started_at: pending.started_at,
                prompt_summary: pending.prompt_summary,
                usage: response.usage.clone().or(pending.usage),
            };

            finish_app_run_from_provider_response(
                &state,
                run_id,
                next_pending,
                response,
                duration_ms,
                Some(serde_json::json!({ "continued_from_tool_results": true })),
            )
            .await
        }
        Err(err) => {
            let error = err.to_string();
            let record = RunRecord {
                id: run_id.clone(),
                app_id: Some(pending.app_id.clone()),
                agent_id: Some(pending.agent_id.clone()),
                model_id: Some(pending.model_id.clone()),
                resolved_tool_ids: pending.resolved_tool_ids,
                provider: provider_id,
                model,
                source_app: Some(pending.app_id.clone()),
                prompt_summary: pending.prompt_summary,
                response_summary: None,
                status: RunStatus::Failed,
                started_at: pending.started_at,
                ended_at,
                duration_ms,
                error: Some(error.clone()),
                usage: pending.usage,
            };
            record_run(&state, record).await;
            record_audit(
                &state,
                audit_record(
                    "app.run_failed",
                    AuditLevel::Error,
                    error.clone(),
                    Some(pending.app_id),
                    Some(pending.agent_id),
                    Some(run_id),
                    Some(serde_json::json!({ "model_id": pending.model_id })),
                ),
            )
            .await;
            Err(api_error(StatusCode::BAD_GATEWAY, error))
        }
    }
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
                app_id: None,
                agent_id: None,
                model_id: None,
                resolved_tool_ids: Vec::new(),
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
            app_id: None,
            agent_id: None,
            model_id: None,
            resolved_tool_ids: Vec::new(),
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

async fn stream_run(
    State(state): State<AppState>,
    Json(payload): Json<RunCreateRequest>,
) -> ApiResult<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    let app_env = load_app_env().await?;
    let config = hydrate_config_secrets(state.config.read().await.clone(), &app_env);
    let catalog = state.catalog.read().await.clone();
    let fallback_model = first_ollama_model_name(&state, &config).await;
    let policy = app_policy::resolve_app_policy(
        &catalog,
        &config,
        &payload.app_id,
        payload.agent_id.as_deref(),
        fallback_model.as_deref(),
    )
    .map_err(app_policy_api_error)?;

    if policy.provider_model.trim().is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!("agent '{}' has no resolved model", policy.agent.id),
        ));
    }

    let mut messages = normalize_messages(payload.messages, payload.input)?;
    messages = apply_agent_context(
        &config.instructions,
        &policy.agent,
        payload.instructions,
        payload.context,
        messages,
    );
    let tools = provider_tools_from_policy(&policy.tools);
    let generation = payload.generation.unwrap_or(policy.generation.clone());
    let metadata = Some(run_metadata(
        payload.metadata,
        &policy.app.id,
        &policy.agent.id,
        &policy.model.id,
    ));
    let (provider_id, model, provider) =
        provider_for_request(&state, &config, &policy.provider_id, &policy.provider_model)?;

    let provider_request = ProviderChatRequest {
        provider: provider_id.clone(),
        model: model.clone(),
        messages: messages.clone(),
        temperature: Some(generation.temperature),
        top_p: Some(generation.top_p),
        max_tokens: Some(generation.max_tokens),
        tools,
        tool_choice: None,
        stream: true,
        metadata,
    };
    let run_id = Uuid::new_v4().to_string();
    let started_at = Utc::now();
    let started_timer = Instant::now();
    let prompt_summary = summarize_messages(&messages);

    let provider_stream = match provider.stream_chat_completion(provider_request).await {
        Ok(provider_stream) => provider_stream,
        Err(err) => {
            let error = err.to_string();
            let ended_at = Utc::now();
            let record = RunRecord {
                id: run_id,
                app_id: Some(policy.app.id),
                agent_id: Some(policy.agent.id),
                model_id: Some(policy.model.id),
                resolved_tool_ids: policy.tools.iter().map(|tool| tool.id.clone()).collect(),
                provider: provider_id,
                model,
                source_app: Some(payload.app_id),
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
        let app_id = policy.app.id.clone();
        let agent_id = policy.agent.id.clone();
        let model_id = policy.model.id.clone();
        let tool_ids = policy.tools.iter().map(|tool| tool.id.clone()).collect();
        let record = RunRecord {
            id: run_id,
            app_id: Some(app_id),
            agent_id: Some(agent_id),
            model_id: Some(model_id),
            resolved_tool_ids: tool_ids,
            provider: provider_id,
            model,
            source_app: Some(policy.app.id),
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

async fn list_audit(
    State(state): State<AppState>,
    Query(query): Query<AuditQuery>,
) -> Json<AuditResponse> {
    let limit = query.limit.unwrap_or(50).min(MAX_RECENT_AUDIT);
    let audit = state
        .audit
        .read()
        .await
        .iter()
        .take(limit)
        .cloned()
        .collect();
    Json(AuditResponse { audit })
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
                app_id: None,
                agent_id: None,
                model_id: None,
                resolved_tool_ids: Vec::new(),
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
                app_id: None,
                agent_id: None,
                model_id: None,
                resolved_tool_ids: Vec::new(),
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

fn apply_agent_context(
    settings: &InstructionSettings,
    agent: &AgentConfig,
    request_instructions: Option<String>,
    app_context: Option<serde_json::Value>,
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

    push_trimmed(&mut parts, &agent.system_prompt);

    if let Some(instructions) = request_instructions {
        let instructions = instructions.trim();
        if !instructions.is_empty() {
            parts.push(format!("Request-specific instructions:\n{instructions}"));
        }
    }

    if let Some(context) = app_context {
        parts.push(format!(
            "App context:\n{}",
            serde_json::to_string_pretty(&context).unwrap_or_else(|_| context.to_string())
        ));
    }

    if !parts.is_empty() {
        messages.insert(0, ChatMessage::text("system", parts.join("\n\n")));
    }

    messages
}

fn agent_from_create(
    payload: AgentCreateRequest,
    existing_ids: &[String],
) -> ApiResult<AgentConfig> {
    let mut agent = AgentConfig::default();
    agent.id = payload
        .id
        .map(|id| normalize_agent_id(&id))
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| unique_agent_id(existing_ids));
    if let Some(name) = payload.name {
        agent.name = trim_or_default(name, "New agent");
    }
    if let Some(role) = payload.role {
        agent.role = trim_or_default(role, "General assistant");
    }
    if let Some(description) = payload.description {
        agent.description = description.trim().to_string();
    }
    if let Some(system_prompt) = payload.system_prompt {
        agent.system_prompt = system_prompt.trim().to_string();
    }
    if let Some(default_model_id) = payload.default_model_id {
        agent.default_model_id = nonempty_trimmed(default_model_id);
    }
    if let Some(provider_id) = payload.default_provider_id {
        agent.default_provider_id = normalize_provider_id(&provider_id);
    }
    if let Some(model) = payload.default_model {
        agent.default_model = model.trim().to_string();
    }
    if let Some(environment) = payload.default_environment {
        agent.default_environment = normalize_agent_enum_value(&environment);
    }
    if let Some(autonomy) = payload.autonomy {
        agent.autonomy = normalize_agent_enum_value(&autonomy);
    }
    if let Some(permissions) = payload.permissions {
        agent.permissions = permissions;
    }
    if let Some(allowed_tool_ids) = payload.allowed_tool_ids {
        agent.allowed_tool_ids = normalize_id_list(allowed_tool_ids);
    }
    if let Some(temperature) = payload.temperature {
        agent.temperature = Some(temperature);
    }
    if let Some(max_tokens) = payload.max_tokens {
        agent.max_tokens = Some(max_tokens);
    }
    if let Some(enabled) = payload.enabled {
        agent.enabled = enabled;
    }
    if let Some(status) = payload.status {
        agent.status = normalize_agent_status(&status)?;
    }
    agent.updated_at = timestamp_now();
    Ok(agent)
}

fn apply_agent_patch(agent: &mut AgentConfig, patch: AgentPatch) -> ApiResult<()> {
    if let Some(name) = patch.name {
        agent.name = name;
    }
    if let Some(description) = patch.description {
        agent.description = description;
    }
    if let Some(system_prompt) = patch.system_prompt {
        agent.system_prompt = system_prompt;
    }
    if let Some(default_model_id) = patch.default_model_id {
        agent.default_model_id = nonempty_trimmed(default_model_id);
    }
    if let Some(provider_id) = patch.default_provider_id {
        agent.default_provider_id = normalize_provider_id(&provider_id);
    }
    if let Some(model) = patch.default_model {
        agent.default_model = model.trim().to_string();
    }
    if let Some(allowed_tool_ids) = patch.allowed_tool_ids {
        agent.allowed_tool_ids = normalize_id_list(allowed_tool_ids);
    }
    if let Some(temperature) = patch.temperature {
        agent.temperature = Some(temperature);
    }
    if let Some(max_tokens) = patch.max_tokens {
        agent.max_tokens = Some(max_tokens);
    }
    if let Some(enabled) = patch.enabled {
        agent.enabled = enabled;
    }
    if let Some(status) = patch.status {
        agent.status = normalize_agent_status(&status)?;
    }
    Ok(())
}

fn reject_protected_agent_patch_fields(value: &serde_json::Value) -> ApiResult<()> {
    let object = value
        .as_object()
        .ok_or_else(|| api_error(StatusCode::BAD_REQUEST, "agent patch must be a JSON object"))?;
    let allowed = [
        "name",
        "description",
        "system_prompt",
        "default_model_id",
        "default_provider_id",
        "default_model",
        "allowed_tool_ids",
        "temperature",
        "max_tokens",
        "enabled",
        "status",
    ];

    for key in object.keys() {
        if !allowed.iter().any(|allowed| allowed == key) {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!("agent field '{key}' cannot be patched"),
            ));
        }
    }

    Ok(())
}

async fn validate_agent_defaults(
    state: &AppState,
    config: &AppConfig,
    agent: &AgentConfig,
) -> ApiResult<()> {
    if agent.id.trim().is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "agent id is required"));
    }
    if !["draft", "paused", "active"].contains(&agent.status.as_str()) {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!("invalid agent status '{}'", agent.status),
        ));
    }

    let provider_id = agent.default_provider_id.trim();
    if provider_id.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "default_provider_id is required",
        ));
    }

    if agent_is_active(agent)
        && agent.default_model.trim().is_empty()
        && agent
            .default_model_id
            .as_deref()
            .is_none_or(|model_id| model_id.trim().is_empty())
    {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "default_model or default_model_id is required for active agents",
        ));
    }

    validate_agent_provider(config, provider_id)?;
    validate_agent_model(state, config, provider_id, &agent.default_model).await
}

fn validate_agent_provider(config: &AppConfig, provider_id: &str) -> ApiResult<()> {
    if provider_id == "ollama" {
        return Ok(());
    }

    let provider = find_litellm_provider(config, provider_id).ok_or_else(|| {
        api_error(
            StatusCode::BAD_REQUEST,
            format!("unknown model provider: {provider_id}"),
        )
    })?;
    if !provider.enabled {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!("provider '{provider_id}' is disabled"),
        ));
    }

    Ok(())
}

async fn validate_agent_model(
    state: &AppState,
    config: &AppConfig,
    provider_id: &str,
    model: &str,
) -> ApiResult<()> {
    let model = model.trim();
    if model.is_empty() {
        return Ok(());
    }

    if provider_id == "ollama" {
        if let Ok(models) = state.providers.list_ollama_models(config).await {
            if !models.is_empty() && !models.iter().any(|item| item.name == model) {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    format!("model '{model}' is not available for provider 'ollama'"),
                ));
            }
        }
        return Ok(());
    }

    let provider = find_litellm_provider(config, provider_id).ok_or_else(|| {
        api_error(
            StatusCode::BAD_REQUEST,
            format!("unknown model provider: {provider_id}"),
        )
    })?;
    let models = suggested_provider_models(&provider);
    if !models.is_empty() {
        let known_model = models.iter().any(|item| item == model)
            || models
                .iter()
                .map(|item| litellm_model_for_provider(&provider, item))
                .any(|item| item == model);
        if !known_model {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!("model '{model}' is not available for provider '{provider_id}'"),
            ));
        }
    }

    Ok(())
}

fn agent_is_active(agent: &AgentConfig) -> bool {
    agent.status == "active"
}

fn normalize_agent_status(status: &str) -> ApiResult<String> {
    let status = normalize_agent_enum_value(status);
    if ["draft", "paused", "active"].contains(&status.as_str()) {
        Ok(status)
    } else {
        Err(api_error(
            StatusCode::BAD_REQUEST,
            format!("invalid agent status '{status}'"),
        ))
    }
}

fn normalize_agent_id(value: &str) -> String {
    let id = normalize_provider_id(value);
    if id.starts_with("ag_") {
        id
    } else if id.is_empty() {
        String::new()
    } else {
        format!("ag_{id}")
    }
}

fn unique_agent_id(existing_ids: &[String]) -> String {
    loop {
        let id = format!("ag_{}", Uuid::new_v4().simple());
        if !existing_ids.iter().any(|existing| existing == &id) {
            return id;
        }
    }
}

fn normalize_agent_enum_value(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace([' ', '_'], "-")
}

fn trim_or_default(value: String, fallback: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        fallback.to_string()
    } else {
        value.to_string()
    }
}

fn timestamp_now() -> String {
    Utc::now().to_rfc3339()
}

async fn increment_agent_tasks_run(state: &AppState, agent_id: &str) -> ApiResult<()> {
    let mut catalog = state.catalog.write().await;
    if let Some(agent) = app_policy::find_agent_mut(&mut catalog, agent_id) {
        agent.tasks_run = agent.tasks_run.saturating_add(1);
        agent.updated_at = timestamp_now();
        app_policy::save_agents(&state.catalog_dir, &catalog.agents)
            .await
            .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    }
    Ok(())
}

fn agent_chat_metadata(
    metadata: Option<serde_json::Value>,
    source_app: &str,
    agent_id: &str,
) -> serde_json::Value {
    let mut metadata = match metadata {
        Some(serde_json::Value::Object(map)) => map,
        Some(value) => {
            let mut map = serde_json::Map::new();
            map.insert("request_metadata".to_string(), value);
            map
        }
        None => serde_json::Map::new(),
    };
    metadata.insert("source_app".to_string(), serde_json::json!(source_app));
    metadata.insert("agent_id".to_string(), serde_json::json!(agent_id));
    serde_json::Value::Object(metadata)
}

fn push_trimmed(parts: &mut Vec<String>, value: &str) {
    let value = value.trim();
    if !value.is_empty() {
        parts.push(value.to_string());
    }
}

async fn first_ollama_model_name(state: &AppState, config: &AppConfig) -> Option<String> {
    state
        .providers
        .list_ollama_models(config)
        .await
        .ok()
        .and_then(|models| models.into_iter().next())
        .map(|model| model.name)
}

fn provider_for_request(
    state: &AppState,
    config: &AppConfig,
    requested_provider_id: &str,
    requested_model: &str,
) -> ApiResult<(String, String, Arc<dyn ModelProvider>)> {
    let requested_provider_id = requested_provider_id.trim().to_ascii_lowercase();
    let requested_model = requested_model.trim();
    if requested_model.is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "model is required"));
    }

    let (provider_registry_id, model) = if requested_provider_id == "ollama"
        || requested_provider_id == "litellm"
    {
        (requested_provider_id.clone(), requested_model.to_string())
    } else if let Some(litellm_provider) = find_litellm_provider(config, &requested_provider_id) {
        (
            "litellm".to_string(),
            litellm_model_for_provider(&litellm_provider, requested_model),
        )
    } else {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!("unknown model provider: {requested_provider_id}"),
        ));
    };
    let provider = state
        .providers
        .get(&provider_registry_id, config)
        .ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                format!("unknown model provider: {requested_provider_id}"),
            )
        })?;

    Ok((requested_provider_id, model, provider))
}

fn provider_tools_from_policy(tools: &[ToolConfig]) -> Option<serde_json::Value> {
    let provider_tools = tools
        .iter()
        .filter(|tool| tool.enabled)
        .map(|tool| {
            let name = provider_tool_name(&tool.id);
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.input_schema.clone().unwrap_or_else(|| serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": true
                    }))
                }
            })
        })
        .collect::<Vec<_>>();

    if provider_tools.is_empty() {
        None
    } else {
        Some(serde_json::Value::Array(provider_tools))
    }
}

fn provider_tool_name(tool_id: &str) -> String {
    tool_id.replace('.', "_")
}

fn normalize_provider_tool_calls(
    provider_tool_calls: &serde_json::Value,
    tools: &[ToolConfig],
) -> Result<(serde_json::Value, Vec<RunToolRequest>), String> {
    let calls = provider_tool_calls
        .as_array()
        .ok_or_else(|| "tool_calls must be an array".to_string())?;
    let tools_by_provider_name = tools
        .iter()
        .filter(|tool| tool.enabled)
        .map(|tool| (provider_tool_name(&tool.id), tool))
        .collect::<HashMap<_, _>>();
    let mut normalized_calls = Vec::new();
    let mut requests = Vec::new();

    for (index, call) in calls.iter().enumerate() {
        let function = call
            .get("function")
            .ok_or_else(|| format!("tool call {index} is missing function"))?;
        let provider_name = function
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("tool call {index} is missing function.name"))?;
        let tool = tools_by_provider_name
            .get(provider_name)
            .ok_or_else(|| format!("unknown tool function '{provider_name}'"))?;
        let mut normalized_call = call.clone();
        let call_id = call
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple()));

        if let serde_json::Value::Object(map) = &mut normalized_call {
            map.insert("id".to_string(), serde_json::Value::String(call_id.clone()));
            map.entry("type".to_string())
                .or_insert_with(|| serde_json::Value::String("function".to_string()));
        }

        let arguments = parse_tool_call_arguments(function.get("arguments"))
            .map_err(|err| format!("tool call '{call_id}' has invalid arguments: {err}"))?;
        normalized_calls.push(normalized_call);
        requests.push(RunToolRequest {
            id: call_id,
            tool_id: tool.id.clone(),
            name: provider_name.to_string(),
            arguments,
            risk_level: tool.risk_level.clone(),
            display_name: tool.name.clone(),
        });
    }

    Ok((serde_json::Value::Array(normalized_calls), requests))
}

fn parse_tool_call_arguments(
    arguments: Option<&serde_json::Value>,
) -> Result<serde_json::Value, String> {
    match arguments {
        Some(serde_json::Value::String(value)) if value.trim().is_empty() => {
            Ok(serde_json::json!({}))
        }
        Some(serde_json::Value::String(value)) => serde_json::from_str(value)
            .map_err(|err| format!("expected JSON object arguments encoded as a string: {err}")),
        Some(serde_json::Value::Object(_)) => Ok(arguments.cloned().unwrap_or_default()),
        Some(serde_json::Value::Null) | None => Ok(serde_json::json!({})),
        Some(_) => Err("expected object arguments".to_string()),
    }
}

fn tool_result_messages_for_pending(
    pending: &PendingRun,
    results: Vec<RunToolResultInput>,
) -> ApiResult<Vec<ChatMessage>> {
    if results.len() != pending.tool_requests.len() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!(
                "expected {} tool result(s), received {}",
                pending.tool_requests.len(),
                results.len()
            ),
        ));
    }

    let mut results_by_call_id = HashMap::new();
    for result in results {
        let call_id = result.tool_call_id.trim().to_string();
        if call_id.is_empty() {
            return Err(api_error(StatusCode::BAD_REQUEST, "toolCallId is required"));
        }
        if results_by_call_id.insert(call_id, result).is_some() {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "duplicate tool result call id",
            ));
        }
    }

    let mut messages = Vec::new();
    for request in &pending.tool_requests {
        let result = results_by_call_id.remove(&request.id).ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                format!("missing tool result for call '{}'", request.id),
            )
        })?;
        if let Some(tool_id) = result.tool_id.as_deref() {
            if tool_id != request.tool_id {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    format!("tool result for '{}' used the wrong tool id", request.id),
                ));
            }
        }

        let content = match nonempty_trimmed(result.error.unwrap_or_default()) {
            Some(error) => serde_json::json!({ "error": error }),
            None => result.result.unwrap_or_else(|| serde_json::json!({})),
        };
        messages.push(ChatMessage {
            role: "tool".to_string(),
            content: MessageContent::Structured(content),
            name: Some(provider_tool_name(&request.tool_id)),
            tool_call_id: Some(request.id.clone()),
            tool_calls: None,
        });
    }

    if !results_by_call_id.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "received tool results that were not requested",
        ));
    }

    Ok(messages)
}

fn tool_ids_from_provider_tools(tools: &Option<serde_json::Value>) -> Vec<String> {
    tools
        .as_ref()
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    item.get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn run_metadata(
    metadata: Option<serde_json::Value>,
    app_id: &str,
    agent_id: &str,
    model_id: &str,
) -> serde_json::Value {
    let mut metadata = match metadata {
        Some(serde_json::Value::Object(map)) => map,
        Some(value) => {
            let mut map = serde_json::Map::new();
            map.insert("request_metadata".to_string(), value);
            map
        }
        None => serde_json::Map::new(),
    };
    metadata.insert("app_id".to_string(), serde_json::json!(app_id));
    metadata.insert("agent_id".to_string(), serde_json::json!(agent_id));
    metadata.insert("model_id".to_string(), serde_json::json!(model_id));
    serde_json::Value::Object(metadata)
}

fn app_policy_api_error(error: AppPolicyError) -> (StatusCode, Json<ApiError>) {
    let status = match error.kind {
        AppPolicyErrorKind::NotFound => StatusCode::NOT_FOUND,
        AppPolicyErrorKind::Disabled => StatusCode::FORBIDDEN,
        AppPolicyErrorKind::Forbidden => StatusCode::FORBIDDEN,
        AppPolicyErrorKind::Misconfigured => StatusCode::BAD_REQUEST,
    };
    api_error(status, error.message)
}

fn audit_level_for_policy_error(error: &AppPolicyError) -> AuditLevel {
    match error.kind {
        AppPolicyErrorKind::Forbidden | AppPolicyErrorKind::Disabled => AuditLevel::Denied,
        AppPolicyErrorKind::NotFound => AuditLevel::Warn,
        AppPolicyErrorKind::Misconfigured => AuditLevel::Error,
    }
}

fn audit_record(
    event: impl Into<String>,
    level: AuditLevel,
    message: impl Into<String>,
    app_id: Option<String>,
    agent_id: Option<String>,
    run_id: Option<String>,
    metadata: Option<serde_json::Value>,
) -> AuditRecord {
    AuditRecord {
        id: format!("audit_{}", Uuid::new_v4().simple()),
        event: event.into(),
        level,
        message: message.into(),
        app_id,
        agent_id,
        run_id,
        metadata,
        created_at: Utc::now(),
    }
}

async fn record_audit(state: &AppState, record: AuditRecord) {
    let logging_enabled = state.config.read().await.logging_enabled;
    {
        let mut history = state.audit.write().await;
        runs::push_recent_audit(&mut history, record.clone(), MAX_RECENT_AUDIT);
    }

    if logging_enabled {
        if let Err(err) = runs::append_audit_jsonl(&state.audit_path, &record).await {
            tracing::warn!(error = %err, "failed to append audit log");
        }
    }
}

fn normalize_id_list(ids: Vec<String>) -> Vec<String> {
    ids.into_iter()
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty())
        .fold(Vec::new(), |mut acc, id| {
            if !acc
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(&id))
            {
                acc.push(id);
            }
            acc
        })
}

fn nonempty_trimmed(value: String) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn nonempty_string(value: String) -> Option<String> {
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

fn merge_json_object(target: &mut serde_json::Value, source: Option<serde_json::Value>) {
    let Some(serde_json::Value::Object(source)) = source else {
        return;
    };
    let Some(target) = target.as_object_mut() else {
        return;
    };

    for (key, value) in source {
        target.insert(key, value);
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
    _config_path: &Path,
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

    secrets::app_data_dir().join(path)
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

fn duration_ms_between(started_at: DateTime<Utc>, ended_at: DateTime<Utc>) -> u64 {
    ended_at
        .signed_duration_since(started_at)
        .num_milliseconds()
        .max(0) as u64
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
    state
        .litellm_runtime
        .ensure_after_apply(config, output_path, app_env)
        .await
        .map_err(runtime_start_api_error)
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
    if let Some(existing_type) = existing_providers
        .iter()
        .find(|existing| normalize_provider_id(&existing.id) == provider.id)
        .map(|existing| normalize_provider_type(&existing.provider_type))
        .filter(|provider_type| !provider_type.is_empty())
    {
        provider.provider_type = existing_type;
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

fn runtime_start_api_error(error: anyhow::Error) -> (StatusCode, Json<ApiError>) {
    let message = error.to_string();
    let status = if message.contains("already using the LiteLLM port") {
        StatusCode::CONFLICT
    } else if message.contains("LiteLLM Python was not found")
        || message.contains("runtime Python does not exist")
    {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };

    api_error(status, message)
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
    fn existing_provider_type_is_preserved_by_id() {
        let existing = vec![LiteLlmProviderConfig {
            id: "openai_main".to_string(),
            enabled: true,
            provider_type: "openai".to_string(),
            display_name: "OpenAI".to_string(),
            api_key_env_var: "OPENAI_API_KEY".to_string(),
            api_key: Some(REDACTED_SECRET.to_string()),
            api_base: None,
        }];
        let providers = vec![LiteLlmProviderConfig {
            provider_type: "ollama".to_string(),
            ..existing[0].clone()
        }];

        let normalized = match normalize_litellm_provider_configs(providers, &existing) {
            Ok(providers) => providers,
            Err(_) => panic!("existing provider type should be preserved"),
        };

        assert_eq!(normalized[0].provider_type, "openai");
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

        let child_env = crate::litellm_runtime::litellm_child_env(&config, &app_env);

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

    #[test]
    fn protected_agent_patch_fields_are_rejected() {
        let patch = serde_json::json!({
            "name": "Notebook helper",
            "tasks_run": 12
        });

        assert!(reject_protected_agent_patch_fields(&patch).is_err());
    }

    #[test]
    fn app_editable_agent_patch_fields_are_allowed() {
        let patch = serde_json::json!({
            "name": "Notebook helper",
            "description": "Helps with current notes",
            "system_prompt": "Stay concise.",
            "default_model_id": "ollama-default",
            "default_provider_id": "ollama",
            "default_model": "qwen2.5:7b",
            "allowed_tool_ids": ["note.getCurrentPage"],
            "temperature": 0.2,
            "max_tokens": 512,
            "enabled": true,
            "status": "active"
        });

        assert!(reject_protected_agent_patch_fields(&patch).is_ok());
    }

    #[test]
    fn agent_patch_preserves_editing_spaces() {
        let mut agent = AgentConfig::default();
        let result = apply_agent_patch(
            &mut agent,
            AgentPatch {
                name: Some("Notebook helper ".to_string()),
                description: Some("Draft with normal spaces ".to_string()),
                system_prompt: Some("Keep user spacing intact. ".to_string()),
                default_model_id: None,
                default_provider_id: None,
                default_model: None,
                allowed_tool_ids: None,
                temperature: None,
                max_tokens: None,
                enabled: None,
                status: None,
            },
        );

        assert!(result.is_ok());
        assert_eq!(agent.name, "Notebook helper ");
        assert_eq!(agent.description, "Draft with normal spaces ");
        assert_eq!(agent.system_prompt, "Keep user spacing intact. ");
    }

    #[test]
    fn agent_context_prepends_agent_prompt_and_app_context() {
        let mut agent = AgentConfig::default();
        agent.system_prompt = "Use the current page context.".to_string();
        let messages = apply_agent_context(
            &InstructionSettings::default(),
            &agent,
            Some("Return a tool call when editing.".to_string()),
            Some(serde_json::json!({ "activePage": { "title": "Daily" } })),
            vec![ChatMessage::text("user", "Append the summary.")],
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert!(messages[0]
            .content_text()
            .contains("Use the current page context."));
        assert!(messages[0].content_text().contains("Return a tool call"));
        assert!(messages[0].content_text().contains("activePage"));
    }

    #[test]
    fn configured_tools_use_openai_function_shape_when_enabled() {
        let tools = provider_tools_from_policy(&[ToolConfig {
            id: "note.getCurrentPage".to_string(),
            name: "Get Current Page".to_string(),
            description: "Read note content.".to_string(),
            risk_level: "low".to_string(),
            enabled: true,
            input_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            })),
            output_schema: None,
        }])
        .expect("enabled tool should produce provider tools");
        let tools = tools.as_array().expect("tools should be an array");

        assert_eq!(tools.len(), 1);
        assert_eq!(
            tools[0]
                .pointer("/function/name")
                .and_then(serde_json::Value::as_str),
            Some("note_getCurrentPage")
        );
        assert_eq!(
            tools[0]
                .pointer("/function/parameters/properties/query/type")
                .and_then(serde_json::Value::as_str),
            Some("string")
        );
    }

    #[test]
    fn provider_tool_calls_map_back_to_note_tool_ids() {
        let tools = vec![ToolConfig {
            id: "note.getCurrentPage".to_string(),
            name: "Get Current Page".to_string(),
            description: "Read the current page.".to_string(),
            risk_level: "low".to_string(),
            enabled: true,
            input_schema: None,
            output_schema: None,
        }];
        let raw_tool_calls = serde_json::json!([
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "note_getCurrentPage",
                    "arguments": "{\"includeBlocks\":true}"
                }
            }
        ]);

        let (normalized, requests) =
            normalize_provider_tool_calls(&raw_tool_calls, &tools).expect("tool call should map");

        assert_eq!(normalized[0]["id"], "call_1");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].tool_id, "note.getCurrentPage");
        assert_eq!(requests[0].arguments["includeBlocks"], true);
    }

    #[test]
    fn tool_result_validation_rejects_wrong_tool_id() {
        let pending = PendingRun {
            app_id: "note".to_string(),
            agent_id: "note-assistant".to_string(),
            model_id: "ollama-default".to_string(),
            provider_id: "ollama".to_string(),
            provider_model: "llama3".to_string(),
            resolved_model_name: "llama3".to_string(),
            resolved_tool_ids: vec!["note.getCurrentPage".to_string()],
            tools: Vec::new(),
            messages: Vec::new(),
            provider_tool_calls: serde_json::json!([]),
            tool_requests: vec![RunToolRequest {
                id: "call_1".to_string(),
                tool_id: "note.getCurrentPage".to_string(),
                name: "note_getCurrentPage".to_string(),
                arguments: serde_json::json!({}),
                risk_level: "low".to_string(),
                display_name: "Get Current Page".to_string(),
            }],
            generation: GenerationSettings::default(),
            metadata: None,
            started_at: Utc::now(),
            prompt_summary: "user: summarize".to_string(),
            usage: None,
        };

        let error = tool_result_messages_for_pending(
            &pending,
            vec![RunToolResultInput {
                tool_call_id: "call_1".to_string(),
                tool_id: Some("note.deleteBlock".to_string()),
                result: Some(serde_json::json!({ "ok": true })),
                error: None,
            }],
        )
        .expect_err("wrong tool id should be rejected");

        assert_eq!(error.0, StatusCode::BAD_REQUEST);
    }
}
