use crate::{
    config::{self, AppConfig, GenerationSettings, InstructionSettings},
    ollama::{ChatMessage, OllamaClient, OllamaModel},
    runs::{self, RunRecord, RunStatus},
};
use axum::{
    extract::{Query, State},
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
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use uuid::Uuid;

const MAX_RECENT_RUNS: usize = 100;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<RwLock<AppConfig>>,
    pub config_path: PathBuf,
    pub runs_path: PathBuf,
    pub runs: Arc<RwLock<VecDeque<RunRecord>>>,
    pub ollama: OllamaClient,
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
    model: Option<String>,
    source_app: Option<String>,
    messages: Option<Vec<ChatMessage>>,
    prompt: Option<String>,
    instructions: Option<String>,
    generation: Option<GenerationSettings>,
}

#[derive(Serialize)]
struct ChatResponse {
    run_id: String,
    model: String,
    message: ChatMessage,
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
struct SettingsPatch {
    ollama_endpoint: Option<String>,
    default_model: Option<String>,
    generation: Option<GenerationSettings>,
    instructions: Option<InstructionSettings>,
    logging_enabled: Option<bool>,
    api_token: Option<String>,
    theme: Option<String>,
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
    let models = state.ollama.list_models(&config.ollama_endpoint).await;

    Json(HealthResponse {
        service: "llama-harness".to_string(),
        running: true,
        ollama_reachable: models.is_ok(),
        ollama_endpoint: config.ollama_endpoint,
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
        .ollama
        .list_models(&config.ollama_endpoint)
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
        config.clone()
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
    let run_id = Uuid::new_v4().to_string();
    let started_at = Utc::now();
    let started_timer = Instant::now();
    let prompt_summary = summarize_messages(&resolved.messages);

    let response = state
        .ollama
        .chat_stream(
            &resolved.ollama_endpoint,
            resolved.model.clone(),
            resolved.messages.clone(),
            resolved.generation.clone(),
        )
        .await;

    let response = match response {
        Ok(response) => response,
        Err(err) => {
            let ended_at = Utc::now();
            let record = RunRecord {
                id: run_id,
                model: resolved.model,
                source_app: resolved.source_app,
                prompt_summary,
                response_summary: None,
                status: RunStatus::Failed,
                started_at,
                ended_at,
                duration_ms: duration_ms(started_timer.elapsed()),
                error: Some(err.to_string()),
            };
            record_run(&state, record).await;
            return Err(api_error(StatusCode::BAD_GATEWAY, err.to_string()));
        }
    };

    let stream_state = state.clone();
    let stream = async_stream::stream! {
        let mut bytes = response.bytes_stream();
        let mut buffer = String::new();
        let mut response_summary = String::new();
        let mut error: Option<String> = None;

        while let Some(chunk) = bytes.next().await {
            match chunk {
                Ok(bytes) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));
                    while let Some(index) = buffer.find('\n') {
                        let line = buffer[..index].trim().to_string();
                        buffer.drain(..=index);
                        if line.is_empty() {
                            continue;
                        }

                        match serde_json::from_str::<serde_json::Value>(&line) {
                            Ok(value) => {
                                if let Some(content) = value.pointer("/message/content").and_then(|value| value.as_str()) {
                                    response_summary.push_str(content);
                                    if let Ok(event) = Event::default()
                                        .event("token")
                                        .json_data(StreamEvent {
                                            run_id: &run_id,
                                            content,
                                        }) {
                                        yield Ok(event);
                                    }
                                }

                                if value.get("done").and_then(|value| value.as_bool()).unwrap_or(false) {
                                    if let Ok(event) = Event::default()
                                        .event("done")
                                        .json_data(serde_json::json!({ "run_id": run_id })) {
                                        yield Ok(event);
                                    }
                                }
                            }
                            Err(_) => {
                                yield Ok(Event::default().event("raw").data(line));
                            }
                        }
                    }
                }
                Err(err) => {
                    let message = err.to_string();
                    error = Some(message.clone());
                    if let Ok(event) = Event::default()
                        .event("error")
                        .json_data(serde_json::json!({ "run_id": run_id, "error": message })) {
                        yield Ok(event);
                    }
                    break;
                }
            }
        }

        let ended_at = Utc::now();
        let record = RunRecord {
            id: run_id,
            model: resolved.model,
            source_app: resolved.source_app,
            prompt_summary,
            response_summary: Some(truncate(&response_summary, 500)),
            status: if error.is_some() { RunStatus::Failed } else { RunStatus::Completed },
            started_at,
            ended_at,
            duration_ms: duration_ms(started_timer.elapsed()),
            error,
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
    Json(state.config.read().await.clone())
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

        config::save_config(&state.config_path, &config)
            .await
            .map_err(|err| api_error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
        config.clone()
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
    let run_id = Uuid::new_v4().to_string();
    let started_at = Utc::now();
    let started_timer = Instant::now();
    let prompt_summary = summarize_messages(&resolved.messages);

    let result = state
        .ollama
        .chat(
            &resolved.ollama_endpoint,
            resolved.model.clone(),
            resolved.messages.clone(),
            resolved.generation,
        )
        .await;

    let ended_at = Utc::now();
    let duration_ms = duration_ms(started_timer.elapsed());

    match result {
        Ok(response) => {
            let message = response.message.unwrap_or(ChatMessage {
                role: "assistant".to_string(),
                content: String::new(),
            });
            let record = RunRecord {
                id: run_id.clone(),
                model: resolved.model.clone(),
                source_app: resolved.source_app,
                prompt_summary,
                response_summary: Some(truncate(&message.content, 500)),
                status: RunStatus::Completed,
                started_at,
                ended_at,
                duration_ms,
                error: None,
            };
            record_run(&state, record).await;

            Ok(Json(ChatResponse {
                run_id,
                model: resolved.model,
                message,
                started_at,
                ended_at,
                duration_ms,
            }))
        }
        Err(err) => {
            let error = err.to_string();
            let record = RunRecord {
                id: run_id,
                model: resolved.model,
                source_app: resolved.source_app,
                prompt_summary,
                response_summary: None,
                status: RunStatus::Failed,
                started_at,
                ended_at,
                duration_ms,
                error: Some(error.clone()),
            };
            record_run(&state, record).await;
            Err(api_error(StatusCode::BAD_GATEWAY, error))
        }
    }
}

#[derive(Clone)]
struct ResolvedChat {
    ollama_endpoint: String,
    model: String,
    source_app: Option<String>,
    messages: Vec<ChatMessage>,
    generation: GenerationSettings,
}

async fn resolve_chat(state: &AppState, payload: ChatRequest) -> ApiResult<ResolvedChat> {
    let config = state.config.read().await.clone();
    let messages = normalize_messages(payload.messages, payload.prompt)?;
    let messages = apply_instructions(&config.instructions, payload.instructions, messages);
    let model = payload
        .model
        .filter(|model| !model.trim().is_empty())
        .or(config.default_model.clone())
        .ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                "model is required because no default model is configured",
            )
        })?;

    Ok(ResolvedChat {
        ollama_endpoint: config.ollama_endpoint,
        model,
        source_app: payload.source_app,
        messages,
        generation: payload.generation.unwrap_or(config.generation),
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
            return Ok(vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }]);
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
        messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: parts.join("\n\n"),
            },
        );
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

fn summarize_messages(messages: &[ChatMessage]) -> String {
    let joined = messages
        .iter()
        .map(|message| {
            if message.role == "system" {
                "system: [instructions applied]".to_string()
            } else {
                format!("{}: {}", message.role, message.content.replace('\n', " "))
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
            vec![ChatMessage {
                role: "user".to_string(),
                content: "Summarize this note.".to_string(),
            }],
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert!(messages[0]
            .content
            .contains("You are a local automation assistant."));
        assert!(messages[0]
            .content
            .contains("Available tools and tool instructions"));
        assert!(messages[0].content.contains("Prefer concise answers."));
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
            vec![ChatMessage {
                role: "user".to_string(),
                content: "List tasks.".to_string(),
            }],
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert!(messages[0].content.contains("Use bullet points."));
        assert!(!messages[0].content.contains("Ignored"));
    }

    #[test]
    fn run_summary_masks_instruction_text() {
        let summary = summarize_messages(&[
            ChatMessage {
                role: "system".to_string(),
                content: "Do not leak this instruction.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ]);

        assert_eq!(summary, "system: [instructions applied] | user: Hello");
    }
}
