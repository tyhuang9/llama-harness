use crate::config::GenerationSettings;
use anyhow::{anyhow, Context, Result};
use reqwest::{Client, Response, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone)]
pub struct OllamaClient {
    http: Client,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
    pub options: OllamaOptions,
}

#[derive(Debug, Serialize)]
pub struct OllamaOptions {
    pub temperature: f32,
    pub top_p: f32,
    pub num_predict: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OllamaChatResponse {
    pub model: Option<String>,
    pub created_at: Option<String>,
    pub message: Option<ChatMessage>,
    pub done: Option<bool>,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<u64>,
    pub eval_duration: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OllamaTagsResponse {
    #[serde(default)]
    pub models: Vec<OllamaModel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    pub model: Option<String>,
    pub modified_at: Option<String>,
    pub size: Option<u64>,
    pub digest: Option<String>,
    pub details: Option<OllamaModelDetails>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OllamaModelDetails {
    pub parent_model: Option<String>,
    pub format: Option<String>,
    pub family: Option<String>,
    pub families: Option<Vec<String>>,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
}

impl OllamaClient {
    pub fn new() -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build reqwest client");

        Self { http }
    }

    pub async fn list_models(&self, endpoint: &str) -> Result<Vec<OllamaModel>> {
        let response = self
            .http
            .get(url(endpoint, "/api/tags"))
            .send()
            .await
            .context("failed to call Ollama /api/tags")?;

        ensure_success(response)
            .await?
            .json::<OllamaTagsResponse>()
            .await
            .context("failed to decode Ollama model list")
            .map(|body| body.models)
    }

    pub async fn chat(
        &self,
        endpoint: &str,
        model: String,
        messages: Vec<ChatMessage>,
        generation: GenerationSettings,
    ) -> Result<OllamaChatResponse> {
        let request = OllamaChatRequest {
            model,
            messages,
            stream: false,
            options: generation.into(),
        };

        let response = self
            .http
            .post(url(endpoint, "/api/chat"))
            .json(&request)
            .send()
            .await
            .context("failed to call Ollama /api/chat")?;

        ensure_success(response)
            .await?
            .json::<OllamaChatResponse>()
            .await
            .context("failed to decode Ollama chat response")
    }

    pub async fn chat_stream(
        &self,
        endpoint: &str,
        model: String,
        messages: Vec<ChatMessage>,
        generation: GenerationSettings,
    ) -> Result<Response> {
        let request = OllamaChatRequest {
            model,
            messages,
            stream: true,
            options: generation.into(),
        };

        let response = self
            .http
            .post(url(endpoint, "/api/chat"))
            .json(&request)
            .send()
            .await
            .context("failed to start Ollama streaming chat")?;

        ensure_success(response).await
    }
}

impl From<GenerationSettings> for OllamaOptions {
    fn from(settings: GenerationSettings) -> Self {
        Self {
            temperature: settings.temperature,
            top_p: settings.top_p,
            num_predict: settings.max_tokens,
        }
    }
}

fn url(endpoint: &str, path: &str) -> String {
    format!("{}{}", endpoint.trim_end_matches('/'), path)
}

async fn ensure_success(response: Response) -> Result<Response> {
    let status = response.status();
    if status == StatusCode::OK {
        return Ok(response);
    }

    let body = response.text().await.unwrap_or_default();
    let message = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(|error| error.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| body.trim().to_string());

    Err(anyhow!(
        "Ollama returned {}{}",
        status,
        if message.is_empty() {
            String::new()
        } else {
            format!(": {message}")
        }
    ))
}
