use crate::{
    config::GenerationSettings,
    providers::{
        ChatMessage, ModelProvider, ProviderChatRequest, ProviderChatResponse, ProviderEventStream,
        ProviderStreamEvent, TokenUsage,
    },
};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{Client, Response, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone)]
pub struct OllamaClient {
    http: Client,
}

#[derive(Clone)]
pub struct OllamaProvider {
    client: OllamaClient,
    endpoint: String,
}

#[derive(Debug, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaWireMessage>,
    pub stream: bool,
    pub options: OllamaOptions,
}

#[derive(Debug, Serialize)]
pub struct OllamaWireMessage {
    pub role: String,
    pub content: String,
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
            messages: to_ollama_messages(&messages),
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
            messages: to_ollama_messages(&messages),
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

impl OllamaProvider {
    pub fn new(client: OllamaClient, endpoint: String) -> Self {
        Self { client, endpoint }
    }
}

#[async_trait]
impl ModelProvider for OllamaProvider {
    fn id(&self) -> &'static str {
        "ollama"
    }

    async fn chat_completion(&self, request: ProviderChatRequest) -> Result<ProviderChatResponse> {
        let _ = (
            &request.provider,
            &request.tools,
            &request.tool_choice,
            request.stream,
            &request.metadata,
        );
        let generation = request.generation_settings();
        let response = self
            .client
            .chat(
                &self.endpoint,
                request.model.clone(),
                request.messages,
                generation,
            )
            .await?;

        let content = response
            .message
            .as_ref()
            .map(ChatMessage::content_text)
            .unwrap_or_default();

        Ok(ProviderChatResponse {
            content,
            tool_calls: None,
            usage: usage_from_ollama_response(&response),
            model: response.model.or(Some(request.model)),
            provider: Some(self.id().to_string()),
            raw_response: None,
        })
    }

    async fn stream_chat_completion(
        &self,
        request: ProviderChatRequest,
    ) -> Result<ProviderEventStream> {
        let _ = (
            &request.provider,
            &request.tools,
            &request.tool_choice,
            request.stream,
            &request.metadata,
        );
        let generation = request.generation_settings();
        let response = self
            .client
            .chat_stream(&self.endpoint, request.model, request.messages, generation)
            .await?;

        Ok(parse_ollama_stream(response))
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

fn to_ollama_messages(messages: &[ChatMessage]) -> Vec<OllamaWireMessage> {
    messages
        .iter()
        .map(|message| OllamaWireMessage {
            role: message.role.clone(),
            content: message.content_text(),
        })
        .collect()
}

fn parse_ollama_stream(response: Response) -> ProviderEventStream {
    Box::pin(async_stream::stream! {
        let mut bytes = response.bytes_stream();
        let mut buffer = String::new();

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
                                if let Some(content) = value
                                    .pointer("/message/content")
                                    .and_then(|value| value.as_str())
                                    .filter(|content| !content.is_empty()) {
                                    yield ProviderStreamEvent::Token {
                                        content: content.to_string(),
                                    };
                                }

                                if value
                                    .get("done")
                                    .and_then(|value| value.as_bool())
                                    .unwrap_or(false) {
                                    yield ProviderStreamEvent::Done {
                                        usage: usage_from_ollama_value(&value),
                                        raw: None,
                                    };
                                }
                            }
                            Err(_) => {
                                yield ProviderStreamEvent::Raw { data: line };
                            }
                        }
                    }
                }
                Err(err) => {
                    yield ProviderStreamEvent::Error {
                        error: err.to_string(),
                    };
                    break;
                }
            }
        }
    })
}

fn usage_from_ollama_response(response: &OllamaChatResponse) -> Option<TokenUsage> {
    let usage = TokenUsage {
        input_tokens: response.prompt_eval_count,
        output_tokens: response.eval_count,
        total_tokens: sum_tokens(response.prompt_eval_count, response.eval_count),
    };
    usage_if_present(usage)
}

fn usage_from_ollama_value(value: &serde_json::Value) -> Option<TokenUsage> {
    let input_tokens = value
        .get("prompt_eval_count")
        .and_then(|value| value.as_u64());
    let output_tokens = value.get("eval_count").and_then(|value| value.as_u64());
    let usage = TokenUsage {
        input_tokens,
        output_tokens,
        total_tokens: sum_tokens(input_tokens, output_tokens),
    };
    usage_if_present(usage)
}

fn usage_if_present(usage: TokenUsage) -> Option<TokenUsage> {
    if usage.input_tokens.is_some() || usage.output_tokens.is_some() || usage.total_tokens.is_some()
    {
        Some(usage)
    } else {
        None
    }
}

fn sum_tokens(input_tokens: Option<u64>, output_tokens: Option<u64>) -> Option<u64> {
    match (input_tokens, output_tokens) {
        (Some(input), Some(output)) => Some(input + output),
        _ => None,
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
