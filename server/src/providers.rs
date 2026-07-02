use crate::{
    config::{AppConfig, GenerationSettings},
    litellm::LiteLlmProvider,
    ollama::{OllamaClient, OllamaModel, OllamaProvider},
};
use anyhow::Result;
use async_trait::async_trait;
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use std::{pin::Pin, sync::Arc};

pub type ProviderEventStream = Pin<Box<dyn Stream<Item = ProviderStreamEvent> + Send>>;

#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn id(&self) -> &'static str;

    async fn chat_completion(&self, request: ProviderChatRequest) -> Result<ProviderChatResponse>;

    async fn stream_chat_completion(
        &self,
        request: ProviderChatRequest,
    ) -> Result<ProviderEventStream>;
}

#[derive(Clone)]
pub struct ProviderRegistry {
    ollama_client: OllamaClient,
}

impl ProviderRegistry {
    pub fn new(ollama_client: OllamaClient) -> Self {
        Self { ollama_client }
    }

    pub fn get(&self, provider_id: &str, config: &AppConfig) -> Option<Arc<dyn ModelProvider>> {
        match provider_id.trim().to_ascii_lowercase().as_str() {
            "ollama" => Some(Arc::new(OllamaProvider::new(
                self.ollama_client.clone(),
                config.ollama_endpoint.clone(),
            ))),
            "litellm" => Some(Arc::new(LiteLlmProvider::new(
                config.litellm.clone(),
                config.litellm_providers.clone(),
                config.model_routes.clone(),
            ))),
            _ => None,
        }
    }

    pub async fn list_ollama_models(&self, config: &AppConfig) -> Result<Vec<OllamaModel>> {
        self.ollama_client
            .list_models(&config.ollama_endpoint)
            .await
    }

    pub async fn litellm_healthy(&self, config: &AppConfig) -> bool {
        LiteLlmProvider::new(
            config.litellm.clone(),
            config.litellm_providers.clone(),
            config.model_routes.clone(),
        )
        .health()
        .await
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
}

impl ChatMessage {
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: MessageContent::Text(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn content_text(&self) -> String {
        self.content.as_text()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Structured(serde_json::Value),
}

impl MessageContent {
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(value) => value.clone(),
            Self::Structured(value) => serde_json::to_string(value).unwrap_or_default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProviderChatRequest {
    pub provider: String,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub tools: Option<serde_json::Value>,
    pub tool_choice: Option<serde_json::Value>,
    pub stream: bool,
    pub metadata: Option<serde_json::Value>,
}

impl ProviderChatRequest {
    pub fn generation_settings(&self) -> GenerationSettings {
        let defaults = GenerationSettings::default();
        GenerationSettings {
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            max_tokens: self.max_tokens.unwrap_or(defaults.max_tokens),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct TokenUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderChatResponse {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<serde_json::Value>,
}

#[derive(Clone, Debug)]
pub enum ProviderStreamEvent {
    Token {
        content: String,
    },
    Done {
        usage: Option<TokenUsage>,
        raw: Option<serde_json::Value>,
    },
    Error {
        error: String,
    },
    Raw {
        data: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_resolves_ollama() {
        let registry = ProviderRegistry::new(OllamaClient::new());
        let config = AppConfig::default();

        assert!(registry.get("ollama", &config).is_some());
        assert!(registry.get("litellm", &config).is_some());
        assert!(registry.get("missing", &config).is_none());
    }
}
