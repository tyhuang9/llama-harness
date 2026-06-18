use crate::{
    config::{LiteLlmProviderConfig, LiteLlmSettings, ModelRoute},
    providers::{
        ProviderChatRequest, ProviderChatResponse, ProviderEventStream, ProviderStreamEvent,
        TokenUsage,
    },
};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{Client, RequestBuilder, Response, StatusCode};
use serde::Serialize;
use serde_json::{Map, Value};
use std::{path::Path, time::Duration};

use crate::providers::ModelProvider;

#[derive(Clone)]
pub struct LiteLlmProvider {
    settings: LiteLlmSettings,
    providers: Vec<LiteLlmProviderConfig>,
    routes: Vec<ModelRoute>,
    http: Client,
}

#[derive(Debug, PartialEq, Eq)]
struct LiteLlmStreamUpdate {
    token: Option<String>,
    done: bool,
    usage: Option<TokenUsage>,
    raw: Option<String>,
}

#[derive(Debug, Serialize)]
struct LiteLlmConfig {
    model_list: Vec<LiteLlmConfigModel>,
    litellm_settings: LiteLlmRuntimeSettings,
    general_settings: LiteLlmGeneralSettings,
}

#[derive(Debug, Serialize)]
struct LiteLlmConfigModel {
    model_name: String,
    litellm_params: LiteLlmConfigParams,
}

#[derive(Debug, Serialize)]
struct LiteLlmConfigParams {
    model: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    api_key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_base: Option<String>,
}

#[derive(Debug, Serialize)]
struct LiteLlmRuntimeSettings {
    check_provider_endpoint: bool,
}

#[derive(Debug, Serialize)]
struct LiteLlmGeneralSettings {
    master_key: String,
}

pub struct LiteLlmConfigGeneration {
    pub routes_written: usize,
    pub providers_written: usize,
    pub entries_written: usize,
}

impl LiteLlmProvider {
    pub fn new(
        settings: LiteLlmSettings,
        providers: Vec<LiteLlmProviderConfig>,
        routes: Vec<ModelRoute>,
    ) -> Self {
        let timeout = Duration::from_millis(settings.timeout_ms.max(1));
        let http = Client::builder()
            .timeout(timeout)
            .build()
            .expect("failed to build reqwest client");

        Self {
            settings,
            providers,
            routes,
            http,
        }
    }

    pub async fn health(&self) -> bool {
        if !self.settings.enabled {
            return false;
        }

        self.health_request("/health/readiness").await
            || self.health_request("/health").await
            || self.health_request("/v1/models").await
    }

    fn ensure_ready(&self, model: &str) -> Result<()> {
        if !self.settings.enabled {
            return Err(anyhow!("LiteLLM is disabled in settings."));
        }
        if model.trim().is_empty() {
            return Err(anyhow!("model is required for LiteLLM requests"));
        }
        if self.settings.allow_unconfigured_models
            || self.has_model_route(model)
            || self.has_provider_for_model(model)
        {
            return Ok(());
        }

        Err(anyhow!(
            "LiteLLM model '{model}' is not configured. Add a provider, add an advanced route, or enable allow_unconfigured_models."
        ))
    }

    fn has_model_route(&self, model: &str) -> bool {
        self.routes
            .iter()
            .any(|route| route.enabled && route.model_alias == model)
    }

    fn has_provider_for_model(&self, model: &str) -> bool {
        self.providers
            .iter()
            .any(|provider| provider.enabled && model_belongs_to_provider(provider, model))
    }

    async fn health_request(&self, path: &str) -> bool {
        let request = self.with_auth(self.http.get(url(&self.settings.base_url, path)));
        request
            .send()
            .await
            .map(|response| response.status().is_success())
            .unwrap_or(false)
    }

    async fn send_chat_request(&self, body: Value) -> Result<Response> {
        let request = self
            .with_auth(
                self.http
                    .post(url(&self.settings.base_url, "/v1/chat/completions")),
            )
            .json(&body);

        let response = request.send().await.map_err(|err| {
            if err.is_connect() || err.is_timeout() {
                anyhow!(
                    "LiteLLM proxy is not reachable at {}.",
                    self.settings.base_url.trim_end_matches('/')
                )
            } else {
                anyhow!("failed to call LiteLLM proxy: {err}")
            }
        })?;

        ensure_litellm_success(response, self.settings.api_key.as_deref()).await
    }

    fn with_auth(&self, request: RequestBuilder) -> RequestBuilder {
        match self.settings.api_key.as_deref().map(str::trim) {
            Some(api_key) if !api_key.is_empty() => request.bearer_auth(api_key),
            _ => request,
        }
    }
}

#[async_trait]
impl ModelProvider for LiteLlmProvider {
    fn id(&self) -> &'static str {
        "litellm"
    }

    async fn chat_completion(&self, request: ProviderChatRequest) -> Result<ProviderChatResponse> {
        self.ensure_ready(&request.model)?;
        let body = build_litellm_chat_body(&request);
        let response = self.send_chat_request(body).await?;
        let raw = response
            .json::<Value>()
            .await
            .context("failed to decode LiteLLM chat response")?;

        parse_litellm_chat_response(raw, &request.model, self.id())
    }

    async fn stream_chat_completion(
        &self,
        mut request: ProviderChatRequest,
    ) -> Result<ProviderEventStream> {
        self.ensure_ready(&request.model)?;
        request.stream = true;
        let body = build_litellm_chat_body(&request);
        let response = self.send_chat_request(body).await?;

        Ok(parse_litellm_stream(response))
    }
}

pub async fn generate_litellm_config(
    providers: &[LiteLlmProviderConfig],
    routes: &[ModelRoute],
    output_path: &Path,
    ollama_endpoint: &str,
) -> Result<LiteLlmConfigGeneration> {
    let provider_entries = providers
        .iter()
        .filter(|provider| provider.enabled)
        .filter_map(|provider| {
            let model_name = provider_wildcard_model(provider)?;
            Some(LiteLlmConfigModel {
                model_name: model_name.clone(),
                litellm_params: LiteLlmConfigParams {
                    model: model_name,
                    api_key: env_reference(&provider.api_key_env_var),
                    api_base: provider_api_base(provider, ollama_endpoint),
                },
            })
        })
        .collect::<Vec<_>>();

    let route_entries = routes
        .iter()
        .filter(|route| {
            route.enabled
                && !route.model_alias.trim().is_empty()
                && !route.litellm_model.trim().is_empty()
        })
        .map(|route| LiteLlmConfigModel {
            model_name: route.model_alias.trim().to_string(),
            litellm_params: LiteLlmConfigParams {
                model: route.litellm_model.trim().to_string(),
                api_key: env_reference(&route.api_key_env_var),
                api_base: route
                    .api_base
                    .as_ref()
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty()),
            },
        })
        .collect::<Vec<_>>();

    let providers_written = provider_entries.len();
    let routes_written = route_entries.len();

    let mut model_list: Vec<LiteLlmConfigModel> =
        provider_entries.into_iter().chain(route_entries).collect();
    append_default_example_routes(&mut model_list, ollama_endpoint);

    let config = LiteLlmConfig {
        model_list,
        litellm_settings: LiteLlmRuntimeSettings {
            check_provider_endpoint: true,
        },
        general_settings: LiteLlmGeneralSettings {
            master_key: "os.environ/LITELLM_MASTER_KEY".to_string(),
        },
    };
    let entries_written = config.model_list.len();
    let contents = serde_yaml::to_string(&config).context("failed to encode LiteLLM config")?;

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    tokio::fs::write(output_path, contents)
        .await
        .with_context(|| format!("failed to write {}", output_path.display()))?;

    Ok(LiteLlmConfigGeneration {
        routes_written,
        providers_written,
        entries_written,
    })
}

fn append_default_example_routes(entries: &mut Vec<LiteLlmConfigModel>, ollama_endpoint: &str) {
    if !entries
        .iter()
        .any(|entry| entry.model_name == "gpt-4o-mini")
    {
        entries.push(LiteLlmConfigModel {
            model_name: "gpt-4o-mini".to_string(),
            litellm_params: LiteLlmConfigParams {
                model: "openai/gpt-4o-mini".to_string(),
                api_key: env_reference("OPENAI_API_KEY"),
                api_base: None,
            },
        });
    }

    if !entries
        .iter()
        .any(|entry| entry.model_name == "local-llama")
    {
        entries.push(LiteLlmConfigModel {
            model_name: "local-llama".to_string(),
            litellm_params: LiteLlmConfigParams {
                model: "ollama/llama3.1".to_string(),
                api_key: String::new(),
                api_base: Some(ollama_endpoint.trim().to_string())
                    .filter(|value| !value.is_empty())
                    .or_else(|| Some("http://127.0.0.1:11434".to_string())),
            },
        });
    }
}

pub fn litellm_model_for_provider(provider: &LiteLlmProviderConfig, model: &str) -> String {
    let model = model.trim();
    let model_prefix = litellm_model_prefix(provider);
    if model.is_empty() || model_prefix.is_empty() {
        return model.to_string();
    }

    let prefix = format!("{model_prefix}/");
    if model.starts_with(&prefix) {
        return model.to_string();
    }

    format!("{prefix}{model}")
}

fn model_belongs_to_provider(provider: &LiteLlmProviderConfig, model: &str) -> bool {
    let model_prefix = litellm_model_prefix(provider);
    if model_prefix.is_empty() {
        return false;
    }

    let model = model.trim();
    model.starts_with(&format!("{model_prefix}/"))
}

fn provider_wildcard_model(provider: &LiteLlmProviderConfig) -> Option<String> {
    let model_prefix = litellm_model_prefix(provider);
    if model_prefix.is_empty() {
        return None;
    }
    Some(format!("{model_prefix}/*"))
}

fn litellm_model_prefix(provider: &LiteLlmProviderConfig) -> String {
    match normalize_provider_type(&provider.provider_type).as_str() {
        "ollama" => "ollama_chat".to_string(),
        provider_type => provider_type.to_string(),
    }
}

fn provider_api_base(provider: &LiteLlmProviderConfig, ollama_endpoint: &str) -> Option<String> {
    provider
        .api_base
        .as_ref()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            if normalize_provider_type(&provider.provider_type) == "ollama" {
                Some(ollama_endpoint.trim().to_string()).filter(|value| !value.is_empty())
            } else {
                None
            }
        })
}

fn normalize_provider_type(provider_type: &str) -> String {
    provider_type
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-'], "_")
}

fn build_litellm_chat_body(request: &ProviderChatRequest) -> Value {
    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(request.model.clone()));
    body.insert(
        "messages".to_string(),
        serde_json::to_value(&request.messages).unwrap_or(Value::Array(Vec::new())),
    );
    insert_optional_f32(&mut body, "temperature", request.temperature);
    insert_optional_f32(&mut body, "top_p", request.top_p);
    if let Some(max_tokens) = request.max_tokens {
        body.insert("max_tokens".to_string(), Value::from(max_tokens));
    }
    if let Some(tools) = &request.tools {
        body.insert("tools".to_string(), tools.clone());
    }
    if let Some(tool_choice) = &request.tool_choice {
        body.insert("tool_choice".to_string(), tool_choice.clone());
    }
    if let Some(metadata) = &request.metadata {
        body.insert("metadata".to_string(), metadata.clone());
    }
    body.insert("stream".to_string(), Value::Bool(request.stream));
    if request.stream {
        body.insert(
            "stream_options".to_string(),
            serde_json::json!({ "include_usage": true }),
        );
    }

    Value::Object(body)
}

fn parse_litellm_chat_response(
    raw: Value,
    fallback_model: &str,
    provider: &str,
) -> Result<ProviderChatResponse> {
    let message = raw
        .pointer("/choices/0/message")
        .ok_or_else(|| anyhow!("LiteLLM response did not include choices[0].message"))?;
    let content = message
        .get("content")
        .map(content_value_to_string)
        .unwrap_or_default();
    let tool_calls = message.get("tool_calls").cloned();
    let usage = usage_from_value(raw.get("usage"));
    let model = raw
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| Some(fallback_model.to_string()));

    Ok(ProviderChatResponse {
        content,
        tool_calls,
        usage,
        model,
        provider: Some(provider.to_string()),
        raw_response: Some(raw),
    })
}

fn parse_litellm_stream(response: Response) -> ProviderEventStream {
    Box::pin(async_stream::stream! {
        let mut bytes = response.bytes_stream();
        let mut buffer = String::new();
        let mut usage: Option<TokenUsage> = None;
        let mut saw_done = false;

        while let Some(chunk) = bytes.next().await {
            match chunk {
                Ok(bytes) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));
                    while let Some(index) = buffer.find('\n') {
                        let line = buffer[..index].trim().to_string();
                        buffer.drain(..=index);
                        let update = parse_litellm_stream_line(&line);
                        if let Some(token) = update.token {
                            yield ProviderStreamEvent::Token { content: token };
                        }
                        if let Some(event_usage) = update.usage {
                            usage = Some(event_usage);
                        }
                        if let Some(raw) = update.raw {
                            yield ProviderStreamEvent::Raw { data: raw };
                        }
                        if update.done {
                            saw_done = true;
                            yield ProviderStreamEvent::Done {
                                usage: usage.clone(),
                                raw: None,
                            };
                        }
                    }
                }
                Err(err) => {
                    yield ProviderStreamEvent::Error {
                        error: err.to_string(),
                    };
                    return;
                }
            }
        }

        if !saw_done {
            yield ProviderStreamEvent::Done { usage, raw: None };
        }
    })
}

fn parse_litellm_stream_line(line: &str) -> LiteLlmStreamUpdate {
    let line = line.trim();
    if line.is_empty() || line.starts_with(':') {
        return empty_stream_update();
    }

    let payload = line.strip_prefix("data:").map(str::trim).unwrap_or(line);
    if payload == "[DONE]" {
        return LiteLlmStreamUpdate {
            token: None,
            done: true,
            usage: None,
            raw: None,
        };
    }

    let value = match serde_json::from_str::<Value>(payload) {
        Ok(value) => value,
        Err(_) => {
            return LiteLlmStreamUpdate {
                token: None,
                done: false,
                usage: None,
                raw: Some(line.to_string()),
            };
        }
    };

    LiteLlmStreamUpdate {
        token: value
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string),
        done: false,
        usage: usage_from_value(value.get("usage")),
        raw: None,
    }
}

async fn ensure_litellm_success(response: Response, api_key: Option<&str>) -> Result<Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = response.text().await.unwrap_or_default();
    if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
        return Err(anyhow!(
            "LiteLLM authentication failed. Check the LiteLLM API key."
        ));
    }

    let message = extract_error_message(&body)
        .filter(|message| !message.trim().is_empty())
        .unwrap_or_else(|| body.trim().to_string());
    let message = redact_secret(&message, api_key);

    Err(anyhow!(
        "LiteLLM returned {}{}",
        status,
        if message.is_empty() {
            String::new()
        } else {
            format!(": {message}")
        }
    ))
}

fn insert_optional_f32(body: &mut Map<String, Value>, key: &str, value: Option<f32>) {
    if let Some(value) = value {
        if let Some(number) = serde_json::Number::from_f64(f64::from(value)) {
            body.insert(key.to_string(), Value::Number(number));
        }
    }
}

fn content_value_to_string(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| serde_json::to_string(value).unwrap_or_default())
}

fn usage_from_value(value: Option<&Value>) -> Option<TokenUsage> {
    let value = value?;
    let usage = TokenUsage {
        input_tokens: value.get("prompt_tokens").and_then(Value::as_u64),
        output_tokens: value.get("completion_tokens").and_then(Value::as_u64),
        total_tokens: value.get("total_tokens").and_then(Value::as_u64),
    };
    if usage.input_tokens.is_some() || usage.output_tokens.is_some() || usage.total_tokens.is_some()
    {
        Some(usage)
    } else {
        None
    }
}

fn extract_error_message(body: &str) -> Option<String> {
    let value = serde_json::from_str::<Value>(body).ok()?;
    value
        .pointer("/error/message")
        .or_else(|| value.get("message"))
        .or_else(|| value.get("detail"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn redact_secret(value: &str, secret: Option<&str>) -> String {
    match secret.map(str::trim).filter(|secret| !secret.is_empty()) {
        Some(secret) => value.replace(secret, "[redacted]"),
        None => value.to_string(),
    }
}

fn env_reference(name: &str) -> String {
    let name = name.trim();
    if name.is_empty() {
        String::new()
    } else {
        format!("os.environ/{name}")
    }
}

fn empty_stream_update() -> LiteLlmStreamUpdate {
    LiteLlmStreamUpdate {
        token: None,
        done: false,
        usage: None,
        raw: None,
    }
}

fn url(endpoint: &str, path: &str) -> String {
    format!("{}{}", endpoint.trim_end_matches('/'), path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::ChatMessage;

    #[test]
    fn request_payload_is_openai_compatible() {
        let request = ProviderChatRequest {
            provider: "litellm".to_string(),
            model: "openai:gpt-4o".to_string(),
            messages: vec![ChatMessage::text("user", "hello")],
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(2048),
            tools: Some(serde_json::json!([{ "type": "function" }])),
            tool_choice: Some(serde_json::json!("auto")),
            stream: true,
            metadata: Some(serde_json::json!({ "source": "test" })),
        };

        let body = build_litellm_chat_body(&request);

        assert_eq!(body["model"], "openai:gpt-4o");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello");
        assert!((body["temperature"].as_f64().unwrap() - 0.7).abs() < 0.000_001);
        assert!((body["top_p"].as_f64().unwrap() - 0.9).abs() < 0.000_001);
        assert_eq!(body["max_tokens"], 2048);
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn response_parser_extracts_content_and_usage() {
        let raw = serde_json::json!({
            "model": "openai:gpt-4o",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello back",
                        "tool_calls": [{ "id": "call_1" }]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7
            }
        });

        let parsed =
            parse_litellm_chat_response(raw, "fallback", "litellm").expect("response parses");

        assert_eq!(parsed.content, "hello back");
        assert_eq!(parsed.model.as_deref(), Some("openai:gpt-4o"));
        assert_eq!(parsed.usage.expect("usage").total_tokens, Some(7));
        assert!(parsed.tool_calls.is_some());
    }

    #[test]
    fn stream_parser_handles_token_and_done_lines() {
        let token = parse_litellm_stream_line(
            r#"data: {"choices":[{"delta":{"content":"hi"}}],"usage":null}"#,
        );
        let done = parse_litellm_stream_line("data: [DONE]");

        assert_eq!(token.token.as_deref(), Some("hi"));
        assert!(!token.done);
        assert!(done.done);
    }

    #[test]
    fn generated_config_uses_env_references() {
        let providers = vec![LiteLlmProviderConfig {
            id: "openai_main".to_string(),
            enabled: true,
            provider_type: "openai".to_string(),
            display_name: "OpenAI".to_string(),
            api_key_env_var: "OPENAI_API_KEY".to_string(),
            api_key: Some("sk-secret".to_string()),
            api_base: None,
        }];
        let routes = vec![ModelRoute {
            id: "route_openai".to_string(),
            enabled: true,
            display_name: "OpenAI".to_string(),
            provider: "litellm".to_string(),
            provider_family: "openai".to_string(),
            model_alias: "openai:gpt-4o".to_string(),
            litellm_model: "openai/gpt-4o".to_string(),
            api_key_env_var: "OPENAI_API_KEY".to_string(),
            api_base: None,
            notes: None,
        }];
        let config = LiteLlmConfig {
            model_list: providers
                .iter()
                .map(|provider| {
                    let model_name = provider_wildcard_model(provider).expect("wildcard");
                    LiteLlmConfigModel {
                        model_name: model_name.clone(),
                        litellm_params: LiteLlmConfigParams {
                            model: model_name,
                            api_key: env_reference(&provider.api_key_env_var),
                            api_base: None,
                        },
                    }
                })
                .chain(routes.iter().map(|route| LiteLlmConfigModel {
                    model_name: route.model_alias.clone(),
                    litellm_params: LiteLlmConfigParams {
                        model: route.litellm_model.clone(),
                        api_key: env_reference(&route.api_key_env_var),
                        api_base: None,
                    },
                }))
                .collect(),
            litellm_settings: LiteLlmRuntimeSettings {
                check_provider_endpoint: true,
            },
            general_settings: LiteLlmGeneralSettings {
                master_key: "os.environ/LITELLM_MASTER_KEY".to_string(),
            },
        };

        let yaml = serde_yaml::to_string(&config).expect("yaml");

        assert!(yaml.contains("model_name: openai/*"));
        assert!(yaml.contains("model: openai/*"));
        assert!(yaml.contains("check_provider_endpoint: true"));
        assert!(yaml.contains("api_key: os.environ/OPENAI_API_KEY"));
        assert!(yaml.contains("master_key: os.environ/LITELLM_MASTER_KEY"));
        assert!(!yaml.contains("sk-"));
    }

    #[test]
    fn ollama_provider_uses_chat_prefix_and_api_base() {
        let provider = LiteLlmProviderConfig {
            id: "local_ollama".to_string(),
            enabled: true,
            provider_type: "ollama".to_string(),
            display_name: "Ollama".to_string(),
            api_key_env_var: String::new(),
            api_key: None,
            api_base: None,
        };

        assert_eq!(
            litellm_model_for_provider(&provider, "llama3.2"),
            "ollama_chat/llama3.2"
        );
        assert_eq!(
            provider_wildcard_model(&provider).as_deref(),
            Some("ollama_chat/*")
        );
        assert_eq!(
            provider_api_base(&provider, "http://localhost:11434").as_deref(),
            Some("http://localhost:11434")
        );
    }
}
