use crate::{
    streaming::{stream_response, OllamaEventStream},
    wire::{chat_request, tool_calls, usage, ChatResponse, TagsResponse},
};
use async_trait::async_trait;
use futures_util::StreamExt;
use llama_harness_core::{
    HarnessError, ModelCapabilities, ModelEventStream, ModelInfo, ModelProvider, ModelRequest,
    ModelResponse, ModelStreamController, ModelStreamEvent, ProviderHealth, ToolCallAssembler,
    ToolCallAssemblyLimits, ToolCallDelta,
};
use reqwest::{Client, RequestBuilder, Response, StatusCode, Url};
use serde::Deserialize;
use std::{net::IpAddr, time::Duration};
use tokio_util::sync::CancellationToken;

/// Default loopback URL used by [`OllamaProviderBuilder`].
pub const DEFAULT_OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";
/// Default timeout applied to Ollama HTTP requests.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_MAX_RESPONSE_BYTES: usize = 1024 * 1024;
const DEFAULT_MAX_STREAM_LINE_BYTES: usize = 1024 * 1024;
const ERROR_BODY_LIMIT_BYTES: usize = 8 * 1024;

/// Builder for a direct, loopback-only Ollama provider.
///
/// DNS hostnames other than `localhost` and non-loopback addresses are rejected
/// deliberately. Applications that need remote inference should provide a distinct
/// provider rather than turning the local Ollama integration into an SSRF primitive.
#[derive(Clone, Debug)]
pub struct OllamaProviderBuilder {
    base_url: String,
    request_timeout: Duration,
    keep_alive: Option<String>,
    max_response_bytes: usize,
    max_stream_bytes: usize,
    max_stream_line_bytes: usize,
}

impl Default for OllamaProviderBuilder {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_OLLAMA_BASE_URL.into(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            keep_alive: None,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            max_stream_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            max_stream_line_bytes: DEFAULT_MAX_STREAM_LINE_BYTES,
        }
    }
}

impl OllamaProviderBuilder {
    /// Sets the loopback URL for the Ollama server.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets the timeout for each Ollama HTTP request.
    pub fn request_timeout(mut self, request_timeout: Duration) -> Self {
        self.request_timeout = request_timeout;
        self
    }

    /// Sets Ollama's per-request model keep-alive value (for example, `"5m"`).
    pub fn keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    /// Sets the maximum size of a non-streaming response body in bytes.
    pub fn max_response_bytes(mut self, max_response_bytes: usize) -> Self {
        self.max_response_bytes = max_response_bytes;
        self
    }

    /// Sets the maximum total size of a streaming response in bytes.
    pub fn max_stream_bytes(mut self, max_stream_bytes: usize) -> Self {
        self.max_stream_bytes = max_stream_bytes;
        self
    }

    /// Sets the maximum size of one newline-delimited streaming response line.
    pub fn max_stream_line_bytes(mut self, max_stream_line_bytes: usize) -> Self {
        self.max_stream_line_bytes = max_stream_line_bytes;
        self
    }

    /// Validates the configuration and constructs an Ollama provider.
    pub fn build(self) -> Result<OllamaProvider, HarnessError> {
        if self.request_timeout.is_zero() {
            return Err(HarnessError::InvalidRequest(
                "Ollama request timeout must be greater than zero".into(),
            ));
        }
        if self.max_response_bytes == 0
            || self.max_stream_bytes == 0
            || self.max_stream_line_bytes == 0
        {
            return Err(HarnessError::InvalidRequest(
                "Ollama response limits must be greater than zero".into(),
            ));
        }
        let base_url = parse_loopback_url(&self.base_url)?;
        let http = Client::builder()
            .timeout(self.request_timeout)
            // A loopback endpoint must not redirect a sidecar request elsewhere.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| HarnessError::Provider(format!("create Ollama client: {error}")))?;
        Ok(OllamaProvider {
            http,
            base_url,
            keep_alive: self.keep_alive.filter(|value| !value.trim().is_empty()),
            max_response_bytes: self.max_response_bytes,
            max_stream_bytes: self.max_stream_bytes,
            max_stream_line_bytes: self.max_stream_line_bytes,
        })
    }
}

#[derive(Clone)]
/// A direct provider for a loopback Ollama server.
pub struct OllamaProvider {
    http: Client,
    base_url: Url,
    keep_alive: Option<String>,
    max_response_bytes: usize,
    max_stream_bytes: usize,
    max_stream_line_bytes: usize,
}

impl OllamaProvider {
    /// Creates a builder initialized with the crate defaults.
    pub fn builder() -> OllamaProviderBuilder {
        OllamaProviderBuilder::default()
    }

    /// Creates a provider using the crate defaults.
    pub fn new() -> Result<Self, HarnessError> {
        Self::builder().build()
    }

    /// Returns the validated base URL used for Ollama requests.
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    /// Starts a typed NDJSON stream for a model request.
    ///
    /// This provider-specific view remains available for compatibility. Generic hosts
    /// can consume the same response through [`ModelProvider::stream`].
    pub async fn stream_chat(
        &self,
        request: ModelRequest,
    ) -> Result<OllamaEventStream, HarnessError> {
        if request.cancellation.is_cancelled() {
            return Err(HarnessError::Cancelled);
        }
        if request.structured_output.is_some() {
            return Err(HarnessError::UnsupportedCapability(
                "Ollama provider does not implement the structured-output contract".into(),
            ));
        }
        let body = chat_request(
            request.model,
            &request.messages,
            &request.tools,
            request.prepared_tools.as_deref(),
            &request.generation,
            self.keep_alive.as_deref(),
            true,
        )?;
        let response = self
            .send(
                self.http.post(self.endpoint("api/chat")?).json(&body),
                &request.cancellation,
            )
            .await?;
        let response = self.ensure_success(response, &request.cancellation).await?;
        Ok(stream_response(
            response,
            request.cancellation,
            self.max_stream_bytes,
            self.max_stream_line_bytes,
        ))
    }

    fn endpoint(&self, path: &str) -> Result<Url, HarnessError> {
        self.base_url
            .join(path)
            .map_err(|error| HarnessError::Provider(format!("build Ollama endpoint: {error}")))
    }

    async fn send(
        &self,
        request: RequestBuilder,
        cancellation: &CancellationToken,
    ) -> Result<Response, HarnessError> {
        tokio::select! {
            _ = cancellation.cancelled() => Err(HarnessError::Cancelled),
            response = request.send() => response.map_err(map_request_error),
        }
    }

    async fn ensure_success(
        &self,
        response: Response,
        cancellation: &CancellationToken,
    ) -> Result<Response, HarnessError> {
        if response.status().is_success() {
            return Ok(response);
        }
        let status = response.status();
        let body = read_bounded(response, ERROR_BODY_LIMIT_BYTES, cancellation)
            .await
            .unwrap_or_default();
        let message = serde_json::from_slice::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| {
                value
                    .get("error")
                    .and_then(|error| error.as_str())
                    .map(str::to_owned)
            })
            .filter(|message| !message.trim().is_empty())
            .unwrap_or_else(|| String::from_utf8_lossy(&body).trim().to_owned());
        let detail = if message.is_empty() {
            format!("Ollama returned {status}")
        } else {
            format!("Ollama returned {status}: {message}")
        };
        if status == StatusCode::REQUEST_TIMEOUT
            || status == StatusCode::TOO_MANY_REQUESTS
            || status.is_server_error()
        {
            Err(HarnessError::RetryableProvider(detail))
        } else {
            Err(HarnessError::Provider(detail))
        }
    }

    async fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
    ) -> Result<T, HarnessError> {
        let cancellation = CancellationToken::new();
        let response = self
            .send(self.http.get(self.endpoint(path)?), &cancellation)
            .await?;
        let response = self.ensure_success(response, &cancellation).await?;
        let bytes = read_bounded(response, self.max_response_bytes, &cancellation).await?;
        serde_json::from_slice(&bytes).map_err(|error| {
            HarnessError::Provider(format!("decode Ollama {path} response: {error}"))
        })
    }
}

#[async_trait]
impl ModelProvider for OllamaProvider {
    fn id(&self) -> &str {
        "ollama"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::new(true, true, false).with_parallel_tool_calls(true)
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        #[derive(Deserialize)]
        struct Version {
            version: Option<String>,
        }

        match self.get_json::<Version>("api/version").await {
            Ok(version) => Ok(ProviderHealth::new(
                true,
                version.version.map(|version| format!("Ollama {version}")),
            )),
            Err(error) => Ok(ProviderHealth::unhealthy(error.to_string())),
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        let response: TagsResponse = self.get_json("api/tags").await?;
        Ok(response
            .models
            .into_iter()
            .map(|model| ModelInfo::new(model.name).with_capabilities(self.capabilities()))
            .collect())
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError> {
        if request.cancellation.is_cancelled() {
            return Err(HarnessError::Cancelled);
        }
        if request.structured_output.is_some() {
            return Err(HarnessError::UnsupportedCapability(
                "Ollama provider does not implement the structured-output contract".into(),
            ));
        }
        let requested_model = request.model.clone();
        let body = chat_request(
            request.model,
            &request.messages,
            &request.tools,
            request.prepared_tools.as_deref(),
            &request.generation,
            self.keep_alive.as_deref(),
            false,
        )?;
        let response = self
            .send(
                self.http.post(self.endpoint("api/chat")?).json(&body),
                &request.cancellation,
            )
            .await?;
        let response = self.ensure_success(response, &request.cancellation).await?;
        let bytes = read_bounded(response, self.max_response_bytes, &request.cancellation).await?;
        let response: ChatResponse = serde_json::from_slice(&bytes).map_err(|error| {
            HarnessError::Provider(format!("decode Ollama chat response: {error}"))
        })?;
        if !response.done {
            return Err(HarnessError::Provider(
                "Ollama non-streaming chat response was not marked done".into(),
            ));
        }
        let message = response.message.as_ref();
        let tool_calls = message.map_or_else(Vec::new, |message| tool_calls(&message.tool_calls));
        let final_output = message.and_then(|message| {
            (!message.content.is_empty() || tool_calls.is_empty()).then(|| message.content.clone())
        });
        let mut model_response =
            ModelResponse::new(response.model.clone().unwrap_or(requested_model))
                .with_tool_calls(tool_calls)
                .with_usage(usage(&response));
        model_response.final_output = final_output;
        Ok(model_response)
    }

    async fn stream(&self, request: ModelRequest) -> Result<ModelEventStream, HarnessError> {
        let assembly_limits = ToolCallAssemblyLimits::for_provider(&self.capabilities().limits)?;
        let assembler = ToolCallAssembler::new(request.tools.clone(), assembly_limits)?;
        let mut controller = ModelStreamController::new(assembler);
        let mut events = OllamaProvider::stream_chat(self, request).await?;
        Ok(Box::pin(async_stream::stream! {
            let mut call_index = 0_usize;
            while let Some(event) = events.next().await {
                let mapped = event.map(|event| match event {
                    crate::OllamaStreamEvent::TextDelta { content } =>
                        ModelStreamEvent::TextDelta { content },
                    crate::OllamaStreamEvent::ToolCall { call } => {
                        let delta = ToolCallDelta::new(
                            call_index,
                            call.arguments_json,
                            true,
                        )
                        .with_call_id(format!("{}-{call_index}", call.id))
                        .with_tool_id(call.tool_id);
                        call_index = call_index.saturating_add(1);
                        ModelStreamEvent::ToolCallDelta(delta)
                    }
                    crate::OllamaStreamEvent::Completed { model, usage } =>
                        ModelStreamEvent::Completed { model, usage },
                });
                match controller.push(mapped) {
                    Ok(validated) => yield Ok(validated.event),
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                }
            }
            if let Err(error) = controller.finish_eof() {
                yield Err(error);
            }
        }))
    }
}

async fn read_bounded(
    response: Response,
    max_bytes: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, HarnessError> {
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes as u64)
    {
        return Err(HarnessError::ResourceLimit(format!(
            "Ollama response exceeds {max_bytes} bytes"
        )));
    }
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = tokio::select! {
        _ = cancellation.cancelled() => return Err(HarnessError::Cancelled),
        chunk = stream.next() => chunk,
    } {
        let chunk = chunk.map_err(map_request_error)?;
        if body.len().saturating_add(chunk.len()) > max_bytes {
            return Err(HarnessError::ResourceLimit(format!(
                "Ollama response exceeds {max_bytes} bytes"
            )));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn parse_loopback_url(value: &str) -> Result<Url, HarnessError> {
    let mut url = Url::parse(value.trim()).map_err(|error| {
        HarnessError::InvalidRequest(format!("invalid Ollama base URL: {error}"))
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(HarnessError::InvalidRequest(
            "Ollama base URL must use http or https".into(),
        ));
    }
    if url.host().is_none() {
        return Err(HarnessError::InvalidRequest(
            "Ollama base URL must include a loopback host".into(),
        ));
    }
    let is_loopback = url.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback())
    });
    if !is_loopback {
        return Err(HarnessError::InvalidRequest(
            "Ollama base URL must use localhost or a loopback address".into(),
        ));
    }
    if !url.path().ends_with('/') {
        url.set_path(&format!("{}/", url.path()));
    }
    Ok(url)
}

fn map_request_error(error: reqwest::Error) -> HarnessError {
    if error.is_timeout() || error.is_connect() {
        HarnessError::RetryableProvider(format!("Ollama connection failed: {error}"))
    } else {
        HarnessError::Provider(format!("Ollama request failed: {error}"))
    }
}
