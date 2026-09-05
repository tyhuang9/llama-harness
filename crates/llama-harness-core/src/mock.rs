use crate::{
    HarnessError, ModelCapabilities, ModelInfo, ModelProvider, ModelRequest, ModelResponse,
    ProviderHealth, Usage,
};
use async_trait::async_trait;
use std::{collections::VecDeque, sync::Mutex};

#[derive(Clone, Debug)]
#[non_exhaustive]
/// One scripted response or failure for the mock provider.
pub enum MockStep {
    /// Return a model response.
    Response(ModelResponse),
    /// Return a harness error.
    Error(HarnessError),
}

/// Deterministic model provider backed by a scripted queue.
pub struct MockModelProvider {
    id: String,
    capabilities: ModelCapabilities,
    steps: Mutex<VecDeque<MockStep>>,
    requests: Mutex<Vec<ModelRequest>>,
}

impl MockModelProvider {
    /// Creates a mock provider that consumes the supplied steps in order.
    pub fn scripted(steps: impl IntoIterator<Item = MockStep>) -> Self {
        Self {
            id: "mock".into(),
            capabilities: ModelCapabilities {
                supports_tools: true,
                supports_streaming: false,
                supports_structured_output: true,
                ..ModelCapabilities::default()
            },
            steps: Mutex::new(steps.into_iter().collect()),
            requests: Mutex::new(vec![]),
        }
    }

    /// Returns the model requests received by the provider.
    pub fn requests(&self) -> Vec<ModelRequest> {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Replaces the capabilities advertised by this scripted provider.
    pub fn with_capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }
}

#[async_trait]
impl ModelProvider for MockModelProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.capabilities.clone()
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth {
            healthy: true,
            detail: None,
        })
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(vec![ModelInfo {
            id: "mock-model".into(),
            capabilities: self.capabilities(),
        }])
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError> {
        if request.cancellation.is_cancelled() {
            return Err(HarnessError::Cancelled);
        }
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(request);
        let step = self
            .steps
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front();
        match step {
            Some(MockStep::Response(response)) => Ok(response),
            Some(MockStep::Error(error)) => Err(error),
            None => Err(HarnessError::Provider("mock script exhausted".into())),
        }
    }
}

/// Creates a scripted successful final response from the mock model.
pub fn final_response(output: impl Into<String>) -> MockStep {
    MockStep::Response(ModelResponse {
        model: "mock-model".into(),
        final_output: Some(output.into()),
        tool_calls: vec![],
        usage: Usage::default(),
    })
}

/// Creates a scripted response containing a tool call.
pub fn tool_response(call: crate::ToolCall) -> MockStep {
    MockStep::Response(ModelResponse {
        model: "mock-model".into(),
        final_output: None,
        tool_calls: vec![call],
        usage: Usage::default(),
    })
}

/// Creates a scripted provider failure.
pub fn error_response(error: HarnessError) -> MockStep {
    MockStep::Error(error)
}
