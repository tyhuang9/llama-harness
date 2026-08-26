use crate::{
    HarnessError, ModelCapabilities, ModelInfo, ModelProvider, ModelRequest, ModelResponse,
    ProviderHealth, Usage,
};
use async_trait::async_trait;
use std::{collections::VecDeque, sync::Mutex};

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum MockStep {
    Response(ModelResponse),
    Error(HarnessError),
}

pub struct MockModelProvider {
    id: String,
    steps: Mutex<VecDeque<MockStep>>,
    requests: Mutex<Vec<ModelRequest>>,
}

impl MockModelProvider {
    pub fn scripted(steps: impl IntoIterator<Item = MockStep>) -> Self {
        Self {
            id: "mock".into(),
            steps: Mutex::new(steps.into_iter().collect()),
            requests: Mutex::new(vec![]),
        }
    }

    pub fn requests(&self) -> Vec<ModelRequest> {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

#[async_trait]
impl ModelProvider for MockModelProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            supports_tools: true,
            supports_streaming: false,
            supports_structured_output: true,
        }
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

pub fn final_response(output: impl Into<String>) -> MockStep {
    MockStep::Response(ModelResponse {
        model: "mock-model".into(),
        final_output: Some(output.into()),
        tool_calls: vec![],
        usage: Usage::default(),
    })
}

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
