//! Deterministic child sidecar used only by workspace SDK integration tests.
//!
//! It exercises the same `llama-harness-runtime` protocol bridge and canonical
//! `AgentRunner` as the production binary, while replacing the networked
//! provider with a fixed tool-call/final-output script.

use std::sync::Arc;

use llama_harness_core::{
    mock::{final_response, tool_response, MockModelProvider},
    HarnessError, ModelProvider, ToolCall,
};
use llama_harness_protocol::ProviderConfiguration;
use llama_harness_runtime::{serve_stdio_with_factory, ProviderFactory};

struct ScriptedProviderFactory;

impl ProviderFactory for ScriptedProviderFactory {
    fn create(&self, _: &ProviderConfiguration) -> Result<Arc<dyn ModelProvider>, HarnessError> {
        Ok(Arc::new(MockModelProvider::scripted([
            tool_response(ToolCall::new(
                "scripted-call-1",
                "notes.search",
                r#"{"query":"harness"}"#,
            )),
            final_response("scripted sidecar completed after host tool callback"),
        ])))
    }

    fn provider_names(&self) -> Vec<String> {
        vec!["scripted-test".into()]
    }
}

#[tokio::main]
async fn main() {
    if let Err(error) = serve_stdio_with_factory(Arc::new(ScriptedProviderFactory)).await {
        eprintln!("llama-harness scripted runtime failed: {error}");
        std::process::exit(1);
    }
}
