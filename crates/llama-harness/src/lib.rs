//! Public embedded API for llama-harness.
//!
//! This crate is the stable, deliberate entry point for Rust applications. The
//! canonical agent loop remains in Rust and is embedded in Rust and Tauri
//! applications; non-Rust SDK support is provided separately by a managed
//! child-process runtime.
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use llama_harness::{
//!     mock::{final_response, MockModelProvider}, AgentDefinition, AgentRunner, RunRequest,
//! };
//!
//! # async fn example() -> Result<(), llama_harness::HarnessError> {
//! let provider = Arc::new(MockModelProvider::scripted([final_response("done")]));
//! let runner = AgentRunner::builder(provider).build();
//! let result = runner
//!     .run(RunRequest::new(
//!         AgentDefinition::new("example", "Example", "1", "mock-model"),
//!         "Reply with done",
//!     ))
//!     .await?;
//! assert_eq!(result.final_output.as_deref(), Some("done"));
//! # Ok(())
//! # }
//! ```

/// Scripted provider utilities intended for deterministic tests and examples.
pub mod mock {
    pub use llama_harness_core::mock::{
        final_response, tool_response, MockModelProvider, MockStep,
    };
}

pub use async_trait::async_trait;
pub use llama_harness_core::{
    load_agent_manifest, load_agent_manifest_path, AgentDefinition, AgentLimits, AgentManifest,
    AgentManifestError, AgentRunner, AgentRunnerBuilder, AllowAllPolicy, ApprovalHandler,
    ApprovalRecord, DenyApproval, EventRecord, EventSink, GenerationOptions, HarnessError,
    InMemoryEventSink, JsonMap, Message, ModelCapabilities, ModelInfo, ModelProvider, ModelRequest,
    ModelResponse, PolicyDecision, PolicyEngine, ProviderHealth, RunError, RunEvent, RunOverrides,
    RunRequest, RunResult, RunStatus, SafeDefaultPolicy, Tool, ToolCall, ToolCallContext,
    ToolDefinition, ToolRegistry, ToolResult, ToolRisk, Usage, AGENT_MANIFEST_VERSION,
};
pub use tokio_util::sync::CancellationToken;

#[cfg(feature = "ollama")]
pub use llama_harness_ollama::{
    OllamaEventStream, OllamaProvider, OllamaProviderBuilder, OllamaStreamEvent,
    DEFAULT_OLLAMA_BASE_URL, DEFAULT_REQUEST_TIMEOUT,
};

#[cfg(feature = "observability")]
pub use llama_harness_observability::{
    AppendOutcome, ExportedRun, PersistedEvent, RedactionConfig, RetentionPolicy, RetentionResult,
    RunListQuery, RunSummary, SqliteEventSink, TraceStoreConfig, TraceStoreError, REDACTED_VALUE,
};

#[cfg(feature = "evals")]
pub use llama_harness_evals::{
    evaluate_expectations, evaluate_suite, export_regression_case, is_json_subset, load_suite,
    load_suite_path, replay_regression, AssertionFailure, EvalCase, EvalDefaults, EvalError,
    EvalExecutionRequest, EvalExecutor, EvalExpected, EvalFixture, EvalObservation, EvalSuite,
    EvaluationCaseResult, EvaluationReport, ExpectedFailure, ExpectedToolCall, RegressionCase,
    RegressionSource, SUPPORTED_SUITE_VERSION,
};

#[cfg(feature = "protocol")]
pub use llama_harness_protocol as protocol;

#[cfg(feature = "tauri")]
pub use llama_harness_tauri as tauri;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facade_exports_manifest_and_tool_call_contracts() {
        let manifest: AgentManifest =
            load_agent_manifest(r#"{"version":1,"agents":[]}"#, Some("json")).unwrap();
        assert_eq!(manifest.version, AGENT_MANIFEST_VERSION);
        let _: Result<AgentManifest, AgentManifestError> = load_agent_manifest_path("missing.yaml");
        let context = ToolCallContext {
            run_id: "run".into(),
            trace_id: "trace".into(),
            call_id: "call".into(),
            tool_id: "tool".into(),
        };
        assert_eq!(context.run_id, "run");
    }
}
