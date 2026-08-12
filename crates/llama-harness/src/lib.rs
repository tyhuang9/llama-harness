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

pub use llama_harness_core::{
    AgentDefinition, AgentLimits, AgentRunner, AgentRunnerBuilder, AllowAllPolicy, ApprovalHandler,
    ApprovalRecord, DenyApproval, EventRecord, EventSink, GenerationOptions, HarnessError,
    InMemoryEventSink, JsonMap, Message, ModelCapabilities, ModelInfo, ModelProvider, ModelRequest,
    ModelResponse, PolicyDecision, PolicyEngine, ProviderHealth, RunError, RunEvent, RunOverrides,
    RunRequest, RunResult, RunStatus, SafeDefaultPolicy, Tool, ToolCall, ToolDefinition,
    ToolRegistry, ToolResult, ToolRisk, Usage,
};

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
