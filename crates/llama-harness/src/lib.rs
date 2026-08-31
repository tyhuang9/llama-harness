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

#![deny(missing_docs)]

/// Scripted provider utilities intended for deterministic tests and examples.
pub mod mock {
    pub use llama_harness_core::mock::{
        error_response, final_response, tool_response, MockModelProvider, MockStep,
    };
}

pub use async_trait::async_trait;
pub use llama_harness_core::{
    load_agent_manifest, load_agent_manifest_path, AgentDefinition, AgentLimits, AgentManifest,
    AgentManifestError, AgentRunner, AgentRunnerBuilder, AllowAllPolicy, ApprovalHandler,
    ApprovalRecord, CancellationSafety, CatalogFingerprint, DenyApproval, EventRecord, EventSink,
    ExecutionLocation, ExecutionPlan, GenerationOptions, HarnessError, InMemoryEventSink,
    IssueSafety, JsonMap, Message, MessageRole, ModelCapabilities, ModelEventStream, ModelInfo,
    ModelProvider, ModelRequest, ModelResponse, ModelStreamController, ModelStreamEvent,
    ModelStreamFailureKind, NetworkEgress, PartialToolCall, PlanConcurrency, PlanNode,
    PolicyDecision, PolicyEngine, ProviderCapabilityLimits, ProviderHealth, ResultBinding,
    ResultRef, RunError, RunEvent, RunOverrides, RunRequest, RunResult, RunStatus, RunStrategy,
    SafeDefaultPolicy, SpeculationPolicy, Tool, ToolCall, ToolCallAssembler,
    ToolCallAssemblyLimits, ToolCallContext, ToolCallDelta, ToolCaller, ToolDefinition,
    ToolDiscoveryLimits, ToolDiscoveryMetadata, ToolExposure, ToolRegistry, ToolResult, ToolRisk,
    ToolScope, Usage, ValidatedModelStreamEvent, AGENT_MANIFEST_VERSION,
    CATALOG_FINGERPRINT_VERSION, MAX_EXECUTION_PLAN_BINDINGS, MAX_EXECUTION_PLAN_BYTES,
    MAX_EXECUTION_PLAN_EDGES, MAX_EXECUTION_PLAN_NODES, MAX_PLAN_ARGUMENT_BYTES,
    MAX_PLAN_ID_LENGTH, MAX_PLAN_JSON_DEPTH, MAX_PLAN_POINTER_LENGTH,
};
pub use serde_json;
pub use serde_json::Value as JsonValue;
pub use tokio_util::sync::CancellationToken;

#[cfg(feature = "ollama")]
/// Direct, loopback-only Ollama provider integration.
pub mod ollama {
    pub use llama_harness_ollama::*;
}

#[cfg(feature = "observability")]
/// Redacted local SQLite observability integration.
pub mod observability {
    pub use llama_harness_observability::*;
}

#[cfg(feature = "evals")]
/// Deterministic evaluation and regression contracts.
pub mod evals {
    pub use llama_harness_evals::*;
}

#[cfg(feature = "tauri")]
/// Embedded Tauri event, approval, cancellation, and path helpers.
pub mod tauri {
    pub use llama_harness_tauri::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facade_exports_manifest_and_tool_call_contracts() {
        let manifest: AgentManifest =
            load_agent_manifest(r#"{"version":1,"agents":[]}"#, Some("json")).unwrap();
        assert_eq!(manifest.version, AGENT_MANIFEST_VERSION);
        let _: Result<AgentManifest, AgentManifestError> = load_agent_manifest_path("missing.yaml");
        let context = ToolCallContext::new("run", "trace", "call", "tool");
        assert_eq!(context.run_id, "run");
    }
}
