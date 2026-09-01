//! Embedded, provider-neutral agent runtime for application-owned integrations.

#![deny(missing_docs)]

mod adaptive;
mod agent;
mod agent_manifest;
mod broker;
mod discovery;
mod error;
mod event;
mod limits;
mod message;
/// Deterministic mock provider and scripted response helpers.
pub mod mock;
mod model;
/// Declarative execution plan contracts and validation.
pub mod plan;
mod policy;
#[cfg(feature = "programmatic")]
mod programmatic;
mod runner;
mod speculation;
mod streaming;
mod tool;

pub use agent::{
    AgentDefinition, JsonMap, RunOverrides, RunRequest, RunResult, RunStatus, RunStrategy,
};
pub use agent_manifest::{
    load_agent_manifest, load_agent_manifest_path, AgentManifest, AgentManifestError,
    AGENT_MANIFEST_VERSION,
};
pub use discovery::{
    CatalogFingerprint, ToolDiscoveryLimits, ToolDiscoveryMetadata, ToolExposure,
    CATALOG_FINGERPRINT_VERSION,
};
pub use error::{HarnessError, ModelStreamFailureKind, RunError};
pub use event::{
    EventRecord, EventSink, InMemoryEventSink, PlanLifecycleOutcome, PlanNodeOutcome, PlanPhase,
    ProgramLifecycleOutcome, RunEvent, StrategyFallbackReason, StrategySelectionReason,
    ToolDiscoveryOutcome, ToolDiscoverySelection,
};
pub use limits::{
    AgentLimits, GenerationOptions, HARD_MAX_PROGRAMMATIC_FANOUT_CONCURRENCY,
    HARD_MAX_PROGRAMMATIC_PROGRAM_BYTES,
};
#[cfg(feature = "programmatic")]
pub use llama_harness_programmatic_sandbox as programmatic_sandbox;
pub use message::{Message, MessageRole};
pub use model::{
    ModelCapabilities, ModelInfo, ModelProvider, ModelRequest, ModelResponse, PreparedToolCatalog,
    ProgrammaticConformance, ProviderCapabilityLimits, ProviderHealth, Usage,
};
pub use plan::{
    ExecutionPlan, PlanConcurrency, PlanNode, ResultBinding, ResultRef,
    MAX_EXECUTION_PLAN_BINDINGS, MAX_EXECUTION_PLAN_BYTES, MAX_EXECUTION_PLAN_EDGES,
    MAX_EXECUTION_PLAN_NODES, MAX_PLAN_ARGUMENT_BYTES, MAX_PLAN_ID_LENGTH, MAX_PLAN_JSON_DEPTH,
    MAX_PLAN_POINTER_LENGTH,
};
pub use policy::{
    AllowAllPolicy, ApprovalHandler, ApprovalRecord, DenyApproval, PolicyDecision, PolicyEngine,
    SafeDefaultPolicy,
};
#[cfg(feature = "programmatic")]
pub use programmatic::ProgrammaticHostConfig;
pub use runner::{AgentRunner, AgentRunnerBuilder};
pub use speculation::{
    SpeculationConfig, SpeculationMetrics, SpeculationMode, SpeculationReadiness,
    HARD_MAX_SPECULATION_DURATION_MS, HARD_MAX_SPECULATION_STREAM_EVENTS,
    MIN_SPECULATION_SHADOW_OBSERVATIONS,
};
pub use streaming::{
    ModelEventStream, ModelStreamController, ModelStreamEvent, PartialToolCall, ToolCallAssembler,
    ToolCallAssemblyLimits, ToolCallDelta, ValidatedModelStreamEvent,
};
pub use tool::{
    CancellationSafety, ExecutionLocation, GroupToolRegistration, IssueSafety, NetworkEgress,
    SpeculationPolicy, Tool, ToolCall, ToolCallContext, ToolCaller, ToolDefinition,
    ToolRegistrationGroup, ToolRegistry, ToolResult, ToolRisk,
};
