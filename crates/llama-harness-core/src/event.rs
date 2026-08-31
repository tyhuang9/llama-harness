use crate::{PolicyDecision, RunStatus, RunStrategy, ToolCaller};
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
/// An ordered event emitted during a run.
pub struct EventRecord {
    /// Identifier of the run that emitted the event.
    pub run_id: String,
    /// Core-generated unique execution identity, independent of an optional
    /// application-visible run ID.
    #[serde(default = "new_execution_id")]
    pub execution_id: String,
    /// Trace identifier associated with the run.
    pub trace_id: String,
    /// Monotonic within one run, starting at one.
    pub sequence: u64,
    /// Milliseconds since the Unix epoch. Never decreases within one emitter.
    pub timestamp_ms: u64,
    /// Event payload.
    pub event: RunEvent,
}

impl EventRecord {
    /// Creates one ordered event record for a run and trace.
    pub fn new(
        run_id: impl Into<String>,
        trace_id: impl Into<String>,
        sequence: u64,
        timestamp_ms: u64,
        event: RunEvent,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            execution_id: new_execution_id(),
            trace_id: trace_id.into(),
            sequence,
            timestamp_ms,
            event,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
/// Lifecycle event emitted by the runner.
pub enum RunEvent {
    /// A run has started.
    Started {
        /// Identifier of the started run.
        run_id: String,
        /// Trace identifier for the started run.
        trace_id: String,
    },
    /// The runner is requesting a model completion.
    ModelRequested {
        /// One-based model call number.
        call_number: u32,
        /// Model identifier being called.
        model: String,
    },
    /// A retryable provider error caused another model call to be scheduled.
    ModelRetrying {
        /// Number of the next model call.
        next_call_number: u32,
        /// Reason the previous call will be retried.
        reason: String,
    },
    /// The model returned a response.
    ModelResponded {
        /// Model call number that returned.
        call_number: u32,
    },
    /// A metadata-only tool catalog selection completed for one caller scope.
    ToolDiscoveryCompleted {
        /// Caller whose immutable scope was selected.
        caller: ToolCaller,
        /// Stable outcome of the completed selection attempt.
        #[serde(default)]
        outcome: ToolDiscoveryOutcome,
        /// Stable category describing how the scope was selected.
        #[serde(default)]
        selection: ToolDiscoverySelection,
        /// Number of allowed, caller-compatible catalog candidates.
        candidate_count: u32,
        /// Number of definitions selected for model exposure.
        selected_count: u32,
        /// Number of candidates explicitly configured as deferred.
        deferred_candidate_count: u32,
        /// Effective maximum number of tools for this caller scope.
        #[serde(default)]
        effective_tool_count_budget: u32,
        /// Effective maximum serialized tool-definition bytes for this scope.
        #[serde(default)]
        effective_schema_byte_budget: u64,
        /// Exact serialized definition-array byte count selected or required;
        /// a size scalar that never includes definition content.
        #[serde(default)]
        selected_schema_bytes: u64,
        /// Number of deferred candidates admitted by bounded expansion.
        #[serde(default)]
        expansion_count: u32,
        /// Configured maximum bounded expansion count.
        #[serde(default)]
        expansion_limit: u32,
        /// Whether the complete candidate catalog exceeded the effective budget.
        catalog_exceeded_budget: bool,
        /// Elapsed time spent selecting this caller scope.
        #[serde(default)]
        duration_ms: u64,
    },
    /// The runner selected an execution strategy using metadata-only inputs.
    StrategySelected {
        /// Strategy requested by the host.
        requested: RunStrategy,
        /// Strategy selected for execution.
        selected: RunStrategy,
        /// Stable reason for the selection.
        reason: StrategySelectionReason,
    },
    /// The runner fell back from one strategy to another.
    StrategyFallback {
        /// Strategy that could not continue.
        from: RunStrategy,
        /// Safe fallback strategy.
        to: RunStrategy,
        /// Stable reason for the fallback.
        reason: StrategyFallbackReason,
    },
    /// A metadata-only declarative planning phase changed lifecycle state.
    PlanLifecycle {
        /// Stable phase whose lifecycle changed.
        phase: PlanPhase,
        /// One-based attempt within the phase.
        attempt: u32,
        /// Stable value-free lifecycle outcome.
        outcome: PlanLifecycleOutcome,
    },
    /// A complete declarative plan passed structural and execution preflight.
    PlanValidated {
        /// One-based execution-plan attempt.
        attempt: u32,
        /// Number of validated plan nodes.
        node_count: u32,
    },
    /// A metadata-only program generation or repair attempt changed lifecycle state.
    ProgramLifecycle {
        /// One-based program generation or repair attempt.
        attempt: u32,
        /// Stable value-free lifecycle outcome.
        outcome: ProgramLifecycleOutcome,
    },
    /// A strict program AST and its private bytecode completed validation.
    ProgramValidated {
        /// One-based successfully validated program attempt.
        attempt: u32,
        /// Number of source AST statements admitted to compilation.
        statement_count: u32,
        /// Number of private verified bytecode instructions.
        instruction_count: u32,
    },
    /// A nonterminal program VM completion before final model synthesis.
    ProgramExecutionCompleted {
        /// One-based successfully executed program attempt.
        attempt: u32,
        /// Deterministic fuel charged by the VM.
        fuel_used: u64,
        /// Executed branch decisions.
        branches: u64,
        /// Entered bounded loop iterations.
        loop_iterations: u64,
        /// Read-only fan-out batches yielded.
        fanout_batches: u32,
        /// Failed or invalid broker results observed before this terminal execution state.
        partial_failures: u32,
        /// Peak conservatively accounted VM bytes.
        peak_accounted_bytes: u64,
        /// VM execution duration in milliseconds.
        duration_ms: u64,
    },
    /// A declarative plan node started execution.
    PlanNodeStarted {
        /// Runner-generated opaque node identifier; never model-provided text.
        node_id: String,
        /// Registered tool identifier.
        tool_id: String,
        /// One-based execution-plan attempt.
        attempt: u32,
        /// One-based deterministic execution wave.
        wave: u32,
    },
    /// A declarative plan node completed execution.
    PlanNodeCompleted {
        /// Runner-generated opaque node identifier; never model-provided text.
        node_id: String,
        /// Registered tool identifier.
        tool_id: String,
        /// One-based execution-plan attempt.
        attempt: u32,
        /// One-based deterministic execution wave.
        wave: u32,
        /// Whether the node returned a successful validated result.
        ok: bool,
        /// Stable metadata-only node outcome.
        outcome: PlanNodeOutcome,
        /// Elapsed node execution time in milliseconds.
        duration_ms: u64,
    },
    /// A previously committed effect was reused instead of being invoked again.
    ToolEffectReused {
        /// Identifier of the call receiving the recorded result.
        call_id: String,
        /// Registered tool identifier.
        tool_id: String,
    },
    /// Aggregate metadata-only strategy usage for the completed run.
    StrategyUsage {
        /// Strategy that performed the run's final execution path.
        strategy: RunStrategy,
        /// Number of provider calls issued.
        model_calls: u32,
        /// Provider calls used for initial plan selection and validation.
        planning_model_calls: u32,
        /// Provider calls used for the single optional invalid-plan repair.
        repair_model_calls: u32,
        /// Provider calls used for the single optional execution recovery.
        recovery_model_calls: u32,
        /// Provider calls used to synthesize a final answer from completed plan results.
        final_synthesis_model_calls: u32,
        /// Provider calls used by direct reactive execution.
        reactive_model_calls: u32,
        /// Number of tool proposals admitted to the broker attempt budget.
        tool_calls: u32,
        /// Admitted proposals that crossed into the execution boundary.
        tool_issued: u32,
        /// Admitted proposals satisfied from an exact completed-effect record.
        tool_reused: u32,
        /// Admitted proposals rejected by validation, policy, approval, or limits.
        tool_rejected: u32,
        /// Admitted proposals aborted by an error before execution was issued.
        tool_pre_dispatch_aborted: u32,
        /// Issued calls that returned a successful validated result.
        tool_completed: u32,
        /// Issued calls that failed execution or result validation.
        tool_failed: u32,
        /// Issued calls cancelled or timed out before a result was available.
        tool_cancelled: u32,
        /// Elapsed run duration in milliseconds.
        duration_ms: u64,
    },
    /// A tool call was rejected before execution.
    ToolRejected {
        /// Identifier of the rejected call.
        call_id: String,
        /// Identifier of the rejected tool.
        tool_id: String,
        /// Reason for rejection.
        reason: String,
    },
    /// A policy decision was recorded for a tool call.
    PolicyDecided {
        /// Identifier of the related tool call.
        call_id: String,
        /// Policy outcome.
        decision: PolicyDecision,
    },
    /// Approval was requested for a tool call.
    ApprovalRequested {
        /// Identifier of the tool call awaiting approval.
        call_id: String,
        /// Identifier of the tool requiring approval.
        tool_id: String,
    },
    /// A tool call finished execution.
    ToolCompleted {
        /// Identifier of the completed call.
        call_id: String,
        /// Identifier of the completed tool.
        tool_id: String,
        /// Whether the tool returned a successful result.
        ok: bool,
    },
    /// The run reached a terminal status.
    Completed {
        /// Terminal run status.
        status: RunStatus,
    },
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Stable, value-free outcome of a completed tool-discovery selection.
pub enum ToolDiscoveryOutcome {
    /// A usable scope, including an intentionally empty scope, was selected.
    #[default]
    Selected,
    /// Mandatory tools could not fit the effective count or schema-byte budget.
    LimitReached,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Stable, value-free category for a completed tool-discovery selection.
pub enum ToolDiscoverySelection {
    /// An event written by an earlier release did not record a selection category.
    #[default]
    LegacyUnclassified,
    /// The authorized caller catalog was empty.
    EmptyCatalog,
    /// The provider advertised insufficient capacity for any tool scope.
    NoCapacity,
    /// The complete authorized catalog fit the effective budget.
    FullCatalog,
    /// Only mandatory hot tools were selected.
    HotOnly,
    /// An exact identifier, name, namespace, or alias selected tools.
    Exact,
    /// A high-margin lexical match selected one deferred tool.
    LexicalConfident,
    /// A low-margin lexical match selected a bounded expansion.
    LexicalExpanded,
    /// No deferred candidate matched the query.
    NoMatch,
    /// Mandatory tools exceeded the effective tool-count budget.
    CountLimit,
    /// Mandatory tools exceeded the effective serialized-schema-byte budget.
    SchemaByteLimit,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Stable, value-free reason for selecting an execution strategy.
pub enum StrategySelectionReason {
    /// The host explicitly forced the selected strategy.
    Forced,
    /// Provider capabilities permit adaptive declarative planning.
    AdaptivePlanner,
    /// The provider-directed planner selected direct reactive execution.
    PlannerSelectedDirect,
    /// The provider-directed planner selected a declarative plan.
    PlannerSelectedPlan,
    /// Provider capabilities require the direct compatibility path.
    CapabilityDowngrade,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Stable, value-free reason for a strategy fallback.
pub enum StrategyFallbackReason {
    /// The provider does not support structured plan generation.
    UnsupportedCapability,
    /// A generated plan remained invalid after one repair attempt.
    InvalidPlan,
    /// A generated program remained invalid after its single repair attempt.
    InvalidProgram,
    /// Execution failed after some node results were safely recorded.
    ExecutionRecovery,
    /// Planner execution failed before any tool effect began.
    PlannerFailure,
}

fn new_execution_id() -> String {
    Uuid::new_v4().to_string()
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Stable value-free lifecycle outcome for a program attempt.
pub enum ProgramLifecycleOutcome {
    /// Program generation or repair started.
    Started,
    /// Strict parsing, compilation, and private-bytecode verification succeeded.
    Validated,
    /// The provider output failed strict parsing or verification.
    Invalid,
    /// Final synthesis completed after VM execution and produced the terminal answer.
    Succeeded,
    /// The attempt was abandoned for the approved direct fallback before an effect.
    Fallback,
    /// The attempt failed after validation.
    Failed,
    /// The attempt was cancelled.
    Cancelled,
    /// The attempt exceeded its deadline.
    TimedOut,
    /// The attempt exhausted a configured resource limit.
    LimitReached,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Stable declarative planning and recovery phase.
pub enum PlanPhase {
    /// Initial strategy-envelope generation and validation.
    Planning,
    /// Bounded repair of an invalid strategy envelope.
    Repair,
    /// Structural and schema validation of a generated strategy envelope.
    Validation,
    /// Structural, broker, policy, and approval preflight.
    Preflight,
    /// Bounded recovery after safely recorded execution.
    Recovery,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Stable value-free outcome for a declarative planning phase.
pub enum PlanLifecycleOutcome {
    /// The phase began.
    Started,
    /// The phase completed successfully.
    Succeeded,
    /// Generated structured content was invalid.
    Invalid,
    /// Validation, policy, approval, or another safety gate rejected the phase.
    Rejected,
    /// The phase failed for another reason.
    Failed,
    /// The phase was cancelled.
    Cancelled,
    /// The phase exceeded its deadline.
    TimedOut,
    /// The phase exhausted a configured resource or call limit.
    LimitReached,
    /// The optional phase was skipped to preserve final-synthesis capacity.
    Skipped,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Stable value-free outcome for one declarative plan node.
pub enum PlanNodeOutcome {
    /// The tool returned a successful validated result.
    Succeeded,
    /// The tool or scheduler failed the node.
    Failed,
    /// The node was cancelled.
    Cancelled,
    /// The node exceeded its deadline.
    TimedOut,
    /// Validation, policy, approval, or another safety gate rejected the node.
    Rejected,
    /// The node exhausted a configured resource or call limit.
    LimitReached,
    /// An exact previously completed effect supplied the node result.
    Reused,
}

/// Receives ordered events from an agent runner.
pub trait EventSink: Send + Sync {
    /// Records one emitted event.
    fn emit(&self, record: EventRecord);
}

#[derive(Default)]
/// Thread-safe in-memory event collector.
pub struct InMemoryEventSink {
    events: Mutex<Vec<EventRecord>>,
}

impl InMemoryEventSink {
    /// Returns a snapshot of all events collected so far.
    pub fn events(&self) -> Vec<EventRecord> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl EventSink for InMemoryEventSink {
    fn emit(&self, record: EventRecord) {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(record);
    }
}

pub(crate) struct EventEmitter {
    run_id: String,
    execution_id: String,
    trace_id: String,
    sequence: u64,
    last_timestamp_ms: u64,
    sink: Arc<dyn EventSink>,
}

impl EventEmitter {
    pub(crate) fn new(run_id: String, trace_id: String, sink: Arc<dyn EventSink>) -> Self {
        Self {
            run_id,
            execution_id: new_execution_id(),
            trace_id,
            sequence: 0,
            last_timestamp_ms: 0,
            sink,
        }
    }

    pub(crate) fn emit(&mut self, event: RunEvent) {
        self.sequence = self.sequence.saturating_add(1);
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis() as u64)
            .max(self.last_timestamp_ms);
        self.last_timestamp_ms = timestamp_ms;
        self.sink.emit(EventRecord {
            run_id: self.run_id.clone(),
            execution_id: self.execution_id.clone(),
            trace_id: self.trace_id.clone(),
            sequence: self.sequence,
            timestamp_ms,
            event,
        });
    }
}
