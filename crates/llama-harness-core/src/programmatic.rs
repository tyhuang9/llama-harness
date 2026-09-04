use crate::{
    broker::{BrokerState, PrepareOutcome, PreparedCall, ToolBroker},
    discovery::{ToolDiscoveryStats, ToolScope, ToolScopeSelection},
    event::{EventEmitter, ProgramLifecycleOutcome, RunEvent, StrategySelectionReason},
    runner::{
        agent_structured_output, apply_terminal_error, await_guarded, check_stopped,
        emit_discovery, ensure_transcript, initial_messages, merge_generation, provider_deadline,
        validate_model_response, validate_output, DirectContinuation, DirectStrategyEvents,
        ProgrammaticUsage, RunPreflight,
    },
    AgentRunner, HarnessError, Message, ModelCapabilities, ModelRequest, ModelResponse,
    ProgrammaticConformance, RunRequest, RunResult, RunStatus, RunStrategy, StrategyFallbackReason,
    StructuredOutputRequest, ToolCall, ToolCallContext, ToolCaller, ToolResult,
};
use futures_util::future::join_all;
use llama_harness_programmatic_sandbox::{
    Execution, ExecutionId, Program, SandboxErrorCode, SandboxLimits, StepOutcome, ToolBatch,
    ToolResponse, HARD_LIMITS,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    io::{self, Write},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant as StdInstant},
};
use tokio::time::Instant;
use uuid::Uuid;

const DEFAULT_PROGRAMMATIC_DURATION_MS: u64 = 60_000;
const HARD_PROGRAMMATIC_DURATION_MS: u64 = 300_000;
const DEFAULT_VM_ADMISSION: usize = 4;
const HARD_VM_ADMISSION: usize = 16;
const MAX_FANOUT_CONCURRENCY: usize = 8;

/// A bounded workload shape that a host may allow Adaptive planning to promote
/// to deterministic Programmatic execution.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProgrammaticWorkloadClass {
    /// Bounded iteration over a collection or repeated operation.
    Loop,
    /// Bounded parallel dispatch of independent read-only work.
    FanOut,
    /// Bounded selection of values from an intermediate collection.
    Filter,
    /// Bounded reduction or aggregation of intermediate values.
    Aggregation,
    /// Work whose bounded intermediate representation is materially larger
    /// than is practical to shuttle through a native tool-calling transcript.
    LargeIntermediateData,
}

impl ProgrammaticWorkloadClass {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Loop => "loop",
            Self::FanOut => "fan_out",
            Self::Filter => "filter",
            Self::Aggregation => "aggregation",
            Self::LargeIntermediateData => "large_intermediate_data",
        }
    }
}

pub(crate) struct AdaptiveProgrammaticPrepared {
    pub(crate) scope: ToolScope,
    pub(crate) discovery: ToolDiscoveryStats,
    capabilities: ModelCapabilities,
    limits: SandboxLimits,
    structured_output: Option<StructuredOutputRequest>,
    pub(crate) deadline: Option<Instant>,
    admission: Option<tokio::sync::OwnedSemaphorePermit>,
}

pub(crate) enum AdaptiveProgrammaticReadiness {
    Ready(Box<AdaptiveProgrammaticPrepared>),
    Fallback(Option<ToolDiscoveryStats>),
}

pub(crate) struct ProgrammaticContinuation {
    pub(crate) result: RunResult,
    pub(crate) events: EventEmitter,
    pub(crate) output_validator: Option<jsonschema::Validator>,
    pub(crate) deadline: Option<Instant>,
    pub(crate) started: StdInstant,
    pub(crate) model: String,
    pub(crate) model_calls: u32,
    pub(crate) planning_calls: u32,
    pub(crate) repair_calls: u32,
    pub(crate) synthesis_calls: u32,
    pub(crate) broker_state: BrokerState,
    pub(crate) prepared_direct_scope: Option<ToolScope>,
}

// The sandbox only needs a process-local opaque resume scope. Public broker
// occurrence and effect identities use the full UUID below; this counter is
// deliberately never used as an externally visible execution identity.
static NEXT_SANDBOX_EXECUTION_ID: AtomicU64 = AtomicU64::new(1);

const PROGRAM_PROMPT: &str = r#"Return only a strict llama-harness program JSON object. Use version 1 and a body array. Statements are let, branch, for_each, map, filter, reduce, invoke, fan_out, and return. Expressions are null, boolean, integer, string, variable, path, array, object, binary, and unary. Tool IDs in invoke/fan_out must be static IDs from the supplied catalog. All loops and collections require explicit bounded limits. Do not use markdown, floats, dynamic tool names, code strings, imports, functions, recursion, mutation, reflection, exceptions, regex, or prose."#;

const REPAIR_PROMPT: &str = "The previous program failed strict structural verification. Return one corrected version-1 program JSON object only. Do not add markdown or prose.";

const SYNTHESIS_PROMPT: &str = "A verified program completed. Produce the final answer using only the bounded program return and value-free broker call summary in the next user message.";

fn program_schema(limits: &SandboxLimits) -> Value {
    serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["version", "body"],
        "properties": {
            "version": {"const": 1},
            "body": {"$ref": "#/$defs/body"}
        },
        "additionalProperties": false,
        "$defs": {
            "body": {
                "type": "array",
                "maxItems": limits.max_ast_nodes,
                "items": {"$ref": "#/$defs/statement"}
            },
            "statement": {
                "oneOf": [
                    {
                        "type": "object",
                        "required": ["kind", "name", "value"],
                        "properties": {
                            "kind": {"const": "let"},
                            "name": {"type": "string", "minLength": 1},
                            "value": {"$ref": "#/$defs/expression"}
                        },
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "required": ["kind", "condition", "then_body"],
                        "properties": {
                            "kind": {"const": "branch"},
                            "condition": {"$ref": "#/$defs/expression"},
                            "then_body": {"$ref": "#/$defs/body"},
                            "else_body": {"$ref": "#/$defs/body"}
                        },
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "required": ["kind", "item", "collection", "max_iterations", "body"],
                        "properties": {
                            "kind": {"const": "for_each"},
                            "item": {"type": "string", "minLength": 1},
                            "collection": {"$ref": "#/$defs/expression"},
                            "max_iterations": {"type": "integer", "minimum": 1, "maximum": limits.max_loop_iterations},
                            "body": {"$ref": "#/$defs/body"}
                        },
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "required": ["kind", "name", "item", "collection", "max_items", "value"],
                        "properties": {
                            "kind": {"const": "map"},
                            "name": {"type": "string", "minLength": 1},
                            "item": {"type": "string", "minLength": 1},
                            "collection": {"$ref": "#/$defs/expression"},
                            "max_items": {"type": "integer", "minimum": 1, "maximum": limits.max_collection_items},
                            "value": {"$ref": "#/$defs/expression"}
                        },
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "required": ["kind", "name", "item", "collection", "max_items", "predicate"],
                        "properties": {
                            "kind": {"const": "filter"},
                            "name": {"type": "string", "minLength": 1},
                            "item": {"type": "string", "minLength": 1},
                            "collection": {"$ref": "#/$defs/expression"},
                            "max_items": {"type": "integer", "minimum": 1, "maximum": limits.max_collection_items},
                            "predicate": {"$ref": "#/$defs/expression"}
                        },
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "required": ["kind", "name", "item", "accumulator", "collection", "max_items", "initial", "value"],
                        "properties": {
                            "kind": {"const": "reduce"},
                            "name": {"type": "string", "minLength": 1},
                            "item": {"type": "string", "minLength": 1},
                            "accumulator": {"type": "string", "minLength": 1},
                            "collection": {"$ref": "#/$defs/expression"},
                            "max_items": {"type": "integer", "minimum": 1, "maximum": limits.max_collection_items},
                            "initial": {"$ref": "#/$defs/expression"},
                            "value": {"$ref": "#/$defs/expression"}
                        },
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "required": ["kind", "name", "tool_id", "arguments"],
                        "properties": {
                            "kind": {"const": "invoke"},
                            "name": {"type": "string", "minLength": 1},
                            "tool_id": {"type": "string", "minLength": 1},
                            "arguments": {"$ref": "#/$defs/expression"}
                        },
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "required": ["kind", "name", "tool_id", "item", "collection", "max_calls", "arguments"],
                        "properties": {
                            "kind": {"const": "fan_out"},
                            "name": {"type": "string", "minLength": 1},
                            "tool_id": {"type": "string", "minLength": 1},
                            "item": {"type": "string", "minLength": 1},
                            "collection": {"$ref": "#/$defs/expression"},
                            "max_calls": {"type": "integer", "minimum": 1, "maximum": limits.max_fanout},
                            "arguments": {"$ref": "#/$defs/expression"}
                        },
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "required": ["kind", "value"],
                        "properties": {
                            "kind": {"const": "return"},
                            "value": {"$ref": "#/$defs/expression"}
                        },
                        "additionalProperties": false
                    }
                ]
            },
            "expression": {
                "oneOf": [
                    {"type": "object", "required": ["kind"], "properties": {"kind": {"const": "null"}}, "additionalProperties": false},
                    {"type": "object", "required": ["kind", "value"], "properties": {"kind": {"const": "boolean"}, "value": {"type": "boolean"}}, "additionalProperties": false},
                    {"type": "object", "required": ["kind", "value"], "properties": {"kind": {"const": "integer"}, "value": {"type": "integer"}}, "additionalProperties": false},
                    {"type": "object", "required": ["kind", "value"], "properties": {"kind": {"const": "string"}, "value": {"type": "string"}}, "additionalProperties": false},
                    {"type": "object", "required": ["kind", "name"], "properties": {"kind": {"const": "variable"}, "name": {"type": "string", "minLength": 1}}, "additionalProperties": false},
                    {"type": "object", "required": ["kind", "value", "pointer"], "properties": {"kind": {"const": "path"}, "value": {"$ref": "#/$defs/expression"}, "pointer": {"type": "string"}}, "additionalProperties": false},
                    {"type": "object", "required": ["kind", "items"], "properties": {"kind": {"const": "array"}, "items": {"type": "array", "maxItems": limits.max_collection_items, "items": {"$ref": "#/$defs/expression"}}}, "additionalProperties": false},
                    {"type": "object", "required": ["kind", "entries"], "properties": {"kind": {"const": "object"}, "entries": {"type": "array", "maxItems": limits.max_collection_items, "items": {"$ref": "#/$defs/object_entry"}}}, "additionalProperties": false},
                    {"type": "object", "required": ["kind", "operator", "left", "right"], "properties": {"kind": {"const": "binary"}, "operator": {"enum": ["add", "subtract", "multiply", "divide", "remainder", "equal", "not_equal", "less_than", "less_than_or_equal", "greater_than", "greater_than_or_equal", "and", "or"]}, "left": {"$ref": "#/$defs/expression"}, "right": {"$ref": "#/$defs/expression"}}, "additionalProperties": false},
                    {"type": "object", "required": ["kind", "operator", "value"], "properties": {"kind": {"const": "unary"}, "operator": {"enum": ["not", "negate", "count", "sum", "all", "any"]}, "value": {"$ref": "#/$defs/expression"}}, "additionalProperties": false}
                ]
            },
            "object_entry": {
                "type": "object",
                "required": ["key", "value"],
                "properties": {
                    "key": {"type": "string"},
                    "value": {"$ref": "#/$defs/expression"}
                },
                "additionalProperties": false
            }
        }
    })
}

fn program_structured_output(
    capabilities: &ModelCapabilities,
    limits: &SandboxLimits,
) -> Result<Option<StructuredOutputRequest>, HarnessError> {
    if !capabilities.supports_structured_output {
        return Ok(None);
    }
    StructuredOutputRequest::new("llama_harness_program_ast_v1", program_schema(limits), true)
        .map(Some)
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct ProgrammaticBrokerSummary {
    total: u64,
    succeeded: u64,
    failed: u64,
}

#[derive(Serialize)]
struct ProgrammaticSynthesisInput<'a> {
    program_return: &'a Value,
    broker_calls: ProgrammaticBrokerSummary,
}

/// Bounded, canonical program return and value-free audit summary passed to
/// the final synthesis call.
///
/// Raw tool arguments and results remain inside the broker/sandbox execution
/// boundary. Accounting reserves the maximum program return plus bounded call
/// counters before the broker can issue an effect, so large intermediate data
/// is reduced by the program instead of being re-injected into the model.
struct ProgrammaticTranscript {
    summary: ProgrammaticBrokerSummary,
    initial_message_bytes: u64,
    limit: u64,
    maximum_program_return_bytes: u64,
}

impl ProgrammaticTranscript {
    fn new(request: &RunRequest, limits: &SandboxLimits) -> Result<Self, HarnessError> {
        let initial_message_bytes = initial_messages(request)
            .iter()
            .map(Message::transcript_bytes)
            .sum();
        let transcript = Self {
            summary: ProgrammaticBrokerSummary::default(),
            initial_message_bytes,
            limit: request.agent.limits.max_transcript_bytes,
            maximum_program_return_bytes: u64::try_from(limits.max_output_bytes).map_err(|_| {
                HarnessError::ResourceLimit(
                    "program return byte limit does not fit accounting".into(),
                )
            })?,
        };
        transcript.ensure_summary(transcript.summary, transcript.maximum_program_return_bytes)?;
        Ok(transcript)
    }

    fn reserve_batch(&self, batch: &ToolBatch) -> Result<(), HarnessError> {
        let additional = u64::try_from(batch.calls().len()).map_err(|_| {
            HarnessError::ResourceLimit("programmatic batch length does not fit accounting".into())
        })?;
        let total = self.summary.total.checked_add(additional).ok_or_else(|| {
            HarnessError::ResourceLimit("programmatic broker call count overflowed".into())
        })?;
        // Either outcome counter can grow to the projected total. Reserving
        // both maxima is conservative by only a few decimal digits and keeps
        // post-effect synthesis admission impossible.
        self.ensure_summary(
            ProgrammaticBrokerSummary {
                total,
                succeeded: total,
                failed: total,
            },
            self.maximum_program_return_bytes,
        )
    }

    fn record(&mut self, ok: bool) -> Result<(), HarnessError> {
        let mut next = self.summary;
        next.total = next.total.checked_add(1).ok_or_else(|| {
            HarnessError::ResourceLimit("programmatic broker call count overflowed".into())
        })?;
        let outcome = if ok {
            &mut next.succeeded
        } else {
            &mut next.failed
        };
        *outcome = outcome.checked_add(1).ok_or_else(|| {
            HarnessError::ResourceLimit("programmatic broker outcome count overflowed".into())
        })?;
        self.ensure_summary(next, self.maximum_program_return_bytes)?;
        self.summary = next;
        Ok(())
    }

    fn synthesis_payload_capacity(&self, program_return: &Value) -> Result<usize, HarnessError> {
        let program_return_bytes = count_json_bytes(program_return)?;
        self.ensure_summary(self.summary, program_return_bytes)?;
        let payload_bytes = synthesis_payload_bytes(program_return_bytes, self.summary)?;
        usize::try_from(payload_bytes).map_err(|_| {
            HarnessError::ResourceLimit("programmatic synthesis payload does not fit memory".into())
        })
    }

    fn ensure_summary(
        &self,
        summary: ProgrammaticBrokerSummary,
        program_return_bytes: u64,
    ) -> Result<(), HarnessError> {
        let payload_bytes = synthesis_payload_bytes(program_return_bytes, summary)?;
        let projected = checked_transcript_sum([
            self.initial_message_bytes,
            SYNTHESIS_PROMPT.len() as u64,
            payload_bytes,
        ])?;
        if projected > self.limit {
            return Err(HarnessError::ResourceLimit(format!(
                "programmatic synthesis transcript exceeds {} bytes",
                self.limit
            )));
        }
        Ok(())
    }
}

fn synthesis_payload_bytes(
    program_return_bytes: u64,
    summary: ProgrammaticBrokerSummary,
) -> Result<u64, HarnessError> {
    // {"program_return":<value>,"broker_calls":<summary>}. The fixed
    // punctuation and member names contribute 35 bytes.
    checked_transcript_sum([35, program_return_bytes, count_json_bytes(&summary)?])
}

fn checked_transcript_sum(values: impl IntoIterator<Item = u64>) -> Result<u64, HarnessError> {
    values.into_iter().try_fold(0u64, |total, value| {
        total.checked_add(value).ok_or_else(|| {
            HarnessError::ResourceLimit("programmatic transcript accounting overflowed".into())
        })
    })
}

struct JsonByteCounter(u64);

impl Write for JsonByteCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(
                u64::try_from(bytes.len()).map_err(|_| io::Error::other("length overflow"))?,
            )
            .ok_or_else(|| io::Error::other("length overflow"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn count_json_bytes(value: &impl Serialize) -> Result<u64, HarnessError> {
    let mut counter = JsonByteCounter(0);
    serde_json::to_writer(&mut counter, value).map_err(|_| {
        HarnessError::InvalidOutput("programmatic transcript could not be serialized".into())
    })?;
    Ok(counter.0)
}

fn serialize_synthesis_input(
    program_return: &Value,
    transcript: &ProgrammaticTranscript,
) -> Result<String, HarnessError> {
    let capacity = transcript.synthesis_payload_capacity(program_return)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|_| {
        HarnessError::ResourceLimit("programmatic synthesis allocation failed".into())
    })?;
    serde_json::to_writer(
        &mut bytes,
        &ProgrammaticSynthesisInput {
            program_return,
            broker_calls: transcript.summary,
        },
    )
    .map_err(|_| {
        HarnessError::InvalidOutput("program synthesis input could not be serialized".into())
    })?;
    if bytes.len() != capacity {
        return Err(HarnessError::InvalidOutput(
            "program synthesis accounting did not match serialization".into(),
        ));
    }
    String::from_utf8(bytes)
        .map_err(|_| HarnessError::InvalidOutput("program synthesis input was not UTF-8".into()))
}

/// Explicit host opt-in and resource bounds for programmatic execution.
#[derive(Clone, Debug)]
pub struct ProgrammaticHostConfig {
    /// Sandbox limits further constrained by immutable library and provider caps.
    pub limits: SandboxLimits,
    /// Finite programmatic run deadline in milliseconds.
    pub max_duration_ms: u64,
    /// Maximum concurrently admitted Programmatic runs for this runner.
    ///
    /// Admission is nonblocking and begins before a candidate program is
    /// requested, parsed, or compiled. When every slot is occupied, the run
    /// fails immediately with a resource limit rather than queueing. A held
    /// permit remains through program/model buffers, VM state, canonical tool
    /// results retained by the VM, the bounded synthesis summary, final
    /// synthesis, and output validation. This conservatively bounds concurrent
    /// Programmatic-run retained memory rather than only VM live-byte accounting.
    pub max_active_vms: usize,
    /// Maximum concurrent read-only, parallel-safe calls in a fan-out batch.
    pub max_fanout_concurrency: usize,
}

impl Default for ProgrammaticHostConfig {
    fn default() -> Self {
        Self {
            limits: SandboxLimits::default(),
            max_duration_ms: DEFAULT_PROGRAMMATIC_DURATION_MS,
            max_active_vms: DEFAULT_VM_ADMISSION,
            max_fanout_concurrency: MAX_FANOUT_CONCURRENCY,
        }
    }
}

impl ProgrammaticHostConfig {
    fn validate(&self) -> Result<(), HarnessError> {
        self.limits.validate().map_err(|_| {
            HarnessError::InvalidRequest("invalid programmatic sandbox limits".into())
        })?;
        if self.max_duration_ms == 0 || self.max_duration_ms > HARD_PROGRAMMATIC_DURATION_MS {
            return Err(HarnessError::InvalidRequest(
                "programmatic duration must be within 1..=300000 milliseconds".into(),
            ));
        }
        if self.max_active_vms == 0 || self.max_active_vms > HARD_VM_ADMISSION {
            return Err(HarnessError::InvalidRequest(
                "programmatic run admission must be within 1..=16".into(),
            ));
        }
        if self.max_fanout_concurrency == 0 || self.max_fanout_concurrency > MAX_FANOUT_CONCURRENCY
        {
            return Err(HarnessError::InvalidRequest(
                "programmatic fan-out concurrency must be within 1..=8".into(),
            ));
        }
        Ok(())
    }
}

fn programmatic_deadline(
    config: &ProgrammaticHostConfig,
    run_deadline: Option<Instant>,
) -> Result<Option<Instant>, HarnessError> {
    let configured_deadline = Instant::now()
        .checked_add(Duration::from_millis(config.max_duration_ms))
        .ok_or_else(|| HarnessError::InvalidRequest("programmatic duration is too large".into()))?;
    Ok(Some(run_deadline.map_or(configured_deadline, |run| {
        run.min(configured_deadline)
    })))
}

fn effective_programmatic_limits(
    config: &ProgrammaticHostConfig,
    request: &RunRequest,
    provider_program_bytes: u64,
) -> Result<SandboxLimits, HarnessError> {
    let mut limits = config.limits.constrained_by(HARD_LIMITS);
    limits.max_program_bytes = limits
        .max_program_bytes
        .min(request.agent.limits.max_programmatic_program_bytes as usize)
        .min(usize::try_from(provider_program_bytes).unwrap_or(usize::MAX));
    limits.validate().map_err(|_| {
        HarnessError::UnsupportedCapability(
            "effective provider program byte limit is invalid".into(),
        )
    })?;
    Ok(limits)
}

impl AgentRunner {
    pub(crate) fn prepare_adaptive_programmatic(
        &self,
        request: &RunRequest,
        workload_class: ProgrammaticWorkloadClass,
        model_calls: u32,
        run_deadline: Option<Instant>,
    ) -> Result<AdaptiveProgrammaticReadiness, HarnessError> {
        // Re-prove promotion authority at the execution boundary. The planner
        // caller performs the same check for an early downgrade, but admission
        // must not depend on one adjacent call site remaining correct.
        if !self
            .adaptive_programmatic_allowlist
            .contains(&workload_class)
        {
            return Ok(AdaptiveProgrammaticReadiness::Fallback(None));
        }
        let Some(config) = self.programmatic.as_ref() else {
            return Ok(AdaptiveProgrammaticReadiness::Fallback(None));
        };
        // An explicitly supplied but invalid host configuration is a terminal
        // host error, not a capability downgrade.
        config.validate()?;
        let capabilities = self.provider.capabilities();
        if !capabilities.supports_tools
            || !capabilities.supports_programmatic_calling
            || capabilities.programmatic_conformance
                != Some(ProgrammaticConformance::StrictJsonAstV1)
        {
            return Ok(AdaptiveProgrammaticReadiness::Fallback(None));
        }
        let Some(provider_program_bytes) = capabilities
            .limits
            .max_program_bytes
            .filter(|bytes| *bytes > 0)
        else {
            return Ok(AdaptiveProgrammaticReadiness::Fallback(None));
        };
        if request
            .agent
            .limits
            .max_model_calls
            .saturating_sub(model_calls)
            < 2
        {
            return Ok(AdaptiveProgrammaticReadiness::Fallback(None));
        }

        let deadline = programmatic_deadline(config, run_deadline)?;
        check_stopped(
            &request.cancellation,
            deadline,
            "programmatic run deadline reached",
        )?;
        // Admission is deliberately acquired before discovery or any model
        // work. Saturation is a fail-fast resource limit, never a queued wait
        // or an Adaptive downgrade.
        let admission = Arc::clone(&self.programmatic_admission)
            .try_acquire_owned()
            .map_err(|_| {
                HarnessError::ResourceLimit("programmatic run admission limit reached".into())
            })?;
        let selection = self.tools.select_scope_for_run(
            &request.input,
            &request.agent.tool_allowlist,
            ToolCaller::Programmatic,
            self.discovery_limits,
            &capabilities.limits,
            &request.cancellation,
            deadline,
        )?;
        let (scope, discovery) = match selection {
            ToolScopeSelection::Selected(scope, stats) => (scope, stats),
            ToolScopeSelection::LimitReached(stats) => {
                return Ok(AdaptiveProgrammaticReadiness::Fallback(Some(stats)));
            }
        };
        let limits = match effective_programmatic_limits(config, request, provider_program_bytes) {
            Ok(limits) => limits,
            Err(HarnessError::UnsupportedCapability(_)) => {
                return Ok(AdaptiveProgrammaticReadiness::Fallback(Some(discovery)));
            }
            Err(error) => return Err(error),
        };
        let structured_output = program_structured_output(&capabilities, &limits)?;
        Ok(AdaptiveProgrammaticReadiness::Ready(Box::new(
            AdaptiveProgrammaticPrepared {
                scope,
                discovery,
                capabilities,
                limits,
                structured_output,
                deadline,
                admission: Some(admission),
            },
        )))
    }

    pub(crate) async fn run_programmatic(
        &self,
        request: RunRequest,
        preflight: RunPreflight,
    ) -> Result<RunResult, HarnessError> {
        let config = self.programmatic.as_ref().ok_or_else(|| {
            HarnessError::UnsupportedCapability(
                "programmatic execution requires explicit host opt-in".into(),
            )
        })?;
        config.validate()?;
        let capabilities = self.provider.capabilities();
        if !capabilities.supports_tools
            || !capabilities.supports_programmatic_calling
            || capabilities.programmatic_conformance
                != Some(ProgrammaticConformance::StrictJsonAstV1)
        {
            return Err(HarnessError::UnsupportedCapability(
                "provider does not explicitly conform to strict programmatic JSON AST V1".into(),
            ));
        }
        let provider_program_bytes = capabilities
            .limits
            .max_program_bytes
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| {
                HarnessError::UnsupportedCapability(
                    "provider must advertise a nonzero program byte limit".into(),
                )
            })?;
        if request.agent.limits.max_model_calls < 2 {
            return Err(HarnessError::UnsupportedCapability(
                "programmatic execution requires at least two model calls".into(),
            ));
        }

        let deadline = programmatic_deadline(config, preflight.deadline)?;
        check_stopped(
            &request.cancellation,
            deadline,
            "programmatic run deadline reached",
        )?;
        let selection = self.tools.select_scope_for_run(
            &request.input,
            &request.agent.tool_allowlist,
            ToolCaller::Programmatic,
            self.discovery_limits,
            &capabilities.limits,
            &request.cancellation,
            deadline,
        )?;
        let (scope, discovery) = match selection {
            ToolScopeSelection::Selected(scope, stats) => (scope, stats),
            ToolScopeSelection::LimitReached(_) => {
                return Err(HarnessError::ResourceLimit(
                    "programmatic tool scope exceeds discovery limits".into(),
                ))
            }
        };

        let limits = effective_programmatic_limits(config, &request, provider_program_bytes)?;
        let program_structured_output = program_structured_output(&capabilities, &limits)?;

        let started = preflight.started;
        let run_id = request
            .run_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let trace_id = request
            .trace_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let model = request
            .overrides
            .model
            .clone()
            .unwrap_or_else(|| request.agent.default_model.clone());
        let result = RunResult::new(&run_id, RunStatus::Failed, &model, &trace_id);
        let mut events = EventEmitter::new(run_id.clone(), trace_id, Arc::clone(&self.events));
        events.emit(RunEvent::Started {
            run_id: run_id.clone(),
            trace_id: result.trace_id.clone(),
        });
        emit_discovery(&mut events, ToolCaller::Programmatic, discovery);
        events.emit(RunEvent::StrategySelected {
            requested: RunStrategy::Programmatic,
            selected: RunStrategy::Programmatic,
            reason: StrategySelectionReason::Forced,
        });

        self.run_programmatic_engine(
            &request,
            AdaptiveProgrammaticPrepared {
                scope,
                discovery,
                capabilities,
                limits,
                structured_output: program_structured_output,
                deadline,
                // Forced execution retains its historical admission point
                // inside the engine.
                admission: None,
            },
            ProgrammaticContinuation {
                result,
                events,
                output_validator: preflight.output_validator,
                deadline,
                started,
                model,
                model_calls: 0,
                planning_calls: 0,
                repair_calls: 0,
                synthesis_calls: 0,
                broker_state: BrokerState::default(),
                prepared_direct_scope: None,
            },
            RunStrategy::Programmatic,
            StrategySelectionReason::Forced,
        )
        .await
    }

    pub(crate) async fn run_adaptive_programmatic(
        &self,
        request: &RunRequest,
        prepared: AdaptiveProgrammaticPrepared,
        continuation: ProgrammaticContinuation,
    ) -> Result<RunResult, HarnessError> {
        self.run_programmatic_engine(
            request,
            prepared,
            continuation,
            RunStrategy::Adaptive,
            StrategySelectionReason::CapabilityDowngrade,
        )
        .await
    }

    async fn run_programmatic_engine(
        &self,
        request: &RunRequest,
        prepared: AdaptiveProgrammaticPrepared,
        continuation: ProgrammaticContinuation,
        requested: RunStrategy,
        fallback_selection_reason: StrategySelectionReason,
    ) -> Result<RunResult, HarnessError> {
        let AdaptiveProgrammaticPrepared {
            scope,
            discovery: _,
            capabilities,
            limits,
            structured_output: program_structured_output,
            deadline: prepared_deadline,
            admission: prepared_admission,
        } = prepared;
        let ProgrammaticContinuation {
            mut result,
            mut events,
            output_validator,
            deadline,
            started,
            model,
            mut model_calls,
            mut planning_calls,
            mut repair_calls,
            mut synthesis_calls,
            mut broker_state,
            prepared_direct_scope,
        } = continuation;
        debug_assert_eq!(deadline, prepared_deadline);
        let broker = ToolBroker::new(
            &self.tools,
            &scope,
            &self.policy,
            &self.approvals,
            &self.concurrency,
        );
        let mut dispatched = false;
        let mut invalid_program_exhausted = false;
        let mut program_attempt = 0u32;
        let terminal = async {
            events.emit(RunEvent::ProgramLifecycle {
                attempt: program_attempt.saturating_add(1),
                outcome: ProgramLifecycleOutcome::Started,
            });
            // This permit intentionally spans the complete Programmatic run:
            // synthesis-summary creation, program generation and repair,
            // parsing, compilation, VM construction, retained VM values, final
            // synthesis, and output validation. It is a whole Programmatic
            // run-memory admission, not a per-slice compute permit. The
            // nonblocking check prevents a tool-held run from deadlocking a
            // reentrant Programmatic run and avoids model or tool work for a
            // rejected candidate.
            let _live_run_permit = match prepared_admission {
                Some(permit) => permit,
                None => Arc::clone(&self.programmatic_admission)
                    .try_acquire_owned()
                    .map_err(|_| {
                        HarnessError::ResourceLimit(
                            "programmatic run admission limit reached".into(),
                        )
                    })?,
            };
            let mut broker_transcript = ProgrammaticTranscript::new(request, &limits)?;
            let mut generation_messages = initial_messages(request);
            generation_messages.push(Message::system(PROGRAM_PROMPT));
            ensure_transcript(&generation_messages, &request.agent.limits)?;

            let verified = loop {
                if program_attempt != 0 {
                    events.emit(RunEvent::ProgramLifecycle {
                        attempt: program_attempt.saturating_add(1),
                        outcome: ProgramLifecycleOutcome::Started,
                    });
                }
                check_stopped(
                    &request.cancellation,
                    deadline,
                    "programmatic run deadline reached",
                )?;
                let phase_calls = if program_attempt == 0 {
                    &mut planning_calls
                } else {
                    &mut repair_calls
                };
                let response = self
                    .programmatic_completion(
                        request,
                        &model,
                        generation_messages.clone(),
                        Some(&scope),
                        program_structured_output.clone(),
                        deadline,
                        &mut model_calls,
                        phase_calls,
                        &mut events,
                    )
                    .await?;
                let source = response.final_output.as_deref().ok_or_else(|| {
                    HarnessError::InvalidOutput("provider returned no program".into())
                });
                let mut statement_count = 0u32;
                let compiled = source.and_then(|source| {
                    Program::from_json(source.as_bytes(), &limits)
                        .and_then(|program| {
                            statement_count =
                                program.statement_count().min(u32::MAX as usize) as u32;
                            program.compile(&limits)
                        })
                        .map_err(sandbox_error)
                });
                match compiled {
                    Ok(program) => {
                        events.emit(RunEvent::ProgramLifecycle {
                            attempt: program_attempt.saturating_add(1),
                            outcome: ProgramLifecycleOutcome::Validated,
                        });
                        events.emit(RunEvent::ProgramValidated {
                            attempt: program_attempt.saturating_add(1),
                            statement_count,
                            instruction_count: program.instruction_count().min(u32::MAX as usize)
                                as u32,
                        });
                        break program;
                    }
                    Err(_error)
                        if program_attempt == 0
                            && request
                                .agent
                                .limits
                                .max_model_calls
                                .saturating_sub(model_calls)
                                > 1 =>
                    {
                        events.emit(RunEvent::ProgramLifecycle {
                            attempt: program_attempt.saturating_add(1),
                            outcome: ProgramLifecycleOutcome::Invalid,
                        });
                        program_attempt = 1;
                        generation_messages.push(Message::assistant(
                            response.final_output.unwrap_or_default(),
                        ));
                        generation_messages.push(Message::system(REPAIR_PROMPT));
                        ensure_transcript(&generation_messages, &request.agent.limits)?;
                    }
                    Err(error) => {
                        events.emit(RunEvent::ProgramLifecycle {
                            attempt: program_attempt.saturating_add(1),
                            outcome: ProgramLifecycleOutcome::Invalid,
                        });
                        invalid_program_exhausted = true;
                        return Err(error);
                    }
                }
            };

            let execution_nonce = Uuid::new_v4();
            let mut vm = Execution::with_attempt(
                verified,
                ExecutionId(next_sandbox_execution_id()),
                program_attempt,
            )
            .map_err(sandbox_error)?;
            let vm_started = StdInstant::now();
            let mut scheduling_slices = 0u64;
            let program_output = loop {
                check_stopped(
                    &request.cancellation,
                    deadline,
                    "programmatic run deadline reached",
                )?;
                scheduling_slices = scheduling_slices.saturating_add(1);
                let step = vm.step(limits.max_slice_fuel).map_err(sandbox_error)?;
                match step {
                    StepOutcome::Sliced => {
                        check_stopped(
                            &request.cancellation,
                            deadline,
                            "programmatic run deadline reached",
                        )?;
                        tokio::task::yield_now().await;
                        continue;
                    }
                    StepOutcome::Complete(value) => break value,
                    StepOutcome::Yielded { batch, resume } => {
                        let responses = self
                            .execute_programmatic_batch(
                                request,
                                &broker,
                                &mut result,
                                &mut events,
                                &mut broker_state,
                                &batch,
                                &execution_nonce,
                                &limits,
                                deadline,
                                &mut dispatched,
                                &mut broker_transcript,
                            )
                            .await?;
                        vm.resume(resume, responses).map_err(sandbox_error)?;
                    }
                    _ => {
                        return Err(HarnessError::Tool(
                            "sandbox returned an unsupported step outcome".into(),
                        ))
                    }
                }
            };
            let metrics = vm.metrics();
            events.emit(RunEvent::ProgramExecutionCompleted {
                attempt: program_attempt.saturating_add(1),
                fuel_used: metrics.fuel_used,
                scheduling_slices,
                tool_yields: metrics.yields,
                branches: metrics.branches,
                loop_iterations: metrics.loop_iterations,
                fanout_batches: metrics.fanout_batches,
                partial_failures: broker_state
                    .tool_failed
                    .saturating_add(broker_state.tool_cancelled)
                    .saturating_add(broker_state.tool_rejected),
                peak_accounted_bytes: metrics.retained_bytes as u64,
                duration_ms: vm_started.elapsed().as_millis() as u64,
            });
            drop(vm);

            let output_json = serialize_synthesis_input(&program_output, &broker_transcript)?;
            let mut synthesis_messages = initial_messages(request);
            synthesis_messages.push(Message::system(SYNTHESIS_PROMPT));
            synthesis_messages.push(Message::user(output_json));
            ensure_transcript(&synthesis_messages, &request.agent.limits)?;
            let response = self
                .programmatic_completion(
                    request,
                    &model,
                    synthesis_messages,
                    None,
                    agent_structured_output(&capabilities, request.agent.output_schema.as_ref()),
                    deadline,
                    &mut model_calls,
                    &mut synthesis_calls,
                    &mut events,
                )
                .await?;
            let output = response.final_output.ok_or_else(|| {
                HarnessError::InvalidOutput("final synthesis returned no output".into())
            })?;
            if output.trim().is_empty() {
                return Err(HarnessError::InvalidOutput(
                    "final synthesis returned empty output".into(),
                ));
            }
            validate_output(
                output_validator.as_ref(),
                &output,
                request.agent.limits.max_json_depth,
            )?;
            events.emit(RunEvent::ProgramLifecycle {
                attempt: program_attempt.saturating_add(1),
                outcome: ProgramLifecycleOutcome::Succeeded,
            });
            result.status = RunStatus::Completed;
            result.final_output = Some(output);
            Ok::<(), HarnessError>(())
        }
        .await;

        if invalid_program_exhausted && !dispatched && broker_state.tool_issued == 0 {
            events.emit(RunEvent::ProgramLifecycle {
                attempt: program_attempt.saturating_add(1),
                outcome: ProgramLifecycleOutcome::Fallback,
            });
            // The continuation is one logical run, including its tightened
            // programmatic host deadline rather than the preflight request
            // deadline that may be longer.
            return self
                .run_direct_continuation(
                    request.clone(),
                    DirectStrategyEvents {
                        requested,
                        reason: fallback_selection_reason,
                        fallback: Some(StrategyFallbackReason::InvalidProgram),
                        fallback_from: Some(RunStrategy::Programmatic),
                        prior_discovery: None,
                    },
                    RunPreflight {
                        output_validator,
                        deadline,
                        started,
                    },
                    DirectContinuation {
                        result,
                        events,
                        broker_state,
                        model_calls,
                        usage: ProgrammaticUsage {
                            planning_model_calls: planning_calls,
                            repair_model_calls: repair_calls,
                            final_synthesis_model_calls: synthesis_calls,
                        },
                        prepared_direct_scope,
                    },
                )
                .await;
        }
        if let Err(error) = terminal {
            events.emit(RunEvent::ProgramLifecycle {
                attempt: program_attempt.saturating_add(1),
                outcome: terminal_lifecycle_outcome(&error),
            });
            if dispatched {
                apply_terminal_error(
                    &mut result,
                    HarnessError::Tool(
                        "programmatic execution ended with an uncertain post-dispatch outcome"
                            .into(),
                    ),
                );
            } else {
                apply_terminal_error(&mut result, error);
            }
        }
        result.duration_ms = started.elapsed().as_millis() as u64;
        broker_state.finalize_usage();
        events.emit(RunEvent::StrategyUsage {
            strategy: RunStrategy::Programmatic,
            model_calls,
            planning_model_calls: planning_calls,
            repair_model_calls: repair_calls,
            recovery_model_calls: 0,
            final_synthesis_model_calls: synthesis_calls,
            reactive_model_calls: 0,
            tool_calls: broker_state.tool_calls,
            tool_issued: broker_state.tool_issued,
            tool_reused: broker_state.tool_reused,
            tool_rejected: broker_state.tool_rejected,
            tool_pre_dispatch_aborted: broker_state.tool_pre_dispatch_aborted,
            tool_completed: broker_state.tool_completed,
            tool_failed: broker_state.tool_failed,
            tool_cancelled: broker_state.tool_cancelled,
            duration_ms: result.duration_ms,
        });
        events.emit(RunEvent::Completed {
            status: result.status.clone(),
        });
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    async fn programmatic_completion(
        &self,
        request: &RunRequest,
        model: &str,
        messages: Vec<Message>,
        scope: Option<&ToolScope>,
        structured_output: Option<StructuredOutputRequest>,
        deadline: Option<Instant>,
        model_calls: &mut u32,
        phase_calls: &mut u32,
        events: &mut EventEmitter,
    ) -> Result<ModelResponse, HarnessError> {
        if *model_calls >= request.agent.limits.max_model_calls {
            return Err(HarnessError::ResourceLimit(
                "model call limit reached".into(),
            ));
        }
        *model_calls += 1;
        *phase_calls += 1;
        events.emit(RunEvent::ModelRequested {
            call_number: *model_calls,
            model: model.into(),
        });
        let call_cancellation = request.cancellation.child_token();
        let call_deadline =
            provider_deadline(deadline, request.agent.limits.max_model_call_duration_ms)?;
        let response = await_guarded(
            self.provider.complete(ModelRequest {
                model: model.into(),
                messages,
                tools: scope.map_or_else(Vec::new, |scope| scope.definitions().to_vec()),
                prepared_tools: scope.and_then(ToolScope::prepared),
                generation: merge_generation(
                    &request.agent.generation,
                    &request.overrides.generation,
                ),
                structured_output,
                metadata: request.metadata.clone(),
                cancellation: call_cancellation.clone(),
            }),
            &request.cancellation,
            call_deadline,
            "provider call deadline reached",
            Some(&call_cancellation),
        )
        .await?;
        validate_model_response(&response, &request.agent.limits)?;
        if !response.tool_calls.is_empty() {
            return Err(HarnessError::InvalidOutput(
                "programmatic model phases cannot return native tool calls".into(),
            ));
        }
        events.emit(RunEvent::ModelResponded {
            call_number: *model_calls,
        });
        Ok(response)
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_programmatic_batch(
        &self,
        request: &RunRequest,
        broker: &ToolBroker<'_>,
        result: &mut RunResult,
        events: &mut EventEmitter,
        state: &mut BrokerState,
        batch: &ToolBatch,
        execution_nonce: &Uuid,
        limits: &SandboxLimits,
        deadline: Option<Instant>,
        dispatched: &mut bool,
        transcript: &mut ProgrammaticTranscript,
    ) -> Result<Vec<ToolResponse>, HarnessError> {
        // A fan-out is an all-or-nothing static admission decision. Check every
        // requested call before the broker can consult policy or approval for
        // any individual entry.
        if batch.requests_read_only_fan_out() {
            for request_call in batch.calls() {
                let Some(tool) = self.tools.get(&request_call.tool_id) else {
                    return Err(HarnessError::InvalidTool(
                        "programmatic fan-out references an unavailable tool".into(),
                    ));
                };
                if !request
                    .agent
                    .tool_allowlist
                    .iter()
                    .any(|id| id == &request_call.tool_id)
                    || !tool.definition().allows_caller(ToolCaller::Programmatic)
                    || !tool.definition().read_only
                    || !tool.definition().parallel_safe
                {
                    return Err(HarnessError::InvalidTool(
                        "programmatic fan-out requires allowed read-only, parallel-safe tools"
                            .into(),
                    ));
                }
            }
            let capabilities = self.provider.capabilities();
            let provider_parallel = capabilities
                .supports_parallel_tool_calls
                .then_some(capabilities.limits.max_parallel_tool_calls)
                .flatten()
                .filter(|limit| *limit > 0)
                .ok_or_else(|| {
                    HarnessError::UnsupportedCapability(
                        "programmatic fan-out requires a nonzero provider parallel-call limit"
                            .into(),
                    )
                })? as usize;
            let effective = self
                .programmatic
                .as_ref()
                .map_or(0, |config| config.max_fanout_concurrency)
                .min(request.agent.limits.max_programmatic_fanout_concurrency as usize)
                .min(provider_parallel)
                .min(MAX_FANOUT_CONCURRENCY);
            if batch.calls().len() > effective {
                return Err(HarnessError::ResourceLimit(
                    "programmatic fan-out exceeds the effective concurrency limit".into(),
                ));
            }
        }

        // Reserve the value-free broker summary before the batch enters policy
        // or approval. Raw arguments and results stay inside the execution
        // boundary and never expand the final synthesis prompt.
        transcript.reserve_batch(batch)?;

        let mut prepared: Vec<(usize, Box<PreparedCall>)> = Vec::new();
        let mut responses: Vec<Option<ToolResponse>> = std::iter::repeat_with(|| None)
            .take(batch.calls().len())
            .collect();
        let mut transcript_slots: Vec<Option<bool>> = std::iter::repeat_with(|| None)
            .take(batch.calls().len())
            .collect();
        for (index, request_call) in batch.calls().iter().enumerate() {
            check_stopped(
                &request.cancellation,
                deadline,
                "programmatic dispatch deadline reached",
            )?;
            let call = ToolCall::new(
                format!(
                    "programmatic-{}-{}-{}-{}",
                    request_call.program_attempt,
                    execution_nonce,
                    request_call.call_site,
                    request_call.dynamic_ordinal
                ),
                request_call.tool_id.clone(),
                serde_json::to_string(&request_call.arguments).map_err(|_| {
                    HarnessError::InvalidArguments(
                        "programmatic arguments could not be serialized".into(),
                    )
                })?,
            );
            let context = ToolCallContext::new(
                result.id.clone(),
                result.trace_id.clone(),
                call.id.clone(),
                call.tool_id.clone(),
            )
            .with_programmatic_occurrence(
                request_call.program_attempt,
                request_call.call_site,
                request_call.dynamic_ordinal,
                call.id.clone(),
            );
            match broker
                .prepare(
                    request,
                    result,
                    events,
                    state,
                    call,
                    ToolCaller::Programmatic,
                    false,
                    false,
                    Some(context),
                    deadline,
                )
                .await?
            {
                PrepareOutcome::Ready(call) => prepared.push((index, call)),
                PrepareOutcome::Rejected(_) => {
                    responses[index] = Some(ToolResponse::failure(request_call));
                    transcript_slots[index] = Some(false);
                }
                PrepareOutcome::Stop => {
                    return Err(HarnessError::ResourceLimit(
                        "programmatic batch exhausted the tool call budget before dispatch".into(),
                    ));
                }
                PrepareOutcome::Reused(value) => {
                    responses[index] = Some(tool_response(request_call, value.as_ref(), limits)?);
                    transcript_slots[index] = Some(value.ok);
                }
            }
        }

        for (_, call) in &prepared {
            broker.mark_dispatched(state, call);
        }
        if !prepared.is_empty() {
            *dispatched = true;
        }
        let executions = if batch.requests_read_only_fan_out() {
            join_all(
                prepared
                    .iter()
                    .map(|(_, call)| broker.execute(call, request, deadline)),
            )
            .await
        } else {
            let mut values = Vec::with_capacity(prepared.len());
            for (_, call) in &prepared {
                values.push(broker.execute(call, request, deadline).await);
            }
            values
        };
        let mut first_execution_error = None;
        for (((index, call), execution), request_call) in prepared
            .iter()
            .zip(executions)
            .zip(prepared.iter().map(|(index, _)| &batch.calls()[*index]))
        {
            let execution = match execution {
                Ok(execution) if execution.result.ok && execution.validation_error.is_none() => {
                    execution
                }
                Ok(execution) => {
                    broker.record_execution(state, call, &execution);
                    broker.mark_uncertain(state, call);
                    events.emit(RunEvent::ToolCompleted {
                        call_id: call.call.id.clone(),
                        tool_id: call.call.tool_id.clone(),
                        ok: false,
                    });
                    first_execution_error.get_or_insert_with(|| {
                        HarnessError::Tool(
                            "programmatic tool returned a failed or invalid result".into(),
                        )
                    });
                    transcript_slots[*index] = Some(false);
                    continue;
                }
                Err(error) => {
                    state.record_execution_error(&error);
                    broker.mark_uncertain(state, call);
                    events.emit(RunEvent::ToolCompleted {
                        call_id: call.call.id.clone(),
                        tool_id: call.call.tool_id.clone(),
                        ok: false,
                    });
                    if first_execution_error.is_none() {
                        first_execution_error = Some(error);
                    }
                    transcript_slots[*index] = Some(false);
                    continue;
                }
            };
            match tool_response(request_call, execution.result.as_ref(), limits) {
                Ok(response) => {
                    responses[*index] = Some(response);
                    events.emit(RunEvent::ToolCompleted {
                        call_id: call.call.id.clone(),
                        tool_id: call.call.tool_id.clone(),
                        ok: true,
                    });
                    broker.record_execution(state, call, &execution);
                }
                Err(error) => {
                    // The tool did run successfully but its response cannot
                    // safely be retained by the VM. Its external effect is
                    // therefore terminally uncertain and never resumed.
                    state.record_execution_error(&error);
                    broker.mark_uncertain(state, call);
                    events.emit(RunEvent::ToolCompleted {
                        call_id: call.call.id.clone(),
                        tool_id: call.call.tool_id.clone(),
                        ok: false,
                    });
                    first_execution_error.get_or_insert(error);
                    transcript_slots[*index] = Some(false);
                    continue;
                }
            }
            transcript_slots[*index] = Some(true);
        }
        for ok in transcript_slots.into_iter().flatten() {
            transcript.record(ok)?;
        }
        if let Some(error) = first_execution_error {
            return Err(error);
        }
        responses
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| HarnessError::Tool("programmatic batch response was incomplete".into()))
    }
}

fn tool_response(
    request: &llama_harness_programmatic_sandbox::ToolRequest,
    result: &ToolResult,
    limits: &SandboxLimits,
) -> Result<ToolResponse, HarnessError> {
    if result.ok {
        ToolResponse::success(request, &result.output, limits).map_err(sandbox_error)
    } else {
        Ok(ToolResponse::failure(request))
    }
}

fn sandbox_error(error: llama_harness_programmatic_sandbox::SandboxError) -> HarnessError {
    match error.code() {
        SandboxErrorCode::ResourceLimit => HarnessError::ResourceLimit(error.to_string()),
        SandboxErrorCode::InvalidResume | SandboxErrorCode::Execution => {
            HarnessError::Tool(error.to_string())
        }
        _ => HarnessError::InvalidOutput(error.to_string()),
    }
}

fn terminal_lifecycle_outcome(error: &HarnessError) -> ProgramLifecycleOutcome {
    match error {
        HarnessError::Cancelled => ProgramLifecycleOutcome::Cancelled,
        HarnessError::TimedOut(_) => ProgramLifecycleOutcome::TimedOut,
        HarnessError::ResourceLimit(_) => ProgramLifecycleOutcome::LimitReached,
        _ => ProgramLifecycleOutcome::Failed,
    }
}

fn next_sandbox_execution_id() -> u64 {
    NEXT_SANDBOX_EXECUTION_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn transcript(limit: u64, maximum_program_return_bytes: u64) -> ProgrammaticTranscript {
        ProgrammaticTranscript {
            summary: ProgrammaticBrokerSummary::default(),
            initial_message_bytes: 17,
            limit,
            maximum_program_return_bytes,
        }
    }

    fn projected_limit(transcript: &ProgrammaticTranscript, program_return: &Value) -> u64 {
        checked_transcript_sum([
            transcript.initial_message_bytes,
            SYNTHESIS_PROMPT.len() as u64,
            transcript
                .synthesis_payload_capacity(program_return)
                .unwrap() as u64,
        ])
        .unwrap()
    }

    fn assert_payload_matches_serde(transcript: &ProgrammaticTranscript, program_return: &Value) {
        let expected = serde_json::to_vec(&ProgrammaticSynthesisInput {
            program_return,
            broker_calls: transcript.summary,
        })
        .unwrap();
        assert_eq!(
            transcript
                .synthesis_payload_capacity(program_return)
                .unwrap(),
            expected.len()
        );
        assert_eq!(
            serialize_synthesis_input(program_return, transcript)
                .unwrap()
                .as_bytes(),
            expected.as_slice()
        );
    }

    #[test]
    fn synthesis_summary_accounting_matches_canonical_serialization() {
        let program_return = json!({"value":"returned","items":[1,2,3]});
        let return_bytes = count_json_bytes(&program_return).unwrap();
        let mut transcript = transcript(u64::MAX, return_bytes);

        assert_payload_matches_serde(&transcript, &program_return);

        transcript.record(true).unwrap();
        assert_payload_matches_serde(&transcript, &program_return);

        transcript.record(false).unwrap();
        transcript.record(true).unwrap();
        assert_payload_matches_serde(&transcript, &program_return);
        assert_eq!(transcript.summary.total, 3);
        assert_eq!(transcript.summary.succeeded, 2);
        assert_eq!(transcript.summary.failed, 1);
    }

    #[test]
    fn synthesis_summary_enforces_exact_record_and_batch_boundaries() {
        let program_return = json!({"summary":"bounded"});
        let return_bytes = count_json_bytes(&program_return).unwrap();
        let mut probe = transcript(u64::MAX, return_bytes);
        for _ in 0..10 {
            probe.record(true).unwrap();
        }
        let limit = projected_limit(&probe, &program_return);

        let mut rejected = transcript(limit - 1, return_bytes);
        for _ in 0..9 {
            rejected.record(true).unwrap();
        }
        assert!(matches!(
            rejected.record(true),
            Err(HarnessError::ResourceLimit(_))
        ));
        assert_eq!(rejected.summary.total, 9);
        assert_payload_matches_serde(&rejected, &program_return);

        let mut at_limit = transcript(limit, return_bytes);
        for _ in 0..10 {
            at_limit.record(true).unwrap();
        }
        assert_payload_matches_serde(&at_limit, &program_return);

        let mut above_limit = transcript(limit + 1, return_bytes);
        for _ in 0..10 {
            above_limit.record(true).unwrap();
        }
        assert_payload_matches_serde(&above_limit, &program_return);

        let fanout_summary = ProgrammaticBrokerSummary {
            total: 3,
            succeeded: 3,
            failed: 3,
        };
        let fanout_payload = synthesis_payload_bytes(return_bytes, fanout_summary).unwrap();
        let fanout_limit =
            checked_transcript_sum([17, SYNTHESIS_PROMPT.len() as u64, fanout_payload]).unwrap();
        assert!(matches!(
            transcript(fanout_limit - 1, return_bytes).ensure_summary(fanout_summary, return_bytes),
            Err(HarnessError::ResourceLimit(_))
        ));
        transcript(fanout_limit, return_bytes)
            .ensure_summary(fanout_summary, return_bytes)
            .unwrap();
        transcript(fanout_limit + 1, return_bytes)
            .ensure_summary(fanout_summary, return_bytes)
            .unwrap();
    }

    #[test]
    fn synthesis_summary_omits_raw_fanout_values() {
        let program_return = json!({"status":"complete"});
        let mut transcript = transcript(u64::MAX, count_json_bytes(&program_return).unwrap());
        for index in 0..512u64 {
            transcript.record(index % 3 != 0).unwrap();
        }
        assert_payload_matches_serde(&transcript, &program_return);
        let serialized = serialize_synthesis_input(&program_return, &transcript).unwrap();
        let payload: Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(payload["broker_calls"]["total"], 512);
        assert_eq!(payload["broker_calls"]["succeeded"], 341);
        assert_eq!(payload["broker_calls"]["failed"], 171);
        assert!(!serialized.contains("fanout_read"));
        assert!(!serialized.contains("arguments"));
        assert!(!serialized.contains("output"));
        assert!(serialized.len() < 128);
    }
}
