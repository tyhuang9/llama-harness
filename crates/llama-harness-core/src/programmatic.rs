use crate::{
    broker::{BrokerState, PrepareOutcome, PreparedCall, ToolBroker},
    discovery::{ToolScope, ToolScopeSelection},
    event::{EventEmitter, ProgramLifecycleOutcome, RunEvent, StrategySelectionReason},
    runner::{
        apply_terminal_error, await_guarded, check_stopped, emit_discovery, ensure_transcript,
        initial_messages, merge_generation, provider_deadline, validate_model_response,
        validate_output, DirectContinuation, DirectStrategyEvents, ProgrammaticUsage, RunPreflight,
    },
    AgentRunner, HarnessError, Message, ModelRequest, ModelResponse, ProgrammaticConformance,
    RunRequest, RunResult, RunStatus, RunStrategy, StrategyFallbackReason, ToolCall,
    ToolCallContext, ToolCaller, ToolResult,
};
use futures_util::future::join_all;
use llama_harness_programmatic_sandbox::{
    Execution, ExecutionId, Program, SandboxErrorCode, SandboxLimits, StepOutcome, ToolBatch,
    ToolResponse, HARD_LIMITS,
};
use serde::Serialize;
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

// The sandbox only needs a process-local opaque resume scope. Public broker
// occurrence and effect identities use the full UUID below; this counter is
// deliberately never used as an externally visible execution identity.
static NEXT_SANDBOX_EXECUTION_ID: AtomicU64 = AtomicU64::new(1);

const PROGRAM_PROMPT: &str = r#"Return only a strict llama-harness program JSON object. Use version 1 and a body array. Statements are let, branch, for_each, map, filter, reduce, invoke, fan_out, and return. Expressions are null, boolean, integer, string, variable, path, array, object, binary, and unary. Tool IDs in invoke/fan_out must be static IDs from the supplied catalog. All loops and collections require explicit bounded limits. Do not use markdown, floats, dynamic tool names, code strings, imports, functions, recursion, mutation, reflection, exceptions, regex, or prose."#;

const REPAIR_PROMPT: &str = "The previous program failed strict structural verification. Return one corrected version-1 program JSON object only. Do not add markdown or prose.";

const SYNTHESIS_PROMPT: &str = "A verified program completed. Produce the final answer using only the inert program return and broker-audited tool transcript in the next user message.";

#[derive(Clone, Serialize)]
struct ProgrammaticTranscriptEntry {
    tool_id: String,
    arguments: Value,
    ok: bool,
    output: Value,
}

#[derive(Serialize)]
struct ProgrammaticTranscriptEntryRef<'a> {
    tool_id: &'a str,
    arguments: &'a Value,
    ok: bool,
    output: &'a Value,
}

#[derive(Serialize)]
struct ProgrammaticSynthesisInput<'a> {
    program_return: &'a Value,
    broker_calls: &'a [ProgrammaticTranscriptEntry],
}

/// Bounded, canonical evidence passed to the final synthesis call.
///
/// The accounting intentionally reserves a full potential batch before the
/// broker can issue any effect, then measures each entry without allocating a
/// second serialized copy before retaining it. This makes a later synthesis
/// capacity failure impossible for accepted sandbox/tool bounds.
struct ProgrammaticTranscript {
    entries: Vec<ProgrammaticTranscriptEntry>,
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
            entries: Vec::new(),
            initial_message_bytes,
            limit: request.agent.limits.max_transcript_bytes,
            maximum_program_return_bytes: u64::try_from(limits.max_output_bytes).map_err(|_| {
                HarnessError::ResourceLimit(
                    "program return byte limit does not fit accounting".into(),
                )
            })?,
        };
        transcript.ensure_projected(0, 0, transcript.maximum_program_return_bytes)?;
        Ok(transcript)
    }

    fn reserve_batch(&self, batch: &ToolBatch, request: &RunRequest) -> Result<(), HarnessError> {
        let mut maximum_new_entries = 0u64;
        for call in batch.calls() {
            let tool_id_bytes = count_json_bytes(&call.tool_id)?;
            let argument_bytes = count_json_bytes(&call.arguments)?;
            let entry_bytes = checked_transcript_sum([
                96,
                tool_id_bytes,
                argument_bytes,
                request.agent.limits.max_tool_result_bytes,
            ])?;
            maximum_new_entries = checked_transcript_sum([maximum_new_entries, entry_bytes])?;
        }
        self.ensure_projected(
            maximum_new_entries,
            u64::try_from(batch.calls().len()).map_err(|_| {
                HarnessError::ResourceLimit(
                    "programmatic batch length does not fit accounting".into(),
                )
            })?,
            self.maximum_program_return_bytes,
        )
    }

    fn append(
        &mut self,
        tool_id: &str,
        arguments: &Value,
        ok: bool,
        output: &Value,
    ) -> Result<(), HarnessError> {
        let entry = ProgrammaticTranscriptEntryRef {
            tool_id,
            arguments,
            ok,
            output,
        };
        let entry_bytes = count_json_bytes(&entry)?;
        self.ensure_projected(entry_bytes, 1, self.maximum_program_return_bytes)?;
        self.entries.try_reserve(1).map_err(|_| {
            HarnessError::ResourceLimit("programmatic transcript allocation failed".into())
        })?;
        self.entries.push(ProgrammaticTranscriptEntry {
            tool_id: tool_id.into(),
            arguments: arguments.clone(),
            ok,
            output: output.clone(),
        });
        Ok(())
    }

    fn synthesis_payload_capacity(&self, program_return: &Value) -> Result<usize, HarnessError> {
        let program_return_bytes = count_json_bytes(program_return)?;
        let entry_bytes = self
            .entries
            .iter()
            .map(count_json_bytes)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .try_fold(0u64, |total, bytes| checked_transcript_sum([total, bytes]))?;
        self.ensure_projected(0, 0, program_return_bytes)?;
        let payload_bytes =
            synthesis_payload_bytes(program_return_bytes, entry_bytes, self.entries.len() as u64)?;
        usize::try_from(payload_bytes).map_err(|_| {
            HarnessError::ResourceLimit("programmatic synthesis payload does not fit memory".into())
        })
    }

    fn ensure_projected(
        &self,
        additional_entry_bytes: u64,
        additional_entries: u64,
        program_return_bytes: u64,
    ) -> Result<(), HarnessError> {
        let retained_entry_bytes = self
            .entries
            .iter()
            .map(count_json_bytes)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .try_fold(0u64, |total, bytes| checked_transcript_sum([total, bytes]))?;
        let payload_bytes = synthesis_payload_bytes(
            program_return_bytes,
            checked_transcript_sum([retained_entry_bytes, additional_entry_bytes])?,
            checked_transcript_sum([self.entries.len() as u64, additional_entries])?,
        )?;
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
    entry_bytes: u64,
    entry_count: u64,
) -> Result<u64, HarnessError> {
    // {"program_return":<value>,"broker_calls":[<entries>]}. The fixed
    // punctuation and member names contribute 37 bytes.
    checked_transcript_sum([
        37,
        program_return_bytes,
        entry_bytes,
        entry_count.saturating_sub(1),
    ])
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
            broker_calls: &transcript.entries,
        },
    )
    .map_err(|_| {
        HarnessError::InvalidOutput("program synthesis input could not be serialized".into())
    })?;
    debug_assert!(bytes.len() <= capacity);
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
    /// Maximum live VMs admitted by this runner.
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
                "programmatic VM admission must be within 1..=16".into(),
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

impl AgentRunner {
    pub(crate) async fn run_programmatic(
        &self,
        request: RunRequest,
        mut preflight: RunPreflight,
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

        let configured_deadline = Instant::now()
            .checked_add(Duration::from_millis(config.max_duration_ms))
            .ok_or_else(|| {
                HarnessError::InvalidRequest("programmatic duration is too large".into())
            })?;
        let deadline = Some(
            preflight
                .deadline
                .map_or(configured_deadline, |run| run.min(configured_deadline)),
        );
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
        let mut result = RunResult::new(&run_id, RunStatus::Failed, &model, &trace_id);
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

        let mut model_calls = 0u32;
        let mut planning_calls = 0u32;
        let mut repair_calls = 0u32;
        let mut synthesis_calls = 0u32;
        let mut broker_state = BrokerState::default();
        let broker = ToolBroker::new(
            &self.tools,
            &scope,
            &self.policy,
            &self.approvals,
            &self.concurrency,
        );
        let mut dispatched = false;
        let mut invalid_program_exhausted = false;
        let terminal = async {
            let mut broker_transcript = ProgrammaticTranscript::new(&request, &limits)?;
            let mut generation_messages = initial_messages(&request);
            generation_messages.push(Message::system(PROGRAM_PROMPT));
            ensure_transcript(&generation_messages, &request.agent.limits)?;

            let mut program_attempt = 0u32;
            let verified = loop {
                events.emit(RunEvent::ProgramLifecycle {
                    attempt: program_attempt.saturating_add(1),
                    outcome: ProgramLifecycleOutcome::Started,
                });
                let phase_calls = if program_attempt == 0 {
                    &mut planning_calls
                } else {
                    &mut repair_calls
                };
                let response = self
                    .programmatic_completion(
                        &request,
                        &model,
                        generation_messages.clone(),
                        Some(&scope),
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
            let program_output = loop {
                check_stopped(
                    &request.cancellation,
                    deadline,
                    "programmatic run deadline reached",
                )?;
                // VM admission covers synchronous compute only. The permit is
                // released before any provider, policy, approval, or tool await.
                let step = {
                    let _admission = await_guarded(
                        async {
                            Arc::clone(&self.programmatic_admission)
                                .acquire_owned()
                                .await
                                .map_err(|_| {
                                    HarnessError::ResourceLimit(
                                        "programmatic VM admission closed".into(),
                                    )
                                })
                        },
                        &request.cancellation,
                        deadline,
                        "programmatic VM admission exceeded run deadline",
                        None,
                    )
                    .await?;
                    vm.step(limits.max_slice_fuel).map_err(sandbox_error)?
                };
                match step {
                    StepOutcome::Sliced => continue,
                    StepOutcome::Complete(value) => break value,
                    StepOutcome::Yielded { batch, resume } => {
                        let responses = self
                            .execute_programmatic_batch(
                                &request,
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
                        // Resuming may materialize and retain a bounded tool
                        // response tree, so it is VM compute just like step.
                        let _admission = await_guarded(
                            async {
                                Arc::clone(&self.programmatic_admission)
                                    .acquire_owned()
                                    .await
                                    .map_err(|_| {
                                        HarnessError::ResourceLimit(
                                            "programmatic VM admission closed".into(),
                                        )
                                    })
                            },
                            &request.cancellation,
                            deadline,
                            "programmatic VM admission exceeded run deadline",
                            None,
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

            let output_json = serialize_synthesis_input(&program_output, &broker_transcript)?;
            let mut synthesis_messages = initial_messages(&request);
            synthesis_messages.push(Message::system(SYNTHESIS_PROMPT));
            synthesis_messages.push(Message::user(output_json));
            ensure_transcript(&synthesis_messages, &request.agent.limits)?;
            let response = self
                .programmatic_completion(
                    &request,
                    &model,
                    synthesis_messages,
                    None,
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
                preflight.output_validator.as_ref(),
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
                attempt: repair_calls.saturating_add(1),
                outcome: ProgramLifecycleOutcome::Fallback,
            });
            // The continuation is one logical run, including its tightened
            // programmatic host deadline rather than the preflight request
            // deadline that may be longer.
            preflight.deadline = deadline;
            return self
                .run_direct_continuation(
                    request,
                    DirectStrategyEvents {
                        requested: RunStrategy::Programmatic,
                        reason: StrategySelectionReason::Forced,
                        fallback: Some(StrategyFallbackReason::InvalidProgram),
                        prior_discovery: None,
                    },
                    preflight,
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
                    },
                )
                .await;
        }
        if let Err(error) = terminal {
            events.emit(RunEvent::ProgramLifecycle {
                attempt: repair_calls.saturating_add(1).max(1),
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

        // Reserve the full worst-case canonical broker envelope before the
        // batch enters policy or approval. A later tool result therefore
        // cannot make the synthesis payload exceed its budget after an effect
        // has started.
        transcript.reserve_batch(batch, request)?;

        let mut prepared: Vec<(usize, Box<PreparedCall>)> = Vec::new();
        let mut responses: Vec<Option<ToolResponse>> = std::iter::repeat_with(|| None)
            .take(batch.calls().len())
            .collect();
        let mut transcript_slots: Vec<Option<ProgrammaticTranscriptEntry>> =
            std::iter::repeat_with(|| None)
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
                    transcript_slots[index] = Some(ProgrammaticTranscriptEntry {
                        tool_id: request_call.tool_id.clone(),
                        arguments: request_call.arguments.clone(),
                        ok: false,
                        output: Value::Null,
                    });
                }
                PrepareOutcome::Stop => {
                    return Err(HarnessError::ResourceLimit(
                        "programmatic batch exhausted the tool call budget before dispatch".into(),
                    ));
                }
                PrepareOutcome::Reused(value) => {
                    responses[index] = Some(tool_response(request_call, value.as_ref(), limits)?);
                    transcript_slots[index] = Some(ProgrammaticTranscriptEntry {
                        tool_id: request_call.tool_id.clone(),
                        arguments: request_call.arguments.clone(),
                        ok: value.ok,
                        output: value.output.clone(),
                    });
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
                    transcript_slots[*index] = Some(ProgrammaticTranscriptEntry {
                        tool_id: call.call.tool_id.clone(),
                        arguments: call.arguments.clone(),
                        ok: false,
                        output: Value::Null,
                    });
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
                    transcript_slots[*index] = Some(ProgrammaticTranscriptEntry {
                        tool_id: call.call.tool_id.clone(),
                        arguments: call.arguments.clone(),
                        ok: false,
                        output: Value::Null,
                    });
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
                    transcript_slots[*index] = Some(ProgrammaticTranscriptEntry {
                        tool_id: call.call.tool_id.clone(),
                        arguments: call.arguments.clone(),
                        ok: false,
                        output: Value::Null,
                    });
                    continue;
                }
            }
            transcript_slots[*index] = Some(ProgrammaticTranscriptEntry {
                tool_id: call.call.tool_id.clone(),
                arguments: call.arguments.clone(),
                ok: true,
                output: execution.result.output.clone(),
            });
        }
        for entry in transcript_slots.into_iter().flatten() {
            transcript.append(&entry.tool_id, &entry.arguments, entry.ok, &entry.output)?;
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
