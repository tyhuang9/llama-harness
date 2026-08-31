use crate::{
    broker::{BrokerState, PrepareOutcome, PreparedCall, ToolBroker},
    discovery::{ToolScope, ToolScopeSelection},
    event::{EventEmitter, RunEvent, StrategySelectionReason},
    runner::{
        apply_terminal_error, await_guarded, check_stopped, emit_discovery, ensure_transcript,
        initial_messages, merge_generation, provider_deadline, validate_model_response,
        validate_output, RunPreflight,
    },
    AgentRunner, HarnessError, Message, ModelRequest, ModelResponse, ProgrammaticConformance,
    RunRequest, RunResult, RunStatus, RunStrategy, ToolCall, ToolCaller, ToolResult,
};
use futures_util::future::join_all;
use llama_harness_programmatic_sandbox::{
    Execution, ExecutionId, Program, SandboxErrorCode, SandboxLimits, StepOutcome, ToolBatch,
    ToolResponse,
};
use std::{sync::Arc, time::Duration};
use tokio::time::Instant;
use uuid::Uuid;

const DEFAULT_PROGRAMMATIC_DURATION_MS: u64 = 60_000;
const HARD_PROGRAMMATIC_DURATION_MS: u64 = 300_000;
const DEFAULT_VM_ADMISSION: usize = 4;
const HARD_VM_ADMISSION: usize = 16;
const MAX_FANOUT_CONCURRENCY: usize = 8;

const PROGRAM_PROMPT: &str = r#"Return only a strict llama-harness program JSON object. Use version 1 and a body array. Statements are let, branch, for_each, map, filter, reduce, invoke, fan_out, and return. Expressions are null, boolean, integer, string, variable, path, array, object, binary, and unary. Tool IDs in invoke/fan_out must be static IDs from the supplied catalog. All loops and collections require explicit bounded limits. Do not use markdown, floats, dynamic tool names, code strings, imports, functions, recursion, mutation, reflection, exceptions, regex, or prose."#;

const REPAIR_PROMPT: &str = "The previous program failed strict structural verification. Return one corrected version-1 program JSON object only. Do not add markdown or prose.";

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
        let admission = await_guarded(
            async {
                Arc::clone(&self.programmatic_admission)
                    .acquire_owned()
                    .await
                    .map_err(|_| {
                        HarnessError::ResourceLimit("programmatic VM admission closed".into())
                    })
            },
            &request.cancellation,
            deadline,
            "programmatic VM admission exceeded run deadline",
            None,
        )
        .await?;

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

        let mut limits = config.limits;
        limits.max_program_bytes = limits
            .max_program_bytes
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
        let terminal = async {
            let mut generation_messages = initial_messages(&request);
            generation_messages.push(Message::system(PROGRAM_PROMPT));
            ensure_transcript(&generation_messages, &request.agent.limits)?;

            let mut program_attempt = 0u32;
            let verified = loop {
                let response = self
                    .programmatic_completion(
                        &request,
                        &model,
                        generation_messages.clone(),
                        Some(&scope),
                        deadline,
                        &mut model_calls,
                        &mut events,
                    )
                    .await?;
                if program_attempt == 0 {
                    planning_calls += 1;
                } else {
                    repair_calls += 1;
                }
                let source = response.final_output.as_deref().ok_or_else(|| {
                    HarnessError::InvalidOutput("provider returned no program".into())
                });
                let compiled = source.and_then(|source| {
                    Program::from_json(source.as_bytes(), &limits)
                        .and_then(|program| program.compile(&limits))
                        .map_err(sandbox_error)
                });
                match compiled {
                    Ok(program) => break program,
                    Err(_error) if program_attempt == 0 && model_calls + 1 < request.agent.limits.max_model_calls => {
                        program_attempt = 1;
                        generation_messages.push(Message::assistant(response.final_output.unwrap_or_default()));
                        generation_messages.push(Message::system(REPAIR_PROMPT));
                        ensure_transcript(&generation_messages, &request.agent.limits)?;
                    }
                    Err(error) => return Err(error),
                }
            };

            let execution_number = execution_id(&run_id);
            let mut vm = Execution::with_attempt(verified, ExecutionId(execution_number), program_attempt)
                .map_err(sandbox_error)?;
            let program_output = loop {
                check_stopped(&request.cancellation, deadline, "programmatic run deadline reached")?;
                match vm.step(limits.max_slice_fuel).map_err(sandbox_error)? {
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
                                deadline,
                                &mut dispatched,
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

            let output_json = serde_json::to_string(&program_output).map_err(|_| {
                HarnessError::InvalidOutput("program output could not be serialized".into())
            })?;
            let mut synthesis_messages = initial_messages(&request);
            synthesis_messages.push(Message::system(
                "A verified program completed. Produce the final answer using only the inert JSON result in the next user message.",
            ));
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
                    &mut events,
                )
                .await?;
            synthesis_calls += 1;
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
            result.status = RunStatus::Completed;
            result.final_output = Some(output);
            Ok::<(), HarnessError>(())
        }
        .await;

        drop(admission);
        if let Err(error) = terminal {
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
        events: &mut EventEmitter,
    ) -> Result<ModelResponse, HarnessError> {
        if *model_calls >= request.agent.limits.max_model_calls {
            return Err(HarnessError::ResourceLimit(
                "model call limit reached".into(),
            ));
        }
        *model_calls += 1;
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
        deadline: Option<Instant>,
        dispatched: &mut bool,
    ) -> Result<Vec<ToolResponse>, HarnessError> {
        let mut prepared: Vec<(usize, Box<PreparedCall>)> = Vec::new();
        let mut responses: Vec<Option<ToolResponse>> = vec![None; batch.calls().len()];
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
                    request_call.execution_id.0,
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
                    deadline,
                )
                .await?
            {
                PrepareOutcome::Ready(call) => prepared.push((index, call)),
                PrepareOutcome::Rejected(_) | PrepareOutcome::Stop => {
                    responses[index] = Some(ToolResponse::failure(request_call));
                }
                PrepareOutcome::Reused(value) => {
                    responses[index] = Some(tool_response(request_call, value.as_ref()));
                }
            }
        }

        if batch.requests_read_only_fan_out()
            && prepared.iter().any(|(_, call)| {
                !call.tool.definition().read_only || !call.tool.definition().parallel_safe
            })
        {
            return Err(HarnessError::InvalidTool(
                "programmatic fan-out requires read-only, parallel-safe tools".into(),
            ));
        }
        if batch.requests_read_only_fan_out()
            && prepared.len()
                > self
                    .programmatic
                    .as_ref()
                    .map_or(0, |config| config.max_fanout_concurrency)
        {
            return Err(HarnessError::ResourceLimit(
                "programmatic fan-out concurrency limit exceeded".into(),
            ));
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
                    continue;
                }
            };
            events.emit(RunEvent::ToolCompleted {
                call_id: call.call.id.clone(),
                tool_id: call.call.tool_id.clone(),
                ok: true,
            });
            broker.record_execution(state, call, &execution);
            responses[*index] = Some(tool_response(request_call, execution.result.as_ref()));
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
) -> ToolResponse {
    if result.ok {
        ToolResponse::success(request, result.output.clone())
    } else {
        ToolResponse::failure(request)
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

fn execution_id(run_id: &str) -> u64 {
    let mut value = 0xcbf29ce484222325u64;
    for byte in run_id.bytes() {
        value ^= u64::from(byte);
        value = value.wrapping_mul(0x100000001b3);
    }
    value
}
