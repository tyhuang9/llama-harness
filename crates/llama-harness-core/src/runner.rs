use crate::{
    agent::{RunRequest, RunResult, RunStatus},
    broker::{
        BrokerState, PrepareOutcome, SpeculativeCommitOutcome, ToolBroker, ToolConcurrencyLimiter,
    },
    discovery::{ToolScope, ToolScopeSelection},
    event::{EventEmitter, EventSink, InMemoryEventSink, RunEvent},
    limits::{compile_trusted_schema, ensure_json_depth, serialized_len, AgentLimits},
    message::Message,
    model::{
        ModelCapabilities, ModelProvider, ModelRequest, ModelResponse, StructuredOutputRequest,
    },
    policy::{ApprovalHandler, DenyApproval, PolicyEngine, SafeDefaultPolicy},
    tool::{ToolCall, ToolCaller, ToolRegistry, ToolResult},
    GenerationOptions, HarnessError, RunError, ToolDiscoveryLimits,
};
use jsonschema::Validator;
use serde_json::Value;
#[cfg(feature = "programmatic")]
use std::collections::BTreeSet;
use std::{
    future::Future,
    sync::Arc,
    time::{Duration, Instant as StdInstant},
};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Executes agent runs against a model provider and registered tools.
pub struct AgentRunner {
    pub(crate) provider: Arc<dyn ModelProvider>,
    pub(crate) tools: ToolRegistry,
    pub(crate) policy: Arc<dyn PolicyEngine>,
    pub(crate) approvals: Arc<dyn ApprovalHandler>,
    pub(crate) events: Arc<dyn EventSink>,
    pub(crate) concurrency: Arc<ToolConcurrencyLimiter>,
    pub(crate) discovery_limits: ToolDiscoveryLimits,
    pub(crate) speculation: Option<Arc<crate::speculation::SpeculationController>>,
    #[cfg(feature = "programmatic")]
    pub(crate) programmatic: Option<crate::ProgrammaticHostConfig>,
    #[cfg(feature = "programmatic")]
    pub(crate) adaptive_programmatic_allowlist: BTreeSet<crate::ProgrammaticWorkloadClass>,
    #[cfg(feature = "programmatic")]
    pub(crate) programmatic_admission: Arc<tokio::sync::Semaphore>,
}

/// Configures an [`AgentRunner`] and its policy, approval, tool, and event integrations.
pub struct AgentRunnerBuilder {
    provider: Arc<dyn ModelProvider>,
    tools: ToolRegistry,
    policy: Arc<dyn PolicyEngine>,
    approvals: Arc<dyn ApprovalHandler>,
    events: Arc<dyn EventSink>,
    concurrency: Arc<ToolConcurrencyLimiter>,
    discovery_limits: ToolDiscoveryLimits,
    speculation: Option<crate::SpeculationConfig>,
    #[cfg(feature = "programmatic")]
    programmatic: Option<crate::ProgrammaticHostConfig>,
    #[cfg(feature = "programmatic")]
    adaptive_programmatic_allowlist: BTreeSet<crate::ProgrammaticWorkloadClass>,
}

pub(crate) struct RunPreflight {
    pub(crate) output_validator: Option<Validator>,
    pub(crate) deadline: Option<Instant>,
    pub(crate) started: StdInstant,
}

impl AgentRunner {
    /// Starts building a runner with conservative policy and in-memory event defaults.
    pub fn builder(provider: Arc<dyn ModelProvider>) -> AgentRunnerBuilder {
        AgentRunnerBuilder {
            provider,
            tools: ToolRegistry::default(),
            policy: Arc::new(SafeDefaultPolicy),
            approvals: Arc::new(DenyApproval),
            events: Arc::new(InMemoryEventSink::default()),
            concurrency: Arc::new(ToolConcurrencyLimiter::default()),
            discovery_limits: ToolDiscoveryLimits::default(),
            speculation: None,
            #[cfg(feature = "programmatic")]
            programmatic: None,
            #[cfg(feature = "programmatic")]
            adaptive_programmatic_allowlist: BTreeSet::new(),
        }
    }

    /// Returns pull-only readiness evidence for one registered tool.
    pub fn speculation_readiness(&self, tool_id: &str) -> crate::SpeculationReadiness {
        self.speculation.as_ref().map_or_else(
            || crate::SpeculationReadiness::disabled(tool_id),
            |controller| controller.readiness(tool_id),
        )
    }

    /// Returns pull-only metadata counters for one registered tool.
    pub fn speculation_metrics(&self, tool_id: &str) -> crate::SpeculationMetrics {
        self.speculation.as_ref().map_or_else(
            || crate::SpeculationMetrics::disabled(tool_id),
            |controller| controller.metrics(tool_id),
        )
    }

    /// Explicitly activates one ready tool without affecting other tools.
    pub fn activate_speculation(&self, tool_id: &str) -> crate::SpeculationReadiness {
        self.speculation.as_ref().map_or_else(
            || crate::SpeculationReadiness::disabled(tool_id),
            |controller| controller.activate(tool_id),
        )
    }

    /// Immediately returns one tool to Shadow and clears its readiness streak.
    pub fn return_speculation_to_shadow(&self, tool_id: &str) -> crate::SpeculationReadiness {
        self.speculation.as_ref().map_or_else(
            || crate::SpeculationReadiness::disabled(tool_id),
            |controller| controller.return_to_shadow(tool_id),
        )
    }

    pub(crate) async fn run_direct(
        &self,
        request: RunRequest,
        strategy_events: Option<DirectStrategyEvents>,
        preflight: RunPreflight,
        speculation_eligibility: SpeculationEligibility,
    ) -> Result<RunResult, HarnessError> {
        self.run_direct_internal(
            request,
            strategy_events,
            preflight,
            None,
            speculation_eligibility,
        )
        .await
    }

    /// Continues a pre-effect strategy fallback in the existing run identity,
    /// event sequence, deadline, model budget, and broker state.
    #[cfg_attr(not(feature = "programmatic"), allow(dead_code))]
    pub(crate) async fn run_direct_continuation(
        &self,
        request: RunRequest,
        strategy_events: DirectStrategyEvents,
        preflight: RunPreflight,
        continuation: DirectContinuation,
    ) -> Result<RunResult, HarnessError> {
        self.run_direct_internal(
            request,
            Some(strategy_events),
            preflight,
            Some(continuation),
            SpeculationEligibility::SequentialOnly,
        )
        .await
    }

    async fn run_direct_internal(
        &self,
        request: RunRequest,
        strategy_events: Option<DirectStrategyEvents>,
        preflight: RunPreflight,
        mut continuation: Option<DirectContinuation>,
        mut speculation_eligibility: SpeculationEligibility,
    ) -> Result<RunResult, HarnessError> {
        let capabilities = self.provider.capabilities();
        let prepared_direct_scope = continuation
            .as_mut()
            .and_then(|continuation| continuation.prepared_direct_scope.take());
        let (tool_scope, discovery) = if let Some(scope) = prepared_direct_scope {
            (scope, None)
        } else {
            let selection = self.tools.select_scope_for_run(
                &request.input,
                &request.agent.tool_allowlist,
                ToolCaller::Direct,
                self.discovery_limits,
                &capabilities.limits,
                &request.cancellation,
                preflight.deadline,
            );
            match selection {
                Ok(ToolScopeSelection::Selected(scope, stats)) => (scope, Some(stats)),
                Ok(ToolScopeSelection::LimitReached(stats)) => {
                    if let Some(mut continuation) = continuation {
                        emit_discovery(&mut continuation.events, ToolCaller::Direct, stats);
                        apply_terminal_error(
                            &mut continuation.result,
                            HarnessError::ResourceLimit("tool discovery budget reached".into()),
                        );
                        continuation.result.duration_ms =
                            preflight.started.elapsed().as_millis() as u64;
                        return Ok(finish_programmatic_continuation(continuation));
                    }
                    let mut completed_scopes = Vec::with_capacity(2);
                    if let Some((caller, prior)) =
                        strategy_events.and_then(|events| events.prior_discovery)
                    {
                        completed_scopes.push((caller, prior));
                    }
                    completed_scopes.push((ToolCaller::Direct, stats));
                    return Ok(discovery_limit_terminal_result_with_scopes(
                        &request,
                        &completed_scopes,
                        &self.events,
                        crate::RunStrategy::Direct,
                        preflight.started,
                        strategy_events,
                    ));
                }
                Err(error @ (HarnessError::Cancelled | HarnessError::TimedOut(_))) => {
                    if let Some(mut continuation) = continuation {
                        apply_terminal_error(&mut continuation.result, error);
                        continuation.result.duration_ms =
                            preflight.started.elapsed().as_millis() as u64;
                        return Ok(finish_programmatic_continuation(continuation));
                    }
                    let prior = strategy_events
                        .and_then(|events| events.prior_discovery)
                        .into_iter()
                        .collect::<Vec<_>>();
                    return Ok(pre_event_terminal_result_with_scopes(
                        &request,
                        error,
                        &self.events,
                        crate::RunStrategy::Direct,
                        preflight.started,
                        &prior,
                    ));
                }
                Err(error) => return Err(error),
            }
        };
        let started = preflight.started;
        let deadline = preflight.deadline;
        let (mut result, mut events, mut model_calls, mut broker_state, continuation_usage) =
            match continuation {
                Some(continuation) => {
                    let DirectContinuation {
                        result,
                        mut events,
                        broker_state,
                        model_calls,
                        usage,
                        prepared_direct_scope: _,
                    } = continuation;
                    if let Some(discovery) = discovery {
                        emit_discovery(&mut events, ToolCaller::Direct, discovery);
                    }
                    if let Some(strategy_events) = strategy_events {
                        if let Some(reason) = strategy_events.fallback {
                            events.emit(RunEvent::StrategyFallback {
                                from: strategy_events
                                    .fallback_from
                                    .unwrap_or(strategy_events.requested),
                                to: crate::RunStrategy::Direct,
                                reason,
                            });
                        }
                        events.emit(RunEvent::StrategySelected {
                            requested: strategy_events.requested,
                            selected: crate::RunStrategy::Direct,
                            reason: strategy_events.reason,
                        });
                    }
                    (result, events, model_calls, broker_state, Some(usage))
                }
                None => {
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
                    let result = RunResult {
                        id: run_id.clone(),
                        status: RunStatus::Failed,
                        final_output: None,
                        model,
                        tool_calls: vec![],
                        policy_decisions: vec![],
                        approvals: vec![],
                        errors: vec![],
                        duration_ms: 0,
                        trace_id: trace_id.clone(),
                        model_call_limit_reached: false,
                        tool_call_limit_reached: false,
                        repeated_tool_call_limit_reached: false,
                        cancelled: false,
                    };
                    let mut events = EventEmitter::new(
                        run_id.clone(),
                        trace_id.clone(),
                        Arc::clone(&self.events),
                    );
                    events.emit(RunEvent::Started { run_id, trace_id });
                    if let Some((caller, prior)) =
                        strategy_events.and_then(|events| events.prior_discovery)
                    {
                        emit_discovery(&mut events, caller, prior);
                    }
                    if let Some(discovery) = discovery {
                        emit_discovery(&mut events, ToolCaller::Direct, discovery);
                    }
                    if let Some(strategy_events) = strategy_events {
                        if let Some(reason) = strategy_events.fallback {
                            events.emit(RunEvent::StrategyFallback {
                                from: strategy_events
                                    .fallback_from
                                    .unwrap_or(strategy_events.requested),
                                to: crate::RunStrategy::Direct,
                                reason,
                            });
                        }
                        events.emit(RunEvent::StrategySelected {
                            requested: strategy_events.requested,
                            selected: crate::RunStrategy::Direct,
                            reason: strategy_events.reason,
                        });
                    }
                    (result, events, 0, BrokerState::default(), None)
                }
            };

        let model = result.model.clone();
        let mut messages = initial_messages(&request);
        let mut output_repairs = 0;
        let structured_output =
            agent_structured_output(&capabilities, request.agent.output_schema.as_ref());
        let broker = ToolBroker::new(
            &self.tools,
            &tool_scope,
            &self.policy,
            &self.approvals,
            &self.concurrency,
        );

        'run: loop {
            if let Err(error) =
                check_stopped(&request.cancellation, deadline, "run deadline reached")
            {
                apply_terminal_error(&mut result, error);
                break;
            }

            let mut provider_retries = 0;
            let (response, mut speculative) = loop {
                if model_calls >= request.agent.limits.max_model_calls {
                    result.status = RunStatus::LimitReached;
                    result.model_call_limit_reached = true;
                    result.errors.push(RunError {
                        code: "model_call_limit".into(),
                        message: "model call limit reached".into(),
                    });
                    break 'run;
                }
                if let Err(error) =
                    check_stopped(&request.cancellation, deadline, "run deadline reached")
                {
                    apply_terminal_error(&mut result, error);
                    break 'run;
                }

                model_calls += 1;
                events.emit(RunEvent::ModelRequested {
                    call_number: model_calls,
                    model: model.clone(),
                });
                let call_cancellation = request.cancellation.child_token();
                let call_deadline = match provider_deadline(
                    deadline,
                    request.agent.limits.max_model_call_duration_ms,
                ) {
                    Ok(deadline) => deadline,
                    Err(error) => {
                        apply_terminal_error(&mut result, error);
                        break 'run;
                    }
                };
                let model_request = ModelRequest {
                    model: model.clone(),
                    messages: messages.clone(),
                    tools: tool_scope.definitions().to_vec(),
                    prepared_tools: tool_scope.prepared(),
                    generation: merge_generation(
                        &request.agent.generation,
                        &request.overrides.generation,
                    ),
                    structured_output: structured_output.clone(),
                    metadata: request.metadata.clone(),
                    cancellation: call_cancellation.clone(),
                };
                let speculate_this_attempt = speculation_eligibility.begin_provider_attempt();
                let completion = if let Some(controller) = self.speculation.as_ref().filter(|_| {
                    speculate_this_attempt && self.provider.capabilities().supports_streaming
                }) {
                    crate::speculation::stream_reactive_response(
                        &self.provider,
                        &broker,
                        controller,
                        &request,
                        model_request,
                        &result.id,
                        &result.trace_id,
                        model_calls,
                        call_deadline,
                    )
                    .await
                    .map(|streamed| (streamed.response, streamed.speculative))
                } else {
                    await_guarded(
                        self.provider.complete(model_request),
                        &request.cancellation,
                        call_deadline,
                        "provider call deadline reached",
                        Some(&call_cancellation),
                    )
                    .await
                    .map(|response| (response, None))
                };

                match completion {
                    Ok(response) => {
                        events.emit(RunEvent::ModelResponded {
                            call_number: model_calls,
                        });
                        break response;
                    }
                    Err(HarnessError::RetryableProvider(_))
                        if provider_retries < request.agent.limits.max_provider_retries =>
                    {
                        provider_retries += 1;
                        events.emit(RunEvent::ModelRetrying {
                            next_call_number: model_calls.saturating_add(1),
                            reason: "retryable provider failure".into(),
                        });
                    }
                    Err(error) => {
                        apply_terminal_error(&mut result, error);
                        break 'run;
                    }
                }
            };

            if let Err(error) = validate_model_response(&response, &request.agent.limits) {
                if let (Some(controller), Some(ready)) =
                    (self.speculation.as_ref(), speculative.take())
                {
                    crate::speculation::discard_ready_commit(controller, ready, false);
                }
                apply_terminal_error(&mut result, error);
                break;
            }

            if let Some(output) = response.final_output {
                if let (Some(controller), Some(ready)) =
                    (self.speculation.as_ref(), speculative.take())
                {
                    crate::speculation::discard_ready_commit(controller, ready, false);
                }
                if output.trim().is_empty() {
                    result.errors.push(RunError {
                        code: "empty_model_response".into(),
                        message: "model returned an empty final output".into(),
                    });
                    break;
                }
                match validate_output(
                    preflight.output_validator.as_ref(),
                    &output,
                    request.agent.limits.max_json_depth,
                ) {
                    Ok(()) => {
                        result.status = RunStatus::Completed;
                        result.final_output = Some(output);
                        break;
                    }
                    Err(error @ HarnessError::ResourceLimit(_)) => {
                        apply_terminal_error(&mut result, error);
                        break;
                    }
                    Err(error) if output_repairs >= request.agent.limits.max_output_repairs => {
                        result.errors.push(error.run_error());
                        break;
                    }
                    Err(_) => {
                        output_repairs += 1;
                        messages.push(Message::assistant(output));
                        messages.push(Message::system(
                            "Return only JSON that satisfies the requested output schema.",
                        ));
                        if let Err(error) = ensure_transcript(&messages, &request.agent.limits) {
                            apply_terminal_error(&mut result, error);
                            break;
                        }
                        continue;
                    }
                }
            }

            if response.tool_calls.is_empty() {
                result.errors.push(RunError {
                    code: "empty_model_response".into(),
                    message: "model returned neither final output nor tool calls".into(),
                });
                break;
            }

            let recorded_calls =
                self.tool_calls_for_transcript(&request, &tool_scope, &response.tool_calls);
            messages.push(Message::assistant_tool_calls(recorded_calls.clone()));
            if let Err(error) = ensure_transcript(&messages, &request.agent.limits) {
                apply_terminal_error(&mut result, error);
                break;
            }

            for (call_index, call) in response.tool_calls.into_iter().enumerate() {
                if call_index == 0 {
                    if let (Some(controller), Some(mut ready)) =
                        (self.speculation.as_ref(), speculative.take())
                    {
                        match broker
                            .commit_speculative(
                                controller,
                                &request,
                                &mut result,
                                &mut events,
                                &mut broker_state,
                                &mut ready.execution,
                                &call,
                                model_calls,
                                ready.deadline,
                                deadline,
                            )
                            .await
                        {
                            Ok(SpeculativeCommitOutcome::Committed(tool_result)) => {
                                if let Err(error) = push_tool_message(
                                    &mut messages,
                                    &call,
                                    &tool_result,
                                    &request.agent.limits,
                                ) {
                                    apply_terminal_error(&mut result, error);
                                    break 'run;
                                }
                                continue;
                            }
                            Ok(SpeculativeCommitOutcome::ExecuteDirect(prepared)) => {
                                crate::speculation::discard_ready_commit(controller, ready, false);
                                broker.mark_dispatched(&mut broker_state, &prepared);
                                let execution =
                                    match broker.execute(&prepared, &request, deadline).await {
                                        Ok(execution) => execution,
                                        Err(error) => {
                                            broker_state.record_execution_error(&error);
                                            broker.mark_uncertain(&mut broker_state, &prepared);
                                            events.emit(RunEvent::ToolCompleted {
                                                call_id: call.id.clone(),
                                                tool_id: call.tool_id.clone(),
                                                ok: false,
                                            });
                                            apply_terminal_error(&mut result, error);
                                            break 'run;
                                        }
                                    };
                                events.emit(RunEvent::ToolCompleted {
                                    call_id: call.id.clone(),
                                    tool_id: call.tool_id.clone(),
                                    ok: execution.result.ok,
                                });
                                broker.record_execution(&mut broker_state, &prepared, &execution);
                                if let Some(error) = execution.validation_error {
                                    result.errors.push(error);
                                } else if !execution.result.ok {
                                    result.errors.push(RunError::new(
                                        "tool_error",
                                        "tool returned a failure result",
                                    ));
                                }
                                if let Err(error) = push_tool_message(
                                    &mut messages,
                                    &call,
                                    &execution.result,
                                    &request.agent.limits,
                                ) {
                                    apply_terminal_error(&mut result, error);
                                    break 'run;
                                }
                                continue;
                            }
                            Ok(SpeculativeCommitOutcome::Resolved(tool_result)) => {
                                crate::speculation::discard_ready_commit(controller, ready, false);
                                if let Err(error) = push_tool_message(
                                    &mut messages,
                                    &call,
                                    &tool_result,
                                    &request.agent.limits,
                                ) {
                                    apply_terminal_error(&mut result, error);
                                    break 'run;
                                }
                                continue;
                            }
                            Ok(SpeculativeCommitOutcome::Stop) => {
                                crate::speculation::discard_ready_commit(controller, ready, false);
                                break 'run;
                            }
                            Ok(SpeculativeCommitOutcome::NotCommitted) => {
                                crate::speculation::discard_ready_commit(controller, ready, false);
                            }
                            Ok(SpeculativeCommitOutcome::DirectError(error)) => {
                                crate::speculation::discard_ready_commit(
                                    controller,
                                    ready,
                                    matches!(
                                        error,
                                        HarnessError::Cancelled | HarnessError::TimedOut(_)
                                    ),
                                );
                                apply_terminal_error(&mut result, error);
                                break 'run;
                            }
                            Err(error) => {
                                let cancelled = matches!(
                                    error,
                                    HarnessError::Cancelled | HarnessError::TimedOut(_)
                                );
                                crate::speculation::discard_ready_commit(
                                    controller, ready, cancelled,
                                );
                                if cancelled {
                                    apply_terminal_error(&mut result, error);
                                    break 'run;
                                }
                            }
                        }
                    }
                }
                let attempts_before = broker_state.tool_calls;
                let classified_before = broker_state.classified_tool_calls();
                let outcome = match broker
                    .prepare(
                        &request,
                        &mut result,
                        &mut events,
                        &mut broker_state,
                        call.clone(),
                        ToolCaller::Direct,
                        false,
                        false,
                        None,
                        deadline,
                    )
                    .await
                {
                    Ok(outcome) => outcome,
                    Err(error) => {
                        if broker_state.tool_calls > attempts_before
                            && broker_state.classified_tool_calls() == classified_before
                        {
                            broker_state.record_pre_dispatch_error(&error);
                        }
                        apply_terminal_error(&mut result, error);
                        break 'run;
                    }
                };
                match outcome {
                    PrepareOutcome::Ready(prepared) => {
                        broker.mark_dispatched(&mut broker_state, &prepared);
                        let execution = match broker.execute(&prepared, &request, deadline).await {
                            Ok(execution) => execution,
                            Err(error) => {
                                broker_state.record_execution_error(&error);
                                broker.mark_uncertain(&mut broker_state, &prepared);
                                events.emit(RunEvent::ToolCompleted {
                                    call_id: call.id.clone(),
                                    tool_id: call.tool_id.clone(),
                                    ok: false,
                                });
                                apply_terminal_error(&mut result, error);
                                break 'run;
                            }
                        };
                        events.emit(RunEvent::ToolCompleted {
                            call_id: call.id.clone(),
                            tool_id: call.tool_id.clone(),
                            ok: execution.result.ok,
                        });
                        broker.record_execution(&mut broker_state, &prepared, &execution);
                        if let Some(error) = execution.validation_error {
                            result.errors.push(error);
                        } else if !execution.result.ok {
                            result.errors.push(RunError::new(
                                "tool_error",
                                "tool returned a failure result",
                            ));
                        }
                        if let Err(error) = push_tool_message(
                            &mut messages,
                            &call,
                            &execution.result,
                            &request.agent.limits,
                        ) {
                            apply_terminal_error(&mut result, error);
                            break 'run;
                        }
                    }
                    PrepareOutcome::Rejected(failure) => {
                        if let Err(error) =
                            push_tool_message(&mut messages, &call, &failure, &request.agent.limits)
                        {
                            apply_terminal_error(&mut result, error);
                            break 'run;
                        }
                    }
                    PrepareOutcome::Reused(failure) => {
                        if let Err(error) =
                            push_tool_message(&mut messages, &call, &failure, &request.agent.limits)
                        {
                            apply_terminal_error(&mut result, error);
                            break 'run;
                        }
                    }
                    PrepareOutcome::Stop => break 'run,
                }
            }
        }

        result.duration_ms = started.elapsed().as_millis() as u64;
        if let Some(usage) = continuation_usage {
            let continuation = DirectContinuation {
                result,
                events,
                broker_state,
                model_calls,
                usage,
                prepared_direct_scope: None,
            };
            return Ok(finish_programmatic_continuation(continuation));
        }
        if strategy_events.is_some() {
            broker_state.finalize_usage();
            events.emit(RunEvent::StrategyUsage {
                strategy: crate::RunStrategy::Direct,
                model_calls,
                planning_model_calls: 0,
                repair_model_calls: 0,
                recovery_model_calls: 0,
                final_synthesis_model_calls: 0,
                reactive_model_calls: model_calls,
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
        }
        events.emit(RunEvent::Completed {
            status: result.status.clone(),
        });
        Ok(result)
    }

    pub(crate) fn tool_calls_for_transcript(
        &self,
        request: &RunRequest,
        scope: &ToolScope,
        calls: &[ToolCall],
    ) -> Vec<ToolCall> {
        calls
            .iter()
            .map(|call| {
                let mut transcript_call = call.clone();
                if !self.tool_arguments_are_valid(request, scope, call) {
                    transcript_call.arguments_json = "{}".into();
                }
                transcript_call
            })
            .collect()
    }

    fn tool_arguments_are_valid(
        &self,
        request: &RunRequest,
        scope: &ToolScope,
        call: &ToolCall,
    ) -> bool {
        if call.arguments_json.len() as u64 > request.agent.limits.max_tool_arguments_bytes {
            return false;
        }
        let Ok(arguments) = serde_json::from_str(&call.arguments_json) else {
            return false;
        };
        if ensure_json_depth(
            "tool arguments",
            &arguments,
            request.agent.limits.max_json_depth,
        )
        .is_err()
        {
            return false;
        }
        if !scope.contains(&call.tool_id) {
            return false;
        }
        let Some(tool) = self.tools.get(&call.tool_id) else {
            return false;
        };
        request
            .agent
            .tool_allowlist
            .iter()
            .any(|id| id == &call.tool_id)
            && tool.definition().allows_caller(ToolCaller::Direct)
            && self.tools.validate(&call.tool_id, &arguments).is_ok()
    }
}

/// One-shot execution state controlling whether a provider attempt may enter
/// guarded streaming speculation. Every provider attempt consumes the only
/// eligible state, including an attempt that fails and is retried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpeculationEligibility {
    FirstProviderAttempt,
    SequentialOnly,
}

impl SpeculationEligibility {
    pub(crate) fn begin_provider_attempt(&mut self) -> bool {
        matches!(
            std::mem::replace(self, Self::SequentialOnly),
            Self::FirstProviderAttempt
        )
    }
}

pub(crate) fn discovery_limit_terminal_result(
    request: &RunRequest,
    caller: ToolCaller,
    stats: crate::discovery::ToolDiscoveryStats,
    event_sink: &Arc<dyn EventSink>,
    strategy: crate::RunStrategy,
    started: StdInstant,
    strategy_events: Option<DirectStrategyEvents>,
) -> RunResult {
    discovery_limit_terminal_result_with_scopes(
        request,
        &[(caller, stats)],
        event_sink,
        strategy,
        started,
        strategy_events,
    )
}

pub(crate) fn discovery_limit_terminal_result_with_scopes(
    request: &RunRequest,
    completed_scopes: &[(ToolCaller, crate::discovery::ToolDiscoveryStats)],
    event_sink: &Arc<dyn EventSink>,
    strategy: crate::RunStrategy,
    started: StdInstant,
    strategy_events: Option<DirectStrategyEvents>,
) -> RunResult {
    let mut result = preflight_terminal_result(
        request,
        HarnessError::ResourceLimit("tool discovery budget reached".into()),
    );
    result.duration_ms = started.elapsed().as_millis() as u64;
    let mut events = EventEmitter::new(
        result.id.clone(),
        result.trace_id.clone(),
        Arc::clone(event_sink),
    );
    events.emit(RunEvent::Started {
        run_id: result.id.clone(),
        trace_id: result.trace_id.clone(),
    });
    for (caller, stats) in completed_scopes {
        emit_discovery(&mut events, *caller, *stats);
    }
    if let Some(strategy_events) = strategy_events {
        if let Some(reason) = strategy_events.fallback {
            events.emit(RunEvent::StrategyFallback {
                from: strategy_events
                    .fallback_from
                    .unwrap_or(strategy_events.requested),
                to: strategy,
                reason,
            });
        }
        events.emit(RunEvent::StrategySelected {
            requested: strategy_events.requested,
            selected: strategy,
            reason: strategy_events.reason,
        });
    }
    emit_zero_strategy_usage(&mut events, strategy, result.duration_ms);
    events.emit(RunEvent::Completed {
        status: result.status.clone(),
    });
    result
}

fn emit_zero_strategy_usage(
    events: &mut EventEmitter,
    strategy: crate::RunStrategy,
    duration_ms: u64,
) {
    events.emit(RunEvent::StrategyUsage {
        strategy,
        model_calls: 0,
        planning_model_calls: 0,
        repair_model_calls: 0,
        recovery_model_calls: 0,
        final_synthesis_model_calls: 0,
        reactive_model_calls: 0,
        tool_calls: 0,
        tool_issued: 0,
        tool_reused: 0,
        tool_rejected: 0,
        tool_pre_dispatch_aborted: 0,
        tool_completed: 0,
        tool_failed: 0,
        tool_cancelled: 0,
        duration_ms,
    });
}

#[derive(Clone, Copy)]
pub(crate) struct DirectStrategyEvents {
    pub(crate) requested: crate::RunStrategy,
    pub(crate) reason: crate::StrategySelectionReason,
    pub(crate) fallback: Option<crate::StrategyFallbackReason>,
    pub(crate) fallback_from: Option<crate::RunStrategy>,
    pub(crate) prior_discovery: Option<(ToolCaller, crate::discovery::ToolDiscoveryStats)>,
}

/// Programmatic counters retained while a pre-effect invalid program falls
/// through to the direct compatibility path.
pub(crate) struct ProgrammaticUsage {
    pub(crate) planning_model_calls: u32,
    pub(crate) repair_model_calls: u32,
    pub(crate) final_synthesis_model_calls: u32,
}

/// Mutable state transferred to direct execution without starting another run.
pub(crate) struct DirectContinuation {
    pub(crate) result: RunResult,
    pub(crate) events: EventEmitter,
    pub(crate) broker_state: BrokerState,
    pub(crate) model_calls: u32,
    pub(crate) usage: ProgrammaticUsage,
    pub(crate) prepared_direct_scope: Option<ToolScope>,
}

fn finish_programmatic_continuation(mut continuation: DirectContinuation) -> RunResult {
    continuation.broker_state.finalize_usage();
    let usage = continuation.usage;
    continuation.events.emit(RunEvent::StrategyUsage {
        strategy: crate::RunStrategy::Direct,
        model_calls: continuation.model_calls,
        planning_model_calls: usage.planning_model_calls,
        repair_model_calls: usage.repair_model_calls,
        recovery_model_calls: 0,
        final_synthesis_model_calls: usage.final_synthesis_model_calls,
        reactive_model_calls: continuation
            .model_calls
            .saturating_sub(usage.planning_model_calls)
            .saturating_sub(usage.repair_model_calls)
            .saturating_sub(usage.final_synthesis_model_calls),
        tool_calls: continuation.broker_state.tool_calls,
        tool_issued: continuation.broker_state.tool_issued,
        tool_reused: continuation.broker_state.tool_reused,
        tool_rejected: continuation.broker_state.tool_rejected,
        tool_pre_dispatch_aborted: continuation.broker_state.tool_pre_dispatch_aborted,
        tool_completed: continuation.broker_state.tool_completed,
        tool_failed: continuation.broker_state.tool_failed,
        tool_cancelled: continuation.broker_state.tool_cancelled,
        duration_ms: continuation.result.duration_ms,
    });
    continuation.events.emit(RunEvent::Completed {
        status: continuation.result.status.clone(),
    });
    continuation.result
}

impl AgentRunnerBuilder {
    #[cfg(feature = "programmatic")]
    /// Explicitly opts this host into bounded programmatic execution.
    pub fn programmatic(mut self, config: crate::ProgrammaticHostConfig) -> Self {
        self.programmatic = Some(config);
        self
    }

    #[cfg(feature = "programmatic")]
    /// Allows Adaptive runs to promote selected workload classes to Programmatic execution.
    ///
    /// The default allowlist is empty, preserving the Adaptive planner contract and
    /// behavior from releases that predate Programmatic promotion. Forced
    /// Programmatic runs do not consult this allowlist.
    pub fn adaptive_programmatic_allowlist(
        mut self,
        workload_classes: impl IntoIterator<Item = crate::ProgrammaticWorkloadClass>,
    ) -> Self {
        self.adaptive_programmatic_allowlist
            .extend(workload_classes);
        self
    }
    /// Replaces the tool registry used by the runner.
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    /// Replaces the bounded host limits used for model-facing tool discovery.
    pub fn discovery_limits(mut self, limits: ToolDiscoveryLimits) -> Self {
        self.discovery_limits = limits;
        self
    }

    /// Explicitly opts this runner into guarded shadow-first speculation.
    pub fn speculation(mut self, config: crate::SpeculationConfig) -> Self {
        self.speculation = Some(config);
        self
    }

    /// Replaces the policy engine used for tool decisions.
    pub fn policy(mut self, policy: Arc<dyn PolicyEngine>) -> Self {
        self.policy = policy;
        self
    }

    /// Replaces the approval handler used for approval-gated tools.
    pub fn approvals(mut self, approvals: Arc<dyn ApprovalHandler>) -> Self {
        self.approvals = approvals;
        self
    }

    /// Replaces the event sink that receives run lifecycle events.
    pub fn event_sink(mut self, events: Arc<dyn EventSink>) -> Self {
        self.events = events;
        self
    }

    /// Builds the configured runner.
    pub fn build(self) -> AgentRunner {
        #[cfg(feature = "programmatic")]
        let admission = Arc::new(tokio::sync::Semaphore::new(
            self.programmatic
                .as_ref()
                // Invalid host configuration is rejected when a run is started.
                // Clamp the construction-time permit count so untrusted builder
                // input (including zero or `usize::MAX`) cannot panic first.
                .map_or(1, |config| config.max_active_vms.clamp(1, 16)),
        ));
        AgentRunner {
            provider: self.provider,
            tools: self.tools,
            policy: self.policy,
            approvals: self.approvals,
            events: self.events,
            concurrency: self.concurrency,
            discovery_limits: self.discovery_limits,
            speculation: self
                .speculation
                .map(crate::speculation::SpeculationController::new)
                .map(Arc::new),
            #[cfg(feature = "programmatic")]
            programmatic: self.programmatic,
            #[cfg(feature = "programmatic")]
            adaptive_programmatic_allowlist: self.adaptive_programmatic_allowlist,
            #[cfg(feature = "programmatic")]
            programmatic_admission: admission,
        }
    }
}

pub(crate) fn emit_discovery(
    events: &mut EventEmitter,
    caller: ToolCaller,
    stats: crate::discovery::ToolDiscoveryStats,
) {
    events.emit(RunEvent::ToolDiscoveryCompleted {
        caller,
        outcome: stats.outcome,
        selection: stats.selection,
        candidate_count: stats.candidate_count,
        selected_count: stats.selected_count,
        deferred_candidate_count: stats.deferred_candidate_count,
        effective_tool_count_budget: stats.effective_tool_count_budget,
        effective_schema_byte_budget: stats.effective_schema_byte_budget,
        selected_schema_bytes: stats.selected_schema_bytes,
        expansion_count: stats.expansion_count,
        expansion_limit: stats.expansion_limit,
        catalog_exceeded_budget: stats.catalog_exceeded_budget,
        duration_ms: stats.duration_ms,
    });
}

pub(crate) fn validate_request(request: &RunRequest) -> Result<Option<Validator>, HarnessError> {
    if request.agent.id.trim().is_empty()
        || request.agent.name.trim().is_empty()
        || request.agent.version.trim().is_empty()
        || request.agent.default_model.trim().is_empty()
        || request
            .overrides
            .model
            .as_ref()
            .is_some_and(|model| model.trim().is_empty())
    {
        return Err(HarnessError::InvalidRequest(
            "agent id, name, version, and selected model are required".into(),
        ));
    }
    if request.input.trim().is_empty() {
        return Err(HarnessError::InvalidRequest("input is required".into()));
    }

    let limits = &request.agent.limits;
    if limits.max_model_calls == 0
        || limits.max_tool_calls == 0
        || limits.max_identical_tool_calls == 0
        || limits.max_input_bytes == 0
        || limits.max_request_payload_bytes == 0
        || limits.max_model_response_bytes == 0
        || limits.max_tool_arguments_bytes == 0
        || limits.max_tool_result_bytes == 0
        || limits.max_transcript_bytes == 0
        || limits.max_json_depth == 0
        || limits.max_programmatic_program_bytes == 0
        || limits.max_programmatic_fanout_concurrency == 0
    {
        return Err(HarnessError::InvalidRequest(
            "call, byte, transcript, and depth limits must be greater than zero".into(),
        ));
    }
    if limits.max_programmatic_program_bytes > crate::limits::HARD_MAX_PROGRAMMATIC_PROGRAM_BYTES
        || limits.max_programmatic_fanout_concurrency
            > crate::limits::HARD_MAX_PROGRAMMATIC_FANOUT_CONCURRENCY
    {
        return Err(HarnessError::InvalidRequest(
            "programmatic limits exceed immutable library ceilings".into(),
        ));
    }
    if request.input.len() as u64 > limits.max_input_bytes {
        return Err(HarnessError::InvalidRequest(format!(
            "input exceeds {} bytes",
            limits.max_input_bytes
        )));
    }
    if serialized_len(request)? > limits.max_request_payload_bytes {
        return Err(HarnessError::InvalidRequest(format!(
            "request payload exceeds {} bytes",
            limits.max_request_payload_bytes
        )));
    }
    for (label, value) in [
        (
            "application context",
            Value::Object(request.application_context.clone()),
        ),
        ("request metadata", Value::Object(request.metadata.clone())),
        (
            "evaluation metadata",
            Value::Object(request.evaluation.clone()),
        ),
        (
            "agent metadata",
            Value::Object(request.agent.metadata.clone()),
        ),
    ] {
        ensure_json_depth(label, &value, limits.max_json_depth)
            .map_err(|error| HarnessError::InvalidRequest(error.to_string()))?;
    }
    ensure_transcript(&initial_messages(request), limits)
        .map_err(|error| HarnessError::InvalidRequest(error.to_string()))?;

    request
        .agent
        .output_schema
        .as_ref()
        .map(|schema| {
            ensure_json_depth("output schema", schema, limits.max_json_depth)
                .map_err(|error| HarnessError::InvalidRequest(error.to_string()))?;
            StructuredOutputRequest::new("llama_harness_agent_output", schema.clone(), true)?;
            compile_trusted_schema(schema, |error| {
                HarnessError::InvalidRequest(format!("invalid output schema: {error}"))
            })
        })
        .transpose()
}

pub(crate) fn agent_structured_output(
    capabilities: &ModelCapabilities,
    schema: Option<&Value>,
) -> Option<StructuredOutputRequest> {
    if !capabilities.supports_structured_output {
        return None;
    }
    schema.cloned().map(|schema| {
        StructuredOutputRequest::from_prevalidated("llama_harness_agent_output", schema, true)
    })
}

pub(crate) fn preflight_request(request: &RunRequest) -> Result<RunPreflight, HarnessError> {
    let output_validator = validate_request(request)?;
    let started = StdInstant::now();
    let deadline = absolute_deadline(request.agent.limits.max_run_duration_ms)?;
    check_stopped(&request.cancellation, deadline, "run deadline reached")?;
    Ok(RunPreflight {
        output_validator,
        deadline,
        started,
    })
}

pub(crate) fn preflight_terminal_result(request: &RunRequest, error: HarnessError) -> RunResult {
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
    let mut result = RunResult::new(run_id, RunStatus::Failed, model, trace_id);
    apply_terminal_error(&mut result, error);
    result
}

pub(crate) fn pre_event_terminal_result(
    request: &RunRequest,
    error: HarnessError,
    event_sink: &Arc<dyn EventSink>,
    strategy: crate::RunStrategy,
    started: StdInstant,
) -> RunResult {
    pre_event_terminal_result_with_scopes(request, error, event_sink, strategy, started, &[])
}

pub(crate) fn pre_event_terminal_result_with_scopes(
    request: &RunRequest,
    error: HarnessError,
    event_sink: &Arc<dyn EventSink>,
    strategy: crate::RunStrategy,
    started: StdInstant,
    completed_scopes: &[(ToolCaller, crate::discovery::ToolDiscoveryStats)],
) -> RunResult {
    debug_assert!(matches!(
        error,
        HarnessError::Cancelled | HarnessError::TimedOut(_)
    ));
    let mut result = preflight_terminal_result(request, error);
    result.duration_ms = started.elapsed().as_millis() as u64;
    let mut events = EventEmitter::new(
        result.id.clone(),
        result.trace_id.clone(),
        Arc::clone(event_sink),
    );
    events.emit(RunEvent::Started {
        run_id: result.id.clone(),
        trace_id: result.trace_id.clone(),
    });
    for (caller, stats) in completed_scopes {
        emit_discovery(&mut events, *caller, *stats);
    }
    emit_zero_strategy_usage(&mut events, strategy, result.duration_ms);
    events.emit(RunEvent::Completed {
        status: result.status.clone(),
    });
    result
}

pub(crate) fn validate_model_response(
    response: &ModelResponse,
    limits: &AgentLimits,
) -> Result<(), HarnessError> {
    if serialized_len(response)? > limits.max_model_response_bytes {
        return Err(HarnessError::ResourceLimit(format!(
            "model response exceeds {} bytes",
            limits.max_model_response_bytes
        )));
    }
    Ok(())
}

pub(crate) fn validate_output(
    validator: Option<&Validator>,
    output: &str,
    max_json_depth: u32,
) -> Result<(), HarnessError> {
    let Some(validator) = validator else {
        return Ok(());
    };
    let value = serde_json::from_str(output)
        .map_err(|error| HarnessError::InvalidOutput(format!("output is not JSON: {error}")))?;
    ensure_json_depth("structured output", &value, max_json_depth)?;
    validator
        .validate(&value)
        .map_err(|error| HarnessError::InvalidOutput(error.to_string()))
}

pub(crate) fn initial_messages(request: &RunRequest) -> Vec<Message> {
    let mut messages = vec![];
    if !request.agent.system_instructions.trim().is_empty() {
        messages.push(Message::system(request.agent.system_instructions.clone()));
    }
    messages.extend(request.history.clone());
    messages.push(Message::user(request.input.clone()));
    messages
}

pub(crate) fn ensure_transcript(
    messages: &[Message],
    limits: &AgentLimits,
) -> Result<(), HarnessError> {
    let bytes = messages.iter().map(Message::transcript_bytes).sum::<u64>();
    if bytes > limits.max_transcript_bytes {
        return Err(HarnessError::ResourceLimit(format!(
            "transcript exceeds {} bytes",
            limits.max_transcript_bytes
        )));
    }
    Ok(())
}

pub(crate) fn push_tool_message(
    messages: &mut Vec<Message>,
    call: &ToolCall,
    result: &ToolResult,
    limits: &AgentLimits,
) -> Result<(), HarnessError> {
    let message = Message::tool(call.id.clone(), result).map_err(|error| {
        HarnessError::Tool(format!("tool result serialization failed: {error}"))
    })?;
    messages.push(message);
    ensure_transcript(messages, limits)
}

pub(crate) fn apply_terminal_error(result: &mut RunResult, error: HarnessError) {
    result.status = match error {
        HarnessError::Cancelled => RunStatus::Cancelled,
        HarnessError::ResourceLimit(_) => RunStatus::LimitReached,
        _ => RunStatus::Failed,
    };
    result.cancelled = matches!(error, HarnessError::Cancelled);
    result.errors.push(error.run_error());
}

pub(crate) fn absolute_deadline(duration_ms: Option<u64>) -> Result<Option<Instant>, HarnessError> {
    duration_ms
        .map(|duration_ms| {
            Instant::now()
                .checked_add(Duration::from_millis(duration_ms))
                .ok_or_else(|| HarnessError::InvalidRequest("run duration is too large".into()))
        })
        .transpose()
}

pub(crate) fn provider_deadline(
    run_deadline: Option<Instant>,
    provider_duration_ms: Option<u64>,
) -> Result<Option<Instant>, HarnessError> {
    let provider_deadline = provider_duration_ms
        .map(|duration_ms| {
            Instant::now()
                .checked_add(Duration::from_millis(duration_ms))
                .ok_or_else(|| {
                    HarnessError::InvalidRequest("model call duration is too large".into())
                })
        })
        .transpose()?;
    match (run_deadline, provider_deadline) {
        (Some(run), Some(provider)) => Ok(Some(run.min(provider))),
        (Some(run), None) => Ok(Some(run)),
        (None, provider) => Ok(provider),
    }
}

pub(crate) fn check_stopped(
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
    timeout_message: &str,
) -> Result<(), HarnessError> {
    if cancellation.is_cancelled() {
        return Err(HarnessError::Cancelled);
    }
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return Err(HarnessError::TimedOut(timeout_message.into()));
    }
    Ok(())
}

pub(crate) async fn await_guarded<T, F>(
    future: F,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
    timeout_message: &str,
    child_cancellation: Option<&CancellationToken>,
) -> Result<T, HarnessError>
where
    F: Future<Output = Result<T, HarnessError>>,
{
    tokio::pin!(future);
    let result = if let Some(deadline) = deadline {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err(HarnessError::Cancelled),
            _ = tokio::time::sleep_until(deadline) => {
                Err(HarnessError::TimedOut(timeout_message.into()))
            },
            result = &mut future => result,
        }
    } else {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err(HarnessError::Cancelled),
            result = &mut future => result,
        }
    };
    if matches!(
        result,
        Err(HarnessError::Cancelled | HarnessError::TimedOut(_))
    ) {
        if let Some(child_cancellation) = child_cancellation {
            child_cancellation.cancel();
        }
    }
    result
}

pub(crate) fn merge_generation(
    base: &GenerationOptions,
    override_options: &GenerationOptions,
) -> GenerationOptions {
    GenerationOptions {
        temperature: override_options.temperature.or(base.temperature),
        top_p: override_options.top_p.or(base.top_p),
        max_output_tokens: override_options
            .max_output_tokens
            .or(base.max_output_tokens),
    }
}
