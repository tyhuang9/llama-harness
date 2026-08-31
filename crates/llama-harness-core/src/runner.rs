use crate::{
    agent::{RunRequest, RunResult, RunStatus},
    broker::{BrokerState, PrepareOutcome, ToolBroker, ToolConcurrencyLimiter},
    discovery::{ToolScope, ToolScopeSelection},
    event::{EventEmitter, EventSink, InMemoryEventSink, RunEvent},
    limits::{compile_trusted_schema, ensure_json_depth, serialized_len, AgentLimits},
    message::Message,
    model::{ModelProvider, ModelRequest, ModelResponse},
    policy::{ApprovalHandler, DenyApproval, PolicyEngine, SafeDefaultPolicy},
    tool::{ToolCall, ToolCaller, ToolRegistry, ToolResult},
    GenerationOptions, HarnessError, RunError, ToolDiscoveryLimits,
};
use jsonschema::Validator;
use serde_json::Value;
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
    #[cfg(feature = "programmatic")]
    pub(crate) programmatic: Option<crate::ProgrammaticHostConfig>,
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
    #[cfg(feature = "programmatic")]
    programmatic: Option<crate::ProgrammaticHostConfig>,
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
            #[cfg(feature = "programmatic")]
            programmatic: None,
        }
    }

    pub(crate) async fn run_direct(
        &self,
        request: RunRequest,
        strategy_events: Option<DirectStrategyEvents>,
        preflight: RunPreflight,
    ) -> Result<RunResult, HarnessError> {
        let selection = self.tools.select_scope_for_run(
            &request.input,
            &request.agent.tool_allowlist,
            ToolCaller::Direct,
            self.discovery_limits,
            &self.provider.capabilities().limits,
            &request.cancellation,
            preflight.deadline,
        );
        let (tool_scope, discovery) = match selection {
            Ok(ToolScopeSelection::Selected(scope, stats)) => (scope, stats),
            Ok(ToolScopeSelection::LimitReached(stats)) => {
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
        };
        let started = preflight.started;
        let deadline = preflight.deadline;
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
        let mut result = RunResult {
            id: run_id.clone(),
            status: RunStatus::Failed,
            final_output: None,
            model: model.clone(),
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
        let mut events =
            EventEmitter::new(run_id.clone(), trace_id.clone(), Arc::clone(&self.events));
        events.emit(RunEvent::Started { run_id, trace_id });
        if let Some((caller, prior)) = strategy_events.and_then(|events| events.prior_discovery) {
            emit_discovery(&mut events, caller, prior);
        }
        emit_discovery(&mut events, ToolCaller::Direct, discovery);
        if let Some(strategy_events) = strategy_events {
            if let Some(reason) = strategy_events.fallback {
                events.emit(RunEvent::StrategyFallback {
                    from: strategy_events.requested,
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

        let mut messages = initial_messages(&request);
        let mut model_calls = 0;
        let mut broker_state = BrokerState::default();
        let mut output_repairs = 0;
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
            let response = loop {
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
                let completion = self.provider.complete(ModelRequest {
                    model: model.clone(),
                    messages: messages.clone(),
                    tools: tool_scope.definitions().to_vec(),
                    prepared_tools: tool_scope.prepared(),
                    generation: merge_generation(
                        &request.agent.generation,
                        &request.overrides.generation,
                    ),
                    metadata: request.metadata.clone(),
                    cancellation: call_cancellation.clone(),
                });

                match await_guarded(
                    completion,
                    &request.cancellation,
                    call_deadline,
                    "provider call deadline reached",
                    Some(&call_cancellation),
                )
                .await
                {
                    Ok(response) => {
                        events.emit(RunEvent::ModelResponded {
                            call_number: model_calls,
                        });
                        break response;
                    }
                    Err(HarnessError::RetryableProvider(reason))
                        if provider_retries < request.agent.limits.max_provider_retries =>
                    {
                        provider_retries += 1;
                        events.emit(RunEvent::ModelRetrying {
                            next_call_number: model_calls.saturating_add(1),
                            reason,
                        });
                    }
                    Err(error) => {
                        apply_terminal_error(&mut result, error);
                        break 'run;
                    }
                }
            };

            if let Err(error) = validate_model_response(&response, &request.agent.limits) {
                apply_terminal_error(&mut result, error);
                break;
            }

            if let Some(output) = response.final_output {
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

            for call in response.tool_calls {
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
                                execution
                                    .result
                                    .error
                                    .clone()
                                    .unwrap_or_else(|| "tool returned a failure result".into()),
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
                from: strategy_events.requested,
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
    pub(crate) prior_discovery: Option<(ToolCaller, crate::discovery::ToolDiscoveryStats)>,
}

impl AgentRunnerBuilder {
    #[cfg(feature = "programmatic")]
    /// Explicitly opts this host into bounded programmatic execution.
    pub fn programmatic(mut self, config: crate::ProgrammaticHostConfig) -> Self {
        self.programmatic = Some(config);
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
                .map_or(1, |config| config.max_active_vms),
        ));
        AgentRunner {
            provider: self.provider,
            tools: self.tools,
            policy: self.policy,
            approvals: self.approvals,
            events: self.events,
            concurrency: self.concurrency,
            discovery_limits: self.discovery_limits,
            #[cfg(feature = "programmatic")]
            programmatic: self.programmatic,
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
    {
        return Err(HarnessError::InvalidRequest(
            "call, byte, transcript, and depth limits must be greater than zero".into(),
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
            compile_trusted_schema(schema, |error| {
                HarnessError::InvalidRequest(format!("invalid output schema: {error}"))
            })
        })
        .transpose()
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
