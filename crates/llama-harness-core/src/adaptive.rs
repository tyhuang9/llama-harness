use crate::{
    broker::{BrokerState, PrepareOutcome, PreparedCall, ToolBroker},
    event::EventEmitter,
    plan::MAX_EXECUTION_PLAN_NODES,
    runner::{
        absolute_deadline, apply_terminal_error, ensure_transcript, initial_messages,
        merge_generation, provider_deadline, push_tool_message, validate_model_response,
        validate_output, validate_request, DirectStrategyEvents,
    },
    AgentRunner, ExecutionPlan, HarnessError, Message, ModelRequest, ModelResponse,
    PlanConcurrency, PlanNode, RunError, RunEvent, RunRequest, RunResult, RunStatus, RunStrategy,
    StrategyFallbackReason, StrategySelectionReason, ToolCall, ToolCaller, ToolDefinition,
    ToolResult,
};
use futures_util::future::join_all;
use jsonschema::Validator;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::Arc,
    time::Instant as StdInstant,
};
use tokio::time::Instant;
use uuid::Uuid;

const MAX_SCHEDULER_PARALLELISM: usize = 8;
const PLANNER_PROMPT: &str = "Select the safest efficient tool strategy. Return only one strict JSON object: {\"strategy\":\"direct\"} when no finite safe plan is justified, or {\"strategy\":\"declarative_plan\",\"plan\":{\"nodes\":[...]}} for a finite dependency DAG. Use only the supplied tools. Every plan node requires id, tool_id, and schema-valid arguments. Optional fields are depends_on, result_bindings, concurrency, approval_barrier, and commit_boundary. Choose direct for mutations, approval-sensitive work, ambiguity, or an uncertain next step.";
const REPAIR_PROMPT: &str = "The previous strategy envelope was invalid. Repair it once. Return only the strict JSON envelope requested by the planning instructions, with no prose or markdown.";
const RECOVERY_PROMPT: &str = "Execution stopped after the recorded completed results. Produce one replacement declarative plan using only those results as prior state. Never repeat a completed mutation. Return only {\"strategy\":\"declarative_plan\",\"plan\":{\"nodes\":[...]}}.";

#[derive(Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case", deny_unknown_fields)]
enum PlannerEnvelope {
    Direct,
    DeclarativePlan { plan: ExecutionPlan },
}

struct StrategyRun<'a> {
    runner: &'a AgentRunner,
    request: &'a RunRequest,
    output_validator: Option<Validator>,
    result: RunResult,
    events: EventEmitter,
    messages: Vec<Message>,
    model: String,
    deadline: Option<Instant>,
    started: StdInstant,
    model_calls: u32,
    output_repairs: u32,
    broker_state: BrokerState,
    selected: RunStrategy,
}

struct PreparedNode {
    node: PlanNode,
    prepared: Option<PreparedCall>,
    reused: Option<ToolResult>,
}

struct PlanExecution {
    completed: HashMap<String, ToolResult>,
    transcript: Vec<(ToolCall, ToolResult)>,
    failed: bool,
}

impl AgentRunner {
    /// Executes one run using the quality-first adaptive strategy.
    ///
    /// Providers without structured-plan support transparently retain the direct
    /// compatibility path. Invalid requests return `Err`; failures after a run
    /// starts are captured in a terminal [`RunResult`].
    pub async fn run(&self, request: RunRequest) -> Result<RunResult, HarnessError> {
        self.run_with_strategy(request, RunStrategy::Adaptive).await
    }

    /// Executes one run with an explicit strategy override.
    ///
    /// Forced direct execution preserves the reactive safety-boundary behavior.
    /// Programmatic execution remains unavailable until the optional sandbox is
    /// installed by a later release.
    pub async fn run_with_strategy(
        &self,
        request: RunRequest,
        strategy: RunStrategy,
    ) -> Result<RunResult, HarnessError> {
        match strategy {
            RunStrategy::Direct => self.run_direct(request, None).await,
            RunStrategy::Programmatic => Err(HarnessError::UnsupportedCapability(
                "programmatic execution requires the optional sandbox runtime".into(),
            )),
            RunStrategy::DeclarativePlan => {
                self.ensure_planning_capability(&request)?;
                self.run_planned(request, RunStrategy::DeclarativePlan)
                    .await
            }
            RunStrategy::Adaptive => {
                if self.planning_downgrade_reason(&request).is_some() {
                    self.run_direct(
                        request,
                        Some(DirectStrategyEvents {
                            reason: StrategySelectionReason::CapabilityDowngrade,
                        }),
                    )
                    .await
                } else {
                    self.run_planned(request, RunStrategy::Adaptive).await
                }
            }
        }
    }

    fn ensure_planning_capability(&self, request: &RunRequest) -> Result<(), HarnessError> {
        if let Some(reason) = self.planning_downgrade_reason(request) {
            return Err(HarnessError::UnsupportedCapability(reason.into()));
        }
        Ok(())
    }

    fn planning_downgrade_reason(&self, request: &RunRequest) -> Option<&'static str> {
        let capabilities = self.provider.capabilities();
        if request.agent.limits.max_model_calls < 2 {
            return Some("run model-call budget cannot support planning and finalization");
        }
        if !capabilities.supports_tools || !capabilities.supports_structured_plans {
            return Some("provider does not support structured plans");
        }
        if capabilities.limits.max_plan_nodes == Some(0)
            || capabilities.limits.max_plan_bytes == Some(0)
        {
            return Some("provider advertises no structured-plan capacity");
        }
        let tools = self
            .tools
            .allowed_definitions_for(&request.agent.tool_allowlist, ToolCaller::DeclarativePlan);
        if tools.is_empty() {
            return Some("no allowed tools permit declarative plan calls");
        }
        if capabilities
            .limits
            .max_tools
            .is_some_and(|limit| tools.len() > limit as usize)
        {
            return Some("tool catalog exceeds provider tool-count capacity");
        }
        let schema_bytes = tools.iter().fold(0u64, |total, tool| {
            total.saturating_add(
                serde_json::to_vec(tool).map_or(u64::MAX, |bytes| bytes.len() as u64),
            )
        });
        if capabilities
            .limits
            .max_tool_schema_bytes
            .is_some_and(|limit| schema_bytes > limit)
        {
            return Some("tool catalog exceeds provider schema capacity");
        }
        None
    }

    async fn run_planned(
        &self,
        request: RunRequest,
        requested: RunStrategy,
    ) -> Result<RunResult, HarnessError> {
        let mut run = StrategyRun::new(self, &request, requested)?;
        if requested == RunStrategy::DeclarativePlan {
            run.events.emit(RunEvent::StrategySelected {
                requested,
                selected: RunStrategy::DeclarativePlan,
                reason: StrategySelectionReason::Forced,
            });
        }

        let plan_tools = self
            .tools
            .allowed_definitions_for(&request.agent.tool_allowlist, ToolCaller::DeclarativePlan);
        let mut planner_messages = run.messages.clone();
        planner_messages.insert(0, Message::system(PLANNER_PROMPT));
        let mut repair_used = false;
        let envelope = loop {
            let response = match run
                .complete(planner_messages.clone(), plan_tools.clone())
                .await
            {
                Ok(Some(response)) => response,
                Ok(None) => return Ok(run.finish()),
                Err(error) => {
                    apply_terminal_error(&mut run.result, error);
                    return Ok(run.finish());
                }
            };
            match run.parse_envelope(response, requested) {
                Ok(envelope) => break envelope,
                Err(error) if !repair_used => {
                    repair_used = true;
                    planner_messages.push(Message::system(REPAIR_PROMPT));
                    run.result
                        .errors
                        .retain(|record| record.code != "invalid_plan");
                    let _ = error;
                }
                Err(error) if requested == RunStrategy::Adaptive => {
                    run.events.emit(RunEvent::StrategyFallback {
                        from: RunStrategy::DeclarativePlan,
                        to: RunStrategy::Direct,
                        reason: StrategyFallbackReason::InvalidPlan,
                    });
                    run.events.emit(RunEvent::StrategySelected {
                        requested,
                        selected: RunStrategy::Direct,
                        reason: StrategySelectionReason::PlannerSelectedDirect,
                    });
                    run.selected = RunStrategy::Direct;
                    run.result
                        .errors
                        .retain(|record| record.code != "invalid_plan");
                    let _ = error;
                    run.run_reactive().await;
                    return Ok(run.finish());
                }
                Err(error) => {
                    run.result.errors.push(RunError::new(
                        "invalid_plan",
                        format!("declarative plan remained invalid after repair: {error}"),
                    ));
                    return Ok(run.finish());
                }
            }
        };

        match envelope {
            PlannerEnvelope::Direct => {
                run.events.emit(RunEvent::StrategySelected {
                    requested,
                    selected: RunStrategy::Direct,
                    reason: StrategySelectionReason::PlannerSelectedDirect,
                });
                run.selected = RunStrategy::Direct;
                run.run_reactive().await;
            }
            PlannerEnvelope::DeclarativePlan { mut plan } => {
                if requested == RunStrategy::Adaptive {
                    run.events.emit(RunEvent::StrategySelected {
                        requested,
                        selected: RunStrategy::DeclarativePlan,
                        reason: StrategySelectionReason::PlannerSelectedPlan,
                    });
                }
                run.selected = RunStrategy::DeclarativePlan;
                run.broker_state.enable_effect_reuse();
                let mut execution = run.execute_plan(&plan).await;

                if execution.failed && !repair_used {
                    run.append_plan_transcript(&execution.transcript);
                    let recovery = json!({
                        "completed": execution
                            .completed
                            .iter()
                            .map(|(node_id, result)| (node_id, &result.output))
                            .collect::<HashMap<_, _>>()
                    });
                    let mut recovery_messages = run.messages.clone();
                    recovery_messages.push(Message::system(RECOVERY_PROMPT));
                    recovery_messages.push(Message::user(recovery.to_string()));
                    if ensure_transcript(&recovery_messages, &request.agent.limits).is_ok() {
                        if let Ok(Some(response)) =
                            run.complete(recovery_messages, plan_tools.clone()).await
                        {
                            if let Ok(PlannerEnvelope::DeclarativePlan { plan: repaired }) =
                                run.parse_envelope(response, RunStrategy::DeclarativePlan)
                            {
                                run.events.emit(RunEvent::StrategyFallback {
                                    from: RunStrategy::DeclarativePlan,
                                    to: RunStrategy::DeclarativePlan,
                                    reason: StrategyFallbackReason::ExecutionRecovery,
                                });
                                plan = repaired;
                                execution = run.execute_plan(&plan).await;
                            }
                        }
                    }
                }

                if execution.failed {
                    if run.result.errors.is_empty() {
                        run.result.errors.push(RunError::new(
                            "plan_execution_failed",
                            "declarative plan execution failed",
                        ));
                    }
                } else {
                    run.append_plan_transcript(&execution.transcript);
                    run.run_reactive().await;
                }
            }
        }
        Ok(run.finish())
    }
}

impl<'a> StrategyRun<'a> {
    fn new(
        runner: &'a AgentRunner,
        request: &'a RunRequest,
        requested: RunStrategy,
    ) -> Result<Self, HarnessError> {
        let output_validator = validate_request(request)?;
        let started = StdInstant::now();
        let deadline = absolute_deadline(request.agent.limits.max_run_duration_ms)?;
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
        let mut events =
            EventEmitter::new(run_id.clone(), trace_id.clone(), Arc::clone(&runner.events));
        events.emit(RunEvent::Started {
            run_id: run_id.clone(),
            trace_id: trace_id.clone(),
        });
        Ok(Self {
            runner,
            request,
            output_validator,
            result: RunResult::new(run_id, RunStatus::Failed, model.clone(), trace_id),
            events,
            messages: initial_messages(request),
            model,
            deadline,
            started,
            model_calls: 0,
            output_repairs: 0,
            broker_state: BrokerState::default(),
            selected: requested,
        })
    }

    async fn complete(
        &mut self,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
    ) -> Result<Option<ModelResponse>, HarnessError> {
        let mut provider_retries = 0;
        loop {
            if self.model_calls >= self.request.agent.limits.max_model_calls {
                self.result.status = RunStatus::LimitReached;
                self.result.model_call_limit_reached = true;
                self.result.errors.push(RunError::new(
                    "model_call_limit",
                    "model call limit reached",
                ));
                return Ok(None);
            }
            self.model_calls += 1;
            self.events.emit(RunEvent::ModelRequested {
                call_number: self.model_calls,
                model: self.model.clone(),
            });
            let call_cancellation = self.request.cancellation.child_token();
            let call_deadline = provider_deadline(
                self.deadline,
                self.request.agent.limits.max_model_call_duration_ms,
            )?;
            let completion = self.runner.provider.complete(ModelRequest {
                model: self.model.clone(),
                messages: messages.clone(),
                tools: tools.clone(),
                generation: merge_generation(
                    &self.request.agent.generation,
                    &self.request.overrides.generation,
                ),
                metadata: self.request.metadata.clone(),
                cancellation: call_cancellation.clone(),
            });
            match crate::runner::await_guarded(
                completion,
                &self.request.cancellation,
                call_deadline,
                "provider call deadline reached",
                Some(&call_cancellation),
            )
            .await
            {
                Ok(response) => {
                    validate_model_response(&response, &self.request.agent.limits)?;
                    self.events.emit(RunEvent::ModelResponded {
                        call_number: self.model_calls,
                    });
                    return Ok(Some(response));
                }
                Err(HarnessError::RetryableProvider(reason))
                    if provider_retries < self.request.agent.limits.max_provider_retries =>
                {
                    provider_retries += 1;
                    self.events.emit(RunEvent::ModelRetrying {
                        next_call_number: self.model_calls.saturating_add(1),
                        reason,
                    });
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn parse_envelope(
        &mut self,
        response: ModelResponse,
        requested: RunStrategy,
    ) -> Result<PlannerEnvelope, HarnessError> {
        if !response.tool_calls.is_empty() {
            return Err(HarnessError::InvalidOutput(
                "planner returned tool calls instead of a strategy envelope".into(),
            ));
        }
        let output = response.final_output.ok_or_else(|| {
            HarnessError::InvalidOutput("planner omitted the strategy envelope".into())
        })?;
        let capabilities = self.runner.provider.capabilities();
        if capabilities
            .limits
            .max_plan_bytes
            .is_some_and(|limit| output.len() as u64 > limit)
        {
            return Err(HarnessError::ResourceLimit(
                "planner envelope exceeds provider plan-byte limit".into(),
            ));
        }
        let envelope: PlannerEnvelope = serde_json::from_str(&output).map_err(|_| {
            HarnessError::InvalidOutput("planner returned an invalid strategy envelope".into())
        })?;
        if requested == RunStrategy::DeclarativePlan && matches!(envelope, PlannerEnvelope::Direct)
        {
            return Err(HarnessError::InvalidOutput(
                "forced declarative planning cannot select direct execution".into(),
            ));
        }
        if let PlannerEnvelope::DeclarativePlan { plan } = &envelope {
            self.static_plan_preflight(plan)?;
        }
        Ok(envelope)
    }

    fn static_plan_preflight(&self, plan: &ExecutionPlan) -> Result<(), HarnessError> {
        let capabilities = self.runner.provider.capabilities();
        let provider_nodes = capabilities
            .limits
            .max_plan_nodes
            .map_or(MAX_EXECUTION_PLAN_NODES, |limit| limit as usize);
        let max_nodes = provider_nodes
            .min(self.request.agent.limits.max_tool_calls as usize)
            .min(MAX_EXECUTION_PLAN_NODES);
        if plan.nodes.is_empty() {
            return Err(HarnessError::InvalidRequest(
                "declarative plan must contain at least one node".into(),
            ));
        }
        plan.validate(max_nodes)?;
        if capabilities.limits.max_plan_bytes.is_some_and(|limit| {
            serde_json::to_vec(plan).map_or(true, |bytes| bytes.len() as u64 > limit)
        }) {
            return Err(HarnessError::ResourceLimit(
                "execution plan exceeds provider plan-byte limit".into(),
            ));
        }

        let mut identical = HashMap::<String, u32>::new();
        for node in &plan.nodes {
            let Some(tool) = self.runner.tools.get(&node.tool_id) else {
                return Err(HarnessError::InvalidTool(format!(
                    "plan node '{}' selects an unknown tool",
                    node.id
                )));
            };
            if !self
                .request
                .agent
                .tool_allowlist
                .iter()
                .any(|tool_id| tool_id == &node.tool_id)
                || !tool.definition().allows_caller(ToolCaller::DeclarativePlan)
            {
                return Err(HarnessError::InvalidTool(format!(
                    "plan node '{}' selects a tool not allowed for declarative plans",
                    node.id
                )));
            }
            self.runner.tools.validate(&node.tool_id, &node.arguments)?;
            let signature = crate::broker::canonical_signature(&node.tool_id, &node.arguments);
            let count = identical.entry(signature).or_default();
            *count += 1;
            if *count > self.request.agent.limits.max_identical_tool_calls {
                return Err(HarnessError::ResourceLimit(
                    "execution plan exceeds the repeated-call limit".into(),
                ));
            }
        }
        Ok(())
    }

    async fn preflight_plan(
        &mut self,
        plan: &ExecutionPlan,
    ) -> Result<Option<Vec<PreparedNode>>, HarnessError> {
        self.static_plan_preflight(plan)?;
        let broker = ToolBroker::new(
            &self.runner.tools,
            &self.runner.policy,
            &self.runner.approvals,
        );
        let mut prepared_nodes = Vec::with_capacity(plan.nodes.len());
        for node in &plan.nodes {
            let arguments_json = serde_json::to_string(&node.arguments).map_err(|error| {
                HarnessError::InvalidArguments(format!(
                    "plan node '{}' arguments cannot be serialized: {error}",
                    node.id
                ))
            })?;
            let call = ToolCall::new(
                format!("plan:{}", node.id),
                node.tool_id.clone(),
                arguments_json,
            );
            match broker
                .prepare(
                    self.request,
                    &mut self.result,
                    &mut self.events,
                    &mut self.broker_state,
                    call,
                    ToolCaller::DeclarativePlan,
                    node.approval_barrier,
                    self.deadline,
                )
                .await?
            {
                PrepareOutcome::Ready(prepared) => prepared_nodes.push(PreparedNode {
                    node: node.clone(),
                    prepared: Some(prepared),
                    reused: None,
                }),
                PrepareOutcome::Reused(result) => prepared_nodes.push(PreparedNode {
                    node: node.clone(),
                    prepared: None,
                    reused: Some(result),
                }),
                PrepareOutcome::Rejected(_) => {
                    return Err(HarnessError::InvalidRequest(format!(
                        "plan node '{}' failed execution preflight",
                        node.id
                    )));
                }
                PrepareOutcome::Stop => return Ok(None),
            }
        }
        self.events.emit(RunEvent::PlanValidated {
            node_count: plan.nodes.len() as u32,
        });
        Ok(Some(prepared_nodes))
    }

    async fn execute_plan(&mut self, plan: &ExecutionPlan) -> PlanExecution {
        let mut prepared = match self.preflight_plan(plan).await {
            Ok(Some(prepared)) => prepared,
            Ok(None) => {
                return PlanExecution {
                    completed: HashMap::new(),
                    transcript: Vec::new(),
                    failed: true,
                };
            }
            Err(error) => {
                apply_terminal_error(&mut self.result, error);
                return PlanExecution {
                    completed: HashMap::new(),
                    transcript: Vec::new(),
                    failed: true,
                };
            }
        };

        let capabilities = self.runner.provider.capabilities();
        let max_parallel = if capabilities.supports_parallel_tool_calls {
            capabilities
                .limits
                .max_parallel_tool_calls
                .map_or(MAX_SCHEDULER_PARALLELISM, |limit| limit as usize)
                .clamp(1, MAX_SCHEDULER_PARALLELISM)
        } else {
            1
        };
        let mut completed = HashMap::new();
        let mut transcript = Vec::new();
        let mut done = HashSet::new();
        let mut wave_number = 0u32;

        while done.len() < prepared.len() {
            let ready = prepared
                .iter()
                .enumerate()
                .filter(|(_, candidate)| {
                    !done.contains(&candidate.node.id)
                        && candidate
                            .node
                            .depends_on
                            .iter()
                            .all(|dependency| done.contains(dependency))
                })
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            if ready.is_empty() {
                self.result.errors.push(RunError::new(
                    "plan_execution_failed",
                    "execution plan made no dependency progress",
                ));
                return PlanExecution {
                    completed,
                    transcript,
                    failed: true,
                };
            }
            let wave = build_wave(&prepared, &ready, max_parallel);
            wave_number += 1;

            let broker = ToolBroker::new(
                &self.runner.tools,
                &self.runner.policy,
                &self.runner.approvals,
            );
            let mut executable = Vec::new();
            for &index in &wave {
                if let Some(reused) = prepared[index].reused.take() {
                    let call = ToolCall::new(
                        format!("plan:{}", prepared[index].node.id),
                        prepared[index].node.tool_id.clone(),
                        serde_json::to_string(&prepared[index].node.arguments)
                            .unwrap_or_else(|_| "{}".into()),
                    );
                    self.events.emit(RunEvent::PlanNodeCompleted {
                        node_id: prepared[index].node.id.clone(),
                        tool_id: prepared[index].node.tool_id.clone(),
                        wave: wave_number,
                        ok: reused.ok,
                        duration_ms: 0,
                    });
                    completed.insert(prepared[index].node.id.clone(), reused.clone());
                    transcript.push((call, reused));
                    done.insert(prepared[index].node.id.clone());
                    continue;
                }

                let arguments = match bind_arguments(&prepared[index].node, &completed) {
                    Ok(arguments) => arguments,
                    Err(error) => {
                        apply_terminal_error(&mut self.result, error);
                        return PlanExecution {
                            completed,
                            transcript,
                            failed: true,
                        };
                    }
                };
                let node_id = prepared[index].node.id.clone();
                let tool_id = prepared[index].node.tool_id.clone();
                let Some(call) = prepared[index].prepared.as_mut() else {
                    self.result.errors.push(RunError::new(
                        "plan_execution_failed",
                        "prepared plan node is missing its invocation",
                    ));
                    return PlanExecution {
                        completed,
                        transcript,
                        failed: true,
                    };
                };
                match broker
                    .revalidate_bound_arguments(
                        call,
                        arguments,
                        self.request,
                        &mut self.result,
                        &mut self.events,
                        &self.broker_state,
                        self.deadline,
                    )
                    .await
                {
                    Ok(Some(reused)) => {
                        self.events.emit(RunEvent::PlanNodeCompleted {
                            node_id: node_id.clone(),
                            tool_id,
                            wave: wave_number,
                            ok: reused.ok,
                            duration_ms: 0,
                        });
                        completed.insert(node_id.clone(), reused.clone());
                        transcript.push((call.call.clone(), reused));
                        done.insert(node_id);
                    }
                    Ok(None) => {
                        if let Some(recorded) = self
                            .result
                            .tool_calls
                            .iter_mut()
                            .rev()
                            .find(|recorded| recorded.id == call.call.id)
                        {
                            *recorded = call.call.clone();
                        }
                        executable.push(index);
                    }
                    Err(error) => {
                        apply_terminal_error(&mut self.result, error);
                        return PlanExecution {
                            completed,
                            transcript,
                            failed: true,
                        };
                    }
                }
            }

            for &index in &executable {
                let call = prepared[index].prepared.as_ref().expect("checked above");
                self.events.emit(RunEvent::PlanNodeStarted {
                    node_id: prepared[index].node.id.clone(),
                    tool_id: call.call.tool_id.clone(),
                    wave: wave_number,
                });
            }
            let executions = join_all(executable.iter().map(|&index| {
                broker.execute(
                    prepared[index].prepared.as_ref().expect("checked above"),
                    self.request,
                    self.deadline,
                )
            }))
            .await;

            let mut failed = false;
            for (&index, execution) in executable.iter().zip(executions) {
                let call = prepared[index].prepared.as_ref().expect("checked above");
                match execution {
                    Ok(execution) => {
                        self.events.emit(RunEvent::ToolCompleted {
                            call_id: call.call.id.clone(),
                            tool_id: call.call.tool_id.clone(),
                            ok: execution.result.ok,
                        });
                        self.events.emit(RunEvent::PlanNodeCompleted {
                            node_id: prepared[index].node.id.clone(),
                            tool_id: call.call.tool_id.clone(),
                            wave: wave_number,
                            ok: execution.result.ok,
                            duration_ms: execution.duration_ms,
                        });
                        broker.record_execution(&mut self.broker_state, call, &execution);
                        if let Some(error) = execution.validation_error {
                            self.result.errors.push(error);
                            failed = true;
                        } else if !execution.result.ok {
                            self.result.errors.push(RunError::new(
                                "tool_error",
                                execution
                                    .result
                                    .error
                                    .clone()
                                    .unwrap_or_else(|| "tool returned a failure result".into()),
                            ));
                            failed = true;
                        } else {
                            completed
                                .insert(prepared[index].node.id.clone(), execution.result.clone());
                            transcript.push((call.call.clone(), execution.result));
                            done.insert(prepared[index].node.id.clone());
                        }
                    }
                    Err(error) => {
                        self.events.emit(RunEvent::ToolCompleted {
                            call_id: call.call.id.clone(),
                            tool_id: call.call.tool_id.clone(),
                            ok: false,
                        });
                        self.events.emit(RunEvent::PlanNodeCompleted {
                            node_id: prepared[index].node.id.clone(),
                            tool_id: call.call.tool_id.clone(),
                            wave: wave_number,
                            ok: false,
                            duration_ms: 0,
                        });
                        apply_terminal_error(&mut self.result, error);
                        failed = true;
                    }
                }
            }
            if failed {
                return PlanExecution {
                    completed,
                    transcript,
                    failed: true,
                };
            }
        }

        PlanExecution {
            completed,
            transcript,
            failed: false,
        }
    }

    fn append_plan_transcript(&mut self, entries: &[(ToolCall, ToolResult)]) {
        if entries.is_empty() {
            return;
        }
        self.messages.push(Message::assistant_tool_calls(
            entries.iter().map(|(call, _)| call.clone()).collect(),
        ));
        for (call, result) in entries {
            if let Err(error) =
                push_tool_message(&mut self.messages, call, result, &self.request.agent.limits)
            {
                apply_terminal_error(&mut self.result, error);
                break;
            }
        }
    }

    async fn run_reactive(&mut self) {
        let broker = ToolBroker::new(
            &self.runner.tools,
            &self.runner.policy,
            &self.runner.approvals,
        );
        loop {
            if !matches!(self.result.status, RunStatus::Failed) && self.result.cancelled {
                return;
            }
            let response = match self
                .complete(
                    self.messages.clone(),
                    self.runner
                        .tools
                        .allowed_definitions(&self.request.agent.tool_allowlist),
                )
                .await
            {
                Ok(Some(response)) => response,
                Ok(None) => return,
                Err(error) => {
                    apply_terminal_error(&mut self.result, error);
                    return;
                }
            };

            if let Some(output) = response.final_output {
                if output.trim().is_empty() {
                    self.result.errors.push(RunError::new(
                        "empty_model_response",
                        "model returned an empty final output",
                    ));
                    return;
                }
                match validate_output(
                    self.output_validator.as_ref(),
                    &output,
                    self.request.agent.limits.max_json_depth,
                ) {
                    Ok(()) => {
                        self.result.status = RunStatus::Completed;
                        self.result.final_output = Some(output);
                        return;
                    }
                    Err(error @ HarnessError::ResourceLimit(_)) => {
                        apply_terminal_error(&mut self.result, error);
                        return;
                    }
                    Err(error)
                        if self.output_repairs >= self.request.agent.limits.max_output_repairs =>
                    {
                        self.result.errors.push(error.run_error());
                        return;
                    }
                    Err(_) => {
                        self.output_repairs += 1;
                        self.messages.push(Message::assistant(output));
                        self.messages.push(Message::system(
                            "Return only JSON that satisfies the requested output schema.",
                        ));
                        if let Err(error) =
                            ensure_transcript(&self.messages, &self.request.agent.limits)
                        {
                            apply_terminal_error(&mut self.result, error);
                            return;
                        }
                        continue;
                    }
                }
            }

            if response.tool_calls.is_empty() {
                self.result.errors.push(RunError::new(
                    "empty_model_response",
                    "model returned neither final output nor tool calls",
                ));
                return;
            }
            let recorded_calls = self
                .runner
                .tool_calls_for_transcript(self.request, &response.tool_calls);
            self.messages
                .push(Message::assistant_tool_calls(recorded_calls));
            if let Err(error) = ensure_transcript(&self.messages, &self.request.agent.limits) {
                apply_terminal_error(&mut self.result, error);
                return;
            }

            for call in response.tool_calls {
                let outcome = match broker
                    .prepare(
                        self.request,
                        &mut self.result,
                        &mut self.events,
                        &mut self.broker_state,
                        call.clone(),
                        ToolCaller::Direct,
                        false,
                        self.deadline,
                    )
                    .await
                {
                    Ok(outcome) => outcome,
                    Err(error) => {
                        apply_terminal_error(&mut self.result, error);
                        return;
                    }
                };
                let tool_result = match outcome {
                    PrepareOutcome::Ready(prepared) => {
                        let execution =
                            match broker.execute(&prepared, self.request, self.deadline).await {
                                Ok(execution) => execution,
                                Err(error) => {
                                    self.events.emit(RunEvent::ToolCompleted {
                                        call_id: call.id.clone(),
                                        tool_id: call.tool_id.clone(),
                                        ok: false,
                                    });
                                    apply_terminal_error(&mut self.result, error);
                                    return;
                                }
                            };
                        self.events.emit(RunEvent::ToolCompleted {
                            call_id: call.id.clone(),
                            tool_id: call.tool_id.clone(),
                            ok: execution.result.ok,
                        });
                        broker.record_execution(&mut self.broker_state, &prepared, &execution);
                        if let Some(error) = execution.validation_error {
                            self.result.errors.push(error);
                        } else if !execution.result.ok {
                            self.result.errors.push(RunError::new(
                                "tool_error",
                                execution
                                    .result
                                    .error
                                    .clone()
                                    .unwrap_or_else(|| "tool returned a failure result".into()),
                            ));
                        }
                        execution.result
                    }
                    PrepareOutcome::Rejected(result) | PrepareOutcome::Reused(result) => result,
                    PrepareOutcome::Stop => return,
                };
                if let Err(error) = push_tool_message(
                    &mut self.messages,
                    &call,
                    &tool_result,
                    &self.request.agent.limits,
                ) {
                    apply_terminal_error(&mut self.result, error);
                    return;
                }
            }
        }
    }

    fn finish(mut self) -> RunResult {
        self.result.duration_ms = self.started.elapsed().as_millis() as u64;
        self.events.emit(RunEvent::StrategyUsage {
            strategy: self.selected,
            model_calls: self.model_calls,
            tool_calls: self.broker_state.tool_calls,
            duration_ms: self.result.duration_ms,
        });
        self.events.emit(RunEvent::Completed {
            status: self.result.status.clone(),
        });
        self.result
    }
}

fn build_wave(prepared: &[PreparedNode], ready: &[usize], max_parallel: usize) -> Vec<usize> {
    let first = ready[0];
    if !parallel_eligible(&prepared[first]) {
        return vec![first];
    }
    let mut keys = BTreeSet::new();
    let mut wave = Vec::new();
    for &index in ready {
        if wave.len() >= max_parallel || !parallel_eligible(&prepared[index]) {
            continue;
        }
        let key = prepared[index]
            .prepared
            .as_ref()
            .and_then(|call| call.tool.definition().concurrency_key.clone());
        if key.as_ref().is_some_and(|key| !keys.insert(key.clone())) {
            continue;
        }
        wave.push(index);
    }
    wave
}

fn parallel_eligible(node: &PreparedNode) -> bool {
    if node.reused.is_some() {
        return true;
    }
    let Some(prepared) = &node.prepared else {
        return false;
    };
    node.node.concurrency == PlanConcurrency::ToolDefault
        && !node.node.approval_barrier
        && !node.node.commit_boundary
        && prepared.tool.definition().read_only
        && prepared.tool.definition().parallel_safe
}

fn bind_arguments(
    node: &PlanNode,
    completed: &HashMap<String, ToolResult>,
) -> Result<Value, HarnessError> {
    let mut arguments = node.arguments.clone();
    for binding in &node.result_bindings {
        let source = completed.get(&binding.source.node_id).ok_or_else(|| {
            HarnessError::InvalidArguments(format!(
                "plan node '{}' cannot bind missing result '{}'",
                node.id, binding.source.node_id
            ))
        })?;
        let value = source
            .output
            .pointer(&binding.source.output_pointer)
            .ok_or_else(|| {
                HarnessError::InvalidArguments(format!(
                    "plan node '{}' source pointer did not resolve",
                    node.id
                ))
            })?
            .clone();
        let target = arguments
            .pointer_mut(&binding.target_pointer)
            .ok_or_else(|| {
                HarnessError::InvalidArguments(format!(
                    "plan node '{}' target pointer did not resolve",
                    node.id
                ))
            })?;
        *target = value;
    }
    Ok(arguments)
}
