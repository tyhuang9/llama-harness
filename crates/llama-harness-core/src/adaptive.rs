use crate::{
    broker::{BrokerState, FinalizeOutcome, PrepareOutcome, PreparedCall, ToolBroker},
    event::EventEmitter,
    limits::serialized_len,
    plan::{MAX_EXECUTION_PLAN_NODES, MAX_PLAN_ARGUMENT_BYTES},
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
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::Arc,
    time::Instant as StdInstant,
};
use tokio::time::Instant;
use uuid::Uuid;

const MAX_SCHEDULER_PARALLELISM: usize = 8;
const MAX_PLAN_RETAINED_BYTES: u64 = 16 * 1024 * 1024;
const PLANNER_PROMPT: &str = "Select the safest efficient tool strategy. Return only one strict JSON object: {\"strategy\":\"direct\"} when no finite safe plan is justified, or {\"strategy\":\"declarative_plan\",\"plan\":{\"nodes\":[...]}} for a finite dependency DAG. Use only the supplied tools. Every plan node requires id, tool_id, and schema-valid arguments. Optional fields are depends_on, result_bindings, concurrency, approval_barrier, and commit_boundary. Choose direct for mutations, approval-sensitive work, ambiguity, or an uncertain next step.";
const REPAIR_PROMPT: &str = "The previous strategy envelope was invalid. Repair it once. Return only the strict JSON envelope requested by the planning instructions, with no prose or markdown.";
const RECOVERY_PROMPT: &str = "Execution stopped after the recorded completed results. Produce one replacement declarative plan using only those results as prior state. Never repeat a completed mutation. Return only {\"strategy\":\"declarative_plan\",\"plan\":{\"nodes\":[...]}}.";

#[derive(Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case", deny_unknown_fields)]
enum PlannerEnvelope {
    Direct,
    DeclarativePlan { plan: ExecutionPlan },
}

#[derive(Serialize)]
struct RecoveryState<'a> {
    completed: BTreeMap<&'a str, &'a Value>,
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
    terminal: bool,
    plan_attempt: u32,
}

struct PreparedNode {
    node: PlanNode,
    prepared: Option<PreparedCall>,
    reused: Option<(ToolCall, Arc<ToolResult>)>,
    event_id: String,
}

struct PlanExecution {
    completed: BTreeMap<String, Arc<ToolResult>>,
    transcript: Vec<(ToolCall, Arc<ToolResult>)>,
    failure: Option<PlanFailureKind>,
    effects_started: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlanFailureKind {
    Preflight,
    Recoverable,
    Terminal,
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
            RunStrategy::Direct => {
                self.run_direct(
                    request,
                    Some(DirectStrategyEvents {
                        requested: RunStrategy::Direct,
                        reason: StrategySelectionReason::Forced,
                        fallback: None,
                    }),
                )
                .await
            }
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
                            requested: RunStrategy::Adaptive,
                            reason: StrategySelectionReason::CapabilityDowngrade,
                            fallback: Some(StrategyFallbackReason::UnsupportedCapability),
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
        let mut invalid_repair_used = false;
        let envelope = loop {
            let response = match run
                .complete(planner_messages.clone(), plan_tools.clone())
                .await
            {
                Ok(Some(response)) => response,
                Ok(None) => return Ok(run.finish()),
                Err(error) => {
                    run.terminate(error);
                    return Ok(run.finish());
                }
            };
            match run.parse_envelope(response, requested) {
                Ok(envelope) => break envelope,
                Err(error) if is_terminal_failure(&error) => {
                    run.terminate(error);
                    return Ok(run.finish());
                }
                Err(error) if !invalid_repair_used => {
                    invalid_repair_used = true;
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

                if execution.failure == Some(PlanFailureKind::Preflight)
                    && requested == RunStrategy::Adaptive
                    && !execution.effects_started
                    && !run.terminal
                {
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
                    run.run_reactive().await;
                    return Ok(run.finish());
                }

                if execution.failure == Some(PlanFailureKind::Recoverable)
                    && execution.effects_started
                    && run.broker_state.recovery_is_safe()
                    && !run.terminal
                {
                    if let Err(error) = run.append_plan_transcript(&execution.transcript) {
                        run.terminate(error);
                        return Ok(run.finish());
                    }
                    let recovery = RecoveryState {
                        completed: execution
                            .completed
                            .iter()
                            .map(|(node_id, result)| (node_id.as_str(), &result.output))
                            .collect(),
                    };
                    let recovery = match serde_json::to_string(&recovery) {
                        Ok(recovery) => recovery,
                        Err(error) => {
                            run.terminate(HarnessError::InvalidRequest(format!(
                                "completed results could not be serialized: {error}"
                            )));
                            return Ok(run.finish());
                        }
                    };
                    let mut recovery_messages = run.messages.clone();
                    recovery_messages.push(Message::system(RECOVERY_PROMPT));
                    recovery_messages.push(Message::user(recovery));
                    if let Err(error) = ensure_transcript(&recovery_messages, &request.agent.limits)
                    {
                        run.terminate(error);
                        return Ok(run.finish());
                    }
                    match run.complete(recovery_messages, plan_tools.clone()).await {
                        Ok(Some(response)) => {
                            match run.parse_envelope(response, RunStrategy::DeclarativePlan) {
                                Ok(PlannerEnvelope::DeclarativePlan { plan: repaired }) => {
                                    run.events.emit(RunEvent::StrategyFallback {
                                        from: RunStrategy::DeclarativePlan,
                                        to: RunStrategy::DeclarativePlan,
                                        reason: StrategyFallbackReason::ExecutionRecovery,
                                    });
                                    plan = repaired;
                                    execution = run.execute_plan(&plan).await;
                                }
                                Ok(PlannerEnvelope::Direct) => unreachable!(
                                    "forced declarative envelope rejects direct recovery"
                                ),
                                Err(error) if is_terminal_failure(&error) => run.terminate(error),
                                Err(error) => run.result.errors.push(RunError::new(
                                    "plan_recovery_failed",
                                    format!("recovery plan was invalid: {error}"),
                                )),
                            }
                        }
                        Ok(None) => {}
                        Err(error) => run.terminate(error),
                    }
                }

                if execution.failure.is_some() {
                    if run.result.errors.is_empty() {
                        run.result.errors.push(RunError::new(
                            "plan_execution_failed",
                            "declarative plan execution failed",
                        ));
                    }
                } else {
                    if let Err(error) = run.append_plan_transcript(&execution.transcript) {
                        run.terminate(error);
                        return Ok(run.finish());
                    }
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
            terminal: false,
            plan_attempt: 0,
        })
    }

    fn terminate(&mut self, error: HarnessError) {
        if self.terminal {
            return;
        }
        apply_terminal_error(&mut self.result, error);
        self.terminal = true;
    }

    fn record_plan_error(
        &mut self,
        error: HarnessError,
        otherwise: PlanFailureKind,
    ) -> PlanFailureKind {
        if is_terminal_failure(&error) {
            self.terminate(error);
            PlanFailureKind::Terminal
        } else {
            self.result.errors.push(error.run_error());
            otherwise
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_reused_plan_node(
        &mut self,
        retained_bytes: &mut u64,
        budget: u64,
        completed: &mut BTreeMap<String, Arc<ToolResult>>,
        transcript: &mut Vec<(ToolCall, Arc<ToolResult>)>,
        done: &mut HashSet<String>,
        node_id: String,
        event_id: String,
        tool_id: String,
        wave: u32,
        call: ToolCall,
        result: Arc<ToolResult>,
    ) -> Result<(), HarnessError> {
        reserve_plan_entry(retained_bytes, budget, &call, &result)?;
        self.events.emit(RunEvent::PlanNodeCompleted {
            node_id: event_id,
            tool_id,
            wave,
            ok: result.ok,
            duration_ms: 0,
        });
        completed.insert(node_id.clone(), Arc::clone(&result));
        transcript.push((call, result));
        done.insert(node_id);
        Ok(())
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
                self.terminal = true;
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
        self.validate_bound_authorization_topology(plan)?;
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
            if node.result_bindings.is_empty() {
                let signature = crate::broker::canonical_signature(&node.tool_id, &node.arguments);
                let count = identical.entry(signature).or_default();
                *count += 1;
                if *count > self.request.agent.limits.max_identical_tool_calls {
                    return Err(HarnessError::ResourceLimit(
                        "execution plan exceeds the repeated-call limit".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_bound_authorization_topology(
        &self,
        plan: &ExecutionPlan,
    ) -> Result<(), HarnessError> {
        let indexes = plan
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id.as_str(), index))
            .collect::<HashMap<_, _>>();
        for (node_index, node) in plan.nodes.iter().enumerate() {
            if node.result_bindings.is_empty() {
                continue;
            }
            let mut pending = node.depends_on.clone();
            let mut visited = HashSet::new();
            while let Some(dependency_id) = pending.pop() {
                if !visited.insert(dependency_id.clone()) {
                    continue;
                }
                let dependency_index = indexes[dependency_id.as_str()];
                let dependency = &plan.nodes[dependency_index];
                let dependency_tool =
                    self.runner.tools.get(&dependency.tool_id).ok_or_else(|| {
                        HarnessError::InvalidTool("plan selects an unknown tool".into())
                    })?;
                if !dependency_tool.definition().read_only {
                    return Err(HarnessError::InvalidRequest(format!(
                        "bound plan node {} transitively depends on a mutation",
                        node_index + 1
                    )));
                }
                pending.extend(dependency.depends_on.iter().cloned());
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
            &self.runner.concurrency,
        );
        self.plan_attempt = self.plan_attempt.checked_add(1).ok_or_else(|| {
            HarnessError::ResourceLimit("declarative plan attempt counter overflow".into())
        })?;
        let plan_attempt = self.plan_attempt;
        let mut prepared_nodes = Vec::with_capacity(plan.nodes.len());
        for (node_index, node) in plan.nodes.iter().enumerate() {
            let arguments_json = serde_json::to_string(&node.arguments).map_err(|error| {
                HarnessError::InvalidArguments(format!(
                    "plan node '{}' arguments cannot be serialized: {error}",
                    node.id
                ))
            })?;
            let event_id = format!("plan-{plan_attempt}-node-{}", node_index + 1);
            let call = ToolCall::new(event_id.clone(), node.tool_id.clone(), arguments_json);
            match broker
                .prepare(
                    self.request,
                    &mut self.result,
                    &mut self.events,
                    &mut self.broker_state,
                    call.clone(),
                    ToolCaller::DeclarativePlan,
                    node.approval_barrier,
                    !node.result_bindings.is_empty(),
                    self.deadline,
                )
                .await?
            {
                PrepareOutcome::Ready(prepared) => prepared_nodes.push(PreparedNode {
                    node: node.clone(),
                    prepared: Some(*prepared),
                    reused: None,
                    event_id,
                }),
                PrepareOutcome::Reused(result) => prepared_nodes.push(PreparedNode {
                    node: node.clone(),
                    prepared: None,
                    reused: Some((call, result)),
                    event_id,
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

    async fn authorize_mutation_gate(
        &mut self,
        prepared: &mut [PreparedNode],
        completed: &BTreeMap<String, Arc<ToolResult>>,
        done: &HashSet<String>,
    ) -> Result<bool, HarnessError> {
        let broker = ToolBroker::new(
            &self.runner.tools,
            &self.runner.policy,
            &self.runner.approvals,
            &self.runner.concurrency,
        );
        for node in prepared.iter_mut() {
            if done.contains(&node.node.id) || node.reused.is_some() {
                continue;
            }
            let Some(call) = node.prepared.as_ref() else {
                continue;
            };
            if call.tool.definition().read_only && !node.node.approval_barrier {
                continue;
            }
            let arguments = bind_arguments(
                &node.node,
                completed,
                self.request.agent.limits.max_tool_arguments_bytes,
            )?;
            let call = node.prepared.as_mut().expect("checked above");
            let outcome = broker
                .revalidate_bound_arguments(
                    call,
                    arguments,
                    self.request,
                    &mut self.result,
                    &mut self.events,
                    &mut self.broker_state,
                    self.deadline,
                )
                .await?;
            match outcome {
                FinalizeOutcome::Ready => {}
                FinalizeOutcome::Reused(result) => {
                    let exact_call = call.call.clone();
                    node.prepared = None;
                    node.reused = Some((exact_call, result));
                }
                FinalizeOutcome::Stop => return Ok(false),
            }
        }
        Ok(true)
    }

    async fn execute_plan(&mut self, plan: &ExecutionPlan) -> PlanExecution {
        let mut prepared = match self.preflight_plan(plan).await {
            Ok(Some(prepared)) => prepared,
            Ok(None) => {
                self.terminal = true;
                return PlanExecution {
                    completed: BTreeMap::new(),
                    transcript: Vec::new(),
                    failure: Some(PlanFailureKind::Terminal),
                    effects_started: false,
                };
            }
            Err(error) => {
                let failure = self.record_plan_error(error, PlanFailureKind::Preflight);
                return PlanExecution {
                    completed: BTreeMap::new(),
                    transcript: Vec::new(),
                    failure: Some(failure),
                    effects_started: false,
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
        let plan_budget = self
            .request
            .agent
            .limits
            .max_transcript_bytes
            .min(MAX_PLAN_RETAINED_BYTES);
        let existing_transcript_bytes = self
            .messages
            .iter()
            .map(Message::transcript_bytes)
            .sum::<u64>();
        let Some(plan_result_budget) = plan_budget.checked_sub(existing_transcript_bytes) else {
            self.terminate(HarnessError::ResourceLimit(
                "plan result budget is exhausted before execution".into(),
            ));
            return PlanExecution {
                completed: BTreeMap::new(),
                transcript: Vec::new(),
                failure: Some(PlanFailureKind::Terminal),
                effects_started: false,
            };
        };
        let mut completed = BTreeMap::new();
        let mut transcript = Vec::new();
        let mut retained_bytes = 0u64;
        let mut done = HashSet::new();
        let mut wave_number = 0u32;
        let mut effects_started = false;
        let mut mutation_gate_open = false;

        while done.len() < prepared.len() {
            let mut ready = prepared
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
                    failure: Some(PlanFailureKind::Recoverable),
                    effects_started,
                };
            }
            if !mutation_gate_open {
                let read_ready = ready
                    .iter()
                    .copied()
                    .filter(|&index| prepared_node_is_read_only(&prepared[index]))
                    .collect::<Vec<_>>();
                if read_ready.is_empty() {
                    match self
                        .authorize_mutation_gate(&mut prepared, &completed, &done)
                        .await
                    {
                        Ok(true) => {
                            mutation_gate_open = true;
                            continue;
                        }
                        Ok(false) => {
                            self.terminal = true;
                            return PlanExecution {
                                completed,
                                transcript,
                                failure: Some(PlanFailureKind::Terminal),
                                effects_started,
                            };
                        }
                        Err(error) => {
                            let failure =
                                self.record_plan_error(error, PlanFailureKind::Recoverable);
                            return PlanExecution {
                                completed,
                                transcript,
                                failure: Some(failure),
                                effects_started,
                            };
                        }
                    }
                }
                ready = read_ready;
            }
            let wave = build_wave(&prepared, &ready, max_parallel);
            wave_number += 1;

            let broker = ToolBroker::new(
                &self.runner.tools,
                &self.runner.policy,
                &self.runner.approvals,
                &self.runner.concurrency,
            );
            let mut executable = Vec::new();
            for &index in &wave {
                if let Some((call, reused)) = prepared[index].reused.take() {
                    if let Err(error) = self.record_reused_plan_node(
                        &mut retained_bytes,
                        plan_result_budget,
                        &mut completed,
                        &mut transcript,
                        &mut done,
                        prepared[index].node.id.clone(),
                        prepared[index].event_id.clone(),
                        prepared[index].node.tool_id.clone(),
                        wave_number,
                        call,
                        reused,
                    ) {
                        self.terminate(error);
                        return PlanExecution {
                            completed,
                            transcript,
                            failure: Some(PlanFailureKind::Terminal),
                            effects_started,
                        };
                    }
                    continue;
                }

                let arguments = match bind_arguments(
                    &prepared[index].node,
                    &completed,
                    self.request.agent.limits.max_tool_arguments_bytes,
                ) {
                    Ok(arguments) => arguments,
                    Err(error) => {
                        let failure = self.record_plan_error(error, PlanFailureKind::Recoverable);
                        return PlanExecution {
                            completed,
                            transcript,
                            failure: Some(failure),
                            effects_started,
                        };
                    }
                };
                let node_id = prepared[index].node.id.clone();
                let tool_id = prepared[index].node.tool_id.clone();
                let event_id = prepared[index].event_id.clone();
                let Some(call) = prepared[index].prepared.as_mut() else {
                    self.result.errors.push(RunError::new(
                        "plan_execution_failed",
                        "prepared plan node is missing its invocation",
                    ));
                    return PlanExecution {
                        completed,
                        transcript,
                        failure: Some(PlanFailureKind::Recoverable),
                        effects_started,
                    };
                };
                match broker
                    .revalidate_bound_arguments(
                        call,
                        arguments,
                        self.request,
                        &mut self.result,
                        &mut self.events,
                        &mut self.broker_state,
                        self.deadline,
                    )
                    .await
                {
                    Ok(FinalizeOutcome::Reused(reused)) => {
                        let exact_call = call.call.clone();
                        if let Err(error) = self.record_reused_plan_node(
                            &mut retained_bytes,
                            plan_result_budget,
                            &mut completed,
                            &mut transcript,
                            &mut done,
                            node_id,
                            event_id,
                            tool_id,
                            wave_number,
                            exact_call,
                            reused,
                        ) {
                            self.terminate(error);
                            return PlanExecution {
                                completed,
                                transcript,
                                failure: Some(PlanFailureKind::Terminal),
                                effects_started,
                            };
                        }
                    }
                    Ok(FinalizeOutcome::Ready) => executable.push(index),
                    Ok(FinalizeOutcome::Stop) => {
                        self.terminal = true;
                        return PlanExecution {
                            completed,
                            transcript,
                            failure: Some(PlanFailureKind::Terminal),
                            effects_started,
                        };
                    }
                    Err(error) => {
                        let failure = self.record_plan_error(error, PlanFailureKind::Recoverable);
                        return PlanExecution {
                            completed,
                            transcript,
                            failure: Some(failure),
                            effects_started,
                        };
                    }
                }
            }

            let mut permitted = 0usize;
            let mut worst_case_bytes = retained_bytes;
            for &index in &executable {
                let call = prepared[index].prepared.as_ref().expect("checked above");
                let entry_bytes = maximum_plan_entry_bytes(
                    &call.call,
                    self.request.agent.limits.max_tool_result_bytes,
                );
                let Some(next) = worst_case_bytes.checked_add(entry_bytes) else {
                    break;
                };
                if next > plan_result_budget {
                    break;
                }
                worst_case_bytes = next;
                permitted += 1;
            }
            if permitted == 0 && !executable.is_empty() {
                self.terminate(HarnessError::ResourceLimit(
                    "aggregate plan result budget is exhausted before the next wave".into(),
                ));
                return PlanExecution {
                    completed,
                    transcript,
                    failure: Some(PlanFailureKind::Terminal),
                    effects_started,
                };
            }
            executable.truncate(permitted);
            effects_started |= !executable.is_empty();
            for &index in &executable {
                let call = prepared[index].prepared.as_ref().expect("checked above");
                broker.mark_dispatched(&mut self.broker_state, call);
                self.events.emit(RunEvent::PlanNodeStarted {
                    node_id: prepared[index].event_id.clone(),
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

            let mut wave_failure = None;
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
                            node_id: prepared[index].event_id.clone(),
                            tool_id: call.call.tool_id.clone(),
                            wave: wave_number,
                            ok: execution.result.ok,
                            duration_ms: execution.duration_ms,
                        });
                        broker.record_execution(&mut self.broker_state, call, &execution);
                        if let Some(error) = execution.validation_error {
                            self.result.errors.push(error);
                            wave_failure.get_or_insert(PlanFailureKind::Recoverable);
                        } else if !execution.result.ok {
                            self.result.errors.push(RunError::new(
                                "tool_error",
                                execution
                                    .result
                                    .error
                                    .clone()
                                    .unwrap_or_else(|| "tool returned a failure result".into()),
                            ));
                            wave_failure.get_or_insert(PlanFailureKind::Recoverable);
                        } else {
                            match reserve_plan_entry(
                                &mut retained_bytes,
                                plan_result_budget,
                                &call.call,
                                &execution.result,
                            ) {
                                Ok(()) => {
                                    completed.insert(
                                        prepared[index].node.id.clone(),
                                        Arc::clone(&execution.result),
                                    );
                                    transcript.push((call.call.clone(), execution.result));
                                    done.insert(prepared[index].node.id.clone());
                                }
                                Err(error) => {
                                    self.terminate(error);
                                    wave_failure = Some(PlanFailureKind::Terminal);
                                }
                            }
                        }
                    }
                    Err(error) => {
                        broker.mark_uncertain(&mut self.broker_state, call);
                        self.events.emit(RunEvent::ToolCompleted {
                            call_id: call.call.id.clone(),
                            tool_id: call.call.tool_id.clone(),
                            ok: false,
                        });
                        self.events.emit(RunEvent::PlanNodeCompleted {
                            node_id: prepared[index].event_id.clone(),
                            tool_id: call.call.tool_id.clone(),
                            wave: wave_number,
                            ok: false,
                            duration_ms: 0,
                        });
                        let failure = self.record_plan_error(error, PlanFailureKind::Recoverable);
                        if failure == PlanFailureKind::Terminal {
                            wave_failure = Some(failure);
                        } else {
                            wave_failure.get_or_insert(failure);
                        }
                    }
                }
            }
            if let Some(failure) = wave_failure {
                return PlanExecution {
                    completed,
                    transcript,
                    failure: Some(failure),
                    effects_started,
                };
            }
        }

        PlanExecution {
            completed,
            transcript,
            failure: None,
            effects_started,
        }
    }

    fn append_plan_transcript(
        &mut self,
        entries: &[(ToolCall, Arc<ToolResult>)],
    ) -> Result<(), HarnessError> {
        if entries.is_empty() {
            return Ok(());
        }
        self.messages.push(Message::assistant_tool_calls(
            entries.iter().map(|(call, _)| call.clone()).collect(),
        ));
        ensure_transcript(&self.messages, &self.request.agent.limits)?;
        for (call, result) in entries {
            push_tool_message(&mut self.messages, call, result, &self.request.agent.limits)?;
        }
        Ok(())
    }

    async fn run_reactive(&mut self) {
        let broker = ToolBroker::new(
            &self.runner.tools,
            &self.runner.policy,
            &self.runner.approvals,
            &self.runner.concurrency,
        );
        loop {
            if self.terminal {
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
                    self.terminate(error);
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
                        self.terminate(error);
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
                            self.terminate(error);
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
                self.terminate(error);
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
                        false,
                        self.deadline,
                    )
                    .await
                {
                    Ok(outcome) => outcome,
                    Err(error) => {
                        self.terminate(error);
                        return;
                    }
                };
                let tool_result = match outcome {
                    PrepareOutcome::Ready(prepared) => {
                        broker.mark_dispatched(&mut self.broker_state, &prepared);
                        let execution =
                            match broker.execute(&prepared, self.request, self.deadline).await {
                                Ok(execution) => execution,
                                Err(error) => {
                                    broker.mark_uncertain(&mut self.broker_state, &prepared);
                                    self.events.emit(RunEvent::ToolCompleted {
                                        call_id: call.id.clone(),
                                        tool_id: call.tool_id.clone(),
                                        ok: false,
                                    });
                                    self.terminate(error);
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
                    PrepareOutcome::Rejected(result) => Arc::new(result),
                    PrepareOutcome::Reused(result) => result,
                    PrepareOutcome::Stop => {
                        self.terminal = true;
                        return;
                    }
                };
                if let Err(error) = push_tool_message(
                    &mut self.messages,
                    &call,
                    &tool_result,
                    &self.request.agent.limits,
                ) {
                    self.terminate(error);
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

fn prepared_node_is_read_only(node: &PreparedNode) -> bool {
    node.prepared
        .as_ref()
        .is_some_and(|call| call.tool.definition().read_only)
}

fn is_terminal_failure(error: &HarnessError) -> bool {
    matches!(
        error,
        HarnessError::Cancelled | HarnessError::TimedOut(_) | HarnessError::ResourceLimit(_)
    )
}

fn maximum_plan_entry_bytes(call: &ToolCall, max_tool_result_bytes: u64) -> u64 {
    (call.id.len() as u64)
        .saturating_mul(2)
        .saturating_add(call.tool_id.len() as u64)
        .saturating_add(call.arguments_json.len() as u64)
        .saturating_add(max_tool_result_bytes)
}

fn reserve_plan_entry(
    retained_bytes: &mut u64,
    budget: u64,
    call: &ToolCall,
    result: &ToolResult,
) -> Result<(), HarnessError> {
    let result_bytes = serialized_len(result)?;
    let entry_bytes = (call.id.len() as u64)
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(call.tool_id.len() as u64))
        .and_then(|bytes| bytes.checked_add(call.arguments_json.len() as u64))
        .and_then(|bytes| bytes.checked_add(result_bytes))
        .ok_or_else(|| HarnessError::ResourceLimit("plan result budget overflow".into()))?;
    let next = retained_bytes
        .checked_add(entry_bytes)
        .ok_or_else(|| HarnessError::ResourceLimit("plan result budget overflow".into()))?;
    if next > budget {
        return Err(HarnessError::ResourceLimit(format!(
            "aggregate plan results exceed {budget} bytes"
        )));
    }
    *retained_bytes = next;
    Ok(())
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
    completed: &BTreeMap<String, Arc<ToolResult>>,
    max_arguments_bytes: u64,
) -> Result<Value, HarnessError> {
    let copy_budget = max_arguments_bytes.min(MAX_PLAN_ARGUMENT_BYTES as u64);
    let mut projected_bytes = serialized_len(&node.arguments)?;
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
            })?;
        if node.arguments.pointer(&binding.target_pointer).is_none() {
            return Err(HarnessError::InvalidArguments(format!(
                "plan node '{}' target pointer did not resolve",
                node.id
            )));
        }
        projected_bytes = projected_bytes
            .checked_add(serialized_len(value)?)
            .ok_or_else(|| {
                HarnessError::ResourceLimit("bound argument copy budget overflow".into())
            })?;
        if projected_bytes > copy_budget {
            return Err(HarnessError::ResourceLimit(format!(
                "bound argument copies exceed {copy_budget} bytes"
            )));
        }
    }

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
