use crate::{
    broker::{BrokerState, FinalizeOutcome, PrepareOutcome, PreparedCall, ToolBroker},
    discovery::{ToolDiscoveryStats, ToolScope},
    event::EventEmitter,
    limits::serialized_len,
    plan::{MAX_EXECUTION_PLAN_NODES, MAX_PLAN_ARGUMENT_BYTES},
    runner::{
        apply_terminal_error, emit_discovery, ensure_transcript, initial_messages,
        merge_generation, pre_event_terminal_result, preflight_request, preflight_terminal_result,
        provider_deadline, push_tool_message, validate_model_response, validate_output,
        DirectStrategyEvents, RunPreflight,
    },
    AgentRunner, ExecutionPlan, HarnessError, Message, ModelRequest, ModelResponse,
    PlanConcurrency, PlanLifecycleOutcome, PlanNode, PlanNodeOutcome, PlanPhase, RunError,
    RunEvent, RunRequest, RunResult, RunStatus, RunStrategy, StrategyFallbackReason,
    StrategySelectionReason, ToolCall, ToolCaller, ToolDefinition, ToolResult,
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
    planning_model_calls: u32,
    repair_model_calls: u32,
    recovery_model_calls: u32,
    final_synthesis_model_calls: u32,
    reactive_model_calls: u32,
    output_repairs: u32,
    broker_state: BrokerState,
    selected: RunStrategy,
    terminal: bool,
    plan_attempt: u32,
    direct_scope: ToolScope,
    plan_scope: ToolScope,
}

struct PreparedPlanScope {
    scope: ToolScope,
    discovery: ToolDiscoveryStats,
}

enum PlanningReadiness {
    Ready(PreparedPlanScope),
    Downgrade(&'static str),
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

#[derive(Clone, Copy)]
enum ModelCallPhase {
    Planning,
    Repair,
    Recovery,
    FinalSynthesis,
    Reactive,
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
        if strategy == RunStrategy::Programmatic {
            return Err(HarnessError::UnsupportedCapability(
                "programmatic execution requires the optional sandbox runtime".into(),
            ));
        }
        let preflight = match preflight_request(&request) {
            Ok(preflight) => preflight,
            Err(error @ (HarnessError::Cancelled | HarnessError::TimedOut(_))) => {
                return Ok(preflight_terminal_result(&request, error));
            }
            Err(error) => return Err(error),
        };
        match strategy {
            RunStrategy::Direct => {
                self.run_direct(
                    request,
                    Some(DirectStrategyEvents {
                        requested: RunStrategy::Direct,
                        reason: StrategySelectionReason::Forced,
                        fallback: None,
                    }),
                    preflight,
                )
                .await
            }
            RunStrategy::Programmatic => unreachable!("programmatic strategy returned above"),
            RunStrategy::DeclarativePlan => {
                let readiness = match self.prepare_plan_scope(&request, preflight.deadline) {
                    Ok(readiness) => readiness,
                    Err(error @ (HarnessError::Cancelled | HarnessError::TimedOut(_))) => {
                        return Ok(pre_event_terminal_result(
                            &request,
                            error,
                            &self.events,
                            RunStrategy::DeclarativePlan,
                        ));
                    }
                    Err(error) => return Err(error),
                };
                match readiness {
                    PlanningReadiness::Ready(prepared) => {
                        self.run_planned(request, RunStrategy::DeclarativePlan, prepared, preflight)
                            .await
                    }
                    PlanningReadiness::Downgrade(reason) => {
                        Err(HarnessError::UnsupportedCapability(reason.into()))
                    }
                }
            }
            RunStrategy::Adaptive => {
                let readiness = match self.prepare_plan_scope(&request, preflight.deadline) {
                    Ok(readiness) => readiness,
                    Err(error @ (HarnessError::Cancelled | HarnessError::TimedOut(_))) => {
                        return Ok(pre_event_terminal_result(
                            &request,
                            error,
                            &self.events,
                            RunStrategy::Adaptive,
                        ));
                    }
                    Err(error) => return Err(error),
                };
                match readiness {
                    PlanningReadiness::Downgrade(_) => {
                        self.run_direct(
                            request,
                            Some(DirectStrategyEvents {
                                requested: RunStrategy::Adaptive,
                                reason: StrategySelectionReason::CapabilityDowngrade,
                                fallback: Some(StrategyFallbackReason::UnsupportedCapability),
                            }),
                            preflight,
                        )
                        .await
                    }
                    PlanningReadiness::Ready(prepared) => {
                        self.run_planned(request, RunStrategy::Adaptive, prepared, preflight)
                            .await
                    }
                }
            }
        }
    }

    fn prepare_plan_scope(
        &self,
        request: &RunRequest,
        deadline: Option<Instant>,
    ) -> Result<PlanningReadiness, HarnessError> {
        let capabilities = self.provider.capabilities();
        if request.agent.limits.max_model_calls < 2 {
            return Ok(PlanningReadiness::Downgrade(
                "run model-call budget cannot support planning and finalization",
            ));
        }
        if !capabilities.supports_tools || !capabilities.supports_structured_plans {
            return Ok(PlanningReadiness::Downgrade(
                "provider does not support structured plans",
            ));
        }
        if capabilities.limits.max_plan_nodes == Some(0)
            || capabilities.limits.max_plan_bytes == Some(0)
        {
            return Ok(PlanningReadiness::Downgrade(
                "provider advertises no structured-plan capacity",
            ));
        }
        let (scope, discovery) = self.tools.select_scope_for_run(
            &request.input,
            &request.agent.tool_allowlist,
            ToolCaller::DeclarativePlan,
            self.discovery_limits,
            &capabilities.limits,
            &request.cancellation,
            deadline,
        )?;
        if scope.is_empty() {
            return Ok(PlanningReadiness::Downgrade(
                "no allowed tools permit declarative plan calls",
            ));
        }
        Ok(PlanningReadiness::Ready(PreparedPlanScope {
            scope,
            discovery,
        }))
    }

    async fn run_planned(
        &self,
        request: RunRequest,
        requested: RunStrategy,
        prepared_plan: PreparedPlanScope,
        preflight: RunPreflight,
    ) -> Result<RunResult, HarnessError> {
        let mut run = match StrategyRun::new(self, &request, requested, prepared_plan, preflight) {
            Ok(run) => run,
            Err(error @ (HarnessError::Cancelled | HarnessError::TimedOut(_))) => {
                return Ok(pre_event_terminal_result(
                    &request,
                    error,
                    &self.events,
                    requested,
                ));
            }
            Err(error) => return Err(error),
        };
        if requested == RunStrategy::DeclarativePlan {
            run.events.emit(RunEvent::StrategySelected {
                requested,
                selected: RunStrategy::DeclarativePlan,
                reason: StrategySelectionReason::Forced,
            });
        }

        let plan_tools = run.plan_scope.definitions().to_vec();
        let mut planner_messages = run.messages.clone();
        planner_messages.insert(0, Message::system(PLANNER_PROMPT));
        let mut invalid_repair_used = false;
        let envelope = loop {
            let (plan_phase, model_phase) = if invalid_repair_used {
                (PlanPhase::Repair, ModelCallPhase::Repair)
            } else {
                (PlanPhase::Planning, ModelCallPhase::Planning)
            };
            run.events.emit(RunEvent::PlanLifecycle {
                phase: plan_phase,
                attempt: 1,
                outcome: PlanLifecycleOutcome::Started,
            });
            let response = match run
                .complete(
                    planner_messages.clone(),
                    plan_tools.clone(),
                    model_phase,
                    true,
                )
                .await
            {
                Ok(Some(response)) => response,
                Ok(None) => {
                    run.events.emit(RunEvent::PlanLifecycle {
                        phase: plan_phase,
                        attempt: 1,
                        outcome: lifecycle_outcome_for_status(&run.result.status),
                    });
                    return Ok(run.finish());
                }
                Err(error) => {
                    run.events.emit(RunEvent::PlanLifecycle {
                        phase: plan_phase,
                        attempt: 1,
                        outcome: lifecycle_outcome_for_error(&error),
                    });
                    if requested == RunStrategy::Adaptive
                        && !is_terminal_failure(&error)
                        && run.has_model_call_capacity()
                    {
                        run.select_direct_fallback(
                            requested,
                            StrategyFallbackReason::PlannerFailure,
                        );
                        run.run_reactive(ModelCallPhase::Reactive).await;
                    } else {
                        run.terminate(error);
                    }
                    return Ok(run.finish());
                }
            };
            run.events.emit(RunEvent::PlanLifecycle {
                phase: plan_phase,
                attempt: 1,
                outcome: PlanLifecycleOutcome::Succeeded,
            });
            let validation_attempt = if invalid_repair_used { 2 } else { 1 };
            run.events.emit(RunEvent::PlanLifecycle {
                phase: PlanPhase::Validation,
                attempt: validation_attempt,
                outcome: PlanLifecycleOutcome::Started,
            });
            match run.parse_envelope(response, requested) {
                Ok(envelope) => {
                    run.events.emit(RunEvent::PlanLifecycle {
                        phase: PlanPhase::Validation,
                        attempt: validation_attempt,
                        outcome: PlanLifecycleOutcome::Succeeded,
                    });
                    break envelope;
                }
                Err(error) if is_terminal_failure(&error) => {
                    run.events.emit(RunEvent::PlanLifecycle {
                        phase: PlanPhase::Validation,
                        attempt: validation_attempt,
                        outcome: lifecycle_outcome_for_error(&error),
                    });
                    run.terminate(error);
                    return Ok(run.finish());
                }
                Err(error) if !invalid_repair_used && run.can_spend_optional_model_call() => {
                    run.events.emit(RunEvent::PlanLifecycle {
                        phase: PlanPhase::Validation,
                        attempt: validation_attempt,
                        outcome: validation_outcome_for_error(&error),
                    });
                    invalid_repair_used = true;
                    planner_messages.push(Message::system(REPAIR_PROMPT));
                    run.result
                        .errors
                        .retain(|record| record.code != "invalid_plan");
                    let _ = error;
                }
                Err(error) if requested == RunStrategy::Adaptive => {
                    run.events.emit(RunEvent::PlanLifecycle {
                        phase: PlanPhase::Validation,
                        attempt: validation_attempt,
                        outcome: validation_outcome_for_error(&error),
                    });
                    if !invalid_repair_used {
                        run.events.emit(RunEvent::PlanLifecycle {
                            phase: PlanPhase::Repair,
                            attempt: 1,
                            outcome: PlanLifecycleOutcome::Skipped,
                        });
                    }
                    if !run.has_model_call_capacity() {
                        run.terminate(HarnessError::ResourceLimit(
                            "no model call remains for direct fallback".into(),
                        ));
                        return Ok(run.finish());
                    }
                    run.select_direct_fallback(requested, StrategyFallbackReason::InvalidPlan);
                    run.result
                        .errors
                        .retain(|record| record.code != "invalid_plan");
                    let _ = error;
                    run.run_reactive(ModelCallPhase::Reactive).await;
                    return Ok(run.finish());
                }
                Err(error) => {
                    run.events.emit(RunEvent::PlanLifecycle {
                        phase: PlanPhase::Validation,
                        attempt: validation_attempt,
                        outcome: validation_outcome_for_error(&error),
                    });
                    if !invalid_repair_used {
                        run.events.emit(RunEvent::PlanLifecycle {
                            phase: PlanPhase::Repair,
                            attempt: 1,
                            outcome: PlanLifecycleOutcome::Skipped,
                        });
                    }
                    run.result.errors.push(RunError::new(
                        "invalid_plan",
                        format!("declarative plan was invalid: {error}"),
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
                run.run_reactive(ModelCallPhase::Reactive).await;
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
                    && run.has_model_call_capacity()
                {
                    run.select_direct_fallback(requested, StrategyFallbackReason::InvalidPlan);
                    run.run_reactive(ModelCallPhase::Reactive).await;
                    return Ok(run.finish());
                }

                if execution.failure == Some(PlanFailureKind::Recoverable)
                    && execution.effects_started
                    && run.broker_state.recovery_is_safe()
                    && !run.terminal
                {
                    if !run.can_spend_optional_model_call() {
                        run.events.emit(RunEvent::PlanLifecycle {
                            phase: PlanPhase::Recovery,
                            attempt: 1,
                            outcome: PlanLifecycleOutcome::Skipped,
                        });
                    } else {
                        run.events.emit(RunEvent::PlanLifecycle {
                            phase: PlanPhase::Recovery,
                            attempt: 1,
                            outcome: PlanLifecycleOutcome::Started,
                        });
                        if let Err(error) = run.append_plan_transcript(&execution.transcript) {
                            run.events.emit(RunEvent::PlanLifecycle {
                                phase: PlanPhase::Recovery,
                                attempt: 1,
                                outcome: lifecycle_outcome_for_error(&error),
                            });
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
                                let error = HarnessError::InvalidRequest(format!(
                                    "completed results could not be serialized: {error}"
                                ));
                                run.events.emit(RunEvent::PlanLifecycle {
                                    phase: PlanPhase::Recovery,
                                    attempt: 1,
                                    outcome: lifecycle_outcome_for_error(&error),
                                });
                                run.terminate(error);
                                return Ok(run.finish());
                            }
                        };
                        let mut recovery_messages = run.messages.clone();
                        recovery_messages.push(Message::system(RECOVERY_PROMPT));
                        recovery_messages.push(Message::user(recovery));
                        if let Err(error) =
                            ensure_transcript(&recovery_messages, &request.agent.limits)
                        {
                            run.events.emit(RunEvent::PlanLifecycle {
                                phase: PlanPhase::Recovery,
                                attempt: 1,
                                outcome: lifecycle_outcome_for_error(&error),
                            });
                            run.terminate(error);
                            return Ok(run.finish());
                        }
                        match run
                            .complete(
                                recovery_messages,
                                plan_tools.clone(),
                                ModelCallPhase::Recovery,
                                true,
                            )
                            .await
                        {
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
                                        run.events.emit(RunEvent::PlanLifecycle {
                                            phase: PlanPhase::Recovery,
                                            attempt: 1,
                                            outcome: recovery_execution_outcome(&execution, &run),
                                        });
                                    }
                                    Ok(PlannerEnvelope::Direct) => unreachable!(
                                        "forced declarative envelope rejects direct recovery"
                                    ),
                                    Err(error) if is_terminal_failure(&error) => {
                                        run.events.emit(RunEvent::PlanLifecycle {
                                            phase: PlanPhase::Recovery,
                                            attempt: 1,
                                            outcome: lifecycle_outcome_for_error(&error),
                                        });
                                        run.terminate(error);
                                    }
                                    Err(error) => {
                                        run.events.emit(RunEvent::PlanLifecycle {
                                            phase: PlanPhase::Recovery,
                                            attempt: 1,
                                            outcome: PlanLifecycleOutcome::Invalid,
                                        });
                                        run.result.errors.push(RunError::new(
                                            "plan_recovery_failed",
                                            format!("recovery plan was invalid: {error}"),
                                        ));
                                    }
                                }
                            }
                            Ok(None) => run.events.emit(RunEvent::PlanLifecycle {
                                phase: PlanPhase::Recovery,
                                attempt: 1,
                                outcome: lifecycle_outcome_for_status(&run.result.status),
                            }),
                            Err(error) => {
                                run.events.emit(RunEvent::PlanLifecycle {
                                    phase: PlanPhase::Recovery,
                                    attempt: 1,
                                    outcome: lifecycle_outcome_for_error(&error),
                                });
                                run.terminate(error);
                            }
                        }
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
                    run.run_reactive(ModelCallPhase::FinalSynthesis).await;
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
        prepared_plan: PreparedPlanScope,
        preflight: RunPreflight,
    ) -> Result<Self, HarnessError> {
        let capabilities = runner.provider.capabilities();
        let (direct_scope, direct_discovery) = runner.tools.select_scope_for_run(
            &request.input,
            &request.agent.tool_allowlist,
            ToolCaller::Direct,
            runner.discovery_limits,
            &capabilities.limits,
            &request.cancellation,
            preflight.deadline,
        )?;
        let started = StdInstant::now();
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
        emit_discovery(
            &mut events,
            ToolCaller::DeclarativePlan,
            prepared_plan.discovery,
        );
        emit_discovery(&mut events, ToolCaller::Direct, direct_discovery);
        Ok(Self {
            runner,
            request,
            output_validator: preflight.output_validator,
            result: RunResult::new(run_id, RunStatus::Failed, model.clone(), trace_id),
            events,
            messages: initial_messages(request),
            model,
            deadline: preflight.deadline,
            started,
            model_calls: 0,
            planning_model_calls: 0,
            repair_model_calls: 0,
            recovery_model_calls: 0,
            final_synthesis_model_calls: 0,
            reactive_model_calls: 0,
            output_repairs: 0,
            broker_state: BrokerState::default(),
            selected: requested,
            terminal: false,
            plan_attempt: 0,
            direct_scope,
            plan_scope: prepared_plan.scope,
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
        duration_ms: u64,
    ) -> Result<(), HarnessError> {
        reserve_plan_entry(retained_bytes, budget, &call, &result)?;
        self.events.emit(RunEvent::PlanNodeCompleted {
            node_id: event_id,
            tool_id,
            attempt: self.plan_attempt,
            wave,
            ok: result.ok,
            outcome: PlanNodeOutcome::Reused,
            duration_ms,
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
        phase: ModelCallPhase,
        reserve_final_synthesis: bool,
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
            match phase {
                ModelCallPhase::Planning => self.planning_model_calls += 1,
                ModelCallPhase::Repair => self.repair_model_calls += 1,
                ModelCallPhase::Recovery => self.recovery_model_calls += 1,
                ModelCallPhase::FinalSynthesis => self.final_synthesis_model_calls += 1,
                ModelCallPhase::Reactive => self.reactive_model_calls += 1,
            }
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
                    if reserve_final_synthesis && !self.can_spend_optional_model_call() {
                        return Err(HarnessError::RetryableProvider(
                            "provider retry skipped to preserve final synthesis capacity".into(),
                        ));
                    }
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

    fn can_spend_optional_model_call(&self) -> bool {
        self.request
            .agent
            .limits
            .max_model_calls
            .saturating_sub(self.model_calls)
            > 1
    }

    fn has_model_call_capacity(&self) -> bool {
        self.model_calls < self.request.agent.limits.max_model_calls
    }

    fn select_direct_fallback(&mut self, requested: RunStrategy, reason: StrategyFallbackReason) {
        self.events.emit(RunEvent::StrategyFallback {
            from: RunStrategy::DeclarativePlan,
            to: RunStrategy::Direct,
            reason,
        });
        self.events.emit(RunEvent::StrategySelected {
            requested,
            selected: RunStrategy::Direct,
            reason: StrategySelectionReason::PlannerSelectedDirect,
        });
        self.selected = RunStrategy::Direct;
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
            if !self.plan_scope.contains(&node.tool_id) {
                return Err(HarnessError::InvalidTool(format!(
                    "plan node '{}' selects an unavailable tool",
                    node.id
                )));
            }
            let Some(tool) = self.runner.tools.get(&node.tool_id) else {
                return Err(HarnessError::InvalidTool(format!(
                    "plan node '{}' selects an unavailable tool",
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
                if !self.plan_scope.contains(&dependency.tool_id) {
                    return Err(HarnessError::InvalidTool(
                        "plan selects an unavailable tool".into(),
                    ));
                }
                let dependency_tool =
                    self.runner.tools.get(&dependency.tool_id).ok_or_else(|| {
                        HarnessError::InvalidTool("plan selects an unavailable tool".into())
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
        plan_attempt: u32,
    ) -> Result<Option<Vec<PreparedNode>>, HarnessError> {
        self.static_plan_preflight(plan)?;
        let scope = self.plan_scope.clone();
        let broker = ToolBroker::new(
            &self.runner.tools,
            &scope,
            &self.runner.policy,
            &self.runner.approvals,
            &self.runner.concurrency,
        );
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
            let started = StdInstant::now();
            let attempts_before = self.broker_state.tool_calls;
            let classified_before = self.broker_state.classified_tool_calls();
            let outcome = broker
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
                .await;
            match outcome {
                Ok(PrepareOutcome::Ready(prepared)) => prepared_nodes.push(PreparedNode {
                    node: node.clone(),
                    prepared: Some(*prepared),
                    reused: None,
                    event_id,
                }),
                Ok(PrepareOutcome::Reused(result)) => prepared_nodes.push(PreparedNode {
                    node: node.clone(),
                    prepared: None,
                    reused: Some((call, result)),
                    event_id,
                }),
                Ok(PrepareOutcome::Rejected(_)) => {
                    self.events.emit(RunEvent::PlanNodeCompleted {
                        node_id: event_id,
                        tool_id: node.tool_id.clone(),
                        attempt: plan_attempt,
                        wave: 0,
                        ok: false,
                        outcome: PlanNodeOutcome::Rejected,
                        duration_ms: elapsed_millis(started),
                    });
                    return Err(HarnessError::InvalidRequest(format!(
                        "plan node '{}' failed execution preflight",
                        node.id
                    )));
                }
                Ok(PrepareOutcome::Stop) => {
                    self.events.emit(RunEvent::PlanNodeCompleted {
                        node_id: event_id,
                        tool_id: node.tool_id.clone(),
                        attempt: plan_attempt,
                        wave: 0,
                        ok: false,
                        outcome: node_outcome_for_status(&self.result.status),
                        duration_ms: elapsed_millis(started),
                    });
                    return Ok(None);
                }
                Err(error) => {
                    if self.broker_state.tool_calls > attempts_before
                        && self.broker_state.classified_tool_calls() == classified_before
                    {
                        self.broker_state.record_pre_dispatch_error(&error);
                    }
                    self.events.emit(RunEvent::PlanNodeCompleted {
                        node_id: event_id,
                        tool_id: node.tool_id.clone(),
                        attempt: plan_attempt,
                        wave: 0,
                        ok: false,
                        outcome: node_outcome_for_error(&error),
                        duration_ms: elapsed_millis(started),
                    });
                    return Err(error);
                }
            }
        }
        self.events.emit(RunEvent::PlanValidated {
            attempt: plan_attempt,
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
        let scope = self.plan_scope.clone();
        let broker = ToolBroker::new(
            &self.runner.tools,
            &scope,
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
            let started = StdInstant::now();
            let arguments = match bind_arguments(
                &node.node,
                completed,
                self.request.agent.limits.max_tool_arguments_bytes,
            ) {
                Ok(arguments) => arguments,
                Err(error) => {
                    self.broker_state.record_pre_dispatch_error(&error);
                    self.events.emit(RunEvent::PlanNodeCompleted {
                        node_id: node.event_id.clone(),
                        tool_id: node.node.tool_id.clone(),
                        attempt: self.plan_attempt,
                        wave: 0,
                        ok: false,
                        outcome: node_outcome_for_error(&error),
                        duration_ms: elapsed_millis(started),
                    });
                    return Err(error);
                }
            };
            let call = node.prepared.as_mut().expect("checked above");
            let classified_before = self.broker_state.classified_tool_calls();
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
                .await;
            match outcome {
                Ok(FinalizeOutcome::Ready) => {}
                Ok(FinalizeOutcome::Reused(result)) => {
                    let exact_call = call.call.clone();
                    node.prepared = None;
                    node.reused = Some((exact_call, result));
                }
                Ok(FinalizeOutcome::Stop) => {
                    self.events.emit(RunEvent::PlanNodeCompleted {
                        node_id: node.event_id.clone(),
                        tool_id: node.node.tool_id.clone(),
                        attempt: self.plan_attempt,
                        wave: 0,
                        ok: false,
                        outcome: node_outcome_for_status(&self.result.status),
                        duration_ms: elapsed_millis(started),
                    });
                    return Ok(false);
                }
                Err(error) => {
                    if self.broker_state.classified_tool_calls() == classified_before {
                        self.broker_state.record_pre_dispatch_error(&error);
                    }
                    self.events.emit(RunEvent::PlanNodeCompleted {
                        node_id: node.event_id.clone(),
                        tool_id: node.node.tool_id.clone(),
                        attempt: self.plan_attempt,
                        wave: 0,
                        ok: false,
                        outcome: node_outcome_for_error(&error),
                        duration_ms: elapsed_millis(started),
                    });
                    return Err(error);
                }
            }
        }
        Ok(true)
    }

    async fn execute_plan(&mut self, plan: &ExecutionPlan) -> PlanExecution {
        self.plan_attempt = match self.plan_attempt.checked_add(1) {
            Some(attempt) => attempt,
            None => {
                self.terminate(HarnessError::ResourceLimit(
                    "declarative plan attempt counter overflow".into(),
                ));
                return PlanExecution {
                    completed: BTreeMap::new(),
                    transcript: Vec::new(),
                    failure: Some(PlanFailureKind::Terminal),
                    effects_started: false,
                };
            }
        };
        let plan_attempt = self.plan_attempt;
        self.events.emit(RunEvent::PlanLifecycle {
            phase: PlanPhase::Preflight,
            attempt: plan_attempt,
            outcome: PlanLifecycleOutcome::Started,
        });
        let mut prepared = match self.preflight_plan(plan, plan_attempt).await {
            Ok(Some(prepared)) => {
                self.events.emit(RunEvent::PlanLifecycle {
                    phase: PlanPhase::Preflight,
                    attempt: plan_attempt,
                    outcome: PlanLifecycleOutcome::Succeeded,
                });
                prepared
            }
            Ok(None) => {
                self.events.emit(RunEvent::PlanLifecycle {
                    phase: PlanPhase::Preflight,
                    attempt: plan_attempt,
                    outcome: lifecycle_outcome_for_status(&self.result.status),
                });
                self.terminal = true;
                return PlanExecution {
                    completed: BTreeMap::new(),
                    transcript: Vec::new(),
                    failure: Some(PlanFailureKind::Terminal),
                    effects_started: false,
                };
            }
            Err(error) => {
                self.events.emit(RunEvent::PlanLifecycle {
                    phase: PlanPhase::Preflight,
                    attempt: plan_attempt,
                    outcome: lifecycle_outcome_for_error(&error),
                });
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

            let scope = self.plan_scope.clone();
            let broker = ToolBroker::new(
                &self.runner.tools,
                &scope,
                &self.runner.policy,
                &self.runner.approvals,
                &self.runner.concurrency,
            );
            let mut executable = Vec::new();
            for &index in &wave {
                let node_started = StdInstant::now();
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
                        elapsed_millis(node_started),
                    ) {
                        self.events.emit(RunEvent::PlanNodeCompleted {
                            node_id: prepared[index].event_id.clone(),
                            tool_id: prepared[index].node.tool_id.clone(),
                            attempt: plan_attempt,
                            wave: wave_number,
                            ok: false,
                            outcome: node_outcome_for_error(&error),
                            duration_ms: elapsed_millis(node_started),
                        });
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
                        self.broker_state.record_pre_dispatch_error(&error);
                        self.events.emit(RunEvent::PlanNodeCompleted {
                            node_id: prepared[index].event_id.clone(),
                            tool_id: prepared[index].node.tool_id.clone(),
                            attempt: plan_attempt,
                            wave: wave_number,
                            ok: false,
                            outcome: node_outcome_for_error(&error),
                            duration_ms: elapsed_millis(node_started),
                        });
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
                    let error =
                        HarnessError::Tool("prepared plan node is missing its invocation".into());
                    self.broker_state.record_pre_dispatch_error(&error);
                    self.events.emit(RunEvent::PlanNodeCompleted {
                        node_id: event_id,
                        tool_id,
                        attempt: plan_attempt,
                        wave: wave_number,
                        ok: false,
                        outcome: PlanNodeOutcome::Failed,
                        duration_ms: elapsed_millis(node_started),
                    });
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
                let classified_before = self.broker_state.classified_tool_calls();
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
                            elapsed_millis(node_started),
                        ) {
                            self.events.emit(RunEvent::PlanNodeCompleted {
                                node_id: prepared[index].event_id.clone(),
                                tool_id: prepared[index].node.tool_id.clone(),
                                attempt: plan_attempt,
                                wave: wave_number,
                                ok: false,
                                outcome: node_outcome_for_error(&error),
                                duration_ms: elapsed_millis(node_started),
                            });
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
                        self.events.emit(RunEvent::PlanNodeCompleted {
                            node_id: prepared[index].event_id.clone(),
                            tool_id: prepared[index].node.tool_id.clone(),
                            attempt: plan_attempt,
                            wave: wave_number,
                            ok: false,
                            outcome: node_outcome_for_status(&self.result.status),
                            duration_ms: elapsed_millis(node_started),
                        });
                        self.terminal = true;
                        return PlanExecution {
                            completed,
                            transcript,
                            failure: Some(PlanFailureKind::Terminal),
                            effects_started,
                        };
                    }
                    Err(error) => {
                        if self.broker_state.classified_tool_calls() == classified_before {
                            self.broker_state.record_pre_dispatch_error(&error);
                        }
                        self.events.emit(RunEvent::PlanNodeCompleted {
                            node_id: prepared[index].event_id.clone(),
                            tool_id: prepared[index].node.tool_id.clone(),
                            attempt: plan_attempt,
                            wave: wave_number,
                            ok: false,
                            outcome: node_outcome_for_error(&error),
                            duration_ms: elapsed_millis(node_started),
                        });
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

            let budget_check_started = StdInstant::now();
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
                let index = executable[0];
                self.events.emit(RunEvent::PlanNodeCompleted {
                    node_id: prepared[index].event_id.clone(),
                    tool_id: prepared[index].node.tool_id.clone(),
                    attempt: plan_attempt,
                    wave: wave_number,
                    ok: false,
                    outcome: PlanNodeOutcome::LimitReached,
                    duration_ms: elapsed_millis(budget_check_started),
                });
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
                    attempt: plan_attempt,
                    wave: wave_number,
                });
            }
            let request = self.request;
            let deadline = self.deadline;
            let executions = join_all(executable.iter().map(|&index| {
                let prepared_call = prepared[index].prepared.as_ref().expect("checked above");
                let broker = &broker;
                async move {
                    let started = StdInstant::now();
                    let outcome = broker.execute(prepared_call, request, deadline).await;
                    (outcome, elapsed_millis(started))
                }
            }))
            .await;

            let mut wave_failure = None;
            for (&index, (execution, elapsed_ms)) in executable.iter().zip(executions) {
                let call = prepared[index].prepared.as_ref().expect("checked above");
                match execution {
                    Ok(execution) => {
                        self.events.emit(RunEvent::ToolCompleted {
                            call_id: call.call.id.clone(),
                            tool_id: call.call.tool_id.clone(),
                            ok: execution.result.ok,
                        });
                        broker.record_execution(&mut self.broker_state, call, &execution);
                        if let Some(error) = execution.validation_error {
                            self.events.emit(RunEvent::PlanNodeCompleted {
                                node_id: prepared[index].event_id.clone(),
                                tool_id: call.call.tool_id.clone(),
                                attempt: plan_attempt,
                                wave: wave_number,
                                ok: false,
                                outcome: PlanNodeOutcome::Rejected,
                                duration_ms: execution.duration_ms,
                            });
                            self.result.errors.push(error);
                            wave_failure.get_or_insert(PlanFailureKind::Recoverable);
                        } else if !execution.result.ok {
                            self.events.emit(RunEvent::PlanNodeCompleted {
                                node_id: prepared[index].event_id.clone(),
                                tool_id: call.call.tool_id.clone(),
                                attempt: plan_attempt,
                                wave: wave_number,
                                ok: false,
                                outcome: PlanNodeOutcome::Failed,
                                duration_ms: execution.duration_ms,
                            });
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
                                    self.events.emit(RunEvent::PlanNodeCompleted {
                                        node_id: prepared[index].event_id.clone(),
                                        tool_id: call.call.tool_id.clone(),
                                        attempt: plan_attempt,
                                        wave: wave_number,
                                        ok: true,
                                        outcome: PlanNodeOutcome::Succeeded,
                                        duration_ms: execution.duration_ms,
                                    });
                                    completed.insert(
                                        prepared[index].node.id.clone(),
                                        Arc::clone(&execution.result),
                                    );
                                    transcript.push((call.call.clone(), execution.result));
                                    done.insert(prepared[index].node.id.clone());
                                }
                                Err(error) => {
                                    self.events.emit(RunEvent::PlanNodeCompleted {
                                        node_id: prepared[index].event_id.clone(),
                                        tool_id: call.call.tool_id.clone(),
                                        attempt: plan_attempt,
                                        wave: wave_number,
                                        ok: false,
                                        outcome: node_outcome_for_error(&error),
                                        duration_ms: execution.duration_ms,
                                    });
                                    self.terminate(error);
                                    wave_failure = Some(PlanFailureKind::Terminal);
                                }
                            }
                        }
                    }
                    Err(error) => {
                        self.broker_state.record_execution_error(&error);
                        broker.mark_uncertain(&mut self.broker_state, call);
                        self.events.emit(RunEvent::ToolCompleted {
                            call_id: call.call.id.clone(),
                            tool_id: call.call.tool_id.clone(),
                            ok: false,
                        });
                        self.events.emit(RunEvent::PlanNodeCompleted {
                            node_id: prepared[index].event_id.clone(),
                            tool_id: call.call.tool_id.clone(),
                            attempt: plan_attempt,
                            wave: wave_number,
                            ok: false,
                            outcome: node_outcome_for_error(&error),
                            duration_ms: elapsed_ms,
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

    async fn run_reactive(&mut self, initial_phase: ModelCallPhase) {
        let scope = self.direct_scope.clone();
        let broker = ToolBroker::new(
            &self.runner.tools,
            &scope,
            &self.runner.policy,
            &self.runner.approvals,
            &self.runner.concurrency,
        );
        let mut phase = initial_phase;
        loop {
            if self.terminal {
                return;
            }
            let response = match self
                .complete(
                    self.messages.clone(),
                    self.direct_scope.definitions().to_vec(),
                    phase,
                    false,
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
            phase = ModelCallPhase::Reactive;

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
            let recorded_calls = self.runner.tool_calls_for_transcript(
                self.request,
                &self.direct_scope,
                &response.tool_calls,
            );
            self.messages
                .push(Message::assistant_tool_calls(recorded_calls));
            if let Err(error) = ensure_transcript(&self.messages, &self.request.agent.limits) {
                self.terminate(error);
                return;
            }

            for call in response.tool_calls {
                let attempts_before = self.broker_state.tool_calls;
                let classified_before = self.broker_state.classified_tool_calls();
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
                        if self.broker_state.tool_calls > attempts_before
                            && self.broker_state.classified_tool_calls() == classified_before
                        {
                            self.broker_state.record_pre_dispatch_error(&error);
                        }
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
                                    self.broker_state.record_execution_error(&error);
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
        debug_assert_eq!(
            self.model_calls,
            self.planning_model_calls
                + self.repair_model_calls
                + self.recovery_model_calls
                + self.final_synthesis_model_calls
                + self.reactive_model_calls
        );
        self.broker_state.finalize_usage();
        self.events.emit(RunEvent::StrategyUsage {
            strategy: self.selected,
            model_calls: self.model_calls,
            planning_model_calls: self.planning_model_calls,
            repair_model_calls: self.repair_model_calls,
            recovery_model_calls: self.recovery_model_calls,
            final_synthesis_model_calls: self.final_synthesis_model_calls,
            reactive_model_calls: self.reactive_model_calls,
            tool_calls: self.broker_state.tool_calls,
            tool_issued: self.broker_state.tool_issued,
            tool_reused: self.broker_state.tool_reused,
            tool_rejected: self.broker_state.tool_rejected,
            tool_pre_dispatch_aborted: self.broker_state.tool_pre_dispatch_aborted,
            tool_completed: self.broker_state.tool_completed,
            tool_failed: self.broker_state.tool_failed,
            tool_cancelled: self.broker_state.tool_cancelled,
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

fn lifecycle_outcome_for_error(error: &HarnessError) -> PlanLifecycleOutcome {
    match error {
        HarnessError::Cancelled => PlanLifecycleOutcome::Cancelled,
        HarnessError::TimedOut(_) => PlanLifecycleOutcome::TimedOut,
        HarnessError::ResourceLimit(_) => PlanLifecycleOutcome::LimitReached,
        HarnessError::InvalidRequest(_)
        | HarnessError::InvalidTool(_)
        | HarnessError::InvalidArguments(_)
        | HarnessError::Policy(_)
        | HarnessError::Approval(_)
        | HarnessError::InvalidOutput(_) => PlanLifecycleOutcome::Rejected,
        _ => PlanLifecycleOutcome::Failed,
    }
}

fn validation_outcome_for_error(error: &HarnessError) -> PlanLifecycleOutcome {
    match error {
        HarnessError::InvalidOutput(_) => PlanLifecycleOutcome::Invalid,
        _ => lifecycle_outcome_for_error(error),
    }
}

fn lifecycle_outcome_for_status(status: &RunStatus) -> PlanLifecycleOutcome {
    match status {
        RunStatus::Completed => PlanLifecycleOutcome::Succeeded,
        RunStatus::Cancelled => PlanLifecycleOutcome::Cancelled,
        RunStatus::LimitReached => PlanLifecycleOutcome::LimitReached,
        RunStatus::Failed => PlanLifecycleOutcome::Failed,
    }
}

fn recovery_execution_outcome(
    execution: &PlanExecution,
    run: &StrategyRun<'_>,
) -> PlanLifecycleOutcome {
    match execution.failure {
        None => PlanLifecycleOutcome::Succeeded,
        Some(PlanFailureKind::Preflight) => PlanLifecycleOutcome::Rejected,
        Some(PlanFailureKind::Recoverable) => PlanLifecycleOutcome::Failed,
        Some(PlanFailureKind::Terminal) => lifecycle_outcome_for_status(&run.result.status),
    }
}

fn node_outcome_for_error(error: &HarnessError) -> PlanNodeOutcome {
    match error {
        HarnessError::Cancelled => PlanNodeOutcome::Cancelled,
        HarnessError::TimedOut(_) => PlanNodeOutcome::TimedOut,
        HarnessError::ResourceLimit(_) => PlanNodeOutcome::LimitReached,
        HarnessError::InvalidRequest(_)
        | HarnessError::InvalidTool(_)
        | HarnessError::InvalidArguments(_)
        | HarnessError::Policy(_)
        | HarnessError::Approval(_)
        | HarnessError::InvalidOutput(_) => PlanNodeOutcome::Rejected,
        _ => PlanNodeOutcome::Failed,
    }
}

fn node_outcome_for_status(status: &RunStatus) -> PlanNodeOutcome {
    match status {
        RunStatus::Completed => PlanNodeOutcome::Succeeded,
        RunStatus::Cancelled => PlanNodeOutcome::Cancelled,
        RunStatus::LimitReached => PlanNodeOutcome::LimitReached,
        RunStatus::Failed => PlanNodeOutcome::Failed,
    }
}

fn elapsed_millis(started: StdInstant) -> u64 {
    started.elapsed().as_millis() as u64
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

#[cfg(test)]
mod discovery_terminal_tests {
    use super::*;
    use crate::{
        mock::{final_response, MockModelProvider},
        InMemoryEventSink, ModelCapabilities, ProviderCapabilityLimits, Tool, ToolDiscoveryLimits,
        ToolDiscoveryMetadata, ToolRegistry,
    };
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::{
        atomic::{AtomicU32, AtomicUsize, Ordering},
        Arc, Barrier,
    };
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    struct CountingTool {
        definition: ToolDefinition,
        calls: AtomicU32,
    }

    #[async_trait]
    impl Tool for CountingTool {
        fn definition(&self) -> &ToolDefinition {
            &self.definition
        }

        async fn execute(
            &self,
            _: Value,
            _: CancellationToken,
        ) -> Result<ToolResult, HarnessError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult::success(json!({"ok": true})))
        }
    }

    #[derive(Clone, Copy)]
    enum StopKind {
        Cancel,
        Deadline,
    }

    fn planning_capabilities() -> ModelCapabilities {
        ModelCapabilities::new(true, false, true)
            .with_structured_plans(true)
            .with_limits(
                ProviderCapabilityLimits::new()
                    .with_max_plan_nodes(64)
                    .with_max_plan_bytes(256 * 1024),
            )
    }

    fn registry_with_checkpoint(
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
        paused_caller: ToolCaller,
    ) -> (ToolRegistry, Vec<Arc<CountingTool>>) {
        let mut registry = ToolRegistry::default();
        let mut tools = Vec::new();
        for index in 0..100 {
            let id = format!("discovery.tool.{index:03}");
            let tool = Arc::new(CountingTool {
                definition: ToolDefinition::new(
                    &id,
                    id.replace('.', " "),
                    "discovery terminal test tool",
                    json!({"type": "object"}),
                )
                .with_allowed_callers([ToolCaller::Direct, ToolCaller::DeclarativePlan]),
                calls: AtomicU32::new(0),
            });
            registry
                .register_with_discovery(tool.clone(), ToolDiscoveryMetadata::deferred())
                .unwrap();
            tools.push(tool);
        }
        let checkpoints = Arc::new(AtomicUsize::new(0));
        registry.set_discovery_checkpoint(Arc::new(move |caller| {
            if caller == paused_caller && checkpoints.fetch_add(1, Ordering::SeqCst) == 1 {
                entered.wait();
                release.wait();
            }
        }));
        (registry, tools)
    }

    fn assert_zero_usage(event: &RunEvent, strategy: RunStrategy) {
        let RunEvent::StrategyUsage {
            strategy: actual,
            model_calls,
            planning_model_calls,
            repair_model_calls,
            recovery_model_calls,
            final_synthesis_model_calls,
            reactive_model_calls,
            tool_calls,
            tool_issued,
            tool_reused,
            tool_rejected,
            tool_pre_dispatch_aborted,
            tool_completed,
            tool_failed,
            tool_cancelled,
            duration_ms,
        } = event
        else {
            panic!("expected strategy usage event");
        };
        assert_eq!(*actual, strategy);
        assert_eq!(
            [
                u64::from(*model_calls),
                u64::from(*planning_model_calls),
                u64::from(*repair_model_calls),
                u64::from(*recovery_model_calls),
                u64::from(*final_synthesis_model_calls),
                u64::from(*reactive_model_calls),
                u64::from(*tool_calls),
                u64::from(*tool_issued),
                u64::from(*tool_reused),
                u64::from(*tool_rejected),
                u64::from(*tool_pre_dispatch_aborted),
                u64::from(*tool_completed),
                u64::from(*tool_failed),
                u64::from(*tool_cancelled),
                *duration_ms,
            ],
            [0; 15]
        );
        assert_eq!(
            *model_calls,
            *planning_model_calls
                + *repair_model_calls
                + *recovery_model_calls
                + *final_synthesis_model_calls
                + *reactive_model_calls
        );
        assert_eq!(
            *tool_calls,
            *tool_issued + *tool_reused + *tool_rejected + *tool_pre_dispatch_aborted
        );
        assert_eq!(
            *tool_issued,
            *tool_completed + *tool_failed + *tool_cancelled
        );
    }

    fn run_stopped_discovery(
        strategy: RunStrategy,
        stop: StopKind,
        paused_caller: ToolCaller,
        capabilities: ModelCapabilities,
        usage_strategy: RunStrategy,
    ) {
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let (registry, tools) =
            registry_with_checkpoint(entered.clone(), release.clone(), paused_caller);
        let provider = Arc::new(
            MockModelProvider::scripted([final_response("unused")]).with_capabilities(capabilities),
        );
        let events = Arc::new(InMemoryEventSink::default());
        let runner = Arc::new(
            AgentRunner::builder(provider.clone())
                .tools(registry)
                .event_sink(events.clone())
                .discovery_limits(ToolDiscoveryLimits::new().with_max_tools(1))
                .build(),
        );
        let mut agent = crate::AgentDefinition::new("discovery", "Discovery", "1", "mock");
        agent.tool_allowlist = (0..100)
            .map(|index| format!("discovery.tool.{index:03}"))
            .collect();
        if matches!(stop, StopKind::Deadline) {
            agent.limits.max_run_duration_ms = Some(500);
        }
        let request = RunRequest::new(agent, "discovery.tool.099")
            .with_run_id(format!("discovery-{strategy:?}"))
            .with_trace_id(format!("trace-{strategy:?}"));
        let cancellation = request.cancellation.clone();
        let run = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .unwrap()
                .block_on(runner.run_with_strategy(request, strategy))
        });

        entered.wait();
        match stop {
            StopKind::Cancel => cancellation.cancel(),
            StopKind::Deadline => std::thread::sleep(Duration::from_millis(550)),
        }
        release.wait();
        let result = run.join().unwrap().unwrap();

        match stop {
            StopKind::Cancel => {
                assert_eq!(result.status, RunStatus::Cancelled);
                assert!(result.cancelled);
                assert_eq!(result.errors.len(), 1);
                assert_eq!(result.errors[0].code, "cancelled");
            }
            StopKind::Deadline => {
                assert_eq!(result.status, RunStatus::Failed);
                assert!(!result.cancelled);
                assert_eq!(result.errors.len(), 1);
                assert_eq!(result.errors[0].code, "timed_out");
            }
        }
        assert!(result.tool_calls.is_empty());
        assert!(result.policy_decisions.is_empty());
        assert!(result.approvals.is_empty());
        assert!(provider.requests().is_empty());
        assert_eq!(
            tools
                .iter()
                .map(|tool| tool.calls.load(Ordering::SeqCst))
                .sum::<u32>(),
            0
        );

        let records = events.events();
        assert_eq!(records.len(), 3);
        assert_eq!(
            records
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(matches!(records[0].event, RunEvent::Started { .. }));
        assert_zero_usage(&records[1].event, usage_strategy);
        assert!(matches!(
            records[2].event,
            RunEvent::Completed { ref status } if status == &result.status
        ));
        assert!(!records.iter().any(|record| matches!(
            record.event,
            RunEvent::ToolDiscoveryCompleted { .. }
                | RunEvent::ModelRequested { .. }
                | RunEvent::PolicyDecided { .. }
                | RunEvent::ApprovalRequested { .. }
                | RunEvent::ToolEffectReused { .. }
                | RunEvent::ToolCompleted { .. }
        )));
    }

    #[test]
    fn discovery_stop_is_a_terminal_result_for_every_runnable_strategy() {
        for strategy in [
            RunStrategy::Direct,
            RunStrategy::Adaptive,
            RunStrategy::DeclarativePlan,
        ] {
            let paused_caller = if strategy == RunStrategy::Direct {
                ToolCaller::Direct
            } else {
                ToolCaller::DeclarativePlan
            };
            run_stopped_discovery(
                strategy,
                StopKind::Cancel,
                paused_caller,
                planning_capabilities(),
                strategy,
            );
            run_stopped_discovery(
                strategy,
                StopKind::Deadline,
                paused_caller,
                planning_capabilities(),
                strategy,
            );
        }
    }

    #[test]
    fn planned_strategy_direct_scope_stop_has_the_same_terminal_contract() {
        for strategy in [RunStrategy::Adaptive, RunStrategy::DeclarativePlan] {
            run_stopped_discovery(
                strategy,
                StopKind::Cancel,
                ToolCaller::Direct,
                planning_capabilities(),
                strategy,
            );
            run_stopped_discovery(
                strategy,
                StopKind::Deadline,
                ToolCaller::Direct,
                planning_capabilities(),
                strategy,
            );
        }
    }

    #[test]
    fn adaptive_downgrade_discovery_stop_reports_direct_zero_usage() {
        for stop in [StopKind::Cancel, StopKind::Deadline] {
            run_stopped_discovery(
                RunStrategy::Adaptive,
                stop,
                ToolCaller::Direct,
                ModelCapabilities::new(true, false, true),
                RunStrategy::Direct,
            );
        }
    }
}
