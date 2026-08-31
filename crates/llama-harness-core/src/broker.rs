use crate::{
    discovery::ToolScope,
    limits::{ensure_json_depth, serialized_len},
    runner::{await_guarded, check_stopped},
    ApprovalHandler, ApprovalRecord, HarnessError, PolicyDecision, PolicyEngine, RunError,
    RunEvent, RunRequest, RunResult, RunStatus, Tool, ToolCall, ToolCallContext, ToolCaller,
    ToolRegistry, ToolResult,
};
use serde_json::Value;
use std::{
    collections::HashMap,
    future::Future,
    sync::{Arc, Weak},
    time::Instant as StdInstant,
};
use tokio::{
    sync::{Mutex, OwnedSemaphorePermit, Semaphore},
    time::Instant,
};

use crate::event::EventEmitter;

/// A dispatched tool has a short, bounded opportunity to observe cooperative
/// cancellation before its keyed-concurrency permit is released. This does not
/// make an interrupted effect retryable: callers still record it as uncertain.
const TOOL_CLEANUP_GRACE: std::time::Duration = std::time::Duration::from_millis(250);

/// Runner-wide keyed permits shared by direct and planned invocations.
#[derive(Default)]
pub(crate) struct ToolConcurrencyLimiter {
    keyed: Mutex<HashMap<String, Weak<Semaphore>>>,
}

impl ToolConcurrencyLimiter {
    async fn acquire(&self, key: &str) -> Result<OwnedSemaphorePermit, HarnessError> {
        let semaphore = {
            let mut keyed = self.keyed.lock().await;
            keyed.retain(|_, semaphore| semaphore.strong_count() > 0);
            if let Some(semaphore) = keyed.get(key).and_then(Weak::upgrade) {
                semaphore
            } else {
                let semaphore = Arc::new(Semaphore::new(1));
                keyed.insert(key.to_owned(), Arc::downgrade(&semaphore));
                semaphore
            }
        };
        semaphore
            .acquire_owned()
            .await
            .map_err(|_| HarnessError::Tool("tool concurrency permit closed".into()))
    }
}

/// Per-run counters and effect records shared by every tool-calling strategy.
#[derive(Default)]
pub(crate) struct BrokerState {
    /// Tool proposals admitted under the total-call limit.
    pub(crate) tool_calls: u32,
    pub(crate) tool_issued: u32,
    pub(crate) tool_reused: u32,
    pub(crate) tool_rejected: u32,
    pub(crate) tool_pre_dispatch_aborted: u32,
    pub(crate) tool_completed: u32,
    pub(crate) tool_failed: u32,
    pub(crate) tool_cancelled: u32,
    identical_calls: HashMap<String, u32>,
    effects: HashMap<String, EffectRecord>,
    reuse_committed_effects: bool,
}

impl BrokerState {
    pub(crate) fn enable_effect_reuse(&mut self) {
        self.reuse_committed_effects = true;
    }

    pub(crate) fn recovery_is_safe(&self) -> bool {
        self.effects
            .values()
            .all(|record| matches!(record, EffectRecord::Completed(_)))
    }

    pub(crate) fn record_pre_dispatch_error(&mut self, error: &HarnessError) {
        if matches!(error, HarnessError::Cancelled | HarnessError::TimedOut(_)) {
            self.tool_pre_dispatch_aborted += 1;
        } else {
            self.tool_rejected += 1;
        }
    }

    pub(crate) fn classified_tool_calls(&self) -> u32 {
        self.tool_issued + self.tool_reused + self.tool_rejected + self.tool_pre_dispatch_aborted
    }

    pub(crate) fn record_execution_error(&mut self, error: &HarnessError) {
        if matches!(error, HarnessError::Cancelled | HarnessError::TimedOut(_)) {
            self.tool_cancelled += 1;
        } else {
            self.tool_failed += 1;
        }
    }

    pub(crate) fn finalize_usage(&mut self) {
        let classified = self.classified_tool_calls();
        debug_assert!(classified <= self.tool_calls);
        self.tool_pre_dispatch_aborted += self.tool_calls.saturating_sub(classified);
        debug_assert_eq!(
            self.tool_calls,
            self.tool_issued
                + self.tool_reused
                + self.tool_rejected
                + self.tool_pre_dispatch_aborted
        );
        debug_assert_eq!(
            self.tool_issued,
            self.tool_completed + self.tool_failed + self.tool_cancelled
        );
    }
}

#[derive(Clone)]
enum EffectRecord {
    Dispatched,
    Uncertain,
    Completed(Arc<ToolResult>),
}

/// A completely validated and authorized invocation ready to execute.
pub(crate) struct PreparedCall {
    pub(crate) call: ToolCall,
    pub(crate) arguments: Value,
    pub(crate) tool: Arc<dyn Tool>,
    pub(crate) context: ToolCallContext,
    signature: Option<String>,
    effect_key: Option<String>,
    caller: ToolCaller,
    authorized_signature: Option<String>,
    signature_accounted: bool,
    approval_barrier: bool,
}

/// Result of preparing one invocation at the shared broker boundary.
pub(crate) enum PrepareOutcome {
    Ready(Box<PreparedCall>),
    Rejected(ToolResult),
    Reused(Arc<ToolResult>),
    Stop,
}

/// Result of resolving and authorizing a call whose arguments contain bindings.
pub(crate) enum FinalizeOutcome {
    Ready,
    Reused(Arc<ToolResult>),
    Stop,
}

/// Result of executing one prepared invocation.
pub(crate) struct BrokerExecution {
    pub(crate) result: Arc<ToolResult>,
    pub(crate) validation_error: Option<RunError>,
    pub(crate) duration_ms: u64,
}

/// Provider-neutral safety boundary used by direct and orchestrated calls.
pub(crate) struct ToolBroker<'a> {
    tools: &'a ToolRegistry,
    scope: &'a ToolScope,
    policy: &'a Arc<dyn PolicyEngine>,
    approvals: &'a Arc<dyn ApprovalHandler>,
    concurrency: &'a Arc<ToolConcurrencyLimiter>,
}

impl<'a> ToolBroker<'a> {
    pub(crate) fn new(
        tools: &'a ToolRegistry,
        scope: &'a ToolScope,
        policy: &'a Arc<dyn PolicyEngine>,
        approvals: &'a Arc<dyn ApprovalHandler>,
        concurrency: &'a Arc<ToolConcurrencyLimiter>,
    ) -> Self {
        Self {
            tools,
            scope,
            policy,
            approvals,
            concurrency,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn prepare(
        &self,
        request: &RunRequest,
        result: &mut RunResult,
        events: &mut EventEmitter,
        state: &mut BrokerState,
        call: ToolCall,
        caller: ToolCaller,
        approval_barrier: bool,
        defer_signature_checks: bool,
        supplied_context: Option<ToolCallContext>,
        deadline: Option<Instant>,
    ) -> Result<PrepareOutcome, HarnessError> {
        let mut call = call;
        check_stopped(
            &request.cancellation,
            deadline,
            "run deadline reached before tool validation",
        )?;
        if state.tool_calls >= request.agent.limits.max_tool_calls {
            result.status = RunStatus::LimitReached;
            result.tool_call_limit_reached = true;
            result
                .errors
                .push(RunError::new("tool_call_limit", "tool call limit reached"));
            return Ok(PrepareOutcome::Stop);
        }
        state.tool_calls += 1;

        if call.arguments_json.len() as u64 > request.agent.limits.max_tool_arguments_bytes {
            self.reject(
                result,
                events,
                state,
                &call,
                "tool arguments exceed byte limit",
            );
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "tool arguments exceed byte limit",
            )));
        }

        let arguments: Value = match serde_json::from_str(&call.arguments_json) {
            Ok(value) => value,
            Err(_) => {
                self.reject(result, events, state, &call, "malformed tool arguments");
                return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                    "malformed tool arguments",
                )));
            }
        };
        if ensure_json_depth(
            "tool arguments",
            &arguments,
            request.agent.limits.max_json_depth,
        )
        .is_err()
        {
            self.reject(
                result,
                events,
                state,
                &call,
                "tool arguments exceed JSON depth limit",
            );
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "tool arguments exceed JSON depth limit",
            )));
        }
        let canonical_arguments = serde_json::to_string(&arguments).map_err(|_| {
            HarnessError::InvalidArguments("tool arguments could not be canonicalized".into())
        })?;
        if canonical_arguments.len() as u64 > request.agent.limits.max_tool_arguments_bytes {
            self.reject(
                result,
                events,
                state,
                &call,
                "tool arguments exceed byte limit",
            );
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "tool arguments exceed byte limit",
            )));
        }
        call.arguments_json = canonical_arguments;

        if !defer_signature_checks {
            let mut recorded_call = self.call_for_transcript(request, &call, caller);
            if caller == ToolCaller::Programmatic {
                recorded_call.arguments_json = "{}".into();
            }
            result.tool_calls.push(recorded_call);
        }

        if !self.scope.contains(&call.tool_id) || self.scope.caller() != caller {
            self.reject(result, events, state, &call, "tool unavailable");
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "tool unavailable",
            )));
        }
        let Some(tool) = self.tools.get(&call.tool_id) else {
            self.reject(result, events, state, &call, "tool unavailable");
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "tool unavailable",
            )));
        };
        if !request
            .agent
            .tool_allowlist
            .iter()
            .any(|id| id == &call.tool_id)
        {
            self.reject(result, events, state, &call, "tool is not allowed");
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "tool is not allowed",
            )));
        }
        if !tool.definition().allows_caller(caller) {
            self.reject(result, events, state, &call, "tool caller is not allowed");
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "tool caller is not allowed",
            )));
        }
        if self.tools.validate(&call.tool_id, &arguments).is_err() {
            self.reject(
                result,
                events,
                state,
                &call,
                "tool arguments failed validation",
            );
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "tool arguments failed validation",
            )));
        }

        let context = match supplied_context {
            Some(context) => context,
            None => {
                let mut context = ToolCallContext::new(
                    result.id.clone(),
                    result.trace_id.clone(),
                    call.id.clone(),
                    call.tool_id.clone(),
                );
                context.caller = Some(caller);
                context
            }
        };
        if context.run_id != result.id
            || context.trace_id != result.trace_id
            || context.call_id != call.id
            || context.tool_id != call.tool_id
            || context.caller != Some(caller)
        {
            return Err(HarnessError::InvalidRequest(
                "tool call context does not match the broker occurrence".into(),
            ));
        }
        let has_programmatic_provenance = context.program_attempt.is_some()
            || context.static_call_site.is_some()
            || context.dynamic_ordinal.is_some()
            || context.effect_key.is_some();
        if caller == ToolCaller::Programmatic {
            if context.program_attempt.is_none()
                || context.static_call_site.is_none()
                || context.dynamic_ordinal.is_none()
                || context.effect_key.as_deref() != Some(call.id.as_str())
            {
                return Err(HarnessError::InvalidRequest(
                    "programmatic tool context is missing occurrence provenance".into(),
                ));
            }
        } else if has_programmatic_provenance {
            return Err(HarnessError::InvalidRequest(
                "non-programmatic tool context contains program occurrence provenance".into(),
            ));
        }

        let signature =
            (!defer_signature_checks).then(|| canonical_signature(&call.tool_id, &arguments));
        if let Some(signature) = &signature {
            if let Some(recorded) = self.reusable_effect(
                state,
                events,
                &call,
                caller,
                tool.definition().read_only,
                signature,
            )? {
                return Ok(PrepareOutcome::Reused(recorded));
            }
            if !self.account_signature(request, result, state, signature) {
                return Ok(PrepareOutcome::Stop);
            }
            if let Some(rejection) = self
                .authorize(
                    request,
                    result,
                    events,
                    state,
                    &call,
                    &context,
                    tool.as_ref(),
                    &arguments,
                    approval_barrier,
                    deadline,
                )
                .await?
            {
                return Ok(PrepareOutcome::Rejected(rejection));
            }
        }

        let authorized_signature = signature.clone();
        let effect_key = if tool.definition().read_only {
            None
        } else if caller == ToolCaller::Programmatic {
            context.effect_key.clone()
        } else {
            signature.clone()
        };
        Ok(PrepareOutcome::Ready(Box::new(PreparedCall {
            call,
            arguments,
            tool,
            context,
            signature,
            effect_key,
            caller,
            authorized_signature,
            signature_accounted: !defer_signature_checks,
            approval_barrier,
        })))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn revalidate_bound_arguments(
        &self,
        prepared: &mut PreparedCall,
        arguments: Value,
        request: &RunRequest,
        result: &mut RunResult,
        events: &mut EventEmitter,
        state: &mut BrokerState,
        deadline: Option<Instant>,
    ) -> Result<FinalizeOutcome, HarnessError> {
        check_stopped(
            &request.cancellation,
            deadline,
            "run deadline reached before bound argument validation",
        )?;
        let arguments_json = serde_json::to_string(&arguments).map_err(|error| {
            HarnessError::InvalidArguments(format!(
                "bound arguments for tool {} could not be serialized: {error}",
                prepared.call.tool_id
            ))
        })?;
        if arguments_json.len() as u64 > request.agent.limits.max_tool_arguments_bytes {
            return Err(HarnessError::ResourceLimit(format!(
                "bound tool arguments exceed {} bytes",
                request.agent.limits.max_tool_arguments_bytes
            )));
        }
        ensure_json_depth(
            "bound tool arguments",
            &arguments,
            request.agent.limits.max_json_depth,
        )?;
        if !prepared.tool.definition().allows_caller(prepared.caller)
            || !self.scope.contains(&prepared.call.tool_id)
            || self.scope.caller() != prepared.caller
            || !request
                .agent
                .tool_allowlist
                .iter()
                .any(|id| id == &prepared.call.tool_id)
        {
            return Err(HarnessError::InvalidTool(
                "bound plan call is no longer allowed".into(),
            ));
        }
        self.tools.validate(&prepared.call.tool_id, &arguments)?;

        let signature = canonical_signature(&prepared.call.tool_id, &arguments);
        if !prepared.signature_accounted {
            prepared.call.arguments_json = arguments_json.clone();
            result.tool_calls.push(prepared.call.clone());
        }
        if let Some(recorded) = self.reusable_effect(
            state,
            events,
            &prepared.call,
            prepared.caller,
            prepared.tool.definition().read_only,
            &signature,
        )? {
            return Ok(FinalizeOutcome::Reused(recorded));
        }
        if !prepared.signature_accounted {
            if !self.account_signature(request, result, state, &signature) {
                return Ok(FinalizeOutcome::Stop);
            }
            prepared.signature_accounted = true;
        }

        if prepared.authorized_signature.as_deref() != Some(signature.as_str()) {
            if self
                .authorize(
                    request,
                    result,
                    events,
                    state,
                    &prepared.call,
                    &prepared.context,
                    prepared.tool.as_ref(),
                    &arguments,
                    prepared.approval_barrier,
                    deadline,
                )
                .await?
                .is_some()
            {
                return Err(HarnessError::Policy(
                    "bound plan call authorization denied".into(),
                ));
            }
            prepared.authorized_signature = Some(signature.clone());
        }

        prepared.arguments = arguments;
        prepared.call.arguments_json = arguments_json;
        prepared.effect_key = (!prepared.tool.definition().read_only).then(|| signature.clone());
        prepared.signature = Some(signature);
        Ok(FinalizeOutcome::Ready)
    }

    fn account_signature(
        &self,
        request: &RunRequest,
        result: &mut RunResult,
        state: &mut BrokerState,
        signature: &str,
    ) -> bool {
        let count = state
            .identical_calls
            .entry(signature.to_owned())
            .or_default();
        *count += 1;
        if *count <= request.agent.limits.max_identical_tool_calls {
            return true;
        }
        result.status = RunStatus::LimitReached;
        result.repeated_tool_call_limit_reached = true;
        state.tool_rejected += 1;
        result.errors.push(RunError::new(
            "repeated_tool_call_limit",
            "repeated identical tool call limit reached",
        ));
        false
    }

    fn reusable_effect(
        &self,
        state: &mut BrokerState,
        events: &mut EventEmitter,
        call: &ToolCall,
        caller: ToolCaller,
        read_only: bool,
        signature: &str,
    ) -> Result<Option<Arc<ToolResult>>, HarnessError> {
        if !state.reuse_committed_effects || read_only || caller == ToolCaller::Programmatic {
            return Ok(None);
        }
        match state.effects.get(signature) {
            Some(EffectRecord::Completed(recorded)) => {
                state.tool_reused += 1;
                events.emit(RunEvent::ToolEffectReused {
                    call_id: call.id.clone(),
                    tool_id: call.tool_id.clone(),
                });
                Ok(Some(recorded.clone()))
            }
            Some(EffectRecord::Dispatched | EffectRecord::Uncertain) => Err(HarnessError::Tool(
                "state-changing tool outcome is uncertain; implicit replay is prohibited".into(),
            )),
            None => Ok(None),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn authorize(
        &self,
        request: &RunRequest,
        result: &mut RunResult,
        events: &mut EventEmitter,
        state: &mut BrokerState,
        call: &ToolCall,
        context: &ToolCallContext,
        tool: &dyn Tool,
        arguments: &Value,
        approval_barrier: bool,
        deadline: Option<Instant>,
    ) -> Result<Option<ToolResult>, HarnessError> {
        let decision = await_guarded(
            self.policy
                .decide_with_context(context, tool.definition(), arguments, request),
            &request.cancellation,
            deadline,
            "policy decision exceeded run deadline",
            None,
        )
        .await?;
        let public_decision = public_policy_decision(&decision);
        events.emit(RunEvent::PolicyDecided {
            call_id: call.id.clone(),
            decision: public_decision.clone(),
        });
        result.policy_decisions.push(public_decision);

        if let PolicyDecision::Deny { .. } = decision {
            self.reject(result, events, state, call, "policy denied");
            return Ok(Some(ToolResult::failure("policy denied")));
        }

        if approval_barrier || matches!(decision, PolicyDecision::RequireApproval { .. }) {
            events.emit(RunEvent::ApprovalRequested {
                call_id: call.id.clone(),
                tool_id: call.tool_id.clone(),
            });
            let mut approval = await_guarded(
                self.approvals
                    .approve_with_context(context, tool.definition(), arguments, request),
                &request.cancellation,
                deadline,
                "approval exceeded run deadline",
                None,
            )
            .await?;
            approval.call_id = call.id.clone();
            approval.tool_id = call.tool_id.clone();
            let granted = approval.granted;
            result.approvals.push(ApprovalRecord {
                call_id: approval.call_id,
                tool_id: approval.tool_id,
                granted,
                reason: if granted {
                    "approval granted".into()
                } else {
                    "approval denied".into()
                },
            });
            if !granted {
                self.reject(result, events, state, call, "approval denied");
                return Ok(Some(ToolResult::failure("approval denied")));
            }
        }

        Ok(None)
    }

    pub(crate) async fn execute(
        &self,
        prepared: &PreparedCall,
        request: &RunRequest,
        deadline: Option<Instant>,
    ) -> Result<BrokerExecution, HarnessError> {
        check_stopped(
            &request.cancellation,
            deadline,
            "run deadline reached before tool invocation",
        )?;
        let started = StdInstant::now();
        let _concurrency_permit = if let Some(key) = &prepared.tool.definition().concurrency_key {
            Some(
                await_guarded(
                    self.concurrency.acquire(key),
                    &request.cancellation,
                    deadline,
                    "tool concurrency wait exceeded run deadline",
                    None,
                )
                .await?,
            )
        } else {
            None
        };
        check_stopped(
            &request.cancellation,
            deadline,
            "run deadline reached while waiting for tool concurrency",
        )?;
        let current = self.tools.get(&prepared.call.tool_id).ok_or_else(|| {
            HarnessError::InvalidTool("tool disappeared before invocation".into())
        })?;
        let canonical_arguments = serde_json::to_string(&prepared.arguments).map_err(|_| {
            HarnessError::InvalidArguments(
                "prepared tool arguments could not be canonicalized".into(),
            )
        })?;
        let current_signature = canonical_signature(&prepared.call.tool_id, &prepared.arguments);
        if !Arc::ptr_eq(&current, &prepared.tool)
            || !self.scope.contains(&prepared.call.tool_id)
            || self.scope.caller() != prepared.caller
            || !prepared.tool.definition().allows_caller(prepared.caller)
            || prepared.call.arguments_json != canonical_arguments
            || prepared.signature.as_deref() != Some(current_signature.as_str())
            || prepared.authorized_signature.as_deref() != Some(current_signature.as_str())
        {
            return Err(HarnessError::Policy(
                "tool identity or bound authorization changed before invocation".into(),
            ));
        }
        let tool_cancellation = request.cancellation.child_token();
        let execution = prepared.tool.execute_with_context(
            &prepared.context,
            prepared.arguments.clone(),
            tool_cancellation.clone(),
        );
        let mut tool_result = await_dispatched_tool(
            execution,
            &request.cancellation,
            deadline,
            "tool execution exceeded run deadline",
            &tool_cancellation,
        )
        .await?;

        if serialized_len(&tool_result)? > request.agent.limits.max_tool_result_bytes {
            return Err(HarnessError::ResourceLimit(format!(
                "tool result exceeds {} bytes",
                request.agent.limits.max_tool_result_bytes
            )));
        }
        ensure_json_depth(
            "tool result",
            &tool_result.output,
            request.agent.limits.max_json_depth,
        )?;

        let validation_error = if tool_result.ok {
            self.tools
                .validate_output(&prepared.call.tool_id, &tool_result.output)
                .err()
                .map(|_| {
                    tool_result = ToolResult::failure("tool output failed validation");
                    RunError::new("tool_error", "tool output failed validation")
                })
        } else {
            None
        };

        Ok(BrokerExecution {
            result: Arc::new(tool_result),
            validation_error,
            duration_ms: started.elapsed().as_millis() as u64,
        })
    }

    pub(crate) fn mark_dispatched(&self, state: &mut BrokerState, prepared: &PreparedCall) {
        state.tool_issued += 1;
        if prepared.tool.definition().read_only {
            return;
        }
        if let Some(signature) = &prepared.effect_key {
            state
                .effects
                .insert(signature.clone(), EffectRecord::Dispatched);
        }
    }

    pub(crate) fn mark_uncertain(&self, state: &mut BrokerState, prepared: &PreparedCall) {
        if prepared.tool.definition().read_only {
            return;
        }
        if let Some(signature) = &prepared.effect_key {
            state
                .effects
                .insert(signature.clone(), EffectRecord::Uncertain);
        }
    }

    pub(crate) fn record_execution(
        &self,
        state: &mut BrokerState,
        prepared: &PreparedCall,
        execution: &BrokerExecution,
    ) {
        if execution.result.ok && execution.validation_error.is_none() {
            state.tool_completed += 1;
        } else {
            state.tool_failed += 1;
        }
        if prepared.tool.definition().read_only {
            return;
        }
        if let Some(signature) = &prepared.effect_key {
            let record = if execution.result.ok && execution.validation_error.is_none() {
                EffectRecord::Completed(Arc::clone(&execution.result))
            } else {
                EffectRecord::Uncertain
            };
            state.effects.insert(signature.clone(), record);
        }
    }

    fn reject(
        &self,
        result: &mut RunResult,
        events: &mut EventEmitter,
        state: &mut BrokerState,
        call: &ToolCall,
        reason: &'static str,
    ) {
        state.tool_rejected += 1;
        events.emit(RunEvent::ToolRejected {
            call_id: call.id.clone(),
            tool_id: call.tool_id.clone(),
            reason: reason.into(),
        });
        result.errors.push(RunError::new("tool_rejected", reason));
    }

    fn call_for_transcript(
        &self,
        request: &RunRequest,
        call: &ToolCall,
        caller: ToolCaller,
    ) -> ToolCall {
        let mut recorded = call.clone();
        if !self.scope.contains(&call.tool_id) || self.scope.caller() != caller {
            recorded.arguments_json = "{}".into();
            return recorded;
        }
        if !self.arguments_are_valid(request, call, caller) {
            recorded.arguments_json = "{}".into();
        }
        recorded
    }

    fn arguments_are_valid(
        &self,
        request: &RunRequest,
        call: &ToolCall,
        caller: ToolCaller,
    ) -> bool {
        if !self.scope.contains(&call.tool_id) || self.scope.caller() != caller {
            return false;
        }
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
        let Some(tool) = self.tools.get(&call.tool_id) else {
            return false;
        };
        request
            .agent
            .tool_allowlist
            .iter()
            .any(|id| id == &call.tool_id)
            && tool.definition().allows_caller(caller)
            && self.tools.validate(&call.tool_id, &arguments).is_ok()
    }
}

fn public_policy_decision(decision: &PolicyDecision) -> PolicyDecision {
    match decision {
        PolicyDecision::Allow { .. } => PolicyDecision::Allow {
            reason: "policy allowed the call".into(),
        },
        PolicyDecision::Deny { .. } => PolicyDecision::Deny {
            reason: "policy denied the call".into(),
        },
        PolicyDecision::RequireApproval { .. } => PolicyDecision::RequireApproval {
            reason: "policy requires approval".into(),
        },
    }
}

/// Awaits a tool after the broker has marked its effect as dispatched.
///
/// Unlike [`await_guarded`], cancellation and deadline expiry do not drop the
/// active future immediately. The child token is signalled first and the same
/// future is polled through a bounded cleanup grace so cooperative tools can
/// quiesce while keyed-concurrency admission remains held. A non-cooperative
/// future is still dropped after that grace; the terminal error ensures the
/// caller records the potentially active effect as uncertain and never replays
/// or recovers it.
async fn await_dispatched_tool<T, F>(
    future: F,
    cancellation: &tokio_util::sync::CancellationToken,
    deadline: Option<Instant>,
    timeout_message: &str,
    child_cancellation: &tokio_util::sync::CancellationToken,
) -> Result<T, HarnessError>
where
    F: Future<Output = Result<T, HarnessError>>,
{
    enum AwaitOutcome<T> {
        Completed(Result<T, HarnessError>),
        Cancelled,
        TimedOut,
    }

    tokio::pin!(future);
    let outcome = if let Some(deadline) = deadline {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => AwaitOutcome::Cancelled,
            _ = tokio::time::sleep_until(deadline) => {
                AwaitOutcome::TimedOut
            },
            result = &mut future => AwaitOutcome::Completed(result),
        }
    } else {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => AwaitOutcome::Cancelled,
            result = &mut future => AwaitOutcome::Completed(result),
        }
    };

    match outcome {
        AwaitOutcome::Completed(result) => result,
        AwaitOutcome::Cancelled => {
            child_cancellation.cancel();
            let _ = tokio::time::timeout(TOOL_CLEANUP_GRACE, &mut future).await;
            Err(HarnessError::Cancelled)
        }
        AwaitOutcome::TimedOut => {
            child_cancellation.cancel();
            let _ = tokio::time::timeout(TOOL_CLEANUP_GRACE, &mut future).await;
            Err(HarnessError::TimedOut(timeout_message.into()))
        }
    }
}

pub(crate) fn canonical_signature(tool_id: &str, arguments: &Value) -> String {
    format!(
        "{}:{}",
        tool_id,
        serde_json::to_string(arguments).unwrap_or_default()
    )
}
