use crate::speculation::{PreIssueSkipReason, SpeculationController, SpeculativeResolution};
use crate::{
    discovery::ToolScope,
    limits::{ensure_json_depth, serialized_len},
    runner::{await_guarded, check_stopped},
    ApprovalHandler, ApprovalRecord, CancellationSafety, ExecutionLocation, HarnessError,
    IssueSafety, NetworkEgress, PolicyDecision, PolicyEngine, RunError, RunEvent, RunRequest,
    RunResult, RunStatus, SpeculationPolicy, Tool, ToolCall, ToolCallContext, ToolCaller,
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

    fn try_acquire(&self, key: &str) -> Result<Option<OwnedSemaphorePermit>, HarnessError> {
        let Some(mut keyed) = self.keyed.try_lock().ok() else {
            return Ok(None);
        };
        keyed.retain(|_, semaphore| semaphore.strong_count() > 0);
        let semaphore = if let Some(semaphore) = keyed.get(key).and_then(Weak::upgrade) {
            semaphore
        } else {
            let semaphore = Arc::new(Semaphore::new(1));
            keyed.insert(key.to_owned(), Arc::downgrade(&semaphore));
            semaphore
        };
        drop(keyed);
        match semaphore.try_acquire_owned() {
            Ok(permit) => Ok(Some(permit)),
            Err(tokio::sync::TryAcquireError::NoPermits) => Ok(None),
            Err(tokio::sync::TryAcquireError::Closed) => {
                Err(HarnessError::Tool("tool concurrency permit closed".into()))
            }
        }
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
    identical_calls: HashMap<InvocationKey, u32>,
    effects: HashMap<InvocationKey, EffectRecord>,
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
    signature: Option<InvocationKey>,
    effect_key: Option<InvocationKey>,
    caller: ToolCaller,
    authorized_signature: Option<InvocationKey>,
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
            // The public functional result is the canonical record of what the
            // broker actually evaluated. Value-bearing parser/VM artifacts are
            // never copied here, but valid tool arguments are preserved.
            result
                .tool_calls
                .push(self.call_for_transcript(request, &call, caller));
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
            (!defer_signature_checks).then(|| InvocationKey::new(&call.tool_id, &arguments));
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
            Some(InvocationKey::programmatic(
                &call.tool_id,
                &arguments,
                context.effect_key.as_deref().unwrap_or_default(),
            ))
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

        let signature = InvocationKey::new(&prepared.call.tool_id, &arguments);
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

        if prepared.authorized_signature.as_ref() != Some(&signature) {
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
        signature: &InvocationKey,
    ) -> bool {
        let count = state.identical_calls.entry(signature.clone()).or_default();
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
        signature: &InvocationKey,
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
        let current_signature = InvocationKey::new(&prepared.call.tool_id, &prepared.arguments);
        if !Arc::ptr_eq(&current, &prepared.tool)
            || !self.scope.contains(&prepared.call.tool_id)
            || self.scope.caller() != prepared.caller
            || !prepared.tool.definition().allows_caller(prepared.caller)
            || prepared.call.arguments_json != canonical_arguments
            || prepared.signature.as_ref() != Some(&current_signature)
            || prepared.authorized_signature.as_ref() != Some(&current_signature)
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
    struct CancellationOnDrop {
        cancellation: tokio_util::sync::CancellationToken,
        armed: bool,
    }

    impl Drop for CancellationOnDrop {
        fn drop(&mut self) {
            if self.armed {
                self.cancellation.cancel();
            }
        }
    }

    enum AwaitOutcome<T> {
        Completed(Result<T, HarnessError>),
        Cancelled,
        TimedOut,
    }

    tokio::pin!(future);
    // This guard is declared after the pinned tool future so task/future drop
    // signals cancellation before the tool future itself is destroyed.
    let mut cancellation_on_drop = CancellationOnDrop {
        cancellation: child_cancellation.clone(),
        armed: true,
    };
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
        AwaitOutcome::Completed(result) => {
            cancellation_on_drop.armed = false;
            result
        }
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct InvocationKey {
    pub(crate) tool_id: String,
    canonical_arguments: String,
    programmatic_occurrence: Option<String>,
}

/// Broker-owned result of one nonblocking speculative issue attempt.
pub(crate) enum SpeculativeAttempt {
    /// The candidate failed closed before crossing the tool issue boundary.
    NotIssued,
    /// The tool crossed the issue boundary and either produced a reusable result or failed.
    Issued(Result<Box<SpeculativeExecution>, HarnessError>),
}

/// Successful bounded speculative execution retained until exact commit or discard.
pub(crate) struct SpeculativeExecution {
    prepared: SpeculativePreparedCall,
    result: Arc<ToolResult>,
    lease: SpeculativeLease,
}

struct SpeculativeLease {
    controller: Arc<SpeculationController>,
    tool_id: String,
    cancellation: tokio_util::sync::CancellationToken,
    slot: Option<OwnedSemaphorePermit>,
    started_at: Instant,
    execution_finished_at: Option<Instant>,
    execution_duration_recorded: bool,
    drop_resolution: SpeculativeResolution,
    settled: bool,
}

impl SpeculativeLease {
    fn new(
        controller: &Arc<SpeculationController>,
        tool_id: &str,
        cancellation: tokio_util::sync::CancellationToken,
        slot: OwnedSemaphorePermit,
    ) -> Self {
        let started_at = Instant::now();
        controller.record_issue_started(tool_id, started_at);
        Self {
            controller: Arc::clone(controller),
            tool_id: tool_id.to_owned(),
            cancellation,
            slot: Some(slot),
            started_at,
            execution_finished_at: None,
            execution_duration_recorded: false,
            drop_resolution: SpeculativeResolution::Cancelled,
            settled: false,
        }
    }

    fn execution_finished(&mut self) {
        self.record_execution_duration();
        self.execution_finished_at = Some(Instant::now());
        self.drop_resolution = SpeculativeResolution::Discarded;
    }

    fn record_execution_duration(&mut self) {
        if !self.execution_duration_recorded {
            self.controller
                .record_execution_duration(&self.tool_id, elapsed_ms(self.started_at));
            self.execution_duration_recorded = true;
        }
    }

    fn publication_wait_ms(&self) -> Option<u64> {
        self.execution_finished_at.map(elapsed_ms)
    }

    fn settle(&mut self, resolution: SpeculativeResolution) {
        if !self.settled {
            self.cancellation.cancel();
            self.record_execution_duration();
            self.controller.record_resolution(
                &self.tool_id,
                resolution,
                self.publication_wait_ms(),
            );
            self.settled = true;
        }
    }

    fn try_commit(
        &mut self,
        candidate_deadline: Instant,
        run_deadline: Option<Instant>,
        cancellation: &tokio_util::sync::CancellationToken,
    ) -> bool {
        if self.settled
            || !self.controller.record_commit_if_active(
                &self.tool_id,
                candidate_deadline,
                run_deadline,
                cancellation,
                self.publication_wait_ms(),
            )
        {
            return false;
        }
        self.cancellation.cancel();
        self.slot.take();
        self.settled = true;
        true
    }
}

impl Drop for SpeculativeLease {
    fn drop(&mut self) {
        if !self.settled {
            self.cancellation.cancel();
            self.record_execution_duration();
            self.controller.record_resolution(
                &self.tool_id,
                self.drop_resolution,
                self.publication_wait_ms(),
            );
            self.settled = true;
        }
    }
}

impl SpeculativeExecution {
    pub(crate) fn settle(&mut self, resolution: SpeculativeResolution) {
        self.lease.settle(resolution);
    }

    fn try_commit(
        &mut self,
        candidate_deadline: Instant,
        run_deadline: Option<Instant>,
        cancellation: &tokio_util::sync::CancellationToken,
    ) -> bool {
        if !self
            .lease
            .try_commit(candidate_deadline, run_deadline, cancellation)
        {
            return false;
        }
        self.prepared._concurrency_permit.take();
        true
    }
}

struct PreIssueGuard {
    controller: Arc<SpeculationController>,
    tool_id: String,
    reason: PreIssueSkipReason,
    settled: bool,
}

impl PreIssueGuard {
    fn new(controller: &Arc<SpeculationController>, tool_id: &str) -> Self {
        Self {
            controller: Arc::clone(controller),
            tool_id: tool_id.to_owned(),
            reason: PreIssueSkipReason::Aborted,
            settled: false,
        }
    }

    fn skip(&mut self, reason: PreIssueSkipReason) {
        self.reason = reason;
    }

    fn issued(&mut self) {
        self.settled = true;
    }
}

impl Drop for PreIssueGuard {
    fn drop(&mut self) {
        if !self.settled {
            self.controller
                .record_pre_issue_skip(&self.tool_id, self.reason);
            self.settled = true;
        }
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(
        Instant::now()
            .saturating_duration_since(started)
            .as_millis(),
    )
    .unwrap_or(u64::MAX)
}

/// Broker outcome after revalidating a completed candidate at the normal
/// authoritative Direct boundary.
pub(crate) enum SpeculativeCommitOutcome {
    /// The cached result passed both policy boundaries and was recorded once.
    Committed(Arc<ToolResult>),
    /// Speculation was disabled or invalidated; execute this already-authorized
    /// Direct preparation without repeating policy or approval.
    ExecuteDirect(Box<PreparedCall>),
    /// Normal Direct preparation produced a canonical rejection or reuse.
    Resolved(Arc<ToolResult>),
    /// Normal Direct limits stopped the run.
    Stop,
    /// The candidate did not reach the ordinary Direct boundary and may use the
    /// normal sequential fallback path.
    NotCommitted,
    /// Ordinary Direct preparation failed after mutating normal accounting.
    DirectError(HarnessError),
}

struct SpeculativePreparedCall {
    call: ToolCall,
    arguments: Value,
    key: InvocationKey,
    tool: Arc<dyn Tool>,
    context: ToolCallContext,
    catalog_version: u64,
    model_call_number: u32,
    _concurrency_permit: Option<OwnedSemaphorePermit>,
}

impl InvocationKey {
    pub(crate) fn new(tool_id: &str, arguments: &Value) -> Self {
        Self {
            tool_id: tool_id.to_owned(),
            canonical_arguments: canonical_json(arguments),
            programmatic_occurrence: None,
        }
    }
}

fn canonical_json(value: &Value) -> String {
    fn normalized(value: &Value) -> Value {
        match value {
            Value::Array(values) => Value::Array(values.iter().map(normalized).collect()),
            Value::Object(values) => {
                let mut entries = values.iter().collect::<Vec<_>>();
                entries.sort_unstable_by_key(|(key, _)| *key);
                let mut object = serde_json::Map::new();
                for (key, value) in entries {
                    object.insert(key.clone(), normalized(value));
                }
                Value::Object(object)
            }
            value => value.clone(),
        }
    }

    serde_json::to_string(&normalized(value)).expect("serde_json::Value always serializes")
}

impl<'a> ToolBroker<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn speculate(
        &self,
        controller: &Arc<SpeculationController>,
        request: &RunRequest,
        mut call: ToolCall,
        model_call_number: u32,
        run_id: &str,
        trace_id: &str,
        deadline: Option<Instant>,
        slot: OwnedSemaphorePermit,
        candidate_cancellation: tokio_util::sync::CancellationToken,
    ) -> SpeculativeAttempt {
        let mut pre_issue = PreIssueGuard::new(controller, &call.tool_id);
        let prepared = match self
            .prepare_speculative(
                controller,
                request,
                &mut call,
                model_call_number,
                run_id,
                trace_id,
                deadline,
                &candidate_cancellation,
                &mut pre_issue,
            )
            .await
        {
            Ok(Some(prepared)) => prepared,
            Ok(None) => return SpeculativeAttempt::NotIssued,
            Err(error) => {
                pre_issue.skip(
                    if matches!(error, HarnessError::Cancelled | HarnessError::TimedOut(_)) {
                        PreIssueSkipReason::Aborted
                    } else {
                        PreIssueSkipReason::Failed
                    },
                );
                return SpeculativeAttempt::NotIssued;
            }
        };
        if check_stopped(
            &candidate_cancellation,
            deadline,
            "speculative execution deadline reached before tool invocation",
        )
        .is_err()
        {
            pre_issue.skip(PreIssueSkipReason::Aborted);
            return SpeculativeAttempt::NotIssued;
        }
        if !controller.is_active(&prepared.call.tool_id)
            || !self.speculative_prepared_is_live(request, &prepared)
        {
            pre_issue.skip(PreIssueSkipReason::Invalidated);
            return SpeculativeAttempt::NotIssued;
        }
        pre_issue.issued();
        SpeculativeAttempt::Issued(
            self.execute_speculative(
                prepared,
                controller,
                request,
                &candidate_cancellation,
                deadline,
                slot,
            )
            .await
            .map(Box::new),
        )
    }

    #[allow(clippy::too_many_arguments)]
    async fn prepare_speculative(
        &self,
        controller: &SpeculationController,
        request: &RunRequest,
        call: &mut ToolCall,
        model_call_number: u32,
        run_id: &str,
        trace_id: &str,
        deadline: Option<Instant>,
        candidate_cancellation: &tokio_util::sync::CancellationToken,
        pre_issue: &mut PreIssueGuard,
    ) -> Result<Option<SpeculativePreparedCall>, HarnessError> {
        controller.config().validate()?;
        if !controller.is_active(&call.tool_id)
            || call.arguments_json.len() as u64 > request.agent.limits.max_tool_arguments_bytes
        {
            pre_issue.skip(PreIssueSkipReason::Validation);
            return Ok(None);
        }
        let arguments: Value = match serde_json::from_str(&call.arguments_json) {
            Ok(arguments) => arguments,
            Err(_) => {
                pre_issue.skip(PreIssueSkipReason::Validation);
                return Ok(None);
            }
        };
        if ensure_json_depth(
            "speculative tool arguments",
            &arguments,
            request.agent.limits.max_json_depth,
        )
        .is_err()
            || self.tools.validate(&call.tool_id, &arguments).is_err()
        {
            pre_issue.skip(PreIssueSkipReason::Validation);
            return Ok(None);
        }
        call.arguments_json = serde_json::to_string(&arguments).map_err(|_| {
            HarnessError::InvalidArguments(
                "speculative tool arguments could not be canonicalized".into(),
            )
        })?;
        if call.arguments_json.len() as u64 > request.agent.limits.max_tool_arguments_bytes {
            pre_issue.skip(PreIssueSkipReason::Validation);
            return Ok(None);
        }
        let key = InvocationKey::new(&call.tool_id, &arguments);
        let Some((tool, catalog_version)) = self.tools.get_versioned(&call.tool_id) else {
            pre_issue.skip(PreIssueSkipReason::Invalidated);
            return Ok(None);
        };
        if !self.speculative_metadata_is_live(request, &call.tool_id, tool.as_ref()) {
            pre_issue.skip(PreIssueSkipReason::Invalidated);
            return Ok(None);
        }
        let mut context =
            ToolCallContext::new(run_id, trace_id, call.id.clone(), call.tool_id.clone());
        context.caller = Some(ToolCaller::Speculative);
        let mut direct_context = context.clone();
        direct_context.caller = Some(ToolCaller::Direct);
        let ordinary = match await_guarded(
            self.policy.decide_with_context(
                &direct_context,
                tool.definition(),
                &arguments,
                request,
            ),
            candidate_cancellation,
            deadline,
            "ordinary policy decision exceeded candidate deadline",
            None,
        )
        .await
        {
            Ok(decision) => decision,
            Err(error) => {
                pre_issue.skip(
                    if matches!(error, HarnessError::Cancelled | HarnessError::TimedOut(_)) {
                        PreIssueSkipReason::Aborted
                    } else {
                        PreIssueSkipReason::Failed
                    },
                );
                return Err(error);
            }
        };
        // Approval is never requested before hidden issue. A visible
        // authoritative Direct call handles RequireApproval later.
        if !matches!(ordinary, PolicyDecision::Allow { .. }) {
            pre_issue.skip(PreIssueSkipReason::Policy);
            return Ok(None);
        }
        let decision = match await_guarded(
            self.policy
                .decide_speculative(&context, tool.definition(), &arguments, request),
            candidate_cancellation,
            deadline,
            "speculative policy decision exceeded candidate deadline",
            None,
        )
        .await
        {
            Ok(decision) => decision,
            Err(error) => {
                pre_issue.skip(
                    if matches!(error, HarnessError::Cancelled | HarnessError::TimedOut(_)) {
                        PreIssueSkipReason::Aborted
                    } else {
                        PreIssueSkipReason::Failed
                    },
                );
                return Err(error);
            }
        };
        if !matches!(decision, PolicyDecision::Allow { .. }) || !controller.is_active(&call.tool_id)
        {
            pre_issue.skip(if matches!(decision, PolicyDecision::Allow { .. }) {
                PreIssueSkipReason::Invalidated
            } else {
                PreIssueSkipReason::Policy
            });
            return Ok(None);
        }
        let Some((current, current_version)) = self.tools.get_versioned(&call.tool_id) else {
            pre_issue.skip(PreIssueSkipReason::Invalidated);
            return Ok(None);
        };
        if current_version != catalog_version
            || !Arc::ptr_eq(&current, &tool)
            || !self.speculative_metadata_is_live(request, &call.tool_id, current.as_ref())
        {
            pre_issue.skip(PreIssueSkipReason::Invalidated);
            return Ok(None);
        }
        let concurrency_permit = if let Some(key) = &tool.definition().concurrency_key {
            let permit = match self.concurrency.try_acquire(key) {
                Ok(Some(permit)) => permit,
                Ok(None) => {
                    pre_issue.skip(PreIssueSkipReason::KeySaturated);
                    return Ok(None);
                }
                Err(error) => {
                    pre_issue.skip(PreIssueSkipReason::Failed);
                    return Err(error);
                }
            };
            Some(permit)
        } else {
            None
        };
        if !controller.is_active(&call.tool_id)
            || !self.speculative_metadata_is_live(request, &call.tool_id, tool.as_ref())
        {
            pre_issue.skip(PreIssueSkipReason::Invalidated);
            return Ok(None);
        }
        Ok(Some(SpeculativePreparedCall {
            call: call.clone(),
            arguments,
            key,
            tool,
            context,
            catalog_version,
            model_call_number,
            _concurrency_permit: concurrency_permit,
        }))
    }

    async fn execute_speculative(
        &self,
        prepared: SpeculativePreparedCall,
        controller: &Arc<SpeculationController>,
        request: &RunRequest,
        candidate_cancellation: &tokio_util::sync::CancellationToken,
        deadline: Option<Instant>,
        slot: OwnedSemaphorePermit,
    ) -> Result<SpeculativeExecution, HarnessError> {
        // The explicit tool policy attests that Direct and Speculative caller
        // contexts have caller-invariant successful-result semantics.
        let tool_cancellation = candidate_cancellation.child_token();
        let mut lease = SpeculativeLease::new(
            controller,
            &prepared.call.tool_id,
            tool_cancellation.clone(),
            slot,
        );
        let execution = prepared.tool.execute_with_context(
            &prepared.context,
            prepared.arguments.clone(),
            tool_cancellation.clone(),
        );
        let tool_result = match await_dispatched_tool(
            execution,
            candidate_cancellation,
            deadline,
            "speculative tool execution exceeded candidate deadline",
            &tool_cancellation,
        )
        .await
        {
            Ok(result) => {
                lease.execution_finished();
                result
            }
            Err(error) => {
                let resolution =
                    if matches!(error, HarnessError::Cancelled | HarnessError::TimedOut(_)) {
                        SpeculativeResolution::Cancelled
                    } else {
                        SpeculativeResolution::Discarded
                    };
                lease.settle(resolution);
                return Err(error);
            }
        };
        if !tool_result.ok {
            return Err(HarnessError::Tool(
                "speculative tool returned a failure result".into(),
            ));
        }
        if serialized_len(&tool_result)? > request.agent.limits.max_tool_result_bytes {
            return Err(HarnessError::ResourceLimit(
                "speculative tool result exceeded its byte limit".into(),
            ));
        }
        ensure_json_depth(
            "speculative tool result",
            &tool_result.output,
            request.agent.limits.max_json_depth,
        )?;
        self.tools
            .validate_output(&prepared.call.tool_id, &tool_result.output)?;
        Ok(SpeculativeExecution {
            prepared,
            result: Arc::new(tool_result),
            lease,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn commit_speculative(
        &self,
        controller: &SpeculationController,
        request: &RunRequest,
        result: &mut RunResult,
        events: &mut EventEmitter,
        state: &mut BrokerState,
        execution: &mut Option<Box<SpeculativeExecution>>,
        authoritative_call: &ToolCall,
        model_call_number: u32,
        candidate_deadline: Instant,
        run_deadline: Option<Instant>,
    ) -> Result<SpeculativeCommitOutcome, HarnessError> {
        controller.config().validate()?;
        let commit_deadline =
            Some(run_deadline.map_or(candidate_deadline, |run| run.min(candidate_deadline)));
        if let Err(error) = check_stopped(
            &request.cancellation,
            commit_deadline,
            "speculative candidate expired before commit",
        ) {
            if matches!(error, HarnessError::Cancelled)
                || run_deadline.is_some_and(|run| run <= candidate_deadline)
            {
                return Err(error);
            }
            execution.take();
            return Ok(SpeculativeCommitOutcome::NotCommitted);
        }
        let Some(candidate) = execution.as_deref() else {
            return Ok(SpeculativeCommitOutcome::NotCommitted);
        };
        if !controller.is_active(&candidate.prepared.call.tool_id)
            || candidate.prepared.model_call_number != model_call_number
            || candidate.prepared.call.id != authoritative_call.id
            || authoritative_call.arguments_json.len() as u64
                > request.agent.limits.max_tool_arguments_bytes
        {
            execution.take();
            return Ok(SpeculativeCommitOutcome::NotCommitted);
        }
        let authoritative_arguments: Value =
            match serde_json::from_str(&authoritative_call.arguments_json) {
                Ok(arguments) => arguments,
                Err(_) => {
                    execution.take();
                    return Ok(SpeculativeCommitOutcome::NotCommitted);
                }
            };
        if ensure_json_depth(
            "authoritative speculative tool arguments",
            &authoritative_arguments,
            request.agent.limits.max_json_depth,
        )
        .is_err()
        {
            execution.take();
            return Ok(SpeculativeCommitOutcome::NotCommitted);
        }
        let authoritative_key =
            InvocationKey::new(&authoritative_call.tool_id, &authoritative_arguments);
        if authoritative_key != candidate.prepared.key
            || !self.speculative_prepared_is_live(request, &candidate.prepared)
            || self
                .tools
                .validate(&authoritative_call.tool_id, &authoritative_arguments)
                .is_err()
            || serialized_len(candidate.result.as_ref())?
                > request.agent.limits.max_tool_result_bytes
            || ensure_json_depth(
                "committed speculative tool result",
                &candidate.result.output,
                request.agent.limits.max_json_depth,
            )
            .is_err()
            || self
                .tools
                .validate_output(&authoritative_call.tool_id, &candidate.result.output)
                .is_err()
        {
            execution.take();
            return Ok(SpeculativeCommitOutcome::NotCommitted);
        }
        // Cross the ordinary Direct authorization and approval boundary exactly
        // once. If cached reuse is later invalidated, the returned preparation is
        // executed normally without re-running policy or approval.
        let attempts_before = state.tool_calls;
        let classified_before = state.classified_tool_calls();
        let ordinary = {
            let ordinary_prepare = self.prepare(
                request,
                result,
                events,
                state,
                authoritative_call.clone(),
                ToolCaller::Direct,
                false,
                false,
                None,
                run_deadline,
            );
            tokio::pin!(ordinary_prepare);
            if Instant::now() >= candidate_deadline {
                execution.take();
                ordinary_prepare.await
            } else {
                tokio::select! {
                    biased;
                    outcome = &mut ordinary_prepare => outcome,
                    _ = tokio::time::sleep_until(candidate_deadline) => {
                        // Expiry owns and synchronously drops the completed cache,
                        // runner slot, keyed permit, and settlement lease without
                        // cancelling or restarting canonical Direct preparation.
                        execution.take();
                        ordinary_prepare.await
                    }
                }
            }
        };
        if Instant::now() >= candidate_deadline {
            execution.take();
        }
        let prepared = match ordinary {
            Ok(outcome) => outcome,
            Err(error) => {
                if state.tool_calls > attempts_before
                    && state.classified_tool_calls() == classified_before
                {
                    state.record_pre_dispatch_error(&error);
                }
                return Ok(SpeculativeCommitOutcome::DirectError(error));
            }
        };
        let prepared = match prepared {
            PrepareOutcome::Ready(prepared) => prepared,
            PrepareOutcome::Rejected(result) => {
                execution.take();
                return Ok(SpeculativeCommitOutcome::Resolved(Arc::new(result)));
            }
            PrepareOutcome::Reused(result) => {
                execution.take();
                return Ok(SpeculativeCommitOutcome::Resolved(result));
            }
            PrepareOutcome::Stop => {
                execution.take();
                return Ok(SpeculativeCommitOutcome::Stop);
            }
        };

        if execution.is_none() {
            return Ok(SpeculativeCommitOutcome::ExecuteDirect(prepared));
        }
        let Some(candidate) = execution.as_deref() else {
            return Ok(SpeculativeCommitOutcome::ExecuteDirect(prepared));
        };
        if !controller.is_active(&candidate.prepared.call.tool_id)
            || !self.speculative_prepared_is_live(request, &candidate.prepared)
        {
            execution.take();
            return Ok(SpeculativeCommitOutcome::ExecuteDirect(prepared));
        }
        let speculative_context = candidate.prepared.context.clone();
        let speculative_tool = Arc::clone(&candidate.prepared.tool);

        let decision = match await_guarded(
            self.policy.decide_speculative(
                &speculative_context,
                speculative_tool.definition(),
                &authoritative_arguments,
                request,
            ),
            &request.cancellation,
            commit_deadline,
            "speculative commit policy decision exceeded candidate deadline",
            None,
        )
        .await
        {
            Ok(decision) => decision,
            Err(error) => {
                if request.cancellation.is_cancelled()
                    || run_deadline.is_some_and(|deadline| Instant::now() >= deadline)
                {
                    return Err(error);
                }
                // A dedicated-policy failure cannot invalidate the already
                // completed ordinary Direct authorization or cause it to run
                // twice. Discard the cache and execute this same preparation.
                execution.take();
                return Ok(SpeculativeCommitOutcome::ExecuteDirect(prepared));
            }
        };
        if !matches!(decision, PolicyDecision::Allow { .. }) {
            execution.take();
            return Ok(SpeculativeCommitOutcome::ExecuteDirect(prepared));
        }
        let candidate_is_live = execution.as_deref().is_some_and(|candidate| {
            controller.is_active(&candidate.prepared.call.tool_id)
                && candidate.prepared.model_call_number == model_call_number
                && candidate.prepared.call.id == authoritative_call.id
                && candidate.prepared.key == authoritative_key
                && self.speculative_prepared_is_live(request, &candidate.prepared)
                && self
                    .tools
                    .validate_output(&authoritative_call.tool_id, &candidate.result.output)
                    .is_ok()
        });
        if !candidate_is_live {
            execution.take();
            return Ok(SpeculativeCommitOutcome::ExecuteDirect(prepared));
        }

        // This is the atomic cache-publication boundary. The controller lock
        // rechecks activation, cancellation, and both absolute deadlines while
        // converting the live lease to Committed. After it succeeds, later
        // synchronous event-sink latency cannot make the authorized result stale.
        let mut candidate = execution
            .take()
            .expect("candidate liveness was checked immediately before take");
        if !candidate.try_commit(candidate_deadline, run_deadline, &request.cancellation) {
            return Ok(SpeculativeCommitOutcome::ExecuteDirect(prepared));
        }
        let tool_result = Arc::clone(&candidate.result);
        // The result is now committed and independently retained. Release the
        // speculative runner/keyed permits before invoking synchronous sinks.
        drop(candidate);
        self.mark_dispatched(state, &prepared);
        let committed = BrokerExecution {
            result: Arc::clone(&tool_result),
            validation_error: None,
            duration_ms: 0,
        };
        self.record_execution(state, &prepared, &committed);
        events.emit(RunEvent::ToolCompleted {
            call_id: prepared.call.id.clone(),
            tool_id: prepared.call.tool_id.clone(),
            ok: true,
        });
        Ok(SpeculativeCommitOutcome::Committed(tool_result))
    }

    pub(crate) fn validate_shadow_candidate(
        &self,
        controller: &SpeculationController,
        request: &RunRequest,
        call: &ToolCall,
    ) -> Option<InvocationKey> {
        if call.arguments_json.len() as u64 > request.agent.limits.max_tool_arguments_bytes {
            return None;
        }
        let arguments: Value = serde_json::from_str(&call.arguments_json).ok()?;
        ensure_json_depth(
            "shadow tool arguments",
            &arguments,
            request.agent.limits.max_json_depth,
        )
        .ok()?;
        self.tools.validate(&call.tool_id, &arguments).ok()?;
        let (tool, _) = self.tools.get_versioned(&call.tool_id)?;
        self.speculative_metadata_is_live(request, &call.tool_id, tool.as_ref())
            .then(|| {
                controller.register_tool(&call.tool_id);
                InvocationKey::new(&call.tool_id, &arguments)
            })
    }

    pub(crate) fn validate_partial_probe(
        &self,
        request: &RunRequest,
        tool_id: &str,
        arguments_json: &str,
    ) -> bool {
        if arguments_json.len() as u64 > request.agent.limits.max_tool_arguments_bytes {
            return false;
        }
        let Ok(arguments) = serde_json::from_str(arguments_json) else {
            return false;
        };
        ensure_json_depth(
            "partial speculative tool arguments",
            &arguments,
            request.agent.limits.max_json_depth,
        )
        .is_ok()
            && self.tools.validate(tool_id, &arguments).is_ok()
    }

    fn speculative_prepared_is_live(
        &self,
        request: &RunRequest,
        prepared: &SpeculativePreparedCall,
    ) -> bool {
        let Some((current, version)) = self.tools.get_versioned(&prepared.call.tool_id) else {
            return false;
        };
        version == prepared.catalog_version
            && Arc::ptr_eq(&current, &prepared.tool)
            && InvocationKey::new(&prepared.call.tool_id, &prepared.arguments) == prepared.key
            && self.speculative_metadata_is_live(request, &prepared.call.tool_id, current.as_ref())
    }

    fn speculative_metadata_is_live(
        &self,
        request: &RunRequest,
        tool_id: &str,
        tool: &dyn Tool,
    ) -> bool {
        let definition = tool.definition();
        self.scope.caller() == ToolCaller::Direct
            && self.scope.contains(tool_id)
            && request.agent.tool_allowlist.iter().any(|id| id == tool_id)
            && definition.id == tool_id
            && definition.allows_caller(ToolCaller::Direct)
            && definition.allows_caller(ToolCaller::Speculative)
            && definition.speculation_policy == SpeculationPolicy::Enabled
            && definition.read_only
            && definition.idempotent
            && definition.parallel_safe
            && definition.cancellation_safety == CancellationSafety::Guaranteed
            && definition.issue_safety == IssueSafety::Guaranteed
            && definition.execution_location == ExecutionLocation::LocalPrivate
            && definition.network_egress == NetworkEgress::Prohibited
    }
}

impl InvocationKey {
    fn programmatic(tool_id: &str, arguments: &Value, occurrence: &str) -> Self {
        Self {
            tool_id: tool_id.to_owned(),
            canonical_arguments: canonical_json(arguments),
            programmatic_occurrence: Some(occurrence.to_owned()),
        }
    }
}

#[cfg(test)]
mod speculation_tests {
    use super::{InvocationKey, ToolConcurrencyLimiter};
    use serde_json::Value;
    use std::collections::{HashMap, HashSet};

    #[tokio::test]
    async fn speculative_keyed_admission_is_nonblocking_and_never_queues() {
        let limiter = ToolConcurrencyLimiter::default();
        let held = limiter.acquire("resource").await.unwrap();
        assert!(limiter.try_acquire("resource").unwrap().is_none());
        drop(held);
        assert!(limiter.try_acquire("resource").unwrap().is_some());
    }

    #[test]
    fn invocation_key_equality_and_hash_share_one_canonical_representation() {
        let negative_zero: Value = serde_json::from_str("-0.0").unwrap();
        let positive_zero: Value = serde_json::from_str("0.0").unwrap();
        let negative = InvocationKey::new("read", &negative_zero);
        let positive = InvocationKey::new("read", &positive_zero);

        assert_ne!(negative, positive);
        let keys = HashSet::from([negative.clone(), positive.clone()]);
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&negative));
        assert!(keys.contains(&positive));
        let mut repeated_calls = HashMap::new();
        *repeated_calls.entry(negative.clone()).or_insert(0_u32) += 1;
        *repeated_calls.entry(positive.clone()).or_insert(0_u32) += 1;
        *repeated_calls.entry(negative.clone()).or_insert(0_u32) += 1;
        assert_eq!(repeated_calls[&negative], 2);
        assert_eq!(repeated_calls[&positive], 1);

        let reordered =
            InvocationKey::new("read", &serde_json::from_str(r#"{"b":2,"a":1}"#).unwrap());
        let canonical = InvocationKey::new("read", &serde_json::json!({"a":1,"b":2}));
        assert_eq!(reordered, canonical);
    }
}
