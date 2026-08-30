use crate::{
    limits::{ensure_json_depth, serialized_len},
    runner::{await_guarded, check_stopped},
    ApprovalHandler, HarnessError, PolicyDecision, PolicyEngine, RunError, RunEvent, RunRequest,
    RunResult, RunStatus, Tool, ToolCall, ToolCallContext, ToolCaller, ToolRegistry, ToolResult,
};
use serde_json::Value;
use std::{collections::HashMap, sync::Arc, time::Instant as StdInstant};
use tokio::time::Instant;

use crate::event::EventEmitter;

/// Per-run counters and effect records shared by every tool-calling strategy.
#[derive(Default)]
pub(crate) struct BrokerState {
    pub(crate) tool_calls: u32,
    identical_calls: HashMap<String, u32>,
    committed_effects: HashMap<String, ToolResult>,
    reuse_committed_effects: bool,
}

impl BrokerState {
    pub(crate) fn enable_effect_reuse(&mut self) {
        self.reuse_committed_effects = true;
    }
}

/// A completely validated and authorized invocation ready to execute.
pub(crate) struct PreparedCall {
    pub(crate) call: ToolCall,
    pub(crate) arguments: Value,
    pub(crate) tool: Arc<dyn Tool>,
    pub(crate) context: ToolCallContext,
    pub(crate) signature: String,
    caller: ToolCaller,
    approved: bool,
}

/// Result of preparing one invocation at the shared broker boundary.
pub(crate) enum PrepareOutcome {
    Ready(PreparedCall),
    Rejected(ToolResult),
    Reused(ToolResult),
    Stop,
}

/// Result of executing one prepared invocation.
pub(crate) struct BrokerExecution {
    pub(crate) result: ToolResult,
    pub(crate) validation_error: Option<RunError>,
    pub(crate) duration_ms: u64,
}

/// Provider-neutral safety boundary used by direct and orchestrated calls.
pub(crate) struct ToolBroker<'a> {
    tools: &'a ToolRegistry,
    policy: &'a Arc<dyn PolicyEngine>,
    approvals: &'a Arc<dyn ApprovalHandler>,
}

impl<'a> ToolBroker<'a> {
    pub(crate) fn new(
        tools: &'a ToolRegistry,
        policy: &'a Arc<dyn PolicyEngine>,
        approvals: &'a Arc<dyn ApprovalHandler>,
    ) -> Self {
        Self {
            tools,
            policy,
            approvals,
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
        deadline: Option<Instant>,
    ) -> Result<PrepareOutcome, HarnessError> {
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

        let recorded_call = self.call_for_transcript(request, &call, caller);
        result.tool_calls.push(recorded_call);

        if call.arguments_json.len() as u64 > request.agent.limits.max_tool_arguments_bytes {
            self.reject(
                result,
                events,
                &call,
                format!(
                    "tool arguments exceed {} bytes",
                    request.agent.limits.max_tool_arguments_bytes
                ),
            );
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "tool arguments exceed byte limit",
            )));
        }

        let arguments: Value = match serde_json::from_str(&call.arguments_json) {
            Ok(value) => value,
            Err(error) => {
                self.reject(result, events, &call, format!("malformed JSON: {error}"));
                return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                    "malformed tool arguments",
                )));
            }
        };
        if let Err(error) = ensure_json_depth(
            "tool arguments",
            &arguments,
            request.agent.limits.max_json_depth,
        ) {
            self.reject(result, events, &call, error.to_string());
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "tool arguments exceed JSON depth limit",
            )));
        }

        let signature = canonical_signature(&call.tool_id, &arguments);
        let count = state.identical_calls.entry(signature.clone()).or_default();
        *count += 1;
        if *count > request.agent.limits.max_identical_tool_calls {
            result.status = RunStatus::LimitReached;
            result.repeated_tool_call_limit_reached = true;
            result.errors.push(RunError::new(
                "repeated_tool_call_limit",
                "repeated identical tool call limit reached",
            ));
            return Ok(PrepareOutcome::Stop);
        }

        let Some(tool) = self.tools.get(&call.tool_id) else {
            self.reject(result, events, &call, "unknown tool".into());
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "unknown tool",
            )));
        };
        if !request
            .agent
            .tool_allowlist
            .iter()
            .any(|id| id == &call.tool_id)
        {
            self.reject(
                result,
                events,
                &call,
                "tool is not allowed for agent".into(),
            );
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "tool is not allowed",
            )));
        }
        if !tool.definition().allows_caller(caller) {
            self.reject(
                result,
                events,
                &call,
                format!("tool does not allow {} calls", caller_name(caller)),
            );
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "tool caller is not allowed",
            )));
        }
        if let Err(error) = self.tools.validate(&call.tool_id, &arguments) {
            self.reject(result, events, &call, error.to_string());
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "tool arguments failed validation",
            )));
        }

        if state.reuse_committed_effects && !tool.definition().read_only {
            if let Some(recorded) = state.committed_effects.get(&signature) {
                events.emit(RunEvent::ToolEffectReused {
                    call_id: call.id.clone(),
                    tool_id: call.tool_id.clone(),
                });
                return Ok(PrepareOutcome::Reused(recorded.clone()));
            }
        }

        let context = ToolCallContext::new(
            result.id.clone(),
            result.trace_id.clone(),
            call.id.clone(),
            call.tool_id.clone(),
        );
        let decision = await_guarded(
            self.policy
                .decide_with_context(&context, tool.definition(), &arguments, request),
            &request.cancellation,
            deadline,
            "policy decision exceeded run deadline",
            None,
        )
        .await?;
        events.emit(RunEvent::PolicyDecided {
            call_id: call.id.clone(),
            decision: decision.clone(),
        });
        result.policy_decisions.push(decision.clone());

        if let PolicyDecision::Deny { reason } = decision {
            self.reject(result, events, &call, format!("policy denied: {reason}"));
            return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                "policy denied",
            )));
        }

        let mut approved = false;
        if approval_barrier || matches!(decision, PolicyDecision::RequireApproval { .. }) {
            events.emit(RunEvent::ApprovalRequested {
                call_id: call.id.clone(),
                tool_id: call.tool_id.clone(),
            });
            let mut approval = await_guarded(
                self.approvals.approve_with_context(
                    &context,
                    tool.definition(),
                    &arguments,
                    request,
                ),
                &request.cancellation,
                deadline,
                "approval exceeded run deadline",
                None,
            )
            .await?;
            approval.call_id = call.id.clone();
            approval.tool_id = call.tool_id.clone();
            let granted = approval.granted;
            let reason = approval.reason.clone();
            result.approvals.push(approval);
            if !granted {
                self.reject(result, events, &call, format!("approval denied: {reason}"));
                return Ok(PrepareOutcome::Rejected(ToolResult::failure(
                    "approval denied",
                )));
            }
            approved = true;
        }

        Ok(PrepareOutcome::Ready(PreparedCall {
            call,
            arguments,
            tool,
            context,
            signature,
            caller,
            approved,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn revalidate_bound_arguments(
        &self,
        prepared: &mut PreparedCall,
        arguments: Value,
        request: &RunRequest,
        result: &mut RunResult,
        events: &mut EventEmitter,
        state: &BrokerState,
        deadline: Option<Instant>,
    ) -> Result<Option<ToolResult>, HarnessError> {
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
        if state.reuse_committed_effects && !prepared.tool.definition().read_only {
            if let Some(recorded) = state.committed_effects.get(&signature) {
                events.emit(RunEvent::ToolEffectReused {
                    call_id: prepared.call.id.clone(),
                    tool_id: prepared.call.tool_id.clone(),
                });
                return Ok(Some(recorded.clone()));
            }
        }

        let decision = await_guarded(
            self.policy.decide_with_context(
                &prepared.context,
                prepared.tool.definition(),
                &arguments,
                request,
            ),
            &request.cancellation,
            deadline,
            "bound argument policy decision exceeded run deadline",
            None,
        )
        .await?;
        events.emit(RunEvent::PolicyDecided {
            call_id: prepared.call.id.clone(),
            decision: decision.clone(),
        });
        result.policy_decisions.push(decision.clone());
        match decision {
            PolicyDecision::Deny { reason } => {
                return Err(HarnessError::Policy(format!(
                    "bound plan call denied: {reason}"
                )));
            }
            PolicyDecision::RequireApproval { .. } if !prepared.approved => {
                events.emit(RunEvent::ApprovalRequested {
                    call_id: prepared.call.id.clone(),
                    tool_id: prepared.call.tool_id.clone(),
                });
                let mut approval = await_guarded(
                    self.approvals.approve_with_context(
                        &prepared.context,
                        prepared.tool.definition(),
                        &arguments,
                        request,
                    ),
                    &request.cancellation,
                    deadline,
                    "bound argument approval exceeded run deadline",
                    None,
                )
                .await?;
                approval.call_id = prepared.call.id.clone();
                approval.tool_id = prepared.call.tool_id.clone();
                let granted = approval.granted;
                result.approvals.push(approval);
                if !granted {
                    return Err(HarnessError::Approval(
                        "bound plan call approval denied".into(),
                    ));
                }
                prepared.approved = true;
            }
            PolicyDecision::RequireApproval { .. } | PolicyDecision::Allow { .. } => {}
        }

        prepared.arguments = arguments;
        prepared.call.arguments_json = arguments_json;
        prepared.signature = signature;
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
        let tool_cancellation = request.cancellation.child_token();
        let execution = prepared.tool.execute_with_context(
            &prepared.context,
            prepared.arguments.clone(),
            tool_cancellation.clone(),
        );
        let mut tool_result = await_guarded(
            execution,
            &request.cancellation,
            deadline,
            "tool execution exceeded run deadline",
            Some(&tool_cancellation),
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
                .map(|error| {
                    tool_result = ToolResult::failure("tool output failed validation");
                    RunError::new("tool_error", error.to_string())
                })
        } else {
            None
        };

        Ok(BrokerExecution {
            result: tool_result,
            validation_error,
            duration_ms: started.elapsed().as_millis() as u64,
        })
    }

    pub(crate) fn record_execution(
        &self,
        state: &mut BrokerState,
        prepared: &PreparedCall,
        execution: &BrokerExecution,
    ) {
        if execution.result.ok && !prepared.tool.definition().read_only {
            state
                .committed_effects
                .insert(prepared.signature.clone(), execution.result.clone());
        }
    }

    fn reject(
        &self,
        result: &mut RunResult,
        events: &mut EventEmitter,
        call: &ToolCall,
        reason: String,
    ) {
        events.emit(RunEvent::ToolRejected {
            call_id: call.id.clone(),
            tool_id: call.tool_id.clone(),
            reason: reason.clone(),
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

pub(crate) fn canonical_signature(tool_id: &str, arguments: &Value) -> String {
    format!(
        "{}:{}",
        tool_id,
        serde_json::to_string(arguments).unwrap_or_default()
    )
}

fn caller_name(caller: ToolCaller) -> &'static str {
    match caller {
        ToolCaller::Direct => "direct",
        ToolCaller::DeclarativePlan => "declarative plan",
        ToolCaller::Programmatic => "programmatic",
        ToolCaller::Speculative => "speculative",
    }
}
