use crate::{
    agent::{RunRequest, RunResult, RunStatus},
    event::{EventEmitter, EventSink, InMemoryEventSink, RunEvent},
    limits::{compile_trusted_schema, ensure_json_depth, serialized_len, AgentLimits},
    message::{Message, MessageRole},
    model::{ModelProvider, ModelRequest, ModelResponse},
    policy::{ApprovalHandler, DenyApproval, PolicyDecision, PolicyEngine, SafeDefaultPolicy},
    tool::{Tool, ToolCall, ToolRegistry, ToolResult},
    GenerationOptions, HarnessError, RunError,
};
use jsonschema::Validator;
use serde_json::Value;
use std::{
    collections::HashMap,
    future::Future,
    sync::Arc,
    time::{Duration, Instant as StdInstant},
};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub struct AgentRunner {
    provider: Arc<dyn ModelProvider>,
    tools: ToolRegistry,
    policy: Arc<dyn PolicyEngine>,
    approvals: Arc<dyn ApprovalHandler>,
    events: Arc<dyn EventSink>,
}

pub struct AgentRunnerBuilder {
    provider: Arc<dyn ModelProvider>,
    tools: ToolRegistry,
    policy: Arc<dyn PolicyEngine>,
    approvals: Arc<dyn ApprovalHandler>,
    events: Arc<dyn EventSink>,
}

impl AgentRunner {
    pub fn builder(provider: Arc<dyn ModelProvider>) -> AgentRunnerBuilder {
        AgentRunnerBuilder {
            provider,
            tools: ToolRegistry::default(),
            policy: Arc::new(SafeDefaultPolicy),
            approvals: Arc::new(DenyApproval),
            events: Arc::new(InMemoryEventSink::default()),
        }
    }

    /// Executes one run. Invalid requests return `Err`; failures after a run starts are captured
    /// in a terminal `RunResult` and always emit a matching terminal event.
    pub async fn run(&self, request: RunRequest) -> Result<RunResult, HarnessError> {
        let output_validator = validate_request(&request)?;
        let started = StdInstant::now();
        let deadline = absolute_deadline(request.agent.limits.max_run_duration_ms)?;
        let run_id = Uuid::new_v4().to_string();
        let trace_id = Uuid::new_v4().to_string();
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

        let mut messages = initial_messages(&request);
        let mut model_calls = 0;
        let mut tool_calls = 0;
        let mut output_repairs = 0;
        let mut identical_calls: HashMap<String, u32> = HashMap::new();

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
                let call_deadline =
                    provider_deadline(deadline, request.agent.limits.max_model_call_duration_ms);
                let completion = self.provider.complete(ModelRequest {
                    model: model.clone(),
                    messages: messages.clone(),
                    tools: self
                        .tools
                        .allowed_definitions(&request.agent.tool_allowlist),
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
                    output_validator.as_ref(),
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

            for call in response.tool_calls {
                if let Err(error) =
                    check_stopped(&request.cancellation, deadline, "run deadline reached")
                {
                    apply_terminal_error(&mut result, error);
                    break 'run;
                }
                if tool_calls >= request.agent.limits.max_tool_calls {
                    result.status = RunStatus::LimitReached;
                    result.tool_call_limit_reached = true;
                    result.errors.push(RunError {
                        code: "tool_call_limit".into(),
                        message: "tool call limit reached".into(),
                    });
                    break 'run;
                }
                tool_calls += 1;
                result.tool_calls.push(call.clone());

                if call.arguments_json.len() as u64 > request.agent.limits.max_tool_arguments_bytes
                {
                    self.reject(
                        &mut result,
                        &mut events,
                        &call,
                        format!(
                            "tool arguments exceed {} bytes",
                            request.agent.limits.max_tool_arguments_bytes
                        ),
                    );
                    if let Err(error) = push_tool_message(
                        &mut messages,
                        &call,
                        &ToolResult::failure("tool arguments exceed byte limit"),
                        &request.agent.limits,
                    ) {
                        apply_terminal_error(&mut result, error);
                        break 'run;
                    }
                    continue;
                }

                let arguments: Value = match serde_json::from_str(&call.arguments_json) {
                    Ok(value) => value,
                    Err(error) => {
                        self.reject(
                            &mut result,
                            &mut events,
                            &call,
                            format!("malformed JSON: {error}"),
                        );
                        if let Err(error) = push_tool_message(
                            &mut messages,
                            &call,
                            &ToolResult::failure("malformed tool arguments"),
                            &request.agent.limits,
                        ) {
                            apply_terminal_error(&mut result, error);
                            break 'run;
                        }
                        continue;
                    }
                };
                if let Err(error) = ensure_json_depth(
                    "tool arguments",
                    &arguments,
                    request.agent.limits.max_json_depth,
                ) {
                    self.reject(&mut result, &mut events, &call, error.to_string());
                    if let Err(error) = push_tool_message(
                        &mut messages,
                        &call,
                        &ToolResult::failure("tool arguments exceed JSON depth limit"),
                        &request.agent.limits,
                    ) {
                        apply_terminal_error(&mut result, error);
                        break 'run;
                    }
                    continue;
                }

                let signature = format!("{}:{}", call.tool_id, canonical_json(&arguments));
                let count = identical_calls.entry(signature).or_default();
                *count += 1;
                if *count > request.agent.limits.max_identical_tool_calls {
                    result.status = RunStatus::LimitReached;
                    result.repeated_tool_call_limit_reached = true;
                    result.errors.push(RunError {
                        code: "repeated_tool_call_limit".into(),
                        message: "repeated identical tool call limit reached".into(),
                    });
                    break 'run;
                }

                let Some(tool) = self.tools.get(&call.tool_id) else {
                    self.reject(&mut result, &mut events, &call, "unknown tool".into());
                    if let Err(error) = push_tool_message(
                        &mut messages,
                        &call,
                        &ToolResult::failure("unknown tool"),
                        &request.agent.limits,
                    ) {
                        apply_terminal_error(&mut result, error);
                        break 'run;
                    }
                    continue;
                };
                if !request
                    .agent
                    .tool_allowlist
                    .iter()
                    .any(|id| id == &call.tool_id)
                {
                    self.reject(
                        &mut result,
                        &mut events,
                        &call,
                        "tool is not allowed for agent".into(),
                    );
                    if let Err(error) = push_tool_message(
                        &mut messages,
                        &call,
                        &ToolResult::failure("tool is not allowed"),
                        &request.agent.limits,
                    ) {
                        apply_terminal_error(&mut result, error);
                        break 'run;
                    }
                    continue;
                }
                if let Err(error) = self.tools.validate(&call.tool_id, &arguments) {
                    self.reject(&mut result, &mut events, &call, error.to_string());
                    if let Err(error) = push_tool_message(
                        &mut messages,
                        &call,
                        &ToolResult::failure("tool arguments failed validation"),
                        &request.agent.limits,
                    ) {
                        apply_terminal_error(&mut result, error);
                        break 'run;
                    }
                    continue;
                }

                let decision = match await_guarded(
                    self.policy.decide(tool.definition(), &arguments, &request),
                    &request.cancellation,
                    deadline,
                    "policy decision exceeded run deadline",
                    None,
                )
                .await
                {
                    Ok(decision) => decision,
                    Err(error) => {
                        apply_terminal_error(&mut result, error);
                        break 'run;
                    }
                };
                events.emit(RunEvent::PolicyDecided {
                    call_id: call.id.clone(),
                    decision: decision.clone(),
                });
                result.policy_decisions.push(decision.clone());

                match decision {
                    PolicyDecision::Deny { reason } => {
                        self.reject(
                            &mut result,
                            &mut events,
                            &call,
                            format!("policy denied: {reason}"),
                        );
                        if let Err(error) = push_tool_message(
                            &mut messages,
                            &call,
                            &ToolResult::failure("policy denied"),
                            &request.agent.limits,
                        ) {
                            apply_terminal_error(&mut result, error);
                            break 'run;
                        }
                    }
                    PolicyDecision::RequireApproval { .. } => {
                        events.emit(RunEvent::ApprovalRequested {
                            call_id: call.id.clone(),
                            tool_id: call.tool_id.clone(),
                        });
                        let mut approval = match await_guarded(
                            self.approvals
                                .approve(tool.definition(), &arguments, &request),
                            &request.cancellation,
                            deadline,
                            "approval exceeded run deadline",
                            None,
                        )
                        .await
                        {
                            Ok(approval) => approval,
                            Err(error) => {
                                apply_terminal_error(&mut result, error);
                                break 'run;
                            }
                        };
                        approval.call_id = call.id.clone();
                        approval.tool_id = call.tool_id.clone();
                        let granted = approval.granted;
                        result.approvals.push(approval.clone());
                        if granted {
                            if let Err(error) = self
                                .run_tool(
                                    &mut result,
                                    &mut events,
                                    &mut messages,
                                    &call,
                                    tool,
                                    arguments,
                                    &request.cancellation,
                                    deadline,
                                    &request.agent.limits,
                                )
                                .await
                            {
                                apply_terminal_error(&mut result, error);
                                break 'run;
                            }
                        } else {
                            self.reject(
                                &mut result,
                                &mut events,
                                &call,
                                format!("approval denied: {}", approval.reason),
                            );
                            if let Err(error) = push_tool_message(
                                &mut messages,
                                &call,
                                &ToolResult::failure("approval denied"),
                                &request.agent.limits,
                            ) {
                                apply_terminal_error(&mut result, error);
                                break 'run;
                            }
                        }
                    }
                    PolicyDecision::Allow { .. } => {
                        if let Err(error) = self
                            .run_tool(
                                &mut result,
                                &mut events,
                                &mut messages,
                                &call,
                                tool,
                                arguments,
                                &request.cancellation,
                                deadline,
                                &request.agent.limits,
                            )
                            .await
                        {
                            apply_terminal_error(&mut result, error);
                            break 'run;
                        }
                    }
                }
            }
        }

        result.duration_ms = started.elapsed().as_millis() as u64;
        events.emit(RunEvent::Completed {
            status: result.status.clone(),
        });
        Ok(result)
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
        result.errors.push(RunError {
            code: "tool_rejected".into(),
            message: reason,
        });
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_tool(
        &self,
        run: &mut RunResult,
        events: &mut EventEmitter,
        messages: &mut Vec<Message>,
        call: &ToolCall,
        tool: Arc<dyn Tool>,
        arguments: Value,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
        limits: &AgentLimits,
    ) -> Result<(), HarnessError> {
        // This check closes the cancellation window between policy/approval and invocation.
        check_stopped(
            cancellation,
            deadline,
            "run deadline reached before tool invocation",
        )?;
        let tool_cancellation = cancellation.child_token();
        let execution = tool.execute(arguments, tool_cancellation.clone());
        let tool_result = await_guarded(
            execution,
            cancellation,
            deadline,
            "tool execution exceeded run deadline",
            Some(&tool_cancellation),
        )
        .await;

        let tool_result = match tool_result {
            Ok(tool_result) => tool_result,
            Err(error) => {
                events.emit(RunEvent::ToolCompleted {
                    call_id: call.id.clone(),
                    tool_id: call.tool_id.clone(),
                    ok: false,
                });
                return Err(error);
            }
        };
        events.emit(RunEvent::ToolCompleted {
            call_id: call.id.clone(),
            tool_id: call.tool_id.clone(),
            ok: tool_result.ok,
        });
        if serialized_len(&tool_result)? > limits.max_tool_result_bytes {
            return Err(HarnessError::ResourceLimit(format!(
                "tool result exceeds {} bytes",
                limits.max_tool_result_bytes
            )));
        }
        ensure_json_depth("tool result", &tool_result.output, limits.max_json_depth)?;
        push_tool_message(messages, call, &tool_result, limits)?;
        if !tool_result.ok {
            run.errors.push(RunError {
                code: "tool_error".into(),
                message: tool_result
                    .error
                    .clone()
                    .unwrap_or_else(|| "tool returned a failure result".into()),
            });
        }
        Ok(())
    }
}

impl AgentRunnerBuilder {
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    pub fn policy(mut self, policy: Arc<dyn PolicyEngine>) -> Self {
        self.policy = policy;
        self
    }

    pub fn approvals(mut self, approvals: Arc<dyn ApprovalHandler>) -> Self {
        self.approvals = approvals;
        self
    }

    pub fn event_sink(mut self, events: Arc<dyn EventSink>) -> Self {
        self.events = events;
        self
    }

    pub fn build(self) -> AgentRunner {
        AgentRunner {
            provider: self.provider,
            tools: self.tools,
            policy: self.policy,
            approvals: self.approvals,
            events: self.events,
        }
    }
}

fn validate_request(request: &RunRequest) -> Result<Option<Validator>, HarnessError> {
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

fn validate_model_response(
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

fn validate_output(
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

fn initial_messages(request: &RunRequest) -> Vec<Message> {
    let mut messages = vec![];
    if !request.agent.system_instructions.trim().is_empty() {
        messages.push(Message::system(request.agent.system_instructions.clone()));
    }
    messages.extend(request.history.clone());
    messages.push(Message::user(request.input.clone()));
    messages
}

fn ensure_transcript(messages: &[Message], limits: &AgentLimits) -> Result<(), HarnessError> {
    let bytes = messages.iter().map(Message::transcript_bytes).sum::<u64>();
    if bytes > limits.max_transcript_bytes {
        return Err(HarnessError::ResourceLimit(format!(
            "transcript exceeds {} bytes",
            limits.max_transcript_bytes
        )));
    }
    Ok(())
}

fn push_tool_message(
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

fn apply_terminal_error(result: &mut RunResult, error: HarnessError) {
    result.status = match error {
        HarnessError::Cancelled => RunStatus::Cancelled,
        HarnessError::ResourceLimit(_) => RunStatus::LimitReached,
        _ => RunStatus::Failed,
    };
    result.cancelled = matches!(error, HarnessError::Cancelled);
    result.errors.push(error.run_error());
}

fn absolute_deadline(duration_ms: Option<u64>) -> Result<Option<Instant>, HarnessError> {
    duration_ms
        .map(|duration_ms| {
            Instant::now()
                .checked_add(Duration::from_millis(duration_ms))
                .ok_or_else(|| HarnessError::InvalidRequest("run duration is too large".into()))
        })
        .transpose()
}

fn provider_deadline(
    run_deadline: Option<Instant>,
    provider_duration_ms: Option<u64>,
) -> Option<Instant> {
    let provider_deadline = provider_duration_ms
        .and_then(|duration_ms| Instant::now().checked_add(Duration::from_millis(duration_ms)));
    match (run_deadline, provider_deadline) {
        (Some(run), Some(provider)) => Some(run.min(provider)),
        (Some(run), None) => Some(run),
        (None, provider) => provider,
    }
}

fn check_stopped(
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

async fn await_guarded<T, F>(
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

fn merge_generation(
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

fn canonical_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

#[allow(dead_code)]
fn _message_role_is_part_of_public_contract(role: MessageRole) -> MessageRole {
    role
}
