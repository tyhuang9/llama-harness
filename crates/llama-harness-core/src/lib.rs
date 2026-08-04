//! A small, embedded agent runtime. Applications own transport, persistence, and tools.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub type JsonMap = serde_json::Map<String, Value>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentDefinition {
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub system_instructions: String,
    pub default_model: String,
    #[serde(default)]
    pub tool_allowlist: Vec<String>,
    #[serde(default)]
    pub limits: AgentLimits,
    #[serde(default)]
    pub generation: GenerationOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    #[serde(default)]
    pub metadata: JsonMap,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentLimits {
    pub max_model_calls: u32,
    pub max_tool_calls: u32,
    pub max_identical_tool_calls: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_run_duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_call_duration_ms: Option<u64>,
    pub max_output_repairs: u32,
}

impl Default for AgentLimits {
    fn default() -> Self {
        Self {
            max_model_calls: 8,
            max_tool_calls: 16,
            max_identical_tool_calls: 2,
            max_run_duration_ms: None,
            max_model_call_duration_ms: None,
            max_output_repairs: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct GenerationOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            tool_call_id: None,
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            tool_call_id: None,
        }
    }
    fn tool(call_id: String, result: &ToolResult) -> Self {
        Self {
            role: MessageRole::Tool,
            content: serde_json::to_string(result).unwrap_or_else(|_| "{\"ok\":false}".into()),
            tool_call_id: Some(call_id),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct RunOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub generation: GenerationOptions,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunRequest {
    pub agent: AgentDefinition,
    pub input: String,
    #[serde(default)]
    pub application_context: JsonMap,
    #[serde(default)]
    pub history: Vec<Message>,
    #[serde(default)]
    pub metadata: JsonMap,
    #[serde(default)]
    pub overrides: RunOverrides,
    #[serde(default)]
    pub evaluation: JsonMap,
    #[serde(skip, default = "CancellationToken::new")]
    pub cancellation: CancellationToken,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Completed,
    Failed,
    Cancelled,
    LimitReached,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunError {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunResult {
    pub id: String,
    pub status: RunStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_output: Option<String>,
    pub model: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub policy_decisions: Vec<PolicyDecision>,
    #[serde(default)]
    pub approvals: Vec<ApprovalRecord>,
    #[serde(default)]
    pub errors: Vec<RunError>,
    pub duration_ms: u64,
    pub trace_id: String,
    pub model_call_limit_reached: bool,
    pub tool_call_limit_reached: bool,
    pub repeated_tool_call_limit_reached: bool,
    pub cancelled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunEvent {
    Started {
        run_id: String,
        trace_id: String,
    },
    ModelRequested {
        call_number: u32,
        model: String,
    },
    ModelResponded {
        call_number: u32,
    },
    ToolRejected {
        call_id: String,
        tool_id: String,
        reason: String,
    },
    PolicyDecided {
        call_id: String,
        decision: PolicyDecision,
    },
    ApprovalRequested {
        call_id: String,
        tool_id: String,
    },
    ToolCompleted {
        call_id: String,
        tool_id: String,
        ok: bool,
    },
    Completed {
        status: RunStatus,
    },
}

pub trait EventSink: Send + Sync {
    fn emit(&self, event: RunEvent);
}

#[derive(Default)]
pub struct InMemoryEventSink {
    events: Mutex<Vec<RunEvent>>,
}
impl InMemoryEventSink {
    pub fn events(&self) -> Vec<RunEvent> {
        self.events
            .lock()
            .expect("event sink lock poisoned")
            .clone()
    }
}
impl EventSink for InMemoryEventSink {
    fn emit(&self, event: RunEvent) {
        self.events
            .lock()
            .expect("event sink lock poisoned")
            .push(event);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub generation: GenerationOptions,
    #[serde(default)]
    pub metadata: JsonMap,
    #[serde(skip)]
    pub cancellation: CancellationToken,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelResponse {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_output: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub supports_tools: bool,
    pub supports_streaming: bool,
    pub supports_structured_output: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub capabilities: ModelCapabilities,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderHealth {
    pub healthy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn id(&self) -> &str;
    fn capabilities(&self) -> ModelCapabilities;
    async fn health(&self) -> Result<ProviderHealth, HarnessError>;
    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError>;
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError>;
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub tool_id: String,
    pub arguments_json: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolRisk {
    Low,
    Medium,
    High,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolDefinition {
    pub id: String,
    pub name: String,
    pub description: String,
    pub arguments_schema: Value,
    pub risk: ToolRisk,
    pub idempotent: bool,
    pub read_only: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    pub ok: bool,
    pub output: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
impl ToolResult {
    pub fn success(output: Value) -> Self {
        Self {
            ok: true,
            output,
            error: None,
        }
    }
    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            output: Value::Null,
            error: Some(message.into()),
        }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> &ToolDefinition;
    async fn execute(
        &self,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError>;
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}
impl ToolRegistry {
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), HarnessError> {
        let id = tool.definition().id.trim().to_owned();
        if id.is_empty() {
            return Err(HarnessError::InvalidTool("tool id is required".into()));
        }
        jsonschema::validator_for(&tool.definition().arguments_schema).map_err(|error| {
            HarnessError::InvalidTool(format!("invalid schema for {id}: {error}"))
        })?;
        if self.tools.insert(id.clone(), tool).is_some() {
            return Err(HarnessError::InvalidTool(format!("duplicate tool: {id}")));
        }
        Ok(())
    }
    pub fn get(&self, id: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(id).cloned()
    }
    fn allowed_definitions(&self, allowlist: &[String]) -> Vec<ToolDefinition> {
        allowlist
            .iter()
            .filter_map(|id| self.tools.get(id).map(|tool| tool.definition().clone()))
            .collect()
    }
    fn validate(&self, tool: &dyn Tool, arguments: &Value) -> Result<(), HarnessError> {
        let validator = jsonschema::validator_for(&tool.definition().arguments_schema)
            .map_err(|error| HarnessError::InvalidTool(format!("invalid tool schema: {error}")))?;
        validator
            .validate(arguments)
            .map_err(|error| HarnessError::InvalidArguments(error.to_string()))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PolicyDecision {
    Allow { reason: String },
    Deny { reason: String },
    RequireApproval { reason: String },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApprovalRecord {
    pub call_id: String,
    pub tool_id: String,
    pub granted: bool,
    pub reason: String,
}
#[async_trait]
pub trait PolicyEngine: Send + Sync {
    async fn decide(
        &self,
        tool: &ToolDefinition,
        arguments: &Value,
        request: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError>;
}
#[async_trait]
pub trait ApprovalHandler: Send + Sync {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        arguments: &Value,
        request: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError>;
}
pub struct AllowAllPolicy;
#[async_trait]
impl PolicyEngine for AllowAllPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(PolicyDecision::Allow {
            reason: "default policy".into(),
        })
    }
}
pub struct DenyApproval;
#[async_trait]
impl ApprovalHandler for DenyApproval {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        Ok(ApprovalRecord {
            call_id: String::new(),
            tool_id: tool.id.clone(),
            granted: false,
            reason: "no approval handler configured".into(),
        })
    }
}

#[derive(Clone, Debug, Error)]
pub enum HarnessError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("invalid tool: {0}")]
    InvalidTool(String),
    #[error("invalid tool arguments: {0}")]
    InvalidArguments(String),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("cancelled")]
    Cancelled,
    #[error("timed out: {0}")]
    TimedOut(String),
    #[error("invalid structured output: {0}")]
    InvalidOutput(String),
}
impl HarnessError {
    fn run_error(&self) -> RunError {
        let code = match self {
            Self::InvalidRequest(_) => "invalid_request",
            Self::InvalidTool(_) => "invalid_tool",
            Self::InvalidArguments(_) => "invalid_arguments",
            Self::Provider(_) => "provider_error",
            Self::Tool(_) => "tool_error",
            Self::Cancelled => "cancelled",
            Self::TimedOut(_) => "timed_out",
            Self::InvalidOutput(_) => "invalid_output",
        };
        RunError {
            code: code.into(),
            message: self.to_string(),
        }
    }
}

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
            policy: Arc::new(AllowAllPolicy),
            approvals: Arc::new(DenyApproval),
            events: Arc::new(InMemoryEventSink::default()),
        }
    }
    pub async fn run(&self, request: RunRequest) -> Result<RunResult, HarnessError> {
        validate_request(&request)?;
        let started = Instant::now();
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
        self.events.emit(RunEvent::Started { run_id, trace_id });
        let mut messages = vec![];
        if !request.agent.system_instructions.trim().is_empty() {
            messages.push(Message::system(request.agent.system_instructions.clone()));
        }
        messages.extend(request.history.clone());
        messages.push(Message::user(request.input.clone()));
        let mut model_calls = 0;
        let mut tool_calls = 0;
        let mut output_repairs = 0;
        let mut identical_calls: HashMap<String, u32> = HashMap::new();
        loop {
            if request
                .agent
                .limits
                .max_run_duration_ms
                .is_some_and(|limit| started.elapsed() >= Duration::from_millis(limit))
            {
                result.status = RunStatus::Failed;
                result
                    .errors
                    .push(HarnessError::TimedOut("run duration limit reached".into()).run_error());
                break;
            }
            if request.cancellation.is_cancelled() {
                result.status = RunStatus::Cancelled;
                result.cancelled = true;
                result.errors.push(HarnessError::Cancelled.run_error());
                break;
            }
            if model_calls >= request.agent.limits.max_model_calls {
                result.status = RunStatus::LimitReached;
                result.model_call_limit_reached = true;
                result.errors.push(RunError {
                    code: "model_call_limit".into(),
                    message: "model call limit reached".into(),
                });
                break;
            }
            model_calls += 1;
            self.events.emit(RunEvent::ModelRequested {
                call_number: model_calls,
                model: model.clone(),
            });
            let call_cancellation = request.cancellation.child_token();
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
            let completion_result = match request.agent.limits.max_model_call_duration_ms {
                Some(limit) => {
                    match tokio::time::timeout(Duration::from_millis(limit), completion).await {
                        Ok(response) => response,
                        Err(_) => {
                            call_cancellation.cancel();
                            Err(HarnessError::TimedOut(
                                "model call duration limit reached".into(),
                            ))
                        }
                    }
                }
                None => completion.await,
            };
            let response = match completion_result {
                Ok(response) => response,
                Err(error) => {
                    result.status = if matches!(error, HarnessError::Cancelled) {
                        RunStatus::Cancelled
                    } else {
                        RunStatus::Failed
                    };
                    result.cancelled = matches!(error, HarnessError::Cancelled);
                    result.errors.push(error.run_error());
                    break;
                }
            };
            self.events.emit(RunEvent::ModelResponded {
                call_number: model_calls,
            });
            if let Some(output) = response.final_output {
                if let Err(error) = validate_output(&request.agent, &output) {
                    if output_repairs >= request.agent.limits.max_output_repairs {
                        result.errors.push(error.run_error());
                        break;
                    }
                    output_repairs += 1;
                    messages.push(Message {
                        role: MessageRole::Assistant,
                        content: output,
                        tool_call_id: None,
                    });
                    messages.push(Message::system(
                        "Return only JSON that satisfies the requested output schema.",
                    ));
                    continue;
                } else {
                    result.status = RunStatus::Completed;
                    result.final_output = Some(output);
                    break;
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
                if tool_calls >= request.agent.limits.max_tool_calls {
                    result.status = RunStatus::LimitReached;
                    result.tool_call_limit_reached = true;
                    result.errors.push(RunError {
                        code: "tool_call_limit".into(),
                        message: "tool call limit reached".into(),
                    });
                    break;
                }
                tool_calls += 1;
                result.tool_calls.push(call.clone());
                let arguments: Value = match serde_json::from_str(&call.arguments_json) {
                    Ok(value) => value,
                    Err(error) => {
                        self.reject(&mut result, &call, format!("malformed JSON: {error}"));
                        messages.push(Message::tool(
                            call.id.clone(),
                            &ToolResult::failure("malformed tool arguments"),
                        ));
                        continue;
                    }
                };
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
                    break;
                }
                let Some(tool) = self.tools.get(&call.tool_id) else {
                    self.reject(&mut result, &call, "unknown tool".into());
                    messages.push(Message::tool(
                        call.id.clone(),
                        &ToolResult::failure("unknown tool"),
                    ));
                    continue;
                };
                if !request
                    .agent
                    .tool_allowlist
                    .iter()
                    .any(|id| id == &call.tool_id)
                {
                    self.reject(&mut result, &call, "tool is not allowed for agent".into());
                    messages.push(Message::tool(
                        call.id.clone(),
                        &ToolResult::failure("tool is not allowed"),
                    ));
                    continue;
                }
                if let Err(error) = self.tools.validate(tool.as_ref(), &arguments) {
                    self.reject(&mut result, &call, error.to_string());
                    messages.push(Message::tool(
                        call.id.clone(),
                        &ToolResult::failure("tool arguments failed validation"),
                    ));
                    continue;
                }
                let decision = self
                    .policy
                    .decide(tool.definition(), &arguments, &request)
                    .await?;
                self.events.emit(RunEvent::PolicyDecided {
                    call_id: call.id.clone(),
                    decision: decision.clone(),
                });
                result.policy_decisions.push(decision.clone());
                match decision {
                    PolicyDecision::Deny { reason } => {
                        self.reject(&mut result, &call, format!("policy denied: {reason}"));
                        messages.push(Message::tool(
                            call.id.clone(),
                            &ToolResult::failure("policy denied"),
                        ));
                    }
                    PolicyDecision::RequireApproval { .. } => {
                        self.events.emit(RunEvent::ApprovalRequested {
                            call_id: call.id.clone(),
                            tool_id: call.tool_id.clone(),
                        });
                        let mut approval = self
                            .approvals
                            .approve(tool.definition(), &arguments, &request)
                            .await?;
                        approval.call_id = call.id.clone();
                        approval.tool_id = call.tool_id.clone();
                        let granted = approval.granted;
                        result.approvals.push(approval.clone());
                        if granted {
                            self.execute_tool(
                                &mut result,
                                &mut messages,
                                &call,
                                tool,
                                arguments,
                                &request.cancellation,
                            )
                            .await;
                        } else {
                            self.reject(
                                &mut result,
                                &call,
                                format!("approval denied: {}", approval.reason),
                            );
                            messages.push(Message::tool(
                                call.id.clone(),
                                &ToolResult::failure("approval denied"),
                            ));
                        }
                    }
                    PolicyDecision::Allow { .. } => {
                        self.execute_tool(
                            &mut result,
                            &mut messages,
                            &call,
                            tool,
                            arguments,
                            &request.cancellation,
                        )
                        .await
                    }
                }
            }
            if result.status == RunStatus::LimitReached {
                break;
            }
        }
        result.duration_ms = started.elapsed().as_millis() as u64;
        self.events.emit(RunEvent::Completed {
            status: result.status.clone(),
        });
        Ok(result)
    }
    fn reject(&self, result: &mut RunResult, call: &ToolCall, reason: String) {
        self.events.emit(RunEvent::ToolRejected {
            call_id: call.id.clone(),
            tool_id: call.tool_id.clone(),
            reason: reason.clone(),
        });
        result.errors.push(RunError {
            code: "tool_rejected".into(),
            message: reason,
        });
    }
    async fn execute_tool(
        &self,
        run: &mut RunResult,
        messages: &mut Vec<Message>,
        call: &ToolCall,
        tool: Arc<dyn Tool>,
        arguments: Value,
        cancellation: &CancellationToken,
    ) {
        let result = match tool.execute(arguments, cancellation.clone()).await {
            Ok(result) => result,
            Err(error) => {
                run.errors.push(error.run_error());
                ToolResult::failure(error.to_string())
            }
        };
        self.events.emit(RunEvent::ToolCompleted {
            call_id: call.id.clone(),
            tool_id: call.tool_id.clone(),
            ok: result.ok,
        });
        messages.push(Message::tool(call.id.clone(), &result));
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
fn validate_request(request: &RunRequest) -> Result<(), HarnessError> {
    if request.agent.id.trim().is_empty()
        || request.agent.name.trim().is_empty()
        || request.agent.version.trim().is_empty()
        || request.agent.default_model.trim().is_empty()
    {
        return Err(HarnessError::InvalidRequest(
            "agent id, name, version, and default model are required".into(),
        ));
    }
    if request.input.trim().is_empty() {
        return Err(HarnessError::InvalidRequest("input is required".into()));
    }
    if request.agent.limits.max_model_calls == 0
        || request.agent.limits.max_tool_calls == 0
        || request.agent.limits.max_identical_tool_calls == 0
    {
        return Err(HarnessError::InvalidRequest(
            "all limits must be greater than zero".into(),
        ));
    }
    if let Some(schema) = &request.agent.output_schema {
        jsonschema::validator_for(schema).map_err(|error| {
            HarnessError::InvalidRequest(format!("invalid output schema: {error}"))
        })?;
    }
    Ok(())
}

fn validate_output(agent: &AgentDefinition, output: &str) -> Result<(), HarnessError> {
    let Some(schema) = &agent.output_schema else {
        return Ok(());
    };
    let value = serde_json::from_str(output)
        .map_err(|error| HarnessError::InvalidOutput(format!("output is not JSON: {error}")))?;
    jsonschema::validator_for(schema)
        .map_err(|error| HarnessError::InvalidOutput(format!("invalid output schema: {error}")))?
        .validate(&value)
        .map_err(|error| HarnessError::InvalidOutput(error.to_string()))
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

pub mod mock {
    use super::*;
    #[derive(Clone, Debug)]
    pub enum MockStep {
        Response(ModelResponse),
        Error(HarnessError),
    }
    pub struct MockModelProvider {
        id: String,
        steps: Mutex<VecDeque<MockStep>>,
        requests: Mutex<Vec<ModelRequest>>,
    }
    impl MockModelProvider {
        pub fn scripted(steps: impl IntoIterator<Item = MockStep>) -> Self {
            Self {
                id: "mock".into(),
                steps: Mutex::new(steps.into_iter().collect()),
                requests: Mutex::new(vec![]),
            }
        }
        pub fn requests(&self) -> Vec<ModelRequest> {
            self.requests.lock().expect("mock lock poisoned").clone()
        }
    }
    #[async_trait]
    impl ModelProvider for MockModelProvider {
        fn id(&self) -> &str {
            &self.id
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                supports_tools: true,
                supports_streaming: false,
                supports_structured_output: true,
            }
        }
        async fn health(&self) -> Result<ProviderHealth, HarnessError> {
            Ok(ProviderHealth {
                healthy: true,
                detail: None,
            })
        }
        async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
            Ok(vec![ModelInfo {
                id: "mock-model".into(),
                capabilities: self.capabilities(),
            }])
        }
        async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError> {
            if request.cancellation.is_cancelled() {
                return Err(HarnessError::Cancelled);
            }
            self.requests
                .lock()
                .expect("mock lock poisoned")
                .push(request);
            match self.steps.lock().expect("mock lock poisoned").pop_front() {
                Some(MockStep::Response(response)) => Ok(response),
                Some(MockStep::Error(error)) => Err(error),
                None => Err(HarnessError::Provider("mock script exhausted".into())),
            }
        }
    }
    pub fn final_response(output: impl Into<String>) -> MockStep {
        MockStep::Response(ModelResponse {
            model: "mock-model".into(),
            final_output: Some(output.into()),
            tool_calls: vec![],
            usage: Usage::default(),
        })
    }
    pub fn tool_response(call: ToolCall) -> MockStep {
        MockStep::Response(ModelResponse {
            model: "mock-model".into(),
            final_output: None,
            tool_calls: vec![call],
            usage: Usage::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{final_response, tool_response, MockModelProvider};
    use serde_json::json;
    use std::sync::atomic::{AtomicU32, Ordering};
    struct TestTool {
        definition: ToolDefinition,
        calls: AtomicU32,
        fail: bool,
    }
    impl TestTool {
        fn new(id: &str, schema: Value) -> Self {
            Self {
                definition: ToolDefinition {
                    id: id.into(),
                    name: id.into(),
                    description: "test".into(),
                    arguments_schema: schema,
                    risk: ToolRisk::Low,
                    idempotent: true,
                    read_only: true,
                },
                calls: AtomicU32::new(0),
                fail: false,
            }
        }
    }
    #[async_trait]
    impl Tool for TestTool {
        fn definition(&self) -> &ToolDefinition {
            &self.definition
        }
        async fn execute(
            &self,
            _: Value,
            _: CancellationToken,
        ) -> Result<ToolResult, HarnessError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                Err(HarnessError::Tool("boom".into()))
            } else {
                Ok(ToolResult::success(json!({"ok":true})))
            }
        }
    }
    struct FixedPolicy(PolicyDecision);
    #[async_trait]
    impl PolicyEngine for FixedPolicy {
        async fn decide(
            &self,
            _: &ToolDefinition,
            _: &Value,
            _: &RunRequest,
        ) -> Result<PolicyDecision, HarnessError> {
            Ok(self.0.clone())
        }
    }
    struct FixedApproval(bool);
    #[async_trait]
    impl ApprovalHandler for FixedApproval {
        async fn approve(
            &self,
            tool: &ToolDefinition,
            _: &Value,
            _: &RunRequest,
        ) -> Result<ApprovalRecord, HarnessError> {
            Ok(ApprovalRecord {
                call_id: String::new(),
                tool_id: tool.id.clone(),
                granted: self.0,
                reason: "test".into(),
            })
        }
    }
    fn request() -> RunRequest {
        RunRequest {
            agent: AgentDefinition {
                id: "agent".into(),
                name: "Agent".into(),
                version: "1".into(),
                system_instructions: String::new(),
                default_model: "mock-model".into(),
                tool_allowlist: vec!["read".into()],
                limits: AgentLimits::default(),
                generation: GenerationOptions::default(),
                output_schema: None,
                metadata: JsonMap::new(),
            },
            input: "hello".into(),
            application_context: JsonMap::new(),
            history: vec![],
            metadata: JsonMap::new(),
            overrides: RunOverrides::default(),
            evaluation: JsonMap::new(),
            cancellation: CancellationToken::new(),
        }
    }
    fn call(id: &str, tool_id: &str, args: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            tool_id: tool_id.into(),
            arguments_json: args.into(),
        }
    }
    fn registry(tool: Arc<dyn Tool>) -> ToolRegistry {
        let mut tools = ToolRegistry::default();
        tools.register(tool).unwrap();
        tools
    }
    #[tokio::test]
    async fn completes_final_response() {
        let runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
            "done",
        )])))
        .build();
        assert_eq!(
            runner.run(request()).await.unwrap().final_output.as_deref(),
            Some("done")
        );
    }
    #[tokio::test]
    async fn executes_allowed_read_only_tool() {
        let tool = Arc::new(TestTool::new(
            "read",
            json!({"type":"object","required":["query"],"properties":{"query":{"type":"string"}}}),
        ));
        let runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
            tool_response(call("1", "read", r#"{"query":"x"}"#)),
            final_response("done"),
        ])))
        .tools(registry(tool.clone()))
        .build();
        assert_eq!(
            runner.run(request()).await.unwrap().status,
            RunStatus::Completed
        );
        assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn rejects_unknown_disallowed_and_malformed_calls() {
        let read = Arc::new(TestTool::new("read", json!({"type":"object"})));
        let write = Arc::new(TestTool::new("write", json!({"type":"object"})));
        let mut tools = registry(read.clone());
        tools.register(write.clone()).unwrap();
        let provider = Arc::new(MockModelProvider::scripted([
            tool_response(call("1", "missing", "{}")),
            tool_response(call("2", "write", "{}")),
            tool_response(call("3", "read", "not json")),
            final_response("done"),
        ]));
        let runner = AgentRunner::builder(provider).tools(tools).build();
        let result = runner.run(request()).await.unwrap();
        assert_eq!(result.status, RunStatus::Completed);
        assert_eq!(read.calls.load(Ordering::SeqCst), 0);
        assert_eq!(write.calls.load(Ordering::SeqCst), 0);
        assert_eq!(result.errors.len(), 3);
    }
    #[tokio::test]
    async fn rejects_schema_invalid_arguments() {
        let tool = Arc::new(TestTool::new(
            "read",
            json!({"type":"object","required":["query"]}),
        ));
        let runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
            tool_response(call("1", "read", "{}")),
            final_response("done"),
        ])))
        .tools(registry(tool.clone()))
        .build();
        let result = runner.run(request()).await.unwrap();
        assert_eq!(result.errors[0].code, "tool_rejected");
        assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    }
    #[tokio::test]
    async fn deny_and_approval_paths_are_recorded() {
        let tool = Arc::new(TestTool::new("read", json!({"type":"object"})));
        let provider = Arc::new(MockModelProvider::scripted([
            tool_response(call("1", "read", "{}")),
            final_response("done"),
        ]));
        let denied = AgentRunner::builder(provider)
            .tools(registry(tool.clone()))
            .policy(Arc::new(FixedPolicy(PolicyDecision::Deny {
                reason: "no".into(),
            })))
            .build()
            .run(request())
            .await
            .unwrap();
        assert_eq!(denied.errors[0].code, "tool_rejected");
        let granted_runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
            tool_response(call("2", "read", "{}")),
            final_response("done"),
        ])))
        .tools(registry(tool.clone()))
        .policy(Arc::new(FixedPolicy(PolicyDecision::RequireApproval {
            reason: "ask".into(),
        })))
        .approvals(Arc::new(FixedApproval(true)))
        .build();
        assert!(granted_runner.run(request()).await.unwrap().approvals[0].granted);
        let rejected_runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
            tool_response(call("3", "read", "{}")),
            final_response("done"),
        ])))
        .tools(registry(tool))
        .policy(Arc::new(FixedPolicy(PolicyDecision::RequireApproval {
            reason: "ask".into(),
        })))
        .approvals(Arc::new(FixedApproval(false)))
        .build();
        assert!(!rejected_runner.run(request()).await.unwrap().approvals[0].granted);
    }
    #[tokio::test]
    async fn tool_error_and_limits_stop_safely() {
        let mut bad = TestTool::new("read", json!({"type":"object"}));
        bad.fail = true;
        let runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
            tool_response(call("1", "read", "{}")),
            final_response("done"),
        ])))
        .tools(registry(Arc::new(bad)))
        .build();
        let result = runner.run(request()).await.unwrap();
        assert_eq!(result.status, RunStatus::Completed);
        assert_eq!(result.errors[0].code, "tool_error");
        let mut limited = request();
        limited.agent.limits.max_model_calls = 1;
        let result = AgentRunner::builder(Arc::new(MockModelProvider::scripted([tool_response(
            call("1", "read", "{}"),
        )])))
        .tools(registry(Arc::new(TestTool::new(
            "read",
            json!({"type":"object"}),
        ))))
        .build()
        .run(limited)
        .await
        .unwrap();
        assert!(result.model_call_limit_reached);
    }
    #[tokio::test]
    async fn tool_and_repeat_limits_and_cancellation_work() {
        let mut limited = request();
        limited.agent.limits.max_tool_calls = 1;
        let tool = Arc::new(TestTool::new("read", json!({"type":"object"})));
        let runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
            tool_response(call("1", "read", "{}")),
            tool_response(call("2", "read", "{}")),
        ])))
        .tools(registry(tool))
        .build();
        assert!(runner.run(limited).await.unwrap().tool_call_limit_reached);
        let mut repeated = request();
        repeated.agent.limits.max_identical_tool_calls = 1;
        let runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
            tool_response(call("1", "read", "{}")),
            tool_response(call("2", "read", "{}")),
        ])))
        .tools(registry(Arc::new(TestTool::new(
            "read",
            json!({"type":"object"}),
        ))))
        .build();
        assert!(
            runner
                .run(repeated)
                .await
                .unwrap()
                .repeated_tool_call_limit_reached
        );
        let cancelled = request();
        cancelled.cancellation.cancel();
        let result = AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
            "never",
        )])))
        .build()
        .run(cancelled)
        .await
        .unwrap();
        assert!(result.cancelled);
    }

    #[tokio::test]
    async fn repairs_invalid_structured_output_once() {
        let provider = Arc::new(MockModelProvider::scripted([
            final_response("not json"),
            final_response(r#"{"answer":"done"}"#),
        ]));
        let mut request = request();
        request.agent.output_schema = Some(json!({"type":"object","required":["answer"]}));
        let result = AgentRunner::builder(provider)
            .build()
            .run(request)
            .await
            .unwrap();
        assert_eq!(result.status, RunStatus::Completed);
        assert_eq!(result.final_output.as_deref(), Some(r#"{"answer":"done"}"#));
    }

    #[tokio::test]
    async fn mock_provider_reports_inventory_and_captures_request() {
        let provider = Arc::new(MockModelProvider::scripted([final_response("done")]));
        assert_eq!(provider.list_models().await.unwrap()[0].id, "mock-model");
        AgentRunner::builder(provider.clone())
            .build()
            .run(request())
            .await
            .unwrap();
        assert_eq!(provider.requests().len(), 1);
    }
}
