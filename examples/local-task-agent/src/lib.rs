//! A small application-owned task agent embedded directly in a Rust process.
//!
//! The example registers only task tools. It has no generic shell, filesystem, or
//! database tool, and uses a local in-memory task store for every runtime instance.

use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, tool_response, MockModelProvider},
    AgentDefinition, AgentLimits, AgentRunner, ApprovalHandler, ApprovalRecord, EventSink,
    GenerationOptions, HarnessError, InMemoryEventSink, JsonMap, ModelProvider, PolicyDecision,
    PolicyEngine, RunOverrides, RunRequest, Tool, ToolCall, ToolDefinition, ToolRegistry,
    ToolResult, ToolRisk,
};
use llama_harness_evals::{EvalError, EvalExecutionRequest, EvalExecutor, EvalObservation};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

pub const LIST_TASKS_TOOL: &str = "list_tasks";
pub const CREATE_TASK_TOOL: &str = "create_task";
pub const UPDATE_TASK_TOOL: &str = "update_task";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub status: String,
}

#[derive(Debug, Error)]
pub enum TaskStoreError {
    #[error("task store mutex is poisoned")]
    Poisoned,
    #[error("task already exists: {0}")]
    Duplicate(String),
    #[error("task was not found: {0}")]
    Missing(String),
    #[error("task title is required")]
    MissingTitle,
    #[error("task ID is required")]
    MissingId,
    #[error("task status is required")]
    MissingStatus,
    #[error("task argument is required: {0}")]
    MissingArgument(String),
}

#[derive(Default)]
pub struct TaskStore {
    tasks: Mutex<Vec<Task>>,
}

impl TaskStore {
    pub fn new(tasks: impl IntoIterator<Item = Task>) -> Result<Self, TaskStoreError> {
        let tasks: Vec<_> = tasks.into_iter().collect();
        for (index, task) in tasks.iter().enumerate() {
            if task.id.trim().is_empty() {
                return Err(TaskStoreError::MissingId);
            }
            if tasks[..index].iter().any(|existing| existing.id == task.id) {
                return Err(TaskStoreError::Duplicate(task.id.clone()));
            }
        }
        Ok(Self {
            tasks: Mutex::new(tasks),
        })
    }

    pub fn snapshot(&self) -> Result<Vec<Task>, TaskStoreError> {
        self.tasks
            .lock()
            .map(|tasks| tasks.clone())
            .map_err(|_| TaskStoreError::Poisoned)
    }

    fn create(&self, title: String) -> Result<Task, TaskStoreError> {
        let title = title.trim().to_owned();
        if title.is_empty() {
            return Err(TaskStoreError::MissingTitle);
        }
        let mut tasks = self.tasks.lock().map_err(|_| TaskStoreError::Poisoned)?;
        if tasks
            .iter()
            .any(|task| task.title.eq_ignore_ascii_case(&title))
        {
            return Err(TaskStoreError::Duplicate(title));
        }
        let task = Task {
            id: format!("task-{}", tasks.len().saturating_add(1)),
            title,
            status: "open".into(),
        };
        tasks.push(task.clone());
        Ok(task)
    }

    fn update(&self, id: &str, status: String) -> Result<Task, TaskStoreError> {
        let status = status.trim().to_owned();
        if status.is_empty() {
            return Err(TaskStoreError::MissingStatus);
        }
        let mut tasks = self.tasks.lock().map_err(|_| TaskStoreError::Poisoned)?;
        let task = tasks
            .iter_mut()
            .find(|task| task.id == id)
            .ok_or_else(|| TaskStoreError::Missing(id.into()))?;
        task.status = status;
        Ok(task.clone())
    }
}

#[derive(Clone, Copy)]
enum TaskToolKind {
    List,
    Create,
    Update,
}

pub struct TaskTool {
    kind: TaskToolKind,
    definition: ToolDefinition,
    store: Arc<TaskStore>,
}

impl TaskTool {
    fn new(kind: TaskToolKind, store: Arc<TaskStore>) -> Self {
        let (id, description, schema, risk, idempotent, read_only) = match kind {
            TaskToolKind::List => (
                LIST_TASKS_TOOL,
                "List application tasks.",
                json!({"type":"object","additionalProperties":false}),
                ToolRisk::Low,
                true,
                true,
            ),
            TaskToolKind::Create => (
                CREATE_TASK_TOOL,
                "Create a new task only after approval.",
                json!({
                    "type":"object",
                    "required":["title"],
                    "properties":{"title":{"type":"string","minLength":1,"maxLength":200}},
                    "additionalProperties":false
                }),
                ToolRisk::High,
                false,
                false,
            ),
            TaskToolKind::Update => (
                UPDATE_TASK_TOOL,
                "Update an existing task only after approval.",
                json!({
                    "type":"object",
                    "required":["id","status"],
                    "properties":{
                        "id":{"type":"string","minLength":1,"maxLength":100},
                        "status":{"type":"string","minLength":1,"maxLength":40}
                    },
                    "additionalProperties":false
                }),
                ToolRisk::High,
                false,
                false,
            ),
        };
        Self {
            kind,
            definition: ToolDefinition {
                id: id.into(),
                name: id.into(),
                description: description.into(),
                arguments_schema: schema,
                risk,
                idempotent,
                read_only,
            },
            store,
        }
    }
}

#[async_trait]
impl Tool for TaskTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(
        &self,
        arguments: Value,
        _: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        let task_result = match self.kind {
            TaskToolKind::List => self.store.snapshot().map(|tasks| json!({"tasks": tasks})),
            TaskToolKind::Create => argument_string(&arguments, "title")
                .and_then(|title| self.store.create(title).map(|task| json!({"task": task}))),
            TaskToolKind::Update => argument_string(&arguments, "id").and_then(|id| {
                argument_string(&arguments, "status").and_then(|status| {
                    self.store
                        .update(&id, status)
                        .map(|task| json!({"task": task}))
                })
            }),
        };
        match task_result {
            Ok(output) => Ok(ToolResult::success(output)),
            Err(error) => Ok(ToolResult::failure(error.to_string())),
        }
    }
}

fn argument_string(arguments: &Value, key: &str) -> Result<String, TaskStoreError> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| TaskStoreError::MissingArgument(key.into()))
}

pub struct TaskPolicy;

#[async_trait]
impl PolicyEngine for TaskPolicy {
    async fn decide(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        if tool.id == LIST_TASKS_TOOL {
            Ok(PolicyDecision::Allow {
                reason: "read-only task listing".into(),
            })
        } else {
            Ok(PolicyDecision::RequireApproval {
                reason: "task mutations require application approval".into(),
            })
        }
    }
}

pub struct StaticApproval {
    pub grant: bool,
}

#[async_trait]
impl ApprovalHandler for StaticApproval {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        Ok(ApprovalRecord {
            call_id: String::new(),
            tool_id: tool.id.clone(),
            granted: self.grant,
            reason: if self.grant {
                "example approval granted".into()
            } else {
                "example approval denied".into()
            },
        })
    }
}

pub fn task_agent_definition(model: impl Into<String>) -> AgentDefinition {
    AgentDefinition {
        id: "local-task-agent".into(),
        name: "Local Task Agent".into(),
        version: "1".into(),
        system_instructions: "Use only registered task tools. Preserve ambiguity as unresolved; never guess a state change.".into(),
        default_model: model.into(),
        tool_allowlist: vec![
            LIST_TASKS_TOOL.into(),
            CREATE_TASK_TOOL.into(),
            UPDATE_TASK_TOOL.into(),
        ],
        limits: AgentLimits {
            max_model_calls: 4,
            max_tool_calls: 3,
            ..AgentLimits::default()
        },
        generation: GenerationOptions::default(),
        output_schema: None,
        metadata: JsonMap::new(),
    }
}

pub struct TaskAgentRuntime {
    pub runner: AgentRunner,
    pub store: Arc<TaskStore>,
    pub agent: AgentDefinition,
}

pub fn build_runtime(
    provider: Arc<dyn ModelProvider>,
    store: Arc<TaskStore>,
    model: impl Into<String>,
    grant_approval: bool,
    event_sink: Arc<dyn EventSink>,
) -> Result<TaskAgentRuntime, HarnessError> {
    let mut tools = ToolRegistry::default();
    for kind in [
        TaskToolKind::List,
        TaskToolKind::Create,
        TaskToolKind::Update,
    ] {
        tools.register(Arc::new(TaskTool::new(kind, Arc::clone(&store))))?;
    }
    let agent = task_agent_definition(model);
    let runner = AgentRunner::builder(provider)
        .tools(tools)
        .policy(Arc::new(TaskPolicy))
        .approvals(Arc::new(StaticApproval {
            grant: grant_approval,
        }))
        .event_sink(event_sink)
        .build();
    Ok(TaskAgentRuntime {
        runner,
        store,
        agent,
    })
}

impl TaskAgentRuntime {
    pub async fn run(
        &self,
        input: impl Into<String>,
        model: Option<String>,
    ) -> Result<llama_harness_core::RunResult, HarnessError> {
        self.runner
            .run(RunRequest {
                agent: self.agent.clone(),
                input: input.into(),
                application_context: JsonMap::new(),
                history: vec![],
                metadata: JsonMap::new(),
                overrides: RunOverrides {
                    model,
                    generation: GenerationOptions::default(),
                },
                evaluation: JsonMap::new(),
                cancellation: CancellationToken::new(),
            })
            .await
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MockScenario {
    CompleteExisting,
    ListDuplicate,
    CreateNew,
    Ambiguous,
    DisallowedTool,
    MalformedArguments,
}

pub fn scripted_provider(scenario: MockScenario) -> MockModelProvider {
    let final_text = match scenario {
        MockScenario::Ambiguous => "I need clarification before changing a task.",
        MockScenario::DisallowedTool => "I cannot perform that action.",
        MockScenario::MalformedArguments => "I could not use that malformed request.",
        MockScenario::ListDuplicate => "That task already exists; no duplicate was created.",
        MockScenario::CreateNew => "Created the new task after approval.",
        MockScenario::CompleteExisting => "Updated the existing task after approval.",
    };
    let step = match scenario {
        MockScenario::CompleteExisting => Some(ToolCall {
            id: "update-1".into(),
            tool_id: UPDATE_TASK_TOOL.into(),
            arguments_json: r#"{"id":"task-1","status":"completed"}"#.into(),
        }),
        MockScenario::ListDuplicate => Some(ToolCall {
            id: "list-1".into(),
            tool_id: LIST_TASKS_TOOL.into(),
            arguments_json: "{}".into(),
        }),
        MockScenario::CreateNew => Some(ToolCall {
            id: "create-1".into(),
            tool_id: CREATE_TASK_TOOL.into(),
            arguments_json: r#"{"title":"Schedule dentist appointment"}"#.into(),
        }),
        MockScenario::DisallowedTool => Some(ToolCall {
            id: "bad-1".into(),
            tool_id: "delete_all_tasks".into(),
            arguments_json: "{}".into(),
        }),
        MockScenario::MalformedArguments => Some(ToolCall {
            id: "bad-json-1".into(),
            tool_id: CREATE_TASK_TOOL.into(),
            arguments_json: "{not-json".into(),
        }),
        MockScenario::Ambiguous => None,
    };
    match step {
        Some(call) => {
            MockModelProvider::scripted([tool_response(call), final_response(final_text)])
        }
        None => MockModelProvider::scripted([final_response(final_text)]),
    }
}

pub fn default_tasks() -> Vec<Task> {
    vec![Task {
        id: "task-1".into(),
        title: "Evening medication".into(),
        status: "open".into(),
    }]
}

pub struct TaskAgentEvalExecutor;

#[async_trait]
impl EvalExecutor for TaskAgentEvalExecutor {
    async fn execute(&self, request: EvalExecutionRequest) -> Result<EvalObservation, EvalError> {
        let tasks = request
            .fixture
            .as_ref()
            .and_then(|fixture| fixture.data.get("tasks"))
            .map(|tasks| serde_json::from_value::<Vec<Task>>(tasks.clone()))
            .transpose()
            .map_err(EvalError::Json)?
            .unwrap_or_else(default_tasks);
        let scenario = match request.case.id.as_str() {
            "explicit-completion" => MockScenario::CompleteExisting,
            "no-duplicate" => MockScenario::ListDuplicate,
            "create-new" => MockScenario::CreateNew,
            "ambiguous" => MockScenario::Ambiguous,
            "disallowed-tool" => MockScenario::DisallowedTool,
            "malformed-arguments" => MockScenario::MalformedArguments,
            "denied-approval" => MockScenario::CompleteExisting,
            "limit-stop" => MockScenario::ListDuplicate,
            case_id => {
                return Err(EvalError::Executor(format!(
                    "unsupported example case: {case_id}"
                )))
            }
        };
        let provider = Arc::new(scripted_provider(scenario));
        let store = Arc::new(
            TaskStore::new(tasks).map_err(|error| EvalError::Executor(error.to_string()))?,
        );
        let mut runtime = build_runtime(
            Arc::clone(&provider) as Arc<dyn ModelProvider>,
            Arc::clone(&store),
            request.model.clone(),
            matches!(
                request.case.id.as_str(),
                "create-new" | "explicit-completion"
            ),
            Arc::new(InMemoryEventSink::default()),
        )
        .map_err(|error| EvalError::Executor(error.to_string()))?;
        if request.case.id == "limit-stop" {
            runtime.agent.limits.max_model_calls = 1;
        }
        let run = runtime
            .run(request.case.input, Some(request.model))
            .await
            .map_err(|error| EvalError::Executor(error.to_string()))?;
        Ok(EvalObservation {
            model_calls: provider.requests().len() as u32,
            final_state: Some(
                json!({"tasks": store.snapshot().map_err(|error| EvalError::Executor(error.to_string()))?}),
            ),
            unresolved_items: (scenario == MockScenario::Ambiguous)
                .then(|| json!(["task action is ambiguous"])),
            agent_version: Some(runtime.agent.version.clone()),
            prompt_version: Some("local-task-agent-prompt-1".into()),
            run,
        })
    }
}
