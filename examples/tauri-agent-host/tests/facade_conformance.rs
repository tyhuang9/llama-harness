use std::sync::{Arc, Mutex};

use llama_harness::{
    async_trait,
    mock::{final_response, tool_response, MockModelProvider},
    tauri::{
        ApprovalRouter, ApprovalRouterSettings, FrontendEmitter, RunRegistry, TauriEventSink,
        DEFAULT_APPROVAL_EVENT_NAME,
    },
    AgentDefinition, AgentRunner, CancellationToken, PolicyDecision, PolicyEngine, RunRequest,
    RunStatus, Tool, ToolCall, ToolDefinition, ToolRegistry, ToolResult, ToolRisk,
};
use serde::Serialize;

const MAIN: &str = "main";

#[derive(Clone, Default)]
struct TargetEmitter(Arc<Mutex<Vec<(String, String, serde_json::Value)>>>);

impl FrontendEmitter for TargetEmitter {
    fn emit<P: Serialize + Clone>(&self, event: &str, payload: P) -> Result<(), String> {
        self.0.lock().unwrap().push((
            MAIN.into(),
            event.into(),
            serde_json::to_value(payload).unwrap(),
        ));
        Ok(())
    }
}

#[derive(Clone)]
struct StateTool {
    definition: ToolDefinition,
    state: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Tool for StateTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(
        &self,
        _: serde_json::Value,
        _: CancellationToken,
    ) -> Result<ToolResult, llama_harness::HarnessError> {
        self.state.lock().unwrap().push(self.definition.id.clone());
        Ok(ToolResult::success(serde_json::json!({"ok": true})))
    }
}

struct ApprovalForWrites;

#[async_trait]
impl PolicyEngine for ApprovalForWrites {
    async fn decide(
        &self,
        tool: &ToolDefinition,
        _: &serde_json::Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, llama_harness::HarnessError> {
        Ok(if tool.read_only {
            PolicyDecision::Allow {
                reason: "read".into(),
            }
        } else {
            PolicyDecision::RequireApproval {
                reason: "write".into(),
            }
        })
    }
}

fn tool(id: &str, read_only: bool, state: Arc<Mutex<Vec<String>>>) -> StateTool {
    StateTool {
        definition: ToolDefinition {
            id: id.into(),
            name: id.into(),
            description: id.into(),
            arguments_schema: serde_json::json!({"type":"object"}),
            risk: ToolRisk::Medium,
            idempotent: read_only,
            read_only,
        },
        state,
    }
}

fn request(tool_id: &str) -> RunRequest {
    let mut agent = AgentDefinition::new("agent", "Agent", "1", "mock-model");
    agent.tool_allowlist = vec![tool_id.into()];
    RunRequest::new(agent, "run")
}

fn runner(
    tool: StateTool,
    responses: impl IntoIterator<Item = llama_harness::mock::MockStep>,
    router: Arc<ApprovalRouter<TargetEmitter>>,
    emitter: TargetEmitter,
) -> AgentRunner {
    let mut tools = ToolRegistry::default();
    tools.register(Arc::new(tool)).unwrap();
    AgentRunner::builder(Arc::new(MockModelProvider::scripted(responses)))
        .tools(tools)
        .policy(Arc::new(ApprovalForWrites))
        .approvals(router)
        .event_sink(Arc::new(TauriEventSink::new(emitter)))
        .build()
}

async fn approval_id_after(emitter: &TargetEmitter, previous_count: usize) -> String {
    for _ in 0..100 {
        let approval = {
            let records = emitter.0.lock().unwrap();
            let approvals = records
                .iter()
                .filter(|(_, event, _)| event == DEFAULT_APPROVAL_EVENT_NAME)
                .collect::<Vec<_>>();
            (approvals.len() > previous_count).then(|| {
                approvals.last().unwrap().2["approvalId"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
        };
        if let Some(id) = approval {
            return id;
        }
        tokio::task::yield_now().await;
    }
    panic!("approval event was not emitted")
}

fn approval_count(emitter: &TargetEmitter) -> usize {
    emitter
        .0
        .lock()
        .unwrap()
        .iter()
        .filter(|(_, event, _)| event == DEFAULT_APPROVAL_EVENT_NAME)
        .count()
}

#[tokio::test]
async fn facade_only_read_and_granted_write_follow_main_window_contract() {
    let emitter = TargetEmitter::default();
    let router = Arc::new(ApprovalRouter::new(emitter.clone()));
    let state = Arc::new(Mutex::new(vec![]));
    let read = tool("notes.read", true, Arc::clone(&state));
    let result = runner(
        read,
        [
            tool_response(ToolCall {
                id: "read-call".into(),
                tool_id: "notes.read".into(),
                arguments_json: "{}".into(),
            }),
            final_response("done"),
        ],
        Arc::clone(&router),
        emitter.clone(),
    )
    .run(request("notes.read"))
    .await
    .unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(*state.lock().unwrap(), ["notes.read"]);

    let write = tool("notes.write", false, Arc::clone(&state));
    let runner = Arc::new(runner(
        write,
        [
            tool_response(ToolCall {
                id: "write-call".into(),
                tool_id: "notes.write".into(),
                arguments_json: "{}".into(),
            }),
            final_response("done"),
        ],
        Arc::clone(&router),
        emitter.clone(),
    ));
    let task = tokio::spawn(async move { runner.run(request("notes.write")).await.unwrap() });
    let id = approval_id_after(&emitter, 0).await;
    assert!(router.respond(&id, true, "granted"));
    assert!(!router.respond(&id, true, "replay"));
    assert_eq!(task.await.unwrap().status, RunStatus::Completed);
    assert_eq!(*state.lock().unwrap(), ["notes.read", "notes.write"]);
    assert!(emitter
        .0
        .lock()
        .unwrap()
        .iter()
        .all(|(target, _, _)| target == MAIN));
}

#[tokio::test]
async fn facade_only_denial_and_cancellation_leave_no_write_and_clear_host_state() {
    let emitter = TargetEmitter::default();
    let router = Arc::new(ApprovalRouter::with_settings(
        emitter.clone(),
        DEFAULT_APPROVAL_EVENT_NAME,
        ApprovalRouterSettings {
            timeout: std::time::Duration::from_secs(1),
            ..ApprovalRouterSettings::default()
        },
    ));
    let state = Arc::new(Mutex::new(vec![]));
    let denied = Arc::new(runner(
        tool("notes.write", false, Arc::clone(&state)),
        [
            tool_response(ToolCall {
                id: "deny".into(),
                tool_id: "notes.write".into(),
                arguments_json: "{}".into(),
            }),
            final_response("done"),
        ],
        Arc::clone(&router),
        emitter.clone(),
    ));
    let task = tokio::spawn(async move { denied.run(request("notes.write")).await.unwrap() });
    let id = approval_id_after(&emitter, 0).await;
    assert!(router.respond(&id, false, "denied"));
    assert_eq!(task.await.unwrap().status, RunStatus::Completed);
    assert!(state.lock().unwrap().is_empty());

    let registry = Arc::new(RunRegistry::default());
    let mut pending_request = request("notes.write");
    let run_id = registry.register(&mut pending_request).unwrap();
    let approvals_before = approval_count(&emitter);
    let cancelling = Arc::new(runner(
        tool("notes.write", false, Arc::clone(&state)),
        [tool_response(ToolCall {
            id: "cancel".into(),
            tool_id: "notes.write".into(),
            arguments_json: "{}".into(),
        })],
        Arc::clone(&router),
        emitter.clone(),
    ));
    let task = tokio::spawn(async move { cancelling.run(pending_request).await.unwrap() });
    let _ = approval_id_after(&emitter, approvals_before).await;
    assert_eq!(router.cancel_run(&run_id), 1);
    assert!(registry.cancel(&run_id));
    assert!(task.await.unwrap().cancelled);
    assert!(registry.complete(&run_id));
    assert_eq!(router.pending_count(), 0);
    assert!(state.lock().unwrap().is_empty());
}
