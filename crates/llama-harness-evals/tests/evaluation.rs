use async_trait::async_trait;
use llama_harness_core::{EventRecord, RunEvent, RunResult, RunStatus, ToolCall};
use llama_harness_evals::{
    evaluate_suite, export_regression_case, is_json_subset, load_suite, replay_regression,
    EvalError, EvalExecutionRequest, EvalExecutor, EvalObservation, RegressionSource,
};
use llama_harness_observability::{SqliteEventSink, TraceStoreConfig};
use serde_json::{json, Value};
use std::sync::Mutex;

const SUITE: &str = r#"
version: 1
id: task-agent-core
name: Task Agent Core Regression Suite
agent: task-agent
agent_version: 2
prompt_version: prompt-7
models:
  - ollama:small
defaults:
  repeat: 2
  max_latency_ms: 100
cases:
  - id: explicit-completion
    fixture:
      id: medication-incomplete
      data:
        tasks:
          - id: task-123
            status: incomplete
    input: Complete the evening medication task
    expected:
      status: completed
      final_output_contains:
        - done
      structured_output_subset:
        done: true
      required_tools:
        - update_task
      forbidden_tools:
        - create_task
      tool_sequence:
        - update_task
      expected_tool_arguments:
        - tool_id: update_task
          arguments_subset:
            id: task-123
            status: completed
      final_state_subset:
        tasks:
          - id: task-123
            status: completed
      unresolved_items: []
      required_approval_tools:
        - update_task
      max_model_calls: 2
      max_tool_calls: 1
      max_latency_ms: 100
      expect_cancelled: false
"#;

#[derive(Default)]
struct RecordingExecutor {
    fixtures: Mutex<Vec<Option<Value>>>,
    requests: Mutex<Vec<EvalExecutionRequest>>,
}

#[async_trait]
impl EvalExecutor for RecordingExecutor {
    async fn execute(&self, request: EvalExecutionRequest) -> Result<EvalObservation, EvalError> {
        self.fixtures
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(request.fixture.as_ref().map(|fixture| fixture.data.clone()));
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(request.clone());
        Ok(EvalObservation {
            run: run_result(&request.model),
            model_calls: 2,
            final_state: Some(json!({
                "tasks": [{"id": "task-123", "status": "completed", "extra": true}]
            })),
            unresolved_items: Some(json!([])),
            agent_version: request.agent_version,
            prompt_version: request.prompt_version,
        })
    }
}

fn run_result(model: &str) -> RunResult {
    RunResult {
        id: "run-1".into(),
        status: RunStatus::Completed,
        final_output: Some(r#"{"done":true,"message":"done"}"#.into()),
        model: model.into(),
        tool_calls: vec![ToolCall {
            id: "call-1".into(),
            tool_id: "update_task".into(),
            arguments_json: r#"{"id":"task-123","status":"completed"}"#.into(),
        }],
        policy_decisions: vec![],
        approvals: vec![llama_harness_core::ApprovalRecord {
            call_id: "call-1".into(),
            tool_id: "update_task".into(),
            granted: true,
            reason: "test".into(),
        }],
        errors: vec![],
        duration_ms: 50,
        trace_id: "trace-1".into(),
        model_call_limit_reached: false,
        tool_call_limit_reached: false,
        repeated_tool_call_limit_reached: false,
        cancelled: false,
    }
}

#[test]
fn yaml_and_json_suites_validate_strictly_and_round_trip() {
    let suite = load_suite(SUITE, Some("yaml")).unwrap();
    assert_eq!(suite.id, "task-agent-core");
    assert_eq!(suite.defaults.repeat, 2);
    let json = serde_json::to_string(&suite).unwrap();
    assert_eq!(load_suite(&json, Some("json")).unwrap(), suite);
    assert!(matches!(
        load_suite(
            "id: invalid\nname: invalid\nagent: agent\nmodels: [model]\ncases: []\nunknown: true",
            Some("yaml")
        ),
        Err(EvalError::Yaml(_))
    ));
    assert!(matches!(
        load_suite(
            "id: invalid\nname: invalid\nagent: agent\nmodels: [model]\ncases: []",
            Some("yaml")
        ),
        Err(EvalError::InvalidSuite(message)) if message.contains("at least one case")
    ));
}

#[tokio::test]
async fn isolated_runs_produce_deterministic_assertions_and_trace_links() {
    let mut suite = load_suite(SUITE, Some("yaml")).unwrap();
    suite.cases[0].model = Some("ollama:case-default".into());
    let executor = RecordingExecutor::default();
    let report = evaluate_suite(&suite, &executor, &["ollama:override".into()], None)
        .await
        .unwrap();

    assert_eq!(report.results.len(), 2);
    assert_eq!(report.passed_count(), 2);
    assert!(report
        .results
        .iter()
        .all(|result| result.trace_id.as_deref() == Some("trace-1")));
    assert!(report
        .results
        .iter()
        .all(|result| result.model == "ollama:override"));
    let fixtures = executor
        .fixtures
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(fixtures.len(), 2);
    assert_eq!(fixtures[0], fixtures[1]);
    let requests = executor
        .requests
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert!(requests.iter().all(|request| request.fixture.is_some()));
    assert_eq!(requests[0].repetition, 1);
    assert_eq!(requests[1].repetition, 2);
}

#[tokio::test]
async fn failed_deterministic_assertion_is_reported_without_stopping_other_cases() {
    let mut suite = load_suite(SUITE, Some("yaml")).unwrap();
    suite.cases[0]
        .expected
        .forbidden_tools
        .push("update_task".into());
    let executor = RecordingExecutor::default();
    let report = evaluate_suite(&suite, &executor, &[], Some(1))
        .await
        .unwrap();

    assert_eq!(report.results.len(), 1);
    assert!(!report.results[0].passed);
    assert!(report.results[0]
        .failures
        .iter()
        .any(|failure| failure.rule == "forbidden_tools"));
}

#[tokio::test]
async fn regression_export_and_replay_use_explicit_fixture_data_not_trace_payloads() {
    let suite = load_suite(SUITE, Some("yaml")).unwrap();
    let executor = RecordingExecutor::default();
    let report = evaluate_suite(&suite, &executor, &[], Some(1))
        .await
        .unwrap();
    let regression = export_regression_case(
        RegressionSource {
            suite_id: suite.id.clone(),
            agent_id: suite.agent.clone(),
            agent_version: suite.agent_version.clone(),
            prompt_version: suite.prompt_version.clone(),
            prompt_override: suite.prompt_override.clone(),
            model: "ollama:small".into(),
        },
        &suite.cases[0],
        &report.results[0],
    )
    .unwrap();

    let serialized = serde_json::to_string(&regression).unwrap();
    assert!(serialized.contains("trace-1"));
    assert!(!serialized.contains("raw_payload"));
    assert!(!serialized.contains("chain_of_thought"));
    let store = SqliteEventSink::open_in_memory(TraceStoreConfig::default()).unwrap();
    store
        .append_with_raw(
            &EventRecord {
                run_id: "run-1".into(),
                trace_id: "trace-1".into(),
                sequence: 1,
                timestamp_ms: 1,
                event: RunEvent::ModelRequested {
                    call_number: 1,
                    model: "ollama:small".into(),
                },
            },
            Some(&json!({"request": "private trace payload"})),
        )
        .unwrap();
    assert_eq!(
        store.export_run("run-1").unwrap().unwrap().events[0].raw_payload,
        None
    );
    assert!(!serialized.contains("private trace payload"));
    let replayed = replay_regression(&regression, &executor).await.unwrap();
    assert!(replayed.passed);
    assert_eq!(replayed.trace_id.as_deref(), Some("trace-1"));
}

#[test]
fn json_subset_handles_nested_objects_and_unordered_array_members() {
    assert!(is_json_subset(
        &json!({"tasks": [{"id": "b"}], "mode": "safe"}),
        &json!({"tasks": [{"id": "a"}, {"id": "b", "status": "done"}], "mode": "safe"})
    ));
    assert!(!is_json_subset(
        &json!({"mode": "safe"}),
        &json!({"mode": "unsafe"})
    ));
}
