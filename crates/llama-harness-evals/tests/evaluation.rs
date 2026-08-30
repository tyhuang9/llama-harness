use async_trait::async_trait;
use llama_harness_core::{EventRecord, RunEvent, RunResult, RunStatus, RunStrategy, ToolCall};
use llama_harness_evals::{
    evaluate_suite, export_regression_case, is_json_subset, load_suite, replay_regression,
    AssertionFailure, EvalError, EvalExecutionRequest, EvalExecutor, EvalObservation,
    EvaluationCaseResult, EvaluationReport, RegressionSource, StrategyMetrics,
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
            strategy_metrics: StrategyMetrics::default(),
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
    let mut result = RunResult::new("run-1", RunStatus::Completed, model, "trace-1");
    result.final_output = Some(r#"{"done":true,"message":"done"}"#.into());
    result.tool_calls = vec![ToolCall::new(
        "call-1",
        "update_task",
        r#"{"id":"task-123","status":"completed"}"#,
    )];
    result.approvals = vec![llama_harness_core::ApprovalRecord::new(
        "call-1",
        "update_task",
        true,
        "test",
    )];
    result.duration_ms = 50;
    result
}

#[test]
fn yaml_and_json_suites_validate_strictly_and_round_trip() {
    let suite = load_suite(SUITE, Some("yaml")).unwrap();
    assert_eq!(suite.id, "task-agent-core");
    assert_eq!(suite.defaults.repeat, 2);
    assert_eq!(suite.strategies, vec![RunStrategy::Adaptive]);
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
async fn strategy_matrix_preserves_order_and_case_override_takes_precedence() {
    let mut suite = load_suite(SUITE, Some("yaml")).unwrap();
    suite.defaults.repeat = 1;
    suite.strategies = vec![
        RunStrategy::Programmatic,
        RunStrategy::Direct,
        RunStrategy::DeclarativePlan,
    ];
    let executor = RecordingExecutor::default();
    let report = evaluate_suite(&suite, &executor, &[], None).await.unwrap();
    assert_eq!(
        report
            .results
            .iter()
            .map(|result| result.strategy)
            .collect::<Vec<_>>(),
        suite.strategies
    );

    suite.cases[0].strategy = Some(RunStrategy::Direct);
    let executor = RecordingExecutor::default();
    let report = evaluate_suite(&suite, &executor, &[], None).await.unwrap();
    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].strategy, RunStrategy::Direct);
}

#[test]
fn suite_rejects_empty_and_duplicate_strategy_matrices() {
    let mut suite = load_suite(SUITE, Some("yaml")).unwrap();
    suite.strategies.clear();
    assert!(matches!(
        suite.validate(),
        Err(EvalError::InvalidSuite(message)) if message == "at least one strategy is required"
    ));

    suite.strategies = vec![RunStrategy::Direct, RunStrategy::Direct];
    assert!(matches!(
        suite.validate(),
        Err(EvalError::InvalidSuite(message)) if message == "duplicate suite strategy: direct"
    ));
}

#[test]
fn strategy_metrics_round_trip_without_inventing_unknown_values() {
    let metrics = StrategyMetrics::default();
    assert_eq!(metrics.unauthorized_effects, None);
    assert_eq!(metrics.duplicate_effects, None);
    assert_eq!(metrics.unintended_effects, None);
    assert_eq!(metrics.task_correct, None);
    assert_eq!(metrics.final_state_correct, None);
    assert_eq!(metrics.recovery_success, None);
    assert_eq!(metrics.tool_selection_accuracy, None);
    assert_eq!(metrics.input_tokens, None);
    assert_eq!(metrics.output_tokens, None);
    assert_eq!(metrics.wasted_tool_calls, None);
    assert!(!metrics.passes_readiness());

    let value = serde_json::to_value(&metrics).unwrap();
    assert_eq!(
        serde_json::from_value::<StrategyMetrics>(value).unwrap(),
        metrics
    );
    let ready = StrategyMetrics {
        unauthorized_effects: Some(0),
        duplicate_effects: Some(0),
        unintended_effects: Some(0),
        task_correct: Some(true),
        final_state_correct: Some(true),
        ..StrategyMetrics::default()
    };
    assert!(ready.passes_readiness());

    let unknown_safety = StrategyMetrics {
        task_correct: Some(true),
        final_state_correct: Some(true),
        ..StrategyMetrics::default()
    };
    assert!(!unknown_safety.passes_readiness());
}

#[test]
fn strategy_metrics_reject_invalid_tool_selection_accuracy() {
    for accuracy in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
        let metrics = StrategyMetrics {
            tool_selection_accuracy: Some(accuracy),
            ..StrategyMetrics::default()
        };
        assert!(metrics.validate().is_err());
        assert!(!metrics.passes_readiness());
    }
    for accuracy in [0.0, 0.5, 1.0] {
        let metrics = StrategyMetrics {
            tool_selection_accuracy: Some(accuracy),
            ..StrategyMetrics::default()
        };
        assert_eq!(metrics.validate(), Ok(()));
    }
}

fn readiness_result(
    strategy: RunStrategy,
    recovery_success: bool,
    duration_ms: u64,
    tokens: u64,
) -> EvaluationCaseResult {
    let mut result = EvaluationCaseResult::new("suite", "case", "model", 1);
    result.strategy = strategy;
    result.passed = true;
    result.status = Some(RunStatus::Completed);
    result.duration_ms = Some(duration_ms);
    result.model_calls = Some(2);
    result.tool_calls = Some(1);
    result.strategy_metrics = StrategyMetrics {
        unauthorized_effects: Some(0),
        duplicate_effects: Some(0),
        unintended_effects: Some(0),
        task_correct: Some(true),
        final_state_correct: Some(true),
        recovery_success: Some(recovery_success),
        tool_selection_accuracy: Some(0.9),
        input_tokens: Some(tokens / 2),
        output_tokens: Some(tokens - tokens / 2),
        wasted_tool_calls: Some(0),
    };
    result
}

#[test]
fn adaptive_readiness_is_input_order_independent_and_efficiency_is_nonblocking() {
    let adaptive = readiness_result(RunStrategy::Adaptive, false, 1_200, 1_200);
    let direct = readiness_result(RunStrategy::Direct, false, 50, 50);
    let declarative = readiness_result(RunStrategy::DeclarativePlan, true, 900, 900);
    assert!(adaptive.passes_readiness());

    let forward = EvaluationReport::new(
        "report-a",
        "suite",
        1,
        vec![adaptive.clone(), direct.clone(), declarative.clone()],
    )
    .adaptive_readiness();
    let reverse =
        EvaluationReport::new("report-b", "suite", 1, vec![declarative, direct, adaptive])
            .adaptive_readiness();
    assert_eq!(forward, reverse);
    assert!(forward.ready);
    assert_eq!(forward.comparisons.len(), 1);
    assert_eq!(
        forward.comparisons[0].best_forced_strategy,
        RunStrategy::DeclarativePlan
    );
    assert!(
        forward.comparisons[0].adaptive_duration_ms
            > forward.comparisons[0].best_forced_duration_ms
    );
}

#[test]
fn adaptive_readiness_fails_on_safety_or_correctness_regression() {
    let mut adaptive = readiness_result(RunStrategy::Adaptive, true, 100, 100);
    adaptive.strategy_metrics.unauthorized_effects = Some(1);
    let forced = readiness_result(RunStrategy::Direct, true, 100, 100);
    let readiness =
        EvaluationReport::new("report", "suite", 1, vec![adaptive, forced]).adaptive_readiness();
    assert!(!readiness.ready);
    assert!(readiness
        .failures
        .iter()
        .any(|failure| failure.code == "adaptive_hard_gate_failed"));
}

#[test]
fn complete_result_readiness_requires_passed_completed_and_failure_free() {
    let mut result = readiness_result(RunStrategy::Adaptive, true, 100, 100);
    result.passed = false;
    assert!(!result.passes_readiness());
    result.passed = true;
    result.status = Some(RunStatus::Failed);
    assert!(!result.passes_readiness());
    result.status = Some(RunStatus::Completed);
    result
        .failures
        .push(AssertionFailure::new("test", "failure"));
    assert!(!result.passes_readiness());
}

#[test]
fn adaptive_readiness_fails_closed_on_unknown_or_invalid_metrics() {
    let mut unknown = readiness_result(RunStrategy::Adaptive, true, 100, 100);
    unknown.strategy_metrics.unauthorized_effects = None;
    let forced = readiness_result(RunStrategy::Direct, true, 100, 100);
    let readiness = EvaluationReport::new("report", "suite", 1, vec![unknown, forced.clone()])
        .adaptive_readiness();
    assert!(!readiness.ready);
    assert!(readiness
        .failures
        .iter()
        .any(|failure| failure.code == "unknown_metrics"));

    let mut invalid = readiness_result(RunStrategy::Adaptive, true, 100, 100);
    invalid.strategy_metrics.tool_selection_accuracy = Some(1.5);
    let readiness =
        EvaluationReport::new("report", "suite", 1, vec![invalid, forced]).adaptive_readiness();
    assert!(!readiness.ready);
    assert!(readiness
        .failures
        .iter()
        .any(|failure| failure.code == "invalid_metrics"));
}

#[test]
fn adaptive_readiness_fails_closed_on_missing_or_duplicate_baselines() {
    let adaptive = readiness_result(RunStrategy::Adaptive, true, 100, 100);
    let forced = readiness_result(RunStrategy::Direct, true, 100, 100);

    let missing_forced =
        EvaluationReport::new("report", "suite", 1, vec![adaptive.clone()]).adaptive_readiness();
    assert!(!missing_forced.ready);
    assert_eq!(missing_forced.failures[0].code, "missing_forced_baseline");

    let missing_adaptive =
        EvaluationReport::new("report", "suite", 1, vec![forced.clone()]).adaptive_readiness();
    assert!(!missing_adaptive.ready);
    assert_eq!(
        missing_adaptive.failures[0].code,
        "missing_adaptive_baseline"
    );

    let duplicate = EvaluationReport::new(
        "report",
        "suite",
        1,
        vec![adaptive.clone(), adaptive, forced],
    )
    .adaptive_readiness();
    assert!(!duplicate.ready);
    assert_eq!(duplicate.failures[0].code, "duplicate_baseline");
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
    let mut suite = load_suite(SUITE, Some("yaml")).unwrap();
    suite.cases[0].strategy = Some(RunStrategy::Programmatic);
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
    assert_eq!(regression.case.strategy, Some(RunStrategy::Programmatic));

    let serialized = serde_json::to_string(&regression).unwrap();
    assert!(serialized.contains("trace-1"));
    assert!(!serialized.contains("raw_payload"));
    assert!(!serialized.contains("chain_of_thought"));
    let store = SqliteEventSink::open_in_memory(TraceStoreConfig::default()).unwrap();
    store
        .append_with_raw(
            &EventRecord::new(
                "run-1",
                "trace-1",
                1,
                1,
                RunEvent::ModelRequested {
                    call_number: 1,
                    model: "ollama:small".into(),
                },
            ),
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
    assert_eq!(replayed.strategy, RunStrategy::Programmatic);
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
