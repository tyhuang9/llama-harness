use async_trait::async_trait;
use llama_harness_core::{EventRecord, RunEvent, RunResult, RunStatus, RunStrategy, ToolCall};
use llama_harness_evals::{
    evaluate_suite, export_regression_case, is_json_subset, load_suite, replay_regression,
    AssertionFailure, EvalError, EvalExecutionRequest, EvalExecutor, EvalObservation,
    EvaluationCaseResult, EvaluationReport, ForcedCandidateDisposition, RegressionSource,
    StrategyMetrics, StrategySelectionCriterion,
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
        Ok(EvalObservation::new(run_result(&request.model), 2)
            .with_final_state(Some(json!({
                "tasks": [{"id": "task-123", "status": "completed", "extra": true}]
            })))
            .with_unresolved_items(Some(json!([])))
            .with_agent_version(request.agent_version)
            .with_prompt_version(request.prompt_version))
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

fn cohort_result(
    case_id: &str,
    strategy: RunStrategy,
    repetition: u32,
    recovery_success: bool,
    tool_selection_accuracy: f64,
    duration_ms: u64,
    tokens: u64,
) -> EvaluationCaseResult {
    let mut result = readiness_result(strategy, recovery_success, duration_ms, tokens);
    result.case_id = case_id.into();
    result.repetition = repetition;
    result.strategy_metrics.tool_selection_accuracy = Some(tool_selection_accuracy);
    result
}

fn cohort_strategy_results(
    case_id: &str,
    strategy: RunStrategy,
    samples: &[(bool, f64, u64, u64)],
) -> Vec<EvaluationCaseResult> {
    samples
        .iter()
        .enumerate()
        .map(
            |(index, &(recovery_success, tool_selection_accuracy, duration_ms, tokens))| {
                cohort_result(
                    case_id,
                    strategy,
                    u32::try_from(index + 1).unwrap(),
                    recovery_success,
                    tool_selection_accuracy,
                    duration_ms,
                    tokens,
                )
            },
        )
        .collect()
}

fn readiness_for_cohort(
    case_id: &str,
    adaptive: &[(bool, f64, u64, u64)],
    direct: &[(bool, f64, u64, u64)],
    declarative: &[(bool, f64, u64, u64)],
) -> llama_harness_evals::AdaptiveReadiness {
    let mut results = cohort_strategy_results(case_id, RunStrategy::Adaptive, adaptive);
    results.extend(cohort_strategy_results(
        case_id,
        RunStrategy::Direct,
        direct,
    ));
    results.extend(cohort_strategy_results(
        case_id,
        RunStrategy::DeclarativePlan,
        declarative,
    ));
    EvaluationReport::new("cohort-ranking", "suite", 1, results).adaptive_readiness()
}

#[test]
fn adaptive_readiness_is_input_order_independent_and_efficiency_is_nonblocking() {
    let adaptive = readiness_result(RunStrategy::Adaptive, false, 1_200, 1_200);
    let direct = readiness_result(RunStrategy::Direct, false, 50, 50);
    let declarative = readiness_result(RunStrategy::DeclarativePlan, true, 900, 900);
    let mut programmatic = readiness_result(RunStrategy::Programmatic, true, 10, 10);
    programmatic.strategy_metrics.unauthorized_effects = Some(1);
    assert!(adaptive.passes_readiness());

    let forward = EvaluationReport::new(
        "report-a",
        "suite",
        1,
        vec![
            adaptive.clone(),
            direct.clone(),
            declarative.clone(),
            programmatic.clone(),
        ],
    )
    .adaptive_readiness();
    let reverse = EvaluationReport::new(
        "report-b",
        "suite",
        1,
        vec![programmatic, declarative, direct, adaptive],
    )
    .adaptive_readiness();
    assert_eq!(forward, reverse);
    assert!(forward.ready);
    assert_eq!(forward.comparisons.len(), 1);
    assert_eq!(forward.comparisons[0].sample_count, 1);
    assert_eq!(forward.comparisons[0].repetition, 0);
    assert_eq!(
        forward.comparisons[0].best_forced_strategy,
        RunStrategy::DeclarativePlan
    );
    assert!(
        forward.comparisons[0].adaptive.duration_ms
            > forward.comparisons[0].best_forced.duration_ms
    );
    assert!(!forward.comparisons[0].adaptive.recovery_success);
    assert_eq!(forward.comparisons[0].adaptive.tool_selection_accuracy, 0.9);
    assert!(forward.comparisons[0].best_forced.recovery_success);
    assert_eq!(forward.comparisons[0].adaptive.total_tokens, 1_200);
    assert_eq!(forward.comparisons[0].adaptive.p50_latency_ms, 1_200);
    assert_eq!(forward.comparisons[0].adaptive.p95_latency_ms, 1_200);
    assert_eq!(forward.comparisons[0].adaptive.model_calls, 2);
    assert_eq!(forward.comparisons[0].adaptive.tool_calls, 1);
    assert_eq!(forward.comparisons[0].adaptive.wasted_tool_calls, 0);
    assert_eq!(forward.comparisons[0].best_forced.model_calls, 2);
    assert_eq!(forward.comparisons[0].best_forced.tool_calls, 1);
    assert_eq!(forward.comparisons[0].best_forced.wasted_tool_calls, 0);
    let candidates = &forward.comparisons[0].forced_candidates;
    assert_eq!(candidates.len(), 3);
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.strategy)
            .collect::<Vec<_>>(),
        vec![
            RunStrategy::Direct,
            RunStrategy::DeclarativePlan,
            RunStrategy::Programmatic,
        ]
    );
    assert!(matches!(
        candidates[0].disposition,
        ForcedCandidateDisposition::Outranked {
            decisive_criterion: StrategySelectionCriterion::RecoverySuccessRate
        }
    ));
    assert_eq!(
        candidates[1].disposition,
        ForcedCandidateDisposition::Selected
    );
    assert!(matches!(
        &candidates[2].disposition,
        ForcedCandidateDisposition::Ineligible { code, .. }
            if code == "safety_hard_gate_failed"
    ));
}

#[test]
fn cohort_percentiles_use_nearest_rank_for_odd_and_even_sample_counts() {
    let mut results = cohort_strategy_results(
        "odd",
        RunStrategy::Adaptive,
        &[
            (true, 0.9, 11, 10),
            (true, 0.9, 21, 10),
            (true, 0.9, 31, 10),
            (true, 0.9, 41, 10),
            (true, 0.9, 51, 10),
        ],
    );
    results.extend(cohort_strategy_results(
        "odd",
        RunStrategy::Direct,
        &[
            (true, 0.9, 10, 10),
            (true, 0.9, 20, 10),
            (true, 0.9, 30, 10),
            (true, 0.9, 40, 10),
            (true, 0.9, 50, 10),
        ],
    ));
    results.extend(cohort_strategy_results(
        "even",
        RunStrategy::Adaptive,
        &[
            (true, 0.9, 11, 10),
            (true, 0.9, 21, 10),
            (true, 0.9, 31, 10),
            (true, 0.9, 41, 10),
        ],
    ));
    results.extend(cohort_strategy_results(
        "even",
        RunStrategy::Direct,
        &[
            (true, 0.9, 10, 10),
            (true, 0.9, 20, 10),
            (true, 0.9, 30, 10),
            (true, 0.9, 40, 10),
        ],
    ));

    let readiness = EvaluationReport::new("percentiles", "suite", 1, results).adaptive_readiness();
    assert!(readiness.ready, "{readiness:#?}");
    assert_eq!(readiness.comparisons.len(), 2);
    let odd = &readiness.comparisons[0];
    assert_eq!(odd.case_id, "even");
    assert_eq!(odd.sample_count, 4);
    assert_eq!(odd.best_forced.p50_latency_ms, 20);
    assert_eq!(odd.best_forced.p95_latency_ms, 40);
    let even = &readiness.comparisons[1];
    assert_eq!(even.case_id, "odd");
    assert_eq!(even.sample_count, 5);
    assert_eq!(even.best_forced.p50_latency_ms, 30);
    assert_eq!(even.best_forced.p95_latency_ms, 50);
}

#[test]
fn cohort_p95_reports_a_high_tail_outlier_at_the_nearest_rank() {
    let direct = [
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 10, 10),
        (true, 0.9, 500, 10),
        (true, 0.9, 1_000, 10),
    ];
    let adaptive = direct
        .map(|(recovery, accuracy, latency, tokens)| (recovery, accuracy, latency + 1, tokens));
    let readiness = readiness_for_cohort("outlier", &adaptive, &direct, &[]);
    assert!(readiness.ready, "{readiness:#?}");
    assert_eq!(readiness.comparisons[0].sample_count, 20);
    assert_eq!(readiness.comparisons[0].best_forced.p50_latency_ms, 10);
    assert_eq!(readiness.comparisons[0].best_forced.p95_latency_ms, 500);
}

#[test]
fn cohort_readiness_fails_closed_on_duplicate_or_mismatched_repetitions() {
    let adaptive = cohort_strategy_results(
        "mismatch",
        RunStrategy::Adaptive,
        &[(true, 0.9, 10, 10), (true, 0.9, 20, 10)],
    );
    let direct = cohort_strategy_results("mismatch", RunStrategy::Direct, &[(true, 0.9, 10, 10)]);
    let mismatch = EvaluationReport::new(
        "mismatch",
        "suite",
        1,
        adaptive.into_iter().chain(direct).collect(),
    )
    .adaptive_readiness();
    assert!(!mismatch.ready);
    assert!(mismatch
        .failures
        .iter()
        .any(|failure| failure.code == "mismatched_repetition_sets"));

    let adaptive = cohort_result("duplicate", RunStrategy::Adaptive, 1, true, 0.9, 10, 10);
    let duplicate = adaptive.clone();
    let direct = cohort_result("duplicate", RunStrategy::Direct, 1, true, 0.9, 10, 10);
    let duplicate =
        EvaluationReport::new("duplicate", "suite", 1, vec![adaptive, duplicate, direct])
            .adaptive_readiness();
    assert!(!duplicate.ready);
    assert!(duplicate
        .failures
        .iter()
        .any(|failure| failure.code == "duplicate_baseline"));
}

#[test]
fn one_unsafe_adaptive_sample_invalidates_its_entire_cohort() {
    let mut unsafe_adaptive = cohort_result("unsafe", RunStrategy::Adaptive, 2, true, 0.9, 20, 10);
    unsafe_adaptive.strategy_metrics.unauthorized_effects = Some(1);
    let readiness = EvaluationReport::new(
        "unsafe",
        "suite",
        1,
        vec![
            cohort_result("unsafe", RunStrategy::Adaptive, 1, true, 0.9, 10, 10),
            unsafe_adaptive,
            cohort_result("unsafe", RunStrategy::Direct, 1, true, 0.9, 10, 10),
            cohort_result("unsafe", RunStrategy::Direct, 2, true, 0.9, 20, 10),
        ],
    )
    .adaptive_readiness();
    assert!(!readiness.ready);
    assert!(readiness.comparisons.is_empty());
    assert!(readiness
        .failures
        .iter()
        .any(|failure| failure.code == "adaptive_hard_gate_failed"));
}

#[test]
fn cohort_ranking_prioritizes_reliability_before_latency() {
    let recovery_winner = readiness_for_cohort(
        "recovery-rate",
        &[(true, 0.9, 10, 10), (true, 0.9, 10, 10)],
        &[(true, 0.9, 1, 1), (false, 0.9, 1, 1)],
        &[(true, 0.9, 100, 100), (true, 0.9, 100, 100)],
    );
    assert!(recovery_winner.ready, "{recovery_winner:#?}");
    assert_eq!(
        recovery_winner.comparisons[0].best_forced_strategy,
        RunStrategy::DeclarativePlan,
        "recovery success rate must outrank P50 latency"
    );
    assert_eq!(
        recovery_winner.comparisons[0]
            .best_forced
            .recovery_success_rate,
        1.0
    );
    assert!(matches!(
        recovery_winner.comparisons[0].forced_candidates[0].disposition,
        ForcedCandidateDisposition::Outranked {
            decisive_criterion: StrategySelectionCriterion::RecoverySuccessRate
        }
    ));

    let accuracy_winner = readiness_for_cohort(
        "accuracy-mean",
        &[(true, 0.9, 10, 10), (true, 0.9, 10, 10)],
        &[(true, 0.4, 1, 1), (true, 0.4, 1, 1)],
        &[(true, 0.8, 100, 100), (true, 0.8, 100, 100)],
    );
    assert!(accuracy_winner.ready, "{accuracy_winner:#?}");
    assert_eq!(
        accuracy_winner.comparisons[0].best_forced_strategy,
        RunStrategy::DeclarativePlan,
        "mean tool-selection accuracy must outrank P50 latency"
    );
    assert!(matches!(
        accuracy_winner.comparisons[0].forced_candidates[0].disposition,
        ForcedCandidateDisposition::Outranked {
            decisive_criterion: StrategySelectionCriterion::ToolSelectionAccuracy
        }
    ));
}

#[test]
fn cohort_ranking_prioritizes_p50_then_p95_before_cost() {
    let p50_winner = readiness_for_cohort(
        "p50-before-p95",
        &[
            (true, 0.9, 10, 10),
            (true, 0.9, 10, 10),
            (true, 0.9, 10, 10),
            (true, 0.9, 10, 10),
        ],
        &[
            (true, 0.9, 1, 10),
            (true, 0.9, 1, 10),
            (true, 0.9, 100, 10),
            (true, 0.9, 100, 10),
        ],
        &[
            (true, 0.9, 2, 10),
            (true, 0.9, 2, 10),
            (true, 0.9, 2, 10),
            (true, 0.9, 2, 10),
        ],
    );
    assert!(p50_winner.ready, "{p50_winner:#?}");
    assert_eq!(
        p50_winner.comparisons[0].best_forced_strategy,
        RunStrategy::Direct,
        "P50 must outrank P95"
    );
    assert!(matches!(
        p50_winner.comparisons[0].forced_candidates[1].disposition,
        ForcedCandidateDisposition::Outranked {
            decisive_criterion: StrategySelectionCriterion::P50LatencyMs
        }
    ));

    let p95_winner = readiness_for_cohort(
        "p95-before-cost",
        &[
            (true, 0.9, 10, 10),
            (true, 0.9, 10, 10),
            (true, 0.9, 10, 10),
            (true, 0.9, 10, 10),
        ],
        &[
            (true, 0.9, 10, 100),
            (true, 0.9, 10, 100),
            (true, 0.9, 10, 100),
            (true, 0.9, 100, 100),
        ],
        &[
            (true, 0.9, 10, 1),
            (true, 0.9, 10, 1),
            (true, 0.9, 10, 1),
            (true, 0.9, 200, 1),
        ],
    );
    assert!(p95_winner.ready, "{p95_winner:#?}");
    assert_eq!(
        p95_winner.comparisons[0].best_forced_strategy,
        RunStrategy::Direct,
        "P95 must outrank aggregate cost"
    );
    assert!(
        p95_winner.comparisons[0].best_forced.total_tokens
            > p95_winner.comparisons[0].forced_candidates[1]
                .metrics
                .total_tokens
    );
    assert!(matches!(
        p95_winner.comparisons[0].forced_candidates[1].disposition,
        ForcedCandidateDisposition::Outranked {
            decisive_criterion: StrategySelectionCriterion::P95LatencyMs
        }
    ));
}

#[test]
fn cohort_aggregation_is_input_order_independent() {
    let adaptive = cohort_strategy_results(
        "order-independent",
        RunStrategy::Adaptive,
        &[
            (true, 0.9, 31, 30),
            (true, 0.9, 11, 10),
            (true, 0.9, 41, 40),
            (true, 0.9, 21, 20),
        ],
    );
    let direct = cohort_strategy_results(
        "order-independent",
        RunStrategy::Direct,
        &[
            (true, 0.9, 30, 30),
            (true, 0.9, 10, 10),
            (true, 0.9, 40, 40),
            (true, 0.9, 20, 20),
        ],
    );
    let declarative = cohort_strategy_results(
        "order-independent",
        RunStrategy::DeclarativePlan,
        &[
            (true, 0.8, 25, 25),
            (true, 0.8, 25, 25),
            (true, 0.8, 25, 25),
            (true, 0.8, 25, 25),
        ],
    );
    let mut forward = adaptive.clone();
    forward.extend(direct.clone());
    forward.extend(declarative.clone());
    let mut reverse = declarative;
    reverse.reverse();
    let mut direct_reverse = direct;
    direct_reverse.reverse();
    reverse.extend(direct_reverse);
    let mut adaptive_reverse = adaptive;
    adaptive_reverse.reverse();
    reverse.extend(adaptive_reverse);

    let forward = EvaluationReport::new("forward", "suite", 1, forward).adaptive_readiness();
    let reverse = EvaluationReport::new("reverse", "suite", 1, reverse).adaptive_readiness();
    assert_eq!(forward, reverse);
    assert!(forward.ready, "{forward:#?}");
    assert_eq!(forward.comparisons[0].sample_count, 4);
    assert_eq!(forward.comparisons[0].best_forced.p50_latency_ms, 20);
    assert_eq!(forward.comparisons[0].best_forced.p95_latency_ms, 40);
}

fn selected_forced(direct: EvaluationCaseResult, declarative: EvaluationCaseResult) -> RunStrategy {
    EvaluationReport::new(
        "ranking",
        "suite",
        1,
        vec![
            readiness_result(RunStrategy::Adaptive, true, 100, 100),
            direct,
            declarative,
        ],
    )
    .adaptive_readiness()
    .comparisons[0]
        .best_forced_strategy
}

#[test]
fn forced_ranking_applies_every_tie_break_in_contract_order() {
    let mut direct = readiness_result(RunStrategy::Direct, true, 1_000, 1_000);
    let declarative = readiness_result(RunStrategy::DeclarativePlan, true, 1, 1);
    direct.strategy_metrics.tool_selection_accuracy = Some(0.95);
    assert_eq!(
        selected_forced(direct, declarative),
        RunStrategy::Direct,
        "higher accuracy must beat lower latency and cost"
    );

    let direct = readiness_result(RunStrategy::Direct, true, 10, 100);
    let declarative = readiness_result(RunStrategy::DeclarativePlan, true, 20, 1);
    assert_eq!(selected_forced(direct, declarative), RunStrategy::Direct);

    let direct = readiness_result(RunStrategy::Direct, true, 10, 10);
    let declarative = readiness_result(RunStrategy::DeclarativePlan, true, 10, 20);
    assert_eq!(selected_forced(direct, declarative), RunStrategy::Direct);

    let mut direct = readiness_result(RunStrategy::Direct, true, 10, 10);
    let mut declarative = readiness_result(RunStrategy::DeclarativePlan, true, 10, 10);
    direct.model_calls = Some(1);
    declarative.model_calls = Some(2);
    assert_eq!(selected_forced(direct, declarative), RunStrategy::Direct);

    let mut direct = readiness_result(RunStrategy::Direct, true, 10, 10);
    let mut declarative = readiness_result(RunStrategy::DeclarativePlan, true, 10, 10);
    direct.tool_calls = Some(1);
    declarative.tool_calls = Some(2);
    assert_eq!(selected_forced(direct, declarative), RunStrategy::Direct);

    let direct = readiness_result(RunStrategy::Direct, true, 10, 10);
    let mut declarative = readiness_result(RunStrategy::DeclarativePlan, true, 10, 10);
    declarative.strategy_metrics.wasted_tool_calls = Some(1);
    assert_eq!(selected_forced(direct, declarative), RunStrategy::Direct);

    let direct = readiness_result(RunStrategy::Direct, true, 10, 10);
    let declarative = readiness_result(RunStrategy::DeclarativePlan, true, 10, 10);
    assert_eq!(
        selected_forced(direct, declarative),
        RunStrategy::Direct,
        "stable enum rank must resolve a complete tie"
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
                "execution-1",
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
