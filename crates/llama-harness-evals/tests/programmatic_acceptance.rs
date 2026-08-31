use async_trait::async_trait;
use llama_harness_core::{ApprovalRecord, RunResult, RunStatus, RunStrategy, ToolCall};
use llama_harness_evals::{
    evaluate_suite, load_suite, EvalError, EvalExecutionRequest, EvalExecutor, EvalObservation,
    StrategyMetrics,
};
use serde_json::{json, Value};
use std::sync::Mutex;

const SUITE: &str = include_str!("fixtures/programmatic-acceptance.yaml");

#[derive(Default)]
struct AcceptanceExecutor {
    requests: Mutex<Vec<EvalExecutionRequest>>,
}

#[async_trait]
impl EvalExecutor for AcceptanceExecutor {
    async fn execute(&self, request: EvalExecutionRequest) -> Result<EvalObservation, EvalError> {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(request.clone());
        let data = &request.fixture.as_ref().expect("acceptance fixture").data;
        let tool_ids = data["tools"]
            .as_array()
            .expect("fixture tools")
            .iter()
            .map(|tool| tool.as_str().expect("tool id"))
            .collect::<Vec<_>>();
        let mut run = RunResult::new(
            format!("{}-{:?}", request.case.id, request.strategy),
            RunStatus::Completed,
            &request.model,
            "fixture-trace",
        );
        run.final_output = Some(format!("{} done", data["scenario"].as_str().unwrap()));
        run.tool_calls = tool_ids
            .iter()
            .enumerate()
            .map(|(index, tool_id)| ToolCall::new(format!("call-{index}"), *tool_id, "{}"))
            .collect();
        run.approvals = data["approval_tools"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|tool| ApprovalRecord::new("fixture", tool.as_str().unwrap(), true, "granted"))
            .collect();
        let metrics = StrategyMetrics {
            unauthorized_effects: Some(0),
            duplicate_effects: Some(0),
            unintended_effects: Some(0),
            task_correct: Some(true),
            final_state_correct: Some(true),
            recovery_success: Some(true),
            ..StrategyMetrics::default()
        };
        let model_calls = if request.strategy == RunStrategy::Programmatic {
            2
        } else {
            1
        };
        Ok(EvalObservation::new(run, model_calls)
            .with_strategy_metrics(metrics)
            .with_final_state(Some(json!({"scenario": data["scenario"].clone()}))))
    }
}

#[tokio::test]
async fn executable_programmatic_acceptance_matrix_forces_strategies_without_adaptive_selection() {
    let suite = load_suite(SUITE, Some("yaml")).unwrap();
    assert_eq!(
        suite.strategies,
        vec![
            RunStrategy::Direct,
            RunStrategy::DeclarativePlan,
            RunStrategy::Programmatic,
        ]
    );
    let executor = AcceptanceExecutor::default();
    let report = evaluate_suite(&suite, &executor, &[], None).await.unwrap();

    assert!(report.results.iter().all(|result| result.passed));
    assert_eq!(report.results.len(), 22);
    assert!(report
        .results
        .iter()
        .all(|result| result.strategy != RunStrategy::Adaptive));
    assert!(report
        .results
        .iter()
        .filter(|result| result.strategy == RunStrategy::Programmatic)
        .all(|result| {
            result.model_calls == Some(2)
                && result.tool_calls.is_some()
                && result.strategy_metrics.passes_readiness()
        }));
    assert_eq!(
        report
            .results
            .iter()
            .filter(|result| result.case_id == "capability-downgrade")
            .map(|result| result.strategy)
            .collect::<Vec<_>>(),
        vec![RunStrategy::Direct]
    );
    let requests = executor
        .requests
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(requests.iter().all(|request| {
        request.strategy != RunStrategy::Adaptive
            && (request.case.id != "capability-downgrade"
                || request.strategy == RunStrategy::Direct)
    }));
}

#[test]
fn programmatic_acceptance_fixture_declares_all_required_scenarios() {
    let suite = load_suite(SUITE, Some("yaml")).unwrap();
    let scenarios = suite
        .cases
        .iter()
        .map(|case| {
            case.fixture
                .as_ref()
                .and_then(|fixture| fixture.data["scenario"].as_str())
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        scenarios,
        vec![
            "branch",
            "loop",
            "fanout",
            "filter",
            "reduce-aggregate",
            "mixed-approval",
            "partial-failure",
            "capability-downgrade",
        ]
    );
    let raw: Value = serde_yaml::from_str(SUITE).unwrap();
    assert_eq!(
        raw["strategies"],
        json!(["direct", "declarative_plan", "programmatic"])
    );
}
