use llama_harness_evals::{
    evaluate_suite, export_regression_case, load_suite_path, replay_regression, RegressionSource,
};
use local_task_agent::TaskAgentEvalExecutor;
use std::path::PathBuf;

fn suite_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../evals/local-task-agent/suite.yaml")
}

#[tokio::test]
async fn example_suite_runs_in_isolated_mock_state_and_links_traces() {
    let suite = load_suite_path(suite_path()).unwrap();
    let executor = TaskAgentEvalExecutor;
    let report = evaluate_suite(&suite, &executor, &[], None).await.unwrap();
    assert_eq!(report.results.len(), 8);
    assert_eq!(report.failed_count(), 0, "{:#?}", report.results);
    assert!(report
        .results
        .iter()
        .all(|result| result.trace_id.as_deref().is_some_and(|id| !id.is_empty())));
}

#[tokio::test]
async fn saved_example_case_replays_without_reading_a_trace_payload() {
    let suite = load_suite_path(suite_path()).unwrap();
    let executor = TaskAgentEvalExecutor;
    let report = evaluate_suite(&suite, &executor, &[], None).await.unwrap();
    let case = &suite.cases[0];
    let regression = export_regression_case(
        RegressionSource {
            suite_id: suite.id.clone(),
            agent_id: suite.agent.clone(),
            agent_version: suite.agent_version.clone(),
            prompt_version: suite.prompt_version.clone(),
            prompt_override: suite.prompt_override.clone(),
            model: "ollama:mock".into(),
        },
        case,
        &report.results[0],
    )
    .unwrap();
    let replayed = replay_regression(&regression, &executor).await.unwrap();
    assert!(replayed.passed, "{:#?}", replayed.failures);
    let serialized = serde_json::to_string(&regression).unwrap();
    assert!(!serialized.contains("raw_payload"));
}
