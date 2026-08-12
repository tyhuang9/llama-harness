use std::sync::Arc;

use llama_harness::{
    mock::{final_response, MockModelProvider},
    AgentDefinition, AgentRunner, RunRequest, RunStatus,
};

#[tokio::test]
async fn facade_runs_the_canonical_core_runner() {
    let provider = Arc::new(MockModelProvider::scripted([final_response(
        "facade result",
    )]));
    let runner = AgentRunner::builder(provider).build();

    let result = runner
        .run(RunRequest::new(
            AgentDefinition::new("facade-test", "Facade test", "1", "mock-model"),
            "Return a fixed response",
        ))
        .await
        .expect("the facade must expose the canonical runner");

    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.final_output.as_deref(), Some("facade result"));
}
