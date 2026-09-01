use std::sync::Arc;

use llama_harness::{
    mock::{final_response, MockModelProvider},
    AgentDefinition, AgentRunner, RunRequest, RunStatus, SpeculationConfig, SpeculationMode,
    MIN_SPECULATION_SHADOW_OBSERVATIONS,
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

#[test]
fn facade_exports_conservative_speculation_contract() {
    let config = SpeculationConfig::default();
    assert_eq!(
        config.required_shadow_observations,
        MIN_SPECULATION_SHADOW_OBSERVATIONS
    );
    let runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([]))).build();
    assert_eq!(
        runner.speculation_readiness("unregistered").mode,
        SpeculationMode::Disabled
    );
    assert_eq!(runner.speculation_metrics("unregistered").issued, 0);
}
