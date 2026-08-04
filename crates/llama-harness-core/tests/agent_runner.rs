use llama_harness_core::{
    mock::{final_response, MockModelProvider},
    AgentDefinition, AgentLimits, AgentRunner, GenerationOptions, JsonMap, RunOverrides,
    RunRequest, RunStatus,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

fn request() -> RunRequest {
    RunRequest {
        agent: AgentDefinition {
            id: "external-test".into(),
            name: "External test".into(),
            version: "1".into(),
            system_instructions: String::new(),
            default_model: "mock-model".into(),
            tool_allowlist: vec![],
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

#[tokio::test]
async fn public_api_runs_a_scripted_final_response() {
    let runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
        "ok",
    )])))
    .build();
    let result = runner.run(request()).await.unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(result.final_output.as_deref(), Some("ok"));
}
