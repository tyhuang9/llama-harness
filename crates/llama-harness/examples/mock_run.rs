use std::sync::Arc;

use llama_harness::{
    mock::{final_response, MockModelProvider},
    AgentDefinition, AgentRunner, RunRequest,
};

#[tokio::main]
async fn main() -> Result<(), llama_harness::HarnessError> {
    let provider = Arc::new(MockModelProvider::scripted([final_response(
        "Hello from llama-harness",
    )]));
    let runner = AgentRunner::builder(provider).build();
    let result = runner
        .run(RunRequest::new(
            AgentDefinition::new("example", "Embedded example", "1", "mock-model"),
            "Say hello",
        ))
        .await?;

    println!("{}", result.final_output.unwrap_or_default());
    Ok(())
}
