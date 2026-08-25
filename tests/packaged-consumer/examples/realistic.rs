use llama_harness::{AgentDefinition, RunRequest};
use llama_harness_packaged_consumer::{cancelled_token, configured_runner};

#[tokio::main]
async fn main() -> Result<(), llama_harness::HarnessError> {
    let (runner, events) = configured_runner()?;
    let agent = AgentDefinition::new("consumer", "Packaged consumer", "1", "consumer-model");
    let result = runner
        .run(RunRequest::new(agent, "Verify the packaged facade"))
        .await?;

    assert_eq!(result.final_output.as_deref(), Some("consumer response"));
    assert!(!events.is_empty());
    assert!(cancelled_token().is_cancelled());
    Ok(())
}
