use clap::{Parser, ValueEnum};
use llama_harness_core::ModelProvider;
use llama_harness_observability::{SqliteEventSink, TraceStoreConfig, TraceStoreError};
use llama_harness_ollama::{OllamaProvider, DEFAULT_OLLAMA_BASE_URL};
use local_task_agent::{
    build_runtime, default_tasks, scripted_provider, MockScenario, TaskStore, TaskStoreError,
};
use std::{path::PathBuf, sync::Arc};
use thiserror::Error;

#[derive(Parser)]
#[command(about = "Embedded local-task-agent reference application")]
struct Arguments {
    /// Use a deterministic scripted mock (default) or an already-installed local Ollama model.
    #[arg(long, value_enum, default_value_t = Provider::Mock)]
    provider: Provider,
    /// Required for --provider ollama; no model is downloaded automatically.
    #[arg(long)]
    model: Option<String>,
    /// Loopback-only Ollama base URL.
    #[arg(long, default_value = DEFAULT_OLLAMA_BASE_URL)]
    ollama_url: String,
    /// SQLite path used for the redacted local run trace.
    #[arg(long, default_value = "local-task-agent-traces.sqlite")]
    trace_db: PathBuf,
    /// Demonstrate denial of state-changing task tools.
    #[arg(long)]
    deny_approval: bool,
    /// User request supplied to the embedded AgentRunner.
    #[arg(long, default_value = "Mark the evening medication task complete.")]
    input: String,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Provider {
    Mock,
    Ollama,
}

#[derive(Debug, Error)]
enum ExampleError {
    #[error(transparent)]
    Harness(#[from] llama_harness_core::HarnessError),
    #[error(transparent)]
    Store(#[from] TaskStoreError),
    #[error(transparent)]
    Trace(#[from] TraceStoreError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("--model is required with --provider ollama")]
    MissingOllamaModel,
    #[error("requested Ollama model is not installed: {0}")]
    OllamaModelMissing(String),
}

#[tokio::main]
async fn main() {
    if let Err(error) = run(Arguments::parse()).await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn run(arguments: Arguments) -> Result<(), ExampleError> {
    let (provider, model): (Arc<dyn ModelProvider>, String) = match arguments.provider {
        Provider::Mock => (
            Arc::new(scripted_provider(MockScenario::CompleteExisting)),
            "mock-model".into(),
        ),
        Provider::Ollama => {
            let model = arguments.model.ok_or(ExampleError::MissingOllamaModel)?;
            let provider = OllamaProvider::builder()
                .base_url(arguments.ollama_url)
                .build()?;
            let health = provider.health().await?;
            if !health.healthy {
                return Err(ExampleError::Harness(
                    llama_harness_core::HarnessError::Provider(
                        health
                            .detail
                            .unwrap_or_else(|| "Ollama is unavailable".into()),
                    ),
                ));
            }
            let installed = provider.list_models().await?;
            if !installed.iter().any(|candidate| candidate.id == model) {
                return Err(ExampleError::OllamaModelMissing(model));
            }
            println!("Ollama healthy; selected installed model {model}");
            (Arc::new(provider), model)
        }
    };

    let store = Arc::new(TaskStore::new(default_tasks())?);
    let trace = Arc::new(SqliteEventSink::open(
        arguments.trace_db,
        TraceStoreConfig::default(),
    )?);
    let runtime = build_runtime(
        provider,
        Arc::clone(&store),
        model.clone(),
        !arguments.deny_approval,
        Arc::clone(&trace) as Arc<dyn llama_harness_core::EventSink>,
    )?;
    let result = runtime.run(arguments.input, Some(model)).await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    println!("tasks: {}", serde_json::to_string(&store.snapshot()?)?);
    if let Some(export) = trace.export_run(&result.id)? {
        println!(
            "persisted redacted trace {} with {} event(s)",
            export.trace_id,
            export.events.len()
        );
    }
    Ok(())
}
