use clap::{Parser, ValueEnum};
use llama_harness::{
    evals::{load_suite_path, EvalError},
    observability::{SqliteEventSink, TraceStoreConfig, TraceStoreError},
    ollama::{OllamaProvider, DEFAULT_OLLAMA_BASE_URL},
    EventSink, HarnessError, ModelProvider, RunEvent, RunResult,
};
use local_task_agent::{
    build_runtime, default_tasks, scripted_provider, MockScenario, Task, TaskStore, TaskStoreError,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{io::Read, path::PathBuf, sync::Arc};
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
    Harness(#[from] HarnessError),
    #[error(transparent)]
    Store(#[from] TaskStoreError),
    #[error(transparent)]
    Trace(#[from] TraceStoreError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Eval(#[from] EvalError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("invalid Promptfoo adapter request: {0}")]
    InvalidPromptfooRequest(String),
    #[error("--model is required with --provider ollama")]
    MissingOllamaModel,
    #[error("requested Ollama model is not installed: {0}")]
    OllamaModelMissing(String),
}

#[tokio::main]
async fn main() {
    let result = if std::env::args().nth(1).as_deref() == Some("promptfoo-adapter") {
        run_promptfoo_adapter().await
    } else {
        run(Arguments::parse()).await
    };
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

/// The only process protocol exposed for the checked-in Promptfoo provider.
/// It deliberately accepts a validated suite case, not arbitrary tool definitions
/// or shell commands, and starts a fresh in-process `AgentRunner` for every call.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptfooAdapterRequest {
    suite_path: PathBuf,
    case_id: String,
    input: String,
    model: String,
    ollama_url: String,
    trace_db: PathBuf,
}

#[derive(Serialize)]
struct PromptfooAdapterResponse {
    output: String,
    run: RunResult,
    model_calls: u32,
    final_state: Value,
    unresolved_items: Option<Value>,
    agent_version: String,
    prompt_version: String,
}

async fn run_promptfoo_adapter() -> Result<(), ExampleError> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let request: PromptfooAdapterRequest = serde_json::from_str(&input)?;
    let model = request
        .model
        .strip_prefix("ollama:")
        .filter(|model| !model.trim().is_empty())
        .ok_or_else(|| {
            ExampleError::InvalidPromptfooRequest(
                "model must be an ollama:<installed-model> ID".into(),
            )
        })?
        .to_owned();
    let suite = load_suite_path(&request.suite_path)?;
    let case = suite
        .cases
        .into_iter()
        .find(|candidate| candidate.id == request.case_id)
        .ok_or_else(|| {
            ExampleError::InvalidPromptfooRequest(format!("unknown suite case {}", request.case_id))
        })?;
    if case.input != request.input {
        return Err(ExampleError::InvalidPromptfooRequest(
            "prompt input must exactly match the selected suite case".into(),
        ));
    }

    let provider = OllamaProvider::builder()
        .base_url(request.ollama_url)
        .build()?;
    let health = provider.health().await?;
    if !health.healthy {
        return Err(ExampleError::Harness(HarnessError::Provider(
            health
                .detail
                .unwrap_or_else(|| "Ollama is unavailable".into()),
        )));
    }
    if !provider
        .list_models()
        .await?
        .iter()
        .any(|candidate| candidate.id == model)
    {
        return Err(ExampleError::OllamaModelMissing(model));
    }
    let tasks = case
        .fixture
        .as_ref()
        .and_then(|fixture| fixture.data.get("tasks"))
        .map(|tasks| serde_json::from_value::<Vec<Task>>(tasks.clone()))
        .transpose()?
        .unwrap_or_else(default_tasks);
    let store = Arc::new(TaskStore::new(tasks)?);
    let trace = Arc::new(SqliteEventSink::open(
        request.trace_db,
        TraceStoreConfig::default(),
    )?);
    let grant_approval = matches!(case.id.as_str(), "create-new" | "explicit-completion");
    let mut runtime = build_runtime(
        Arc::new(provider),
        Arc::clone(&store),
        model.clone(),
        grant_approval,
        Arc::clone(&trace) as Arc<dyn EventSink>,
    )?;
    if case.id == "limit-stop" {
        runtime.agent.limits.max_model_calls = 1;
    }
    let run = runtime.run(case.input, Some(model)).await?;
    let model_calls = trace
        .export_run(&run.id)?
        .map(|export| {
            export
                .events
                .iter()
                .filter(|event| matches!(event.record.event, RunEvent::ModelRequested { .. }))
                .count() as u32
        })
        .unwrap_or_default();
    let output = run.final_output.clone().ok_or_else(|| {
        ExampleError::InvalidPromptfooRequest("agent run did not produce final output".into())
    })?;
    println!(
        "{}",
        serde_json::to_string(&PromptfooAdapterResponse {
            output,
            run,
            model_calls,
            final_state: serde_json::json!({"tasks": store.snapshot()?}),
            unresolved_items: (case.id == "ambiguous")
                .then(|| serde_json::json!(["task action is ambiguous"])),
            agent_version: runtime.agent.version,
            prompt_version: "local-task-agent-prompt-1".into(),
        })?
    );
    Ok(())
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
                return Err(ExampleError::Harness(HarnessError::Provider(
                    health
                        .detail
                        .unwrap_or_else(|| "Ollama is unavailable".into()),
                )));
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
        Arc::clone(&trace) as Arc<dyn EventSink>,
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
