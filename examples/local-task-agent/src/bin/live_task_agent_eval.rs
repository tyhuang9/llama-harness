use clap::{Parser, ValueEnum};
use llama_harness::{
    evals::load_suite_path,
    ollama::{OllamaProvider, DEFAULT_OLLAMA_BASE_URL},
    GenerationOptions, ModelProvider, RunStrategy,
};
use local_task_agent::live_eval::{
    evaluate_live_suite, LiveEvalConfig, LiveEvalExecutor, LiveEvalLimits,
};
use serde::Serialize;
use serde_json::Value;
use std::{path::PathBuf, process::Command, sync::Arc};

#[derive(Parser)]
#[command(about = "Opt-in live local-model task-agent evaluation")]
struct Arguments {
    /// One or more installed Ollama model names. Models are never downloaded.
    #[arg(long, required = true, num_args = 1..)]
    model: Vec<String>,
    /// Loopback-only Ollama base URL, validated by the existing provider builder.
    #[arg(long, default_value = DEFAULT_OLLAMA_BASE_URL)]
    ollama_url: String,
    /// Live suite file. It is intentionally separate from normal deterministic regression suites.
    #[arg(long, default_value_os_t = default_suite_path())]
    suite: PathBuf,
    /// Evaluate only these case IDs.
    #[arg(long)]
    case: Vec<String>,
    /// Requested strategies. The default runs Direct first, then Adaptive.
    #[arg(long, value_enum, num_args = 1.., default_values_t = [CliStrategy::Direct, CliStrategy::Adaptive])]
    strategy: Vec<CliStrategy>,
    /// Repetitions per case, model, and strategy.
    #[arg(long, default_value_t = 3)]
    repeat: u32,
    /// Supported sampling temperature.
    #[arg(long)]
    temperature: Option<f32>,
    /// Supported nucleus-sampling probability.
    #[arg(long)]
    top_p: Option<f32>,
    /// Supported maximum output-token budget.
    #[arg(long)]
    output_tokens: Option<u32>,
    /// Maximum model calls per sample.
    #[arg(long, default_value_t = 4)]
    max_model_calls: u32,
    /// Maximum tool calls per sample.
    #[arg(long, default_value_t = 3)]
    max_tool_calls: u32,
    /// Maximum complete run duration in milliseconds.
    #[arg(long, default_value_t = 300_000)]
    max_run_duration_ms: u64,
    /// Maximum duration of one model request in milliseconds.
    #[arg(long, default_value_t = 90_000)]
    max_model_call_duration_ms: u64,
    /// JSON collected by a local environment probe, retained verbatim as sidecar metadata.
    #[arg(long)]
    environment_json: Option<PathBuf>,
    /// Output path for the normalized report and safe per-sample evidence.
    #[arg(long, default_value = "live-task-agent-eval-report.json")]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ValueEnum)]
enum CliStrategy {
    Direct,
    Adaptive,
    DeclarativePlan,
    Programmatic,
}

impl From<CliStrategy> for RunStrategy {
    fn from(value: CliStrategy) -> Self {
        match value {
            CliStrategy::Direct => Self::Direct,
            CliStrategy::Adaptive => Self::Adaptive,
            CliStrategy::DeclarativePlan => Self::DeclarativePlan,
            CliStrategy::Programmatic => Self::Programmatic,
        }
    }
}

#[derive(Serialize)]
struct OutputArtifact {
    format_version: u32,
    invocation: InvocationMetadata,
    evaluation: local_task_agent::live_eval::LiveEvaluationArtifact,
}

#[derive(Serialize)]
struct InvocationMetadata {
    source_commit: String,
    source_dirty: bool,
    ollama_base_url: String,
    ollama_health: llama_harness::ProviderHealth,
    selected_models: Vec<String>,
    available_model_ids: Vec<String>,
    generation: GenerationOptions,
    limits: LiveEvalLimits,
    environment: Option<Value>,
}

#[tokio::main]
async fn main() {
    if let Err(error) = run(Arguments::parse()).await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn run(arguments: Arguments) -> Result<(), String> {
    let source_revision = source_revision()?;
    if arguments.repeat == 0 {
        return Err("--repeat must be greater than zero".into());
    }
    if arguments.max_model_calls == 0 || arguments.max_tool_calls == 0 {
        return Err("model and tool call limits must be greater than zero".into());
    }
    if arguments.max_run_duration_ms == 0 || arguments.max_model_call_duration_ms == 0 {
        return Err("run duration limits must be greater than zero".into());
    }
    reject_duplicates(&arguments.model, "--model")?;
    let requested_strategies: Vec<_> = arguments
        .strategy
        .iter()
        .map(|strategy| format!("{strategy:?}"))
        .collect();
    reject_duplicates(&requested_strategies, "--strategy")?;
    reject_duplicates(&arguments.case, "--case")?;
    let provider = OllamaProvider::builder()
        .base_url(arguments.ollama_url.clone())
        .build()
        .map_err(|error| format!("invalid Ollama configuration: {error}"))?;
    let health = provider
        .health()
        .await
        .map_err(|error| format!("Ollama health check failed: {error}"))?;
    if !health.healthy {
        return Err(format!(
            "Ollama is unreachable at {}: {}",
            provider.base_url(),
            health.detail.as_deref().unwrap_or("no detail returned")
        ));
    }
    let installed = provider
        .list_models()
        .await
        .map_err(|error| format!("could not list installed Ollama models: {error}"))?;
    let available_model_ids: Vec<_> = installed.iter().map(|model| model.id.clone()).collect();
    let missing: Vec<_> = arguments
        .model
        .iter()
        .filter(|model| !available_model_ids.contains(model))
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "requested Ollama model(s) are not installed: {}; installed: {}",
            missing.join(", "),
            available_model_ids.join(", ")
        ));
    }
    let mut suite = load_suite_path(&arguments.suite).map_err(|error| {
        format!(
            "could not load live suite {}: {error}",
            arguments.suite.display()
        )
    })?;
    filter_requested_cases(&mut suite, &arguments.case)?;
    suite.strategies = arguments.strategy.iter().copied().map(Into::into).collect();
    suite.models = arguments.model.clone();
    suite
        .validate()
        .map_err(|error| format!("invalid selected live suite: {error}"))?;
    let generation = GenerationOptions {
        temperature: arguments.temperature,
        top_p: arguments.top_p,
        max_output_tokens: arguments.output_tokens,
    };
    let limits = LiveEvalLimits {
        max_model_calls: arguments.max_model_calls,
        max_tool_calls: arguments.max_tool_calls,
        max_run_duration_ms: arguments.max_run_duration_ms,
        max_model_call_duration_ms: arguments.max_model_call_duration_ms,
    };
    let environment = match arguments.environment_json.as_ref() {
        Some(path) => Some(read_json(path)?),
        None => None,
    };
    let mut config = LiveEvalConfig::new(Arc::new(provider));
    config.generation = generation.clone();
    config.limits = limits.clone();
    config.progress = true;
    let executor = LiveEvalExecutor::new(config);
    let artifact = evaluate_live_suite(&suite, &executor, &arguments.model, Some(arguments.repeat))
        .await
        .map_err(|error| format!("live evaluation could not execute: {error}"))?;
    let output = OutputArtifact {
        format_version: 1,
        invocation: InvocationMetadata {
            source_commit: source_revision.commit,
            source_dirty: source_revision.dirty,
            ollama_base_url: arguments.ollama_url,
            ollama_health: health,
            selected_models: arguments.model,
            available_model_ids,
            generation,
            limits,
            environment,
        },
        evaluation: artifact,
    };
    let serialized = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("could not serialize evaluation artifact: {error}"))?;
    std::fs::write(&arguments.output, serialized)
        .map_err(|error| format!("could not write {}: {error}", arguments.output.display()))?;
    let failed = output.evaluation.report.failed_count();
    let total = output.evaluation.report.results.len();
    println!(
        "saved {} sample result(s) to {}; {} failed",
        total,
        arguments.output.display(),
        failed
    );
    if failed > 0 {
        return Err(format!("{failed} live evaluation assertion(s) failed"));
    }
    Ok(())
}

fn default_suite_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../evals/local-task-agent/live-suite.yaml")
}

fn read_json(path: &PathBuf) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "could not read environment metadata {}: {error}",
            path.display()
        )
    })?;
    serde_json::from_str(&text).map_err(|error| {
        format!(
            "environment metadata {} was not valid JSON: {error}",
            path.display()
        )
    })
}

struct SourceRevision {
    commit: String,
    dirty: bool,
}

fn source_revision() -> Result<SourceRevision, String> {
    let source_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = Command::new("git")
        .current_dir(&source_root)
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|error| format!("could not capture source commit: {error}"))?;
    if !output.status.success() {
        return Err("could not capture source commit from repository".into());
    }
    let commit = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if commit.is_empty() {
        return Err("source repository returned an empty commit ID".into());
    }
    let status = Command::new("git")
        .current_dir(source_root)
        .args(["status", "--porcelain"])
        .output()
        .map_err(|error| format!("could not capture source dirty state: {error}"))?;
    if !status.status.success() {
        return Err("could not capture source dirty state from repository".into());
    }
    Ok(SourceRevision {
        commit,
        dirty: !status.stdout.is_empty(),
    })
}

fn reject_duplicates(values: &[String], flag: &str) -> Result<(), String> {
    let mut unique = std::collections::BTreeSet::new();
    let duplicates: Vec<_> = values
        .iter()
        .filter(|value| !unique.insert(value.as_str()))
        .cloned()
        .collect();
    if duplicates.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{flag} contains duplicate value(s): {}",
            duplicates.join(", ")
        ))
    }
}

fn filter_requested_cases(
    suite: &mut llama_harness::evals::EvalSuite,
    requested: &[String],
) -> Result<(), String> {
    if requested.is_empty() {
        return Ok(());
    }
    let unknown: Vec<_> = requested
        .iter()
        .filter(|requested| !suite.cases.iter().any(|case| &case.id == *requested))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        return Err(format!("unknown live case ID(s): {}", unknown.join(", ")));
    }
    suite.cases.retain(|case| requested.contains(&case.id));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_cli_cohorts_are_rejected() {
        assert!(reject_duplicates(&["model".into(), "model".into()], "--model").is_err());
        assert!(reject_duplicates(&["Direct".into(), "Direct".into()], "--strategy").is_err());
        assert!(reject_duplicates(&["no-tool".into(), "no-tool".into()], "--case").is_err());
    }

    #[test]
    fn case_filter_rejects_partial_unknown_selection_without_dropping_known_cases() {
        let mut suite = llama_harness::evals::load_suite(
            include_str!("../../../../evals/local-task-agent/live-suite.yaml"),
            Some("yaml"),
        )
        .unwrap();
        let original_count = suite.cases.len();
        let error = filter_requested_cases(&mut suite, &["no-tool".into(), "mistyped-case".into()])
            .unwrap_err();
        assert!(error.contains("mistyped-case"));
        assert_eq!(suite.cases.len(), original_count);
    }
}
