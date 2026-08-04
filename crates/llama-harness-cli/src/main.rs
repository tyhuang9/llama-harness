use clap::{Args, Parser, Subcommand};
use llama_harness_core::{load_agent_manifest_path, AgentDefinition, ModelProvider};
use llama_harness_evals::{load_suite_path, EvaluationReport, RegressionCase};
use llama_harness_observability::{ExportedRun, SqliteEventSink, TraceStoreConfig};
use llama_harness_ollama::{OllamaProvider, DEFAULT_OLLAMA_BASE_URL};
use std::{
    fs,
    path::{Path, PathBuf},
};
use thiserror::Error;

#[derive(Parser)]
#[command(
    name = "llama-harness",
    version,
    about = "Local developer tools for llama-harness"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate, summarize, and prepare deterministic evaluation artifacts.
    Eval {
        #[command(subcommand)]
        command: EvalCommand,
    },
    /// Inspect a saved regression artifact. Executing replay requires an embedding application's adapter.
    Replay(ReplayArgs),
    /// Inspect redacted persisted traces.
    Inspect {
        #[command(subcommand)]
        command: InspectCommand,
    },
    /// Inspect the configured local Ollama instance.
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
    /// Inspect project-owned agent definition manifests.
    Agents {
        #[command(subcommand)]
        command: AgentsCommand,
    },
}

#[derive(Subcommand)]
enum EvalCommand {
    /// Parse and validate a YAML or JSON Harness evaluation suite.
    Validate(SuiteArgs),
    /// Validate a suite, then explain how to run it through an application-owned executor.
    Run(EvalRunArgs),
    /// Print a normalized evaluation report written by an embedding application.
    Results(ReportArgs),
}

#[derive(Args)]
struct SuiteArgs {
    suite: PathBuf,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct EvalRunArgs {
    suite: PathBuf,
    /// Override one or more suite models. Execution still requires an application adapter.
    #[arg(long = "model")]
    models: Vec<String>,
    /// Override repetitions. Must be greater than zero.
    #[arg(long)]
    repeat: Option<u32>,
}

#[derive(Args)]
struct ReportArgs {
    report: PathBuf,
    /// Emit the stored normalized report as JSON.
    #[arg(long)]
    json: bool,
    /// Include only failed cases.
    #[arg(long)]
    failed: bool,
    /// Include only one case ID.
    #[arg(long)]
    case: Option<String>,
}

#[derive(Args)]
struct ReplayArgs {
    regression: PathBuf,
    /// Print the explicit replay request as JSON without attempting execution.
    #[arg(long)]
    json: bool,
}

#[derive(Subcommand)]
enum InspectCommand {
    /// Read one persisted run from a local trace database.
    Run(InspectRunArgs),
}

#[derive(Args)]
struct InspectRunArgs {
    run_id: String,
    /// Project-local SQLite trace database.
    #[arg(long)]
    db: PathBuf,
    /// Export the full redacted run JSON instead of a summary.
    #[arg(long)]
    export_json: bool,
}

#[derive(Subcommand)]
enum ModelsCommand {
    /// Check whether direct local Ollama is reachable.
    Health(OllamaArgs),
    /// List locally installed direct Ollama models.
    List(OllamaArgs),
}

#[derive(Subcommand)]
enum AgentsCommand {
    /// List validated agent definitions in a YAML or JSON manifest.
    List(AgentManifestArgs),
    /// Inspect one validated agent definition by its stable ID.
    Inspect(AgentInspectArgs),
    /// Validate a project-owned agent manifest without executing an agent.
    Validate(AgentManifestArgs),
}

#[derive(Args)]
struct AgentManifestArgs {
    /// Explicit project-owned YAML or JSON manifest path.
    manifest: PathBuf,
    /// Emit machine-readable JSON where applicable.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct AgentInspectArgs {
    /// Explicit project-owned YAML or JSON manifest path.
    manifest: PathBuf,
    /// Stable agent ID.
    agent_id: String,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct OllamaArgs {
    /// Loopback Ollama base URL.
    #[arg(long, default_value = DEFAULT_OLLAMA_BASE_URL)]
    ollama_url: String,
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Eval(#[from] llama_harness_evals::EvalError),
    #[error(transparent)]
    Trace(#[from] llama_harness_observability::TraceStoreError),
    #[error(transparent)]
    Provider(#[from] llama_harness_core::HarnessError),
    #[error(transparent)]
    AgentManifest(#[from] llama_harness_core::AgentManifestError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("application executor required: {0}")]
    ApplicationExecutorRequired(String),
    #[error("not found: {0}")]
    NotFound(String),
}

#[tokio::main]
async fn main() {
    let result = run(Cli::parse()).await;
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<(), CliError> {
    match cli.command {
        Command::Eval { command } => run_eval(command),
        Command::Replay(arguments) => run_replay(arguments),
        Command::Inspect { command } => run_inspect(command),
        Command::Models { command } => run_models(command).await,
        Command::Agents { command } => run_agents(command),
    }
}

fn run_agents(command: AgentsCommand) -> Result<(), CliError> {
    match command {
        AgentsCommand::Validate(arguments) => {
            let manifest = load_agent_manifest_path(arguments.manifest)?;
            if arguments.json {
                println!("{}", serde_json::to_string_pretty(&manifest)?);
            } else {
                println!(
                    "valid agent manifest v{}: {} agent(s)",
                    manifest.version,
                    manifest.agents.len()
                );
            }
            Ok(())
        }
        AgentsCommand::List(arguments) => {
            let manifest = load_agent_manifest_path(arguments.manifest)?;
            if arguments.json {
                println!("{}", serde_json::to_string_pretty(&manifest.agents)?);
            } else if manifest.agents.is_empty() {
                println!("no agents in the validated manifest");
            } else {
                for agent in manifest.agents {
                    println!(
                        "{}\t{}\tv{}\tmodel={}\ttools={}",
                        agent.id,
                        agent.name,
                        agent.version,
                        agent.default_model,
                        agent.tool_allowlist.join(",")
                    );
                }
            }
            Ok(())
        }
        AgentsCommand::Inspect(arguments) => {
            let manifest = load_agent_manifest_path(arguments.manifest)?;
            let agent = manifest
                .agents
                .into_iter()
                .find(|agent| agent.id == arguments.agent_id)
                .ok_or_else(|| CliError::NotFound(format!("agent {}", arguments.agent_id)))?;
            print_agent(&agent, arguments.json)
        }
    }
}

fn print_agent(agent: &AgentDefinition, as_json: bool) -> Result<(), CliError> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(agent)?);
    } else {
        println!(
            "agent {} ({})\nversion: {}\ndefault model: {}\nallowed tools: {}\nmodel calls: {}\ntool calls: {}",
            agent.id,
            agent.name,
            agent.version,
            agent.default_model,
            if agent.tool_allowlist.is_empty() {
                "none".to_owned()
            } else {
                agent.tool_allowlist.join(", ")
            },
            agent.limits.max_model_calls,
            agent.limits.max_tool_calls,
        );
    }
    Ok(())
}

fn run_eval(command: EvalCommand) -> Result<(), CliError> {
    match command {
        EvalCommand::Validate(arguments) => {
            let suite = load_suite_path(arguments.suite)?;
            if arguments.json {
                println!("{}", serde_json::to_string_pretty(&suite)?);
            } else {
                println!(
                    "valid suite {} (v{}): {} cases, {} model(s)",
                    suite.id,
                    suite.version,
                    suite.cases.len(),
                    suite.models.len()
                );
            }
            Ok(())
        }
        EvalCommand::Run(arguments) => {
            let suite = load_suite_path(arguments.suite)?;
            if arguments.repeat == Some(0) {
                return Err(CliError::ApplicationExecutorRequired(
                    "--repeat must be greater than zero".into(),
                ));
            }
            let models = if arguments.models.is_empty() {
                suite.models.join(", ")
            } else {
                arguments.models.join(", ")
            };
            Err(CliError::ApplicationExecutorRequired(format!(
                "suite {} was valid for model(s) {models}, but the standalone CLI cannot construct application-owned tools, fixtures, policies, or approvals. Run llama_harness_evals::evaluate_suite from the embedding application (or the example adapter when available).",
                suite.id
            )))
        }
        EvalCommand::Results(arguments) => {
            let report = filter_report(
                read_report(&arguments.report)?,
                arguments.failed,
                arguments.case.as_deref(),
            );
            if arguments.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("{}", format_report(&report));
            }
            Ok(())
        }
    }
}

fn run_replay(arguments: ReplayArgs) -> Result<(), CliError> {
    let input = fs::read_to_string(&arguments.regression)?;
    let regression: RegressionCase = match arguments
        .regression
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("yaml") | Some("yml") => {
            serde_yaml::from_str(&input).map_err(llama_harness_evals::EvalError::from)?
        }
        _ => serde_json::from_str(&input)?,
    };
    regression.validate()?;
    if arguments.json {
        println!("{}", serde_json::to_string_pretty(&regression)?);
        return Ok(());
    }
    Err(CliError::ApplicationExecutorRequired(format!(
        "replay case {} is valid input for agent {}. The standalone CLI will not infer application tools or fixture loading from a trace. Invoke llama_harness_evals::replay_regression from the embedding application.",
        regression.source_case_id, regression.agent_id
    )))
}

fn run_inspect(command: InspectCommand) -> Result<(), CliError> {
    match command {
        InspectCommand::Run(arguments) => {
            let store = SqliteEventSink::open(arguments.db, TraceStoreConfig::default())?;
            let Some(export) = store.export_run(&arguments.run_id)? else {
                return Err(CliError::NotFound(format!(
                    "run {} in the selected trace database",
                    arguments.run_id
                )));
            };
            println!("{}", render_inspected_run(&export, arguments.export_json)?);
            Ok(())
        }
    }
}

async fn run_models(command: ModelsCommand) -> Result<(), CliError> {
    match command {
        ModelsCommand::Health(arguments) => {
            let provider = OllamaProvider::builder()
                .base_url(arguments.ollama_url)
                .build()?;
            let health = provider.health().await?;
            println!(
                "Ollama {}{}",
                if health.healthy {
                    "healthy"
                } else {
                    "unavailable"
                },
                health
                    .detail
                    .as_deref()
                    .map(|detail| format!(": {detail}"))
                    .unwrap_or_default()
            );
            Ok(())
        }
        ModelsCommand::List(arguments) => {
            let provider = OllamaProvider::builder()
                .base_url(arguments.ollama_url)
                .build()?;
            for model in provider.list_models().await? {
                println!(
                    "{}\ttools={} streaming={} structured_output={}",
                    model.id,
                    model.capabilities.supports_tools,
                    model.capabilities.supports_streaming,
                    model.capabilities.supports_structured_output
                );
            }
            Ok(())
        }
    }
}

fn read_report(path: &Path) -> Result<EvaluationReport, CliError> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn format_report(report: &EvaluationReport) -> String {
    let mut lines = vec![format!(
        "evaluation {}: {}/{} passed",
        report.suite_id,
        report.passed_count(),
        report.results.len()
    )];
    for result in &report.results {
        let outcome = if result.passed { "PASS" } else { "FAIL" };
        lines.push(format!(
            "{outcome} {} [{}] model={} trace={}",
            result.case_id,
            result.repetition,
            result.model,
            result.trace_id.as_deref().unwrap_or("none")
        ));
        for failure in &result.failures {
            lines.push(format!("  {}: {}", failure.rule, failure.message));
        }
    }
    lines.join("\n")
}

fn filter_report(
    mut report: EvaluationReport,
    failed_only: bool,
    case_id: Option<&str>,
) -> EvaluationReport {
    report.results.retain(|result| {
        (!failed_only || !result.passed)
            && case_id.map_or(true, |case_id| result.case_id == case_id)
    });
    report
}

fn render_inspected_run(export: &ExportedRun, as_json: bool) -> Result<String, CliError> {
    if as_json {
        return Ok(serde_json::to_string_pretty(export)?);
    }
    let mut lines = vec![format!(
        "run {} trace {}: {} event(s)",
        export.run_id,
        export.trace_id,
        export.events.len()
    )];
    for event in &export.events {
        lines.push(format!(
            "  #{} {}",
            event.record.sequence,
            serde_json::to_string(&event.record.event)?
        ));
    }
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_harness_core::{EventRecord, RunEvent, RunStatus};
    use llama_harness_evals::{AssertionFailure, EvaluationCaseResult};
    use serde_json::json;

    fn report() -> EvaluationReport {
        EvaluationReport {
            format_version: 1,
            id: "report-1".into(),
            suite_id: "suite".into(),
            suite_version: 1,
            results: vec![
                EvaluationCaseResult {
                    suite_id: "suite".into(),
                    case_id: "passing".into(),
                    model: "ollama:model".into(),
                    repetition: 1,
                    passed: true,
                    failures: vec![],
                    run_id: Some("run-pass".into()),
                    trace_id: Some("trace-pass".into()),
                    status: Some(RunStatus::Completed),
                    duration_ms: Some(10),
                    model_calls: Some(1),
                    tool_calls: Some(0),
                    agent_version: Some("1".into()),
                    prompt_version: Some("1".into()),
                    final_state: None,
                    unresolved_items: None,
                },
                EvaluationCaseResult {
                    suite_id: "suite".into(),
                    case_id: "failing".into(),
                    model: "ollama:model".into(),
                    repetition: 1,
                    passed: false,
                    failures: vec![AssertionFailure {
                        rule: "status".into(),
                        message: "expected completed".into(),
                    }],
                    run_id: Some("run-fail".into()),
                    trace_id: Some("trace-fail".into()),
                    status: Some(RunStatus::Failed),
                    duration_ms: Some(20),
                    model_calls: Some(2),
                    tool_calls: Some(1),
                    agent_version: Some("1".into()),
                    prompt_version: Some("1".into()),
                    final_state: None,
                    unresolved_items: None,
                },
            ],
        }
    }

    #[test]
    fn parser_accepts_the_supported_local_commands() {
        assert!(Cli::try_parse_from(["llama-harness", "eval", "validate", "suite.yaml"]).is_ok());
        assert!(Cli::try_parse_from([
            "llama-harness",
            "inspect",
            "run",
            "run-1",
            "--db",
            "traces.sqlite",
            "--export-json",
        ])
        .is_ok());
        assert!(Cli::try_parse_from(["llama-harness", "models", "list"]).is_ok());
        assert!(Cli::try_parse_from([
            "llama-harness",
            "agents",
            "inspect",
            "agents.yaml",
            "task-agent",
            "--json",
        ])
        .is_ok());
    }

    #[test]
    fn report_filters_and_human_output_are_stable() {
        let filtered = filter_report(report(), true, Some("failing"));
        assert_eq!(filtered.results.len(), 1);
        let output = format_report(&filtered);
        assert_eq!(
            output,
            "evaluation suite: 0/1 passed\nFAIL failing [1] model=ollama:model trace=trace-fail\n  status: expected completed"
        );
        let json = serde_json::to_string(&filtered).unwrap();
        assert!(json.contains("failing"));
        assert!(!json.contains("passing"));
    }

    #[test]
    fn inspect_export_never_recovers_raw_disabled_payloads() {
        let store = SqliteEventSink::open_in_memory(TraceStoreConfig::default()).unwrap();
        store
            .append_with_raw(
                &EventRecord {
                    run_id: "run-1".into(),
                    trace_id: "trace-1".into(),
                    sequence: 1,
                    timestamp_ms: 1,
                    event: RunEvent::ModelRequested {
                        call_number: 1,
                        model: "ollama:model".into(),
                    },
                },
                Some(&json!({"authorization": "private credential"})),
            )
            .unwrap();
        let export = store.export_run("run-1").unwrap().unwrap();
        let json = render_inspected_run(&export, true).unwrap();
        let human = render_inspected_run(&export, false).unwrap();
        assert!(!json.contains("private credential"));
        assert!(!human.contains("private credential"));
        assert!(human.contains("run run-1 trace trace-1: 1 event(s)"));
    }

    #[test]
    fn missing_report_is_an_error() {
        let missing = std::env::temp_dir().join("llama-harness-cli-missing-report.json");
        let error = read_report(&missing).unwrap_err();
        assert!(matches!(error, CliError::Io(_)));
    }

    #[test]
    fn agent_rendering_keeps_manifest_metadata_visible() {
        let agent = AgentDefinition {
            id: "task-agent".into(),
            name: "Task Agent".into(),
            version: "2".into(),
            system_instructions: "No secrets".into(),
            default_model: "ollama:qwen3".into(),
            tool_allowlist: vec!["list_tasks".into()],
            limits: Default::default(),
            generation: Default::default(),
            output_schema: None,
            metadata: Default::default(),
        };
        let json = serde_json::to_string(&agent).unwrap();
        assert!(json.contains("task-agent"));
        assert!(json.contains("list_tasks"));
    }
}
