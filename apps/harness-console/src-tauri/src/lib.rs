//! Local-only Tauri commands for the Llama Harness developer console.
//!
//! This module deliberately talks to project files, the Harness crates, and a
//! loopback Ollama process directly. It never starts or contacts the retired
//! HTTP daemon, and it never sends raw trace payloads to the webview.

use llama_harness_core::{
    load_agent_manifest_path, AgentDefinition, ModelInfo, ModelProvider, ProviderHealth,
};
use llama_harness_evals::EvaluationReport;
use llama_harness_observability::{RunListQuery, SqliteEventSink};
use llama_harness_ollama::OllamaProvider;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    sync::Mutex,
};
use tauri::{Manager, State};

const MAX_RUNS: u32 = 200;
const MAX_EVENTS_PER_RUN: u32 = 1_000;
const MAX_PROMPTFOO_ARTIFACT_BYTES: u64 = 512 * 1024;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectWorkspace {
    pub project_root: String,
    pub trace_db_path: String,
    pub evaluation_results_path: Option<String>,
    pub agent_manifest_path: Option<String>,
    pub ollama_url: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConsolePreferences {
    pub workspace: Option<ProjectWorkspace>,
    /// This only records a console preference. It cannot enable raw persistence
    /// or restore raw payloads from an existing trace database.
    pub raw_payload_preference: bool,
    pub redaction_key_fragments: Vec<String>,
    pub retention_days: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunQuery {
    pub trace_id: Option<String>,
    pub status: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConsoleRun {
    pub run_id: String,
    pub trace_id: String,
    pub started_at_ms: u64,
    pub updated_at_ms: u64,
    pub event_count: u64,
    pub status: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConsoleEvent {
    pub sequence: u64,
    pub timestamp_ms: u64,
    pub event: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConsoleModels {
    pub health: ProviderHealth,
    pub models: Vec<ModelInfo>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluationArtifact {
    pub path: String,
    pub report: EvaluationReport,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluationArtifacts {
    pub reports: Vec<EvaluationArtifact>,
    pub skipped_files: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptfooArtifact {
    pub kind: String,
    pub path: String,
    pub content: String,
    pub truncated: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalLaunchRequest {
    pub suite_path: String,
    pub models: Vec<String>,
    pub repeat: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayLaunchRequest {
    pub regression_path: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandPreview {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandResult {
    pub command: CommandPreview,
    pub success: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

struct ConsoleState {
    preferences_path: PathBuf,
    preferences: Mutex<ConsolePreferences>,
}

impl ConsoleState {
    fn workspace(&self) -> Result<ProjectWorkspace, String> {
        self.preferences
            .lock()
            .map_err(|_| "console preference lock is unavailable".to_owned())?
            .workspace
            .clone()
            .ok_or_else(|| "Connect a local project workspace first.".to_owned())
    }
}

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let directory = app.path().app_data_dir()?;
            fs::create_dir_all(&directory)?;
            let preferences_path = directory.join("console-preferences.json");
            let preferences = load_preferences(&preferences_path);
            app.manage(ConsoleState {
                preferences_path,
                preferences: Mutex::new(preferences),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_preferences,
            connect_workspace,
            save_preferences,
            list_runs,
            list_run_events,
            list_models,
            list_agents,
            list_evaluation_artifacts,
            list_promptfoo_artifacts,
            preview_eval_command,
            launch_eval_command,
            preview_replay_command,
            launch_replay_command,
        ])
        .run(tauri::generate_context!())
        .expect("error while running llama-harness developer console");
}

#[tauri::command]
fn get_preferences(state: State<'_, ConsoleState>) -> Result<ConsolePreferences, String> {
    state
        .preferences
        .lock()
        .map_err(|_| "console preference lock is unavailable".to_owned())
        .map(|preferences| preferences.clone())
}

#[tauri::command]
fn connect_workspace(
    workspace: ProjectWorkspace,
    state: State<'_, ConsoleState>,
) -> Result<ConsolePreferences, String> {
    let workspace = validate_workspace(workspace)?;
    let mut preferences = state
        .preferences
        .lock()
        .map_err(|_| "console preference lock is unavailable".to_owned())?;
    preferences.workspace = Some(workspace);
    persist_preferences(&state.preferences_path, &preferences)?;
    Ok(preferences.clone())
}

#[tauri::command]
fn save_preferences(
    update: ConsolePreferences,
    state: State<'_, ConsoleState>,
) -> Result<ConsolePreferences, String> {
    let mut next = update;
    if let Some(workspace) = next.workspace.take() {
        next.workspace = Some(validate_workspace(workspace)?);
    }
    next.redaction_key_fragments = next
        .redaction_key_fragments
        .into_iter()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .take(32)
        .collect();
    let mut preferences = state
        .preferences
        .lock()
        .map_err(|_| "console preference lock is unavailable".to_owned())?;
    *preferences = next;
    persist_preferences(&state.preferences_path, &preferences)?;
    Ok(preferences.clone())
}

#[tauri::command]
fn list_runs(query: RunQuery, state: State<'_, ConsoleState>) -> Result<Vec<ConsoleRun>, String> {
    let workspace = state.workspace()?;
    let store = open_trace_store(&workspace)?;
    let status = query
        .status
        .as_deref()
        .filter(|value| !value.is_empty())
        .map(parse_run_status)
        .transpose()?;
    store
        .list_runs(RunListQuery {
            trace_id: query.trace_id.filter(|value| !value.trim().is_empty()),
            status,
            limit: MAX_RUNS,
            ..RunListQuery::default()
        })
        .map_err(|error| format!("Could not read local trace runs: {error}"))
        .map(|runs| {
            runs.into_iter()
                .map(|run| ConsoleRun {
                    run_id: run.run_id,
                    trace_id: run.trace_id,
                    started_at_ms: run.started_at_ms,
                    updated_at_ms: run.updated_at_ms,
                    event_count: run.event_count,
                    status: run.status.map(run_status_name),
                })
                .collect()
        })
}

#[tauri::command]
fn list_run_events(
    run_id: String,
    state: State<'_, ConsoleState>,
) -> Result<Vec<ConsoleEvent>, String> {
    let workspace = state.workspace()?;
    let store = open_trace_store(&workspace)?;
    store
        .events_for_run(&run_id, MAX_EVENTS_PER_RUN, 0)
        .map_err(|error| format!("Could not read local trace events: {error}"))
        .and_then(|events| {
            events
                .into_iter()
                .map(|event| {
                    // Deliberately drop PersistedEvent::raw_payload. The console
                    // never exposes it even when a database was written with raw
                    // capture enabled.
                    let record = event.record;
                    serde_json::to_value(record.event)
                        .map(|kind| ConsoleEvent {
                            sequence: record.sequence,
                            timestamp_ms: record.timestamp_ms,
                            event: kind,
                        })
                        .map_err(|error| format!("Could not encode trace event: {error}"))
                })
                .collect()
        })
}

#[tauri::command]
async fn list_models(state: State<'_, ConsoleState>) -> Result<ConsoleModels, String> {
    let workspace = state.workspace()?;
    let provider = OllamaProvider::builder()
        .base_url(workspace.ollama_url)
        .build()
        .map_err(|error| format!("Invalid local Ollama URL: {error}"))?;
    let health = provider
        .health()
        .await
        .map_err(|error| format!("Could not reach local Ollama: {error}"))?;
    let models = if health.healthy {
        provider
            .list_models()
            .await
            .map_err(|error| format!("Could not list local Ollama models: {error}"))?
    } else {
        Vec::new()
    };
    Ok(ConsoleModels { health, models })
}

#[tauri::command]
fn list_agents(state: State<'_, ConsoleState>) -> Result<Vec<AgentDefinition>, String> {
    let workspace = state.workspace()?;
    let Some(path) = workspace.agent_manifest_path else {
        return Ok(Vec::new());
    };
    load_agent_manifest_path(&path)
        .map(|manifest| manifest.agents)
        .map_err(|error| format!("Could not read project agent manifest: {error}"))
}

#[tauri::command]
fn list_evaluation_artifacts(
    state: State<'_, ConsoleState>,
) -> Result<EvaluationArtifacts, String> {
    let workspace = state.workspace()?;
    let Some(path) = workspace.evaluation_results_path else {
        return Ok(EvaluationArtifacts {
            reports: Vec::new(),
            skipped_files: Vec::new(),
        });
    };
    let path = PathBuf::from(path);
    let candidates = evaluation_candidates(&path)?;
    let mut reports = Vec::new();
    let mut skipped_files = Vec::new();
    for candidate in candidates {
        let content = fs::read_to_string(&candidate).map_err(|error| {
            format!(
                "Could not read evaluation artifact {}: {error}",
                candidate.display()
            )
        })?;
        match serde_json::from_str::<EvaluationReport>(&content) {
            Ok(report) => reports.push(EvaluationArtifact {
                path: candidate.display().to_string(),
                report,
            }),
            Err(_) => skipped_files.push(candidate.display().to_string()),
        }
    }
    reports.sort_by(|left, right| right.report.id.cmp(&left.report.id));
    Ok(EvaluationArtifacts {
        reports,
        skipped_files,
    })
}

/// Exposes only the two fixed, project-local artifacts generated by the
/// Promptfoo adapter. This is not a general file browser and has a strict size
/// cap so large raw result files cannot overwhelm the webview.
#[tauri::command]
fn list_promptfoo_artifacts(
    state: State<'_, ConsoleState>,
) -> Result<Vec<PromptfooArtifact>, String> {
    let workspace = state.workspace()?;
    let root = PathBuf::from(workspace.project_root);
    let candidates = [
        ("generated_config", root.join(".llama-harness/generated/promptfooconfig.yaml")),
        ("raw_result", root.join(".llama-harness/results/promptfoo-results.json")),
    ];
    candidates
        .into_iter()
        .filter(|(_, path)| path.is_file())
        .map(|(kind, path)| read_promptfoo_artifact(kind, &path))
        .collect()
}

#[tauri::command]
fn preview_eval_command(
    request: EvalLaunchRequest,
    state: State<'_, ConsoleState>,
) -> Result<CommandPreview, String> {
    build_eval_command(&state.workspace()?, &request)
}

#[tauri::command]
async fn launch_eval_command(
    request: EvalLaunchRequest,
    state: State<'_, ConsoleState>,
) -> Result<CommandResult, String> {
    let command = build_eval_command(&state.workspace()?, &request)?;
    run_command(command).await
}

#[tauri::command]
fn preview_replay_command(
    request: ReplayLaunchRequest,
    state: State<'_, ConsoleState>,
) -> Result<CommandPreview, String> {
    build_replay_command(&state.workspace()?, &request)
}

#[tauri::command]
async fn launch_replay_command(
    request: ReplayLaunchRequest,
    state: State<'_, ConsoleState>,
) -> Result<CommandResult, String> {
    let command = build_replay_command(&state.workspace()?, &request)?;
    run_command(command).await
}

fn load_preferences(path: &Path) -> ConsolePreferences {
    fs::read_to_string(path)
        .ok()
        .and_then(|content| serde_json::from_str(&content).ok())
        .unwrap_or_default()
}

fn persist_preferences(path: &Path, preferences: &ConsolePreferences) -> Result<(), String> {
    let content = serde_json::to_string_pretty(preferences)
        .map_err(|error| format!("Could not serialize console preferences: {error}"))?;
    fs::write(path, content).map_err(|error| format!("Could not save console preferences: {error}"))
}

fn validate_workspace(mut workspace: ProjectWorkspace) -> Result<ProjectWorkspace, String> {
    let root = canonical_existing_directory(&workspace.project_root, "Project root")?;
    let trace_database = canonical_existing_file(&workspace.trace_db_path, "Trace database")?;
    let evaluation_results = workspace
        .evaluation_results_path
        .as_deref()
        .map(|path| canonical_existing_path(path, "Evaluation results"))
        .transpose()?;
    let agent_manifest = workspace
        .agent_manifest_path
        .as_deref()
        .map(|path| canonical_project_file(&root, path, "Agent manifest"))
        .transpose()?;
    if let Some(path) = &agent_manifest {
        load_agent_manifest_path(path)
            .map_err(|error| format!("Agent manifest is not valid: {error}"))?;
    }
    let ollama_url = workspace.ollama_url.trim();
    if ollama_url.is_empty() {
        return Err("Ollama URL is required.".to_owned());
    }
    OllamaProvider::builder()
        .base_url(ollama_url)
        .build()
        .map_err(|error| format!("Ollama URL must be a loopback address: {error}"))?;

    workspace.project_root = root.display().to_string();
    workspace.trace_db_path = trace_database.display().to_string();
    workspace.evaluation_results_path = evaluation_results.map(|path| path.display().to_string());
    workspace.agent_manifest_path = agent_manifest.map(|path| path.display().to_string());
    workspace.ollama_url = ollama_url.to_owned();
    Ok(workspace)
}

fn canonical_project_file(root: &Path, value: &str, label: &str) -> Result<PathBuf, String> {
    let path = canonical_existing_file(value, label)?;
    path.strip_prefix(root)
        .map_err(|_| format!("{label} must be inside the selected project root."))?;
    Ok(path)
}

fn canonical_existing_directory(value: &str, label: &str) -> Result<PathBuf, String> {
    let path = canonical_existing_path(value, label)?;
    if !path.is_dir() {
        return Err(format!("{label} must be an existing directory."));
    }
    Ok(path)
}

fn canonical_existing_file(value: &str, label: &str) -> Result<PathBuf, String> {
    let path = canonical_existing_path(value, label)?;
    if !path.is_file() {
        return Err(format!("{label} must be an existing file."));
    }
    Ok(path)
}

fn canonical_existing_path(value: &str, label: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(value.trim());
    if !path.is_absolute() {
        return Err(format!("{label} must be an absolute path."));
    }
    path.canonicalize()
        .map_err(|error| format!("{label} does not exist or cannot be opened: {error}"))
}

fn open_trace_store(workspace: &ProjectWorkspace) -> Result<SqliteEventSink, String> {
    SqliteEventSink::open_read_only(&workspace.trace_db_path)
        .map_err(|error| format!("Could not open trace database read-only: {error}"))
}

fn evaluation_candidates(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let entries = fs::read_dir(path)
        .map_err(|error| format!("Could not list evaluation results: {error}"))?;
    let mut candidates = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|entry| {
            entry.is_file()
                && entry
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
        })
        .collect::<Vec<_>>();
    candidates.sort();
    Ok(candidates)
}

fn read_promptfoo_artifact(kind: &str, path: &Path) -> Result<PromptfooArtifact, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("Could not inspect Promptfoo artifact {}: {error}", path.display()))?;
    let file = fs::File::open(path)
        .map_err(|error| format!("Could not read Promptfoo artifact {}: {error}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(MAX_PROMPTFOO_ARTIFACT_BYTES)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Could not read Promptfoo artifact {}: {error}", path.display()))?;
    let truncated = metadata.len() > MAX_PROMPTFOO_ARTIFACT_BYTES;
    let content = String::from_utf8_lossy(&bytes).into_owned();
    Ok(PromptfooArtifact {
        kind: kind.to_owned(),
        path: path.display().to_string(),
        content,
        truncated,
    })
}

fn build_eval_command(
    workspace: &ProjectWorkspace,
    request: &EvalLaunchRequest,
) -> Result<CommandPreview, String> {
    if request.repeat == Some(0) {
        return Err("Repeat must be greater than zero.".to_owned());
    }
    let suite = project_relative_existing_file(workspace, &request.suite_path, "Evaluation suite")?;
    match Path::new(&suite)
        .extension()
        .and_then(|extension| extension.to_str())
    {
        Some("yaml" | "yml" | "json") => {}
        _ => return Err("Evaluation suite must be a YAML or JSON file.".to_owned()),
    }
    let mut args = vec![
        "run".to_owned(),
        "-q".to_owned(),
        "-p".to_owned(),
        "llama-harness-cli".to_owned(),
        "--".to_owned(),
        "eval".to_owned(),
        "run".to_owned(),
        suite,
    ];
    for model in &request.models {
        validate_model_id(model)?;
        args.push("--model".to_owned());
        args.push(model.clone());
    }
    if let Some(repeat) = request.repeat {
        args.push("--repeat".to_owned());
        args.push(repeat.to_string());
    }
    Ok(CommandPreview {
        program: "cargo".to_owned(),
        args,
        cwd: workspace.project_root.clone(),
    })
}

fn build_replay_command(
    workspace: &ProjectWorkspace,
    request: &ReplayLaunchRequest,
) -> Result<CommandPreview, String> {
    let regression =
        project_relative_existing_file(workspace, &request.regression_path, "Regression artifact")?;
    match Path::new(&regression)
        .extension()
        .and_then(|extension| extension.to_str())
    {
        Some("yaml" | "yml" | "json") => {}
        _ => return Err("Regression artifact must be a YAML or JSON file.".to_owned()),
    }
    Ok(CommandPreview {
        program: "cargo".to_owned(),
        args: vec![
            "run".to_owned(),
            "-q".to_owned(),
            "-p".to_owned(),
            "llama-harness-cli".to_owned(),
            "--".to_owned(),
            "replay".to_owned(),
            regression,
            "--json".to_owned(),
        ],
        cwd: workspace.project_root.clone(),
    })
}

fn project_relative_existing_file(
    workspace: &ProjectWorkspace,
    supplied: &str,
    label: &str,
) -> Result<String, String> {
    let root = PathBuf::from(&workspace.project_root);
    let candidate = if Path::new(supplied).is_absolute() {
        PathBuf::from(supplied)
    } else {
        root.join(supplied)
    };
    let canonical = candidate
        .canonicalize()
        .map_err(|error| format!("{label} does not exist or cannot be opened: {error}"))?;
    if !canonical.is_file() {
        return Err(format!("{label} must be a file."));
    }
    canonical
        .strip_prefix(&root)
        .map_err(|_| format!("{label} must be inside the selected project root."))
        .map(|path| path.to_string_lossy().replace('\\', "/"))
}

fn validate_model_id(model: &str) -> Result<(), String> {
    if model.is_empty()
        || model.len() > 200
        || !model.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.' | ':' | '/')
        })
    {
        return Err(
            "Model IDs may only contain letters, digits, '-', '_', '.', ':', and '/'.".to_owned(),
        );
    }
    Ok(())
}

async fn run_command(command: CommandPreview) -> Result<CommandResult, String> {
    let cloned = command.clone();
    tauri::async_runtime::spawn_blocking(move || {
        Command::new(&cloned.program)
            .args(&cloned.args)
            .current_dir(&cloned.cwd)
            .output()
            .map(|output| CommandResult {
                command: cloned,
                success: output.status.success(),
                exit_code: output.status.code(),
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            })
            .map_err(|error| format!("Could not start constrained Harness CLI command: {error}"))
    })
    .await
    .map_err(|error| format!("Constrained Harness CLI task failed: {error}"))?
}

fn parse_run_status(status: &str) -> Result<llama_harness_core::RunStatus, String> {
    serde_json::from_value(serde_json::Value::String(status.to_owned())).map_err(|_| {
        "Run status must be one of: completed, failed, cancelled, limit_reached.".to_owned()
    })
}

fn run_status_name(status: llama_harness_core::RunStatus) -> String {
    serde_json::to_value(status)
        .ok()
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| "unknown".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_workspace() -> (PathBuf, ProjectWorkspace) {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("llama-harness-console-{suffix}"));
        fs::create_dir_all(root.join("evals")).expect("temporary project directory should exist");
        fs::write(root.join("traces.sqlite"), []).expect("temporary trace file should exist");
        fs::write(root.join("evals/suite.yaml"), "id: suite")
            .expect("temporary suite should exist");
        fs::write(root.join("evals/replay.json"), "{}").expect("temporary replay should exist");
        let workspace = ProjectWorkspace {
            project_root: root.display().to_string(),
            trace_db_path: root.join("traces.sqlite").display().to_string(),
            evaluation_results_path: None,
            agent_manifest_path: None,
            ollama_url: "http://127.0.0.1:11434".to_owned(),
        };
        (root, workspace)
    }

    #[test]
    fn workspace_requires_existing_absolute_project_data_and_loopback_ollama() {
        let (root, workspace) = temporary_workspace();
        let valid = validate_workspace(workspace.clone()).expect("valid local workspace");
        assert!(Path::new(&valid.project_root).is_absolute());

        let mut relative = workspace.clone();
        relative.project_root = "relative".to_owned();
        assert!(validate_workspace(relative)
            .unwrap_err()
            .contains("absolute"));

        let mut remote_ollama = workspace;
        remote_ollama.ollama_url = "http://192.168.1.8:11434".to_owned();
        assert!(validate_workspace(remote_ollama)
            .unwrap_err()
            .contains("loopback"));
        fs::remove_dir_all(root).expect("temporary project should be removable");
    }

    #[test]
    fn eval_commands_are_project_scoped_and_validate_model_arguments() {
        let (root, workspace) = temporary_workspace();
        let command = build_eval_command(
            &validate_workspace(workspace.clone()).expect("valid workspace"),
            &EvalLaunchRequest {
                suite_path: "evals/suite.yaml".to_owned(),
                models: vec!["ollama/qwen3:latest".to_owned()],
                repeat: Some(2),
            },
        )
        .expect("safe eval command");
        assert_eq!(command.program, "cargo");
        assert!(command.args.contains(&"evals/suite.yaml".to_owned()));

        let outside = build_eval_command(
            &validate_workspace(workspace.clone()).expect("valid workspace"),
            &EvalLaunchRequest {
                suite_path: "../outside.yaml".to_owned(),
                models: vec![],
                repeat: None,
            },
        );
        assert!(outside.is_err());

        let invalid_model = build_eval_command(
            &validate_workspace(workspace).expect("valid workspace"),
            &EvalLaunchRequest {
                suite_path: "evals/suite.yaml".to_owned(),
                models: vec!["model; unexpected".to_owned()],
                repeat: None,
            },
        );
        assert!(invalid_model.is_err());
        fs::remove_dir_all(root).expect("temporary project should be removable");
    }

    #[test]
    fn promptfoo_artifact_reader_is_fixed_and_size_bounded() {
        let (root, _) = temporary_workspace();
        let config = root.join(".llama-harness/generated/promptfooconfig.yaml");
        fs::create_dir_all(config.parent().expect("config parent"))
            .expect("Promptfoo directory should exist");
        fs::write(&config, "providers: []").expect("config should be writable");
        let artifact = read_promptfoo_artifact("generated_config", &config)
            .expect("generated config should be readable");
        assert_eq!(artifact.kind, "generated_config");
        assert!(artifact.content.contains("providers"));
        assert!(!artifact.truncated);
        fs::remove_dir_all(root).expect("temporary project should be removable");
    }
}
