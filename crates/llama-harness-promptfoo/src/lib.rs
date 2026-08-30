//! Development-only generation of a visible Promptfoo workspace.
//!
//! Generated providers invoke an application-owned adapter. The bundled example
//! adapter runs a fresh full `AgentRunner` for each case; this crate never makes
//! Promptfoo a dependency of the embedded runtime.

use llama_harness_core::RunResult;
use llama_harness_evals::{
    evaluate_expectations, load_suite_path, EvalObservation, EvalSuite, EvaluationCaseResult,
    EvaluationReport,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};
use thiserror::Error;

const PROVIDER_SOURCE: &str = include_str!("../assets/agent-provider.mjs");

#[derive(Clone, Debug)]
pub struct PromptfooWorkspace {
    pub config_path: PathBuf,
    pub provider_path: PathBuf,
    pub raw_result_path: PathBuf,
    pub observation_path: PathBuf,
    pub normalized_report_path: PathBuf,
    pub trace_db_path: PathBuf,
}

#[derive(Debug, Error)]
pub enum PromptfooError {
    #[error(transparent)]
    Eval(#[from] llama_harness_evals::EvalError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("invalid Promptfoo configuration: {0}")]
    Invalid(String),
}

/// Generates a debug-visible Promptfoo configuration and custom provider for a
/// validated suite. `project_root` must contain the embedding example/adapter.
pub fn generate_workspace(
    suite_path: impl AsRef<Path>,
    project_root: impl AsRef<Path>,
    output_root: impl AsRef<Path>,
    models: &[String],
    ollama_url: &str,
) -> Result<PromptfooWorkspace, PromptfooError> {
    let suite_path = suite_path.as_ref().canonicalize()?;
    let project_root = project_root.as_ref().canonicalize()?;
    let suite = load_suite_path(&suite_path)?;
    if suite.agent != "local-task-agent" {
        return Err(PromptfooError::Invalid(format!(
            "suite agent {} needs an application-owned Promptfoo adapter; this bundled wrapper supports only local-task-agent",
            suite.agent
        )));
    }
    let selected_models = if models.is_empty() {
        &suite.models
    } else {
        models
    };
    if selected_models.is_empty() {
        return Err(PromptfooError::Invalid(
            "at least one model is required".into(),
        ));
    }
    for model in selected_models {
        if !model.starts_with("ollama:") || model.trim().len() <= "ollama:".len() {
            return Err(PromptfooError::Invalid(
                "Promptfoo adapter models must use an ollama:<installed-model> ID".into(),
            ));
        }
    }
    let output_root = output_root.as_ref();
    let generated = output_root.join("generated");
    let results = output_root.join("results");
    fs::create_dir_all(&generated)?;
    fs::create_dir_all(&results)?;
    let provider_path = generated.join("agent-provider.mjs");
    let config_path = generated.join("promptfooconfig.yaml");
    let raw_result_path = results.join("promptfoo-results.json");
    let observation_path = results.join("promptfoo-observations.jsonl");
    let normalized_report_path = results.join("promptfoo-normalized-report.json");
    let trace_db_path = results.join("promptfoo-traces.sqlite");
    fs::write(&provider_path, PROVIDER_SOURCE)?;
    let config = PromptfooConfig::from_suite(
        &suite,
        &suite_path,
        &project_root,
        &trace_db_path,
        &observation_path,
        selected_models,
        ollama_url,
    );
    fs::write(&config_path, serde_yaml::to_string(&config)?)?;
    Ok(PromptfooWorkspace {
        config_path,
        provider_path,
        raw_result_path,
        observation_path,
        normalized_report_path,
        trace_db_path,
    })
}

#[derive(Serialize)]
struct PromptfooConfig {
    description: String,
    prompts: Vec<String>,
    providers: Vec<PromptfooProvider>,
    tests: Vec<PromptfooTest>,
}

#[derive(Serialize)]
struct PromptfooProvider {
    id: String,
    label: String,
    config: PromptfooProviderConfig,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PromptfooProviderConfig {
    model: String,
    suite_path: String,
    project_root: String,
    trace_db_path: String,
    observation_path: String,
    ollama_url: String,
}

#[derive(Serialize)]
struct PromptfooTest {
    vars: PromptfooVars,
}

#[derive(Serialize)]
struct PromptfooVars {
    case_id: String,
    input: String,
    repetition: u32,
}

impl PromptfooConfig {
    fn from_suite(
        suite: &EvalSuite,
        suite_path: &Path,
        project_root: &Path,
        trace_db_path: &Path,
        observation_path: &Path,
        models: &[String],
        ollama_url: &str,
    ) -> Self {
        Self {
            description: format!("Llama Harness {} v{}", suite.id, suite.version),
            prompts: vec!["{{input}}".into()],
            providers: models
                .iter()
                .map(|model| PromptfooProvider {
                    // Promptfoo resolves `file://` custom providers relative to
                    // the configuration file. A relative reference works on
                    // Windows too; an absolute `C:/...` path is treated as POSIX.
                    id: "file://agent-provider.mjs".into(),
                    label: model.clone(),
                    config: PromptfooProviderConfig {
                        model: model.clone(),
                        suite_path: portable_path(suite_path),
                        project_root: portable_path(project_root),
                        trace_db_path: portable_path(trace_db_path),
                        observation_path: portable_path(observation_path),
                        ollama_url: ollama_url.into(),
                    },
                })
                .collect(),
            tests: suite
                .cases
                .iter()
                .flat_map(|case| {
                    let repetitions = case.repeat.unwrap_or(suite.defaults.repeat);
                    (1..=repetitions).map(move |repetition| PromptfooTest {
                        vars: PromptfooVars {
                            case_id: case.id.clone(),
                            input: case.input.clone(),
                            repetition,
                        },
                    })
                })
                .collect(),
        }
    }
}

fn portable_path(path: &Path) -> String {
    path.to_string_lossy()
        .trim_start_matches("\\\\?\\")
        .replace('\\', "/")
}

/// Converts the sidecar observations written by the custom provider into the
/// repository's provider-neutral result format. This is intentionally separate
/// from Promptfoo's raw output so the trace/run links survive format changes in
/// Promptfoo itself.
pub fn normalize_observations(
    suite_path: impl AsRef<Path>,
    observation_path: impl AsRef<Path>,
    report_path: impl AsRef<Path>,
) -> Result<EvaluationReport, PromptfooError> {
    let suite = load_suite_path(suite_path)?;
    let observation_path = observation_path.as_ref();
    let input = fs::read_to_string(observation_path)?;
    let mut results = Vec::new();
    for line in input.lines().filter(|line| !line.trim().is_empty()) {
        let observation: PromptfooObservation = serde_json::from_str(line).map_err(|error| {
            PromptfooError::Invalid(format!("invalid provider observation: {error}"))
        })?;
        let case = suite
            .cases
            .iter()
            .find(|case| case.id == observation.case_id)
            .ok_or_else(|| {
                PromptfooError::Invalid(format!(
                    "observation references unknown case {}",
                    observation.case_id
                ))
            })?;
        let result = match observation.response {
            Some(response) => {
                let harness_observation = EvalObservation::new(response.run, response.model_calls)
                    .with_final_state(Some(response.final_state))
                    .with_unresolved_items(response.unresolved_items)
                    .with_agent_version(Some(response.agent_version))
                    .with_prompt_version(Some(response.prompt_version));
                let failures = evaluate_expectations(&case.expected, &harness_observation);
                let mut result = EvaluationCaseResult::new(
                    suite.id.clone(),
                    case.id.clone(),
                    observation.model,
                    observation.repetition,
                );
                result.passed = failures.is_empty();
                result.failures = failures;
                result.run_id = Some(harness_observation.run.id.clone());
                result.trace_id = Some(harness_observation.run.trace_id.clone());
                result.status = Some(harness_observation.run.status.clone());
                result.duration_ms = Some(harness_observation.run.duration_ms);
                result.model_calls = Some(harness_observation.model_calls);
                result.tool_calls = Some(harness_observation.run.tool_calls.len() as u32);
                result.strategy_metrics = harness_observation.strategy_metrics;
                result.agent_version = harness_observation.agent_version;
                result.prompt_version = harness_observation.prompt_version;
                result.final_state = harness_observation.final_state;
                result.unresolved_items = harness_observation.unresolved_items;
                result
            }
            None => {
                let mut result = EvaluationCaseResult::new(
                    suite.id.clone(),
                    case.id.clone(),
                    observation.model,
                    observation.repetition,
                );
                result.failures = vec![llama_harness_evals::AssertionFailure::new(
                    "executor",
                    observation
                        .error
                        .unwrap_or_else(|| "Promptfoo provider did not record a response".into()),
                )];
                result
            }
        };
        results.push(result);
    }
    if results.is_empty() {
        return Err(PromptfooError::Invalid(
            "Promptfoo produced no provider observations; inspect the raw result artifact".into(),
        ));
    }
    let report = EvaluationReport::new(
        format!("promptfoo-{}", uuid::Uuid::new_v4()),
        suite.id,
        suite.version,
        results,
    );
    fs::write(
        report_path,
        serde_json::to_string_pretty(&report)
            .map_err(|error| PromptfooError::Invalid(error.to_string()))?,
    )?;
    Ok(report)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptfooObservation {
    case_id: String,
    model: String,
    repetition: u32,
    #[serde(default)]
    response: Option<PromptfooAgentResponse>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptfooAgentResponse {
    #[serde(default, rename = "output")]
    _output: String,
    run: RunResult,
    model_calls: u32,
    final_state: Value,
    #[serde(default)]
    unresolved_items: Option<Value>,
    agent_version: String,
    prompt_version: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_harness_core::RunStatus;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn generates_visible_config_and_provider_for_each_case() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("llama-harness-promptfoo-{stamp}"));
        fs::create_dir_all(&root).unwrap();
        let suite_path = root.join("suite.yaml");
        fs::write(&suite_path, "version: 1\nid: suite\nname: Suite\nagent: local-task-agent\nmodels: [ollama:qwen3]\ncases:\n  - id: case-a\n    input: hello\n").unwrap();
        let workspace = generate_workspace(
            &suite_path,
            &root,
            root.join(".llama-harness"),
            &[],
            "http://127.0.0.1:11434",
        )
        .unwrap();
        let config = fs::read_to_string(&workspace.config_path).unwrap();
        assert!(workspace.provider_path.is_file());
        assert!(config.contains("case-a"));
        assert!(config.contains("ollama:qwen3"));
        assert!(config.contains("agent-provider.mjs"));
        assert!(config.contains("observationPath"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn normalizer_retains_provider_run_and_trace_links() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("llama-harness-promptfoo-normalize-{stamp}"));
        fs::create_dir_all(&root).unwrap();
        let suite_path = root.join("suite.yaml");
        fs::write(&suite_path, "version: 1\nid: suite\nname: Suite\nagent: local-task-agent\nmodels: [ollama:qwen3]\ncases:\n  - id: case-a\n    input: hello\n    expected: {status: completed}\n").unwrap();
        let observations = root.join("observations.jsonl");
        let mut run = RunResult::new("run-1", RunStatus::Completed, "qwen3", "trace-1");
        run.final_output = Some("hello".to_owned());
        run.duration_ms = 12;
        fs::write(
            &observations,
            format!(
                "{}\n",
                json!({
                    "case_id": "case-a", "model": "ollama:qwen3", "repetition": 1,
                    "response": {
                        "output": "hello", "model_calls": 1, "final_state": {"tasks": []},
                        "agent_version": "1", "prompt_version": "prompt-1",
                        "run": run
                    }
                })
            ),
        )
        .unwrap();
        let report =
            normalize_observations(&suite_path, &observations, root.join("report.json")).unwrap();
        assert!(report.results[0].passed);
        assert_eq!(report.results[0].run_id.as_deref(), Some("run-1"));
        assert_eq!(report.results[0].trace_id.as_deref(), Some("trace-1"));
        fs::remove_dir_all(root).unwrap();
    }
}
