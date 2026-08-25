use llama_harness_core::RunStatus;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct AssertionFailure {
    pub rule: String,
    pub message: String,
}

impl AssertionFailure {
    /// Creates one failed evaluation assertion.
    pub fn new(rule: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            rule: rule.into(),
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct EvaluationCaseResult {
    pub suite_id: String,
    pub case_id: String,
    pub model: String,
    pub repetition: u32,
    pub passed: bool,
    pub failures: Vec<AssertionFailure>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<RunStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_state: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unresolved_items: Option<Value>,
}

impl EvaluationCaseResult {
    /// Creates an empty case result for an evaluator or adapter to populate.
    pub fn new(
        suite_id: impl Into<String>,
        case_id: impl Into<String>,
        model: impl Into<String>,
        repetition: u32,
    ) -> Self {
        Self {
            suite_id: suite_id.into(),
            case_id: case_id.into(),
            model: model.into(),
            repetition,
            passed: false,
            failures: Vec::new(),
            run_id: None,
            trace_id: None,
            status: None,
            duration_ms: None,
            model_calls: None,
            tool_calls: None,
            agent_version: None,
            prompt_version: None,
            final_state: None,
            unresolved_items: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct EvaluationReport {
    pub format_version: u32,
    pub id: String,
    pub suite_id: String,
    pub suite_version: u32,
    pub results: Vec<EvaluationCaseResult>,
}

impl EvaluationReport {
    /// Creates a version-one evaluation report.
    pub fn new(
        id: impl Into<String>,
        suite_id: impl Into<String>,
        suite_version: u32,
        results: Vec<EvaluationCaseResult>,
    ) -> Self {
        Self {
            format_version: 1,
            id: id.into(),
            suite_id: suite_id.into(),
            suite_version,
            results,
        }
    }

    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|result| result.passed).count()
    }

    pub fn failed_count(&self) -> usize {
        self.results.len().saturating_sub(self.passed_count())
    }
}
