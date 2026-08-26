use llama_harness_core::RunStatus;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
/// Describes one failed evaluation expectation.
pub struct AssertionFailure {
    /// Stable name of the expectation that failed.
    pub rule: String,
    /// Human-readable explanation of the failure.
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
/// Normalized result for one case and repetition.
pub struct EvaluationCaseResult {
    /// Identifier of the evaluated suite.
    pub suite_id: String,
    /// Identifier of the evaluated case.
    pub case_id: String,
    /// Model used for the execution.
    pub model: String,
    /// One-based repetition number.
    pub repetition: u32,
    /// Whether all expectations passed.
    pub passed: bool,
    /// Expectations that failed for the case.
    pub failures: Vec<AssertionFailure>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Application-visible run identifier, when available.
    pub run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Trace identifier, when available.
    pub trace_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Final runner status, when available.
    pub status: Option<RunStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Run duration in milliseconds, when available.
    pub duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Number of model calls, when available.
    pub model_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Number of tool calls, when available.
    pub tool_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Agent implementation version, when available.
    pub agent_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Prompt version, when available.
    pub prompt_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Application-owned final state snapshot, when available.
    pub final_state: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Application-owned unresolved-items snapshot, when available.
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
/// Collection of normalized results for one suite evaluation.
pub struct EvaluationReport {
    /// Version of the report serialization format.
    pub format_version: u32,
    /// Unique identifier for this report.
    pub id: String,
    /// Identifier of the evaluated suite.
    pub suite_id: String,
    /// Version of the evaluated suite.
    pub suite_version: u32,
    /// Case results contained in the report.
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

    /// Returns the number of passing case results.
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|result| result.passed).count()
    }

    /// Returns the number of non-passing case results.
    pub fn failed_count(&self) -> usize {
        self.results.len().saturating_sub(self.passed_count())
    }
}
