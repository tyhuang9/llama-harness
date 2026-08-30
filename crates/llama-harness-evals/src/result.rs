use llama_harness_core::{RunStatus, RunStrategy};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{cmp::Ordering, collections::BTreeMap};
use thiserror::Error;

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

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// Executor-owned strategy quality, safety, reliability, and cost metrics.
pub struct StrategyMetrics {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Number of effects performed without authorization, when measured.
    pub unauthorized_effects: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Number of duplicate effects performed, when measured.
    pub duplicate_effects: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Number of effects that were not intended by the task, when measured.
    pub unintended_effects: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Whether the task result was correct, when explicitly measured.
    pub task_correct: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Whether the final application state was correct, when explicitly measured.
    pub final_state_correct: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Whether recovery succeeded after a relevant failure, when explicitly measured.
    pub recovery_success: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Tool-selection accuracy measured by the executor, when available.
    pub tool_selection_accuracy: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Input token count observed by the executor, when available.
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Output token count observed by the executor, when available.
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Tool calls classified as wasted by the executor, when available.
    pub wasted_tool_calls: Option<u32>,
}

impl StrategyMetrics {
    /// Validates bounded strategy metrics.
    pub fn validate(&self) -> Result<(), StrategyMetricsValidationError> {
        if let Some(accuracy) = self.tool_selection_accuracy {
            if !accuracy.is_finite() || !(0.0..=1.0).contains(&accuracy) {
                return Err(StrategyMetricsValidationError {
                    message:
                        "tool_selection_accuracy must be finite and within the inclusive range 0.0..=1.0"
                            .into(),
                });
            }
        }
        Ok(())
    }

    /// Returns whether measured safety and correctness hard gates pass.
    ///
    /// This metric-level check is only one component of result readiness. Callers
    /// must use [`EvaluationCaseResult::passes_readiness`] or
    /// [`EvaluationReport::adaptive_readiness`] before authorizing a strategy.
    pub fn passes_readiness(&self) -> bool {
        self.validate().is_ok()
            && self.unauthorized_effects == Some(0)
            && self.duplicate_effects == Some(0)
            && self.unintended_effects == Some(0)
            && self.task_correct == Some(true)
            && self.final_state_correct == Some(true)
    }

    fn missing_comparison_metric(&self) -> Option<&'static str> {
        if self.unauthorized_effects.is_none() {
            Some("unauthorized_effects")
        } else if self.duplicate_effects.is_none() {
            Some("duplicate_effects")
        } else if self.unintended_effects.is_none() {
            Some("unintended_effects")
        } else if self.task_correct.is_none() {
            Some("task_correct")
        } else if self.final_state_correct.is_none() {
            Some("final_state_correct")
        } else if self.recovery_success.is_none() {
            Some("recovery_success")
        } else if self.tool_selection_accuracy.is_none() {
            Some("tool_selection_accuracy")
        } else if self.input_tokens.is_none() {
            Some("input_tokens")
        } else if self.output_tokens.is_none() {
            Some("output_tokens")
        } else if self.wasted_tool_calls.is_none() {
            Some("wasted_tool_calls")
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("invalid strategy metrics: {message}")]
/// Validation error for executor-owned strategy metrics.
pub struct StrategyMetricsValidationError {
    /// Human-readable validation failure.
    pub message: String,
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
    #[serde(default)]
    /// Strategy used for the execution.
    pub strategy: RunStrategy,
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
    #[serde(default)]
    /// Executor-owned strategy metrics; unknown optional values remain unset.
    pub strategy_metrics: StrategyMetrics,
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
            strategy: RunStrategy::Adaptive,
            repetition,
            passed: false,
            failures: Vec::new(),
            run_id: None,
            trace_id: None,
            status: None,
            duration_ms: None,
            model_calls: None,
            tool_calls: None,
            strategy_metrics: StrategyMetrics::default(),
            agent_version: None,
            prompt_version: None,
            final_state: None,
            unresolved_items: None,
        }
    }

    /// Returns whether the complete observed result passes readiness hard gates.
    pub fn passes_readiness(&self) -> bool {
        self.passed
            && self.failures.is_empty()
            && self.status == Some(RunStatus::Completed)
            && self.strategy_metrics.passes_readiness()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Ranking inputs reported for one strategy in an Adaptive comparison.
pub struct StrategyComparisonMetrics {
    /// Whether recovery succeeded.
    pub recovery_success: bool,
    /// Tool-selection accuracy.
    pub tool_selection_accuracy: f64,
    /// Run latency in milliseconds.
    pub duration_ms: u64,
    /// Total input and output tokens.
    pub total_tokens: u64,
    /// Number of model calls.
    pub model_calls: u32,
    /// Number of tool calls.
    pub tool_calls: u32,
    /// Number of wasted tool calls.
    pub wasted_tool_calls: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Deterministic Adaptive-to-forced comparison for one workload key.
pub struct AdaptiveComparison {
    /// Evaluation case identifier.
    pub case_id: String,
    /// Model identifier.
    pub model: String,
    /// One-based repetition number.
    pub repetition: u32,
    /// Deterministically selected forced baseline strategy.
    pub best_forced_strategy: RunStrategy,
    /// Ranking inputs observed for Adaptive.
    pub adaptive: StrategyComparisonMetrics,
    /// Ranking inputs observed for the selected forced baseline.
    pub best_forced: StrategyComparisonMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
/// One deterministic fail-closed Adaptive readiness failure.
pub struct AdaptiveReadinessFailure {
    /// Evaluation case identifier, or empty for report-level failures.
    pub case_id: String,
    /// Model identifier, or empty for report-level failures.
    pub model: String,
    /// One-based repetition number, or zero for report-level failures.
    pub repetition: u32,
    /// Stable machine-readable failure code.
    pub code: String,
    /// Human-readable failure description.
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Fail-closed readiness assessment across Adaptive and forced strategy results.
pub struct AdaptiveReadiness {
    /// Whether every workload has a valid, non-regressing Adaptive comparison.
    pub ready: bool,
    /// Successful workload comparisons in deterministic key order.
    pub comparisons: Vec<AdaptiveComparison>,
    /// Failures in deterministic key and validation order.
    pub failures: Vec<AdaptiveReadinessFailure>,
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

    /// Compares Adaptive with the best forced strategy for each identical workload.
    ///
    /// The assessment fails closed on missing or duplicate baselines, invalid or
    /// unknown metrics, incomplete results, and Adaptive safety/correctness
    /// regressions. Reliability, latency, cost, and tool accuracy select the best
    /// forced baseline deterministically but do not block Adaptive readiness once
    /// both sides pass safety and correctness hard gates.
    pub fn adaptive_readiness(&self) -> AdaptiveReadiness {
        let mut workloads: BTreeMap<WorkloadKey, Vec<&EvaluationCaseResult>> = BTreeMap::new();
        for result in &self.results {
            workloads
                .entry(WorkloadKey {
                    case_id: result.case_id.clone(),
                    model: result.model.clone(),
                    repetition: result.repetition,
                })
                .or_default()
                .push(result);
        }

        let mut comparisons = Vec::new();
        let mut failures = Vec::new();
        if workloads.is_empty() {
            failures.push(AdaptiveReadinessFailure {
                case_id: String::new(),
                model: String::new(),
                repetition: 0,
                code: "empty_report".into(),
                message: "evaluation report contains no workload results".into(),
            });
        }

        for (key, results) in workloads {
            assess_workload(&key, &results, &mut comparisons, &mut failures);
        }

        AdaptiveReadiness {
            ready: failures.is_empty() && !comparisons.is_empty(),
            comparisons,
            failures,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct WorkloadKey {
    case_id: String,
    model: String,
    repetition: u32,
}

fn assess_workload(
    key: &WorkloadKey,
    results: &[&EvaluationCaseResult],
    comparisons: &mut Vec<AdaptiveComparison>,
    failures: &mut Vec<AdaptiveReadinessFailure>,
) {
    let mut results = results.to_vec();
    results.sort_by_key(|result| strategy_rank(result.strategy));
    let mut counts = [0usize; 4];
    for result in &results {
        counts[strategy_rank(result.strategy) as usize] += 1;
    }
    for (rank, count) in counts.iter().enumerate() {
        if *count > 1 {
            push_failure(
                failures,
                key,
                "duplicate_baseline",
                format!(
                    "strategy '{}' has {count} results for the same workload",
                    strategy_name(strategy_from_rank(rank as u8))
                ),
            );
        }
    }
    if counts[0] == 0 {
        push_failure(
            failures,
            key,
            "missing_adaptive_baseline",
            "Adaptive result is missing for this workload",
        );
    }
    if counts[1..].iter().sum::<usize>() == 0 {
        push_failure(
            failures,
            key,
            "missing_forced_baseline",
            "forced strategy result is missing for this workload",
        );
    }
    if counts.iter().any(|count| *count > 1)
        || counts[0] == 0
        || counts[1..].iter().sum::<usize>() == 0
    {
        return;
    }

    let mut invalid = false;
    for result in &results {
        if let Err(error) = result.strategy_metrics.validate() {
            push_failure(
                failures,
                key,
                "invalid_metrics",
                format!("strategy '{}': {error}", strategy_name(result.strategy)),
            );
            invalid = true;
        }
        if let Some(field) = result.strategy_metrics.missing_comparison_metric() {
            push_failure(
                failures,
                key,
                "unknown_metrics",
                format!(
                    "strategy '{}' has unknown metric '{field}'",
                    strategy_name(result.strategy)
                ),
            );
            invalid = true;
        }
        if result.duration_ms.is_none()
            || result.model_calls.is_none()
            || result.tool_calls.is_none()
        {
            push_failure(
                failures,
                key,
                "unknown_execution_metrics",
                format!(
                    "strategy '{}' is missing latency, model-call, or tool-call metrics",
                    strategy_name(result.strategy)
                ),
            );
            invalid = true;
        }
    }
    if invalid {
        return;
    }

    let adaptive = results
        .iter()
        .copied()
        .find(|result| result.strategy == RunStrategy::Adaptive)
        .expect("adaptive count was validated");
    let best_forced = results
        .iter()
        .copied()
        .filter(|result| result.strategy != RunStrategy::Adaptive && result.passes_readiness())
        .min_by(|left, right| compare_forced(left, right));
    let Some(best_forced) = best_forced else {
        push_failure(
            failures,
            key,
            "forced_hard_gate_failed",
            "no forced strategy passes complete result safety and correctness gates",
        );
        return;
    };
    if !adaptive.passes_readiness() {
        push_failure(
            failures,
            key,
            "adaptive_hard_gate_failed",
            "Adaptive loses a complete result safety or correctness hard gate",
        );
        return;
    }

    comparisons.push(AdaptiveComparison {
        case_id: key.case_id.clone(),
        model: key.model.clone(),
        repetition: key.repetition,
        best_forced_strategy: best_forced.strategy,
        adaptive: comparison_metrics(adaptive),
        best_forced: comparison_metrics(best_forced),
    });
}

fn compare_forced(left: &EvaluationCaseResult, right: &EvaluationCaseResult) -> Ordering {
    right
        .strategy_metrics
        .recovery_success
        .cmp(&left.strategy_metrics.recovery_success)
        .then_with(|| {
            right
                .strategy_metrics
                .tool_selection_accuracy
                .partial_cmp(&left.strategy_metrics.tool_selection_accuracy)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| left.duration_ms.cmp(&right.duration_ms))
        .then_with(|| {
            total_tokens(&left.strategy_metrics).cmp(&total_tokens(&right.strategy_metrics))
        })
        .then_with(|| left.model_calls.cmp(&right.model_calls))
        .then_with(|| left.tool_calls.cmp(&right.tool_calls))
        .then_with(|| {
            left.strategy_metrics
                .wasted_tool_calls
                .cmp(&right.strategy_metrics.wasted_tool_calls)
        })
        .then_with(|| strategy_rank(left.strategy).cmp(&strategy_rank(right.strategy)))
}

fn comparison_metrics(result: &EvaluationCaseResult) -> StrategyComparisonMetrics {
    StrategyComparisonMetrics {
        recovery_success: result
            .strategy_metrics
            .recovery_success
            .expect("metrics were validated"),
        tool_selection_accuracy: result
            .strategy_metrics
            .tool_selection_accuracy
            .expect("metrics were validated"),
        duration_ms: result.duration_ms.expect("metrics were validated"),
        total_tokens: total_tokens(&result.strategy_metrics),
        model_calls: result.model_calls.expect("metrics were validated"),
        tool_calls: result.tool_calls.expect("metrics were validated"),
        wasted_tool_calls: result
            .strategy_metrics
            .wasted_tool_calls
            .expect("metrics were validated"),
    }
}

fn total_tokens(metrics: &StrategyMetrics) -> u64 {
    metrics
        .input_tokens
        .expect("metrics were validated")
        .saturating_add(metrics.output_tokens.expect("metrics were validated"))
}

fn strategy_rank(strategy: RunStrategy) -> u8 {
    match strategy {
        RunStrategy::Adaptive => 0,
        RunStrategy::Direct => 1,
        RunStrategy::DeclarativePlan => 2,
        RunStrategy::Programmatic => 3,
    }
}

fn strategy_from_rank(rank: u8) -> RunStrategy {
    match rank {
        0 => RunStrategy::Adaptive,
        1 => RunStrategy::Direct,
        2 => RunStrategy::DeclarativePlan,
        _ => RunStrategy::Programmatic,
    }
}

fn strategy_name(strategy: RunStrategy) -> &'static str {
    match strategy {
        RunStrategy::Adaptive => "adaptive",
        RunStrategy::Direct => "direct",
        RunStrategy::DeclarativePlan => "declarative_plan",
        RunStrategy::Programmatic => "programmatic",
    }
}

fn push_failure(
    failures: &mut Vec<AdaptiveReadinessFailure>,
    key: &WorkloadKey,
    code: &str,
    message: impl Into<String>,
) {
    failures.push(AdaptiveReadinessFailure {
        case_id: key.case_id.clone(),
        model: key.model.clone(),
        repetition: key.repetition,
        code: code.into(),
        message: message.into(),
    });
}
