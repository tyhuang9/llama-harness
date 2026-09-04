use llama_harness_core::{RunStatus, RunStrategy};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
/// Aggregate ranking inputs reported for one strategy in an Adaptive cohort comparison.
pub struct StrategyComparisonMetrics {
    /// Whether recovery succeeded for every sample in the cohort.
    ///
    /// This retained compatibility field is not used for ranking; use
    /// [`Self::recovery_success_rate`] for the aggregate reliability measure.
    pub recovery_success: bool,
    /// Fraction of samples for which recovery succeeded.
    pub recovery_success_rate: f64,
    /// Mean tool-selection accuracy across every sample in the cohort.
    pub tool_selection_accuracy: f64,
    /// P50 latency in milliseconds, using the deterministic nearest-rank method.
    ///
    /// This retained compatibility field equals [`Self::p50_latency_ms`].
    pub duration_ms: u64,
    /// P50 latency in milliseconds, using the deterministic nearest-rank method.
    pub p50_latency_ms: u64,
    /// P95 latency in milliseconds, using the deterministic nearest-rank method.
    pub p95_latency_ms: u64,
    /// Total input and output tokens.
    pub total_tokens: u64,
    /// Number of model calls.
    pub model_calls: u32,
    /// Number of tool calls.
    pub tool_calls: u32,
    /// Number of wasted tool calls.
    pub wasted_tool_calls: u32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// First lexicographic ranking field that determined a forced-candidate outcome.
pub enum StrategySelectionCriterion {
    /// Legacy single-sample recovery criterion.
    ///
    /// Cohort comparisons use [`Self::RecoverySuccessRate`] instead.
    RecoverySuccess,
    /// A higher recovery success rate won.
    RecoverySuccessRate,
    /// Higher tool-selection accuracy won.
    ToolSelectionAccuracy,
    /// Legacy single-sample latency criterion.
    ///
    /// Cohort comparisons use [`Self::P50LatencyMs`] instead.
    DurationMs,
    /// Lower P50 latency won.
    P50LatencyMs,
    /// Lower P95 latency won after P50 latency tied.
    P95LatencyMs,
    /// Lower total token usage won.
    TotalTokens,
    /// Fewer model calls won.
    ModelCalls,
    /// Fewer tool calls won.
    ToolCalls,
    /// Fewer wasted tool calls won.
    WastedToolCalls,
    /// Stable strategy enum order resolved an otherwise complete tie.
    StableStrategyOrder,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
/// Auditable disposition assigned to one forced strategy candidate.
pub enum ForcedCandidateDisposition {
    /// This candidate was selected as the best forced baseline.
    Selected,
    /// This candidate failed a complete-result hard gate.
    Ineligible {
        /// Stable machine-readable ineligibility code.
        code: String,
        /// Human-readable ineligibility reason.
        reason: String,
    },
    /// This eligible candidate lost to the selected baseline.
    Outranked {
        /// First ranking field on which the selected baseline won.
        decisive_criterion: StrategySelectionCriterion,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Ranking inputs and disposition for one forced strategy candidate.
pub struct ForcedCandidateComparison {
    /// Forced strategy evaluated.
    pub strategy: RunStrategy,
    /// Complete ranking inputs observed for the candidate.
    pub metrics: StrategyComparisonMetrics,
    /// Selection outcome and rationale.
    pub disposition: ForcedCandidateDisposition,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Deterministic Adaptive-to-forced comparison for one `(case_id, model)` cohort.
pub struct AdaptiveComparison {
    /// Evaluation case identifier.
    pub case_id: String,
    /// Model identifier.
    pub model: String,
    /// Number of matched repetitions aggregated into this comparison.
    pub sample_count: u32,
    /// Legacy per-repetition field.
    ///
    /// Cohort comparisons aggregate every matched repetition, so this is always
    /// zero. Use [`Self::sample_count`] and the P50/P95 fields in the strategy
    /// metrics instead.
    pub repetition: u32,
    /// Deterministically selected forced baseline strategy.
    pub best_forced_strategy: RunStrategy,
    /// Ranking inputs observed for Adaptive.
    pub adaptive: StrategyComparisonMetrics,
    /// Ranking inputs observed for the selected forced baseline.
    pub best_forced: StrategyComparisonMetrics,
    /// Every forced candidate in stable strategy order with selection rationale.
    pub forced_candidates: Vec<ForcedCandidateComparison>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
/// One deterministic fail-closed Adaptive readiness failure.
pub struct AdaptiveReadinessFailure {
    /// Evaluation case identifier, or empty for report-level failures.
    pub case_id: String,
    /// Model identifier, or empty for report-level failures.
    pub model: String,
    /// One-based repetition number for a per-repetition failure, or zero for a
    /// cohort- or report-level failure.
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

    /// Compares Adaptive with the best forced strategy for each `(case_id, model)`
    /// cohort.
    ///
    /// Every participating strategy must provide the same unique, nonzero
    /// repetition set. The assessment fails closed on missing or duplicate
    /// baselines, mismatched repetitions, invalid or unknown metrics, incomplete
    /// results, and Adaptive safety/correctness regressions. Hard gates aggregate
    /// across every repetition before ranking. Eligible forced baselines rank by
    /// recovery-success rate, mean tool-selection accuracy, P50 latency, P95
    /// latency, and then total tokens, model calls, tool calls, and wasted tool
    /// calls. These ranking fields do not block Adaptive readiness once both sides
    /// pass the complete safety and correctness hard gates.
    pub fn adaptive_readiness(&self) -> AdaptiveReadiness {
        let mut workloads: BTreeMap<CohortKey, Vec<&EvaluationCaseResult>> = BTreeMap::new();
        for result in &self.results {
            workloads
                .entry(CohortKey {
                    case_id: result.case_id.clone(),
                    model: result.model.clone(),
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
            assess_cohort(&key, &results, &mut comparisons, &mut failures);
        }

        AdaptiveReadiness {
            ready: failures.is_empty() && !comparisons.is_empty(),
            comparisons,
            failures,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CohortKey {
    case_id: String,
    model: String,
}

fn assess_cohort(
    key: &CohortKey,
    results: &[&EvaluationCaseResult],
    comparisons: &mut Vec<AdaptiveComparison>,
    failures: &mut Vec<AdaptiveReadinessFailure>,
) {
    let mut samples: [BTreeMap<u32, &EvaluationCaseResult>; 4] =
        std::array::from_fn(|_| BTreeMap::new());
    let mut invalid = false;

    for result in results {
        if result.repetition == 0 {
            push_failure_at_repetition(
                failures,
                key,
                result.repetition,
                "invalid_repetition",
                format!(
                    "strategy '{}' has invalid zero repetition",
                    strategy_name(result.strategy)
                ),
            );
            invalid = true;
            continue;
        }
        let repetitions = &mut samples[strategy_rank(result.strategy) as usize];
        if repetitions.insert(result.repetition, *result).is_some() {
            push_failure_at_repetition(
                failures,
                key,
                result.repetition,
                "duplicate_baseline",
                format!(
                    "strategy '{}' has duplicate repetition {} in one cohort",
                    strategy_name(result.strategy),
                    result.repetition
                ),
            );
            invalid = true;
        }
    }

    if samples[0].is_empty() {
        push_failure(
            failures,
            key,
            "missing_adaptive_baseline",
            "Adaptive result is missing for this workload",
        );
    }
    if samples[1..].iter().all(BTreeMap::is_empty) {
        push_failure(
            failures,
            key,
            "missing_forced_baseline",
            "forced strategy result is missing for this workload",
        );
    }
    if invalid || samples[0].is_empty() || samples[1..].iter().all(BTreeMap::is_empty) {
        return;
    }

    let expected_repetitions = samples[0].keys().copied().collect::<BTreeSet<_>>();
    for repetitions in samples
        .iter()
        .skip(1)
        .filter(|repetitions| !repetitions.is_empty())
    {
        let actual_repetitions = repetitions.keys().copied().collect::<BTreeSet<_>>();
        if actual_repetitions != expected_repetitions {
            let strategy = repetitions
                .values()
                .next()
                .expect("nonempty repetitions have a strategy")
                .strategy;
            push_failure(
                failures,
                key,
                "mismatched_repetition_sets",
                format!(
                    "strategy '{}' repetitions {:?} do not match Adaptive repetitions {:?}",
                    strategy_name(strategy),
                    actual_repetitions,
                    expected_repetitions
                ),
            );
            invalid = true;
        }
    }

    for result in results {
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

    let sample_count = match u32::try_from(expected_repetitions.len()) {
        Ok(sample_count) => sample_count,
        Err(_) => {
            push_failure(
                failures,
                key,
                "metrics_overflow",
                "cohort sample count exceeds its supported range",
            );
            return;
        }
    };

    let mut aggregates = Vec::new();
    for (rank, repetitions) in samples.iter().enumerate() {
        if repetitions.is_empty() {
            continue;
        }
        match aggregate_strategy(
            strategy_from_rank(rank as u8),
            repetitions.values().copied().collect(),
        ) {
            Ok(aggregate) => aggregates.push(aggregate),
            Err(error) => {
                push_failure(
                    failures,
                    key,
                    "metrics_overflow",
                    format!(
                        "strategy '{}': {}",
                        strategy_name(strategy_from_rank(rank as u8)),
                        error.message()
                    ),
                );
                return;
            }
        }
    }

    let adaptive = aggregates
        .iter()
        .find(|aggregate| aggregate.strategy == RunStrategy::Adaptive)
        .expect("adaptive cohort was validated");
    let best_forced = aggregates
        .iter()
        .filter(|aggregate| {
            aggregate.strategy != RunStrategy::Adaptive && aggregate.passes_readiness
        })
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
    if !adaptive.passes_readiness {
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
        sample_count,
        repetition: 0,
        best_forced_strategy: best_forced.strategy,
        adaptive: adaptive.metrics,
        best_forced: best_forced.metrics,
        forced_candidates: aggregates
            .iter()
            .filter(|aggregate| aggregate.strategy != RunStrategy::Adaptive)
            .map(|candidate| ForcedCandidateComparison {
                strategy: candidate.strategy,
                metrics: candidate.metrics,
                disposition: forced_candidate_disposition(best_forced, candidate),
            })
            .collect(),
    });
}

struct StrategyCohort<'a> {
    strategy: RunStrategy,
    samples: Vec<&'a EvaluationCaseResult>,
    metrics: StrategyComparisonMetrics,
    passes_readiness: bool,
}

enum AggregateError {
    CounterOverflow(&'static str),
}

impl AggregateError {
    fn message(&self) -> String {
        match self {
            Self::CounterOverflow(field) => format!("'{field}' exceeds its supported range"),
        }
    }
}

fn aggregate_strategy<'a>(
    strategy: RunStrategy,
    samples: Vec<&'a EvaluationCaseResult>,
) -> Result<StrategyCohort<'a>, AggregateError> {
    let sample_count = u64::try_from(samples.len())
        .map_err(|_| AggregateError::CounterOverflow("sample_count"))?;
    let mut successful_recoveries = 0_u64;
    let mut accuracy_sum = 0.0_f64;
    let mut latencies = Vec::with_capacity(samples.len());
    let mut total_token_count = 0_u64;
    let mut model_call_count = 0_u32;
    let mut tool_call_count = 0_u32;
    let mut wasted_tool_call_count = 0_u32;

    for sample in &samples {
        let metrics = &sample.strategy_metrics;
        if metrics.recovery_success.expect("metrics were validated") {
            successful_recoveries = successful_recoveries
                .checked_add(1)
                .ok_or(AggregateError::CounterOverflow("recovery_success"))?;
        }
        accuracy_sum += metrics
            .tool_selection_accuracy
            .expect("metrics were validated");
        if !accuracy_sum.is_finite() {
            return Err(AggregateError::CounterOverflow("tool_selection_accuracy"));
        }
        latencies.push(sample.duration_ms.expect("metrics were validated"));
        total_token_count = total_token_count
            .checked_add(checked_total_tokens(metrics)?)
            .ok_or(AggregateError::CounterOverflow("total_tokens"))?;
        model_call_count = model_call_count
            .checked_add(sample.model_calls.expect("metrics were validated"))
            .ok_or(AggregateError::CounterOverflow("model_calls"))?;
        tool_call_count = tool_call_count
            .checked_add(sample.tool_calls.expect("metrics were validated"))
            .ok_or(AggregateError::CounterOverflow("tool_calls"))?;
        wasted_tool_call_count = wasted_tool_call_count
            .checked_add(metrics.wasted_tool_calls.expect("metrics were validated"))
            .ok_or(AggregateError::CounterOverflow("wasted_tool_calls"))?;
    }

    let p50_latency_ms = nearest_rank_percentile(&mut latencies, 50);
    let p95_latency_ms = nearest_rank_percentile(&mut latencies, 95);
    Ok(StrategyCohort {
        strategy,
        passes_readiness: samples.iter().all(|sample| sample.passes_readiness()),
        samples,
        metrics: StrategyComparisonMetrics {
            recovery_success: successful_recoveries == sample_count,
            recovery_success_rate: successful_recoveries as f64 / sample_count as f64,
            tool_selection_accuracy: accuracy_sum / sample_count as f64,
            duration_ms: p50_latency_ms,
            p50_latency_ms,
            p95_latency_ms,
            total_tokens: total_token_count,
            model_calls: model_call_count,
            tool_calls: tool_call_count,
            wasted_tool_calls: wasted_tool_call_count,
        },
    })
}

fn nearest_rank_percentile(values: &mut [u64], percentile: u8) -> u64 {
    debug_assert!(!values.is_empty());
    debug_assert!((1..=100).contains(&percentile));
    values.sort_unstable();
    let percentile = usize::from(percentile);
    let hundreds = values.len() / 100;
    let remainder = values.len() % 100;
    let rank = hundreds * percentile + (remainder * percentile).div_ceil(100);
    values[rank - 1]
}

fn checked_total_tokens(metrics: &StrategyMetrics) -> Result<u64, AggregateError> {
    metrics
        .input_tokens
        .expect("metrics were validated")
        .checked_add(metrics.output_tokens.expect("metrics were validated"))
        .ok_or(AggregateError::CounterOverflow("total_tokens"))
}

fn forced_candidate_disposition(
    selected: &StrategyCohort<'_>,
    candidate: &StrategyCohort<'_>,
) -> ForcedCandidateDisposition {
    if candidate.strategy == selected.strategy {
        return ForcedCandidateDisposition::Selected;
    }
    if !candidate.passes_readiness {
        let (code, reason) = ineligible_reason(candidate);
        return ForcedCandidateDisposition::Ineligible {
            code: code.into(),
            reason: reason.into(),
        };
    }
    ForcedCandidateDisposition::Outranked {
        decisive_criterion: decisive_criterion(&selected.metrics, &candidate.metrics),
    }
}

fn ineligible_reason(cohort: &StrategyCohort<'_>) -> (&'static str, &'static str) {
    if cohort.samples.iter().any(|sample| !sample.passed) {
        ("case_not_passed", "evaluation expectations did not pass")
    } else if cohort
        .samples
        .iter()
        .any(|sample| sample.status != Some(RunStatus::Completed))
    {
        ("run_not_completed", "run status was not completed")
    } else if cohort
        .samples
        .iter()
        .any(|sample| !sample.failures.is_empty())
    {
        (
            "assertion_failures",
            "evaluation result contains assertion failures",
        )
    } else if cohort.samples.iter().any(|sample| {
        sample.strategy_metrics.unauthorized_effects != Some(0)
            || sample.strategy_metrics.duplicate_effects != Some(0)
            || sample.strategy_metrics.unintended_effects != Some(0)
    }) {
        (
            "safety_hard_gate_failed",
            "one or more measured safety effect counters were nonzero",
        )
    } else {
        (
            "correctness_hard_gate_failed",
            "task or final-state correctness did not pass",
        )
    }
}

fn decisive_criterion(
    selected: &StrategyComparisonMetrics,
    candidate: &StrategyComparisonMetrics,
) -> StrategySelectionCriterion {
    if selected.recovery_success_rate != candidate.recovery_success_rate {
        StrategySelectionCriterion::RecoverySuccessRate
    } else if selected.tool_selection_accuracy != candidate.tool_selection_accuracy {
        StrategySelectionCriterion::ToolSelectionAccuracy
    } else if selected.p50_latency_ms != candidate.p50_latency_ms {
        StrategySelectionCriterion::P50LatencyMs
    } else if selected.p95_latency_ms != candidate.p95_latency_ms {
        StrategySelectionCriterion::P95LatencyMs
    } else if selected.total_tokens != candidate.total_tokens {
        StrategySelectionCriterion::TotalTokens
    } else if selected.model_calls != candidate.model_calls {
        StrategySelectionCriterion::ModelCalls
    } else if selected.tool_calls != candidate.tool_calls {
        StrategySelectionCriterion::ToolCalls
    } else if selected.wasted_tool_calls != candidate.wasted_tool_calls {
        StrategySelectionCriterion::WastedToolCalls
    } else {
        StrategySelectionCriterion::StableStrategyOrder
    }
}

fn compare_forced(left: &StrategyCohort<'_>, right: &StrategyCohort<'_>) -> Ordering {
    right
        .metrics
        .recovery_success_rate
        .partial_cmp(&left.metrics.recovery_success_rate)
        .expect("validated recovery rates are finite")
        .then_with(|| {
            right
                .metrics
                .tool_selection_accuracy
                .partial_cmp(&left.metrics.tool_selection_accuracy)
                .expect("validated tool-selection accuracy is finite")
        })
        .then_with(|| {
            left.metrics
                .p50_latency_ms
                .cmp(&right.metrics.p50_latency_ms)
        })
        .then_with(|| {
            left.metrics
                .p95_latency_ms
                .cmp(&right.metrics.p95_latency_ms)
        })
        .then_with(|| left.metrics.total_tokens.cmp(&right.metrics.total_tokens))
        .then_with(|| left.metrics.model_calls.cmp(&right.metrics.model_calls))
        .then_with(|| left.metrics.tool_calls.cmp(&right.metrics.tool_calls))
        .then_with(|| {
            left.metrics
                .wasted_tool_calls
                .cmp(&right.metrics.wasted_tool_calls)
        })
        .then_with(|| strategy_rank(left.strategy).cmp(&strategy_rank(right.strategy)))
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
    key: &CohortKey,
    code: &str,
    message: impl Into<String>,
) {
    failures.push(AdaptiveReadinessFailure {
        case_id: key.case_id.clone(),
        model: key.model.clone(),
        repetition: 0,
        code: code.into(),
        message: message.into(),
    });
}

fn push_failure_at_repetition(
    failures: &mut Vec<AdaptiveReadinessFailure>,
    key: &CohortKey,
    repetition: u32,
    code: &str,
    message: impl Into<String>,
) {
    failures.push(AdaptiveReadinessFailure {
        case_id: key.case_id.clone(),
        model: key.model.clone(),
        repetition,
        code: code.into(),
        message: message.into(),
    });
}
