//! Provider-neutral, deterministic evaluation and replay contracts.
//!
//! Evaluation execution remains application-owned: this crate provides the suite
//! format, assertion engine, isolated fixture handoff, and normalized results.
//! It never reconstructs a fixture from a trace or stores hidden reasoning.

#![deny(missing_docs)]

mod assertions;
mod executor;
mod replay;
mod result;
mod suite;

pub use assertions::{evaluate_expectations, is_json_subset};
pub use executor::{evaluate_suite, EvalExecutionRequest, EvalExecutor, EvalObservation};
pub use replay::{export_regression_case, replay_regression, RegressionCase, RegressionSource};
pub use result::{
    AdaptiveComparison, AdaptiveReadiness, AdaptiveReadinessFailure, AssertionFailure,
    EvaluationCaseResult, EvaluationReport, StrategyComparisonMetrics, StrategyMetrics,
    StrategyMetricsValidationError,
};
pub use suite::{
    load_suite, load_suite_path, EvalCase, EvalDefaults, EvalExpected, EvalFixture, EvalSuite,
    ExpectedFailure, ExpectedToolCall, SUPPORTED_SUITE_VERSION,
};

use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
/// Errors returned while loading, validating, or executing evaluations.
pub enum EvalError {
    #[error("I/O error: {0}")]
    /// Reading an evaluation suite failed.
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    /// JSON parsing or serialization failed.
    Json(#[from] serde_json::Error),
    #[error("YAML error: {0}")]
    /// YAML parsing failed.
    Yaml(#[from] serde_yaml::Error),
    #[error("invalid evaluation suite: {0}")]
    /// A suite or regression input failed validation.
    InvalidSuite(String),
    #[error("evaluation executor error: {0}")]
    /// An application-owned executor returned an error.
    Executor(String),
}
