use crate::{
    executor::execute_case, EvalCase, EvalError, EvalExecutionRequest, EvalExecutor,
    EvaluationCaseResult,
};
use llama_harness_core::RunStrategy;
use serde::{Deserialize, Serialize};

/// A replayable, explicit regression input. It deliberately contains no payload read
/// back from a trace database: a trace ID is evidence only, not a fixture source.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// Explicit regression input that can be replayed without trace reconstruction.
pub struct RegressionCase {
    /// Version of the regression case format.
    pub version: u32,
    /// Identifier of the source suite.
    pub source_suite_id: String,
    /// Identifier of the source case.
    pub source_case_id: String,
    /// Identifier of the agent to execute.
    pub agent_id: String,
    #[serde(default)]
    /// Optional agent implementation version.
    pub agent_version: Option<String>,
    #[serde(default)]
    /// Optional prompt version.
    pub prompt_version: Option<String>,
    #[serde(default)]
    /// Optional prompt replacement.
    pub prompt_override: Option<String>,
    /// Model to use for replay.
    pub model: String,
    #[serde(default)]
    /// Trace identifier retained as provenance only.
    pub source_trace_id: Option<String>,
    /// Explicit case and fixture inputs for replay.
    pub case: EvalCase,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Provenance and execution metadata used to create a regression case.
pub struct RegressionSource {
    /// Identifier of the source suite.
    pub suite_id: String,
    /// Identifier of the agent.
    pub agent_id: String,
    /// Optional agent implementation version.
    pub agent_version: Option<String>,
    /// Optional prompt version.
    pub prompt_version: Option<String>,
    /// Optional prompt replacement.
    pub prompt_override: Option<String>,
    /// Model used for the source result.
    pub model: String,
}

impl RegressionCase {
    /// Validates the regression format, provenance, and embedded case.
    pub fn validate(&self) -> Result<(), EvalError> {
        if self.version != 1 {
            return Err(EvalError::InvalidSuite(format!(
                "regression version {} is not supported",
                self.version
            )));
        }
        if self.source_suite_id.trim().is_empty()
            || self.source_case_id.trim().is_empty()
            || self.agent_id.trim().is_empty()
            || self.model.trim().is_empty()
        {
            return Err(EvalError::InvalidSuite(
                "regression source suite, source case, agent, and model are required".into(),
            ));
        }
        if self.case.id != self.source_case_id {
            return Err(EvalError::InvalidSuite(
                "regression source case ID does not match its embedded case".into(),
            ));
        }
        self.case.validate()
    }
}

/// Builds a replayable regression case from an evaluated case result.
pub fn export_regression_case(
    source: RegressionSource,
    case: &EvalCase,
    result: &EvaluationCaseResult,
) -> Result<RegressionCase, EvalError> {
    if result.case_id != case.id {
        return Err(EvalError::InvalidSuite(
            "cannot export regression for a different evaluation case".into(),
        ));
    }
    let regression = RegressionCase {
        version: 1,
        source_suite_id: source.suite_id,
        source_case_id: case.id.clone(),
        agent_id: source.agent_id,
        agent_version: source.agent_version,
        prompt_version: source.prompt_version,
        prompt_override: source.prompt_override,
        model: source.model,
        source_trace_id: result.trace_id.clone(),
        case: case.clone(),
    };
    regression.validate()?;
    Ok(regression)
}

/// Re-executes a validated regression case through an application executor.
pub async fn replay_regression(
    regression: &RegressionCase,
    executor: &dyn EvalExecutor,
) -> Result<EvaluationCaseResult, EvalError> {
    regression.validate()?;
    Ok(execute_case(
        executor,
        EvalExecutionRequest {
            suite_id: regression.source_suite_id.clone(),
            agent_id: regression.agent_id.clone(),
            agent_version: regression.agent_version.clone(),
            prompt_version: regression.prompt_version.clone(),
            prompt_override: regression.prompt_override.clone(),
            case: regression.case.clone(),
            fixture: regression.case.fixture.clone(),
            model: regression.model.clone(),
            strategy: regression.case.strategy.unwrap_or(RunStrategy::Adaptive),
            repetition: 1,
        },
    )
    .await)
}
