use crate::{
    executor::execute_case, EvalCase, EvalError, EvalExecutionRequest, EvalExecutor,
    EvaluationCaseResult,
};
use serde::{Deserialize, Serialize};

/// A replayable, explicit regression input. It deliberately contains no payload read
/// back from a trace database: a trace ID is evidence only, not a fixture source.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RegressionCase {
    pub version: u32,
    pub source_suite_id: String,
    pub source_case_id: String,
    pub agent_id: String,
    #[serde(default)]
    pub agent_version: Option<String>,
    #[serde(default)]
    pub prompt_version: Option<String>,
    #[serde(default)]
    pub prompt_override: Option<String>,
    pub model: String,
    #[serde(default)]
    pub source_trace_id: Option<String>,
    pub case: EvalCase,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegressionSource {
    pub suite_id: String,
    pub agent_id: String,
    pub agent_version: Option<String>,
    pub prompt_version: Option<String>,
    pub prompt_override: Option<String>,
    pub model: String,
}

impl RegressionCase {
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
            repetition: 1,
        },
    )
    .await)
}
