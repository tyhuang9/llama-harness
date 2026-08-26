use crate::{
    evaluate_expectations, EvalCase, EvalError, EvalFixture, EvalSuite, EvaluationCaseResult,
    EvaluationReport,
};
use async_trait::async_trait;
use llama_harness_core::RunResult;
use serde_json::Value;
use uuid::Uuid;

/// The application owns fixture construction, tool registration, and agent resolution.
/// Each execution receives an owned fixture clone, so cases and repetitions cannot share
/// mutable fixture state unless an application explicitly chooses to do so.
#[async_trait]
pub trait EvalExecutor: Send + Sync {
    async fn execute(&self, request: EvalExecutionRequest) -> Result<EvalObservation, EvalError>;
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EvalExecutionRequest {
    pub suite_id: String,
    pub agent_id: String,
    pub agent_version: Option<String>,
    pub prompt_version: Option<String>,
    pub prompt_override: Option<String>,
    pub case: EvalCase,
    pub fixture: Option<EvalFixture>,
    pub model: String,
    pub repetition: u32,
}

#[derive(Clone, Debug)]
pub struct EvalObservation {
    pub run: RunResult,
    pub model_calls: u32,
    pub final_state: Option<Value>,
    pub unresolved_items: Option<Value>,
    pub agent_version: Option<String>,
    pub prompt_version: Option<String>,
}

pub async fn evaluate_suite(
    suite: &EvalSuite,
    executor: &dyn EvalExecutor,
    model_overrides: &[String],
    repeat_override: Option<u32>,
) -> Result<EvaluationReport, EvalError> {
    suite.validate()?;
    if repeat_override == Some(0) {
        return Err(EvalError::InvalidSuite(
            "repeat override must be greater than zero".into(),
        ));
    }
    let models = if model_overrides.is_empty() {
        suite.models.clone()
    } else {
        model_overrides.to_vec()
    };
    if models.iter().any(|model| model.trim().is_empty()) {
        return Err(EvalError::InvalidSuite(
            "model overrides must not be empty".into(),
        ));
    }

    let mut results = Vec::new();
    for source_case in &suite.cases {
        let mut case = source_case.clone();
        if case.expected.max_latency_ms.is_none() {
            case.expected.max_latency_ms = suite.defaults.max_latency_ms;
        }
        let repeats = repeat_override
            .or(case.repeat)
            .unwrap_or(suite.defaults.repeat);
        for model in &models {
            let model = if model_overrides.is_empty() {
                case.model.as_ref().unwrap_or(model)
            } else {
                model
            };
            for repetition in 1..=repeats {
                let request = EvalExecutionRequest {
                    suite_id: suite.id.clone(),
                    agent_id: suite.agent.clone(),
                    agent_version: case
                        .agent_version
                        .clone()
                        .or_else(|| suite.agent_version.clone()),
                    prompt_version: case
                        .prompt_version
                        .clone()
                        .or_else(|| suite.prompt_version.clone()),
                    prompt_override: case
                        .prompt_override
                        .clone()
                        .or_else(|| suite.prompt_override.clone()),
                    case: case.clone(),
                    fixture: case.fixture.clone(),
                    model: model.clone(),
                    repetition,
                };
                results.push(execute_case(executor, request).await);
            }
        }
    }

    Ok(EvaluationReport {
        format_version: 1,
        id: Uuid::new_v4().to_string(),
        suite_id: suite.id.clone(),
        suite_version: suite.version,
        results,
    })
}

pub(crate) async fn execute_case(
    executor: &dyn EvalExecutor,
    request: EvalExecutionRequest,
) -> EvaluationCaseResult {
    let suite_id = request.suite_id.clone();
    let case_id = request.case.id.clone();
    let model = request.model.clone();
    let repetition = request.repetition;
    match executor.execute(request.clone()).await {
        Ok(observation) => {
            let failures = evaluate_expectations(&request.case.expected, &observation);
            EvaluationCaseResult {
                suite_id,
                case_id,
                model,
                repetition,
                passed: failures.is_empty(),
                failures,
                run_id: Some(observation.run.id.clone()),
                trace_id: Some(observation.run.trace_id.clone()),
                status: Some(observation.run.status.clone()),
                duration_ms: Some(observation.run.duration_ms),
                model_calls: Some(observation.model_calls),
                tool_calls: Some(observation.run.tool_calls.len() as u32),
                agent_version: observation.agent_version,
                prompt_version: observation.prompt_version,
                final_state: observation.final_state,
                unresolved_items: observation.unresolved_items,
            }
        }
        Err(error) => EvaluationCaseResult {
            suite_id,
            case_id,
            model,
            repetition,
            passed: false,
            failures: vec![crate::AssertionFailure {
                rule: "executor".into(),
                message: error.to_string(),
            }],
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
        },
    }
}
