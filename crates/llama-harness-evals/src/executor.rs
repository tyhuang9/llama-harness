use crate::{
    evaluate_expectations, EvalCase, EvalError, EvalFixture, EvalSuite, EvaluationCaseResult,
    EvaluationReport, StrategyMetrics,
};
use async_trait::async_trait;
use llama_harness_core::{RunResult, RunStrategy};
use serde_json::Value;
use uuid::Uuid;

/// The application owns fixture construction, tool registration, and agent resolution.
/// Each execution receives an owned fixture clone, so cases and repetitions cannot share
/// mutable fixture state unless an application explicitly chooses to do so.
#[async_trait]
pub trait EvalExecutor: Send + Sync {
    /// Executes one evaluation request and returns its observable result.
    async fn execute(&self, request: EvalExecutionRequest) -> Result<EvalObservation, EvalError>;
}

#[derive(Clone, Debug)]
#[non_exhaustive]
/// Inputs supplied to an application-owned evaluation executor.
pub struct EvalExecutionRequest {
    /// Identifier of the suite being evaluated.
    pub suite_id: String,
    /// Identifier of the agent under test.
    pub agent_id: String,
    /// Optional agent implementation version.
    pub agent_version: Option<String>,
    /// Optional prompt version used for the execution.
    pub prompt_version: Option<String>,
    /// Optional prompt replacement for the case.
    pub prompt_override: Option<String>,
    /// Evaluation case to execute.
    pub case: EvalCase,
    /// Isolated fixture supplied to the application executor.
    pub fixture: Option<EvalFixture>,
    /// Model selected for this execution.
    pub model: String,
    /// Strategy selected for this execution.
    pub strategy: RunStrategy,
    /// One-based repetition number within the evaluation.
    pub repetition: u32,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
/// Observable output returned by an evaluation executor.
pub struct EvalObservation {
    /// Canonical result produced by the agent runner.
    pub run: RunResult,
    /// Number of model calls made during the run.
    pub model_calls: u32,
    /// Executor-owned strategy metrics; leave optional fields unset when unknown.
    pub strategy_metrics: StrategyMetrics,
    /// Optional application-owned final state snapshot.
    pub final_state: Option<Value>,
    /// Optional application-owned unresolved-items snapshot.
    pub unresolved_items: Option<Value>,
    /// Agent version observed by the executor.
    pub agent_version: Option<String>,
    /// Prompt version observed by the executor.
    pub prompt_version: Option<String>,
}

impl EvalObservation {
    /// Creates an observation with unknown strategy metrics and optional state unset.
    ///
    /// This constructor is the stable first-contract entry point; builders allow
    /// future observation fields without requiring downstream struct literals.
    pub fn new(run: RunResult, model_calls: u32) -> Self {
        Self {
            run,
            model_calls,
            strategy_metrics: StrategyMetrics::default(),
            final_state: None,
            unresolved_items: None,
            agent_version: None,
            prompt_version: None,
        }
    }

    /// Sets executor-owned strategy metrics.
    pub fn with_strategy_metrics(mut self, metrics: StrategyMetrics) -> Self {
        self.strategy_metrics = metrics;
        self
    }

    /// Sets an optional application-owned final-state snapshot.
    pub fn with_final_state(mut self, final_state: Option<Value>) -> Self {
        self.final_state = final_state;
        self
    }

    /// Sets an optional application-owned unresolved-items snapshot.
    pub fn with_unresolved_items(mut self, unresolved_items: Option<Value>) -> Self {
        self.unresolved_items = unresolved_items;
        self
    }

    /// Sets the observed agent implementation version.
    pub fn with_agent_version(mut self, agent_version: Option<String>) -> Self {
        self.agent_version = agent_version;
        self
    }

    /// Sets the observed prompt version.
    pub fn with_prompt_version(mut self, prompt_version: Option<String>) -> Self {
        self.prompt_version = prompt_version;
        self
    }
}

/// Executes every suite case for the selected models and repetitions.
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
            let strategies = case
                .strategy
                .map(|strategy| vec![strategy])
                .unwrap_or_else(|| suite.strategies.clone());
            for strategy in strategies {
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
                        strategy,
                        repetition,
                    };
                    results.push(execute_case(executor, request).await);
                }
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
    let strategy = request.strategy;
    let repetition = request.repetition;
    match executor.execute(request.clone()).await {
        Ok(observation) => {
            let failures = evaluate_expectations(&request.case.expected, &observation);
            EvaluationCaseResult {
                suite_id,
                case_id,
                model,
                strategy,
                repetition,
                passed: failures.is_empty(),
                failures,
                run_id: Some(observation.run.id.clone()),
                trace_id: Some(observation.run.trace_id.clone()),
                status: Some(observation.run.status.clone()),
                duration_ms: Some(observation.run.duration_ms),
                model_calls: Some(observation.model_calls),
                tool_calls: Some(observation.run.tool_calls.len() as u32),
                strategy_metrics: observation.strategy_metrics,
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
            strategy,
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
            strategy_metrics: StrategyMetrics::default(),
            agent_version: None,
            prompt_version: None,
            final_state: None,
            unresolved_items: None,
        },
    }
}
