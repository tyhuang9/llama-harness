use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, MockModelProvider, MockStep},
    AgentDefinition, AgentRunner, HarnessError, InMemoryEventSink, ModelCapabilities,
    ModelResponse, PolicyDecision, PolicyEngine, ProgrammaticConformance, ProgrammaticHostConfig,
    ProviderCapabilityLimits, RunEvent, RunRequest, RunStatus, RunStrategy, Tool, ToolCall,
    ToolCaller, ToolDefinition, ToolRegistry, ToolResult, ToolRisk,
};
use llama_harness_evals::{
    evaluate_suite, load_suite, EvalError, EvalExecutionRequest, EvalExecutor, EvalObservation,
    EvaluationReport, StrategyMetrics,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};
use tokio_util::sync::CancellationToken;

const MANIFEST: &str = include_str!("fixtures/acceptance-matrix.yaml");

// The compact runner matrix below is deliberately small. The manifest audit
// consolidates the deeper DAG, sandbox, catalog, and speculation coverage that
// is already exercised through their real production boundaries elsewhere.
const RUNNER_SUITE: &str = r#"
version: 1
id: compatibility-runner-matrix
name: Compatibility Runner Matrix
agent: matrix-agent
models: [matrix-model]
strategies: [direct, declarative_plan, programmatic, adaptive]
defaults: {repeat: 1}
cases:
  - id: no-tool-direct
    strategy: direct
    fixture: {id: no-tool, data: {scenario: no-tool, tools: []}}
    input: Answer without tools.
    expected: {status: completed, final_output_contains: [done], tool_sequence: [], max_model_calls: 3, max_tool_calls: 0}
  - id: no-tool-programmatic
    strategy: programmatic
    fixture: {id: no-tool, data: {scenario: no-tool, tools: []}}
    input: Answer without tools.
    expected: {status: completed, final_output_contains: [done], tool_sequence: [], max_model_calls: 3, max_tool_calls: 0}
  - id: no-tool-adaptive
    strategy: adaptive
    fixture: {id: no-tool, data: {scenario: no-tool, tools: []}}
    input: Answer without tools.
    expected: {status: completed, final_output_contains: [done], tool_sequence: [], max_model_calls: 3, max_tool_calls: 0}
  - id: single-call
    fixture: {id: single-call, data: {scenario: single-call, tools: [read]}}
    input: Read one deterministic value.
    expected: {status: completed, final_output_contains: [done], required_tools: [read], tool_sequence: [read], expected_tool_arguments: [{tool_id: read, arguments_subset: {}}], max_model_calls: 3, max_tool_calls: 1}
"#;

const ADAPTIVE_RUNNER: &str = include_str!("../../llama-harness-core/tests/adaptive_runner.rs");
const PROGRAMMATIC_RUNNER: &str =
    include_str!("../../llama-harness-core/tests/programmatic_runner.rs");
const SPECULATION_RUNNER: &str =
    include_str!("../../llama-harness-core/tests/speculation_runner.rs");
const DEFERRED_DISCOVERY: &str =
    include_str!("../../llama-harness-core/tests/deferred_discovery.rs");
const DIRECT_RUNNER: &str = include_str!("../../llama-harness-core/tests/agent_runner.rs");
const PROGRAMMATIC_ACCEPTANCE: &str = include_str!("fixtures/programmatic-acceptance.yaml");
const SPECULATION_ACCEPTANCE: &str = include_str!("fixtures/speculation-acceptance.yaml");
const SPECULATION_ACCEPTANCE_TEST: &str = include_str!("speculation_acceptance.rs");

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AcceptanceManifest {
    version: u32,
    id: String,
    name: String,
    execution_boundary: String,
    quality_ranking: QualityRanking,
    workloads: Vec<ManifestWorkload>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualityRanking {
    hard_gates: Vec<String>,
    rank_after_hard_gates: Vec<String>,
    latency: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWorkload {
    id: String,
    strategies: Vec<RunStrategy>,
    evidence: Vec<ManifestEvidence>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestEvidence {
    source: String,
    marker: String,
}

#[derive(Default)]
struct State {
    effects: Vec<String>,
    policy_calls: Vec<String>,
}

struct MatrixTool {
    definition: ToolDefinition,
    state: Arc<Mutex<State>>,
}

impl MatrixTool {
    fn new(state: Arc<Mutex<State>>) -> Self {
        Self {
            definition: ToolDefinition::new(
                "read",
                "Read",
                "deterministic read fixture through the production broker",
                json!({"type": "object", "additionalProperties": false}),
            )
            .with_risk(ToolRisk::Low)
            .with_read_only(true)
            .with_idempotent(true)
            .with_parallel_safe(true)
            .with_allowed_callers([
                ToolCaller::Direct,
                ToolCaller::DeclarativePlan,
                ToolCaller::Programmatic,
            ]),
            state,
        }
    }
}

#[async_trait]
impl Tool for MatrixTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .effects
            .push(self.definition.id.clone());
        Ok(ToolResult::success(json!({"value": "read"})))
    }
}

struct MatrixPolicy(Arc<Mutex<State>>);

#[async_trait]
impl PolicyEngine for MatrixPolicy {
    async fn decide(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .policy_calls
            .push(tool.id.clone());
        Ok(PolicyDecision::Allow {
            reason: "deterministic acceptance read".into(),
        })
    }
}

#[derive(Default)]
struct RunnerMatrixExecutor;

fn capabilities() -> ModelCapabilities {
    ModelCapabilities::new(true, false, true)
        .with_parallel_tool_calls(true)
        .with_structured_plans(true)
        .with_programmatic_conformance(ProgrammaticConformance::StrictJsonAstV1)
        .with_limits(
            ProviderCapabilityLimits::new()
                .with_max_program_bytes(64 * 1024)
                .with_max_parallel_tool_calls(8)
                .with_max_plan_nodes(32)
                .with_max_plan_bytes(64 * 1024),
        )
}

fn program() -> String {
    json!({
        "version": 1,
        "body": [
            {
                "kind": "invoke",
                "name": "read",
                "tool_id": "read",
                "arguments": {"kind": "object", "entries": []}
            },
            {"kind": "return", "value": {"kind": "string", "value": "read"}}
        ]
    })
    .to_string()
}

fn no_tool_program() -> String {
    json!({
        "version": 1,
        "body": [{"kind": "return", "value": {"kind": "string", "value": "done"}}]
    })
    .to_string()
}

fn plan(with_read: bool) -> String {
    let nodes = if with_read {
        vec![json!({
            "id": "read",
            "tool_id": "read",
            "arguments": {},
            "depends_on": []
        })]
    } else {
        Vec::new()
    };
    json!({"strategy": "declarative_plan", "plan": {"nodes": nodes}}).to_string()
}

fn steps(scenario: &str, strategy: RunStrategy) -> Vec<MockStep> {
    let with_read = scenario == "single-call";
    let direct = || {
        if with_read {
            vec![
                MockStep::Response(
                    ModelResponse::new("matrix-model")
                        .with_tool_calls(vec![ToolCall::new("read-1", "read", "{}")]),
                ),
                final_response("done"),
            ]
        } else {
            vec![final_response("done")]
        }
    };
    match strategy {
        RunStrategy::Direct => direct(),
        RunStrategy::DeclarativePlan => {
            vec![final_response(plan(with_read)), final_response("done")]
        }
        RunStrategy::Programmatic => vec![
            final_response(if with_read {
                program()
            } else {
                no_tool_program()
            }),
            final_response("done"),
        ],
        RunStrategy::Adaptive => {
            let mut values = vec![final_response(r#"{"strategy":"direct"}"#)];
            values.extend(direct());
            values
        }
    }
}

fn evidence_source(source: &str) -> Option<&'static str> {
    match source {
        "adaptive_runner" => Some(ADAPTIVE_RUNNER),
        "programmatic_runner" => Some(PROGRAMMATIC_RUNNER),
        "speculation_runner" => Some(SPECULATION_RUNNER),
        "deferred_discovery" => Some(DEFERRED_DISCOVERY),
        "direct_runner" => Some(DIRECT_RUNNER),
        "programmatic_acceptance_fixture" => Some(PROGRAMMATIC_ACCEPTANCE),
        "speculation_acceptance_fixture" => Some(SPECULATION_ACCEPTANCE),
        "speculation_acceptance_test" => Some(SPECULATION_ACCEPTANCE_TEST),
        _ => None,
    }
}

#[async_trait]
impl EvalExecutor for RunnerMatrixExecutor {
    async fn execute(&self, request: EvalExecutionRequest) -> Result<EvalObservation, EvalError> {
        let fixture = request.fixture.as_ref().expect("matrix fixture");
        let scenario = fixture.data["scenario"].as_str().expect("scenario");
        let expected_effects = fixture.data["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .map(|value| value.as_str().expect("tool id").to_owned())
            .collect::<Vec<_>>();
        let state = Arc::new(Mutex::new(State::default()));
        let mut registry = ToolRegistry::default();
        registry
            .register(Arc::new(MatrixTool::new(state.clone())))
            .map_err(|error| EvalError::Executor(error.to_string()))?;
        let events = Arc::new(InMemoryEventSink::default());
        let provider = Arc::new(
            MockModelProvider::scripted(steps(scenario, request.strategy))
                .with_capabilities(capabilities()),
        );
        let runner = AgentRunner::builder(provider.clone())
            .tools(registry)
            .policy(Arc::new(MatrixPolicy(state.clone())))
            .event_sink(events.clone())
            .programmatic(ProgrammaticHostConfig::default())
            .build();
        let mut agent = AgentDefinition::new("matrix-agent", "Matrix Agent", "1", &request.model);
        agent.tool_allowlist = vec!["read".into()];
        agent.limits.max_model_calls = 3;
        let run = runner
            .run_with_strategy(
                RunRequest::new(agent, request.case.input.clone()).with_run_id(format!(
                    "{}-{:?}-{}",
                    request.case.id, request.strategy, request.repetition
                )),
                request.strategy,
            )
            .await
            .map_err(|error| EvalError::Executor(error.to_string()))?;

        let selected = events
            .events()
            .iter()
            .rev()
            .find_map(|record| match record.event {
                RunEvent::StrategySelected { selected, .. } => Some(selected),
                _ => None,
            });
        let expected_selected = if request.strategy == RunStrategy::Adaptive {
            RunStrategy::Direct
        } else {
            request.strategy
        };
        if selected != Some(expected_selected) {
            return Err(EvalError::Executor(format!(
                "expected {expected_selected:?} strategy selection, observed {selected:?}"
            )));
        }
        if request.strategy == RunStrategy::Adaptive && selected == Some(RunStrategy::Programmatic)
        {
            return Err(EvalError::Executor(
                "Adaptive selected Programmatic in a compatibility workload".into(),
            ));
        }

        let state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let exact_effects = state.effects == expected_effects;
        let distinct_effects = state.effects.iter().collect::<BTreeSet<_>>().len();
        let unauthorized_effects = state
            .effects
            .iter()
            .filter(|effect| !state.policy_calls.contains(effect))
            .count() as u32;
        if run
            .tool_calls
            .iter()
            .map(|call| &call.tool_id)
            .collect::<Vec<_>>()
            != expected_effects.iter().collect::<Vec<_>>()
        {
            return Err(EvalError::Executor(
                "broker tool-call order diverged from the fixture order".into(),
            ));
        }
        let metrics = StrategyMetrics {
            unauthorized_effects: Some(unauthorized_effects),
            duplicate_effects: Some((state.effects.len() - distinct_effects) as u32),
            unintended_effects: Some((!exact_effects) as u32),
            task_correct: Some(run.status == RunStatus::Completed && exact_effects),
            final_state_correct: Some(run.status == RunStatus::Completed && exact_effects),
            recovery_success: Some(true),
            tool_selection_accuracy: Some(if exact_effects { 1.0 } else { 0.0 }),
            // The mock provider reports no token accounting; zero is the exact
            // known fixture value rather than a fabricated estimate.
            input_tokens: Some(0),
            output_tokens: Some(0),
            wasted_tool_calls: Some((!exact_effects) as u32),
        };
        let final_state = json!({"effects": state.effects});
        drop(state);
        Ok(EvalObservation::new(run, provider.requests().len() as u32)
            .with_strategy_metrics(metrics)
            .with_final_state(Some(final_state)))
    }
}

#[tokio::test]
async fn cross_strategy_runner_matrix_uses_real_agent_runner_and_broker() {
    let suite = load_suite(RUNNER_SUITE, Some("yaml")).expect("valid runner matrix");
    let report = evaluate_suite(&suite, &RunnerMatrixExecutor, &[], None)
        .await
        .expect("matrix executes");
    assert!(
        report.results.iter().all(|result| result.passed),
        "{report:#?}"
    );
    assert!(report.results.iter().all(|result| {
        result.strategy_metrics.unauthorized_effects == Some(0)
            && result.strategy_metrics.duplicate_effects == Some(0)
            && result.strategy_metrics.unintended_effects == Some(0)
            && result.strategy_metrics.task_correct == Some(true)
            && result.strategy_metrics.final_state_correct == Some(true)
            && result.strategy_metrics.recovery_success == Some(true)
    }));
    let readiness = EvaluationReport::new(
        "cross-strategy-readiness",
        report.suite_id.clone(),
        report.suite_version,
        report
            .results
            .iter()
            .filter(|result| result.case_id == "single-call")
            .cloned()
            .collect(),
    )
    .adaptive_readiness();
    assert!(readiness.ready, "{readiness:#?}");
    assert_eq!(readiness.comparisons.len(), 1);
    // Durations are preserved in the normalized report for release analysis,
    // but no wall-clock threshold or expected winner is asserted here.
    assert!(report
        .results
        .iter()
        .all(|result| result.duration_ms.is_some()));
}

#[test]
fn acceptance_manifest_audits_every_required_workload_against_executable_coverage() {
    let manifest: AcceptanceManifest = serde_yaml::from_str(MANIFEST).expect("valid manifest");
    assert_eq!(manifest.version, 1);
    assert_eq!(manifest.id, "compatibility-acceptance-matrix");
    assert_eq!(manifest.name, "Compatibility Release Acceptance Matrix");
    assert_eq!(
        manifest.execution_boundary,
        "real_agent_runner_broker_sandbox_and_speculation"
    );
    assert_eq!(
        manifest.quality_ranking.latency,
        "measured_and_reported_only"
    );
    assert_eq!(
        manifest.quality_ranking.hard_gates,
        vec![
            "unauthorized_effects_zero",
            "duplicate_effects_zero",
            "unintended_effects_zero",
            "task_correct",
            "final_state_correct",
            "recovery_success",
            "deterministic_accounting_and_order",
        ]
    );
    assert_eq!(
        manifest.quality_ranking.rank_after_hard_gates,
        vec![
            "tool_selection_accuracy",
            "model_calls",
            "tool_calls",
            "wasted_tool_calls",
        ]
    );

    let required = BTreeSet::from([
        "no-tool",
        "ambiguous-uncertain-direct",
        "single-call",
        "independent-parallel",
        "dependent-dag",
        "partial-failure-recovery",
        "approval",
        "mixed-read-write",
        "loop",
        "fan-out",
        "aggregation",
        "catalog-30",
        "catalog-100",
        "catalog-1000",
        "provider-downgrade",
        "malformed-plan-repair-and-fallback",
        "sandbox-safety-categories",
        "speculation-hit",
        "speculation-miss",
        "speculation-race",
        "speculation-privacy",
        "speculation-exact",
        "speculation-no-writes",
    ]);
    let discovered = manifest
        .workloads
        .iter()
        .map(|workload| workload.id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(discovered, required);

    for workload in &manifest.workloads {
        assert!(
            !workload.strategies.is_empty(),
            "{} has no forced or Adaptive strategy coverage",
            workload.id
        );
        assert!(
            !workload.evidence.is_empty(),
            "{} has no executable coverage evidence",
            workload.id
        );
        for evidence in &workload.evidence {
            let source = evidence_source(&evidence.source).unwrap_or_else(|| {
                panic!(
                    "{} declares an unknown executable evidence source: {}",
                    workload.id, evidence.source
                )
            });
            assert!(
                source.contains(&evidence.marker),
                "{} evidence is not discovered in {}: {}",
                workload.id,
                evidence.source,
                evidence.marker,
            );
        }
    }

    let by_id = |id: &str| {
        manifest
            .workloads
            .iter()
            .find(|workload| workload.id == id)
            .expect("required workload")
    };
    assert_eq!(
        by_id("no-tool").strategies,
        vec![
            RunStrategy::Direct,
            RunStrategy::Programmatic,
            RunStrategy::Adaptive,
        ],
        "no-tool omits DeclarativePlan because an empty plan is invalid"
    );
    for id in ["single-call"] {
        assert_eq!(
            by_id(id).strategies,
            vec![
                RunStrategy::Direct,
                RunStrategy::DeclarativePlan,
                RunStrategy::Programmatic,
                RunStrategy::Adaptive,
            ],
            "{id} is the executable cross-strategy baseline"
        );
    }
    assert!(by_id("speculation-no-writes")
        .strategies
        .contains(&RunStrategy::Direct));
}
