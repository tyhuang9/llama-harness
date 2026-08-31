use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, MockModelProvider, MockStep},
    AgentDefinition, AgentRunner, ApprovalHandler, ApprovalRecord, HarnessError, InMemoryEventSink,
    ModelCapabilities, ModelResponse, PolicyDecision, PolicyEngine, ProgrammaticConformance,
    ProgrammaticHostConfig, ProviderCapabilityLimits, RunEvent, RunRequest, RunStatus, RunStrategy,
    Tool, ToolCaller, ToolDefinition, ToolRegistry, ToolResult, ToolRisk,
};
use llama_harness_evals::{
    evaluate_suite, load_suite, EvalError, EvalExecutionRequest, EvalExecutor, EvalObservation,
    StrategyMetrics,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};
use tokio_util::sync::CancellationToken;

const SUITE: &str = include_str!("fixtures/programmatic-acceptance.yaml");

#[derive(Default)]
struct State {
    effects: Vec<(String, Value)>,
    approvals: u32,
}

struct FixtureTool {
    definition: ToolDefinition,
    state: Arc<Mutex<State>>,
}

impl FixtureTool {
    fn new(id: &str, read_only: bool, state: Arc<Mutex<State>>) -> Self {
        Self {
            definition: ToolDefinition::new(
                id,
                id,
                "deterministic acceptance fixture tool",
                json!({
                    "type": "object",
                    "properties": {"value": {"type": "integer"}, "fail": {"type": "boolean"}},
                    "required": ["value"],
                    "additionalProperties": false
                }),
            )
            .with_risk(if read_only {
                ToolRisk::Low
            } else {
                ToolRisk::High
            })
            .with_read_only(read_only)
            .with_idempotent(read_only)
            .with_parallel_safe(read_only)
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
impl Tool for FixtureTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(
        &self,
        arguments: Value,
        _: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .effects
            .push((self.definition.id.clone(), arguments.clone()));
        if arguments["fail"] == Value::Bool(true) {
            Ok(ToolResult::failure("fixture partial failure"))
        } else {
            Ok(ToolResult::success(
                json!({"tool": self.definition.id, "value": arguments["value"]}),
            ))
        }
    }
}

struct FixturePolicy;

#[async_trait]
impl PolicyEngine for FixturePolicy {
    async fn decide(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(if tool.id == "write" {
            PolicyDecision::RequireApproval {
                reason: "fixture writes require approval".into(),
            }
        } else {
            PolicyDecision::Allow {
                reason: "fixture reads are allowed".into(),
            }
        })
    }
}

struct FixtureApprovals(Arc<Mutex<State>>);

#[async_trait]
impl ApprovalHandler for FixtureApprovals {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .approvals += 1;
        Ok(ApprovalRecord::new(
            "",
            tool.id.clone(),
            true,
            "fixture approval granted",
        ))
    }
}

#[derive(Default)]
struct AcceptanceExecutor {
    requests: Mutex<Vec<EvalExecutionRequest>>,
}

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

fn calls(tool_ids: &[String], partial_failure: bool) -> Vec<llama_harness_core::ToolCall> {
    tool_ids
        .iter()
        .enumerate()
        .map(|(index, tool_id)| {
            let arguments = if partial_failure && index == 1 {
                json!({"value": index as i64 + 1, "fail": true})
            } else {
                json!({"value": index as i64 + 1})
            };
            llama_harness_core::ToolCall::new(
                format!("fixture-call-{index}"),
                tool_id,
                arguments.to_string(),
            )
        })
        .collect()
}

fn plan(tool_ids: &[String]) -> MockStep {
    final_response(
        json!({
            "strategy": "declarative_plan",
            "plan": {"nodes": tool_ids.iter().enumerate().map(|(index, tool_id)| json!({
                "id": format!("fixture-node-{index}"),
                "tool_id": tool_id,
                "arguments": {"value": index as i64 + 1},
                "depends_on": if index == 0 { Vec::<String>::new() } else { vec![format!("fixture-node-{}", index - 1)] }
            })).collect::<Vec<_>>()}
        })
        .to_string(),
    )
}

fn invoke(name: &str, tool_id: &str, value: Value) -> Value {
    json!({
        "kind": "invoke", "name": name, "tool_id": tool_id,
        "arguments": {"kind": "object", "entries": [{"key": "value", "value": value}]}
    })
}

fn ret(scenario: &str) -> Value {
    json!({"kind": "return", "value": {"kind": "string", "value": scenario}})
}

fn program(scenario: &str, tool_ids: &[String]) -> String {
    let body = match scenario {
        "branch" => vec![
            json!({"kind": "branch", "condition": {"kind": "boolean", "value": true},
                "then_body": [invoke("branch_read", "read", json!({"kind": "integer", "value": 1}))], "else_body": []}),
            ret(scenario),
        ],
        "loop" => vec![
            json!({"kind": "for_each", "item": "item", "max_iterations": 2,
                "collection": {"kind": "array", "items": [{"kind": "integer", "value": 1}, {"kind": "integer", "value": 2}]},
                "body": [invoke("loop_read", "read", json!({"kind": "variable", "name": "item"}))]}),
            ret(scenario),
        ],
        "fanout" => vec![
            json!({"kind": "fan_out", "name": "results", "tool_id": "read", "item": "item", "max_calls": 3,
                "collection": {"kind": "array", "items": [{"kind": "integer", "value": 1}, {"kind": "integer", "value": 2}, {"kind": "integer", "value": 3}]},
                "arguments": {"kind": "object", "entries": [{"key": "value", "value": {"kind": "variable", "name": "item"}}]}}),
            ret(scenario),
        ],
        "filter" => vec![
            invoke(
                "filter_read",
                "read",
                json!({"kind": "integer", "value": 1}),
            ),
            json!({"kind": "filter", "name": "filtered", "item": "item", "max_items": 2,
                "collection": {"kind": "array", "items": [{"kind": "integer", "value": 1}, {"kind": "integer", "value": 2}]},
                "predicate": {"kind": "binary", "operator": "greater_than", "left": {"kind": "variable", "name": "item"}, "right": {"kind": "integer", "value": 1}}}),
            ret(scenario),
        ],
        "reduce-aggregate" => vec![
            invoke("first_read", "read", json!({"kind": "integer", "value": 1})),
            invoke(
                "second_read",
                "read",
                json!({"kind": "integer", "value": 2}),
            ),
            json!({"kind": "reduce", "name": "sum", "item": "item", "accumulator": "acc", "max_items": 2,
                "collection": {"kind": "array", "items": [{"kind": "integer", "value": 1}, {"kind": "integer", "value": 2}]},
                "initial": {"kind": "integer", "value": 0},
                "value": {"kind": "binary", "operator": "add", "left": {"kind": "variable", "name": "acc"}, "right": {"kind": "variable", "name": "item"}}}),
            ret(scenario),
        ],
        "mixed-approval" => vec![
            invoke(
                "read_before_write",
                "read",
                json!({"kind": "integer", "value": 1}),
            ),
            invoke(
                "approved_write",
                "write",
                json!({"kind": "integer", "value": 2}),
            ),
            ret(scenario),
        ],
        "partial-failure" => vec![
            invoke(
                "successful_read",
                "read",
                json!({"kind": "integer", "value": 1}),
            ),
            json!({
                "kind": "invoke", "name": "failed_read", "tool_id": "read",
                "arguments": {"kind": "object", "entries": [
                    {"key": "value", "value": {"kind": "integer", "value": 2}},
                    {"key": "fail", "value": {"kind": "boolean", "value": true}}
                ]}
            }),
            ret(scenario),
        ],
        _ => tool_ids
            .iter()
            .enumerate()
            .map(|(index, tool_id)| {
                invoke(
                    &format!("call_{index}"),
                    tool_id,
                    json!({"kind": "integer", "value": index as i64 + 1}),
                )
            })
            .chain(std::iter::once(ret(scenario)))
            .collect(),
    };
    json!({"version": 1, "body": body}).to_string()
}

fn steps(
    scenario: &str,
    strategy: RunStrategy,
    tool_ids: &[String],
    partial: bool,
) -> Vec<MockStep> {
    match scenario {
        "repair" => vec![
            final_response("not-a-program"),
            final_response(program(scenario, tool_ids)),
            final_response("repair done"),
        ],
        "fallback" => vec![
            final_response("not-a-program"),
            final_response("still-not-a-program"),
            MockStep::Response(
                ModelResponse::new("fixture-model").with_tool_calls(calls(tool_ids, false)),
            ),
            final_response("fallback done"),
        ],
        _ => match strategy {
            RunStrategy::Direct => vec![
                MockStep::Response(
                    ModelResponse::new("fixture-model").with_tool_calls(calls(tool_ids, partial)),
                ),
                final_response(format!("{scenario} done")),
            ],
            RunStrategy::DeclarativePlan => {
                vec![plan(tool_ids), final_response(format!("{scenario} done"))]
            }
            RunStrategy::Programmatic => vec![
                final_response(program(scenario, tool_ids)),
                final_response(format!("{scenario} done")),
            ],
            RunStrategy::Adaptive => unreachable!("the matrix forces each strategy"),
        },
    }
}

#[async_trait]
impl EvalExecutor for AcceptanceExecutor {
    async fn execute(&self, request: EvalExecutionRequest) -> Result<EvalObservation, EvalError> {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(request.clone());
        let data = &request.fixture.as_ref().expect("acceptance fixture").data;
        let scenario = data["scenario"].as_str().expect("fixture scenario");
        let tool_ids = data["tools"]
            .as_array()
            .expect("fixture tools")
            .iter()
            .map(|tool| tool.as_str().expect("tool id").to_owned())
            .collect::<Vec<_>>();
        let partial = data["partial_failure"] == Value::Bool(true);
        let state = Arc::new(Mutex::new(State::default()));
        let mut registry = ToolRegistry::default();
        registry
            .register(Arc::new(FixtureTool::new("read", true, state.clone())))
            .map_err(|error| EvalError::Executor(error.to_string()))?;
        registry
            .register(Arc::new(FixtureTool::new("write", false, state.clone())))
            .map_err(|error| EvalError::Executor(error.to_string()))?;
        let provider = Arc::new(
            MockModelProvider::scripted(steps(scenario, request.strategy, &tool_ids, partial))
                .with_capabilities(capabilities()),
        );
        let events = Arc::new(InMemoryEventSink::default());
        let runner = AgentRunner::builder(provider.clone())
            .tools(registry)
            .policy(Arc::new(FixturePolicy))
            .approvals(Arc::new(FixtureApprovals(state.clone())))
            .event_sink(events)
            .programmatic(ProgrammaticHostConfig::default())
            .build();
        let mut agent = AgentDefinition::new("fixture-agent", "Fixture Agent", "1", &request.model);
        agent.tool_allowlist = vec!["read".into(), "write".into()];
        agent.limits.max_model_calls = match scenario {
            "repair" => 3,
            "fallback" => 4,
            _ => 2,
        };
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
        let expected_status = request
            .case
            .expected
            .status
            .clone()
            .expect("acceptance cases declare a terminal status");
        let state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let observed = state
            .effects
            .iter()
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        let exact = observed == tool_ids;
        let distinct = state
            .effects
            .iter()
            .map(|(id, args)| format!("{id}:{args}"))
            .collect::<BTreeSet<_>>();
        let final_state = json!({
            "scenario": scenario,
            "effects": state.effects.iter().map(|(id, args)| json!({"tool_id": id, "arguments": args})).collect::<Vec<_>>(),
            "approvals": state.approvals,
        });
        let metrics = StrategyMetrics {
            unauthorized_effects: Some(0),
            duplicate_effects: Some((state.effects.len() - distinct.len()) as u32),
            unintended_effects: Some((!exact) as u32),
            task_correct: Some(run.status == expected_status && exact),
            final_state_correct: Some(run.status == expected_status && exact),
            recovery_success: Some(
                !matches!(scenario, "repair" | "fallback") || run.status == RunStatus::Completed,
            ),
            tool_selection_accuracy: Some(if exact { 1.0 } else { 0.0 }),
            input_tokens: None,
            output_tokens: None,
            wasted_tool_calls: Some((!exact) as u32),
        };
        drop(state);
        Ok(EvalObservation::new(run, provider.requests().len() as u32)
            .with_strategy_metrics(metrics)
            .with_final_state(Some(final_state)))
    }
}

#[tokio::test]
async fn executable_programmatic_acceptance_matrix_runs_real_strategies() {
    let suite = load_suite(SUITE, Some("yaml")).unwrap();
    let executor = AcceptanceExecutor::default();
    let report = evaluate_suite(&suite, &executor, &[], None).await.unwrap();
    assert!(
        report.results.iter().all(|result| result.passed),
        "{report:#?}"
    );
    assert!(report
        .results
        .iter()
        .all(|result| result.strategy != RunStrategy::Adaptive));
    assert!(report
        .results
        .iter()
        .all(|result| result.strategy_metrics.passes_readiness()));
}

#[tokio::test]
async fn adaptive_never_selects_programmatic_even_when_advertised() {
    let state = Arc::new(Mutex::new(State::default()));
    let mut registry = ToolRegistry::default();
    registry
        .register(Arc::new(FixtureTool::new("read", true, state)))
        .unwrap();
    let provider = Arc::new(
        MockModelProvider::scripted([
            MockStep::Response(
                ModelResponse::new("fixture-model").with_tool_calls(calls(&["read".into()], false)),
            ),
            final_response("adaptive done"),
        ])
        .with_capabilities(
            ModelCapabilities::new(true, false, true)
                .with_programmatic_conformance(ProgrammaticConformance::StrictJsonAstV1)
                .with_limits(ProviderCapabilityLimits::new().with_max_program_bytes(64 * 1024)),
        ),
    );
    let events = Arc::new(InMemoryEventSink::default());
    let mut agent = AgentDefinition::new("fixture-agent", "Fixture Agent", "1", "fixture-model");
    agent.tool_allowlist = vec!["read".into()];
    agent.limits.max_model_calls = 2;
    let result = AgentRunner::builder(provider)
        .tools(registry)
        .policy(Arc::new(FixturePolicy))
        .event_sink(events.clone())
        .programmatic(ProgrammaticHostConfig::default())
        .build()
        .run(RunRequest::new(agent, "adaptive compatibility"))
        .await
        .unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    assert!(events.events().iter().all(|record| !matches!(
        record.event,
        RunEvent::StrategySelected {
            selected: RunStrategy::Programmatic,
            ..
        }
    )));
}

#[test]
fn programmatic_acceptance_fixture_declares_required_real_scenarios() {
    let suite = load_suite(SUITE, Some("yaml")).unwrap();
    assert_eq!(
        suite
            .cases
            .iter()
            .map(|case| case.fixture.as_ref().unwrap().data["scenario"]
                .as_str()
                .unwrap())
            .collect::<Vec<_>>(),
        vec![
            "branch",
            "loop",
            "fanout",
            "filter",
            "reduce-aggregate",
            "mixed-approval",
            "partial-failure",
            "repair",
            "fallback",
            "capability-downgrade",
        ]
    );
}
