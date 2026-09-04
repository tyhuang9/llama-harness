use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, MockModelProvider, MockStep},
    AgentDefinition, AgentRunner, ApprovalHandler, ApprovalRecord, HarnessError, InMemoryEventSink,
    ModelCapabilities, ModelResponse, PolicyDecision, PolicyEngine, ProgrammaticConformance,
    ProgrammaticHostConfig, ProgrammaticWorkloadClass, ProviderCapabilityLimits, RunEvent,
    RunRequest, RunStatus, RunStrategy, Tool, ToolCaller, ToolDefinition, ToolRegistry, ToolResult,
    ToolRisk,
};
use llama_harness_evals::{
    evaluate_suite, load_suite, EvalError, EvalExecutionRequest, EvalExecutor, EvalObservation,
    EvaluationReport, ForcedCandidateDisposition, StrategyMetrics,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};
use tokio_util::sync::CancellationToken;

const SUITE: &str = include_str!("fixtures/programmatic-acceptance.yaml");
const ADVANCED_WORKLOADS: [&str; 5] = [
    "loop",
    "fanout",
    "filter",
    "reduce-aggregate",
    "large-intermediate-data",
];

#[derive(Default)]
struct State {
    effects: Vec<(String, Value)>,
    policy_calls: Vec<String>,
    approval_calls: Vec<(String, bool)>,
    approvals: u32,
    result_payload_bytes: u64,
}

struct FixtureTool {
    definition: ToolDefinition,
    state: Arc<Mutex<State>>,
    response_payload_bytes: usize,
}

impl FixtureTool {
    fn new(
        id: &str,
        read_only: bool,
        state: Arc<Mutex<State>>,
        response_payload_bytes: usize,
    ) -> Self {
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
            response_payload_bytes,
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
        let failed = arguments["fail"] == Value::Bool(true);
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state
                .effects
                .push((self.definition.id.clone(), arguments.clone()));
            if !failed {
                state.result_payload_bytes = state
                    .result_payload_bytes
                    .saturating_add(self.response_payload_bytes as u64);
            }
        }
        if failed {
            Ok(ToolResult::failure("fixture partial failure"))
        } else if self.response_payload_bytes == 0 {
            Ok(ToolResult::success(
                json!({"tool": self.definition.id, "value": arguments["value"]}),
            ))
        } else {
            Ok(ToolResult::success(json!({
                "tool": self.definition.id,
                "value": arguments["value"],
                "payload": "x".repeat(self.response_payload_bytes),
            })))
        }
    }
}

struct FixturePolicy(Arc<Mutex<State>>);

#[async_trait]
impl PolicyEngine for FixturePolicy {
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
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .approval_calls
            .push((tool.id.clone(), true));
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

fn fixture_workload_class(data: &Value) -> Result<Option<ProgrammaticWorkloadClass>, EvalError> {
    match data.get("workload_class").and_then(Value::as_str) {
        None => Ok(None),
        Some("loop") => Ok(Some(ProgrammaticWorkloadClass::Loop)),
        Some("fan_out") => Ok(Some(ProgrammaticWorkloadClass::FanOut)),
        Some("filter") => Ok(Some(ProgrammaticWorkloadClass::Filter)),
        Some("aggregation") => Ok(Some(ProgrammaticWorkloadClass::Aggregation)),
        Some("large_intermediate_data") => {
            Ok(Some(ProgrammaticWorkloadClass::LargeIntermediateData))
        }
        Some(other) => Err(EvalError::Executor(format!(
            "unknown fixture Programmatic workload class: {other}"
        ))),
    }
}

fn expected_selected_strategy(
    data: &Value,
    requested_strategy: RunStrategy,
) -> Result<RunStrategy, EvalError> {
    if requested_strategy != RunStrategy::Adaptive {
        if data["expected_forced_selection"].as_str() == Some("direct") {
            return Ok(RunStrategy::Direct);
        }
        return Ok(requested_strategy);
    }

    match data
        .get("expected_adaptive_selection")
        .and_then(Value::as_str)
    {
        Some("direct") => Ok(RunStrategy::Direct),
        Some("programmatic") => Ok(RunStrategy::Programmatic),
        Some(other) => Err(EvalError::Executor(format!(
            "unknown expected Adaptive selection: {other}"
        ))),
        None => Err(EvalError::Executor(
            "Adaptive fixtures must declare expected_adaptive_selection".into(),
        )),
    }
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
        "large-intermediate-data" => vec![
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
    workload_class: Option<&str>,
    programmatic_available: bool,
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
            RunStrategy::Adaptive => match workload_class {
                Some(workload_class) => {
                    let mut responses = vec![final_response(
                        json!({
                            "strategy": "programmatic",
                            "workload_class": workload_class,
                        })
                        .to_string(),
                    )];
                    if programmatic_available {
                        responses.push(final_response(program(scenario, tool_ids)));
                    } else {
                        responses.push(MockStep::Response(
                            ModelResponse::new("fixture-model")
                                .with_tool_calls(calls(tool_ids, partial)),
                        ));
                    }
                    responses.push(final_response(format!("{scenario} done")));
                    responses
                }
                None => vec![
                    final_response(r#"{"strategy":"direct"}"#),
                    MockStep::Response(
                        ModelResponse::new("fixture-model")
                            .with_tool_calls(calls(tool_ids, partial)),
                    ),
                    final_response(format!("{scenario} done")),
                ],
            },
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
        let programmatic_available = data["programmatic_available"].as_bool().unwrap_or(true);
        let response_payload_bytes = data["response_payload_bytes"]
            .as_u64()
            .map(usize::try_from)
            .transpose()
            .map_err(|_| EvalError::Executor("fixture payload does not fit usize".into()))?
            .unwrap_or(0);
        let workload_class = fixture_workload_class(data)?;
        let state = Arc::new(Mutex::new(State::default()));
        let mut registry = ToolRegistry::default();
        registry
            .register(Arc::new(FixtureTool::new(
                "read",
                true,
                state.clone(),
                response_payload_bytes,
            )))
            .map_err(|error| EvalError::Executor(error.to_string()))?;
        registry
            .register(Arc::new(FixtureTool::new("write", false, state.clone(), 0)))
            .map_err(|error| EvalError::Executor(error.to_string()))?;
        let provider = Arc::new(
            MockModelProvider::scripted(steps(
                scenario,
                request.strategy,
                &tool_ids,
                partial,
                data["workload_class"].as_str(),
                programmatic_available,
            ))
            .with_capabilities(capabilities().with_programmatic_calling(programmatic_available)),
        );
        let events = Arc::new(InMemoryEventSink::default());
        let runner_builder = AgentRunner::builder(provider.clone())
            .tools(registry)
            .policy(Arc::new(FixturePolicy(state.clone())))
            .approvals(Arc::new(FixtureApprovals(state.clone())))
            .event_sink(events.clone())
            .programmatic(ProgrammaticHostConfig::default());
        let runner = match workload_class {
            Some(workload_class) => runner_builder
                .adaptive_programmatic_allowlist([workload_class])
                .build(),
            None => runner_builder.build(),
        };
        let mut agent = AgentDefinition::new("fixture-agent", "Fixture Agent", "1", &request.model);
        agent.tool_allowlist = vec!["read".into(), "write".into()];
        agent.limits.max_model_calls = match scenario {
            "repair" => 3,
            "fallback" => 4,
            _ if request.strategy == RunStrategy::Adaptive => 3,
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
        let selected_strategy =
            events
                .events()
                .iter()
                .rev()
                .find_map(|record| match &record.event {
                    RunEvent::StrategySelected { selected, .. } => Some(*selected),
                    _ => None,
                });
        let expected_selected_strategy = expected_selected_strategy(data, request.strategy)?;
        if selected_strategy != Some(expected_selected_strategy) {
            return Err(EvalError::Executor(
                format!(
                    "expected {expected_selected_strategy:?} strategy selection, observed {selected_strategy:?}"
                ),
            ));
        }
        let provider_requests = provider.requests();
        if scenario == "large-intermediate-data"
            && expected_selected_strategy == RunStrategy::Programmatic
        {
            let synthesis_input = &provider_requests
                .last()
                .and_then(|model_request| model_request.messages.last())
                .ok_or_else(|| {
                    EvalError::Executor(
                        "programmatic large-intermediate fixture omitted synthesis input".into(),
                    )
                })?
                .content;
            let raw_result_bytes = response_payload_bytes
                .checked_mul(tool_ids.len())
                .ok_or_else(|| EvalError::Executor("fixture byte count overflowed".into()))?;
            if synthesis_input.len() >= raw_result_bytes / 100 {
                return Err(EvalError::Executor(format!(
                    "programmatic synthesis retained {} bytes for {raw_result_bytes} raw result bytes",
                    synthesis_input.len()
                )));
            }
            if synthesis_input.contains("\"payload\"") {
                return Err(EvalError::Executor(
                    "programmatic synthesis reinjected raw tool-result payloads".into(),
                ));
            }
        }
        for (index, call) in run.tool_calls.iter().enumerate() {
            let expected_arguments = if partial && index == 1 {
                json!({"value": index as i64 + 1, "fail": true})
            } else {
                json!({"value": index as i64 + 1})
            };
            let expected_arguments_json = expected_arguments.to_string();
            if call.arguments_json != expected_arguments_json {
                return Err(EvalError::Executor(
                    "runner did not retain canonical tool arguments".into(),
                ));
            }
        }
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
            "result_payload_bytes": state.result_payload_bytes,
        });
        let unauthorized_effects = state
            .effects
            .iter()
            .filter(|(tool_id, _)| {
                let policy_authorized = state
                    .policy_calls
                    .iter()
                    .any(|policy_tool| policy_tool == tool_id);
                let approval_authorized = tool_id != "write"
                    || state
                        .approval_calls
                        .iter()
                        .any(|(approval_tool, granted)| approval_tool == tool_id && *granted);
                !policy_authorized || !approval_authorized
            })
            .count() as u32;
        let metrics = StrategyMetrics {
            unauthorized_effects: Some(unauthorized_effects),
            duplicate_effects: Some((state.effects.len() - distinct.len()) as u32),
            unintended_effects: Some((!exact) as u32),
            task_correct: Some(run.status == expected_status && exact),
            final_state_correct: Some(run.status == expected_status && exact),
            recovery_success: Some(
                !matches!(scenario, "repair" | "fallback") || run.status == RunStatus::Completed,
            ),
            tool_selection_accuracy: Some(if exact { 1.0 } else { 0.0 }),
            // The mock provider reports no token usage. Zero is the exact
            // fixture counter, not a production estimate.
            input_tokens: Some(0),
            output_tokens: Some(0),
            wasted_tool_calls: Some((!exact) as u32),
        };
        drop(state);
        Ok(EvalObservation::new(run, provider_requests.len() as u32)
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
        .any(|result| result.strategy == RunStrategy::Adaptive));
    assert!(report
        .results
        .iter()
        .all(|result| result.strategy_metrics.passes_readiness()));
    for case_id in ADVANCED_WORKLOADS {
        for repetition in 1..=3 {
            let adaptive = report
                .results
                .iter()
                .find(|result| {
                    result.case_id == case_id
                        && result.strategy == RunStrategy::Adaptive
                        && result.repetition == repetition
                })
                .expect("advanced workload has an Adaptive result");
            for forced_strategy in [
                RunStrategy::Direct,
                RunStrategy::DeclarativePlan,
                RunStrategy::Programmatic,
            ] {
                let forced = report
                    .results
                    .iter()
                    .find(|result| {
                        result.case_id == case_id
                            && result.strategy == forced_strategy
                            && result.repetition == repetition
                    })
                    .expect("advanced workload has every forced comparison");
                assert_eq!(
                    adaptive.final_state, forced.final_state,
                    "Adaptive final state diverged from {forced_strategy:?} for {case_id}, repetition {repetition}"
                );
            }
        }
    }

    // This validates deterministic cohort safety/correctness/ranking plumbing,
    // not a real latency advantage. Production's default promotion allowlist
    // remains empty; this fixture opts in only for acceptance coverage.
    let advanced_report = EvaluationReport::new(
        "programmatic-advanced-cohort-readiness",
        report.suite_id.clone(),
        report.suite_version,
        report
            .results
            .iter()
            .filter(|result| ADVANCED_WORKLOADS.contains(&result.case_id.as_str()))
            .cloned()
            .collect(),
    );
    let readiness = advanced_report.adaptive_readiness();
    assert!(readiness.ready, "{readiness:#?}");
    assert!(readiness.failures.is_empty(), "{readiness:#?}");
    assert_eq!(readiness.comparisons.len(), ADVANCED_WORKLOADS.len());
    for comparison in &readiness.comparisons {
        assert_eq!(comparison.sample_count, 3, "{comparison:#?}");
        assert_eq!(
            comparison
                .forced_candidates
                .iter()
                .map(|candidate| candidate.strategy)
                .collect::<Vec<_>>(),
            vec![
                RunStrategy::Direct,
                RunStrategy::DeclarativePlan,
                RunStrategy::Programmatic,
            ],
            "{comparison:#?}"
        );
        assert!(comparison.forced_candidates.iter().all(|candidate| {
            matches!(
                candidate.disposition,
                ForcedCandidateDisposition::Selected | ForcedCandidateDisposition::Outranked { .. }
            )
        }));
    }
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
            "large-intermediate-data",
            "mixed-approval",
            "partial-failure",
            "repair",
            "fallback",
            "capability-downgrade",
        ]
    );

    let adaptive_expectations = suite
        .cases
        .iter()
        .filter(|case| case.strategy != Some(RunStrategy::Programmatic))
        .map(|case| {
            let data = &case.fixture.as_ref().expect("fixture").data;
            (
                data["scenario"].as_str().expect("scenario"),
                data["workload_class"].as_str(),
                data["expected_adaptive_selection"]
                    .as_str()
                    .expect("Adaptive selection"),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        adaptive_expectations,
        vec![
            ("branch", None, "direct"),
            ("loop", Some("loop"), "programmatic"),
            ("fanout", Some("fan_out"), "programmatic"),
            ("filter", Some("filter"), "programmatic"),
            ("reduce-aggregate", Some("aggregation"), "programmatic"),
            (
                "large-intermediate-data",
                Some("large_intermediate_data"),
                "programmatic",
            ),
            ("mixed-approval", None, "direct"),
            ("capability-downgrade", Some("loop"), "direct"),
        ]
    );
}
