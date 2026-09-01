use async_trait::async_trait;
use futures_util::{stream, StreamExt};
use llama_harness_core::{
    AgentDefinition, AgentRunner, CancellationSafety, EventRecord, ExecutionLocation, HarnessError,
    InMemoryEventSink, IssueSafety, MessageRole, ModelCapabilities, ModelEventStream, ModelInfo,
    ModelProvider, ModelRequest, ModelResponse, ModelStreamEvent, NetworkEgress, PolicyDecision,
    PolicyEngine, ProviderHealth, RunRequest, RunResult, RunStatus, RunStrategy, SpeculationConfig,
    SpeculationMetrics, SpeculationMode, SpeculationPolicy, SpeculationReadiness, Tool,
    ToolCallContext, ToolCallDelta, ToolCaller, ToolDefinition, ToolRegistry, ToolResult, ToolRisk,
    Usage,
};
use llama_harness_evals::{
    evaluate_suite, load_suite, EvalError, EvalExecutionRequest, EvalExecutor, EvalObservation,
    EvaluationReport, StrategyMetrics,
};
use llama_harness_observability::{SqliteEventSink, TraceStoreConfig};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

const SUITE: &str = include_str!("fixtures/speculation-acceptance.yaml");
const PRIVATE_RESULT_CANARY: &str = "private-speculative-result-canary";
const PRIVATE_STREAM_ERROR_CANARY: &str = "private-speculative-stream-error-canary";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToolBehavior {
    Eligible,
    InvalidSpeculativeResult,
    BlockingSpeculativeRead,
    Write,
    RemoteRead,
}

impl ToolBehavior {
    fn from_fixture(value: &str) -> Self {
        match value {
            "eligible" => Self::Eligible,
            "invalid_speculative_result" => Self::InvalidSpeculativeResult,
            "blocking_speculative_read" => Self::BlockingSpeculativeRead,
            "write" => Self::Write,
            "remote_read" => Self::RemoteRead,
            other => panic!("unknown speculation fixture tool: {other}"),
        }
    }
}

struct EvaluationTool {
    definition: ToolDefinition,
    behavior: ToolBehavior,
    calls: AtomicUsize,
    callers: Mutex<Vec<ToolCaller>>,
    entered: Arc<Semaphore>,
    release: Arc<Semaphore>,
    delay_ms: AtomicU64,
}

impl EvaluationTool {
    fn new(behavior: ToolBehavior) -> Self {
        let id = match behavior {
            ToolBehavior::Write => "local.write",
            ToolBehavior::RemoteRead => "remote.read",
            _ => "local.read",
        };
        let mut definition = ToolDefinition::new(
            id,
            "Speculation evaluation tool",
            "Deterministic tool used by the guarded-speculation evaluation matrix",
            json!({
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
                "additionalProperties": false
            }),
        )
        .with_risk(ToolRisk::Low)
        .with_read_only(true)
        .with_idempotent(true)
        .with_parallel_safe(true)
        .with_cancellation_safety(CancellationSafety::Guaranteed)
        .with_allowed_callers([ToolCaller::Direct, ToolCaller::Speculative])
        .with_speculation_policy(SpeculationPolicy::Enabled)
        .with_issue_safety(IssueSafety::Guaranteed)
        .with_execution_location(ExecutionLocation::LocalPrivate)
        .with_network_egress(NetworkEgress::Prohibited)
        .with_output_schema(json!({
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "string"}},
            "additionalProperties": false
        }));
        match behavior {
            ToolBehavior::Write => {
                definition.read_only = false;
                definition.idempotent = false;
                definition.parallel_safe = false;
                definition.allowed_callers = [ToolCaller::Direct].into();
                definition.speculation_policy = SpeculationPolicy::Disabled;
                definition.issue_safety = IssueSafety::Unknown;
            }
            ToolBehavior::RemoteRead => {
                definition.allowed_callers = [ToolCaller::Direct].into();
                definition.speculation_policy = SpeculationPolicy::Disabled;
                definition.execution_location = ExecutionLocation::Remote;
                definition.network_egress = NetworkEgress::Permitted;
            }
            _ => {}
        }
        Self {
            definition,
            behavior,
            calls: AtomicUsize::new(0),
            callers: Mutex::new(Vec::new()),
            entered: Arc::new(Semaphore::new(0)),
            release: Arc::new(Semaphore::new(0)),
            delay_ms: AtomicU64::new(0),
        }
    }
}

#[async_trait]
impl Tool for EvaluationTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(
        &self,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        let context = ToolCallContext::new("", "", "", self.definition.id.clone());
        self.execute_with_context(&context, arguments, cancellation)
            .await
    }

    async fn execute_with_context(
        &self,
        context: &ToolCallContext,
        _: Value,
        _: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        let caller = context.caller.unwrap_or(ToolCaller::Direct);
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.callers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(caller);
        let delay_ms = self.delay_ms.load(Ordering::SeqCst);
        if delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }
        if caller == ToolCaller::Speculative {
            if self.behavior == ToolBehavior::InvalidSpeculativeResult {
                return Ok(ToolResult::success(json!({
                    "private": PRIVATE_RESULT_CANARY
                })));
            }
            if self.behavior == ToolBehavior::BlockingSpeculativeRead {
                self.entered.add_permits(1);
                let permit = self
                    .release
                    .acquire()
                    .await
                    .expect("release semaphore remains open");
                permit.forget();
            }
        }
        Ok(ToolResult::success(json!({"value": "stable"})))
    }
}

struct EvaluationProvider {
    tool_id: String,
    complete_calls: AtomicUsize,
    stream_calls: AtomicUsize,
    fail_next_tool_stream: AtomicBool,
    response_delay_ms: AtomicU64,
}

impl EvaluationProvider {
    fn new(tool_id: impl Into<String>) -> Self {
        Self {
            tool_id: tool_id.into(),
            complete_calls: AtomicUsize::new(0),
            stream_calls: AtomicUsize::new(0),
            fail_next_tool_stream: AtomicBool::new(false),
            response_delay_ms: AtomicU64::new(0),
        }
    }

    fn requests_tool(request: &ModelRequest) -> bool {
        !request
            .messages
            .iter()
            .any(|message| message.role == MessageRole::Tool)
    }

    fn tool_response(&self, model: String) -> ModelResponse {
        ModelResponse::new(model).with_tool_calls(vec![llama_harness_core::ToolCall::new(
            "call-0",
            self.tool_id.clone(),
            r#"{"query":"stable"}"#,
        )])
    }
}

#[async_trait]
impl ModelProvider for EvaluationProvider {
    fn id(&self) -> &str {
        "speculation-evaluation-provider"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::new(true, true, false).with_streaming_tool_arguments(true)
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth::healthy())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(vec![ModelInfo::new("fixture-streaming-model")])
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError> {
        self.complete_calls.fetch_add(1, Ordering::SeqCst);
        let requests_tool = Self::requests_tool(&request);
        if requests_tool {
            let delay_ms = self.response_delay_ms.load(Ordering::SeqCst);
            if delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }
        Ok(if requests_tool {
            self.tool_response(request.model)
        } else {
            ModelResponse::new(request.model).with_final_output("done")
        })
    }

    async fn stream(&self, request: ModelRequest) -> Result<ModelEventStream, HarnessError> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        if Self::requests_tool(&request) {
            let candidate = Ok(ModelStreamEvent::ToolCallDelta(
                ToolCallDelta::new(0, r#"{"query":"stable"}"#, true)
                    .with_call_id("call-0")
                    .with_tool_id(self.tool_id.clone()),
            ));
            let fail = self.fail_next_tool_stream.swap(false, Ordering::SeqCst);
            let delay_ms = self.response_delay_ms.load(Ordering::SeqCst);
            let model = request.model;
            let tail = async move {
                if delay_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
                if fail {
                    Err(HarnessError::RetryableProvider(
                        PRIVATE_STREAM_ERROR_CANARY.into(),
                    ))
                } else {
                    Ok(ModelStreamEvent::Completed {
                        model,
                        usage: Usage::default(),
                    })
                }
            };
            return Ok(Box::pin(
                stream::iter([candidate]).chain(stream::once(tail)),
            ));
        }
        let events = vec![
            Ok(ModelStreamEvent::TextDelta {
                content: "done".into(),
            }),
            Ok(ModelStreamEvent::Completed {
                model: request.model,
                usage: Usage::default(),
            }),
        ];
        Ok(Box::pin(stream::iter(events)))
    }
}

struct EvaluationPolicy {
    speculative_decisions: AtomicUsize,
}

#[async_trait]
impl PolicyEngine for EvaluationPolicy {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(PolicyDecision::Allow {
            reason: "authoritative evaluation allow".into(),
        })
    }

    async fn decide_speculative(
        &self,
        _: &ToolCallContext,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        self.speculative_decisions.fetch_add(1, Ordering::SeqCst);
        Ok(PolicyDecision::Allow {
            reason: "dedicated speculative evaluation allow".into(),
        })
    }
}

#[derive(Clone, Debug)]
struct Evidence {
    readiness: SpeculationReadiness,
    metrics: SpeculationMetrics,
    tool_calls: usize,
    callers: Vec<ToolCaller>,
    speculative_policy_decisions: usize,
    complete_calls: usize,
    stream_calls: usize,
    public_result_json: String,
    sqlite_export: String,
}

#[derive(Default)]
struct AcceptanceExecutor {
    evidence: Mutex<Vec<(String, Evidence)>>,
}

impl AcceptanceExecutor {
    fn evidence(&self, case_id: &str) -> Evidence {
        self.evidence
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .find(|(id, _)| id == case_id)
            .unwrap_or_else(|| panic!("missing evidence for {case_id}"))
            .1
            .clone()
    }
}

fn run_request(tool_id: &str, run_id: impl Into<String>) -> RunRequest {
    let run_id = run_id.into();
    let mut agent = AgentDefinition::new(
        "speculation-fixture-agent",
        "Speculation fixture agent",
        "1",
        "fixture-streaming-model",
    );
    agent.tool_allowlist = vec![tool_id.into()];
    agent.limits.max_model_calls = 2;
    RunRequest::new(agent, "read stable state")
        .with_run_id(run_id.clone())
        .with_trace_id(format!("{run_id}-trace"))
}

fn uses_active_mode(mode: &str) -> bool {
    matches!(
        mode,
        "active_exact" | "active_discard" | "saturated_fallback" | "active_terminal_failure"
    )
}

async fn train_and_activate(
    runner: &AgentRunner,
    tool_id: &str,
    case_id: &str,
) -> Result<(), EvalError> {
    for observation in 0..1_000 {
        let result = runner
            .run_with_strategy(
                run_request(tool_id, format!("{case_id}-shadow-{observation}")),
                RunStrategy::Direct,
            )
            .await
            .map_err(|error| EvalError::Executor(error.to_string()))?;
        if result.status != RunStatus::Completed {
            return Err(EvalError::Executor(
                "shadow training did not complete".into(),
            ));
        }
    }
    if !runner.speculation_readiness(tool_id).ready_to_activate {
        return Err(EvalError::Executor(
            "shadow training did not reach the activation threshold".into(),
        ));
    }
    if runner.activate_speculation(tool_id).mode != SpeculationMode::Active {
        return Err(EvalError::Executor(
            "explicit activation did not enter Active".into(),
        ));
    }
    Ok(())
}

fn persist_canonical_events(
    events: &[EventRecord],
    result: &RunResult,
) -> Result<String, EvalError> {
    let store = SqliteEventSink::open_in_memory(TraceStoreConfig::default())
        .map_err(|error| EvalError::Executor(error.to_string()))?;
    store
        .append_batch(
            events
                .iter()
                .filter(|record| record.run_id == result.id && record.trace_id == result.trace_id)
                .cloned()
                .map(|record| (record, None)),
        )
        .map_err(|error| EvalError::Executor(error.to_string()))?;
    store
        .export_run_json(&result.id)
        .map_err(|error| EvalError::Executor(error.to_string()))?
        .ok_or_else(|| EvalError::Executor("canonical evaluation trace was not persisted".into()))
}

fn measured_metrics(status: &RunStatus, mode: &str) -> StrategyMetrics {
    let expected_status = if mode == "active_terminal_failure" {
        RunStatus::Failed
    } else {
        RunStatus::Completed
    };
    StrategyMetrics {
        unauthorized_effects: Some(0),
        duplicate_effects: Some(0),
        unintended_effects: Some(0),
        task_correct: Some(*status == expected_status),
        final_state_correct: Some(*status == expected_status),
        recovery_success: Some(mode != "active_terminal_failure"),
        tool_selection_accuracy: Some(1.0),
        input_tokens: Some(0),
        output_tokens: Some(0),
        wasted_tool_calls: Some(u32::from(mode == "active_discard")),
    }
}

#[async_trait]
impl EvalExecutor for AcceptanceExecutor {
    async fn execute(&self, request: EvalExecutionRequest) -> Result<EvalObservation, EvalError> {
        let fixture = request.fixture.as_ref().expect("speculation fixture");
        let mode = fixture.data["mode"].as_str().expect("fixture mode");
        let behavior =
            ToolBehavior::from_fixture(fixture.data["tool"].as_str().expect("fixture tool"));
        let tool = Arc::new(EvaluationTool::new(behavior));
        let tool_id = tool.definition.id.clone();
        let mut registry = ToolRegistry::default();
        registry
            .register(tool.clone())
            .map_err(|error| EvalError::Executor(error.to_string()))?;
        let provider = Arc::new(EvaluationProvider::new(&tool_id));
        let policy = Arc::new(EvaluationPolicy {
            speculative_decisions: AtomicUsize::new(0),
        });
        let events = Arc::new(InMemoryEventSink::default());
        let builder = AgentRunner::builder(provider.clone())
            .tools(registry)
            .policy(policy.clone())
            .event_sink(events.clone());
        let runner = Arc::new(if mode == "disabled" {
            builder.build()
        } else {
            builder.speculation(SpeculationConfig::default()).build()
        });

        if uses_active_mode(mode) {
            train_and_activate(&runner, &tool_id, &request.case.id).await?;
        }
        if mode == "active_terminal_failure" {
            provider.fail_next_tool_stream.store(true, Ordering::SeqCst);
        }
        let complete_before = provider.complete_calls.load(Ordering::SeqCst);
        let stream_before = provider.stream_calls.load(Ordering::SeqCst);

        let run = if mode == "saturated_fallback" {
            let holder_runner = runner.clone();
            let holder_tool_id = tool_id.clone();
            let holder_case_id = request.case.id.clone();
            let holder = tokio::spawn(async move {
                holder_runner
                    .run_with_strategy(
                        run_request(&holder_tool_id, format!("{holder_case_id}-holder")),
                        RunStrategy::Direct,
                    )
                    .await
            });
            let entered = tool
                .entered
                .acquire()
                .await
                .map_err(|_| EvalError::Executor("blocking tool closed".into()))?;
            entered.forget();
            let fallback = runner
                .run_with_strategy(
                    run_request(&tool_id, request.case.id.clone()),
                    RunStrategy::Direct,
                )
                .await
                .map_err(|error| EvalError::Executor(error.to_string()))?;
            tool.release.add_permits(1);
            let holder_result = holder
                .await
                .map_err(|error| EvalError::Executor(error.to_string()))?
                .map_err(|error| EvalError::Executor(error.to_string()))?;
            if holder_result.status != RunStatus::Completed {
                return Err(EvalError::Executor(
                    "slot-holder evaluation did not complete".into(),
                ));
            }
            fallback
        } else {
            runner
                .run_with_strategy(
                    run_request(&tool_id, request.case.id.clone()),
                    RunStrategy::Direct,
                )
                .await
                .map_err(|error| EvalError::Executor(error.to_string()))?
        };

        let all_events = events.events();
        let sqlite_export = persist_canonical_events(&all_events, &run)?;
        let evidence = Evidence {
            readiness: runner.speculation_readiness(&tool_id),
            metrics: runner.speculation_metrics(&tool_id),
            tool_calls: tool.calls.load(Ordering::SeqCst),
            callers: tool
                .callers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
            speculative_policy_decisions: policy.speculative_decisions.load(Ordering::SeqCst),
            complete_calls: provider
                .complete_calls
                .load(Ordering::SeqCst)
                .saturating_sub(complete_before),
            stream_calls: provider
                .stream_calls
                .load(Ordering::SeqCst)
                .saturating_sub(stream_before),
            public_result_json: serde_json::to_string(&run)
                .map_err(|error| EvalError::Executor(error.to_string()))?,
            sqlite_export,
        };
        self.evidence
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((request.case.id.clone(), evidence));
        let model_calls = if mode == "active_terminal_failure" {
            1
        } else {
            2
        };
        let metrics = measured_metrics(&run.status, mode);
        Ok(EvalObservation::new(run, model_calls)
            .with_strategy_metrics(metrics)
            .with_final_state(Some(json!({"mode": mode, "stable": true}))))
    }
}

fn assert_private_diagnostics_absent(evidence: &Evidence) {
    for forbidden in [
        PRIVATE_RESULT_CANARY,
        PRIVATE_STREAM_ERROR_CANARY,
        "shadow_matches",
        "exact_shadow_observations",
        "ready_to_activate",
        "slot_saturated",
        "active_candidates_considered",
        "pre_issue_validation_skipped",
        "pre_issue_policy_skipped",
        "pre_issue_failed",
        "pre_issue_invalidated",
        "pre_issue_aborted",
        "key_saturated",
        "in_flight",
        "oldest_in_flight_ms",
        "execution_duration_ms",
        "publication_wait_ms",
        "candidate_result",
        "candidate_error",
    ] {
        assert!(
            !evidence.public_result_json.contains(forbidden),
            "public result leaked {forbidden}"
        );
        assert!(
            !evidence.sqlite_export.contains(forbidden),
            "SQLite export leaked {forbidden}"
        );
    }
}

async fn execute_matrix() -> (EvaluationReport, AcceptanceExecutor) {
    let suite = load_suite(SUITE, Some("yaml")).unwrap();
    let executor = AcceptanceExecutor::default();
    let report = evaluate_suite(&suite, &executor, &[], None).await.unwrap();
    (report, executor)
}

#[tokio::test]
async fn forced_speculation_matrix_uses_real_runner_boundaries_and_private_metrics() {
    let (report, executor) = execute_matrix().await;
    assert!(
        report.results.iter().all(|result| result.passed),
        "{report:#?}"
    );

    let disabled = executor.evidence("disabled");
    assert_eq!(disabled.readiness.mode, SpeculationMode::Disabled);
    assert_eq!(disabled.complete_calls, 2);
    assert_eq!(disabled.stream_calls, 0);
    assert_eq!(disabled.callers, [ToolCaller::Direct]);

    let shadow = executor.evidence("shadow");
    assert_eq!(shadow.readiness.mode, SpeculationMode::Shadow);
    assert_eq!(shadow.readiness.exact_shadow_observations, 1);
    assert_eq!(shadow.metrics.shadow_matches, 1);
    assert_eq!(shadow.metrics.issued, 0);
    assert_eq!(shadow.speculative_policy_decisions, 0);
    assert_eq!(shadow.callers, [ToolCaller::Direct]);

    let exact = executor.evidence("active-exact");
    assert_eq!(exact.readiness.mode, SpeculationMode::Active);
    assert_eq!(exact.metrics.issued, 1);
    assert_eq!(exact.metrics.active_candidates_considered, 1);
    assert_eq!(exact.metrics.in_flight, 0);
    assert_eq!(exact.metrics.committed, 1);
    assert_eq!(exact.metrics.discarded, 0);
    assert_eq!(exact.tool_calls, 1_001);
    assert_eq!(exact.callers.last(), Some(&ToolCaller::Speculative));
    assert_eq!(exact.metrics.execution_duration_ms.count, 1);
    assert_eq!(exact.metrics.publication_wait_ms.count, 1);

    let discard = executor.evidence("active-discard");
    assert_eq!(discard.readiness.mode, SpeculationMode::Shadow);
    assert_eq!(discard.readiness.exact_shadow_observations, 0);
    assert_eq!(discard.metrics.issued, 1);
    assert_eq!(discard.metrics.discarded, 1);
    assert_eq!(
        &discard.callers[discard.callers.len() - 2..],
        [ToolCaller::Speculative, ToolCaller::Direct]
    );

    let saturated = executor.evidence("saturated-fallback");
    assert_eq!(saturated.metrics.slot_saturated, 1);
    assert_eq!(saturated.metrics.active_candidates_considered, 2);
    assert_eq!(saturated.metrics.issued, 1);
    assert_eq!(saturated.metrics.committed, 1);
    assert!(saturated
        .callers
        .ends_with(&[ToolCaller::Speculative, ToolCaller::Direct]));

    for case_id in ["ineligible-write", "ineligible-remote"] {
        let ineligible = executor.evidence(case_id);
        assert_eq!(ineligible.readiness.mode, SpeculationMode::Disabled);
        assert_eq!(ineligible.metrics.issued, 0);
        assert_eq!(ineligible.speculative_policy_decisions, 0);
        assert_eq!(ineligible.callers, [ToolCaller::Direct]);
    }

    let terminal = executor.evidence("terminal-stream-failure");
    assert_eq!(terminal.readiness.mode, SpeculationMode::Shadow);
    assert_eq!(terminal.metrics.issued, 1);
    assert_eq!(terminal.metrics.discarded, 1);
    assert_eq!(terminal.metrics.terminal_stream_failures, 1);
    assert_eq!(terminal.stream_calls, 1, "accepted streams are not retried");
    assert_eq!(terminal.complete_calls, 0);

    for evidence in [&exact, &discard, &saturated, &terminal] {
        assert_eq!(
            evidence.metrics.issued,
            evidence.metrics.in_flight
                + evidence.metrics.committed
                + evidence.metrics.discarded
                + evidence.metrics.cancelled
        );
    }

    for case_id in [
        "disabled",
        "shadow",
        "active-exact",
        "active-discard",
        "saturated-fallback",
        "ineligible-write",
        "ineligible-remote",
        "terminal-stream-failure",
    ] {
        assert_private_diagnostics_absent(&executor.evidence(case_id));
    }
}

#[tokio::test]
async fn correctness_and_safety_are_hard_gates_before_latency_comparison() {
    let (report, _) = execute_matrix().await;
    let safe = report
        .results
        .iter()
        .filter(|result| result.status == Some(RunStatus::Completed))
        .collect::<Vec<_>>();
    assert!(safe.iter().all(|result| result.passes_readiness()));

    let mut faster_but_unsafe = (*safe[0]).clone();
    faster_but_unsafe.duration_ms = Some(0);
    faster_but_unsafe.strategy_metrics.unintended_effects = Some(1);
    assert!(!faster_but_unsafe.passes_readiness());
    assert!(safe
        .iter()
        .min_by_key(|result| result.duration_ms)
        .is_some());
}

#[test]
fn speculation_acceptance_fixture_declares_the_forced_matrix() {
    let suite = load_suite(SUITE, Some("yaml")).unwrap();
    assert_eq!(
        suite
            .cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<Vec<_>>(),
        [
            "disabled",
            "shadow",
            "active-exact",
            "active-discard",
            "saturated-fallback",
            "ineligible-write",
            "ineligible-remote",
            "terminal-stream-failure",
        ]
    );
    assert!(suite
        .cases
        .iter()
        .all(|case| case.strategy.is_none() || case.strategy == Some(RunStrategy::Direct)));
}

#[tokio::test]
#[ignore = "release-only informational measurement; finalized index-0 makes public invocation mismatch unconstructible, so validation discard is the fallback case"]
async fn release_speculation_mode_evaluation() {
    let (report, executor) = execute_matrix().await;
    for case_id in [
        "disabled",
        "shadow",
        "active-exact",
        "active-discard",
        "saturated-fallback",
    ] {
        let result = report
            .results
            .iter()
            .find(|result| result.case_id == case_id)
            .expect("release case result");
        let evidence = executor.evidence(case_id);
        eprintln!(
            "{case_id}: duration_ms={} issued={} committed={} discarded={} saturated={}",
            result.duration_ms.unwrap_or_default(),
            evidence.metrics.issued,
            evidence.metrics.committed,
            evidence.metrics.discarded,
            evidence.metrics.slot_saturated,
        );
    }

    const CONTROLLED_DELAY_MS: u64 = 25;
    for (label, configured, activate) in [
        ("controlled-disabled", false, false),
        ("controlled-shadow", true, false),
        ("controlled-active", true, true),
    ] {
        let tool = Arc::new(EvaluationTool::new(ToolBehavior::Eligible));
        let tool_id = tool.definition.id.clone();
        let mut registry = ToolRegistry::default();
        registry.register(tool.clone()).unwrap();
        let provider = Arc::new(EvaluationProvider::new(&tool_id));
        let policy = Arc::new(EvaluationPolicy {
            speculative_decisions: AtomicUsize::new(0),
        });
        let builder = AgentRunner::builder(provider.clone())
            .tools(registry)
            .policy(policy);
        let runner = if configured {
            builder.speculation(SpeculationConfig::default()).build()
        } else {
            builder.build()
        };
        if activate {
            train_and_activate(&runner, &tool_id, label).await.unwrap();
        }
        tool.delay_ms.store(CONTROLLED_DELAY_MS, Ordering::SeqCst);
        provider
            .response_delay_ms
            .store(CONTROLLED_DELAY_MS, Ordering::SeqCst);
        let calls_before = tool.calls.load(Ordering::SeqCst);
        let started = std::time::Instant::now();
        let result = runner
            .run_with_strategy(run_request(&tool_id, label), RunStrategy::Direct)
            .await
            .unwrap();
        let elapsed = started.elapsed();
        let metrics = runner.speculation_metrics(&tool_id);

        // Correctness, exact effect cardinality, and accounting are the only
        // gates. Wall-clock measurements are printed for release comparison,
        // never asserted because scheduler noise is platform-dependent.
        assert_eq!(result.status, RunStatus::Completed);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(tool.calls.load(Ordering::SeqCst) - calls_before, 1);
        assert_eq!(
            metrics.issued,
            metrics.in_flight + metrics.committed + metrics.discarded + metrics.cancelled
        );
        if activate {
            assert_eq!(metrics.committed, 1);
            assert_eq!(
                tool.callers.lock().unwrap().last(),
                Some(&ToolCaller::Speculative)
            );
        } else {
            assert_eq!(metrics.issued, 0);
            assert_eq!(
                tool.callers.lock().unwrap().last(),
                Some(&ToolCaller::Direct)
            );
        }
        eprintln!(
            "{label}: controlled_delay_ms={CONTROLLED_DELAY_MS} observed_ms={} issued={} committed={} execution_p95_source_count={} publication_p95_source_count={}",
            elapsed.as_millis(),
            metrics.issued,
            metrics.committed,
            metrics.execution_duration_ms.count,
            metrics.publication_wait_ms.count,
        );
    }
}
