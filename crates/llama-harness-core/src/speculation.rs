use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError},
    time::Instant,
};

/// Minimum exact same-runner shadow observations required before activation.
pub const MIN_SPECULATION_SHADOW_OBSERVATIONS: u64 = 1_000;
/// Immutable ceiling for one speculative candidate lifetime, in milliseconds.
pub const HARD_MAX_SPECULATION_DURATION_MS: u64 = 5_000;
/// Immutable ceiling for stream events processed in one speculative model turn.
pub const HARD_MAX_SPECULATION_STREAM_EVENTS: u32 = 4_096;

const DEFAULT_SPECULATION_DURATION_MS: u64 = 1_000;
const DEFAULT_SPECULATION_STREAM_EVENTS: u32 = 1_024;

/// Explicit host opt-in and conservative bounds for speculative execution.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SpeculationConfig {
    /// Maximum issue-to-commit lifetime of one candidate, in milliseconds.
    pub max_execution_duration_ms: u64,
    /// Exact consecutive shadow observations required before activation.
    ///
    /// Values below [`MIN_SPECULATION_SHADOW_OBSERVATIONS`] are rejected.
    pub required_shadow_observations: u64,
    /// Maximum provider stream events processed in one speculative model turn.
    pub max_stream_events: u32,
}

impl Default for SpeculationConfig {
    fn default() -> Self {
        Self {
            max_execution_duration_ms: DEFAULT_SPECULATION_DURATION_MS,
            required_shadow_observations: MIN_SPECULATION_SHADOW_OBSERVATIONS,
            max_stream_events: DEFAULT_SPECULATION_STREAM_EVENTS,
        }
    }
}

impl SpeculationConfig {
    pub(crate) fn validate(&self) -> Result<(), crate::HarnessError> {
        if self.max_execution_duration_ms == 0
            || self.max_execution_duration_ms > HARD_MAX_SPECULATION_DURATION_MS
        {
            return Err(crate::HarnessError::InvalidRequest(
                "speculation duration must be within 1..=5000 milliseconds".into(),
            ));
        }
        if self.required_shadow_observations < MIN_SPECULATION_SHADOW_OBSERVATIONS {
            return Err(crate::HarnessError::InvalidRequest(format!(
                "speculation requires at least {MIN_SPECULATION_SHADOW_OBSERVATIONS} shadow observations"
            )));
        }
        if self.max_stream_events == 0
            || self.max_stream_events > HARD_MAX_SPECULATION_STREAM_EVENTS
        {
            return Err(crate::HarnessError::InvalidRequest(
                "speculation stream events must be within 1..=4096".into(),
            ));
        }
        Ok(())
    }
}

/// Current per-tool speculative execution mode.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SpeculationMode {
    /// The runner or requested tool has no eligible host-owned speculation state.
    #[default]
    Disabled,
    /// Candidates are compared with authoritative responses but never executed.
    Shadow,
    /// Eligible candidates may execute before the authoritative stream completes.
    Active,
}

/// Current same-runner evidence for one tool's explicit activation decision.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct SpeculationReadiness {
    /// Registered tool identifier whose state was queried.
    pub tool_id: String,
    /// Current per-tool mode.
    pub mode: SpeculationMode,
    /// Consecutive exact shadow observations since this tool's last incident.
    pub exact_shadow_observations: u64,
    /// Exact shadow observations required by the host configuration.
    pub required_shadow_observations: u64,
    /// Whether an explicit activation request can currently enter active mode.
    pub ready_to_activate: bool,
}

impl SpeculationReadiness {
    pub(crate) fn disabled(tool_id: &str) -> Self {
        Self::disabled_with_required(tool_id, MIN_SPECULATION_SHADOW_OBSERVATIONS)
    }

    fn disabled_with_required(tool_id: &str, required_shadow_observations: u64) -> Self {
        Self {
            tool_id: tool_id.to_owned(),
            mode: SpeculationMode::Disabled,
            exact_shadow_observations: 0,
            required_shadow_observations,
            ready_to_activate: false,
        }
    }
}

/// Pull-only metadata counters for one tool's speculative execution state.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct SpeculationMetrics {
    /// Registered tool identifier whose state was queried.
    pub tool_id: String,
    /// Current per-tool mode.
    pub mode: SpeculationMode,
    /// Exact shadow candidates observed over the runner lifetime.
    pub shadow_matches: u64,
    /// Candidate/authoritative invocation mismatches.
    pub mismatches: u64,
    /// Candidate-bearing streams that failed before authoritative completion.
    pub terminal_stream_failures: u64,
    /// Speculative tool executions issued through the broker.
    pub issued: u64,
    /// Issued executions committed as the exact authoritative first call.
    pub committed: u64,
    /// Issued executions resolved but were not reusable.
    pub discarded: u64,
    /// Issued executions cancelled before a reusable result was available.
    pub cancelled: u64,
    /// Active candidates skipped because the runner-wide slot was occupied.
    pub slot_saturated: u64,
}

impl SpeculationMetrics {
    pub(crate) fn disabled(tool_id: &str) -> Self {
        Self {
            tool_id: tool_id.to_owned(),
            mode: SpeculationMode::Disabled,
            shadow_matches: 0,
            mismatches: 0,
            terminal_stream_failures: 0,
            issued: 0,
            committed: 0,
            discarded: 0,
            cancelled: 0,
            slot_saturated: 0,
        }
    }
}

struct ToolState {
    mode: SpeculationMode,
    exact_shadow_observations: u64,
    metrics: SpeculationMetrics,
}

pub(crate) struct SpeculationController {
    config: SpeculationConfig,
    tools: Mutex<HashMap<String, ToolState>>,
    slot: Arc<Semaphore>,
}

impl SpeculationController {
    pub(crate) fn new(config: SpeculationConfig) -> Self {
        Self {
            config,
            tools: Mutex::new(HashMap::new()),
            slot: Arc::new(Semaphore::new(1)),
        }
    }

    pub(crate) fn config(&self) -> &SpeculationConfig {
        &self.config
    }

    /// Registers only a broker-validated registry identity, keeping state
    /// bounded by the immutable runner registry rather than model-controlled IDs.
    pub(crate) fn register_tool(&self, tool_id: &str) {
        let mut tools = self
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        tools
            .entry(tool_id.to_owned())
            .or_insert_with(|| ToolState {
                mode: SpeculationMode::Shadow,
                exact_shadow_observations: 0,
                metrics: SpeculationMetrics {
                    tool_id: tool_id.to_owned(),
                    mode: SpeculationMode::Shadow,
                    shadow_matches: 0,
                    mismatches: 0,
                    terminal_stream_failures: 0,
                    issued: 0,
                    committed: 0,
                    discarded: 0,
                    cancelled: 0,
                    slot_saturated: 0,
                },
            });
    }

    pub(crate) fn readiness(&self, tool_id: &str) -> SpeculationReadiness {
        let tools = self
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(state) = tools.get(tool_id) else {
            return SpeculationReadiness::disabled_with_required(
                tool_id,
                self.config.required_shadow_observations,
            );
        };
        SpeculationReadiness {
            tool_id: tool_id.to_owned(),
            mode: state.mode,
            exact_shadow_observations: state.exact_shadow_observations,
            required_shadow_observations: self.config.required_shadow_observations,
            ready_to_activate: state_is_ready(state, &self.config),
        }
    }

    pub(crate) fn metrics(&self, tool_id: &str) -> SpeculationMetrics {
        let tools = self
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(state) = tools.get(tool_id) else {
            return SpeculationMetrics::disabled(tool_id);
        };
        let mut metrics = state.metrics.clone();
        metrics.mode = state.mode;
        debug_assert_eq!(
            metrics.issued,
            metrics.committed + metrics.discarded + metrics.cancelled
        );
        metrics
    }

    pub(crate) fn activate(&self, tool_id: &str) -> SpeculationReadiness {
        let mut tools = self
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(state) = tools.get_mut(tool_id) else {
            return SpeculationReadiness::disabled_with_required(
                tool_id,
                self.config.required_shadow_observations,
            );
        };
        if state_is_ready(state, &self.config) {
            state.mode = SpeculationMode::Active;
            state.metrics.mode = SpeculationMode::Active;
        }
        SpeculationReadiness {
            tool_id: tool_id.to_owned(),
            mode: state.mode,
            exact_shadow_observations: state.exact_shadow_observations,
            required_shadow_observations: self.config.required_shadow_observations,
            ready_to_activate: state_is_ready(state, &self.config),
        }
    }

    pub(crate) fn return_to_shadow(&self, tool_id: &str) -> SpeculationReadiness {
        let mut tools = self
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(state) = tools.get_mut(tool_id) else {
            return SpeculationReadiness::disabled_with_required(
                tool_id,
                self.config.required_shadow_observations,
            );
        };
        trip(state);
        SpeculationReadiness {
            tool_id: tool_id.to_owned(),
            mode: SpeculationMode::Shadow,
            exact_shadow_observations: 0,
            required_shadow_observations: self.config.required_shadow_observations,
            ready_to_activate: false,
        }
    }

    pub(crate) fn mode(&self, tool_id: &str) -> SpeculationMode {
        self.tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(tool_id)
            .map_or(SpeculationMode::Disabled, |state| state.mode)
    }

    pub(crate) fn is_active(&self, tool_id: &str) -> bool {
        self.mode(tool_id) == SpeculationMode::Active
    }

    pub(crate) fn try_acquire_slot(&self) -> Result<OwnedSemaphorePermit, TryAcquireError> {
        Arc::clone(&self.slot).try_acquire_owned()
    }

    pub(crate) fn record_shadow_match(&self, tool_id: &str) {
        let mut tools = self
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(state) = tools.get_mut(tool_id) {
            state.exact_shadow_observations = state.exact_shadow_observations.saturating_add(1);
            state.metrics.shadow_matches = state.metrics.shadow_matches.saturating_add(1);
        }
    }

    pub(crate) fn record_mismatch_and_trip(&self, tool_id: &str) {
        let mut tools = self
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(state) = tools.get_mut(tool_id) {
            state.metrics.mismatches = state.metrics.mismatches.saturating_add(1);
            trip(state);
        }
    }

    pub(crate) fn record_terminal_stream_failure_and_trip(&self, tool_id: &str) {
        let mut tools = self
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(state) = tools.get_mut(tool_id) {
            state.metrics.terminal_stream_failures =
                state.metrics.terminal_stream_failures.saturating_add(1);
            trip(state);
        }
    }

    pub(crate) fn record_slot_saturated(&self, tool_id: &str) {
        let mut tools = self
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(state) = tools.get_mut(tool_id) {
            state.metrics.slot_saturated = state.metrics.slot_saturated.saturating_add(1);
        }
    }

    pub(crate) fn record_resolution(&self, tool_id: &str, resolution: SpeculativeResolution) {
        let mut tools = self
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(state) = tools.get_mut(tool_id) else {
            return;
        };
        state.metrics.issued = state.metrics.issued.saturating_add(1);
        match resolution {
            SpeculativeResolution::Discarded => {
                state.metrics.discarded = state.metrics.discarded.saturating_add(1);
                trip(state);
            }
            SpeculativeResolution::Cancelled => {
                state.metrics.cancelled = state.metrics.cancelled.saturating_add(1);
                trip(state);
            }
        }
        debug_assert_eq!(
            state.metrics.issued,
            state.metrics.committed + state.metrics.discarded + state.metrics.cancelled
        );
    }

    pub(crate) fn record_commit_if_active(
        &self,
        tool_id: &str,
        candidate_deadline: Instant,
        run_deadline: Option<Instant>,
        cancellation: &tokio_util::sync::CancellationToken,
    ) -> bool {
        let mut tools = self
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(state) = tools.get_mut(tool_id) else {
            return false;
        };
        let now = Instant::now();
        if state.mode != SpeculationMode::Active
            || cancellation.is_cancelled()
            || now >= candidate_deadline
            || run_deadline.is_some_and(|deadline| now >= deadline)
        {
            return false;
        }
        state.metrics.issued = state.metrics.issued.saturating_add(1);
        state.metrics.committed = state.metrics.committed.saturating_add(1);
        debug_assert_eq!(
            state.metrics.issued,
            state.metrics.committed + state.metrics.discarded + state.metrics.cancelled
        );
        true
    }
}

fn trip(state: &mut ToolState) {
    state.mode = SpeculationMode::Shadow;
    state.metrics.mode = SpeculationMode::Shadow;
    state.exact_shadow_observations = 0;
}

fn state_is_ready(state: &ToolState, config: &SpeculationConfig) -> bool {
    state.mode == SpeculationMode::Shadow
        && state.exact_shadow_observations >= config.required_shadow_observations
        && state.metrics.mismatches == 0
        && state.metrics.terminal_stream_failures == 0
        && state.metrics.discarded == 0
        && state.metrics.cancelled == 0
}

#[derive(Clone, Copy)]
pub(crate) enum SpeculativeResolution {
    Discarded,
    Cancelled,
}

use crate::{
    broker::{InvocationKey, SpeculativeAttempt, SpeculativeExecution, ToolBroker},
    runner::await_guarded,
    AgentLimits, HarnessError, ModelProvider, ModelRequest, ModelResponse, ModelStreamController,
    ModelStreamEvent, ProviderCapabilityLimits, ToolCall, ToolCallAssembler,
    ToolCallAssemblyLimits, Usage,
};
use futures_util::StreamExt;
use std::{collections::BTreeMap, time::Duration};

const MAX_PARTIAL_PROBES: u8 = 8;

pub(crate) struct StreamedReactiveResponse {
    pub(crate) response: ModelResponse,
    pub(crate) speculative: Option<ReadySpeculativeCommit>,
}

pub(crate) struct ReadySpeculativeCommit {
    pub(crate) execution: Option<Box<SpeculativeExecution>>,
    pub(crate) deadline: Instant,
}

pub(crate) fn discard_ready_commit(
    _controller: &SpeculationController,
    mut ready: ReadySpeculativeCommit,
    cancelled: bool,
) {
    if cancelled {
        if let Some(execution) = ready.execution.as_mut() {
            execution.settle(SpeculativeResolution::Cancelled);
        }
    }
}

enum CandidateState {
    Shadow {
        call: ToolCall,
        key: InvocationKey,
    },
    Active {
        call: ToolCall,
        key: InvocationKey,
        execution: Box<SpeculativeExecution>,
        deadline: Instant,
    },
}

struct StreamResponseAssembler {
    model: Option<String>,
    usage: Usage,
    text: String,
    calls: BTreeMap<usize, ToolCall>,
    retained_bytes: u64,
    max_bytes: u64,
}

impl StreamResponseAssembler {
    fn new(max_bytes: u64) -> Self {
        Self {
            model: None,
            usage: Usage::default(),
            text: String::new(),
            calls: BTreeMap::new(),
            retained_bytes: 0,
            max_bytes,
        }
    }

    fn retain(&mut self, bytes: usize) -> Result<(), HarnessError> {
        self.retained_bytes = self
            .retained_bytes
            .checked_add(bytes as u64)
            .ok_or_else(|| HarnessError::ResourceLimit("stream response size overflow".into()))?;
        if self.retained_bytes > self.max_bytes {
            return Err(HarnessError::ResourceLimit(
                "stream response exceeded model response byte limit".into(),
            ));
        }
        Ok(())
    }

    fn text(&mut self, content: String) -> Result<(), HarnessError> {
        self.retain(content.len())?;
        self.text.try_reserve(content.len()).map_err(|_| {
            HarnessError::ResourceLimit("stream response text allocation failed".into())
        })?;
        self.text.push_str(&content);
        Ok(())
    }

    fn call(&mut self, index: usize, call: ToolCall) -> Result<(), HarnessError> {
        self.retain(
            call.id
                .len()
                .saturating_add(call.tool_id.len())
                .saturating_add(call.arguments_json.len()),
        )?;
        if self.calls.insert(index, call).is_some() {
            return Err(HarnessError::InvalidRequest(
                "stream finalized one call index more than once".into(),
            ));
        }
        Ok(())
    }

    fn complete(&mut self, model: String, usage: Usage) -> Result<(), HarnessError> {
        self.retain(model.len())?;
        self.model = Some(model);
        self.usage = usage;
        Ok(())
    }

    fn finish(self) -> Result<ModelResponse, HarnessError> {
        let model = self.model.ok_or_else(|| {
            HarnessError::InvalidRequest("stream completed without model metadata".into())
        })?;
        Ok(ModelResponse {
            model,
            final_output: (!self.text.is_empty()).then_some(self.text),
            tool_calls: self.calls.into_values().collect(),
            usage: self.usage,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn stream_reactive_response(
    provider: &Arc<dyn ModelProvider>,
    broker: &ToolBroker<'_>,
    controller: &Arc<SpeculationController>,
    request: &crate::RunRequest,
    model_request: ModelRequest,
    run_id: &str,
    trace_id: &str,
    model_call_number: u32,
    call_deadline: Option<Instant>,
) -> Result<StreamedReactiveResponse, HarnessError> {
    controller.config().validate()?;
    let capabilities = provider.capabilities();
    let limits = assembly_limits(
        &request.agent.limits,
        &capabilities.limits,
        model_request.tools.len(),
    )?;
    let assembler = ToolCallAssembler::new(model_request.tools.clone(), limits)?;
    let mut stream_controller = ModelStreamController::new(assembler);
    let model_cancellation = model_request.cancellation.clone();
    let stream = await_guarded(
        provider.stream(model_request),
        &request.cancellation,
        call_deadline,
        "provider stream start deadline reached",
        Some(&model_cancellation),
    )
    .await?;
    tokio::pin!(stream);

    let mut response = StreamResponseAssembler::new(request.agent.limits.max_model_response_bytes);
    let mut candidate = None;
    let mut candidate_considered = false;
    let mut partial_probes = 0_u8;
    let mut stream_events = 0_u32;

    loop {
        let candidate_deadline = match &candidate {
            Some(CandidateState::Active { deadline, .. }) => Some(*deadline),
            _ => None,
        };
        let next_deadline = match (call_deadline, candidate_deadline) {
            (Some(call), Some(candidate)) => Some(call.min(candidate)),
            (Some(call), None) => Some(call),
            (None, candidate) => candidate,
        };
        let next = if let Some(deadline) = next_deadline {
            tokio::select! {
                biased;
                _ = request.cancellation.cancelled() => {
                    model_cancellation.cancel();
                    discard_candidate(controller, candidate.take(), true);
                    return Err(HarnessError::Cancelled);
                }
                _ = tokio::time::sleep_until(deadline) => {
                    if candidate_deadline.is_some_and(|candidate| candidate <= deadline)
                        && call_deadline.is_none_or(|call| candidate_deadline != Some(call))
                    {
                        discard_candidate(controller, candidate.take(), false);
                        continue;
                    }
                    model_cancellation.cancel();
                    discard_candidate(controller, candidate.take(), true);
                    return Err(HarnessError::TimedOut(
                        "provider stream deadline reached".into(),
                    ));
                }
                next = stream.next() => next,
            }
        } else {
            tokio::select! {
                biased;
                _ = request.cancellation.cancelled() => {
                    model_cancellation.cancel();
                    discard_candidate(controller, candidate.take(), true);
                    return Err(HarnessError::Cancelled);
                }
                next = stream.next() => next,
            }
        };

        let Some(item) = next else {
            if let Err(error) = stream_controller.finish_eof() {
                model_cancellation.cancel();
                fail_candidate_stream(controller, candidate.take());
                return Err(error);
            }
            break;
        };
        stream_events = stream_events.saturating_add(1);
        if stream_events > controller.config().max_stream_events {
            model_cancellation.cancel();
            fail_candidate_stream(controller, candidate.take());
            return Err(HarnessError::ResourceLimit(
                "speculative stream event limit reached".into(),
            ));
        }

        let delta_index = match &item {
            Ok(ModelStreamEvent::ToolCallDelta(delta)) => Some((delta.index, delta.is_final)),
            _ => None,
        };
        let validated = match stream_controller.push(item) {
            Ok(validated) => validated,
            Err(error) => {
                model_cancellation.cancel();
                fail_candidate_stream(controller, candidate.take());
                return Err(error);
            }
        };

        if capabilities.supports_streaming_tool_arguments
            && partial_probes < MAX_PARTIAL_PROBES
            && matches!(delta_index, Some((0, false)))
        {
            partial_probes += 1;
            if let Some(partial) = stream_controller.partial_call(0) {
                if let Some(tool_id) = partial.tool_id.as_deref() {
                    let _ =
                        broker.validate_partial_probe(request, tool_id, &partial.arguments_json);
                }
            }
        }

        match validated.event {
            ModelStreamEvent::TextDelta { content } => {
                if let Err(error) = response.text(content) {
                    model_cancellation.cancel();
                    fail_candidate_stream(controller, candidate.take());
                    return Err(error);
                }
            }
            ModelStreamEvent::ToolCallDelta(delta) => {
                if let Some(call) = validated.completed_tool_call {
                    let index = delta.index;
                    if let Err(error) = response.call(index, call.clone()) {
                        model_cancellation.cancel();
                        fail_candidate_stream(controller, candidate.take());
                        return Err(error);
                    }
                    if index == 0 && !candidate_considered {
                        candidate_considered = true;
                        if let Some(key) =
                            broker.validate_shadow_candidate(controller, request, &call)
                        {
                            candidate = match controller.mode(&call.tool_id) {
                                SpeculationMode::Shadow => {
                                    Some(CandidateState::Shadow { call, key })
                                }
                                SpeculationMode::Active => match controller.try_acquire_slot() {
                                    Ok(slot) => {
                                        let deadline =
                                            match candidate_deadline_for(controller, call_deadline)
                                            {
                                                Ok(deadline) => deadline,
                                                Err(error) => {
                                                    model_cancellation.cancel();
                                                    return Err(error);
                                                }
                                            };
                                        match broker
                                            .speculate(
                                                controller,
                                                request,
                                                call.clone(),
                                                model_call_number,
                                                run_id,
                                                trace_id,
                                                Some(deadline),
                                                slot,
                                            )
                                            .await
                                        {
                                            SpeculativeAttempt::NotIssued => None,
                                            SpeculativeAttempt::Issued(Ok(execution)) => {
                                                Some(CandidateState::Active {
                                                    call,
                                                    key,
                                                    execution,
                                                    deadline,
                                                })
                                            }
                                            SpeculativeAttempt::Issued(Err(_)) => None,
                                        }
                                    }
                                    Err(_) => {
                                        controller.record_slot_saturated(&call.tool_id);
                                        None
                                    }
                                },
                                SpeculationMode::Disabled => None,
                            };
                        }
                    }
                }
            }
            ModelStreamEvent::Completed { model, usage } => {
                if let Err(error) = response.complete(model, usage) {
                    model_cancellation.cancel();
                    fail_candidate_stream(controller, candidate.take());
                    return Err(error);
                }
                break;
            }
        }
    }

    let response = match response.finish() {
        Ok(response) => response,
        Err(error) => {
            model_cancellation.cancel();
            fail_candidate_stream(controller, candidate.take());
            return Err(error);
        }
    };
    let authoritative = response.tool_calls.first();
    let speculative = match candidate {
        Some(CandidateState::Shadow { call, key }) => {
            if invocation_matches(&call, &key, authoritative) {
                controller.record_shadow_match(&call.tool_id);
            } else {
                controller.record_mismatch_and_trip(&call.tool_id);
            }
            None
        }
        Some(CandidateState::Active {
            call,
            key,
            execution,
            deadline,
        }) => {
            if Instant::now() >= deadline {
                None
            } else if invocation_matches(&call, &key, authoritative) {
                Some(ReadySpeculativeCommit {
                    execution: Some(execution),
                    deadline,
                })
            } else {
                controller.record_mismatch_and_trip(&call.tool_id);
                None
            }
        }
        None => None,
    };
    Ok(StreamedReactiveResponse {
        response,
        speculative,
    })
}

fn assembly_limits(
    run: &AgentLimits,
    provider: &ProviderCapabilityLimits,
    tool_count: usize,
) -> Result<ToolCallAssemblyLimits, HarnessError> {
    let mut limits = ToolCallAssemblyLimits::for_provider(provider)?;
    let argument_bytes = bounded_usize(run.max_tool_arguments_bytes);
    let request_bytes = bounded_usize(run.max_request_payload_bytes);
    let response_bytes = bounded_usize(run.max_model_response_bytes);
    limits.max_calls = limits.max_calls.min(run.max_tool_calls as usize);
    limits.max_argument_bytes = limits.max_argument_bytes.min(argument_bytes);
    limits.max_call_bytes = limits
        .max_call_bytes
        .min(argument_bytes.saturating_add(limits.max_field_bytes.saturating_mul(2)));
    limits.max_total_buffered_bytes = limits
        .max_total_buffered_bytes
        .min(request_bytes)
        .min(response_bytes);
    limits.max_json_depth = limits.max_json_depth.min(run.max_json_depth);
    limits.max_allowed_tools = limits.max_allowed_tools.min(tool_count.max(1));
    limits.max_aggregate_schema_bytes = limits.max_aggregate_schema_bytes.min(request_bytes);
    limits.validate()?;
    Ok(limits)
}

fn bounded_usize(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn candidate_deadline_for(
    controller: &SpeculationController,
    call_deadline: Option<Instant>,
) -> Result<Instant, HarnessError> {
    let candidate = Instant::now()
        .checked_add(Duration::from_millis(
            controller.config().max_execution_duration_ms,
        ))
        .ok_or_else(|| HarnessError::InvalidRequest("speculation duration is too large".into()))?;
    Ok(call_deadline.map_or(candidate, |deadline| deadline.min(candidate)))
}

fn invocation_matches(
    candidate_call: &ToolCall,
    candidate_key: &InvocationKey,
    authoritative: Option<&ToolCall>,
) -> bool {
    let Some(authoritative) = authoritative else {
        return false;
    };
    let Ok(arguments) = serde_json::from_str(&authoritative.arguments_json) else {
        return false;
    };
    candidate_call.id == authoritative.id
        && *candidate_key == InvocationKey::new(&authoritative.tool_id, &arguments)
}

fn discard_candidate(
    _controller: &SpeculationController,
    candidate: Option<CandidateState>,
    cancelled: bool,
) {
    if let Some(CandidateState::Active { mut execution, .. }) = candidate {
        if cancelled {
            execution.settle(SpeculativeResolution::Cancelled);
        }
    }
}

fn fail_candidate_stream(controller: &SpeculationController, candidate: Option<CandidateState>) {
    match candidate {
        Some(CandidateState::Shadow { call, .. }) => {
            controller.record_terminal_stream_failure_and_trip(&call.tool_id);
        }
        Some(CandidateState::Active {
            call,
            mut execution,
            ..
        }) => {
            controller.record_terminal_stream_failure_and_trip(&call.tool_id);
            execution.settle(SpeculativeResolution::Discarded);
        }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{
        invocation_matches, SpeculationConfig, SpeculationController, SpeculationMode,
        HARD_MAX_SPECULATION_DURATION_MS, HARD_MAX_SPECULATION_STREAM_EVENTS,
        MIN_SPECULATION_SHADOW_OBSERVATIONS,
    };
    use crate::{broker::InvocationKey, ToolCall};
    use serde_json::json;
    use tokio::time::Instant;

    #[test]
    fn speculation_config_boundaries_are_exact() {
        let valid = [
            SpeculationConfig {
                max_execution_duration_ms: 1,
                required_shadow_observations: MIN_SPECULATION_SHADOW_OBSERVATIONS,
                max_stream_events: 1,
            },
            SpeculationConfig {
                max_execution_duration_ms: HARD_MAX_SPECULATION_DURATION_MS,
                required_shadow_observations: u64::MAX,
                max_stream_events: HARD_MAX_SPECULATION_STREAM_EVENTS,
            },
        ];
        for config in valid {
            assert!(config.validate().is_ok());
        }

        let invalid = [
            SpeculationConfig {
                max_execution_duration_ms: 0,
                ..SpeculationConfig::default()
            },
            SpeculationConfig {
                max_execution_duration_ms: HARD_MAX_SPECULATION_DURATION_MS + 1,
                ..SpeculationConfig::default()
            },
            SpeculationConfig {
                required_shadow_observations: MIN_SPECULATION_SHADOW_OBSERVATIONS - 1,
                ..SpeculationConfig::default()
            },
            SpeculationConfig {
                max_stream_events: 0,
                ..SpeculationConfig::default()
            },
            SpeculationConfig {
                max_stream_events: HARD_MAX_SPECULATION_STREAM_EVENTS + 1,
                ..SpeculationConfig::default()
            },
        ];
        for config in invalid {
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn activation_and_incidents_are_scoped_per_tool() {
        let controller = SpeculationController::new(SpeculationConfig::default());
        controller.register_tool("first");
        controller.register_tool("second");
        for _ in 0..1_000 {
            controller.record_shadow_match("first");
            controller.record_shadow_match("second");
        }
        assert_eq!(controller.activate("first").mode, SpeculationMode::Active);
        assert_eq!(controller.activate("second").mode, SpeculationMode::Active);

        controller.record_mismatch_and_trip("first");
        assert_eq!(controller.mode("first"), SpeculationMode::Shadow);
        assert_eq!(controller.mode("second"), SpeculationMode::Active);
        assert_eq!(controller.metrics("first").mismatches, 1);
        for _ in 0..1_000 {
            controller.record_shadow_match("first");
        }
        assert!(!controller.readiness("first").ready_to_activate);
        assert_eq!(controller.activate("first").mode, SpeculationMode::Shadow);

        assert!(controller.record_commit_if_active(
            "second",
            Instant::now() + std::time::Duration::from_secs(1),
            None,
            &tokio_util::sync::CancellationToken::new(),
        ));
        let metrics = controller.metrics("second");
        assert_eq!(metrics.issued, 1);
        assert_eq!(metrics.committed, 1);
        assert_eq!(
            metrics.issued,
            metrics.committed + metrics.discarded + metrics.cancelled
        );
    }

    #[test]
    fn invocation_matching_is_typed_exact_and_call_occurrence_bound() {
        let candidate = ToolCall::new("call-0", "read", r#"{"a":1,"b":true}"#);
        let key = InvocationKey::new("read", &json!({"a":1,"b":true}));
        assert!(invocation_matches(
            &candidate,
            &key,
            Some(&ToolCall::new("call-0", "read", r#"{"b":true,"a":1}"#))
        ));
        assert!(!invocation_matches(
            &candidate,
            &key,
            Some(&ToolCall::new(
                "different-call",
                "read",
                r#"{"a":1,"b":true}"#
            ))
        ));
        assert!(!invocation_matches(
            &candidate,
            &key,
            Some(&ToolCall::new("call-0", "other", r#"{"a":1,"b":true}"#))
        ));
        assert!(!invocation_matches(
            &candidate,
            &key,
            Some(&ToolCall::new("call-0", "read", r#"{"a":"1","b":true}"#))
        ));
        let negative_zero = ToolCall::new("call-zero", "read", "-0.0");
        let negative_key = InvocationKey::new("read", &json!(-0.0));
        assert!(!invocation_matches(
            &negative_zero,
            &negative_key,
            Some(&ToolCall::new("call-zero", "read", "0.0"))
        ));
    }

    #[test]
    fn runner_wide_slot_is_nonblocking_and_has_no_queue() {
        let controller = SpeculationController::new(SpeculationConfig::default());
        let held = controller.try_acquire_slot().unwrap();
        assert!(controller.try_acquire_slot().is_err());
        drop(held);
        assert!(controller.try_acquire_slot().is_ok());
    }
}
