use crate::{
    limits::{compile_trusted_schema, ensure_json_depth, serialized_len},
    AgentLimits, HarnessError, ProviderCapabilityLimits, ToolCall, ToolDefinition, Usage,
};

const MAX_ASSEMBLY_CALLS: usize = 16;
const MAX_ASSEMBLY_ARGUMENT_BYTES: usize = 64 * 1024;
const MAX_ASSEMBLY_CALL_BYTES: usize = MAX_ASSEMBLY_ARGUMENT_BYTES + 2 * 1024;
const MAX_ASSEMBLY_BUFFERED_BYTES: usize = 256 * 1024;
const MAX_ASSEMBLY_FIELD_BYTES: usize = 1024;
const MAX_ASSEMBLY_JSON_DEPTH: u32 = 64;
const MAX_ASSEMBLY_ALLOWED_TOOLS: usize = 1024;
const MAX_ASSEMBLY_SCHEMA_BYTES: usize = 256 * 1024;
use futures_core::Stream;
use jsonschema::Validator;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    pin::Pin,
    sync::Arc,
};

/// Provider-neutral stream of model response events.
pub type ModelEventStream =
    Pin<Box<dyn Stream<Item = Result<ModelStreamEvent, HarnessError>> + Send>>;

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
/// Ephemeral events emitted while a provider produces a model response.
pub enum ModelStreamEvent {
    /// A fragment of assistant text.
    TextDelta {
        /// Text received in this response fragment.
        content: String,
    },
    /// An incremental fragment of one model-requested tool call.
    ToolCallDelta(ToolCallDelta),
    /// The response completed with provider-reported model and usage metadata.
    Completed {
        /// Model name reported by the provider.
        model: String,
        /// Token usage reported by the provider.
        usage: Usage,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
/// Incremental provider data for one indexed tool call.
pub struct ToolCallDelta {
    /// Provider-local call index used to interleave multiple calls.
    pub index: usize,
    /// Stable call identifier, when first supplied by the provider.
    pub call_id: Option<String>,
    /// Stable registered tool identifier, when first supplied by the provider.
    pub tool_id: Option<String>,
    /// JSON argument bytes appended by this fragment.
    pub arguments_fragment: String,
    /// Whether this is the final fragment for the indexed call.
    pub is_final: bool,
}

impl ToolCallDelta {
    /// Creates one incremental tool-call fragment.
    pub fn new(index: usize, arguments_fragment: impl Into<String>, is_final: bool) -> Self {
        Self {
            index,
            call_id: None,
            tool_id: None,
            arguments_fragment: arguments_fragment.into(),
            is_final,
        }
    }

    /// Supplies the immutable provider call identifier.
    pub fn with_call_id(mut self, call_id: impl Into<String>) -> Self {
        self.call_id = Some(call_id.into());
        self
    }

    /// Supplies the immutable registered tool identifier.
    pub fn with_tool_id(mut self, tool_id: impl Into<String>) -> Self {
        self.tool_id = Some(tool_id.into());
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
/// Resource limits for bounded incremental tool-call assembly.
pub struct ToolCallAssemblyLimits {
    /// Maximum distinct call indices accepted in one response.
    pub max_calls: usize,
    /// Maximum bytes retained for one call, including identifiers and arguments.
    pub max_call_bytes: usize,
    /// Maximum JSON argument bytes retained for one call.
    pub max_argument_bytes: usize,
    /// Maximum bytes retained across all incomplete calls.
    pub max_total_buffered_bytes: usize,
    /// Maximum bytes accepted for either call identifier field.
    pub max_field_bytes: usize,
    /// Maximum JSON nesting depth accepted for completed arguments.
    pub max_json_depth: u32,
    /// Maximum tool definitions accepted by one assembler.
    pub max_allowed_tools: usize,
    /// Maximum aggregate serialized argument-schema bytes accepted by one assembler.
    pub max_aggregate_schema_bytes: usize,
}

impl Default for ToolCallAssemblyLimits {
    fn default() -> Self {
        let limits = AgentLimits::default();
        Self {
            max_calls: limits.max_tool_calls as usize,
            max_call_bytes: MAX_ASSEMBLY_CALL_BYTES,
            max_argument_bytes: limits.max_tool_arguments_bytes as usize,
            max_total_buffered_bytes: limits.max_request_payload_bytes as usize,
            max_field_bytes: 1024,
            max_json_depth: limits.max_json_depth,
            max_allowed_tools: MAX_ASSEMBLY_ALLOWED_TOOLS,
            max_aggregate_schema_bytes: limits.max_request_payload_bytes as usize,
        }
    }
}

impl ToolCallAssemblyLimits {
    /// Validates that every assembly limit is nonzero.
    pub fn validate(&self) -> Result<(), HarnessError> {
        if self.max_calls == 0
            || self.max_call_bytes == 0
            || self.max_argument_bytes == 0
            || self.max_total_buffered_bytes == 0
            || self.max_field_bytes == 0
            || self.max_json_depth == 0
            || self.max_allowed_tools == 0
            || self.max_aggregate_schema_bytes == 0
        {
            return Err(HarnessError::InvalidRequest(
                "tool-call assembly limits must be greater than zero".into(),
            ));
        }
        if self.max_calls > MAX_ASSEMBLY_CALLS
            || self.max_call_bytes > MAX_ASSEMBLY_CALL_BYTES
            || self.max_argument_bytes > MAX_ASSEMBLY_ARGUMENT_BYTES
            || self.max_total_buffered_bytes > MAX_ASSEMBLY_BUFFERED_BYTES
            || self.max_field_bytes > MAX_ASSEMBLY_FIELD_BYTES
            || self.max_json_depth > MAX_ASSEMBLY_JSON_DEPTH
            || self.max_allowed_tools > MAX_ASSEMBLY_ALLOWED_TOOLS
            || self.max_aggregate_schema_bytes > MAX_ASSEMBLY_SCHEMA_BYTES
            || self.max_argument_bytes > self.max_call_bytes
        {
            return Err(HarnessError::InvalidRequest(
                "tool-call assembly limits exceed immutable local hard limits".into(),
            ));
        }
        Ok(())
    }

    /// Derives effective local limits from conservative defaults and provider caps.
    pub fn for_provider(capabilities: &ProviderCapabilityLimits) -> Result<Self, HarnessError> {
        let mut limits = Self::default();
        if let Some(value) = capabilities.max_tools {
            limits.max_allowed_tools =
                checked_provider_cap("maximum tools", value as u64, MAX_ASSEMBLY_ALLOWED_TOOLS)?;
        }
        if let Some(value) = capabilities.max_tool_schema_bytes {
            limits.max_aggregate_schema_bytes = checked_provider_cap(
                "maximum tool schema bytes",
                value,
                MAX_ASSEMBLY_SCHEMA_BYTES,
            )?;
        }
        if let Some(value) = capabilities.max_streamed_tool_calls {
            limits.max_calls = checked_provider_cap(
                "maximum streamed tool calls",
                value as u64,
                MAX_ASSEMBLY_CALLS,
            )?;
        }
        if let Some(value) = capabilities.max_streamed_argument_bytes {
            limits.max_argument_bytes = checked_provider_cap(
                "maximum streamed argument bytes",
                value,
                MAX_ASSEMBLY_ARGUMENT_BYTES,
            )?;
        }
        limits.validate()?;
        Ok(limits)
    }
}

fn checked_provider_cap(
    label: &str,
    value: u64,
    local_hard_limit: usize,
) -> Result<usize, HarnessError> {
    let value = usize::try_from(value).map_err(|_| {
        HarnessError::InvalidRequest(format!("provider {label} does not fit this platform"))
    })?;
    if value == 0 || value > local_hard_limit {
        return Err(HarnessError::InvalidRequest(format!(
            "provider {label} must be between 1 and {local_hard_limit}"
        )));
    }
    Ok(value)
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
/// Observable but ephemeral state for an incomplete streamed tool call.
///
/// Partial arguments are never executable and must not be persisted as a [`crate::RunEvent`].
pub struct PartialToolCall {
    /// Provider-local call index.
    pub index: usize,
    /// Stable call identifier observed so far.
    pub call_id: Option<String>,
    /// Stable registered tool identifier observed so far.
    pub tool_id: Option<String>,
    /// Unvalidated partial JSON argument text.
    pub arguments_json: String,
}

struct AllowedTool {
    validator: Arc<Validator>,
}

#[derive(Default)]
struct PendingCall {
    call_id: Option<String>,
    tool_id: Option<String>,
    arguments_json: String,
    buffered_bytes: usize,
}

/// Bounded assembler that validates final streamed tool calls without executing them.
pub struct ToolCallAssembler {
    allowed_tools: HashMap<String, AllowedTool>,
    limits: ToolCallAssemblyLimits,
    pending: HashMap<usize, PendingCall>,
    seen_indices: HashSet<usize>,
    finalized_indices: HashSet<usize>,
    call_ids: HashSet<String>,
    total_buffered_bytes: usize,
}

impl ToolCallAssembler {
    /// Creates an assembler from the exact tool definitions allowed for one model request.
    pub fn new(
        allowed_tools: impl IntoIterator<Item = ToolDefinition>,
        limits: ToolCallAssemblyLimits,
    ) -> Result<Self, HarnessError> {
        limits.validate()?;
        let schema_limits = AgentLimits::default();
        let mut definitions = Vec::with_capacity(limits.max_allowed_tools.min(64));
        for definition in allowed_tools {
            if definitions.len() >= limits.max_allowed_tools {
                return Err(HarnessError::ResourceLimit(format!(
                    "streamed tool catalog exceeds {} tools",
                    limits.max_allowed_tools
                )));
            }
            definitions.push(definition);
        }
        let mut seen_ids = HashSet::with_capacity(definitions.len());
        let mut aggregate_schema_bytes = 0_usize;
        for definition in &definitions {
            if definition.id.trim().is_empty() || definition.id.len() > limits.max_field_bytes {
                return Err(HarnessError::InvalidTool(
                    "allowed streamed tool id is empty or too large".into(),
                ));
            }
            if !seen_ids.insert(definition.id.as_str()) {
                return Err(HarnessError::InvalidTool(format!(
                    "duplicate allowed streamed tool: {}",
                    definition.id
                )));
            }
            let schema_bytes = usize::try_from(serialized_len(&definition.arguments_schema)?)
                .map_err(|_| {
                    HarnessError::ResourceLimit(
                        "streamed tool schema size does not fit this platform".into(),
                    )
                })?;
            aggregate_schema_bytes = checked_schema_total(aggregate_schema_bytes, schema_bytes)?;
            if schema_bytes > schema_limits.max_request_payload_bytes as usize {
                return Err(HarnessError::InvalidTool(format!(
                    "schema for {} exceeds trusted schema limit",
                    definition.id
                )));
            }
            if aggregate_schema_bytes > limits.max_aggregate_schema_bytes {
                return Err(HarnessError::ResourceLimit(format!(
                    "streamed tool schemas exceed {} aggregate bytes",
                    limits.max_aggregate_schema_bytes
                )));
            }
            ensure_json_depth(
                "streamed tool schema",
                &definition.arguments_schema,
                schema_limits.max_json_depth,
            )
            .map_err(|error| HarnessError::InvalidTool(error.to_string()))?;
        }

        let mut compiled = HashMap::with_capacity(definitions.len());
        for definition in definitions {
            let id = definition.id;
            let validator = compile_trusted_schema(&definition.arguments_schema, |error| {
                HarnessError::InvalidTool(format!("invalid schema for {id}: {error}"))
            })?;
            compiled.insert(
                id,
                AllowedTool {
                    validator: Arc::new(validator),
                },
            );
        }
        Ok(Self {
            allowed_tools: compiled,
            limits,
            pending: HashMap::new(),
            seen_indices: HashSet::new(),
            finalized_indices: HashSet::new(),
            call_ids: HashSet::new(),
            total_buffered_bytes: 0,
        })
    }

    /// Returns the number of incomplete tool calls currently buffered.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Validates that no incomplete calls remain at stream completion or EOF.
    pub fn finish(&self) -> Result<(), HarnessError> {
        if self.pending.is_empty() {
            Ok(())
        } else {
            Err(HarnessError::Provider(format!(
                "model stream ended with {} incomplete tool call(s)",
                self.pending.len()
            )))
        }
    }

    /// Returns ephemeral partial JSON for an incomplete call index.
    pub fn partial_call(&self, index: usize) -> Option<PartialToolCall> {
        self.pending.get(&index).map(|call| PartialToolCall {
            index,
            call_id: call.call_id.clone(),
            tool_id: call.tool_id.clone(),
            arguments_json: call.arguments_json.clone(),
        })
    }

    /// Returns the bytes currently retained across incomplete calls.
    pub fn buffered_bytes(&self) -> usize {
        self.total_buffered_bytes
    }

    /// Adds a fragment and yields a validated call only when its final fragment arrives.
    pub fn push(&mut self, delta: ToolCallDelta) -> Result<Option<ToolCall>, HarnessError> {
        if self.finalized_indices.contains(&delta.index) {
            return Err(HarnessError::Provider(format!(
                "tool call {} received a fragment after completion",
                delta.index
            )));
        }
        if !self.seen_indices.contains(&delta.index) {
            if self.seen_indices.len() >= self.limits.max_calls {
                return Err(HarnessError::ResourceLimit(format!(
                    "streamed tool calls exceed {}",
                    self.limits.max_calls
                )));
            }
            self.seen_indices.insert(delta.index);
        }

        validate_field(
            "call id",
            delta.call_id.as_deref(),
            self.limits.max_field_bytes,
        )?;
        validate_field(
            "tool id",
            delta.tool_id.as_deref(),
            self.limits.max_field_bytes,
        )?;
        let current = self.pending.get(&delta.index);
        check_immutable_field(
            "call id",
            current.and_then(|pending| pending.call_id.as_deref()),
            delta.call_id.as_deref(),
        )?;
        check_immutable_field(
            "tool id",
            current.and_then(|pending| pending.tool_id.as_deref()),
            delta.tool_id.as_deref(),
        )?;
        let added_bytes = delta.arguments_fragment.len()
            + new_field_bytes(
                current.and_then(|pending| pending.call_id.as_deref()),
                delta.call_id.as_deref(),
            )
            + new_field_bytes(
                current.and_then(|pending| pending.tool_id.as_deref()),
                delta.tool_id.as_deref(),
            );
        let current_call_bytes = current.map_or(0, |pending| pending.buffered_bytes);
        let current_argument_bytes = current.map_or(0, |pending| pending.arguments_json.len());
        let next_argument_bytes = current_argument_bytes
            .checked_add(delta.arguments_fragment.len())
            .ok_or_else(|| {
                HarnessError::ResourceLimit("streamed tool argument size overflow".into())
            })?;
        if next_argument_bytes > self.limits.max_argument_bytes {
            return Err(HarnessError::ResourceLimit(format!(
                "streamed tool arguments {} exceed {} bytes",
                delta.index, self.limits.max_argument_bytes
            )));
        }
        let next_call_bytes = current_call_bytes.checked_add(added_bytes).ok_or_else(|| {
            HarnessError::ResourceLimit("streamed tool call size overflow".into())
        })?;
        if next_call_bytes > self.limits.max_call_bytes {
            return Err(HarnessError::ResourceLimit(format!(
                "streamed tool call {} exceeds {} bytes",
                delta.index, self.limits.max_call_bytes
            )));
        }
        let next_total = self
            .total_buffered_bytes
            .checked_add(added_bytes)
            .ok_or_else(|| HarnessError::ResourceLimit("streamed tool buffer overflow".into()))?;
        if next_total > self.limits.max_total_buffered_bytes {
            return Err(HarnessError::ResourceLimit(format!(
                "streamed tool buffer exceeds {} bytes",
                self.limits.max_total_buffered_bytes
            )));
        }
        let pending = self.pending.entry(delta.index).or_default();
        if pending.call_id.is_none() {
            pending.call_id = delta.call_id;
        }
        if pending.tool_id.is_none() {
            pending.tool_id = delta.tool_id;
        }
        pending.arguments_json.push_str(&delta.arguments_fragment);
        pending.buffered_bytes = next_call_bytes;
        self.total_buffered_bytes = next_total;

        if !delta.is_final {
            return Ok(None);
        }

        let pending = self
            .pending
            .remove(&delta.index)
            .expect("pending call exists");
        self.total_buffered_bytes = self
            .total_buffered_bytes
            .saturating_sub(pending.buffered_bytes);
        self.finalized_indices.insert(delta.index);
        let call_id = required_field("call id", pending.call_id)?;
        let tool_id = required_field("tool id", pending.tool_id)?;
        if !self.call_ids.insert(call_id.clone()) {
            return Err(HarnessError::Provider(format!(
                "duplicate streamed tool call id: {call_id}"
            )));
        }
        let allowed = self.allowed_tools.get(&tool_id).ok_or_else(|| {
            HarnessError::InvalidTool(format!("unknown streamed tool: {tool_id}"))
        })?;
        let arguments: Value = serde_json::from_str(&pending.arguments_json).map_err(|error| {
            HarnessError::InvalidArguments(format!("streamed tool arguments are not JSON: {error}"))
        })?;
        ensure_json_depth(
            "streamed tool arguments",
            &arguments,
            self.limits.max_json_depth,
        )?;
        allowed
            .validator
            .validate(&arguments)
            .map_err(|error| HarnessError::InvalidArguments(error.to_string()))?;
        Ok(Some(ToolCall::new(
            call_id,
            tool_id,
            pending.arguments_json,
        )))
    }
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
/// One stream event after fused terminal-state and tool-call validation.
pub struct ValidatedModelStreamEvent {
    /// The original ephemeral provider-neutral stream event.
    pub event: ModelStreamEvent,
    /// A complete validated tool call produced by a final delta, when present.
    pub completed_tool_call: Option<ToolCall>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelStreamState {
    Active,
    Completed,
    Failed,
}

/// Fused controller that makes the first completion, error, or cancellation terminal.
pub struct ModelStreamController {
    assembler: ToolCallAssembler,
    state: ModelStreamState,
}

impl ModelStreamController {
    /// Creates an active controller around a bounded tool-call assembler.
    pub fn new(assembler: ToolCallAssembler) -> Self {
        Self {
            assembler,
            state: ModelStreamState::Active,
        }
    }

    /// Validates one stream item and permanently fuses the controller on terminal state.
    pub fn push(
        &mut self,
        event: Result<ModelStreamEvent, HarnessError>,
    ) -> Result<ValidatedModelStreamEvent, HarnessError> {
        match self.state {
            ModelStreamState::Completed => {
                return Err(HarnessError::Provider(
                    "model stream received an event after completion".into(),
                ));
            }
            ModelStreamState::Failed => {
                return Err(HarnessError::Provider(
                    "model stream received an event after terminal failure".into(),
                ));
            }
            ModelStreamState::Active => {}
        }

        let event = match event {
            Ok(event) => event,
            Err(error) => {
                self.state = ModelStreamState::Failed;
                return Err(error);
            }
        };
        let completed_tool_call = match &event {
            ModelStreamEvent::ToolCallDelta(delta) => match self.assembler.push(delta.clone()) {
                Ok(call) => call,
                Err(error) => {
                    self.state = ModelStreamState::Failed;
                    return Err(error);
                }
            },
            ModelStreamEvent::Completed { .. } => {
                if let Err(error) = self.assembler.finish() {
                    self.state = ModelStreamState::Failed;
                    return Err(error);
                }
                self.state = ModelStreamState::Completed;
                None
            }
            ModelStreamEvent::TextDelta { .. } => None,
        };
        Ok(ValidatedModelStreamEvent {
            event,
            completed_tool_call,
        })
    }

    /// Finalizes the controller when the underlying transport reaches EOF.
    pub fn finish_eof(&mut self) -> Result<(), HarnessError> {
        match self.state {
            ModelStreamState::Completed => Ok(()),
            ModelStreamState::Failed => Err(HarnessError::Provider(
                "model stream reached EOF after terminal failure".into(),
            )),
            ModelStreamState::Active => {
                if let Err(error) = self.assembler.finish() {
                    self.state = ModelStreamState::Failed;
                    return Err(error);
                }
                self.state = ModelStreamState::Failed;
                Err(HarnessError::Provider(
                    "model stream reached EOF before completion".into(),
                ))
            }
        }
    }

    /// Returns whether completion or failure has fused the stream.
    pub fn is_terminal(&self) -> bool {
        self.state != ModelStreamState::Active
    }
}

fn checked_schema_total(current: usize, next: usize) -> Result<usize, HarnessError> {
    current
        .checked_add(next)
        .ok_or_else(|| HarnessError::ResourceLimit("streamed tool schema size overflow".into()))
}

fn validate_field(label: &str, value: Option<&str>, max_bytes: usize) -> Result<(), HarnessError> {
    if value.is_some_and(|value| value.is_empty() || value.len() > max_bytes) {
        return Err(HarnessError::Provider(format!(
            "streamed {label} is empty or exceeds {max_bytes} bytes"
        )));
    }
    Ok(())
}

fn check_immutable_field(
    label: &str,
    current: Option<&str>,
    incoming: Option<&str>,
) -> Result<(), HarnessError> {
    if matches!((current, incoming), (Some(current), Some(incoming)) if current != incoming) {
        return Err(HarnessError::Provider(format!(
            "conflicting streamed {label}"
        )));
    }
    Ok(())
}

fn new_field_bytes(current: Option<&str>, incoming: Option<&str>) -> usize {
    if current.is_none() {
        incoming.map_or(0, str::len)
    } else {
        0
    }
}

fn required_field(label: &str, value: Option<String>) -> Result<String, HarnessError> {
    value.ok_or_else(|| HarnessError::Provider(format!("final streamed {label} is missing")))
}

#[cfg(test)]
mod tests {
    use super::checked_schema_total;
    use crate::HarnessError;

    #[test]
    fn aggregate_schema_accounting_rejects_integer_overflow() {
        assert!(matches!(
            checked_schema_total(usize::MAX, 1),
            Err(HarnessError::ResourceLimit(message)) if message.contains("overflow")
        ));
    }
}
