use crate::{
    limits::{compile_trusted_schema, ensure_json_depth, serialized_len},
    AgentLimits, HarnessError, ToolCall, ToolDefinition, Usage,
};
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
    /// Maximum bytes retained across all incomplete calls.
    pub max_total_buffered_bytes: usize,
    /// Maximum bytes accepted for either call identifier field.
    pub max_field_bytes: usize,
    /// Maximum JSON nesting depth accepted for completed arguments.
    pub max_json_depth: u32,
}

impl Default for ToolCallAssemblyLimits {
    fn default() -> Self {
        let limits = AgentLimits::default();
        Self {
            max_calls: limits.max_tool_calls as usize,
            max_call_bytes: limits.max_tool_arguments_bytes as usize + 2 * 1024,
            max_total_buffered_bytes: limits.max_request_payload_bytes as usize,
            max_field_bytes: 1024,
            max_json_depth: limits.max_json_depth,
        }
    }
}

impl ToolCallAssemblyLimits {
    /// Validates that every assembly limit is nonzero.
    pub fn validate(&self) -> Result<(), HarnessError> {
        if self.max_calls == 0
            || self.max_call_bytes == 0
            || self.max_total_buffered_bytes == 0
            || self.max_field_bytes == 0
            || self.max_json_depth == 0
        {
            return Err(HarnessError::InvalidRequest(
                "tool-call assembly limits must be greater than zero".into(),
            ));
        }
        Ok(())
    }
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
        let mut compiled = HashMap::new();
        for definition in allowed_tools {
            let id = definition.id;
            if id.trim().is_empty() || id.len() > limits.max_field_bytes {
                return Err(HarnessError::InvalidTool(
                    "allowed streamed tool id is empty or too large".into(),
                ));
            }
            if compiled.contains_key(&id) {
                return Err(HarnessError::InvalidTool(format!(
                    "duplicate allowed streamed tool: {id}"
                )));
            }
            if serialized_len(&definition.arguments_schema)?
                > schema_limits.max_request_payload_bytes
            {
                return Err(HarnessError::InvalidTool(format!(
                    "schema for {id} exceeds trusted schema limit"
                )));
            }
            ensure_json_depth(
                "streamed tool schema",
                &definition.arguments_schema,
                schema_limits.max_json_depth,
            )
            .map_err(|error| HarnessError::InvalidTool(error.to_string()))?;
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
