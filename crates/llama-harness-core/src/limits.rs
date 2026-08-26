use crate::HarnessError;
use jsonschema::{Retrieve, Uri, Validator};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io;

const DEFAULT_MAX_INPUT_BYTES: u64 = 64 * 1024;
const DEFAULT_MAX_REQUEST_PAYLOAD_BYTES: u64 = 256 * 1024;
const DEFAULT_MAX_MODEL_RESPONSE_BYTES: u64 = 1024 * 1024;
const DEFAULT_MAX_TOOL_ARGUMENTS_BYTES: u64 = 64 * 1024;
const DEFAULT_MAX_TOOL_RESULT_BYTES: u64 = 1024 * 1024;
const DEFAULT_MAX_TRANSCRIPT_BYTES: u64 = 4 * 1024 * 1024;
const DEFAULT_MAX_JSON_DEPTH: u32 = 64;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
/// Resource and payload limits applied to an agent run.
pub struct AgentLimits {
    /// Maximum number of model calls in one run.
    pub max_model_calls: u32,
    /// Maximum number of tool calls in one run.
    pub max_tool_calls: u32,
    /// Maximum number of identical consecutive tool calls.
    pub max_identical_tool_calls: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional wall-clock limit for the complete run, in milliseconds.
    pub max_run_duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional wall-clock limit for each model call, in milliseconds.
    pub max_model_call_duration_ms: Option<u64>,
    /// Maximum number of structured-output repair attempts.
    pub max_output_repairs: u32,
    /// Retry budget per logical model turn. Only `RetryableProvider` errors qualify.
    pub max_provider_retries: u32,
    /// Maximum input transcript size in bytes.
    pub max_input_bytes: u64,
    /// Maximum serialized model request size in bytes.
    pub max_request_payload_bytes: u64,
    /// Maximum serialized model response size in bytes.
    pub max_model_response_bytes: u64,
    /// Maximum serialized tool argument size in bytes.
    pub max_tool_arguments_bytes: u64,
    /// Maximum serialized tool result size in bytes.
    pub max_tool_result_bytes: u64,
    /// Maximum transcript size in bytes.
    pub max_transcript_bytes: u64,
    /// Maximum nesting depth for JSON values.
    pub max_json_depth: u32,
}

impl Default for AgentLimits {
    fn default() -> Self {
        Self {
            max_model_calls: 8,
            max_tool_calls: 16,
            max_identical_tool_calls: 2,
            max_run_duration_ms: None,
            max_model_call_duration_ms: None,
            max_output_repairs: 1,
            max_provider_retries: 2,
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            max_request_payload_bytes: DEFAULT_MAX_REQUEST_PAYLOAD_BYTES,
            max_model_response_bytes: DEFAULT_MAX_MODEL_RESPONSE_BYTES,
            max_tool_arguments_bytes: DEFAULT_MAX_TOOL_ARGUMENTS_BYTES,
            max_tool_result_bytes: DEFAULT_MAX_TOOL_RESULT_BYTES,
            max_transcript_bytes: DEFAULT_MAX_TRANSCRIPT_BYTES,
            max_json_depth: DEFAULT_MAX_JSON_DEPTH,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
/// Optional sampling and output-token settings for model generation.
pub struct GenerationOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Sampling temperature, when supported by the provider.
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Nucleus sampling probability, when supported by the provider.
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Maximum number of output tokens, when supported by the provider.
    pub max_output_tokens: Option<u32>,
}

struct RejectExternalReferences;

impl Retrieve for RejectExternalReferences {
    fn retrieve(
        &self,
        uri: &Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err(io::Error::other(format!("external schema reference is disabled: {uri}")).into())
    }
}

pub(crate) fn compile_trusted_schema(
    schema: &Value,
    error: impl Fn(String) -> HarnessError,
) -> Result<Validator, HarnessError> {
    validate_schema_references(schema).map_err(&error)?;
    jsonschema::options()
        .with_retriever(RejectExternalReferences)
        .build(schema)
        .map_err(|validation_error| error(validation_error.to_string()))
}

fn validate_schema_references(schema: &Value) -> Result<(), String> {
    match schema {
        Value::Object(values) => {
            for (key, value) in values {
                if matches!(key.as_str(), "$ref" | "$dynamicRef" | "$recursiveRef") {
                    let reference = value
                        .as_str()
                        .ok_or_else(|| format!("{key} must be a string"))?;
                    if !reference.starts_with('#') {
                        return Err(format!(
                            "external schema reference is disabled: {reference}"
                        ));
                    }
                }
                validate_schema_references(value)?;
            }
            Ok(())
        }
        Value::Array(values) => {
            for value in values {
                validate_schema_references(value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

pub(crate) fn serialized_len(value: &impl Serialize) -> Result<u64, HarnessError> {
    serde_json::to_vec(value)
        .map(|bytes| bytes.len() as u64)
        .map_err(|error| {
            HarnessError::InvalidRequest(format!("payload is not serializable: {error}"))
        })
}

pub(crate) fn json_depth(value: &Value) -> u32 {
    match value {
        Value::Array(values) => 1 + values.iter().map(json_depth).max().unwrap_or(0),
        Value::Object(values) => 1 + values.values().map(json_depth).max().unwrap_or(0),
        _ => 0,
    }
}

pub(crate) fn ensure_json_depth(
    label: &str,
    value: &Value,
    max_depth: u32,
) -> Result<(), HarnessError> {
    if json_depth(value) > max_depth {
        return Err(HarnessError::ResourceLimit(format!(
            "{label} exceeds maximum JSON depth of {max_depth}"
        )));
    }
    Ok(())
}
