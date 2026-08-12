use std::collections::HashSet;

use schemars::{schema_for, Schema};
use serde::de::{DeserializeSeed, Error as _, MapAccess, SeqAccess, Visitor};
use serde_json::Value;
use thiserror::Error;

use crate::{
    Envelope, ProtocolMessage, ProtocolVersion, WireToolResult, MAX_ERROR_MESSAGE_BYTES,
    MAX_IDENTIFIER_BYTES, MAX_JSON_DEPTH, MAX_MESSAGE_BYTES,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProtocolValidationError {
    #[error("protocol message exceeds {MAX_MESSAGE_BYTES} bytes")]
    MessageTooLarge,
    #[error("protocol message exceeds maximum JSON depth of {MAX_JSON_DEPTH}")]
    JsonTooDeep,
    #[error("invalid JSON: {0}")]
    InvalidJson(String),
    #[error("invalid envelope: {0}")]
    InvalidEnvelope(String),
    #[error("duplicate JSON object key: {0}")]
    DuplicateKey(String),
    #[error("unknown protocol message type: {0}")]
    UnknownMessageType(String),
    #[error("incompatible protocol version: runtime supports {supported}, peer sent {received}")]
    IncompatibleVersion {
        supported: ProtocolVersion,
        received: ProtocolVersion,
    },
}

/// Parses one complete JSONL line with the v1 resource limits and compatibility rules.
pub fn decode_line(line: &[u8]) -> Result<Envelope, ProtocolValidationError> {
    if line.len() > MAX_MESSAGE_BYTES {
        return Err(ProtocolValidationError::MessageTooLarge);
    }

    let mut duplicate_check = serde_json::Deserializer::from_slice(line);
    NoDuplicateKeys
        .deserialize(&mut duplicate_check)
        .map_err(|error| duplicate_key_or_json_error(error.to_string()))?;

    let value: Value = serde_json::from_slice(line)
        .map_err(|error| ProtocolValidationError::InvalidJson(error.to_string()))?;
    if json_depth(&value) > MAX_JSON_DEPTH {
        return Err(ProtocolValidationError::JsonTooDeep);
    }
    let message_type = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| ProtocolValidationError::InvalidEnvelope("type is required".into()))?
        .to_owned();
    let envelope: Envelope = serde_json::from_value(value).map_err(|error| {
        if error.to_string().contains("unknown variant") {
            ProtocolValidationError::UnknownMessageType(message_type)
        } else {
            ProtocolValidationError::InvalidEnvelope(error.to_string())
        }
    })?;
    validate_envelope(&envelope)?;
    Ok(envelope)
}

pub fn validate_envelope(envelope: &Envelope) -> Result<(), ProtocolValidationError> {
    if !ProtocolVersion::V1.is_compatible_with(envelope.protocol_version) {
        return Err(ProtocolValidationError::IncompatibleVersion {
            supported: ProtocolVersion::V1,
            received: envelope.protocol_version,
        });
    }
    validate_identifier("request_id", &envelope.request_id)?;
    if let Some(run_id) = &envelope.run_id {
        validate_identifier("run_id", run_id)?;
    }
    match &envelope.message {
        ProtocolMessage::ClientHello(hello) => {
            validate_identifier("sdk.name", &hello.sdk.name)?;
            validate_identifier("sdk.version", &hello.sdk.version)?;
        }
        ProtocolMessage::ToolResult(response) => {
            validate_identifier("callback_id", &response.callback_id)?;
            validate_tool_result(&response.result)?;
        }
        ProtocolMessage::PolicyDecision(response) => {
            validate_identifier("callback_id", &response.callback_id)?
        }
        ProtocolMessage::ApprovalDecision(response) => {
            validate_identifier("callback_id", &response.callback_id)?
        }
        ProtocolMessage::Ping(ping) => validate_identifier("nonce", &ping.nonce)?,
        ProtocolMessage::Pong(pong) => validate_identifier("nonce", &pong.nonce)?,
        _ => {}
    }
    Ok(())
}

pub fn envelope_schema() -> Schema {
    schema_for!(Envelope)
}

pub fn envelope_schema_json() -> String {
    serde_json::to_string_pretty(&envelope_schema()).expect("schemas are serializable") + "\n"
}

fn validate_identifier(label: &str, value: &str) -> Result<(), ProtocolValidationError> {
    if value.trim().is_empty() {
        return Err(ProtocolValidationError::InvalidEnvelope(format!(
            "{label} is required"
        )));
    }
    if value.len() > MAX_IDENTIFIER_BYTES {
        return Err(ProtocolValidationError::InvalidEnvelope(format!(
            "{label} exceeds {MAX_IDENTIFIER_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_tool_result(result: &WireToolResult) -> Result<(), ProtocolValidationError> {
    match (result.ok, &result.error) {
        (true, None) => Ok(()),
        (true, Some(_)) => Err(ProtocolValidationError::InvalidEnvelope(
            "successful tool result must not contain an error".into(),
        )),
        (false, Some(error))
            if !error.trim().is_empty() && error.len() <= MAX_ERROR_MESSAGE_BYTES =>
        {
            Ok(())
        }
        (false, Some(_)) => Err(ProtocolValidationError::InvalidEnvelope(format!(
            "failed tool result error must be non-empty and at most {MAX_ERROR_MESSAGE_BYTES} bytes"
        ))),
        (false, None) => Err(ProtocolValidationError::InvalidEnvelope(
            "failed tool result must contain an error".into(),
        )),
    }
}

fn duplicate_key_or_json_error(message: String) -> ProtocolValidationError {
    if let Some(key) = message.strip_prefix("duplicate JSON object key: ") {
        ProtocolValidationError::DuplicateKey(key.split(" at line").next().unwrap_or(key).into())
    } else {
        ProtocolValidationError::InvalidJson(message)
    }
}

fn json_depth(value: &Value) -> u32 {
    match value {
        Value::Object(values) => 1 + values.values().map(json_depth).max().unwrap_or(0),
        Value::Array(values) => 1 + values.iter().map(json_depth).max().unwrap_or(0),
        _ => 0,
    }
}

struct NoDuplicateKeys;

impl<'de> DeserializeSeed<'de> for NoDuplicateKeys {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for NoDuplicateKeys {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("valid JSON without duplicate object keys")
    }

    fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_u64<E>(self, _: u64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_str<E>(self, _: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_string<E>(self, _: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(())
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence.next_element_seed(NoDuplicateKeys)?.is_some() {}
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(A::Error::custom(format!(
                    "duplicate JSON object key: {key}"
                )));
            }
            map.next_value_seed(NoDuplicateKeys)?;
        }
        Ok(())
    }
}
