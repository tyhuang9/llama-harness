use std::{fs, path::Path};

use llama_harness_protocol::{
    decode_line, envelope_schema_json, ClientHello, ClientIdentity, Envelope, ProtocolMessage,
    ProtocolValidationError, ProtocolVersion, RuntimeCapabilities, RuntimeHello,
    ToolResultResponse, WireToolResult, MAX_MESSAGE_BYTES,
};

fn fixture(name: &str) -> String {
    fs::read_to_string(protocol_root().join("fixtures").join(name))
        .expect("fixture must be checked in")
}

fn protocol_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("protocol")
}

#[test]
fn serializes_the_client_hello_golden_fixture() {
    let message = Envelope::new(
        "hello-1",
        None,
        ProtocolMessage::ClientHello(ClientHello {
            sdk: ClientIdentity {
                name: "@llama-harness/sdk".into(),
                version: "0.1.0".into(),
            },
            capabilities: ["async_callbacks".into()].into_iter().collect(),
        }),
    );

    assert_eq!(
        serde_json::to_string_pretty(&message).unwrap() + "\n",
        fixture("client_hello.json")
    );
    assert_eq!(
        decode_line(fixture("client_hello.json").as_bytes()).unwrap(),
        message
    );
}

#[test]
fn serializes_the_runtime_hello_golden_fixture() {
    let message = Envelope::new(
        "hello-1",
        None,
        ProtocolMessage::RuntimeHello(RuntimeHello {
            runtime_version: "0.1.0".into(),
            capabilities: RuntimeCapabilities {
                supports_output_deltas: false,
                supports_structured_output: true,
                supports_trace_persistence: true,
                concurrent_runs: 16,
                max_pending_callbacks: 128,
                max_queue_depth: 256,
            },
            providers: vec!["ollama".into()],
        }),
    );

    assert_eq!(
        serde_json::to_string_pretty(&message).unwrap() + "\n",
        fixture("runtime_hello.json")
    );
    assert_eq!(
        decode_line(fixture("runtime_hello.json").as_bytes()).unwrap(),
        message
    );
}

#[test]
fn compatible_minor_versions_and_optional_fields_are_accepted() {
    let message = r#"{
  "protocol_version": "1.99",
  "request_id": "hello-1",
  "type": "ping",
  "payload": { "nonce": "ping-1", "future_optional_field": true },
  "future_envelope_field": true
}"#;

    let parsed = decode_line(message.as_bytes()).unwrap();
    assert_eq!(
        parsed.protocol_version,
        ProtocolVersion {
            major: 1,
            minor: 99
        }
    );
}

#[test]
fn incompatible_major_unknown_messages_and_bounds_are_rejected() {
    let incompatible = r#"{"protocol_version":"2.0","request_id":"hello-1","type":"ping","payload":{"nonce":"ping-1"}}"#;
    assert!(matches!(
        decode_line(incompatible.as_bytes()),
        Err(ProtocolValidationError::IncompatibleVersion { .. })
    ));

    let unknown =
        r#"{"protocol_version":"1.0","request_id":"hello-1","type":"future_command","payload":{}}"#;
    assert!(matches!(
        decode_line(unknown.as_bytes()),
        Err(ProtocolValidationError::UnknownMessageType(message)) if message == "future_command"
    ));

    let oversized = vec![b'x'; MAX_MESSAGE_BYTES + 1];
    assert_eq!(
        decode_line(&oversized),
        Err(ProtocolValidationError::MessageTooLarge)
    );

    let duplicate_key = r#"{"protocol_version":"1.0","request_id":"first","request_id":"second","type":"ping","payload":{"nonce":"ping-1"}}"#;
    assert!(matches!(
        decode_line(duplicate_key.as_bytes()),
        Err(ProtocolValidationError::DuplicateKey(key)) if key == "request_id"
    ));
}

#[test]
fn tool_result_has_exactly_one_semantic_outcome() {
    let invalid = Envelope::new(
        "callback-1",
        None,
        ProtocolMessage::ToolResult(ToolResultResponse {
            callback_id: "callback-1".into(),
            result: WireToolResult {
                ok: true,
                output: serde_json::Value::Null,
                error: Some("must not coexist with success".into()),
            },
        }),
    );
    let encoded = serde_json::to_vec(&invalid).unwrap();
    assert!(matches!(
        decode_line(&encoded),
        Err(ProtocolValidationError::InvalidEnvelope(message))
            if message == "successful tool result must not contain an error"
    ));
}

#[test]
fn generated_schema_is_deterministic_and_the_checked_in_envelope_schema_is_versioned() {
    assert_eq!(envelope_schema_json(), envelope_schema_json());
    let schema: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(protocol_root().join("schema/v1-envelope.schema.json"))
            .expect("schema must be checked in"),
    )
    .expect("schema must be JSON");
    assert_eq!(
        schema["$id"],
        "https://llama-harness.dev/protocol/v1/envelope.schema.json"
    );
    assert_eq!(
        schema["properties"]["protocol_version"]["pattern"],
        "^1\\.[0-9]+$"
    );
}
