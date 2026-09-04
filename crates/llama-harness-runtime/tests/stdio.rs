use std::{
    io::Write,
    process::{Command, Stdio},
};

use llama_harness_protocol::{ProtocolErrorCode, ProtocolMessage, ProtocolVersion};

fn run_with_stdin(line: &str) -> Vec<llama_harness_protocol::Envelope> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_llama-harness-runtime"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("runtime binary must start");
    child
        .stdin
        .take()
        .expect("runtime stdin must be piped")
        .write_all(line.as_bytes())
        .expect("runtime stdin write must succeed");
    let output = child
        .wait_with_output()
        .expect("runtime process must exit after parent pipe closes");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("protocol stdout is UTF-8 JSONL")
        .lines()
        .map(|line| {
            llama_harness_protocol::decode_line(line.as_bytes()).expect("valid protocol output")
        })
        .collect()
}

#[test]
fn stdio_handshake_uses_protocol_stdout_only() {
    let output = run_with_stdin(
        "{\"protocol_version\":\"1.0\",\"request_id\":\"hello-1\",\"type\":\"client_hello\",\"payload\":{\"sdk\":{\"name\":\"runtime-test\",\"version\":\"0.1.0\"},\"capabilities\":[]}}\n",
    );
    assert!(
        matches!(output.as_slice(), [envelope] if matches!(envelope.message, ProtocolMessage::RuntimeHello(_)))
    );
}

#[test]
fn stdio_negotiates_the_client_offered_minor_and_pins_runtime_hello() {
    for (offered, selected) in [
        ("1.0", ProtocolVersion::V1_0),
        ("1.99", ProtocolVersion::V1_1),
    ] {
        let output = run_with_stdin(&format!(
            "{{\"protocol_version\":\"{offered}\",\"request_id\":\"hello-1\",\"type\":\"client_hello\",\"payload\":{{\"sdk\":{{\"name\":\"runtime-test\",\"version\":\"0.1.0\"}},\"capabilities\":[]}}}}\n"
        ));
        assert!(matches!(
            output.as_slice(),
            [envelope]
                if envelope.protocol_version == selected
                    && matches!(envelope.message, ProtocolMessage::RuntimeHello(_))
        ));
    }
}

#[test]
fn post_handshake_version_drift_is_rejected_before_the_command_is_processed() {
    let output = run_with_stdin(
        "{\"protocol_version\":\"1.1\",\"request_id\":\"hello-1\",\"type\":\"client_hello\",\"payload\":{\"sdk\":{\"name\":\"runtime-test\",\"version\":\"0.1.0\"},\"capabilities\":[]}}\n{\"protocol_version\":\"1.0\",\"request_id\":\"ping-1\",\"type\":\"ping\",\"payload\":{\"nonce\":\"n\"}}\n",
    );
    assert!(matches!(
        output.as_slice(),
        [hello, error]
            if hello.protocol_version == ProtocolVersion::V1_1
                && matches!(hello.message, ProtocolMessage::RuntimeHello(_))
                && matches!(
                    &error.message,
                    ProtocolMessage::ProtocolError(payload)
                        if payload.code == ProtocolErrorCode::IncompatibleVersion
                )
    ));
}

#[test]
fn malformed_major_and_unknown_type_preserve_structured_error_codes() {
    let major = run_with_stdin(
        "{\"protocol_version\":\"2.0\",\"request_id\":\"major-1\",\"type\":\"ping\",\"payload\":{\"nonce\":\"n\"}}\n",
    );
    let unknown = run_with_stdin(
        "{\"protocol_version\":\"1.0\",\"request_id\":\"unknown-1\",\"type\":\"future_command\",\"payload\":{}}\n",
    );
    assert!(matches!(
        major.as_slice(),
        [envelope] if matches!(&envelope.message, ProtocolMessage::ProtocolError(error) if error.code == ProtocolErrorCode::IncompatibleVersion)
    ));
    assert!(matches!(
        unknown.as_slice(),
        [envelope] if matches!(&envelope.message, ProtocolMessage::ProtocolError(error) if error.code == ProtocolErrorCode::UnknownMessageType)
    ));
}

#[test]
fn forced_advanced_strategy_on_v1_0_fails_without_starting_a_tool_effect() {
    let output = run_with_stdin(
        "{\"protocol_version\":\"1.0\",\"request_id\":\"hello-1\",\"type\":\"client_hello\",\"payload\":{\"sdk\":{\"name\":\"runtime-test\",\"version\":\"0.1.0\"},\"capabilities\":[]}}\n{\"protocol_version\":\"1.0\",\"request_id\":\"run-1\",\"type\":\"start_run\",\"payload\":{\"request\":{\"provider\":{\"kind\":\"ollama\",\"base_url\":\"http://127.0.0.1:9\"},\"agent\":{\"id\":\"agent\",\"name\":\"Agent\",\"version\":\"1\",\"default_model\":\"model\"},\"input\":\"hello\",\"strategy\":\"declarative_plan\"}}}\n",
    );
    assert!(output.iter().all(|envelope| {
        !matches!(envelope.message, ProtocolMessage::ToolExecutionRequested(_))
    }));
    assert!(output.iter().any(|envelope| {
        matches!(
            &envelope.message,
            ProtocolMessage::RunFailed(error) if error.error.code == "unsupported_strategy"
        )
    }));
}

#[test]
fn commands_before_the_handshake_fail_closed() {
    let output = run_with_stdin(
        "{\"protocol_version\":\"1.0\",\"request_id\":\"ping-1\",\"type\":\"ping\",\"payload\":{\"nonce\":\"n\"}}\n",
    );
    assert!(matches!(
        output.as_slice(),
        [envelope] if matches!(
            &envelope.message,
            ProtocolMessage::ProtocolError(error) if error.code == ProtocolErrorCode::InvalidState
        )
    ));
}
