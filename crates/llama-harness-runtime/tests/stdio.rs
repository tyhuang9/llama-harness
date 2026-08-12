use std::{
    io::Write,
    process::{Command, Stdio},
};

use llama_harness_protocol::{ProtocolErrorCode, ProtocolMessage};

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
