use crate::{AssertionFailure, EvalExpected, EvalObservation, ExpectedFailure, ExpectedToolCall};
use serde_json::Value;

/// Evaluates all configured expectations against one observed run.
pub fn evaluate_expectations(
    expected: &EvalExpected,
    observation: &EvalObservation,
) -> Vec<AssertionFailure> {
    let mut failures = Vec::new();
    let result = &observation.run;
    if let Some(status) = &expected.status {
        if result.status != *status {
            failure(
                &mut failures,
                "status",
                format!("expected status {status:?}, got {:?}", result.status),
            );
        }
    }
    if let Some(output) = &expected.final_output_equals {
        if result.final_output.as_deref() != Some(output) {
            failure(
                &mut failures,
                "final_output_equals",
                "final output did not exactly match expected output",
            );
        }
    }
    for required_text in &expected.final_output_contains {
        if !result
            .final_output
            .as_deref()
            .is_some_and(|output| output.contains(required_text))
        {
            failure(
                &mut failures,
                "final_output_contains",
                format!("final output did not contain {required_text:?}"),
            );
        }
    }
    if let Some(expected_json) = &expected.structured_output_subset {
        match result
            .final_output
            .as_deref()
            .map(serde_json::from_str::<Value>)
        {
            Some(Ok(actual_json)) if is_json_subset(expected_json, &actual_json) => {}
            Some(Ok(_)) => failure(
                &mut failures,
                "structured_output_subset",
                "final JSON output did not contain the expected subset",
            ),
            Some(Err(_)) => failure(
                &mut failures,
                "structured_output_subset",
                "final output was not valid JSON",
            ),
            None => failure(
                &mut failures,
                "structured_output_subset",
                "final output was missing",
            ),
        }
    }

    let tool_ids: Vec<&str> = result
        .tool_calls
        .iter()
        .map(|call| call.tool_id.as_str())
        .collect();
    for required_tool in &expected.required_tools {
        if !tool_ids.iter().any(|tool| *tool == required_tool) {
            failure(
                &mut failures,
                "required_tools",
                format!("required tool {required_tool:?} was not called"),
            );
        }
    }
    for forbidden_tool in &expected.forbidden_tools {
        if tool_ids.iter().any(|tool| *tool == forbidden_tool) {
            failure(
                &mut failures,
                "forbidden_tools",
                format!("forbidden tool {forbidden_tool:?} was called"),
            );
        }
    }
    if let Some(expected_sequence) = &expected.tool_sequence {
        if tool_ids
            != expected_sequence
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        {
            failure(
                &mut failures,
                "tool_sequence",
                format!("expected tool sequence {expected_sequence:?}, got {tool_ids:?}"),
            );
        }
    }
    for expected_call in &expected.expected_tool_arguments {
        evaluate_tool_arguments(&mut failures, expected_call, observation);
    }

    if let Some(expected_state) = &expected.final_state_subset {
        match observation.final_state.as_ref() {
            Some(actual) if is_json_subset(expected_state, actual) => {}
            Some(_) => failure(
                &mut failures,
                "final_state_subset",
                "final state did not contain the expected subset",
            ),
            None => failure(
                &mut failures,
                "final_state_subset",
                "executor did not return a final state snapshot",
            ),
        }
    }
    if let Some(expected_items) = &expected.unresolved_items {
        match observation.unresolved_items.as_ref() {
            Some(actual) if is_json_subset(expected_items, actual) => {}
            Some(_) => failure(
                &mut failures,
                "unresolved_items",
                "unresolved items did not contain the expected subset",
            ),
            None => failure(
                &mut failures,
                "unresolved_items",
                "executor did not return unresolved items",
            ),
        }
    }

    let approved_tools: Vec<&str> = result
        .approvals
        .iter()
        .filter(|approval| approval.granted)
        .map(|approval| approval.tool_id.as_str())
        .collect();
    for tool in &expected.required_approval_tools {
        if !approved_tools.iter().any(|approved| *approved == tool) {
            failure(
                &mut failures,
                "required_approval_tools",
                format!("tool {tool:?} did not receive recorded approval"),
            );
        }
    }
    for tool in &expected.forbidden_approval_tools {
        if approved_tools.iter().any(|approved| *approved == tool) {
            failure(
                &mut failures,
                "forbidden_approval_tools",
                format!("tool {tool:?} received approval"),
            );
        }
    }
    if let Some(max_calls) = expected.max_model_calls {
        if observation.model_calls > max_calls {
            failure(
                &mut failures,
                "max_model_calls",
                format!(
                    "model calls {} exceeded {max_calls}",
                    observation.model_calls
                ),
            );
        }
    }
    if let Some(max_calls) = expected.max_tool_calls {
        let calls = result.tool_calls.len() as u32;
        if calls > max_calls {
            failure(
                &mut failures,
                "max_tool_calls",
                format!("tool calls {calls} exceeded {max_calls}"),
            );
        }
    }
    if let Some(max_latency) = expected.max_latency_ms {
        if result.duration_ms > max_latency {
            failure(
                &mut failures,
                "max_latency_ms",
                format!("duration {}ms exceeded {max_latency}ms", result.duration_ms),
            );
        }
    }
    if let Some(expected_cancelled) = expected.expect_cancelled {
        if result.cancelled != expected_cancelled {
            failure(
                &mut failures,
                "expect_cancelled",
                format!(
                    "expected cancelled={expected_cancelled}, got {}",
                    result.cancelled
                ),
            );
        }
    }
    if let Some(expected_failure) = &expected.expected_failure {
        evaluate_expected_failure(&mut failures, expected_failure, observation);
    }

    failures
}

/// Returns whether every value in `expected` is present in `actual`.
pub fn is_json_subset(expected: &Value, actual: &Value) -> bool {
    match (expected, actual) {
        (Value::Object(expected), Value::Object(actual)) => {
            expected.iter().all(|(key, expected)| {
                actual
                    .get(key)
                    .is_some_and(|actual| is_json_subset(expected, actual))
            })
        }
        (Value::Array(expected), Value::Array(actual)) => expected
            .iter()
            .all(|expected| actual.iter().any(|actual| is_json_subset(expected, actual))),
        _ => expected == actual,
    }
}

fn evaluate_tool_arguments(
    failures: &mut Vec<AssertionFailure>,
    expected: &ExpectedToolCall,
    observation: &EvalObservation,
) {
    let matching_calls = observation
        .run
        .tool_calls
        .iter()
        .filter(|call| call.tool_id == expected.tool_id);
    let mut valid = false;
    let mut had_invalid_json = false;
    for call in matching_calls {
        match serde_json::from_str::<Value>(&call.arguments_json) {
            Ok(arguments) if is_json_subset(&expected.arguments_subset, &arguments) => {
                valid = true;
                break;
            }
            Ok(_) => {}
            Err(_) => had_invalid_json = true,
        }
    }
    if !valid {
        let message = if had_invalid_json {
            format!("tool {} had malformed recorded arguments", expected.tool_id)
        } else {
            format!(
                "no {} call contained the expected argument subset",
                expected.tool_id
            )
        };
        failure(failures, "expected_tool_arguments", message);
    }
}

fn evaluate_expected_failure(
    failures: &mut Vec<AssertionFailure>,
    expected: &ExpectedFailure,
    observation: &EvalObservation,
) {
    let matching = observation.run.errors.iter().any(|error| {
        expected
            .code
            .as_ref()
            .is_none_or(|code| error.code == *code)
            && expected
                .message_contains
                .as_ref()
                .is_none_or(|message| error.message.contains(message))
    });
    if !matching {
        failure(
            failures,
            "expected_failure",
            "run errors did not match expected failure metadata",
        );
    }
}

fn failure(failures: &mut Vec<AssertionFailure>, rule: &str, message: impl Into<String>) {
    failures.push(AssertionFailure {
        rule: rule.into(),
        message: message.into(),
    });
}
