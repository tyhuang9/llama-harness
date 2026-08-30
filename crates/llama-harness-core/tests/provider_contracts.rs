use async_trait::async_trait;
use llama_harness_core::{
    mock::MockModelProvider, CancellationSafety, ExecutionLocation, HarnessError, IssueSafety,
    ModelCapabilities, ModelProvider, ModelRequest, ModelStreamController, ModelStreamEvent,
    ModelStreamFailureKind, NetworkEgress, ProviderCapabilityLimits, SpeculationPolicy, Tool,
    ToolCallAssembler, ToolCallAssemblyLimits, ToolCallDelta, ToolCaller, ToolDefinition,
    ToolRegistry, ToolResult, Usage,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

struct TestTool(ToolDefinition);

#[async_trait]
impl Tool for TestTool {
    fn definition(&self) -> &ToolDefinition {
        &self.0
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        Ok(ToolResult::success(Value::Null))
    }
}

fn register(definition: ToolDefinition) -> Result<(), HarnessError> {
    let mut registry = ToolRegistry::default();
    registry.register(Arc::new(TestTool(definition)))
}

#[test]
fn tool_metadata_defaults_are_conservative() {
    let definition = ToolDefinition::new("read", "Read", "Read", json!({"type":"object"}));
    assert_eq!(definition.allowed_callers, [ToolCaller::Direct].into());
    assert!(!definition.parallel_safe);
    assert_eq!(definition.cancellation_safety, CancellationSafety::Unknown);
    assert_eq!(definition.speculation_policy, SpeculationPolicy::Disabled);
    assert!(definition.output_schema.is_none());

    let serialized = serde_json::to_value(&definition).unwrap();
    let mut legacy = serialized.as_object().unwrap().clone();
    for key in [
        "output_schema",
        "parallel_safe",
        "concurrency_key",
        "cancellation_safety",
        "expected_latency_ms",
        "allowed_callers",
        "speculation_policy",
        "issue_safety",
        "execution_location",
        "network_egress",
    ] {
        legacy.remove(key);
    }
    let decoded: ToolDefinition = serde_json::from_value(Value::Object(legacy)).unwrap();
    assert_eq!(decoded.allowed_callers, [ToolCaller::Direct].into());
    assert_eq!(decoded.speculation_policy, SpeculationPolicy::Disabled);
    assert_eq!(decoded.issue_safety, IssueSafety::Unknown);
    assert_eq!(decoded.execution_location, ExecutionLocation::Unknown);
    assert_eq!(decoded.network_egress, NetworkEgress::Unknown);
}

#[test]
fn registry_rejects_untrusted_output_schemas() {
    let definition = ToolDefinition::new("read", "Read", "Read", json!({"type":"object"}))
        .with_output_schema(json!({"$ref":"https://attacker.invalid/schema.json"}));
    assert!(
        matches!(register(definition), Err(HarnessError::InvalidTool(message)) if message.contains("external schema reference"))
    );
}

#[test]
fn speculative_registration_is_fail_closed() {
    let unsafe_definition =
        ToolDefinition::new("unsafe", "Unsafe", "Unsafe", json!({"type":"object"}))
            .with_allowed_callers([ToolCaller::Direct, ToolCaller::Speculative])
            .with_speculation_policy(SpeculationPolicy::Enabled)
            .with_read_only(true)
            .with_idempotent(true)
            .with_parallel_safe(true)
            .with_cancellation_safety(CancellationSafety::Cooperative);
    assert!(
        matches!(register(unsafe_definition), Err(HarnessError::InvalidTool(message)) if message.contains("not eligible"))
    );

    let safe_definition = ToolDefinition::new("safe", "Safe", "Safe", json!({"type":"object"}))
        .with_allowed_callers([ToolCaller::Direct, ToolCaller::Speculative])
        .with_speculation_policy(SpeculationPolicy::Enabled)
        .with_read_only(true)
        .with_idempotent(true)
        .with_parallel_safe(true)
        .with_cancellation_safety(CancellationSafety::Guaranteed)
        .with_issue_safety(IssueSafety::Guaranteed)
        .with_execution_location(ExecutionLocation::LocalPrivate)
        .with_network_egress(NetworkEgress::Prohibited);
    register(safe_definition).unwrap();
}

#[test]
fn speculation_requires_issue_time_and_privacy_guarantees() {
    let base = ToolDefinition::new(
        "candidate",
        "Candidate",
        "Candidate",
        json!({"type":"object"}),
    )
    .with_allowed_callers([ToolCaller::Direct, ToolCaller::Speculative])
    .with_speculation_policy(SpeculationPolicy::Enabled)
    .with_read_only(true)
    .with_idempotent(true)
    .with_parallel_safe(true)
    .with_cancellation_safety(CancellationSafety::Guaranteed);

    assert!(
        matches!(register(base.clone()), Err(HarnessError::InvalidTool(message)) if message.contains("not eligible"))
    );
    assert!(matches!(
        register(
            base.clone()
                .with_issue_safety(IssueSafety::Guaranteed)
                .with_execution_location(ExecutionLocation::Remote)
                .with_network_egress(NetworkEgress::Prohibited)
        ),
        Err(HarnessError::InvalidTool(message)) if message.contains("not eligible")
    ));
    assert!(matches!(
        register(
            base.with_issue_safety(IssueSafety::Guaranteed)
                .with_execution_location(ExecutionLocation::LocalPrivate)
                .with_network_egress(NetworkEgress::Permitted)
        ),
        Err(HarnessError::InvalidTool(message)) if message.contains("not eligible")
    ));
}

#[test]
fn new_model_capabilities_remain_false_unless_advertised() {
    let capabilities = ModelCapabilities::new(true, true, false);
    assert!(!capabilities.supports_strict_tool_schemas);
    assert!(!capabilities.supports_streaming_tool_arguments);
    assert!(!capabilities.supports_parallel_tool_calls);
    assert!(!capabilities.supports_structured_plans);
    assert!(!capabilities.supports_programmatic_calling);
    assert_eq!(capabilities.limits, ProviderCapabilityLimits::default());

    let limits = ProviderCapabilityLimits::new()
        .with_max_tools(100)
        .with_max_streamed_argument_bytes(65_536)
        .with_max_plan_nodes(32);
    let advertised = capabilities
        .with_strict_tool_schemas(true)
        .with_streaming_tool_arguments(true)
        .with_parallel_tool_calls(true)
        .with_structured_plans(true)
        .with_programmatic_calling(true)
        .with_limits(limits.clone());
    assert!(advertised.supports_programmatic_calling);
    assert_eq!(advertised.limits, limits);
}

fn assembler(limits: ToolCallAssemblyLimits) -> ToolCallAssembler {
    ToolCallAssembler::new(
        [ToolDefinition::new(
            "lookup",
            "Lookup",
            "Lookup",
            json!({
                "type":"object",
                "required":["query"],
                "properties":{"query":{"type":"string"}},
                "additionalProperties":false
            }),
        )],
        limits,
    )
    .unwrap()
}

fn assert_stream_failure(error: HarnessError, kind: ModelStreamFailureKind) {
    assert_eq!(error, HarnessError::ModelStream { kind });
    assert_eq!(error.to_string(), kind.message());
    assert!(kind.code().starts_with("model_stream."));
    for sentinel in [
        "sentinel-call-id",
        "sentinel-tool-id",
        "sentinel-arguments",
        "sentinel-streamed-argument-secret",
        "sentinel-provider-message",
    ] {
        assert!(!error.to_string().contains(sentinel));
        assert!(!kind.code().contains(sentinel));
    }
}

#[test]
fn assembler_interleaves_calls_and_only_yields_validated_finals() {
    let mut assembler = assembler(ToolCallAssemblyLimits::default());
    assert!(assembler
        .push(
            ToolCallDelta::new(0, "{\"query\":", false)
                .with_call_id("call-0")
                .with_tool_id("lookup")
        )
        .unwrap()
        .is_none());
    assert!(assembler
        .push(
            ToolCallDelta::new(1, "{\"query\":\"second\"}", true)
                .with_call_id("call-1")
                .with_tool_id("lookup")
        )
        .unwrap()
        .is_some());
    assert_eq!(
        assembler.partial_call(0).unwrap().arguments_json,
        "{\"query\":"
    );
    let call = assembler
        .push(ToolCallDelta::new(0, "\"first\"}", true))
        .unwrap()
        .unwrap();
    assert_eq!(call.id, "call-0");
    assert_eq!(call.tool_id, "lookup");
    assert_eq!(call.arguments_json, "{\"query\":\"first\"}");
    assert_eq!(assembler.buffered_bytes(), 0);
}

#[test]
fn assembler_rejects_identifier_failures_with_exact_value_free_kinds() {
    let mut invalid_call_id = assembler(ToolCallAssemblyLimits::default());
    assert_stream_failure(
        invalid_call_id
            .push(
                ToolCallDelta::new(0, "{}", true)
                    .with_call_id("")
                    .with_tool_id("lookup"),
            )
            .unwrap_err(),
        ModelStreamFailureKind::InvalidCallId,
    );

    let mut invalid_tool_id = assembler(ToolCallAssemblyLimits::default());
    assert_stream_failure(
        invalid_tool_id
            .push(
                ToolCallDelta::new(0, "{}", true)
                    .with_call_id("sentinel-call-id")
                    .with_tool_id(""),
            )
            .unwrap_err(),
        ModelStreamFailureKind::InvalidToolId,
    );

    let mut conflicting_call_id = assembler(ToolCallAssemblyLimits::default());
    conflicting_call_id
        .push(
            ToolCallDelta::new(0, "{", false)
                .with_call_id("sentinel-call-id")
                .with_tool_id("lookup"),
        )
        .unwrap();
    assert_stream_failure(
        conflicting_call_id
            .push(ToolCallDelta::new(0, "}", true).with_call_id("changed"))
            .unwrap_err(),
        ModelStreamFailureKind::ConflictingCallId,
    );

    let mut conflicting_tool_id = assembler(ToolCallAssemblyLimits::default());
    conflicting_tool_id
        .push(
            ToolCallDelta::new(0, "{", false)
                .with_call_id("call")
                .with_tool_id("lookup"),
        )
        .unwrap();
    assert_stream_failure(
        conflicting_tool_id
            .push(ToolCallDelta::new(0, "}", true).with_tool_id("sentinel-tool-id"))
            .unwrap_err(),
        ModelStreamFailureKind::ConflictingToolId,
    );

    let mut missing_call_id = assembler(ToolCallAssemblyLimits::default());
    assert_stream_failure(
        missing_call_id
            .push(ToolCallDelta::new(0, "{}", true).with_tool_id("lookup"))
            .unwrap_err(),
        ModelStreamFailureKind::MissingCallId,
    );

    let mut missing_tool_id = assembler(ToolCallAssemblyLimits::default());
    assert_stream_failure(
        missing_tool_id
            .push(ToolCallDelta::new(0, "{}", true).with_call_id("sentinel-call-id"))
            .unwrap_err(),
        ModelStreamFailureKind::MissingToolId,
    );

    let mut duplicate_call_id = assembler(ToolCallAssemblyLimits::default());
    duplicate_call_id
        .push(
            ToolCallDelta::new(0, r#"{"query":"first"}"#, true)
                .with_call_id("sentinel-call-id")
                .with_tool_id("lookup"),
        )
        .unwrap();
    assert_stream_failure(
        duplicate_call_id
            .push(
                ToolCallDelta::new(1, r#"{"query":"second"}"#, true)
                    .with_call_id("sentinel-call-id")
                    .with_tool_id("lookup"),
            )
            .unwrap_err(),
        ModelStreamFailureKind::DuplicateCallId,
    );
}

#[test]
fn assembler_rejects_invalid_finals_with_exact_value_free_kinds() {
    let mut unknown = assembler(ToolCallAssemblyLimits::default());
    assert_stream_failure(
        unknown
            .push(
                ToolCallDelta::new(0, "{}", true)
                    .with_call_id("sentinel-call-id")
                    .with_tool_id("sentinel-tool-id"),
            )
            .unwrap_err(),
        ModelStreamFailureKind::UnknownTool,
    );

    let mut malformed = assembler(ToolCallAssemblyLimits::default());
    assert_stream_failure(
        malformed
            .push(
                ToolCallDelta::new(0, "{sentinel-arguments", true)
                    .with_call_id("sentinel-call-id")
                    .with_tool_id("lookup"),
            )
            .unwrap_err(),
        ModelStreamFailureKind::MalformedArgumentsJson,
    );

    let mut mismatch = assembler(ToolCallAssemblyLimits::default());
    assert_stream_failure(
        mismatch
            .push(
                ToolCallDelta::new(0, r#"{"query":1}"#, true)
                    .with_call_id("sentinel-call-id")
                    .with_tool_id("lookup"),
            )
            .unwrap_err(),
        ModelStreamFailureKind::ArgumentsSchemaMismatch,
    );

    let mut finalized = assembler(ToolCallAssemblyLimits::default());
    finalized
        .push(
            ToolCallDelta::new(0, r#"{"query":"ok"}"#, true)
                .with_call_id("sentinel-call-id")
                .with_tool_id("lookup"),
        )
        .unwrap();
    assert_stream_failure(
        finalized
            .push(ToolCallDelta::new(0, "sentinel-arguments", false))
            .unwrap_err(),
        ModelStreamFailureKind::FragmentAfterCallCompletion,
    );
}

#[test]
fn streamed_argument_validation_errors_redact_instance_values() {
    const SECRET: &str = "sentinel-streamed-argument-secret";
    let assembler = ToolCallAssembler::new(
        [ToolDefinition::new(
            "lookup",
            "Lookup",
            "Lookup",
            json!({
                "type":"object",
                "required":["query"],
                "properties":{"query":{"type":"string","enum":["allowed"]}},
                "additionalProperties":false
            }),
        )],
        ToolCallAssemblyLimits::default(),
    )
    .unwrap();
    let mut controller = ModelStreamController::new(assembler);

    let error = controller
        .push(Ok(ModelStreamEvent::ToolCallDelta(
            ToolCallDelta::new(0, format!(r#"{{"query":"{SECRET}"}}"#), true)
                .with_call_id("call")
                .with_tool_id("lookup"),
        )))
        .unwrap_err();

    assert_stream_failure(error, ModelStreamFailureKind::ArgumentsSchemaMismatch);
    assert!(controller.is_terminal());
}

#[test]
fn assembler_enforces_call_field_and_total_limits() {
    let limits = ToolCallAssemblyLimits {
        max_calls: 1,
        max_call_bytes: 64,
        max_argument_bytes: 48,
        max_total_buffered_bytes: 32,
        max_field_bytes: 8,
        max_json_depth: 4,
        ..ToolCallAssemblyLimits::default()
    };
    let mut fields = assembler(limits.clone());
    assert_stream_failure(
        fields
            .push(ToolCallDelta::new(0, "{}", true).with_call_id("too-long-id"))
            .unwrap_err(),
        ModelStreamFailureKind::InvalidCallId,
    );

    let mut calls = assembler(limits.clone());
    calls
        .push(
            ToolCallDelta::new(0, "", false)
                .with_call_id("one")
                .with_tool_id("lookup"),
        )
        .unwrap();
    assert!(matches!(
        calls.push(ToolCallDelta::new(1, "", false)),
        Err(HarnessError::ResourceLimit(_))
    ));

    let mut total = assembler(limits);
    assert!(matches!(
        total.push(
            ToolCallDelta::new(0, "{\"query\":\"012345678901234567890123456789\"}", false)
                .with_call_id("call")
                .with_tool_id("lookup")
        ),
        Err(HarnessError::ResourceLimit(_))
    ));
}

#[test]
fn assembler_bounds_catalogs_and_provider_advertised_caps() {
    assert!(matches!(
        ToolCallAssemblyLimits::for_provider(&ProviderCapabilityLimits::new().with_max_tools(0)),
        Err(HarnessError::InvalidRequest(_))
    ));
    assert!(matches!(
        ToolCallAssemblyLimits::for_provider(
            &ProviderCapabilityLimits::new().with_max_parallel_tool_calls(0)
        ),
        Err(HarnessError::InvalidRequest(_))
    ));
    assert!(matches!(
        ToolCallAssemblyLimits::for_provider(
            &ProviderCapabilityLimits::new().with_max_streamed_tool_calls(0)
        ),
        Err(HarnessError::InvalidRequest(_))
    ));
    let local = ToolCallAssemblyLimits::default();
    assert_eq!(
        ToolCallAssemblyLimits::for_provider(&ProviderCapabilityLimits::new())
            .unwrap()
            .max_calls,
        local.max_calls
    );
    let clipped = ToolCallAssemblyLimits::for_provider(
        &ProviderCapabilityLimits::new()
            .with_max_tools(u32::MAX)
            .with_max_tool_schema_bytes(u64::MAX)
            .with_max_parallel_tool_calls(u32::MAX)
            .with_max_streamed_tool_calls(u32::MAX)
            .with_max_streamed_argument_bytes(u64::MAX),
    )
    .unwrap();
    assert_eq!(clipped.max_allowed_tools, 1_024);
    assert_eq!(clipped.max_aggregate_schema_bytes, 256 * 1_024);
    assert_eq!(clipped.max_calls, 16);
    assert_eq!(clipped.max_argument_bytes, 64 * 1_024);
    let effective = ToolCallAssemblyLimits::for_provider(
        &ProviderCapabilityLimits::new()
            .with_max_tools(30)
            .with_max_tool_schema_bytes(4_096)
            .with_max_streamed_tool_calls(2)
            .with_max_streamed_argument_bytes(1_024),
    )
    .unwrap();
    assert_eq!(effective.max_allowed_tools, 30);
    assert_eq!(effective.max_aggregate_schema_bytes, 4_096);
    assert_eq!(effective.max_calls, 2);
    assert_eq!(effective.max_argument_bytes, 1_024);
    let combined = ToolCallAssemblyLimits::for_provider(
        &ProviderCapabilityLimits::new()
            .with_max_streamed_tool_calls(7)
            .with_max_parallel_tool_calls(3),
    )
    .unwrap();
    assert_eq!(combined.max_calls, 3);

    let parallel_one = ToolCallAssemblyLimits::for_provider(
        &ProviderCapabilityLimits::new().with_max_parallel_tool_calls(1),
    )
    .unwrap();
    assert_eq!(parallel_one.max_calls, 1);
    let mut one_call = assembler(parallel_one);
    assert!(one_call
        .push(
            ToolCallDelta::new(0, r#"{"query":"first"}"#, true)
                .with_call_id("first")
                .with_tool_id("lookup")
        )
        .unwrap()
        .is_some());
    assert!(matches!(
        one_call.push(
            ToolCallDelta::new(1, r#"{"query":"second"}"#, true)
                .with_call_id("second")
                .with_tool_id("lookup")
        ),
        Err(HarnessError::ResourceLimit(_))
    ));
    let caller_raised = ToolCallAssemblyLimits {
        max_allowed_tools: 1_025,
        ..ToolCallAssemblyLimits::default()
    };
    assert!(matches!(
        caller_raised.validate(),
        Err(HarnessError::InvalidRequest(_))
    ));

    let thousand = (0..1_000)
        .map(|index| {
            ToolDefinition::new(
                format!("tool-{index}"),
                format!("Tool {index}"),
                "test",
                json!(true),
            )
        })
        .collect::<Vec<_>>();
    ToolCallAssembler::new(thousand, ToolCallAssemblyLimits::default()).unwrap();

    let two_tools = ToolCallAssemblyLimits {
        max_allowed_tools: 2,
        ..ToolCallAssemblyLimits::default()
    };
    let three = (0..3)
        .map(|index| {
            ToolDefinition::new(format!("bounded-{index}"), "Bounded", "test", json!(true))
        })
        .collect::<Vec<_>>();
    assert!(matches!(
        ToolCallAssembler::new(three, two_tools),
        Err(HarnessError::ResourceLimit(_))
    ));

    let seven_schema_bytes = ToolCallAssemblyLimits {
        max_aggregate_schema_bytes: 7,
        ..ToolCallAssemblyLimits::default()
    };
    let schemas = [
        ToolDefinition::new("one", "One", "test", json!(true)),
        ToolDefinition::new("two", "Two", "test", json!(true)),
    ];
    assert!(matches!(
        ToolCallAssembler::new(schemas, seven_schema_bytes),
        Err(HarnessError::ResourceLimit(_))
    ));
}

fn controller() -> ModelStreamController {
    ModelStreamController::new(assembler(ToolCallAssemblyLimits::default()))
}

fn completed() -> ModelStreamEvent {
    ModelStreamEvent::Completed {
        model: "test".into(),
        usage: Usage::default(),
    }
}

#[test]
fn stream_controller_fuses_completion_and_rejects_post_terminal_events() {
    let mut controller = controller();
    let update = controller
        .push(Ok(ModelStreamEvent::ToolCallDelta(
            ToolCallDelta::new(0, "{\"query\":\"ok\"}", true)
                .with_call_id("call")
                .with_tool_id("lookup"),
        )))
        .unwrap();
    assert!(update.completed_tool_call.is_some());
    controller.push(Ok(completed())).unwrap();
    assert!(controller.is_terminal());
    assert!(controller.finish_eof().is_ok());
    assert_stream_failure(
        controller.push(Ok(completed())).unwrap_err(),
        ModelStreamFailureKind::EventAfterCompletion,
    );
    assert_stream_failure(
        controller
            .push(Ok(ModelStreamEvent::TextDelta {
                content: "sentinel-provider-message".into(),
            }))
            .unwrap_err(),
        ModelStreamFailureKind::EventAfterCompletion,
    );
}

#[test]
fn stream_controller_rejects_pending_completion_and_eof() {
    let mut completion = controller();
    completion
        .push(Ok(ModelStreamEvent::ToolCallDelta(
            ToolCallDelta::new(0, "{sentinel-arguments", false)
                .with_call_id("sentinel-call-id")
                .with_tool_id("lookup"),
        )))
        .unwrap();
    assert_stream_failure(
        completion.push(Ok(completed())).unwrap_err(),
        ModelStreamFailureKind::CompletionWithPendingCall,
    );
    assert!(completion.is_terminal());

    let mut eof = controller();
    eof.push(Ok(ModelStreamEvent::ToolCallDelta(
        ToolCallDelta::new(0, "{sentinel-arguments", false)
            .with_call_id("sentinel-call-id")
            .with_tool_id("lookup"),
    )))
    .unwrap();
    assert_stream_failure(
        eof.finish_eof().unwrap_err(),
        ModelStreamFailureKind::EofWithPendingCall,
    );
    assert!(eof.is_terminal());

    let mut direct = assembler(ToolCallAssemblyLimits::default());
    direct
        .push(
            ToolCallDelta::new(0, "{sentinel-arguments", false)
                .with_call_id("sentinel-call-id")
                .with_tool_id("lookup"),
        )
        .unwrap();
    assert_stream_failure(
        direct.finish().unwrap_err(),
        ModelStreamFailureKind::IncompletePendingCall,
    );
}

#[test]
fn stream_controller_makes_first_error_or_cancellation_terminal() {
    let mut upstream = controller();
    assert_stream_failure(
        upstream
            .push(Err(HarnessError::Provider(
                "sentinel-provider-message".into(),
            )))
            .unwrap_err(),
        ModelStreamFailureKind::UpstreamProviderFailure,
    );
    assert_stream_failure(
        upstream
            .push(Ok(ModelStreamEvent::TextDelta {
                content: "sentinel-provider-message".into(),
            }))
            .unwrap_err(),
        ModelStreamFailureKind::EventAfterFailure,
    );
    assert_stream_failure(
        upstream.finish_eof().unwrap_err(),
        ModelStreamFailureKind::EofAfterFailure,
    );

    let mut cancelled = controller();
    assert!(matches!(
        cancelled.push(Err(HarnessError::Cancelled)),
        Err(HarnessError::Cancelled)
    ));
    assert_stream_failure(
        cancelled
            .push(Ok(ModelStreamEvent::TextDelta {
                content: "sentinel-provider-message".into(),
            }))
            .unwrap_err(),
        ModelStreamFailureKind::EventAfterFailure,
    );

    let mut limited = controller();
    assert!(matches!(
        limited.push(Err(HarnessError::ResourceLimit("bounded".into()))),
        Err(HarnessError::ResourceLimit(message)) if message == "bounded"
    ));
    assert!(limited.is_terminal());

    let mut eof = controller();
    assert_stream_failure(
        eof.finish_eof().unwrap_err(),
        ModelStreamFailureKind::EofBeforeCompletion,
    );
}

#[tokio::test]
async fn model_provider_stream_defaults_to_unsupported_capability() {
    let provider = MockModelProvider::scripted([]);
    assert!(matches!(
        provider.stream(ModelRequest::new("mock")).await,
        Err(HarnessError::UnsupportedCapability(message)) if message.contains("mock")
    ));
}
