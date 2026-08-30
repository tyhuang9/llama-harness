use async_trait::async_trait;
use llama_harness_core::{
    mock::MockModelProvider, CancellationSafety, HarnessError, ModelCapabilities, ModelProvider,
    ModelRequest, ProviderCapabilityLimits, SpeculationPolicy, Tool, ToolCallAssembler,
    ToolCallAssemblyLimits, ToolCallDelta, ToolCaller, ToolDefinition, ToolRegistry, ToolResult,
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
    ] {
        legacy.remove(key);
    }
    let decoded: ToolDefinition = serde_json::from_value(Value::Object(legacy)).unwrap();
    assert_eq!(decoded.allowed_callers, [ToolCaller::Direct].into());
    assert_eq!(decoded.speculation_policy, SpeculationPolicy::Disabled);
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
        .with_cancellation_safety(CancellationSafety::Guaranteed);
    register(safe_definition).unwrap();
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
fn assembler_rejects_conflicts_invalid_finals_and_post_final_fragments() {
    let mut conflict = assembler(ToolCallAssemblyLimits::default());
    conflict
        .push(
            ToolCallDelta::new(0, "{", false)
                .with_call_id("call")
                .with_tool_id("lookup"),
        )
        .unwrap();
    assert!(matches!(
        conflict.push(ToolCallDelta::new(0, "}", true).with_tool_id("other")),
        Err(HarnessError::Provider(message)) if message.contains("conflicting")
    ));

    let mut malformed = assembler(ToolCallAssemblyLimits::default());
    assert!(matches!(
        malformed.push(
            ToolCallDelta::new(0, "{", true)
                .with_call_id("call")
                .with_tool_id("lookup")
        ),
        Err(HarnessError::InvalidArguments(_))
    ));

    let mut finalized = assembler(ToolCallAssemblyLimits::default());
    finalized
        .push(
            ToolCallDelta::new(0, "{\"query\":\"ok\"}", true)
                .with_call_id("call")
                .with_tool_id("lookup"),
        )
        .unwrap();
    assert!(matches!(
        finalized.push(ToolCallDelta::new(0, "", false)),
        Err(HarnessError::Provider(message)) if message.contains("after completion")
    ));
}

#[test]
fn assembler_enforces_call_field_and_total_limits() {
    let limits = ToolCallAssemblyLimits {
        max_calls: 1,
        max_call_bytes: 64,
        max_total_buffered_bytes: 32,
        max_field_bytes: 8,
        max_json_depth: 4,
    };
    let mut fields = assembler(limits.clone());
    assert!(matches!(
        fields.push(ToolCallDelta::new(0, "{}", true).with_call_id("too-long-id")),
        Err(HarnessError::Provider(_))
    ));

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
fn assembler_rejects_unknown_tools_and_schema_mismatches() {
    let mut unknown = assembler(ToolCallAssemblyLimits::default());
    assert!(matches!(
        unknown.push(
            ToolCallDelta::new(0, "{}", true)
                .with_call_id("call")
                .with_tool_id("missing")
        ),
        Err(HarnessError::InvalidTool(_))
    ));

    let mut invalid = assembler(ToolCallAssemblyLimits::default());
    assert!(matches!(
        invalid.push(
            ToolCallDelta::new(0, "{\"query\":1}", true)
                .with_call_id("call")
                .with_tool_id("lookup")
        ),
        Err(HarnessError::InvalidArguments(_))
    ));
}

#[tokio::test]
async fn model_provider_stream_defaults_to_unsupported_capability() {
    let provider = MockModelProvider::scripted([]);
    assert!(matches!(
        provider.stream(ModelRequest::new("mock")).await,
        Err(HarnessError::UnsupportedCapability(message)) if message.contains("mock")
    ));
}
