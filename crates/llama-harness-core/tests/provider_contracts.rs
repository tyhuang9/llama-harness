use async_trait::async_trait;
use llama_harness_core::{
    CancellationSafety, HarnessError, ModelCapabilities, ProviderCapabilityLimits,
    SpeculationPolicy, Tool, ToolCaller, ToolDefinition, ToolRegistry, ToolResult,
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
