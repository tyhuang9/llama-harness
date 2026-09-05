use async_trait::async_trait;
use llama_harness_core::{
    GroupToolRegistration, HarnessError, Tool, ToolDefinition, ToolDiscoveryMetadata,
    ToolRegistrationGroup, ToolRegistry, ToolResult,
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
        Ok(ToolResult::success(json!(null)))
    }
}

fn tool(id: &str, name: &str) -> Arc<dyn Tool> {
    Arc::new(TestTool(ToolDefinition::new(
        id,
        name,
        "test",
        json!({"type":"object"}),
    )))
}

fn grouped(tool: Arc<dyn Tool>) -> GroupToolRegistration {
    GroupToolRegistration::new(tool, ToolDiscoveryMetadata::deferred())
}

#[test]
fn group_replacement_is_transactional_and_snapshot_based() {
    let group = ToolRegistrationGroup::new("provider:remote").expect("group");
    let mut base = ToolRegistry::default();
    base.register_with_discovery(tool("local", "local"), ToolDiscoveryMetadata::deferred())
        .expect("local");

    let installed = base
        .replace_group(&group, [grouped(tool("remote-a", "a"))])
        .expect("install");
    assert!(base.get("remote-a").is_none(), "old snapshot is unchanged");
    assert!(installed.get("local").is_some());
    assert!(installed.get("remote-a").is_some());

    let replaced = installed
        .replace_group(&group, [grouped(tool("remote-b", "b"))])
        .expect("replace");
    assert!(
        installed.get("remote-a").is_some(),
        "in-flight owner retains old snapshot"
    );
    assert!(replaced.get("remote-a").is_none());
    assert!(replaced.get("remote-b").is_some());
    assert!(replaced.get("local").is_some());

    let failed = replaced.replace_group(&group, [grouped(tool("local", "collision"))]);
    assert!(failed.is_err(), "non-group collision is rejected");
    assert!(
        replaced.get("remote-b").is_some(),
        "failed replacement leaves source usable"
    );
    assert!(replaced.get("local").is_some());
}
