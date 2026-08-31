use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, tool_response, MockModelProvider},
    AgentDefinition, AgentRunner, ApprovalHandler, ApprovalRecord, HarnessError, PolicyDecision,
    PolicyEngine, RunRequest, ToolCall, ToolDefinition, ToolDiscoveryMetadata, ToolRegistry,
};
use llama_harness_mcp::{
    McpCallRequest, McpCallResult, McpCatalogManager, McpContext, McpDispatchState, McpLimits,
    McpOperation, McpProtocolEra, McpTool, McpToolPage, McpTransport, McpTransportError,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};
use tokio_util::sync::CancellationToken;

fn remote_tool(name: &str) -> McpTool {
    McpTool {
        name: name.into(),
        description: "untrusted remote description".into(),
        input_schema: json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        }),
        output_schema: Some(json!({
            "type": "object",
            "properties": { "ok": { "const": true } },
            "required": ["ok"]
        })),
    }
}

struct FakeTransport {
    calls: AtomicU64,
    tools: Mutex<Vec<McpTool>>,
    fail_list: AtomicU64,
}

impl FakeTransport {
    fn with_tools(tools: Vec<McpTool>) -> Self {
        Self {
            calls: AtomicU64::new(0),
            tools: Mutex::new(tools),
            fail_list: AtomicU64::new(0),
        }
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    fn replace_tools(&self, tools: Vec<McpTool>) {
        *self.tools.lock().expect("test mutex") = tools;
    }
}

#[async_trait]
impl McpTransport for FakeTransport {
    async fn connect(&self, _: CancellationToken) -> Result<McpContext, McpTransportError> {
        Ok(McpContext {
            era: McpProtocolEra::Modern20260728,
            version: "2026-07-28".into(),
            capabilities: BTreeSet::from(["tools".into()]),
            request_context: Some("host-private-context".into()),
        })
    }

    async fn list_tools(
        &self,
        _: &McpContext,
        cursor: Option<&str>,
        _: CancellationToken,
    ) -> Result<McpToolPage, McpTransportError> {
        if self.fail_list.load(Ordering::Relaxed) != 0 {
            return Err(McpTransportError {
                operation: McpOperation::ListTools,
                dispatch: McpDispatchState::NotDispatched,
            });
        }
        if cursor.is_some() {
            return Err(McpTransportError {
                operation: McpOperation::ListTools,
                dispatch: McpDispatchState::NotDispatched,
            });
        }
        Ok(McpToolPage {
            tools: self.tools.lock().expect("test mutex").clone(),
            next_cursor: None,
            ttl_ms: Some(1_000),
            cache_scope: Some("public".into()),
        })
    }

    async fn call_tool(
        &self,
        _: &McpContext,
        request: McpCallRequest,
        _: CancellationToken,
    ) -> Result<McpCallResult, McpTransportError> {
        assert!(matches!(
            request.name.as_str(),
            "remote-write" | "remote-read"
        ));
        assert_eq!(request.arguments, json!({"value":"safe"}));
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(McpCallResult {
            structured_content: Some(json!({"ok": true})),
            content: None,
            is_error: false,
        })
    }

    async fn close(&self, _: McpContext, _: CancellationToken) -> Result<(), McpTransportError> {
        Ok(())
    }
}

struct RequireApproval;

#[async_trait]
impl PolicyEngine for RequireApproval {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(PolicyDecision::RequireApproval {
            reason: "host requires confirmation".into(),
        })
    }
}

struct GrantApproval;

#[async_trait]
impl ApprovalHandler for GrantApproval {
    async fn approve(
        &self,
        tool: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<ApprovalRecord, HarnessError> {
        Ok(ApprovalRecord::new("ignored", &tool.id, true, "granted"))
    }
}

async fn imported_registry(
    transport: Arc<FakeTransport>,
) -> (McpCatalogManager, ToolRegistry, String) {
    let manager = McpCatalogManager::new(transport, "server", McpLimits::default())
        .expect("valid host configuration");
    manager
        .refresh(CancellationToken::new())
        .await
        .expect("catalog imports");
    let id = manager.active_tools()[0].definition().id.clone();
    let registry = manager
        .replace_registered(&ToolRegistry::default())
        .expect("catalog registers atomically");
    (manager, registry, id)
}

fn request(tool_id: String) -> RunRequest {
    let mut agent = AgentDefinition::new("agent", "Agent", "1", "mock-model");
    agent.tool_allowlist = vec![tool_id.clone()];
    RunRequest::new(agent, "call the imported tool")
}

#[tokio::test]
async fn default_policy_denies_imported_mutation_before_transport_dispatch() {
    let transport = Arc::new(FakeTransport::with_tools(vec![remote_tool("remote-write")]));
    let (_manager, registry, tool_id) = imported_registry(transport.clone()).await;
    let provider = Arc::new(MockModelProvider::scripted([
        tool_response(ToolCall::new(
            "call",
            tool_id.clone(),
            r#"{"value":"safe"}"#,
        )),
        final_response("done"),
    ]));
    let result = AgentRunner::builder(provider)
        .tools(registry)
        .build()
        .run(request(tool_id))
        .await
        .expect("policy denial is represented in the run result");
    assert_eq!(result.final_output.as_deref(), Some("done"), "{result:#?}");
    assert_eq!(transport.calls(), 0);
    assert!(matches!(
        result.policy_decisions.as_slice(),
        [PolicyDecision::Deny { .. }]
    ));
}

#[tokio::test]
async fn approval_gated_imported_tool_dispatches_through_the_runner_once() {
    let transport = Arc::new(FakeTransport::with_tools(vec![remote_tool("remote-write")]));
    let (_manager, registry, tool_id) = imported_registry(transport.clone()).await;
    let provider = Arc::new(MockModelProvider::scripted([
        tool_response(ToolCall::new(
            "call",
            tool_id.clone(),
            r#"{"value":"safe"}"#,
        )),
        final_response("done"),
    ]));
    let result = AgentRunner::builder(provider)
        .tools(registry)
        .policy(Arc::new(RequireApproval))
        .approvals(Arc::new(GrantApproval))
        .build()
        .run(request(tool_id))
        .await
        .expect("approved call succeeds");
    assert_eq!(result.final_output.as_deref(), Some("done"), "{result:#?}");
    assert_eq!(transport.calls(), 1);
    assert!(matches!(
        result.policy_decisions.as_slice(),
        [PolicyDecision::RequireApproval { .. }]
    ));
    assert!(result.approvals.iter().all(|approval| approval.granted));
}

#[tokio::test]
async fn mcp_registry_replacement_is_atomic_and_keeps_old_runner_snapshot_immutable() {
    let transport = Arc::new(FakeTransport::with_tools(vec![remote_tool("remote-write")]));
    let manager =
        McpCatalogManager::new(transport.clone(), "server", McpLimits::default()).expect("manager");
    manager
        .refresh(CancellationToken::new())
        .await
        .expect("initial refresh");
    let first_id = manager.active_tools()[0].definition().id.clone();
    let old_registry = manager
        .replace_registered(&ToolRegistry::default())
        .expect("initial install");

    transport.replace_tools(vec![
        remote_tool("remote-write"),
        remote_tool("remote-read"),
    ]);
    let second = manager
        .refresh(CancellationToken::new())
        .await
        .expect("add refresh");
    let added_id = manager.active_tools()[1].definition().id.clone();
    let added_registry = manager
        .replace_registered(&old_registry)
        .expect("same-id plus add replacement");
    assert_eq!(second.generation, 2);
    assert!(old_registry.get(&first_id).is_some());
    assert!(old_registry.get(&added_id).is_none());
    assert!(added_registry.get(&first_id).is_some());
    assert!(added_registry.get(&added_id).is_some());
    let old_runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
        tool_response(ToolCall::new(
            "call",
            first_id.clone(),
            r#"{"value":"safe"}"#,
        )),
        final_response("done"),
    ])))
    .tools(old_registry)
    .policy(Arc::new(RequireApproval))
    .approvals(Arc::new(GrantApproval))
    .build();

    transport.replace_tools(vec![remote_tool("remote-read")]);
    manager
        .refresh(CancellationToken::new())
        .await
        .expect("remove refresh");
    let removed_registry = manager
        .replace_registered(&added_registry)
        .expect("remove replacement");
    assert!(
        added_registry.get(&first_id).is_some(),
        "prior registry is immutable"
    );
    assert!(removed_registry.get(&first_id).is_none());
    assert!(removed_registry.get(&added_id).is_some());

    transport.fail_list.store(1, Ordering::Relaxed);
    assert!(manager.refresh(CancellationToken::new()).await.is_err());
    let after_failed_refresh = manager
        .replace_registered(&removed_registry)
        .expect("prior active catalog remains installable");
    assert!(after_failed_refresh.get(&added_id).is_some());

    let mut collision_base = ToolRegistry::default();
    collision_base
        .register_with_discovery(
            manager.active_tools()[0].clone(),
            ToolDiscoveryMetadata::deferred(),
        )
        .expect("non-group tool");
    assert!(manager.replace_registered(&collision_base).is_err());
    assert!(collision_base.get(&added_id).is_some());

    let old_result = old_runner
        .run(request(first_id))
        .await
        .expect("old runner returns a completed run");
    assert!(!old_result.errors.is_empty());
    assert_eq!(transport.calls(), 0, "old generation must not dispatch");

    let new_runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([
        tool_response(ToolCall::new(
            "call",
            added_id.clone(),
            r#"{"value":"safe"}"#,
        )),
        final_response("done"),
    ])))
    .tools(removed_registry)
    .policy(Arc::new(RequireApproval))
    .approvals(Arc::new(GrantApproval))
    .build();
    let new_result = new_runner
        .run(request(added_id))
        .await
        .expect("new runner dispatches current catalog");
    assert_eq!(new_result.final_output.as_deref(), Some("done"));
    assert_eq!(transport.calls(), 1);
}
