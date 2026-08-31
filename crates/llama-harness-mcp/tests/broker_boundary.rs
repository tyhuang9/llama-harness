use async_trait::async_trait;
use llama_harness_core::{
    mock::{final_response, tool_response, MockModelProvider},
    AgentDefinition, AgentRunner, ApprovalHandler, ApprovalRecord, HarnessError, PolicyDecision,
    PolicyEngine, RunRequest, ToolCall, ToolDefinition, ToolRegistry,
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
        Arc,
    },
};
use tokio_util::sync::CancellationToken;

struct FakeTransport {
    calls: AtomicU64,
}

impl FakeTransport {
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
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
        if cursor.is_some() {
            return Err(McpTransportError {
                operation: McpOperation::ListTools,
                dispatch: McpDispatchState::NotDispatched,
            });
        }
        Ok(McpToolPage {
            tools: vec![McpTool {
                name: "remote-write".into(),
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
            }],
            next_cursor: None,
            ttl_ms: Some(1_000),
            cache_scope: Some("server".into()),
        })
    }

    async fn call_tool(
        &self,
        _: &McpContext,
        request: McpCallRequest,
        _: CancellationToken,
    ) -> Result<McpCallResult, McpTransportError> {
        assert_eq!(request.name, "remote-write");
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
    let mut registry = ToolRegistry::default();
    manager
        .register_active(&mut registry)
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
    let transport = Arc::new(FakeTransport {
        calls: AtomicU64::new(0),
    });
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
    let transport = Arc::new(FakeTransport {
        calls: AtomicU64::new(0),
    });
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
