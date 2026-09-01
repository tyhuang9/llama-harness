use async_trait::async_trait;
use futures_util::stream;
use llama_harness_core::{
    mock::{final_response, tool_response, MockModelProvider},
    AgentDefinition, AgentRunner, ApprovalHandler, ApprovalRecord, HarnessError, MessageRole,
    ModelCapabilities, ModelEventStream, ModelInfo, ModelProvider, ModelRequest, ModelResponse,
    ModelStreamEvent, PolicyDecision, PolicyEngine, ProviderHealth, RunRequest, SpeculationConfig,
    SpeculationMode, Tool, ToolCall, ToolCallContext, ToolCallDelta, ToolDefinition,
    ToolDiscoveryMetadata, ToolRegistry, ToolResult, Usage,
};
use llama_harness_mcp::{
    McpCallRequest, McpCallResult, McpCatalogManager, McpContext, McpDispatchState, McpError,
    McpLimits, McpOperation, McpProtocolEra, McpTool, McpToolPage, McpTransport, McpTransportError,
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

struct LocalTool(ToolDefinition);

#[async_trait]
impl Tool for LocalTool {
    fn definition(&self) -> &ToolDefinition {
        &self.0
    }

    async fn execute(&self, _: Value, _: CancellationToken) -> Result<ToolResult, HarnessError> {
        Ok(ToolResult::success(json!({"local": true})))
    }
}

fn local_tool(id: &str) -> Arc<dyn Tool> {
    Arc::new(LocalTool(ToolDefinition::new(
        id,
        "local-tool",
        "local test tool",
        json!({"type":"object"}),
    )))
}

struct FakeTransport {
    calls: AtomicU64,
    connects: AtomicU64,
    lists: AtomicU64,
    tools: Mutex<Vec<McpTool>>,
    fail_list: AtomicU64,
}

impl FakeTransport {
    fn with_tools(tools: Vec<McpTool>) -> Self {
        Self {
            calls: AtomicU64::new(0),
            connects: AtomicU64::new(0),
            lists: AtomicU64::new(0),
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
        self.connects.fetch_add(1, Ordering::Relaxed);
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
        self.lists.fetch_add(1, Ordering::Relaxed);
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

struct ExplicitSpeculativeAllow {
    decisions: AtomicU64,
}

#[async_trait]
impl PolicyEngine for ExplicitSpeculativeAllow {
    async fn decide(
        &self,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(PolicyDecision::Allow {
            reason: "authoritative MCP test allow".into(),
        })
    }

    async fn decide_speculative(
        &self,
        _: &ToolCallContext,
        _: &ToolDefinition,
        _: &Value,
        _: &RunRequest,
    ) -> Result<PolicyDecision, HarnessError> {
        self.decisions.fetch_add(1, Ordering::Relaxed);
        Ok(PolicyDecision::Allow {
            reason: "explicit speculative test allow".into(),
        })
    }
}

struct StreamingToolProvider {
    tool_id: String,
}

#[async_trait]
impl ModelProvider for StreamingToolProvider {
    fn id(&self) -> &str {
        "mcp-streaming-test"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::new(true, true, false).with_streaming_tool_arguments(true)
    }

    async fn health(&self) -> Result<ProviderHealth, HarnessError> {
        Ok(ProviderHealth::healthy())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, HarnessError> {
        Ok(vec![ModelInfo::new("mock-model")])
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, HarnessError> {
        Ok(ModelResponse::new(request.model).with_final_output("unexpected completion path"))
    }

    async fn stream(&self, request: ModelRequest) -> Result<ModelEventStream, HarnessError> {
        let events = if request
            .messages
            .iter()
            .any(|message| message.role == MessageRole::Tool)
        {
            vec![
                Ok(ModelStreamEvent::TextDelta {
                    content: "done".into(),
                }),
                Ok(ModelStreamEvent::Completed {
                    model: request.model,
                    usage: Usage::default(),
                }),
            ]
        } else {
            vec![
                Ok(ModelStreamEvent::ToolCallDelta(
                    ToolCallDelta::new(0, r#"{"value":"safe"}"#, true)
                        .with_call_id("call")
                        .with_tool_id(self.tool_id.clone()),
                )),
                Ok(ModelStreamEvent::Completed {
                    model: request.model,
                    usage: Usage::default(),
                }),
            ]
        };
        Ok(Box::pin(stream::iter(events)))
    }
}

async fn imported_registry(
    transport: Arc<FakeTransport>,
) -> (McpCatalogManager, ToolRegistry, String) {
    let manager = McpCatalogManager::new(transport, "server", McpLimits::default())
        .expect("valid host configuration");
    let (_, registry) = manager
        .refresh_registered(&ToolRegistry::default(), CancellationToken::new())
        .await
        .expect("catalog imports");
    let id = manager.active_tools()[0].definition().id.clone();
    (manager, registry, id)
}

fn request(tool_id: String) -> RunRequest {
    let mut agent = AgentDefinition::new("agent", "Agent", "1", "mock-model");
    agent.tool_allowlist = vec![tool_id.clone()];
    RunRequest::new(agent, "call the imported tool")
}

async fn assert_approved_call_succeeds(registry: ToolRegistry, tool_id: String) {
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
        .expect("prior registry remains broker-callable");
    assert_eq!(result.final_output.as_deref(), Some("done"), "{result:#?}");
    assert!(result.errors.is_empty(), "{result:#?}");
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
async fn shadow_and_activation_configuration_cannot_speculatively_dispatch_imported_mcp_tools() {
    let transport = Arc::new(FakeTransport::with_tools(vec![remote_tool("remote-read")]));
    let (_manager, registry, tool_id) = imported_registry(transport.clone()).await;
    let imported = registry.get(&tool_id).expect("imported tool is registered");
    assert!(
        !imported
            .definition()
            .allows_caller(llama_harness_core::ToolCaller::Speculative),
        "MCP imports must remain Direct-only"
    );

    let policy = Arc::new(ExplicitSpeculativeAllow {
        decisions: AtomicU64::new(0),
    });
    let runner = AgentRunner::builder(Arc::new(StreamingToolProvider {
        tool_id: tool_id.clone(),
    }))
    .tools(registry)
    .policy(policy.clone())
    .speculation(SpeculationConfig::default())
    .build();

    assert_eq!(
        runner.activate_speculation(&tool_id).mode,
        SpeculationMode::Disabled
    );
    let result = runner
        .run(request(tool_id.clone()))
        .await
        .expect("authoritative MCP call succeeds");

    assert_eq!(result.final_output.as_deref(), Some("done"));
    assert_eq!(
        transport.calls(),
        1,
        "only the Direct broker call dispatches"
    );
    assert_eq!(policy.decisions.load(Ordering::Relaxed), 0);
    assert_eq!(
        runner.speculation_readiness(&tool_id).mode,
        SpeculationMode::Disabled
    );
    assert_eq!(runner.speculation_metrics(&tool_id).issued, 0);
    assert_eq!(
        runner.activate_speculation(&tool_id).mode,
        SpeculationMode::Disabled,
        "an imported tool cannot be forced into Active"
    );
}

#[tokio::test]
async fn mcp_registry_replacement_is_atomic_and_keeps_old_runner_snapshot_immutable() {
    let transport = Arc::new(FakeTransport::with_tools(vec![remote_tool("remote-write")]));
    let manager =
        McpCatalogManager::new(transport.clone(), "server", McpLimits::default()).expect("manager");
    let (_, old_registry) = manager
        .refresh_registered(&ToolRegistry::default(), CancellationToken::new())
        .await
        .expect("initial refresh and install");
    let first_id = manager.active_tools()[0].definition().id.clone();

    transport.replace_tools(vec![
        remote_tool("remote-write"),
        remote_tool("remote-read"),
    ]);
    let (second, added_registry) = manager
        .refresh_registered(&old_registry, CancellationToken::new())
        .await
        .expect("add refresh");
    let added_id = manager.active_tools()[1].definition().id.clone();
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
    let (_, removed_registry) = manager
        .refresh_registered(&added_registry, CancellationToken::new())
        .await
        .expect("remove refresh");
    assert!(
        added_registry.get(&first_id).is_some(),
        "prior registry is immutable"
    );
    assert!(removed_registry.get(&first_id).is_none());
    assert!(removed_registry.get(&added_id).is_some());

    transport.fail_list.store(1, Ordering::Relaxed);
    assert!(manager
        .refresh_registered(&removed_registry, CancellationToken::new())
        .await
        .is_err());
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

#[tokio::test]
async fn empty_refresh_removes_only_the_manager_group_and_remains_replaceable() {
    let transport = Arc::new(FakeTransport::with_tools(vec![remote_tool("remote-write")]));
    let manager =
        McpCatalogManager::new(transport.clone(), "server", McpLimits::default()).expect("manager");
    let mut base = ToolRegistry::default();
    base.register_with_discovery(local_tool("local.keep"), ToolDiscoveryMetadata::deferred())
        .expect("unrelated local tool");

    let (_, installed) = manager
        .refresh_registered(&base, CancellationToken::new())
        .await
        .expect("nonempty catalog installs");
    let remote_id = manager.active_tools()[0].definition().id.clone();
    assert!(installed.get("local.keep").is_some());
    assert!(installed.get(&remote_id).is_some());

    transport.replace_tools(Vec::new());
    let (empty, removed) = manager
        .refresh_registered(&installed, CancellationToken::new())
        .await
        .expect("empty catalog removes the manager group");
    assert_eq!(empty.tool_count, 0);
    assert!(manager.active_snapshot().is_some());
    assert!(manager.active_tools().is_empty());
    assert!(removed.get(&remote_id).is_none());
    assert!(removed.get("local.keep").is_some());

    let replaced_again = manager
        .replace_registered(&removed)
        .expect("active empty catalog remains replaceable");
    assert!(replaced_again.get(&remote_id).is_none());
    assert!(replaced_again.get("local.keep").is_some());
}

#[tokio::test]
async fn bound_manager_rejects_catalog_only_refresh_before_transport_io() {
    let transport = Arc::new(FakeTransport::with_tools(vec![remote_tool("remote-write")]));
    let manager =
        McpCatalogManager::new(transport.clone(), "server", McpLimits::default()).expect("manager");
    let initial = manager
        .refresh(CancellationToken::new())
        .await
        .expect("catalog-only initial refresh");
    let tool_id = manager.active_tools()[0].definition().id.clone();
    let registry = manager
        .replace_registered(&ToolRegistry::default())
        .expect("initial catalog binds to the registry workflow");
    let connects = transport.connects.load(Ordering::Relaxed);
    let lists = transport.lists.load(Ordering::Relaxed);

    transport.replace_tools(vec![remote_tool("remote-read")]);
    assert!(matches!(
        manager.refresh(CancellationToken::new()).await,
        Err(McpError::CatalogUnavailable)
    ));
    assert_eq!(transport.connects.load(Ordering::Relaxed), connects);
    assert_eq!(transport.lists.load(Ordering::Relaxed), lists);
    assert_eq!(manager.active_snapshot(), Some(initial));

    let replacement = manager
        .replace_registered(&registry)
        .expect("prior generation remains replaceable");
    assert_approved_call_succeeds(replacement, tool_id).await;
    assert_eq!(transport.calls(), 1);
}

#[tokio::test]
async fn transactional_refresh_collision_preserves_the_prior_callable_generation() {
    let transport = Arc::new(FakeTransport::with_tools(vec![remote_tool("remote-read")]));
    let manager =
        McpCatalogManager::new(transport.clone(), "server", McpLimits::default()).expect("manager");
    manager
        .refresh(CancellationToken::new())
        .await
        .expect("probe generated read identifier");
    let colliding_id = manager.active_tools()[0].definition().id.clone();

    transport.replace_tools(vec![remote_tool("remote-write")]);
    let prior = manager
        .refresh(CancellationToken::new())
        .await
        .expect("install prior catalog before binding");
    let prior_id = manager.active_tools()[0].definition().id.clone();
    let mut base = ToolRegistry::default();
    base.register_with_discovery(local_tool(&colliding_id), ToolDiscoveryMetadata::deferred())
        .expect("host tool with the future generated identifier");
    let installed = manager
        .replace_registered(&base)
        .expect("prior generation and host tool do not collide");

    transport.replace_tools(vec![remote_tool("remote-read")]);
    assert!(matches!(
        manager
            .refresh_registered(&installed, CancellationToken::new())
            .await,
        Err(McpError::Core(_))
    ));
    assert_eq!(manager.active_snapshot(), Some(prior));
    assert_eq!(manager.active_tools()[0].definition().id, prior_id);

    let replacement = manager
        .replace_registered(&installed)
        .expect("prior generation remains replaceable after collision");
    assert!(replacement.get(&colliding_id).is_some());
    assert!(replacement.get(&prior_id).is_some());
    assert_approved_call_succeeds(replacement, prior_id).await;
    assert_eq!(transport.calls(), 1);
}

#[tokio::test]
async fn invalid_candidate_schema_preserves_the_prior_callable_generation() {
    let transport = Arc::new(FakeTransport::with_tools(vec![remote_tool("remote-write")]));
    let manager =
        McpCatalogManager::new(transport.clone(), "server", McpLimits::default()).expect("manager");
    let (prior, registry) = manager
        .refresh_registered(&ToolRegistry::default(), CancellationToken::new())
        .await
        .expect("initial registered refresh");
    let prior_id = manager.active_tools()[0].definition().id.clone();

    let mut invalid = remote_tool("remote-read");
    invalid.input_schema = json!({"type": 7});
    transport.replace_tools(vec![invalid]);
    assert!(matches!(
        manager
            .refresh_registered(&registry, CancellationToken::new())
            .await,
        Err(McpError::Core(_))
    ));
    assert_eq!(manager.active_snapshot(), Some(prior));
    assert_eq!(manager.active_tools()[0].definition().id, prior_id);

    let replacement = manager
        .replace_registered(&registry)
        .expect("prior generation remains replaceable after invalid schema");
    assert_approved_call_succeeds(replacement, prior_id).await;
    assert_eq!(transport.calls(), 1);
}

#[tokio::test]
async fn invalid_candidate_names_preserve_the_prior_callable_generation() {
    let transport = Arc::new(FakeTransport::with_tools(vec![remote_tool("remote-write")]));
    let manager =
        McpCatalogManager::new(transport.clone(), "server", McpLimits::default()).expect("manager");
    let (prior, registry) = manager
        .refresh_registered(&ToolRegistry::default(), CancellationToken::new())
        .await
        .expect("initial registered refresh");
    let prior_id = manager.active_tools()[0].definition().id.clone();

    for invalid_name in [" remote-read", "réad"] {
        transport.replace_tools(vec![remote_tool(invalid_name)]);
        assert!(matches!(
            manager
                .refresh_registered(&registry, CancellationToken::new())
                .await,
            Err(McpError::Core(_))
        ));
        assert_eq!(manager.active_snapshot(), Some(prior.clone()));
        assert_eq!(manager.active_tools()[0].definition().id, prior_id);
    }

    let replacement = manager
        .replace_registered(&registry)
        .expect("prior generation remains replaceable after invalid names");
    assert_approved_call_succeeds(replacement, prior_id).await;
    assert_eq!(transport.calls(), 1);
}
