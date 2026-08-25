use crate::{
    limits::{compile_trusted_schema, ensure_json_depth, serialized_len},
    AgentLimits, HarnessError,
};
use async_trait::async_trait;
use jsonschema::Validator;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, sync::Arc};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
/// A tool invocation requested by a model.
pub struct ToolCall {
    /// Provider- or model-generated call identifier.
    pub id: String,
    /// Registered tool identifier.
    pub tool_id: String,
    /// JSON-encoded arguments supplied to the tool.
    pub arguments_json: String,
}

impl ToolCall {
    /// Creates a model-requested tool call from its wire-level JSON arguments.
    pub fn new(
        id: impl Into<String>,
        tool_id: impl Into<String>,
        arguments_json: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            tool_id: tool_id.into(),
            arguments_json: arguments_json.into(),
        }
    }
}

/// Immutable correlation data for one validated tool call.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
/// Immutable correlation data for one tool invocation.
pub struct ToolCallContext {
    /// Identifier of the run containing the call.
    pub run_id: String,
    /// Trace identifier associated with the run.
    pub trace_id: String,
    /// Identifier of the tool call.
    pub call_id: String,
    /// Identifier of the registered tool.
    pub tool_id: String,
}

impl ToolCallContext {
    /// Creates immutable correlation data for one validated tool invocation.
    pub fn new(
        run_id: impl Into<String>,
        trace_id: impl Into<String>,
        call_id: impl Into<String>,
        tool_id: impl Into<String>,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            trace_id: trace_id.into(),
            call_id: call_id.into(),
            tool_id: tool_id.into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Application-assessed risk level for a tool.
pub enum ToolRisk {
    /// Low-impact tool operation.
    Low,
    /// Moderate-impact tool operation.
    Medium,
    /// High-impact tool operation.
    High,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
/// Description and safety metadata for a registered tool.
pub struct ToolDefinition {
    /// Stable tool identifier used in allowlists and calls.
    pub id: String,
    /// Human-readable tool name.
    pub name: String,
    /// Human-readable description presented to the model.
    pub description: String,
    /// JSON Schema for validating tool arguments.
    pub arguments_schema: Value,
    /// Application-assessed risk level.
    pub risk: ToolRisk,
    /// Whether repeating the same call is safe.
    pub idempotent: bool,
    /// Whether the tool is guaranteed not to change state.
    pub read_only: bool,
}

impl ToolDefinition {
    /// Creates a tool declaration with conservative mutation defaults.
    ///
    /// New definitions are high-risk, non-idempotent, and state-changing until
    /// the application explicitly declares otherwise.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        arguments_schema: Value,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            arguments_schema,
            risk: ToolRisk::High,
            idempotent: false,
            read_only: false,
        }
    }

    /// Declares the tool's application-assessed risk level.
    pub fn with_risk(mut self, risk: ToolRisk) -> Self {
        self.risk = risk;
        self
    }

    /// Declares whether repeating the same invocation is safe.
    pub fn with_idempotent(mut self, idempotent: bool) -> Self {
        self.idempotent = idempotent;
        self
    }

    /// Declares whether the tool is guaranteed not to change application state.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
/// Result returned by a tool execution.
pub struct ToolResult {
    /// Whether the tool execution succeeded.
    pub ok: bool,
    /// JSON output returned by the tool.
    pub output: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Optional error detail for a failed execution.
    pub error: Option<String>,
}

impl ToolResult {
    /// Creates a tool result from an adapter's explicit semantic fields.
    pub fn new(ok: bool, output: Value, error: Option<String>) -> Self {
        Self { ok, output, error }
    }

    /// Creates a successful result with the supplied JSON output.
    pub fn success(output: Value) -> Self {
        Self {
            ok: true,
            output,
            error: None,
        }
    }

    /// Creates a failed result with a null output and error message.
    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            output: Value::Null,
            error: Some(message.into()),
        }
    }
}

#[async_trait]
/// Interface implemented by tools executable by the runner.
pub trait Tool: Send + Sync {
    /// Returns the tool's declaration and validation metadata.
    fn definition(&self) -> &ToolDefinition;
    /// Cancellation is cooperative and cannot undo external effects already started by a tool.
    async fn execute(
        &self,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError>;

    /// Executes with immutable run and call correlation. Existing embedded tools
    /// may implement only [`Self::execute`]; adapters can override this method
    /// when the correlation data must cross a process boundary.
    async fn execute_with_context(
        &self,
        _: &ToolCallContext,
        arguments: Value,
        cancellation: CancellationToken,
    ) -> Result<ToolResult, HarnessError> {
        self.execute(arguments, cancellation).await
    }
}

struct RegisteredTool {
    tool: Arc<dyn Tool>,
    validator: Arc<Validator>,
}

#[derive(Default)]
/// Registry of tools and their compiled argument validators.
pub struct ToolRegistry {
    tools: HashMap<String, RegisteredTool>,
}

impl ToolRegistry {
    /// Validates and registers a tool, rejecting duplicate IDs and invalid schemas.
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), HarnessError> {
        let id = tool.definition().id.trim().to_owned();
        if id.is_empty() {
            return Err(HarnessError::InvalidTool("tool id is required".into()));
        }
        if self.tools.contains_key(&id) {
            return Err(HarnessError::InvalidTool(format!("duplicate tool: {id}")));
        }

        let schema = &tool.definition().arguments_schema;
        let defaults = AgentLimits::default();
        if serialized_len(schema)? > defaults.max_request_payload_bytes {
            return Err(HarnessError::InvalidTool(format!(
                "schema for {id} exceeds {} bytes",
                defaults.max_request_payload_bytes
            )));
        }
        ensure_json_depth("tool schema", schema, defaults.max_json_depth)
            .map_err(|error| HarnessError::InvalidTool(error.to_string()))?;
        let validator = compile_trusted_schema(schema, |error| {
            HarnessError::InvalidTool(format!("invalid schema for {id}: {error}"))
        })?;
        self.tools.insert(
            id,
            RegisteredTool {
                tool,
                validator: Arc::new(validator),
            },
        );
        Ok(())
    }

    /// Returns a registered tool by ID.
    pub fn get(&self, id: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(id).map(|entry| Arc::clone(&entry.tool))
    }

    pub(crate) fn allowed_definitions(&self, allowlist: &[String]) -> Vec<ToolDefinition> {
        allowlist
            .iter()
            .filter_map(|id| {
                self.tools
                    .get(id)
                    .map(|entry| entry.tool.definition().clone())
            })
            .collect()
    }

    pub(crate) fn validate(&self, tool_id: &str, arguments: &Value) -> Result<(), HarnessError> {
        let entry = self
            .tools
            .get(tool_id)
            .ok_or_else(|| HarnessError::InvalidTool(format!("unknown tool: {tool_id}")))?;
        entry
            .validator
            .validate(arguments)
            .map_err(|error| HarnessError::InvalidArguments(error.to_string()))
    }
}
