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
pub struct ToolCall {
    pub id: String,
    pub tool_id: String,
    pub arguments_json: String,
}

/// Immutable correlation data for one validated tool call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCallContext {
    pub run_id: String,
    pub trace_id: String,
    pub call_id: String,
    pub tool_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolRisk {
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolDefinition {
    pub id: String,
    pub name: String,
    pub description: String,
    pub arguments_schema: Value,
    pub risk: ToolRisk,
    pub idempotent: bool,
    pub read_only: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    pub ok: bool,
    pub output: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ToolResult {
    pub fn success(output: Value) -> Self {
        Self {
            ok: true,
            output,
            error: None,
        }
    }

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            output: Value::Null,
            error: Some(message.into()),
        }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
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
pub struct ToolRegistry {
    tools: HashMap<String, RegisteredTool>,
}

impl ToolRegistry {
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
