use crate::{
    limits::{compile_trusted_schema, ensure_json_depth, serialized_len},
    AgentLimits, HarnessError,
};
use async_trait::async_trait;
use jsonschema::Validator;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};
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

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Strength of a tool's cooperative cancellation guarantee.
pub enum CancellationSafety {
    /// The tool has made no cancellation-safety guarantee.
    #[default]
    Unknown,
    /// The tool observes cancellation, but work already started may still complete.
    Cooperative,
    /// Cancellation is guaranteed to prevent externally visible effects after it is observed.
    Guaranteed,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Execution contexts permitted to invoke a tool through the core broker.
pub enum ToolCaller {
    /// A direct reactive model call.
    Direct,
    /// A node in a validated declarative plan.
    DeclarativePlan,
    /// A nested call from the optional programmatic sandbox.
    Programmatic,
    /// A shadow or committed speculative invocation.
    Speculative,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Explicit policy controlling whether a tool may be invoked speculatively.
pub enum SpeculationPolicy {
    /// Speculative execution is prohibited.
    #[default]
    Disabled,
    /// Speculative execution is permitted when all registry safety gates pass.
    Enabled,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Guarantee that merely issuing a tool cannot create externally visible effects.
pub enum IssueSafety {
    /// No issue-time safety guarantee has been made.
    #[default]
    Unknown,
    /// Issuing the tool is guaranteed not to create externally visible effects.
    Guaranteed,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Execution location relevant to speculative privacy guarantees.
pub enum ExecutionLocation {
    /// The execution location has not been declared.
    #[default]
    Unknown,
    /// Execution is confined to a local, private environment.
    LocalPrivate,
    /// Execution may occur in a remote environment.
    Remote,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
/// Network-egress behavior relevant to speculative privacy guarantees.
pub enum NetworkEgress {
    /// Network-egress behavior has not been declared.
    #[default]
    Unknown,
    /// The tool is guaranteed not to perform network egress.
    Prohibited,
    /// The tool may perform network egress.
    Permitted,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional JSON Schema describing successful tool output.
    pub output_schema: Option<Value>,
    /// Application-assessed risk level.
    pub risk: ToolRisk,
    /// Whether repeating the same call is safe.
    pub idempotent: bool,
    /// Whether the tool is guaranteed not to change state.
    pub read_only: bool,
    #[serde(default)]
    /// Whether independent invocations may safely execute concurrently.
    pub parallel_safe: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional key used to serialize tools that share an external resource.
    pub concurrency_key: Option<String>,
    #[serde(default)]
    /// Strength of the tool's cooperative cancellation guarantee.
    pub cancellation_safety: CancellationSafety,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Expected execution latency in milliseconds, when known.
    pub expected_latency_ms: Option<u64>,
    #[serde(default = "direct_caller_only")]
    /// Execution contexts permitted to invoke the tool.
    pub allowed_callers: BTreeSet<ToolCaller>,
    #[serde(default)]
    /// Explicit speculative-execution policy.
    pub speculation_policy: SpeculationPolicy,
    #[serde(default)]
    /// Independent guarantee that issuing the tool is itself side-effect free.
    pub issue_safety: IssueSafety,
    #[serde(default)]
    /// Execution location used by speculative privacy gates.
    pub execution_location: ExecutionLocation,
    #[serde(default)]
    /// Network-egress behavior used by speculative privacy gates.
    pub network_egress: NetworkEgress,
}

fn direct_caller_only() -> BTreeSet<ToolCaller> {
    BTreeSet::from([ToolCaller::Direct])
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
            output_schema: None,
            risk: ToolRisk::High,
            idempotent: false,
            read_only: false,
            parallel_safe: false,
            concurrency_key: None,
            cancellation_safety: CancellationSafety::Unknown,
            expected_latency_ms: None,
            allowed_callers: direct_caller_only(),
            speculation_policy: SpeculationPolicy::Disabled,
            issue_safety: IssueSafety::Unknown,
            execution_location: ExecutionLocation::Unknown,
            network_egress: NetworkEgress::Unknown,
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

    /// Declares the JSON Schema expected for successful tool output.
    pub fn with_output_schema(mut self, output_schema: Value) -> Self {
        self.output_schema = Some(output_schema);
        self
    }

    /// Declares whether independent invocations may execute concurrently.
    pub fn with_parallel_safe(mut self, parallel_safe: bool) -> Self {
        self.parallel_safe = parallel_safe;
        self
    }

    /// Sets a key used to serialize calls sharing an external resource.
    pub fn with_concurrency_key(mut self, concurrency_key: impl Into<String>) -> Self {
        self.concurrency_key = Some(concurrency_key.into());
        self
    }

    /// Declares the tool's cooperative cancellation guarantee.
    pub fn with_cancellation_safety(mut self, cancellation_safety: CancellationSafety) -> Self {
        self.cancellation_safety = cancellation_safety;
        self
    }

    /// Declares expected execution latency in milliseconds.
    pub fn with_expected_latency_ms(mut self, expected_latency_ms: u64) -> Self {
        self.expected_latency_ms = Some(expected_latency_ms);
        self
    }

    /// Replaces the set of execution contexts permitted to invoke the tool.
    pub fn with_allowed_callers(
        mut self,
        allowed_callers: impl IntoIterator<Item = ToolCaller>,
    ) -> Self {
        self.allowed_callers = allowed_callers.into_iter().collect();
        self
    }

    /// Declares the tool's explicit speculative-execution policy.
    pub fn with_speculation_policy(mut self, speculation_policy: SpeculationPolicy) -> Self {
        self.speculation_policy = speculation_policy;
        self
    }

    /// Declares the tool's issue-time side-effect guarantee.
    pub fn with_issue_safety(mut self, issue_safety: IssueSafety) -> Self {
        self.issue_safety = issue_safety;
        self
    }

    /// Declares where tool execution occurs.
    pub fn with_execution_location(mut self, execution_location: ExecutionLocation) -> Self {
        self.execution_location = execution_location;
        self
    }

    /// Declares whether tool execution can perform network egress.
    pub fn with_network_egress(mut self, network_egress: NetworkEgress) -> Self {
        self.network_egress = network_egress;
        self
    }

    /// Returns whether this tool permits the supplied execution context.
    pub fn allows_caller(&self, caller: ToolCaller) -> bool {
        self.allowed_callers.contains(&caller)
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
    output_validator: Option<Arc<Validator>>,
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
        let output_validator = if let Some(output_schema) = &tool.definition().output_schema {
            if serialized_len(output_schema)? > defaults.max_request_payload_bytes {
                return Err(HarnessError::InvalidTool(format!(
                    "output schema for {id} exceeds {} bytes",
                    defaults.max_request_payload_bytes
                )));
            }
            ensure_json_depth("tool output schema", output_schema, defaults.max_json_depth)
                .map_err(|error| HarnessError::InvalidTool(error.to_string()))?;
            Some(Arc::new(compile_trusted_schema(output_schema, |error| {
                HarnessError::InvalidTool(format!("invalid output schema for {id}: {error}"))
            })?))
        } else {
            None
        };
        validate_scheduling_metadata(tool.definition(), &id)?;
        self.tools.insert(
            id,
            RegisteredTool {
                tool,
                validator: Arc::new(validator),
                output_validator,
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
                    .filter(|entry| entry.tool.definition().allows_caller(ToolCaller::Direct))
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

    pub(crate) fn validate_output(
        &self,
        tool_id: &str,
        output: &Value,
    ) -> Result<(), HarnessError> {
        let entry = self
            .tools
            .get(tool_id)
            .ok_or_else(|| HarnessError::InvalidTool(format!("unknown tool: {tool_id}")))?;
        if let Some(validator) = &entry.output_validator {
            validator.validate(output).map_err(|_| {
                HarnessError::InvalidOutput(format!("tool {tool_id} output failed validation"))
            })?;
        }
        Ok(())
    }
}

fn validate_scheduling_metadata(definition: &ToolDefinition, id: &str) -> Result<(), HarnessError> {
    if definition.allowed_callers.is_empty() {
        return Err(HarnessError::InvalidTool(format!(
            "tool {id} must allow at least one caller"
        )));
    }
    if definition
        .concurrency_key
        .as_ref()
        .is_some_and(|key| key.trim().is_empty() || key.len() > 256)
    {
        return Err(HarnessError::InvalidTool(format!(
            "tool {id} concurrency key must contain 1 to 256 bytes"
        )));
    }
    if definition.expected_latency_ms == Some(0) {
        return Err(HarnessError::InvalidTool(format!(
            "tool {id} expected latency must be greater than zero"
        )));
    }

    let speculative_caller = definition
        .allowed_callers
        .contains(&ToolCaller::Speculative);
    let speculation_enabled = definition.speculation_policy == SpeculationPolicy::Enabled;
    if speculative_caller != speculation_enabled {
        return Err(HarnessError::InvalidTool(format!(
            "tool {id} must enable both the speculative caller and speculation policy"
        )));
    }
    if speculation_enabled
        && (!definition.read_only
            || !definition.idempotent
            || !definition.parallel_safe
            || definition.cancellation_safety != CancellationSafety::Guaranteed
            || definition.issue_safety != IssueSafety::Guaranteed
            || definition.execution_location != ExecutionLocation::LocalPrivate
            || definition.network_egress != NetworkEgress::Prohibited)
    {
        return Err(HarnessError::InvalidTool(format!(
            "tool {id} is not eligible for speculative execution"
        )));
    }
    Ok(())
}
