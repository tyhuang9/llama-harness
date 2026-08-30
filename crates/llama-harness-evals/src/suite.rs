use crate::EvalError;
use llama_harness_core::{JsonMap, Message, RunStatus, RunStrategy};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, fs, path::Path};

/// Current evaluation-suite schema version accepted by this crate.
pub const SUPPORTED_SUITE_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// Declarative evaluation suite containing models, defaults, and cases.
pub struct EvalSuite {
    #[serde(default = "default_suite_version")]
    /// Suite schema version.
    pub version: u32,
    /// Stable suite identifier.
    pub id: String,
    /// Human-readable suite name.
    pub name: String,
    /// Agent identifier under evaluation.
    pub agent: String,
    #[serde(default)]
    /// Optional agent implementation version.
    pub agent_version: Option<String>,
    #[serde(default)]
    /// Optional prompt version.
    pub prompt_version: Option<String>,
    #[serde(default)]
    /// Optional prompt replacement applied to cases.
    pub prompt_override: Option<String>,
    #[serde(default)]
    /// Optional suite description.
    pub description: Option<String>,
    /// Models evaluated by default.
    pub models: Vec<String>,
    #[serde(default = "default_strategies")]
    /// Execution strategies evaluated when a case does not select one.
    pub strategies: Vec<RunStrategy>,
    #[serde(default)]
    /// Defaults inherited by cases.
    pub defaults: EvalDefaults,
    /// Cases included in the suite.
    pub cases: Vec<EvalCase>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// Defaults inherited by evaluation cases.
pub struct EvalDefaults {
    #[serde(default = "default_repeat")]
    /// Number of repetitions when a case does not override it.
    pub repeat: u32,
    #[serde(default)]
    /// Optional maximum run latency in milliseconds.
    pub max_latency_ms: Option<u64>,
    #[serde(default)]
    /// Tags applied by the suite author.
    pub tags: Vec<String>,
}

impl Default for EvalDefaults {
    fn default() -> Self {
        Self {
            repeat: default_repeat(),
            max_latency_ms: None,
            tags: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// One input, fixture, and expectation set within an evaluation suite.
pub struct EvalCase {
    /// Stable case identifier.
    pub id: String,
    #[serde(default)]
    /// Optional case description.
    pub description: Option<String>,
    #[serde(default)]
    /// Optional isolated fixture input.
    pub fixture: Option<EvalFixture>,
    /// User input supplied to the agent.
    pub input: String,
    #[serde(default)]
    /// Conversation history preceding the input.
    pub history: Vec<Message>,
    #[serde(default)]
    /// Application context supplied to the agent.
    pub context: JsonMap,
    #[serde(default)]
    /// Optional per-case model override.
    pub model: Option<String>,
    #[serde(default)]
    /// Optional per-case strategy override.
    pub strategy: Option<RunStrategy>,
    #[serde(default)]
    /// Optional per-case agent version.
    pub agent_version: Option<String>,
    #[serde(default)]
    /// Optional per-case prompt version.
    pub prompt_version: Option<String>,
    #[serde(default)]
    /// Optional per-case prompt replacement.
    pub prompt_override: Option<String>,
    #[serde(default)]
    /// Optional repetition count for this case.
    pub repeat: Option<u32>,
    #[serde(default)]
    /// Expectations applied to the observed result.
    pub expected: EvalExpected,
    #[serde(default)]
    /// Tags applied by the case author.
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// Named fixture data supplied to an application-owned executor.
pub struct EvalFixture {
    /// Stable fixture identifier.
    pub id: String,
    #[serde(default)]
    /// Fixture payload.
    pub data: Value,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// Expectations evaluated against one observed case.
pub struct EvalExpected {
    #[serde(default)]
    /// Expected terminal run status.
    pub status: Option<RunStatus>,
    #[serde(default)]
    /// Exact expected final text.
    pub final_output_equals: Option<String>,
    #[serde(default)]
    /// Text fragments expected in the final output.
    pub final_output_contains: Vec<String>,
    #[serde(default)]
    /// JSON subset expected in the final output.
    pub structured_output_subset: Option<Value>,
    #[serde(default)]
    /// Tool identifiers that must be called.
    pub required_tools: Vec<String>,
    #[serde(default)]
    /// Tool identifiers that must not be called.
    pub forbidden_tools: Vec<String>,
    #[serde(default)]
    /// Exact sequence of tool identifiers expected.
    pub tool_sequence: Option<Vec<String>>,
    #[serde(default)]
    /// Tool calls whose arguments must contain a JSON subset.
    pub expected_tool_arguments: Vec<ExpectedToolCall>,
    #[serde(default)]
    /// JSON subset expected in the final application state.
    pub final_state_subset: Option<Value>,
    #[serde(default)]
    /// JSON subset expected in unresolved items.
    pub unresolved_items: Option<Value>,
    #[serde(default)]
    /// Tools that must receive approval.
    pub required_approval_tools: Vec<String>,
    #[serde(default)]
    /// Tools that must not receive approval.
    pub forbidden_approval_tools: Vec<String>,
    #[serde(default)]
    /// Maximum number of model calls.
    pub max_model_calls: Option<u32>,
    #[serde(default)]
    /// Maximum number of tool calls.
    pub max_tool_calls: Option<u32>,
    #[serde(default)]
    /// Maximum run latency in milliseconds.
    pub max_latency_ms: Option<u64>,
    #[serde(default)]
    /// Whether cancellation is expected.
    pub expect_cancelled: Option<bool>,
    #[serde(default)]
    /// Error metadata expected from a failed run.
    pub expected_failure: Option<ExpectedFailure>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// Expected arguments for a named tool call.
pub struct ExpectedToolCall {
    /// Tool identifier to match.
    pub tool_id: String,
    /// JSON subset expected in the call arguments.
    pub arguments_subset: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// Error metadata expected from a run.
pub struct ExpectedFailure {
    #[serde(default)]
    /// Optional error code to match.
    pub code: Option<String>,
    #[serde(default)]
    /// Optional substring required in the error message.
    pub message_contains: Option<String>,
}

impl EvalSuite {
    /// Validates the suite schema, identifiers, models, and cases.
    pub fn validate(&self) -> Result<(), EvalError> {
        if self.version != SUPPORTED_SUITE_VERSION {
            return Err(EvalError::InvalidSuite(format!(
                "suite version {} is not supported (expected {SUPPORTED_SUITE_VERSION})",
                self.version
            )));
        }
        validate_identifier("suite ID", &self.id)?;
        validate_nonempty("suite name", &self.name)?;
        validate_identifier("agent ID", &self.agent)?;
        if self.models.is_empty() {
            return Err(EvalError::InvalidSuite(
                "at least one model is required".into(),
            ));
        }
        for model in &self.models {
            validate_nonempty("model", model)?;
        }
        if self.strategies.is_empty() {
            return Err(EvalError::InvalidSuite(
                "at least one strategy is required".into(),
            ));
        }
        let mut strategies = BTreeSet::new();
        for strategy in &self.strategies {
            let name = strategy_name(*strategy);
            if !strategies.insert(name) {
                return Err(EvalError::InvalidSuite(format!(
                    "duplicate suite strategy: {name}"
                )));
            }
        }
        if self.defaults.repeat == 0 {
            return Err(EvalError::InvalidSuite(
                "default repeat must be greater than zero".into(),
            ));
        }
        if self.cases.is_empty() {
            return Err(EvalError::InvalidSuite(
                "at least one case is required".into(),
            ));
        }
        let mut case_ids = BTreeSet::new();
        for case in &self.cases {
            validate_identifier("case ID", &case.id)?;
            if !case_ids.insert(&case.id) {
                return Err(EvalError::InvalidSuite(format!(
                    "duplicate case ID: {}",
                    case.id
                )));
            }
            case.validate()?;
        }
        Ok(())
    }
}

impl EvalCase {
    pub(crate) fn validate(&self) -> Result<(), EvalError> {
        validate_nonempty("case input", &self.input)?;
        if self.repeat == Some(0) {
            return Err(EvalError::InvalidSuite(format!(
                "case {} repeat must be greater than zero",
                self.id
            )));
        }
        if let Some(fixture) = &self.fixture {
            validate_identifier("fixture ID", &fixture.id)?;
        }
        if let Some(model) = &self.model {
            validate_nonempty("case model", model)?;
        }
        for tool in self
            .expected
            .required_tools
            .iter()
            .chain(self.expected.forbidden_tools.iter())
            .chain(self.expected.required_approval_tools.iter())
            .chain(self.expected.forbidden_approval_tools.iter())
        {
            validate_identifier("tool ID", tool)?;
        }
        for tool in &self.expected.expected_tool_arguments {
            validate_identifier("expected tool argument tool ID", &tool.tool_id)?;
        }
        Ok(())
    }
}

/// Parses and validates a suite from JSON or YAML text.
pub fn load_suite(input: &str, extension: Option<&str>) -> Result<EvalSuite, EvalError> {
    let suite: EvalSuite = match extension.map(str::to_ascii_lowercase).as_deref() {
        Some("json") => serde_json::from_str(input)?,
        Some("yaml") | Some("yml") | None => serde_yaml::from_str(input)?,
        Some(extension) => {
            return Err(EvalError::InvalidSuite(format!(
                "unsupported suite extension: {extension}; use .yaml, .yml, or .json"
            )))
        }
    };
    suite.validate()?;
    Ok(suite)
}

/// Reads, parses, and validates a suite from a `.json`, `.yaml`, or `.yml` path.
pub fn load_suite_path(path: impl AsRef<Path>) -> Result<EvalSuite, EvalError> {
    let path = path.as_ref();
    let input = fs::read_to_string(path)?;
    load_suite(&input, path.extension().and_then(|value| value.to_str()))
}

fn default_suite_version() -> u32 {
    SUPPORTED_SUITE_VERSION
}

fn default_repeat() -> u32 {
    1
}

fn default_strategies() -> Vec<RunStrategy> {
    vec![RunStrategy::Adaptive]
}

fn strategy_name(strategy: RunStrategy) -> &'static str {
    match strategy {
        RunStrategy::Adaptive => "adaptive",
        RunStrategy::Direct => "direct",
        RunStrategy::DeclarativePlan => "declarative_plan",
        RunStrategy::Programmatic => "programmatic",
    }
}

fn validate_identifier(label: &str, value: &str) -> Result<(), EvalError> {
    validate_nonempty(label, value)?;
    if !value
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.'))
    {
        return Err(EvalError::InvalidSuite(format!(
            "{label} contains unsupported characters: {value}"
        )));
    }
    Ok(())
}

fn validate_nonempty(label: &str, value: &str) -> Result<(), EvalError> {
    if value.trim().is_empty() {
        return Err(EvalError::InvalidSuite(format!("{label} is required")));
    }
    Ok(())
}
