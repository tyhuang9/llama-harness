use crate::EvalError;
use llama_harness_core::{JsonMap, Message, RunStatus};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, fs, path::Path};

pub const SUPPORTED_SUITE_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvalSuite {
    #[serde(default = "default_suite_version")]
    pub version: u32,
    pub id: String,
    pub name: String,
    pub agent: String,
    #[serde(default)]
    pub agent_version: Option<String>,
    #[serde(default)]
    pub prompt_version: Option<String>,
    #[serde(default)]
    pub prompt_override: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    pub models: Vec<String>,
    #[serde(default)]
    pub defaults: EvalDefaults,
    pub cases: Vec<EvalCase>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvalDefaults {
    #[serde(default = "default_repeat")]
    pub repeat: u32,
    #[serde(default)]
    pub max_latency_ms: Option<u64>,
    #[serde(default)]
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
pub struct EvalCase {
    pub id: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub fixture: Option<EvalFixture>,
    pub input: String,
    #[serde(default)]
    pub history: Vec<Message>,
    #[serde(default)]
    pub context: JsonMap,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub agent_version: Option<String>,
    #[serde(default)]
    pub prompt_version: Option<String>,
    #[serde(default)]
    pub prompt_override: Option<String>,
    #[serde(default)]
    pub repeat: Option<u32>,
    #[serde(default)]
    pub expected: EvalExpected,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvalFixture {
    pub id: String,
    #[serde(default)]
    pub data: Value,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvalExpected {
    #[serde(default)]
    pub status: Option<RunStatus>,
    #[serde(default)]
    pub final_output_equals: Option<String>,
    #[serde(default)]
    pub final_output_contains: Vec<String>,
    #[serde(default)]
    pub structured_output_subset: Option<Value>,
    #[serde(default)]
    pub required_tools: Vec<String>,
    #[serde(default)]
    pub forbidden_tools: Vec<String>,
    #[serde(default)]
    pub tool_sequence: Option<Vec<String>>,
    #[serde(default)]
    pub expected_tool_arguments: Vec<ExpectedToolCall>,
    #[serde(default)]
    pub final_state_subset: Option<Value>,
    #[serde(default)]
    pub unresolved_items: Option<Value>,
    #[serde(default)]
    pub required_approval_tools: Vec<String>,
    #[serde(default)]
    pub forbidden_approval_tools: Vec<String>,
    #[serde(default)]
    pub max_model_calls: Option<u32>,
    #[serde(default)]
    pub max_tool_calls: Option<u32>,
    #[serde(default)]
    pub max_latency_ms: Option<u64>,
    #[serde(default)]
    pub expect_cancelled: Option<bool>,
    #[serde(default)]
    pub expected_failure: Option<ExpectedFailure>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExpectedToolCall {
    pub tool_id: String,
    pub arguments_subset: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExpectedFailure {
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub message_contains: Option<String>,
}

impl EvalSuite {
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
