use crate::AgentDefinition;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fs, path::Path};
use thiserror::Error;

/// The only supported on-disk agent manifest format version.
pub const AGENT_MANIFEST_VERSION: u32 = 1;

/// A project-owned collection of inspectable agent definitions.
///
/// Loading this file does not register tools or execute an agent. Applications
/// remain responsible for mapping a listed definition to their own tools,
/// policy, approval handler, and model provider.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[non_exhaustive]
pub struct AgentManifest {
    /// On-disk manifest format version.
    pub version: u32,
    #[serde(default)]
    /// Agent definitions included in the manifest.
    pub agents: Vec<AgentDefinition>,
}

#[derive(Debug, Error)]
#[non_exhaustive]
/// Errors returned while loading or validating an agent manifest.
pub enum AgentManifestError {
    #[error("I/O error: {0}")]
    /// The manifest could not be read from disk.
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    /// JSON parsing failed.
    Json(#[from] serde_json::Error),
    #[error("YAML error: {0}")]
    /// YAML parsing failed.
    Yaml(#[from] serde_yaml::Error),
    #[error("invalid agent manifest: {0}")]
    /// The manifest contents violate the runtime contract.
    Invalid(String),
}

impl AgentManifest {
    /// Creates a manifest for the current supported format version.
    pub fn new(agents: Vec<AgentDefinition>) -> Self {
        Self {
            version: AGENT_MANIFEST_VERSION,
            agents,
        }
    }

    /// Validates the manifest version, agent definitions, and uniqueness rules.
    pub fn validate(&self) -> Result<(), AgentManifestError> {
        if self.version != AGENT_MANIFEST_VERSION {
            return Err(AgentManifestError::Invalid(format!(
                "version {} is unsupported; expected {AGENT_MANIFEST_VERSION}",
                self.version
            )));
        }
        let mut ids = HashSet::new();
        for agent in &self.agents {
            validate_agent_definition(agent)?;
            if !ids.insert(agent.id.as_str()) {
                return Err(AgentManifestError::Invalid(format!(
                    "agent ID {} appears more than once",
                    agent.id
                )));
            }
        }
        Ok(())
    }
}

/// Parses and validates an agent manifest from YAML or JSON input.
pub fn load_agent_manifest(
    input: &str,
    extension: Option<&str>,
) -> Result<AgentManifest, AgentManifestError> {
    let manifest: AgentManifest = match extension.map(str::to_ascii_lowercase).as_deref() {
        Some("yaml") | Some("yml") => serde_yaml::from_str(input)?,
        Some("json") => serde_json::from_str(input)?,
        _ => {
            return Err(AgentManifestError::Invalid(
                "agent manifest must use a .yaml, .yml, or .json extension".into(),
            ))
        }
    };
    manifest.validate()?;
    Ok(manifest)
}

/// Loads and validates a project-owned YAML or JSON agent manifest.
pub fn load_agent_manifest_path(
    path: impl AsRef<Path>,
) -> Result<AgentManifest, AgentManifestError> {
    let path = path.as_ref();
    load_agent_manifest(
        &fs::read_to_string(path)?,
        path.extension().and_then(|extension| extension.to_str()),
    )
}

fn validate_agent_definition(agent: &AgentDefinition) -> Result<(), AgentManifestError> {
    if agent.id.trim().is_empty()
        || agent.name.trim().is_empty()
        || agent.version.trim().is_empty()
        || agent.default_model.trim().is_empty()
    {
        return Err(AgentManifestError::Invalid(
            "agent id, name, version, and default model are required".into(),
        ));
    }
    let limits = &agent.limits;
    if limits.max_model_calls == 0
        || limits.max_tool_calls == 0
        || limits.max_identical_tool_calls == 0
        || limits.max_input_bytes == 0
        || limits.max_request_payload_bytes == 0
        || limits.max_model_response_bytes == 0
        || limits.max_tool_arguments_bytes == 0
        || limits.max_tool_result_bytes == 0
        || limits.max_transcript_bytes == 0
        || limits.max_json_depth == 0
    {
        return Err(AgentManifestError::Invalid(
            "agent call, byte, transcript, and depth limits must be greater than zero".into(),
        ));
    }
    if agent
        .tool_allowlist
        .iter()
        .any(|tool| tool.trim().is_empty())
    {
        return Err(AgentManifestError::Invalid(format!(
            "agent {} has an empty allowed tool ID",
            agent.id
        )));
    }
    let unique_tools = agent.tool_allowlist.iter().collect::<HashSet<_>>();
    if unique_tools.len() != agent.tool_allowlist.len() {
        return Err(AgentManifestError::Invalid(format!(
            "agent {} lists an allowed tool more than once",
            agent.id
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaml_and_json_manifests_are_validated_without_registering_tools() {
        let yaml = r#"
version: 1
agents:
  - id: task
    name: Task Agent
    version: "1"
    default_model: ollama:qwen3
    tool_allowlist: [list_tasks]
"#;
        let yaml_manifest = load_agent_manifest(yaml, Some("yaml")).unwrap();
        assert_eq!(yaml_manifest.agents[0].id, "task");
        let json = serde_json::to_string(&yaml_manifest).unwrap();
        assert_eq!(
            load_agent_manifest(&json, Some("json")).unwrap(),
            yaml_manifest
        );
    }

    #[test]
    fn invalid_versions_duplicate_ids_and_duplicate_tools_are_rejected() {
        let unsupported = r#"{"version":2,"agents":[]}"#;
        assert!(load_agent_manifest(unsupported, Some("json")).is_err());
        let duplicate_id = r#"
version: 1
agents:
  - {id: a, name: A, version: "1", default_model: ollama:a}
  - {id: a, name: B, version: "1", default_model: ollama:b}
"#;
        assert!(load_agent_manifest(duplicate_id, Some("yaml")).is_err());
        let duplicate_tool = r#"
version: 1
agents:
  - id: a
    name: A
    version: "1"
    default_model: ollama:a
    tool_allowlist: [list, list]
"#;
        assert!(load_agent_manifest(duplicate_tool, Some("yaml")).is_err());
    }
}
