use crate::config::{AgentConfig, AppConfig, GenerationSettings};
use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::json;
use std::{collections::HashSet, path::Path};
use tokio::fs;

pub const NOTE_APP_ID: &str = "note";
pub const NOTE_AGENT_ID: &str = "note-assistant";
pub const NOTE_TOOL_IDS: [&str; 3] = ["notes.read", "notes.search", "reminders.create"];

#[derive(Clone, Debug, Default)]
pub struct DomainCatalog {
    pub models: Vec<ModelConfig>,
    pub agents: Vec<AgentConfig>,
    pub apps: Vec<ClientAppConfig>,
    pub tools: Vec<ToolConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ModelConfig {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub model_name: String,
    pub status: String,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ToolConfig {
    pub id: String,
    pub name: String,
    pub description: String,
    pub risk_level: String,
    pub enabled: bool,
    pub input_schema: Option<serde_json::Value>,
    pub output_schema: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ClientAppConfig {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub default_agent_id: String,
    pub allowed_agent_ids: Vec<String>,
    pub allowed_tool_ids: Option<Vec<String>>,
    pub enabled: bool,
}

#[derive(Clone, Debug)]
pub struct ResolvedAppPolicy {
    pub app: ClientAppConfig,
    pub agent: AgentConfig,
    pub allowed_agents: Vec<AgentConfig>,
    pub tools: Vec<ToolConfig>,
    pub model: ResolvedModel,
    pub provider_id: String,
    pub provider_model: String,
    pub generation: GenerationSettings,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedModel {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub model_name: String,
    pub status: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppCapabilitiesResponse {
    pub app_id: String,
    pub app_name: String,
    pub default_agent: AgentSummary,
    pub allowed_agents: Vec<AgentSummary>,
    pub tools: Vec<ToolSummary>,
    pub model: ResolvedModel,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSummary {
    pub id: String,
    pub name: String,
    pub description: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSummary {
    pub id: String,
    pub name: String,
    pub description: String,
    pub risk_level: String,
    pub enabled: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppPolicyErrorKind {
    NotFound,
    Disabled,
    Forbidden,
    Misconfigured,
}

#[derive(Clone, Debug)]
pub struct AppPolicyError {
    pub kind: AppPolicyErrorKind,
    pub message: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            provider: "ollama".to_string(),
            model_name: String::new(),
            status: "requires_selection".to_string(),
            metadata: None,
        }
    }
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            description: String::new(),
            risk_level: "low".to_string(),
            enabled: false,
            input_schema: None,
            output_schema: None,
        }
    }
}

impl Default for ClientAppConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            description: None,
            default_agent_id: String::new(),
            allowed_agent_ids: Vec::new(),
            allowed_tool_ids: None,
            enabled: true,
        }
    }
}

impl AppPolicyError {
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            kind: AppPolicyErrorKind::NotFound,
            message: message.into(),
        }
    }

    pub fn disabled(message: impl Into<String>) -> Self {
        Self {
            kind: AppPolicyErrorKind::Disabled,
            message: message.into(),
        }
    }

    pub fn forbidden(message: impl Into<String>) -> Self {
        Self {
            kind: AppPolicyErrorKind::Forbidden,
            message: message.into(),
        }
    }

    pub fn misconfigured(message: impl Into<String>) -> Self {
        Self {
            kind: AppPolicyErrorKind::Misconfigured,
            message: message.into(),
        }
    }
}

impl AppCapabilitiesResponse {
    pub fn from_policy(policy: &ResolvedAppPolicy) -> Self {
        Self {
            app_id: policy.app.id.clone(),
            app_name: policy.app.name.clone(),
            default_agent: AgentSummary::from_agent(&policy.agent),
            allowed_agents: policy
                .allowed_agents
                .iter()
                .map(AgentSummary::from_agent)
                .collect(),
            tools: policy.tools.iter().map(ToolSummary::from_tool).collect(),
            model: policy.model.clone(),
            warnings: policy.warnings.clone(),
        }
    }
}

impl AgentSummary {
    fn from_agent(agent: &AgentConfig) -> Self {
        Self {
            id: agent.id.clone(),
            name: agent.name.clone(),
            description: agent.description.clone(),
        }
    }
}

impl ToolSummary {
    fn from_tool(tool: &ToolConfig) -> Self {
        Self {
            id: tool.id.clone(),
            name: tool.name.clone(),
            description: tool.description.clone(),
            risk_level: tool.risk_level.clone(),
            enabled: tool.enabled,
        }
    }
}

pub async fn load_domain_catalog(dir: &Path, runtime_config: &AppConfig) -> Result<DomainCatalog> {
    fs::create_dir_all(dir)
        .await
        .with_context(|| format!("failed to create {}", dir.display()))?;

    let models = load_or_seed(&dir.join("models.json"), seed_models(runtime_config)).await?;
    let tools = load_or_seed(&dir.join("tools.json"), seed_tools()).await?;
    let mut agents = load_or_seed(
        &dir.join("agents.json"),
        seed_agents(runtime_config, &models),
    )
    .await?;
    if merge_legacy_agents(&mut agents, &runtime_config.agents) {
        save_agents(dir, &agents).await?;
    }
    let apps = load_or_seed(&dir.join("apps.json"), seed_apps()).await?;

    Ok(DomainCatalog {
        models,
        agents,
        apps,
        tools,
    })
}

pub async fn save_agents(dir: &Path, agents: &[AgentConfig]) -> Result<()> {
    save_pretty_json(&dir.join("agents.json"), agents).await
}

pub async fn save_apps(dir: &Path, apps: &[ClientAppConfig]) -> Result<()> {
    save_pretty_json(&dir.join("apps.json"), apps).await
}

pub async fn save_tools(dir: &Path, tools: &[ToolConfig]) -> Result<()> {
    save_pretty_json(&dir.join("tools.json"), tools).await
}

pub fn resolve_app_policy(
    catalog: &DomainCatalog,
    runtime_config: &AppConfig,
    app_id: &str,
    requested_agent_id: Option<&str>,
    fallback_ollama_model: Option<&str>,
) -> Result<ResolvedAppPolicy, AppPolicyError> {
    let app_id = app_id.trim();
    if app_id.is_empty() {
        return Err(AppPolicyError::misconfigured("appId is required"));
    }

    let app = find_app(catalog, app_id)
        .cloned()
        .ok_or_else(|| AppPolicyError::not_found(format!("unknown app: {app_id}")))?;
    if !app.enabled {
        return Err(AppPolicyError::disabled(format!(
            "app '{}' is disabled",
            app.id
        )));
    }

    let selected_agent_id = requested_agent_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(app.default_agent_id.trim());
    if selected_agent_id.is_empty() {
        return Err(AppPolicyError::misconfigured(format!(
            "app '{}' has no default agent",
            app.id
        )));
    }
    if !contains_id(&app.allowed_agent_ids, selected_agent_id) {
        return Err(AppPolicyError::forbidden(format!(
            "agent '{selected_agent_id}' is not allowed for app '{}'",
            app.id
        )));
    }

    let agent = find_agent(catalog, selected_agent_id)
        .cloned()
        .ok_or_else(|| AppPolicyError::not_found(format!("unknown agent: {selected_agent_id}")))?;
    if !agent_enabled(&agent) {
        return Err(AppPolicyError::disabled(format!(
            "agent '{}' is disabled",
            agent.id
        )));
    }

    let allowed_agents = app
        .allowed_agent_ids
        .iter()
        .filter_map(|agent_id| find_agent(catalog, agent_id).cloned())
        .collect::<Vec<_>>();

    let mut warnings = Vec::new();
    for agent_id in &app.allowed_agent_ids {
        if find_agent(catalog, agent_id).is_none() {
            warnings.push(format!(
                "app '{}' references missing agent '{}'",
                app.id, agent_id
            ));
        }
    }

    let tools = resolve_tools(catalog, &app, &agent, &mut warnings);
    let model = resolve_model(
        catalog,
        runtime_config,
        &agent,
        fallback_ollama_model,
        &mut warnings,
    );

    let provider_id = if model.provider.trim().is_empty() {
        agent.default_provider_id.clone()
    } else {
        model.provider.clone()
    };
    let provider_id = if provider_id.trim().is_empty() {
        "ollama".to_string()
    } else {
        provider_id.trim().to_ascii_lowercase()
    };
    let provider_model = if model.model_name.trim().is_empty() {
        agent.default_model.trim().to_string()
    } else {
        model.model_name.clone()
    };

    let mut generation = runtime_config.generation.clone();
    if let Some(temperature) = agent.temperature {
        generation.temperature = temperature;
    }
    if let Some(max_tokens) = agent.max_tokens {
        generation.max_tokens = max_tokens;
    }

    Ok(ResolvedAppPolicy {
        app,
        agent,
        allowed_agents,
        tools,
        model,
        provider_id,
        provider_model,
        generation,
        warnings,
    })
}

pub fn find_agent<'a>(catalog: &'a DomainCatalog, agent_id: &str) -> Option<&'a AgentConfig> {
    catalog
        .agents
        .iter()
        .find(|agent| agent.id.eq_ignore_ascii_case(agent_id.trim()))
}

pub fn find_agent_mut<'a>(
    catalog: &'a mut DomainCatalog,
    agent_id: &str,
) -> Option<&'a mut AgentConfig> {
    catalog
        .agents
        .iter_mut()
        .find(|agent| agent.id.eq_ignore_ascii_case(agent_id.trim()))
}

pub fn find_app<'a>(catalog: &'a DomainCatalog, app_id: &str) -> Option<&'a ClientAppConfig> {
    catalog
        .apps
        .iter()
        .find(|app| app.id.eq_ignore_ascii_case(app_id.trim()))
}

pub fn find_app_mut<'a>(
    catalog: &'a mut DomainCatalog,
    app_id: &str,
) -> Option<&'a mut ClientAppConfig> {
    catalog
        .apps
        .iter_mut()
        .find(|app| app.id.eq_ignore_ascii_case(app_id.trim()))
}

pub fn agent_enabled(agent: &AgentConfig) -> bool {
    agent.enabled && agent.status != "paused"
}

pub fn model_config_for_agent<'a>(
    catalog: &'a DomainCatalog,
    agent: &AgentConfig,
) -> Option<&'a ModelConfig> {
    agent.default_model_id.as_deref().and_then(|model_id| {
        catalog
            .models
            .iter()
            .find(|model| model.id.eq_ignore_ascii_case(model_id.trim()))
    })
}

pub fn normalize_config_id(value: &str) -> String {
    value
        .trim()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn resolve_tools(
    catalog: &DomainCatalog,
    app: &ClientAppConfig,
    agent: &AgentConfig,
    warnings: &mut Vec<String>,
) -> Vec<ToolConfig> {
    let mut allowed_ids = agent
        .allowed_tool_ids
        .iter()
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty())
        .collect::<Vec<_>>();

    if let Some(app_tool_ids) = &app.allowed_tool_ids {
        let app_tool_ids = app_tool_ids
            .iter()
            .map(|id| id.trim().to_ascii_lowercase())
            .collect::<HashSet<_>>();
        allowed_ids.retain(|id| app_tool_ids.contains(&id.to_ascii_lowercase()));
    }

    let mut tools = Vec::new();
    for tool_id in allowed_ids {
        match catalog
            .tools
            .iter()
            .find(|tool| tool.id.eq_ignore_ascii_case(&tool_id))
        {
            Some(tool) => {
                if !tool.enabled {
                    warnings.push(format!("tool '{}' is configured but disabled", tool.id));
                }
                tools.push(tool.clone());
            }
            None => warnings.push(format!("configured tool '{tool_id}' is missing")),
        }
    }
    tools
}

fn resolve_model(
    catalog: &DomainCatalog,
    runtime_config: &AppConfig,
    agent: &AgentConfig,
    fallback_ollama_model: Option<&str>,
    warnings: &mut Vec<String>,
) -> ResolvedModel {
    if let Some(model) = model_config_for_agent(catalog, agent) {
        let model_name = if model.model_name.trim().is_empty() {
            fallback_model_name(runtime_config, agent, fallback_ollama_model)
        } else {
            model.model_name.clone()
        };
        let status = if model_name.is_empty() {
            warnings.push(format!(
                "model '{}' requires a concrete Ollama model selection",
                model.id
            ));
            "requires_selection".to_string()
        } else if model.status.trim().is_empty() {
            "available".to_string()
        } else {
            model.status.clone()
        };
        return ResolvedModel {
            id: model.id.clone(),
            name: if model.name.trim().is_empty() || model.model_name.trim().is_empty() {
                model_name.clone()
            } else {
                model.name.clone()
            },
            provider: model.provider.clone(),
            model_name,
            status,
        };
    }

    let model_name = fallback_model_name(runtime_config, agent, fallback_ollama_model);

    let provider = if agent.default_provider_id.trim().is_empty() {
        "ollama".to_string()
    } else {
        agent.default_provider_id.trim().to_ascii_lowercase()
    };

    let status = if model_name.is_empty() {
        warnings.push(format!(
            "agent '{}' has no resolved default model",
            agent.id
        ));
        "requires_selection".to_string()
    } else {
        "available".to_string()
    };
    let id = if model_name.is_empty() {
        "ollama-default".to_string()
    } else {
        format!("{}-{}", provider, normalize_config_id(&model_name))
    };

    ResolvedModel {
        id,
        name: if model_name.is_empty() {
            "Default model".to_string()
        } else {
            model_name.clone()
        },
        provider,
        model_name,
        status,
    }
}

fn fallback_model_name(
    runtime_config: &AppConfig,
    agent: &AgentConfig,
    fallback_ollama_model: Option<&str>,
) -> String {
    agent
        .default_model
        .trim()
        .to_string()
        .or_nonempty()
        .or_else(|| {
            runtime_config
                .default_model_for_provider(&agent.default_provider_id)
                .map(|model| model.trim().to_string())
                .filter(|model| !model.is_empty())
        })
        .or_else(|| {
            fallback_ollama_model
                .map(str::trim)
                .filter(|model| !model.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_default()
}

trait NonEmptyString {
    fn or_nonempty(self) -> Option<String>;
}

impl NonEmptyString for String {
    fn or_nonempty(self) -> Option<String> {
        if self.is_empty() {
            None
        } else {
            Some(self)
        }
    }
}

fn contains_id(ids: &[String], target: &str) -> bool {
    ids.iter().any(|id| id.eq_ignore_ascii_case(target.trim()))
}

async fn load_or_seed<T>(path: &Path, seed: Vec<T>) -> Result<Vec<T>>
where
    T: Serialize + DeserializeOwned,
{
    match fs::read_to_string(path).await {
        Ok(contents) => serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse {}", path.display())),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            save_pretty_json(path, &seed).await?;
            Ok(seed)
        }
        Err(err) => Err(err).with_context(|| format!("failed to read {}", path.display())),
    }
}

async fn save_pretty_json<T>(path: &Path, value: &T) -> Result<()>
where
    T: Serialize + ?Sized,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let contents = serde_json::to_string_pretty(value)?;
    fs::write(path, format!("{contents}\n"))
        .await
        .with_context(|| format!("failed to write {}", path.display()))
}

fn seed_models(runtime_config: &AppConfig) -> Vec<ModelConfig> {
    let model_name = seed_model_name(runtime_config);
    let model_id = if model_name.is_empty() {
        "ollama-default".to_string()
    } else {
        format!("ollama-{}", normalize_config_id(&model_name))
    };

    vec![ModelConfig {
        id: model_id,
        name: if model_name.is_empty() {
            "Default Ollama model".to_string()
        } else {
            model_name.clone()
        },
        provider: "ollama".to_string(),
        model_name,
        status: "available".to_string(),
        metadata: Some(json!({ "source": "seed" })),
    }]
}

fn seed_agents(runtime_config: &AppConfig, models: &[ModelConfig]) -> Vec<AgentConfig> {
    let seed_model = models.first().cloned().unwrap_or_default();
    let mut note_agent = AgentConfig {
        id: NOTE_AGENT_ID.to_string(),
        name: "Note Assistant".to_string(),
        role: "Note assistant".to_string(),
        description: "Helps summarize, analyze, and reason about notes.".to_string(),
        system_prompt: "You are a local assistant for a note-taking app. Help summarize notes, extract action items, analyze selected text, and answer questions using provided note context.".to_string(),
        default_model_id: Some(seed_model.id.clone()),
        default_provider_id: seed_model.provider.clone(),
        default_model: seed_model.model_name.clone(),
        allowed_tool_ids: NOTE_TOOL_IDS.iter().map(|id| id.to_string()).collect(),
        status: "active".to_string(),
        enabled: true,
        updated_at: "seed".to_string(),
        ..AgentConfig::default()
    };
    if note_agent.default_provider_id.trim().is_empty() {
        note_agent.default_provider_id = "ollama".to_string();
    }

    let mut agents = vec![note_agent];
    let mut seen = HashSet::from([NOTE_AGENT_ID.to_string()]);
    for agent in &runtime_config.agents {
        let key = agent.id.to_ascii_lowercase();
        if !key.is_empty() && !seen.contains(&key) {
            agents.push(agent.clone());
            seen.insert(key);
        }
    }
    agents
}

fn merge_legacy_agents(agents: &mut Vec<AgentConfig>, legacy_agents: &[AgentConfig]) -> bool {
    let mut changed = false;
    let mut seen = agents
        .iter()
        .map(|agent| agent.id.to_ascii_lowercase())
        .collect::<HashSet<_>>();

    for agent in legacy_agents {
        let key = agent.id.trim().to_ascii_lowercase();
        if key.is_empty() || seen.contains(&key) {
            continue;
        }
        agents.push(agent.clone());
        seen.insert(key);
        changed = true;
    }

    changed
}

fn seed_apps() -> Vec<ClientAppConfig> {
    vec![ClientAppConfig {
        id: NOTE_APP_ID.to_string(),
        name: "Note".to_string(),
        description: Some("Local Note app integration".to_string()),
        default_agent_id: NOTE_AGENT_ID.to_string(),
        allowed_agent_ids: vec![NOTE_AGENT_ID.to_string()],
        allowed_tool_ids: Some(NOTE_TOOL_IDS.iter().map(|id| id.to_string()).collect()),
        enabled: true,
    }]
}

fn seed_tools() -> Vec<ToolConfig> {
    vec![
        ToolConfig {
            id: "notes.read".to_string(),
            name: "Read Notes".to_string(),
            description: "Read note content supplied by the Note app.".to_string(),
            risk_level: "low".to_string(),
            enabled: false,
            input_schema: None,
            output_schema: None,
        },
        ToolConfig {
            id: "notes.search".to_string(),
            name: "Search Notes".to_string(),
            description: "Search note content supplied by the Note app.".to_string(),
            risk_level: "low".to_string(),
            enabled: false,
            input_schema: None,
            output_schema: None,
        },
        ToolConfig {
            id: "reminders.create".to_string(),
            name: "Create Reminder".to_string(),
            description: "Create a reminder requested from note context.".to_string(),
            risk_level: "medium".to_string(),
            enabled: false,
            input_schema: None,
            output_schema: None,
        },
    ]
}

fn seed_model_name(runtime_config: &AppConfig) -> String {
    runtime_config
        .default_model
        .as_deref()
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_string)
        .or_else(|| {
            runtime_config
                .agents
                .iter()
                .find(|agent| !agent.default_model.trim().is_empty())
                .map(|agent| agent.default_model.trim().to_string())
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_policy_intersects_app_and_agent_tools() {
        let mut agent = AgentConfig {
            id: NOTE_AGENT_ID.to_string(),
            name: "Note Assistant".to_string(),
            default_provider_id: "ollama".to_string(),
            default_model: "llama3".to_string(),
            allowed_tool_ids: vec!["notes.read".to_string(), "reminders.create".to_string()],
            status: "active".to_string(),
            enabled: true,
            ..AgentConfig::default()
        };
        agent.default_model_id = None;
        let catalog = DomainCatalog {
            agents: vec![agent],
            apps: vec![ClientAppConfig {
                id: NOTE_APP_ID.to_string(),
                name: "Note".to_string(),
                default_agent_id: NOTE_AGENT_ID.to_string(),
                allowed_agent_ids: vec![NOTE_AGENT_ID.to_string()],
                allowed_tool_ids: Some(vec!["notes.read".to_string()]),
                enabled: true,
                description: None,
            }],
            tools: seed_tools(),
            models: Vec::new(),
        };

        let resolved = resolve_app_policy(&catalog, &AppConfig::default(), NOTE_APP_ID, None, None)
            .expect("policy should resolve");

        assert_eq!(resolved.agent.id, NOTE_AGENT_ID);
        assert_eq!(resolved.tools.len(), 1);
        assert_eq!(resolved.tools[0].id, "notes.read");
        assert_eq!(resolved.model.model_name, "llama3");
    }

    #[test]
    fn app_policy_rejects_disallowed_agent() {
        let catalog = DomainCatalog {
            agents: vec![AgentConfig {
                id: "other-agent".to_string(),
                status: "active".to_string(),
                enabled: true,
                ..AgentConfig::default()
            }],
            apps: vec![ClientAppConfig {
                id: NOTE_APP_ID.to_string(),
                name: "Note".to_string(),
                default_agent_id: NOTE_AGENT_ID.to_string(),
                allowed_agent_ids: vec![NOTE_AGENT_ID.to_string()],
                allowed_tool_ids: None,
                enabled: true,
                description: None,
            }],
            tools: Vec::new(),
            models: Vec::new(),
        };

        let error = resolve_app_policy(
            &catalog,
            &AppConfig::default(),
            NOTE_APP_ID,
            Some("other-agent"),
            None,
        )
        .expect_err("agent should be denied");

        assert_eq!(error.kind, AppPolicyErrorKind::Forbidden);
    }
}
