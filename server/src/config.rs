use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    env,
    path::{Path, PathBuf},
};
use tokio::fs;

pub const REDACTED_SECRET: &str = "__configured__";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub ollama_endpoint: String,
    pub default_provider: String,
    pub default_model: Option<String>,
    pub generation: GenerationSettings,
    pub instructions: InstructionSettings,
    pub logging_enabled: bool,
    pub api_token: Option<String>,
    pub theme: String,
    pub litellm: LiteLlmSettings,
    pub litellm_providers: Vec<LiteLlmProviderConfig>,
    pub model_routes: Vec<ModelRoute>,
    pub agents: Vec<AgentConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GenerationSettings {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct InstructionSettings {
    pub enabled: bool,
    pub system_prompt: String,
    pub tool_context: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LiteLlmSettings {
    pub enabled: bool,
    pub base_url: String,
    pub api_key: Option<String>,
    pub timeout_ms: u64,
    pub default_model: Option<String>,
    pub managed_config_path: Option<String>,
    pub allow_unconfigured_models: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LiteLlmProviderConfig {
    pub id: String,
    pub enabled: bool,
    pub provider_type: String,
    pub display_name: String,
    pub api_key_env_var: String,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelRoute {
    pub id: String,
    pub enabled: bool,
    pub display_name: String,
    pub provider: String,
    pub provider_family: String,
    pub model_alias: String,
    pub litellm_model: String,
    pub api_key_env_var: String,
    pub api_base: Option<String>,
    pub notes: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentConfig {
    pub id: String,
    pub name: String,
    pub role: String,
    pub description: String,
    #[serde(alias = "systemPrompt")]
    pub system_prompt: String,
    #[serde(alias = "defaultModelId")]
    pub default_model_id: Option<String>,
    #[serde(alias = "defaultProviderId")]
    pub default_provider_id: String,
    #[serde(alias = "defaultModel")]
    pub default_model: String,
    #[serde(alias = "defaultEnvironment")]
    pub default_environment: String,
    pub autonomy: String,
    pub permissions: AgentPermissions,
    #[serde(alias = "allowedToolIds")]
    pub allowed_tool_ids: Vec<String>,
    pub temperature: Option<f32>,
    #[serde(alias = "maxTokens")]
    pub max_tokens: Option<u32>,
    pub enabled: bool,
    pub status: String,
    pub tasks_run: u64,
    #[serde(alias = "updatedAt")]
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentPermissions {
    pub browser: bool,
    pub file_read: bool,
    pub file_write: bool,
    pub terminal: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            ollama_endpoint: "http://localhost:11434".to_string(),
            default_provider: "ollama".to_string(),
            default_model: None,
            generation: GenerationSettings::default(),
            instructions: InstructionSettings::default(),
            logging_enabled: true,
            api_token: None,
            theme: "dark".to_string(),
            litellm: LiteLlmSettings::default(),
            litellm_providers: Vec::new(),
            model_routes: Vec::new(),
            agents: Vec::new(),
        }
    }
}

impl Default for GenerationSettings {
    fn default() -> Self {
        Self {
            temperature: 0.2,
            top_p: 0.9,
            max_tokens: 512,
        }
    }
}

impl Default for InstructionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            system_prompt: String::new(),
            tool_context: String::new(),
        }
    }
}

impl Default for LiteLlmSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: "http://127.0.0.1:4000".to_string(),
            api_key: None,
            timeout_ms: 120_000,
            default_model: None,
            managed_config_path: None,
            allow_unconfigured_models: false,
        }
    }
}

impl Default for LiteLlmProviderConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            enabled: true,
            provider_type: "openai".to_string(),
            display_name: String::new(),
            api_key_env_var: "OPENAI_API_KEY".to_string(),
            api_key: None,
            api_base: None,
        }
    }
}

impl Default for ModelRoute {
    fn default() -> Self {
        Self {
            id: String::new(),
            enabled: true,
            display_name: String::new(),
            provider: "litellm".to_string(),
            provider_family: "custom".to_string(),
            model_alias: String::new(),
            litellm_model: String::new(),
            api_key_env_var: String::new(),
            api_base: None,
            notes: None,
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: "New agent".to_string(),
            role: "General assistant".to_string(),
            description: String::new(),
            system_prompt: String::new(),
            default_model_id: None,
            default_provider_id: "ollama".to_string(),
            default_model: String::new(),
            default_environment: "planner".to_string(),
            autonomy: "ask".to_string(),
            permissions: AgentPermissions::default(),
            allowed_tool_ids: Vec::new(),
            temperature: None,
            max_tokens: None,
            enabled: true,
            status: "draft".to_string(),
            tasks_run: 0,
            updated_at: String::new(),
        }
    }
}

impl Default for AgentPermissions {
    fn default() -> Self {
        Self {
            browser: true,
            file_read: true,
            file_write: false,
            terminal: false,
        }
    }
}

impl AppConfig {
    pub fn default_model_for_provider(&self, provider: &str) -> Option<String> {
        match provider {
            "litellm" => self.litellm.default_model.clone(),
            _ => self.default_model.clone(),
        }
    }

    pub fn redacted_for_response(&self) -> Self {
        let mut config = self.clone();
        if config.litellm.api_key.is_some() {
            config.litellm.api_key = Some(REDACTED_SECRET.to_string());
        }
        for provider in &mut config.litellm_providers {
            if provider.api_key.is_some() {
                provider.api_key = Some(REDACTED_SECRET.to_string());
            }
        }
        config
    }
}

pub fn default_config_path() -> PathBuf {
    if let Ok(path) = env::var("LLAMA_HARNESS_CONFIG") {
        return PathBuf::from(path);
    }

    let local = PathBuf::from("config.json");
    if local.exists() {
        return local;
    }

    let parent = PathBuf::from("../config.json");
    if parent.exists() {
        return parent;
    }

    local
}

pub fn default_runs_path() -> PathBuf {
    if let Ok(path) = env::var("LLAMA_HARNESS_RUNS_LOG") {
        return PathBuf::from(path);
    }

    let local = PathBuf::from("runs.jsonl");
    if local.exists() {
        return local;
    }

    let parent = PathBuf::from("../runs.jsonl");
    if parent.parent().is_some_and(Path::exists) {
        return parent;
    }

    local
}

pub fn default_audit_path() -> PathBuf {
    if let Ok(path) = env::var("LLAMA_HARNESS_AUDIT_LOG") {
        return PathBuf::from(path);
    }

    let local = PathBuf::from("logs/audit.jsonl");
    if local.exists() {
        return local;
    }

    let parent = PathBuf::from("../logs/audit.jsonl");
    if parent.parent().is_some_and(Path::exists) {
        return parent;
    }

    local
}

pub fn default_catalog_dir(config_path: &Path) -> PathBuf {
    if let Ok(path) = env::var("LLAMA_HARNESS_CONFIG_DIR") {
        return PathBuf::from(path);
    }

    let parent = config_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty());
    parent
        .map(|path| path.join("config"))
        .unwrap_or_else(|| PathBuf::from("config"))
}

pub async fn load_config(path: &Path) -> Result<AppConfig> {
    match fs::read_to_string(path).await {
        Ok(contents) => serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse config at {}", path.display())),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            let config = AppConfig::default();
            save_config(path, &config).await?;
            Ok(config)
        }
        Err(err) => Err(err).with_context(|| format!("failed to read {}", path.display())),
    }
}

pub async fn save_config(path: &Path, config: &AppConfig) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    let existing_permissions = fs::metadata(path)
        .await
        .ok()
        .map(|metadata| metadata.permissions());
    let contents = serde_json::to_string_pretty(config)?;
    fs::write(path, format!("{contents}\n"))
        .await
        .with_context(|| format!("failed to write {}", path.display()))?;
    if let Some(permissions) = existing_permissions {
        fs::set_permissions(path, permissions)
            .await
            .with_context(|| format!("failed to restore permissions for {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_config_deserializes_with_defaults() {
        let config: AppConfig = serde_json::from_str(
            r#"{
              "ollama_endpoint": "http://localhost:11434",
              "default_model": "llama3.2",
              "generation": {
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 512
              },
              "instructions": {
                "enabled": false,
                "system_prompt": "",
                "tool_context": ""
              },
              "logging_enabled": true,
              "api_token": null,
              "theme": "dark"
            }"#,
        )
        .expect("legacy config should deserialize");

        assert_eq!(config.default_provider, "ollama");
        assert_eq!(config.litellm.base_url, "http://127.0.0.1:4000");
        assert!(!config.litellm.enabled);
        assert!(config.litellm_providers.is_empty());
        assert!(config.model_routes.is_empty());
        assert!(config.agents.is_empty());
    }

    #[test]
    fn litellm_config_deserializes() {
        let config: AppConfig = serde_json::from_str(
            r#"{
              "ollama_endpoint": "http://localhost:11434",
              "default_provider": "litellm",
              "default_model": "llama3.2",
              "generation": {
                "temperature": 0.4,
                "top_p": 0.8,
                "max_tokens": 1024
              },
              "instructions": {
                "enabled": true,
                "system_prompt": "You are concise.",
                "tool_context": ""
              },
              "logging_enabled": true,
              "api_token": null,
              "theme": "dark",
              "litellm": {
                "enabled": true,
                "base_url": "http://127.0.0.1:4000",
                "api_key": null,
                "default_model": "openai:gpt-4o",
                "timeout_ms": 120000,
                "managed_config_path": "litellm.yaml",
                "allow_unconfigured_models": false
              },
              "litellm_providers": [
                {
                  "id": "openai_main",
                  "enabled": true,
                  "provider_type": "openai",
                  "display_name": "OpenAI",
                  "api_key_env_var": "OPENAI_API_KEY",
                  "api_key": null,
                  "api_base": null
                }
              ],
              "model_routes": [
                {
                  "id": "route_openai_gpt4o",
                  "enabled": true,
                  "display_name": "OpenAI GPT-4o",
                  "provider": "litellm",
                  "provider_family": "openai",
                  "model_alias": "openai:gpt-4o",
                  "litellm_model": "openai/gpt-4o",
                  "api_key_env_var": "OPENAI_API_KEY",
                  "api_base": null,
                  "notes": null
                }
              ]
            }"#,
        )
        .expect("litellm config should deserialize");

        assert_eq!(config.default_provider, "litellm");
        assert_eq!(
            config.litellm.default_model.as_deref(),
            Some("openai:gpt-4o")
        );
        assert_eq!(config.model_routes[0].provider_family, "openai");
        assert_eq!(config.litellm_providers[0].provider_type, "openai");
    }

    #[test]
    fn redacted_response_masks_provider_api_keys() {
        let mut config = AppConfig::default();
        config.litellm_providers.push(LiteLlmProviderConfig {
            id: "openai_main".to_string(),
            enabled: true,
            provider_type: "openai".to_string(),
            display_name: "OpenAI".to_string(),
            api_key_env_var: "OPENAI_API_KEY".to_string(),
            api_key: Some("sk-secret".to_string()),
            api_base: None,
        });

        let redacted = config.redacted_for_response();

        assert_eq!(
            redacted.litellm_providers[0].api_key.as_deref(),
            Some(REDACTED_SECRET)
        );
    }
}
