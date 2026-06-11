use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    env,
    path::{Path, PathBuf},
};
use tokio::fs;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub ollama_endpoint: String,
    pub default_model: Option<String>,
    pub generation: GenerationSettings,
    pub instructions: InstructionSettings,
    pub logging_enabled: bool,
    pub api_token: Option<String>,
    pub theme: String,
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

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            ollama_endpoint: "http://localhost:11434".to_string(),
            default_model: None,
            generation: GenerationSettings::default(),
            instructions: InstructionSettings::default(),
            logging_enabled: true,
            api_token: None,
            theme: "dark".to_string(),
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

    let contents = serde_json::to_string_pretty(config)?;
    fs::write(path, format!("{contents}\n"))
        .await
        .with_context(|| format!("failed to write {}", path.display()))
}
