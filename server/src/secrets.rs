use anyhow::{Context, Result};
use std::{
    collections::{HashMap, HashSet},
    env,
    path::{Path, PathBuf},
};
use tokio::fs;

pub const LITELLM_MASTER_KEY_ENV: &str = "LITELLM_MASTER_KEY";

#[derive(Clone, Debug)]
enum EnvLine {
    Entry { key: String, value: String },
    Raw(String),
}

pub fn app_data_dir() -> PathBuf {
    if let Ok(path) = env::var("LLAMA_HARNESS_DATA_DIR") {
        let path = path.trim();
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }

    os_app_data_dir()
}

pub fn env_file_path() -> PathBuf {
    app_data_dir().join(".env")
}

pub async fn load_env_file(path: &Path) -> Result<HashMap<String, String>> {
    let lines = read_env_lines(path).await?;
    Ok(lines
        .into_iter()
        .filter_map(|line| match line {
            EnvLine::Entry { key, value } => Some((key, value)),
            EnvLine::Raw(_) => None,
        })
        .collect())
}

pub async fn write_env_updates(
    path: &Path,
    updates: &HashMap<String, Option<String>>,
) -> Result<()> {
    let mut lines = read_env_lines(path).await?;
    let mut seen = HashSet::new();

    for line in &mut lines {
        if let EnvLine::Entry { key, value } = line {
            if let Some(update) = updates.get(key) {
                seen.insert(key.clone());
                match update {
                    Some(next_value) => *value = next_value.clone(),
                    None => {
                        *line = EnvLine::Raw(String::new());
                    }
                }
            }
        }
    }

    for (key, value) in updates {
        if seen.contains(key) || value.is_none() {
            continue;
        }
        lines.push(EnvLine::Entry {
            key: key.clone(),
            value: value.clone().unwrap_or_default(),
        });
    }

    write_env_lines(path, &lines).await
}

pub fn env_value_configured(vars: &HashMap<String, String>, key: &str) -> bool {
    vars.get(key)
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}

async fn read_env_lines(path: &Path) -> Result<Vec<EnvLine>> {
    let contents = match fs::read_to_string(path).await {
        Ok(contents) => contents,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err).with_context(|| format!("failed to read {}", path.display())),
    };

    Ok(contents.lines().map(parse_env_line).collect())
}

async fn write_env_lines(path: &Path, lines: &[EnvLine]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    let mut contents = String::new();
    for line in lines {
        match line {
            EnvLine::Entry { key, value } => {
                contents.push_str(key);
                contents.push('=');
                contents.push_str(&quote_env_value(value));
                contents.push('\n');
            }
            EnvLine::Raw(value) if !value.is_empty() => {
                contents.push_str(value);
                contents.push('\n');
            }
            EnvLine::Raw(_) => {}
        }
    }

    fs::write(path, contents)
        .await
        .with_context(|| format!("failed to write {}", path.display()))?;
    set_private_permissions(path).await
}

fn parse_env_line(line: &str) -> EnvLine {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return EnvLine::Raw(line.to_string());
    }

    let Some((raw_key, raw_value)) = line.split_once('=') else {
        return EnvLine::Raw(line.to_string());
    };
    let key = raw_key
        .trim()
        .strip_prefix("export ")
        .unwrap_or(raw_key.trim())
        .trim()
        .to_string();
    if key.is_empty()
        || !key
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return EnvLine::Raw(line.to_string());
    }

    EnvLine::Entry {
        key,
        value: unquote_env_value(raw_value.trim()),
    }
}

fn quote_env_value(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    format!("\"{escaped}\"")
}

fn unquote_env_value(value: &str) -> String {
    if value.len() >= 2 && value.starts_with('"') && value.ends_with('"') {
        return value[1..value.len() - 1]
            .replace("\\n", "\n")
            .replace("\\\"", "\"")
            .replace("\\\\", "\\");
    }
    if value.len() >= 2 && value.starts_with('\'') && value.ends_with('\'') {
        return value[1..value.len() - 1].to_string();
    }
    value.to_string()
}

#[cfg(unix)]
async fn set_private_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let permissions = std::fs::Permissions::from_mode(0o600);
    fs::set_permissions(path, permissions)
        .await
        .with_context(|| format!("failed to set private permissions on {}", path.display()))
}

#[cfg(not(unix))]
async fn set_private_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(target_os = "macos")]
fn os_app_data_dir() -> PathBuf {
    home_dir()
        .map(|home| {
            home.join("Library")
                .join("Application Support")
                .join("com.tyhuang.llama-harness")
        })
        .unwrap_or_else(|| PathBuf::from(".llama-harness"))
}

#[cfg(target_os = "windows")]
fn os_app_data_dir() -> PathBuf {
    env::var("APPDATA")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("AppData")
                .join("Roaming")
        })
        .join("com.tyhuang.llama-harness")
}

#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
fn os_app_data_dir() -> PathBuf {
    env::var("XDG_DATA_HOME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".local")
                .join("share")
        })
        .join("llama-harness")
}

fn home_dir() -> Option<PathBuf> {
    env::var("HOME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_quotes_env_values() {
        assert_eq!(
            unquote_env_value("\"sk-test\\\"value\""),
            "sk-test\"value".to_string()
        );
        assert_eq!(quote_env_value("line\nvalue"), "\"line\\nvalue\"");
    }

    #[tokio::test]
    async fn preserves_unrelated_env_keys() {
        let path = std::env::temp_dir().join(format!(
            "llama-harness-env-test-{}.env",
            uuid::Uuid::new_v4()
        ));
        fs::write(&path, "KEEP_ME=1\nOPENAI_API_KEY=old\n")
            .await
            .unwrap();

        let mut updates = HashMap::new();
        updates.insert("OPENAI_API_KEY".to_string(), Some("new".to_string()));
        updates.insert("REMOVE_ME".to_string(), None);
        write_env_updates(&path, &updates).await.unwrap();

        let vars = load_env_file(&path).await.unwrap();
        assert_eq!(vars.get("KEEP_ME").map(String::as_str), Some("1"));
        assert_eq!(vars.get("OPENAI_API_KEY").map(String::as_str), Some("new"));
        assert!(!vars.contains_key("REMOVE_ME"));

        let _ = fs::remove_file(path).await;
    }

    #[test]
    fn env_file_path_uses_data_dir_override() {
        let previous = std::env::var("LLAMA_HARNESS_DATA_DIR").ok();
        let dir = std::env::temp_dir().join(format!(
            "llama-harness-data-dir-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::env::set_var("LLAMA_HARNESS_DATA_DIR", &dir);

        assert_eq!(env_file_path(), dir.join(".env"));

        if let Some(previous) = previous {
            std::env::set_var("LLAMA_HARNESS_DATA_DIR", previous);
        } else {
            std::env::remove_var("LLAMA_HARNESS_DATA_DIR");
        }
    }
}
