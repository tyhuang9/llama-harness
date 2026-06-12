use crate::providers::TokenUsage;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, path::Path};
use tokio::{
    fs::{self, OpenOptions},
    io::AsyncWriteExt,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunRecord {
    pub id: String,
    #[serde(default = "default_provider")]
    pub provider: String,
    pub model: String,
    pub source_app: Option<String>,
    pub prompt_summary: String,
    pub response_summary: Option<String>,
    pub status: RunStatus,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    pub duration_ms: u64,
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Completed,
    Failed,
}

fn default_provider() -> String {
    "ollama".to_string()
}

pub async fn load_runs(path: &Path, max_records: usize) -> Result<VecDeque<RunRecord>> {
    match fs::read_to_string(path).await {
        Ok(contents) => {
            let mut records = VecDeque::new();
            for line in contents
                .lines()
                .rev()
                .filter(|line| !line.trim().is_empty())
            {
                if records.len() >= max_records {
                    break;
                }
                match serde_json::from_str::<RunRecord>(line) {
                    Ok(record) => records.push_back(record),
                    Err(err) => tracing::warn!(error = %err, "skipping malformed run log line"),
                }
            }
            Ok(records)
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(VecDeque::new()),
        Err(err) => Err(err).with_context(|| format!("failed to read {}", path.display())),
    }
}

pub fn push_recent(history: &mut VecDeque<RunRecord>, record: RunRecord, max_records: usize) {
    history.push_front(record);
    while history.len() > max_records {
        history.pop_back();
    }
}

pub async fn append_jsonl(path: &Path, record: &RunRecord) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    let line = serde_json::to_string(record)?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
        .with_context(|| format!("failed to open {}", path.display()))?;

    file.write_all(line.as_bytes()).await?;
    file.write_all(b"\n").await?;
    Ok(())
}
