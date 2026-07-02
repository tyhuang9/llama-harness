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
    #[serde(
        default,
        alias = "appId",
        alias = "app_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub app_id: Option<String>,
    #[serde(
        default,
        alias = "agentId",
        alias = "agent_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub agent_id: Option<String>,
    #[serde(
        default,
        alias = "modelId",
        alias = "model_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resolved_tool_ids: Vec<String>,
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
pub struct AuditRecord {
    pub id: String,
    pub event: String,
    pub level: AuditLevel,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuditLevel {
    Info,
    Warn,
    Denied,
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Completed,
    RequiresAction,
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

pub async fn load_audit(path: &Path, max_records: usize) -> Result<VecDeque<AuditRecord>> {
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
                match serde_json::from_str::<AuditRecord>(line) {
                    Ok(record) => records.push_back(record),
                    Err(err) => tracing::warn!(error = %err, "skipping malformed audit log line"),
                }
            }
            Ok(records)
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(VecDeque::new()),
        Err(err) => Err(err).with_context(|| format!("failed to read {}", path.display())),
    }
}

pub fn push_recent(history: &mut VecDeque<RunRecord>, record: RunRecord, max_records: usize) {
    history.retain(|existing| existing.id != record.id);
    history.push_front(record);
    while history.len() > max_records {
        history.pop_back();
    }
}

pub fn push_recent_audit(
    history: &mut VecDeque<AuditRecord>,
    record: AuditRecord,
    max_records: usize,
) {
    history.push_front(record);
    while history.len() > max_records {
        history.pop_back();
    }
}

pub async fn append_jsonl(path: &Path, record: &RunRecord) -> Result<()> {
    append_record_jsonl(path, record).await
}

pub async fn append_audit_jsonl(path: &Path, record: &AuditRecord) -> Result<()> {
    append_record_jsonl(path, record).await
}

async fn append_record_jsonl<T>(path: &Path, record: &T) -> Result<()>
where
    T: Serialize,
{
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
