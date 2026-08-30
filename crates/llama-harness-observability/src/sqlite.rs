use crate::RedactionConfig;
use llama_harness_core::{EventRecord, EventSink, RunEvent, RunStatus};
use rusqlite::{
    params,
    types::{ToSql, Value as SqlValue},
    Connection, OpenFlags, OptionalExtension, Transaction,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};
use thiserror::Error;

const CURRENT_SCHEMA_VERSION: i64 = 1;
const DEFAULT_MAX_EVENT_BYTES: usize = 64 * 1024;
const DEFAULT_MAX_RAW_PAYLOAD_BYTES: usize = 64 * 1024;
const MAX_QUERY_LIMIT: u32 = 1_000;

#[derive(Clone, Debug, PartialEq, Eq)]
/// Configuration for a local SQLite trace store.
pub struct TraceStoreConfig {
    /// Raw payloads are disabled by default. Structured causal events are always persisted.
    pub persist_raw_payloads: bool,
    /// Maximum serialized size of a structured event in bytes.
    pub max_event_bytes: usize,
    /// Maximum serialized size of an optional raw payload in bytes.
    pub max_raw_payload_bytes: usize,
    /// SQLite busy timeout used while opening and writing the database.
    pub busy_timeout: Duration,
    /// Redaction rules applied before persistence.
    pub redaction: RedactionConfig,
}

impl Default for TraceStoreConfig {
    fn default() -> Self {
        Self {
            persist_raw_payloads: false,
            max_event_bytes: DEFAULT_MAX_EVENT_BYTES,
            max_raw_payload_bytes: DEFAULT_MAX_RAW_PAYLOAD_BYTES,
            busy_timeout: Duration::from_secs(5),
            redaction: RedactionConfig::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
/// Result of appending an event to a trace store.
pub enum AppendOutcome {
    /// The event was inserted into the store.
    Inserted,
    /// An identical event was already present.
    Duplicate,
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
/// A stored event and its optional raw payload.
pub struct PersistedEvent {
    /// The canonical event record.
    pub record: EventRecord,
    /// The optional redacted raw payload stored with the event.
    pub raw_payload: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
/// Aggregate information for one persisted run.
pub struct RunSummary {
    /// Application-visible run identifier.
    pub run_id: String,
    /// Trace identifier shared by the run's events.
    pub trace_id: String,
    /// Timestamp of the earliest event in milliseconds since the Unix epoch.
    pub started_at_ms: u64,
    /// Timestamp of the latest event in milliseconds since the Unix epoch.
    pub updated_at_ms: u64,
    /// Number of persisted events for the run.
    pub event_count: u64,
    /// Terminal status, when a completion event has been persisted.
    pub status: Option<RunStatus>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
/// Filters and pagination for listing persisted runs.
pub struct RunListQuery {
    /// Optional trace identifier filter.
    pub trace_id: Option<String>,
    /// Optional terminal status filter.
    pub status: Option<RunStatus>,
    /// Optional inclusive lower bound for event timestamps.
    pub started_after_ms: Option<u64>,
    /// Optional inclusive upper bound for event timestamps.
    pub started_before_ms: Option<u64>,
    /// Maximum number of summaries to return; zero selects the default limit.
    pub limit: u32,
    /// Number of summaries to skip before returning results.
    pub offset: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
/// Age and count limits for deleting old trace data.
pub struct RetentionPolicy {
    /// Delete events older than this age relative to the supplied current time.
    pub max_age_ms: Option<u64>,
    /// Keep at most this many most-recent runs.
    pub max_runs: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
/// Counts returned by a retention operation.
pub struct RetentionResult {
    /// Number of events deleted by the retention operation.
    pub events_deleted: u64,
    /// Number of runs deleted by the retention operation.
    pub runs_deleted: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
/// Serializable export of one run and its events.
pub struct ExportedRun {
    /// Application-visible run identifier.
    pub run_id: String,
    /// Trace identifier shared by the exported events.
    pub trace_id: String,
    /// Events included in the export, ordered by sequence.
    pub events: Vec<PersistedEventExport>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
/// Serializable representation of one exported event.
pub struct PersistedEventExport {
    /// The canonical event record.
    pub record: EventRecord,
    /// The optional redacted raw payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_payload: Option<Value>,
}

#[derive(Debug, Error)]
#[non_exhaustive]
/// Errors returned by trace-store operations.
pub enum TraceStoreError {
    #[error("SQLite error: {0}")]
    /// The SQLite operation failed.
    Sqlite(#[from] rusqlite::Error),
    #[error("serialization error: {0}")]
    /// A record or payload could not be serialized or decoded.
    Serialization(#[from] serde_json::Error),
    #[error("invalid trace store configuration: {0}")]
    /// The store configuration or query parameters are invalid.
    InvalidConfiguration(String),
    #[error("trace record is invalid: {0}")]
    /// A record does not satisfy the store's invariants.
    InvalidRecord(String),
    #[error("trace payload exceeds configured limit: {0}")]
    /// A serialized event or payload exceeds its configured size limit.
    ResourceLimit(String),
    #[error("conflicting event for run {run_id} sequence {sequence}")]
    /// An existing sequence contains different event data.
    Conflict {
        /// Run containing the conflicting event.
        run_id: String,
        /// Sequence number occupied by different event data.
        sequence: u64,
    },
    #[error("trace store mutex is poisoned")]
    /// A store lock was poisoned by a failed thread.
    Poisoned,
}

#[derive(Clone)]
/// Thread-safe SQLite-backed event sink and trace query interface.
pub struct SqliteEventSink {
    inner: Arc<StoreInner>,
}

struct StoreInner {
    connection: Mutex<Connection>,
    config: TraceStoreConfig,
    last_emit_error: Mutex<Option<String>>,
}

impl SqliteEventSink {
    /// Opens or creates a trace database and applies the current schema.
    pub fn open(path: impl AsRef<Path>, config: TraceStoreConfig) -> Result<Self, TraceStoreError> {
        validate_config(&config)?;
        let mut connection = Connection::open(path)?;
        configure_and_migrate(&mut connection, &config)?;
        Ok(Self::from_connection(connection, config))
    }

    /// Opens an existing trace database for inspection without changing its schema or
    /// SQLite journal settings. This is intended for local developer tooling that
    /// must never create a database or mutate a project's trace store while reading it.
    /// Opens an existing trace database without creating or migrating it.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self, TraceStoreError> {
        let config = TraceStoreConfig::default();
        let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        connection.busy_timeout(config.busy_timeout)?;
        Ok(Self::from_connection(connection, config))
    }

    /// Opens an in-memory trace database using the supplied configuration.
    pub fn open_in_memory(config: TraceStoreConfig) -> Result<Self, TraceStoreError> {
        validate_config(&config)?;
        let mut connection = Connection::open_in_memory()?;
        configure_and_migrate(&mut connection, &config)?;
        Ok(Self::from_connection(connection, config))
    }

    fn from_connection(connection: Connection, config: TraceStoreConfig) -> Self {
        Self {
            inner: Arc::new(StoreInner {
                connection: Mutex::new(connection),
                config,
                last_emit_error: Mutex::new(None),
            }),
        }
    }

    /// Appends one redacted structured event. Use [`Self::append_with_raw`] only when
    /// opt-in raw persistence is explicitly enabled in the store config.
    pub fn append(&self, record: &EventRecord) -> Result<AppendOutcome, TraceStoreError> {
        self.append_with_raw(record, None)
    }

    /// Appends one event and optional raw payload in a single transaction.
    pub fn append_with_raw(
        &self,
        record: &EventRecord,
        raw_payload: Option<&Value>,
    ) -> Result<AppendOutcome, TraceStoreError> {
        let prepared = self.prepare(record, raw_payload)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction()?;
        let outcome = append_prepared(&transaction, &prepared)?;
        transaction.commit()?;
        Ok(outcome)
    }

    /// Appends a group of events in one transaction. Existing identical events remain
    /// idempotent; conflicting `(run_id, sequence)` records roll the transaction back.
    pub fn append_batch(
        &self,
        events: impl IntoIterator<Item = (EventRecord, Option<Value>)>,
    ) -> Result<Vec<AppendOutcome>, TraceStoreError> {
        let prepared = events
            .into_iter()
            .map(|(record, raw)| self.prepare(&record, raw.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction()?;
        let outcomes = prepared
            .iter()
            .map(|event| append_prepared(&transaction, event))
            .collect::<Result<Vec<_>, _>>()?;
        transaction.commit()?;
        Ok(outcomes)
    }

    /// Returns persisted events for a run in sequence order.
    pub fn events_for_run(
        &self,
        run_id: &str,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<PersistedEvent>, TraceStoreError> {
        let limit = checked_limit(limit)?;
        let connection = self.connection()?;
        let mut statement = connection.prepare(
            "SELECT event_json, raw_payload_json
             FROM trace_events
             WHERE run_id = ?1
             ORDER BY sequence ASC
             LIMIT ?2 OFFSET ?3",
        )?;
        let rows = statement.query_map(
            params![run_id, i64::from(limit), i64::from(offset)],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
        )?;
        rows.map(|row| decode_persisted_event(row?)).collect()
    }

    /// Lists run summaries matching the supplied filters.
    pub fn list_runs(&self, query: RunListQuery) -> Result<Vec<RunSummary>, TraceStoreError> {
        let limit = checked_limit(query.limit)?;
        let mut sql = String::from(
            "SELECT run_id, trace_id, MIN(timestamp_ms), MAX(timestamp_ms), COUNT(*), MAX(status)
             FROM trace_events WHERE 1 = 1",
        );
        let mut values: Vec<SqlValue> = Vec::new();
        if let Some(trace_id) = query.trace_id {
            sql.push_str(" AND trace_id = ?");
            values.push(SqlValue::Text(trace_id));
        }
        if let Some(after) = query.started_after_ms {
            sql.push_str(" AND timestamp_ms >= ?");
            values.push(SqlValue::Integer(to_sql_integer(after)?));
        }
        if let Some(before) = query.started_before_ms {
            sql.push_str(" AND timestamp_ms <= ?");
            values.push(SqlValue::Integer(to_sql_integer(before)?));
        }
        sql.push_str(" GROUP BY run_id, trace_id");
        if let Some(status) = query.status {
            sql.push_str(" HAVING MAX(status) = ?");
            values.push(SqlValue::Text(status_text(&status).into()));
        }
        sql.push_str(" ORDER BY MAX(timestamp_ms) DESC, run_id DESC LIMIT ? OFFSET ?");
        values.push(SqlValue::Integer(i64::from(limit)));
        values.push(SqlValue::Integer(i64::from(query.offset)));

        let connection = self.connection()?;
        let mut statement = connection.prepare(&sql)?;
        let rows = statement.query_map(
            rusqlite::params_from_iter(values.iter().map(|value| value as &dyn ToSql)),
            |row| {
                let status = row
                    .get::<_, Option<String>>(5)?
                    .map(|status| parse_status(&status))
                    .transpose()
                    .map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            5,
                            rusqlite::types::Type::Text,
                            Box::new(error),
                        )
                    })?;
                Ok(RunSummary {
                    run_id: row.get(0)?,
                    trace_id: row.get(1)?,
                    started_at_ms: from_sql_integer(row.get(2)?)?,
                    updated_at_ms: from_sql_integer(row.get(3)?)?,
                    event_count: from_sql_integer(row.get(4)?)?,
                    status,
                })
            },
        )?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(TraceStoreError::from)
    }

    /// Exports all events for a run, or `None` when the run is unknown.
    pub fn export_run(&self, run_id: &str) -> Result<Option<ExportedRun>, TraceStoreError> {
        let event_count = self.event_count_for_run(run_id)?;
        if event_count == 0 {
            return Ok(None);
        }
        if event_count > u64::from(MAX_QUERY_LIMIT) {
            return Err(TraceStoreError::ResourceLimit(format!(
                "run {run_id} has {event_count} events; JSON export is limited to {MAX_QUERY_LIMIT} events"
            )));
        }
        let events = self.events_for_run(run_id, MAX_QUERY_LIMIT, 0)?;
        let Some(first) = events.first() else {
            return Err(TraceStoreError::InvalidRecord(format!(
                "run {run_id} disappeared during export"
            )));
        };
        Ok(Some(ExportedRun {
            run_id: run_id.into(),
            trace_id: first.record.trace_id.clone(),
            events: events
                .into_iter()
                .map(|event| PersistedEventExport {
                    record: event.record,
                    raw_payload: event.raw_payload,
                })
                .collect(),
        }))
    }

    /// Serializes a run export as pretty-printed JSON.
    pub fn export_run_json(&self, run_id: &str) -> Result<Option<String>, TraceStoreError> {
        self.export_run(run_id)?
            .map(|export| serde_json::to_string_pretty(&export).map_err(TraceStoreError::from))
            .transpose()
    }

    /// Deletes events according to the supplied age and run-count policy.
    pub fn apply_retention(
        &self,
        policy: &RetentionPolicy,
        now_ms: u64,
    ) -> Result<RetentionResult, TraceStoreError> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction()?;
        let mut result = RetentionResult::default();
        if let Some(max_age_ms) = policy.max_age_ms {
            let cutoff = now_ms.saturating_sub(max_age_ms);
            result.events_deleted += transaction.execute(
                "DELETE FROM trace_events WHERE timestamp_ms < ?1",
                params![to_sql_integer(cutoff)?],
            )? as u64;
        }
        if let Some(max_runs) = policy.max_runs {
            let run_ids = {
                let mut statement = transaction.prepare(
                    "SELECT run_id
                     FROM trace_events
                     GROUP BY run_id
                     ORDER BY MAX(timestamp_ms) DESC, run_id DESC
                     LIMIT -1 OFFSET ?1",
                )?;
                let run_ids = statement
                    .query_map(params![i64::from(max_runs)], |row| row.get::<_, String>(0))?
                    .collect::<Result<Vec<_>, _>>()?;
                run_ids
            };
            for run_id in run_ids {
                result.events_deleted += transaction.execute(
                    "DELETE FROM trace_events WHERE run_id = ?1",
                    params![run_id],
                )? as u64;
                result.runs_deleted += 1;
            }
        }
        transaction.commit()?;
        Ok(result)
    }

    /// Deletes all events belonging to a run and returns the number removed.
    pub fn delete_run(&self, run_id: &str) -> Result<u64, TraceStoreError> {
        let connection = self.connection()?;
        Ok(connection.execute(
            "DELETE FROM trace_events WHERE run_id = ?1",
            params![run_id],
        )? as u64)
    }

    fn event_count_for_run(&self, run_id: &str) -> Result<u64, TraceStoreError> {
        let connection = self.connection()?;
        let count = connection.query_row(
            "SELECT COUNT(*) FROM trace_events WHERE run_id = ?1",
            params![run_id],
            |row| row.get::<_, i64>(0),
        )?;
        from_sql_integer(count).map_err(TraceStoreError::from)
    }

    /// Reclaims unused SQLite pages after deletions.
    pub fn compact(&self) -> Result<(), TraceStoreError> {
        let connection = self.connection()?;
        connection.execute_batch("VACUUM")?;
        Ok(())
    }

    /// Returns the most recent `EventSink::emit` persistence error, if one occurred.
    /// Use `append` when the caller must receive write failures synchronously.
    pub fn last_emit_error(&self) -> Option<String> {
        self.inner
            .last_emit_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn prepare(
        &self,
        record: &EventRecord,
        raw_payload: Option<&Value>,
    ) -> Result<PreparedEvent, TraceStoreError> {
        if record.run_id.trim().is_empty()
            || record.trace_id.trim().is_empty()
            || record.sequence == 0
        {
            return Err(TraceStoreError::InvalidRecord(
                "run ID, trace ID, and nonzero sequence are required".into(),
            ));
        }
        let event_value = self
            .inner
            .config
            .redaction
            .redact(&serde_json::to_value(record)?);
        let event_json = serde_json::to_string(&event_value)?;
        ensure_bytes("event", event_json.len(), self.inner.config.max_event_bytes)?;
        let raw_payload_json = if self.inner.config.persist_raw_payloads {
            raw_payload
                .map(|payload| {
                    let payload = self.inner.config.redaction.redact(payload);
                    let serialized = serde_json::to_string(&payload)?;
                    ensure_bytes(
                        "raw payload",
                        serialized.len(),
                        self.inner.config.max_raw_payload_bytes,
                    )?;
                    Ok::<String, TraceStoreError>(serialized)
                })
                .transpose()?
        } else {
            None
        };
        Ok(PreparedEvent {
            record: record.clone(),
            event_kind: event_kind(&record.event).into(),
            status: completed_status(&record.event)
                .map(status_text)
                .map(str::to_owned),
            event_json,
            raw_payload_json,
        })
    }

    fn connection(&self) -> Result<std::sync::MutexGuard<'_, Connection>, TraceStoreError> {
        self.inner
            .connection
            .lock()
            .map_err(|_| TraceStoreError::Poisoned)
    }
}

impl EventSink for SqliteEventSink {
    fn emit(&self, record: EventRecord) {
        if let Err(error) = self.append(&record) {
            let mut last_error = self
                .inner
                .last_emit_error
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *last_error = Some(error.to_string());
        }
    }
}

struct PreparedEvent {
    record: EventRecord,
    event_kind: String,
    status: Option<String>,
    event_json: String,
    raw_payload_json: Option<String>,
}

fn append_prepared(
    transaction: &Transaction<'_>,
    event: &PreparedEvent,
) -> Result<AppendOutcome, TraceStoreError> {
    let existing_trace_id = transaction
        .query_row(
            "SELECT trace_id FROM trace_events WHERE run_id = ?1 LIMIT 1",
            params![event.record.run_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if let Some(existing_trace_id) = existing_trace_id {
        if existing_trace_id != event.record.trace_id {
            return Err(TraceStoreError::InvalidRecord(format!(
                "run {} already belongs to trace {existing_trace_id}",
                event.record.run_id
            )));
        }
    }
    let inserted = transaction.execute(
        "INSERT OR IGNORE INTO trace_events
         (run_id, trace_id, sequence, timestamp_ms, event_kind, status, event_json, raw_payload_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            event.record.run_id,
            event.record.trace_id,
            to_sql_integer(event.record.sequence)?,
            to_sql_integer(event.record.timestamp_ms)?,
            event.event_kind,
            event.status,
            event.event_json,
            event.raw_payload_json,
        ],
    )?;
    if inserted == 1 {
        return Ok(AppendOutcome::Inserted);
    }
    let existing = transaction
        .query_row(
            "SELECT event_json, raw_payload_json FROM trace_events WHERE run_id = ?1 AND sequence = ?2",
            params![event.record.run_id, to_sql_integer(event.record.sequence)?],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
        )
        .optional()?;
    match existing {
        Some((event_json, raw_payload_json))
            if event_json == event.event_json && raw_payload_json == event.raw_payload_json =>
        {
            Ok(AppendOutcome::Duplicate)
        }
        _ => Err(TraceStoreError::Conflict {
            run_id: event.record.run_id.clone(),
            sequence: event.record.sequence,
        }),
    }
}

fn configure_and_migrate(
    connection: &mut Connection,
    config: &TraceStoreConfig,
) -> Result<(), TraceStoreError> {
    connection.busy_timeout(config.busy_timeout)?;
    connection.pragma_update(None, "journal_mode", "WAL")?;
    connection.pragma_update(None, "foreign_keys", "ON")?;
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY NOT NULL);
         CREATE TABLE IF NOT EXISTS trace_events (
             run_id TEXT NOT NULL,
             trace_id TEXT NOT NULL,
             sequence INTEGER NOT NULL,
             timestamp_ms INTEGER NOT NULL,
             event_kind TEXT NOT NULL,
             status TEXT,
             event_json TEXT NOT NULL,
             raw_payload_json TEXT,
             PRIMARY KEY (run_id, sequence)
         );
         CREATE INDEX IF NOT EXISTS trace_events_trace_timestamp_idx
             ON trace_events(trace_id, timestamp_ms DESC);
         CREATE INDEX IF NOT EXISTS trace_events_timestamp_idx
             ON trace_events(timestamp_ms DESC);
         CREATE INDEX IF NOT EXISTS trace_events_status_idx
             ON trace_events(status);",
    )?;
    let version = connection.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
        [],
        |row| row.get::<_, i64>(0),
    )?;
    if version > CURRENT_SCHEMA_VERSION {
        return Err(TraceStoreError::InvalidConfiguration(format!(
            "database schema version {version} is newer than supported version {CURRENT_SCHEMA_VERSION}"
        )));
    }
    if version < CURRENT_SCHEMA_VERSION {
        connection.execute(
            "INSERT INTO schema_migrations(version) VALUES (?1)",
            params![CURRENT_SCHEMA_VERSION],
        )?;
    }
    Ok(())
}

fn validate_config(config: &TraceStoreConfig) -> Result<(), TraceStoreError> {
    if config.max_event_bytes == 0 || config.max_raw_payload_bytes == 0 {
        return Err(TraceStoreError::InvalidConfiguration(
            "trace payload limits must be greater than zero".into(),
        ));
    }
    if config.busy_timeout.is_zero() {
        return Err(TraceStoreError::InvalidConfiguration(
            "SQLite busy timeout must be greater than zero".into(),
        ));
    }
    Ok(())
}

fn decode_persisted_event(
    (event_json, raw_payload_json): (String, Option<String>),
) -> Result<PersistedEvent, TraceStoreError> {
    Ok(PersistedEvent {
        record: serde_json::from_str(&event_json)?,
        raw_payload: raw_payload_json
            .map(|payload| serde_json::from_str(&payload))
            .transpose()?,
    })
}

fn checked_limit(limit: u32) -> Result<u32, TraceStoreError> {
    let limit = if limit == 0 { 100 } else { limit };
    if limit > MAX_QUERY_LIMIT {
        return Err(TraceStoreError::InvalidConfiguration(format!(
            "query limit cannot exceed {MAX_QUERY_LIMIT}"
        )));
    }
    Ok(limit)
}

fn ensure_bytes(label: &str, bytes: usize, max_bytes: usize) -> Result<(), TraceStoreError> {
    if bytes > max_bytes {
        return Err(TraceStoreError::ResourceLimit(format!(
            "{label} is {bytes} bytes, above the {max_bytes}-byte limit"
        )));
    }
    Ok(())
}

fn event_kind(event: &RunEvent) -> &'static str {
    match event {
        RunEvent::Started { .. } => "run.started",
        RunEvent::ModelRequested { .. } => "model.requested",
        RunEvent::ModelRetrying { .. } => "model.retrying",
        RunEvent::ModelResponded { .. } => "model.responded",
        RunEvent::StrategySelected { .. } => "strategy.selected",
        RunEvent::StrategyFallback { .. } => "strategy.fallback",
        RunEvent::PlanLifecycle { .. } => "plan.lifecycle",
        RunEvent::PlanValidated { .. } => "plan.validated",
        RunEvent::PlanNodeStarted { .. } => "plan.node.started",
        RunEvent::PlanNodeCompleted { .. } => "plan.node.completed",
        RunEvent::ToolEffectReused { .. } => "tool.effect_reused",
        RunEvent::StrategyUsage { .. } => "strategy.usage",
        RunEvent::ToolRejected { .. } => "tool.rejected",
        RunEvent::PolicyDecided { .. } => "policy.decided",
        RunEvent::ApprovalRequested { .. } => "approval.requested",
        RunEvent::ToolCompleted { .. } => "tool.completed",
        RunEvent::Completed { .. } => "run.completed",
        _ => "run.unknown",
    }
}

fn completed_status(event: &RunEvent) -> Option<&RunStatus> {
    match event {
        RunEvent::Completed { status } => Some(status),
        _ => None,
    }
}

fn status_text(status: &RunStatus) -> &'static str {
    match status {
        RunStatus::Completed => "completed",
        RunStatus::Failed => "failed",
        RunStatus::Cancelled => "cancelled",
        RunStatus::LimitReached => "limit_reached",
        _ => "unknown",
    }
}

fn parse_status(value: &str) -> Result<RunStatus, TraceStoreError> {
    match value {
        "completed" => Ok(RunStatus::Completed),
        "failed" => Ok(RunStatus::Failed),
        "cancelled" => Ok(RunStatus::Cancelled),
        "limit_reached" => Ok(RunStatus::LimitReached),
        status => Err(TraceStoreError::InvalidRecord(format!(
            "unknown stored run status: {status}"
        ))),
    }
}

fn to_sql_integer(value: u64) -> Result<i64, TraceStoreError> {
    i64::try_from(value).map_err(|_| {
        TraceStoreError::InvalidRecord(format!("integer value {value} exceeds SQLite range"))
    })
}

fn from_sql_integer(value: i64) -> rusqlite::Result<u64> {
    u64::try_from(value).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Integer,
            Box::new(error),
        )
    })
}
