use llama_harness_core::{EventRecord, EventSink, RunEvent, RunStatus};
use llama_harness_observability::{
    AppendOutcome, RedactionConfig, RetentionPolicy, RunListQuery, SqliteEventSink,
    TraceStoreConfig, TraceStoreError, REDACTED_VALUE,
};
use serde_json::json;
use std::{
    fs,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

fn record(
    run_id: &str,
    trace_id: &str,
    sequence: u64,
    timestamp_ms: u64,
    event: RunEvent,
) -> EventRecord {
    EventRecord {
        run_id: run_id.into(),
        trace_id: trace_id.into(),
        sequence,
        timestamp_ms,
        event,
    }
}

fn started(run_id: &str, trace_id: &str, timestamp_ms: u64) -> EventRecord {
    record(
        run_id,
        trace_id,
        1,
        timestamp_ms,
        RunEvent::Started {
            run_id: run_id.into(),
            trace_id: trace_id.into(),
        },
    )
}

fn completed(
    run_id: &str,
    trace_id: &str,
    sequence: u64,
    timestamp_ms: u64,
    status: RunStatus,
) -> EventRecord {
    record(
        run_id,
        trace_id,
        sequence,
        timestamp_ms,
        RunEvent::Completed { status },
    )
}

fn temporary_database(name: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "llama-harness-{name}-{}-{stamp}.sqlite",
        std::process::id()
    ))
}

#[test]
fn migration_append_query_reopen_and_conflict_are_deterministic() {
    let path = temporary_database("migration");
    let store = SqliteEventSink::open(&path, TraceStoreConfig::default()).unwrap();
    let first = started("run-a", "trace-a", 10);
    let second = completed("run-a", "trace-a", 2, 20, RunStatus::Completed);

    assert_eq!(store.append(&first).unwrap(), AppendOutcome::Inserted);
    assert_eq!(store.append(&second).unwrap(), AppendOutcome::Inserted);
    assert_eq!(store.append(&first).unwrap(), AppendOutcome::Duplicate);
    assert!(matches!(
        store.append(&record("run-a", "trace-a", 1, 11, RunEvent::ModelResponded { call_number: 1 })),
        Err(TraceStoreError::Conflict { run_id, sequence }) if run_id == "run-a" && sequence == 1
    ));
    assert!(matches!(
        store.append(&record("run-a", "trace-other", 3, 30, RunEvent::ModelResponded { call_number: 2 })),
        Err(TraceStoreError::InvalidRecord(message)) if message.contains("already belongs")
    ));
    let events = store.events_for_run("run-a", 10, 0).unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].record.sequence, 1);
    assert_eq!(events[1].record.sequence, 2);
    drop(store);

    let reopened = SqliteEventSink::open(&path, TraceStoreConfig::default()).unwrap();
    assert_eq!(reopened.events_for_run("run-a", 10, 0).unwrap().len(), 2);
    drop(reopened);
    fs::remove_file(path).unwrap();
}

#[test]
fn raw_payloads_are_opt_in_bounded_and_redacted_before_write_and_export() {
    let raw_payload = json!({
        "authorization": "Bearer top-secret",
        "nested": {"api_key": "abc123"},
        "note": "contains top-secret"
    });
    let event = record(
        "run-redacted",
        "trace-redacted",
        1,
        10,
        RunEvent::ToolRejected {
            call_id: "call".into(),
            tool_id: "tool".into(),
            reason: "top-secret".into(),
        },
    );
    let config = TraceStoreConfig {
        redaction: RedactionConfig {
            secret_values: vec!["top-secret".into()],
            ..RedactionConfig::default()
        },
        ..TraceStoreConfig::default()
    };
    let disabled = SqliteEventSink::open_in_memory(config.clone()).unwrap();
    disabled
        .append_with_raw(&event, Some(&raw_payload))
        .unwrap();
    assert_eq!(
        disabled.events_for_run("run-redacted", 10, 0).unwrap()[0].raw_payload,
        None
    );

    let enabled = SqliteEventSink::open_in_memory(TraceStoreConfig {
        persist_raw_payloads: true,
        ..config
    })
    .unwrap();
    enabled.append_with_raw(&event, Some(&raw_payload)).unwrap();
    let persisted = enabled.events_for_run("run-redacted", 10, 0).unwrap();
    let raw = persisted[0].raw_payload.as_ref().unwrap();
    assert_eq!(raw["authorization"], REDACTED_VALUE);
    assert_eq!(raw["nested"]["api_key"], REDACTED_VALUE);
    assert_eq!(raw["note"], format!("contains {REDACTED_VALUE}"));
    assert!(matches!(
        &persisted[0].record.event,
        RunEvent::ToolRejected { reason, .. } if reason == REDACTED_VALUE
    ));
    let export = enabled.export_run_json("run-redacted").unwrap().unwrap();
    assert!(!export.contains("top-secret"));
    assert!(!export.contains("abc123"));

    let limited = SqliteEventSink::open_in_memory(TraceStoreConfig {
        persist_raw_payloads: true,
        max_raw_payload_bytes: 8,
        ..TraceStoreConfig::default()
    })
    .unwrap();
    assert!(matches!(
        limited.append_with_raw(&event, Some(&raw_payload)),
        Err(TraceStoreError::ResourceLimit(message)) if message.contains("raw payload")
    ));
}

#[test]
fn batch_writes_are_transactional_and_run_queries_filter_paginate_export_and_retain() {
    let store = SqliteEventSink::open_in_memory(TraceStoreConfig::default()).unwrap();
    let inserted = store
        .append_batch(vec![
            (started("run-old", "trace-a", 10), None),
            (
                completed("run-old", "trace-a", 2, 20, RunStatus::Failed),
                None,
            ),
            (started("run-new", "trace-b", 30), None),
            (
                completed("run-new", "trace-b", 2, 40, RunStatus::Completed),
                None,
            ),
        ])
        .unwrap();
    assert_eq!(inserted, vec![AppendOutcome::Inserted; 4]);
    assert!(matches!(
        store.append_batch(vec![
            (started("run-atomic", "trace-c", 1), None),
            (
                record(
                    "run-atomic",
                    "trace-c",
                    1,
                    2,
                    RunEvent::ModelResponded { call_number: 1 }
                ),
                None
            ),
        ]),
        Err(TraceStoreError::Conflict { .. })
    ));
    assert!(store
        .events_for_run("run-atomic", 10, 0)
        .unwrap()
        .is_empty());

    let completed_runs = store
        .list_runs(RunListQuery {
            status: Some(RunStatus::Completed),
            limit: 10,
            ..RunListQuery::default()
        })
        .unwrap();
    assert_eq!(completed_runs.len(), 1);
    assert_eq!(completed_runs[0].run_id, "run-new");
    let trace_a = store
        .list_runs(RunListQuery {
            trace_id: Some("trace-a".into()),
            started_before_ms: Some(25),
            limit: 1,
            ..RunListQuery::default()
        })
        .unwrap();
    assert_eq!(trace_a[0].run_id, "run-old");
    assert_eq!(
        store
            .list_runs(RunListQuery {
                limit: 10,
                offset: 1,
                ..RunListQuery::default()
            })
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        store.export_run("run-new").unwrap().unwrap().events.len(),
        2
    );

    let retention = store
        .apply_retention(
            &RetentionPolicy {
                max_age_ms: None,
                max_runs: Some(1),
            },
            50,
        )
        .unwrap();
    assert_eq!(retention.runs_deleted, 1);
    assert!(store.events_for_run("run-old", 10, 0).unwrap().is_empty());
    assert_eq!(store.delete_run("run-new").unwrap(), 2);
}

#[test]
fn sink_supports_concurrent_event_emission_and_exposes_write_errors() {
    let sink = Arc::new(SqliteEventSink::open_in_memory(TraceStoreConfig::default()).unwrap());
    std::thread::scope(|scope| {
        for index in 0..12_u64 {
            let sink = Arc::clone(&sink);
            scope.spawn(move || {
                sink.emit(started(
                    &format!("run-{index}"),
                    "trace-concurrent",
                    index + 1,
                ));
            });
        }
    });
    for index in 0..12_u64 {
        assert_eq!(
            sink.events_for_run(&format!("run-{index}"), 10, 0)
                .unwrap()
                .len(),
            1
        );
    }
    assert_eq!(sink.last_emit_error(), None);

    sink.emit(record(
        "",
        "trace",
        1,
        1,
        RunEvent::ModelResponded { call_number: 1 },
    ));
    assert!(sink.last_emit_error().unwrap().contains("run ID"));
}

#[test]
fn export_refuses_a_trace_that_cannot_be_complete_within_the_export_bound() {
    let store = SqliteEventSink::open_in_memory(TraceStoreConfig::default()).unwrap();
    store
        .append_batch((1..=1_001).map(|sequence| {
            (
                record(
                    "run-large",
                    "trace-large",
                    sequence,
                    sequence,
                    RunEvent::ModelResponded {
                        call_number: sequence as u32,
                    },
                ),
                None,
            )
        }))
        .unwrap();
    assert!(matches!(
        store.export_run("run-large"),
        Err(TraceStoreError::ResourceLimit(message)) if message.contains("limited to 1000")
    ));
}
