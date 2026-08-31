use llama_harness_core::{
    mock::{final_response, MockModelProvider},
    AgentDefinition, AgentRunner, EventRecord, EventSink, ProgramLifecycleOutcome, RunEvent,
    RunRequest, RunStatus, RunStrategy, ToolCaller, ToolDiscoveryOutcome, ToolDiscoverySelection,
};
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
    let mut record = EventRecord::new(run_id, trace_id, sequence, timestamp_ms, event);
    // Test records model one EventEmitter execution unless a test deliberately
    // overrides this identity to model reuse of a public run ID.
    record.execution_id = format!("test-execution:{run_id}:{trace_id}");
    record
}

#[test]
fn runner_discovery_events_reopen_with_additive_legacy_compatibility() {
    let path = temporary_database("runner-discovery");
    let store = SqliteEventSink::open(&path, TraceStoreConfig::default()).unwrap();
    let provider = Arc::new(MockModelProvider::scripted([final_response("done")]));
    let runner = AgentRunner::builder(provider)
        .event_sink(Arc::new(store.clone()))
        .build();
    let request = RunRequest::new(
        AgentDefinition::new("observability", "Observability", "1", "mock"),
        "answer without tools",
    )
    .with_run_id("runner-discovery")
    .with_trace_id("runner-discovery-trace");
    let result = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(runner.run_with_strategy(request, RunStrategy::Direct))
        .unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    drop(runner);
    drop(store);

    let connection = rusqlite::Connection::open(&path).unwrap();
    let kind: String = connection
        .query_row(
            "SELECT event_kind FROM trace_events WHERE run_id = ?1 AND event_kind = ?2",
            ["runner-discovery", "tool.discovery.completed"],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(kind, "tool.discovery.completed");
    let legacy = json!({
        "run_id": "legacy-discovery",
        "trace_id": "legacy-discovery-trace",
        "sequence": 1,
        "timestamp_ms": 1,
        "event": {
            "type": "tool_discovery_completed",
            "caller": "direct",
            "candidate_count": 3,
            "selected_count": 1,
            "deferred_candidate_count": 3,
            "catalog_exceeded_budget": true
        }
    })
    .to_string();
    connection
        .execute(
            "INSERT INTO trace_events
             (run_id, execution_id, trace_id, sequence, timestamp_ms, event_kind, status, event_json)
             VALUES (?1, ?2, ?3, 1, 1, ?4, NULL, ?5)",
            rusqlite::params![
                "legacy-discovery",
                "legacy-discovery-execution",
                "legacy-discovery-trace",
                "tool.discovery.completed",
                legacy
            ],
        )
        .unwrap();
    drop(connection);

    let reopened = SqliteEventSink::open(&path, TraceStoreConfig::default()).unwrap();
    let records = reopened.events_for_run("runner-discovery", 20, 0).unwrap();
    let discovery = records
        .iter()
        .find_map(|persisted| match persisted.record.event {
            RunEvent::ToolDiscoveryCompleted {
                caller,
                outcome,
                selection,
                candidate_count,
                selected_count,
                deferred_candidate_count,
                effective_tool_count_budget,
                effective_schema_byte_budget,
                selected_schema_bytes,
                expansion_count,
                expansion_limit,
                catalog_exceeded_budget,
                duration_ms,
            } => Some((
                caller,
                outcome,
                selection,
                candidate_count,
                selected_count,
                deferred_candidate_count,
                effective_tool_count_budget,
                effective_schema_byte_budget,
                selected_schema_bytes,
                expansion_count,
                expansion_limit,
                catalog_exceeded_budget,
                duration_ms,
            )),
            _ => None,
        })
        .unwrap();
    assert_eq!(discovery.0, ToolCaller::Direct);
    assert_eq!(discovery.1, ToolDiscoveryOutcome::Selected);
    assert_eq!(discovery.2, ToolDiscoverySelection::EmptyCatalog);
    assert_eq!(discovery.3, 0);
    assert_eq!(discovery.4, 0);
    assert_eq!(discovery.5, 0);
    assert_eq!(discovery.8, 2);
    assert!(!discovery.11);

    let legacy = reopened.events_for_run("legacy-discovery", 10, 0).unwrap();
    assert!(matches!(
        legacy[0].record.event,
        RunEvent::ToolDiscoveryCompleted {
            outcome: ToolDiscoveryOutcome::Selected,
            selection: ToolDiscoverySelection::LegacyUnclassified,
            effective_tool_count_budget: 0,
            effective_schema_byte_budget: 0,
            selected_schema_bytes: 0,
            expansion_count: 0,
            expansion_limit: 0,
            duration_ms: 0,
            ..
        }
    ));
    drop(reopened);
    fs::remove_file(path).unwrap();
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
fn discovery_events_persist_metadata_only() {
    let store = SqliteEventSink::open_in_memory(TraceStoreConfig::default()).unwrap();
    store
        .append(&record(
            "run-discovery",
            "trace-discovery",
            1,
            10,
            RunEvent::ToolDiscoveryCompleted {
                caller: ToolCaller::Direct,
                outcome: ToolDiscoveryOutcome::Selected,
                selection: ToolDiscoverySelection::LexicalExpanded,
                candidate_count: 1_000,
                selected_count: 2,
                deferred_candidate_count: 999,
                effective_tool_count_budget: 4,
                effective_schema_byte_budget: 16_384,
                selected_schema_bytes: 512,
                expansion_count: 2,
                expansion_limit: 4,
                catalog_exceeded_budget: true,
                duration_ms: 3,
            },
        ))
        .unwrap();
    let persisted = store.events_for_run("run-discovery", 10, 0).unwrap();
    assert!(matches!(
        &persisted[0].record.event,
        RunEvent::ToolDiscoveryCompleted {
            caller: ToolCaller::Direct,
            outcome: ToolDiscoveryOutcome::Selected,
            selection: ToolDiscoverySelection::LexicalExpanded,
            candidate_count: 1_000,
            selected_count: 2,
            deferred_candidate_count: 999,
            effective_tool_count_budget: 4,
            effective_schema_byte_budget: 16_384,
            selected_schema_bytes: 512,
            expansion_count: 2,
            expansion_limit: 4,
            catalog_exceeded_budget: true,
            duration_ms: 3,
        }
    ));
    let export = store.export_run_json("run-discovery").unwrap().unwrap();
    assert!(export.contains("tool_discovery_completed"));
    for forbidden in [
        "\"query\":",
        "\"tool_ids\":",
        "\"aliases\":",
        "\"schema\":",
        "\"fingerprint\":",
        "\"cache_hit\":",
    ] {
        assert!(!export.contains(forbidden));
    }
}

#[test]
fn programmatic_event_kinds_are_ordered_additive_and_redacted_on_reopen() {
    let path = temporary_database("programmatic-events");
    let canaries = [
        "PROGRAM_SOURCE_CANARY",
        "AST_VALUE_CANARY",
        "TOOL_ID_CANARY",
        "RAW_ERROR_CANARY",
    ];
    let store = SqliteEventSink::open(
        &path,
        TraceStoreConfig {
            persist_raw_payloads: true,
            redaction: RedactionConfig {
                secret_values: canaries.iter().map(ToString::to_string).collect(),
                ..RedactionConfig::default()
            },
            ..TraceStoreConfig::default()
        },
    )
    .unwrap();
    let run = "programmatic-events";
    let trace = "programmatic-events-trace";
    let events = vec![
        (
            record(
                run,
                trace,
                1,
                10,
                RunEvent::ProgramLifecycle {
                    attempt: 1,
                    outcome: ProgramLifecycleOutcome::Started,
                },
            ),
            Some(json!({
                "program": "PROGRAM_SOURCE_CANARY",
                "ast_value": "AST_VALUE_CANARY",
                "tool": "TOOL_ID_CANARY"
            })),
        ),
        (
            record(
                run,
                trace,
                2,
                20,
                RunEvent::ProgramValidated {
                    attempt: 1,
                    statement_count: 3,
                    instruction_count: 5,
                },
            ),
            None,
        ),
        (
            record(
                run,
                trace,
                3,
                30,
                RunEvent::ProgramExecutionCompleted {
                    attempt: 1,
                    fuel_used: 9,
                    branches: 1,
                    loop_iterations: 2,
                    fanout_batches: 1,
                    partial_failures: 0,
                    peak_accounted_bytes: 64,
                    duration_ms: 4,
                },
            ),
            None,
        ),
        (
            record(
                run,
                trace,
                4,
                40,
                RunEvent::ToolRejected {
                    call_id: "opaque-call".into(),
                    tool_id: "registered-tool".into(),
                    reason: "RAW_ERROR_CANARY".into(),
                },
            ),
            None,
        ),
        (completed(run, trace, 5, 50, RunStatus::Completed), None),
    ];
    assert_eq!(
        store.append_batch(events).unwrap(),
        vec![AppendOutcome::Inserted; 5]
    );
    let export = store.export_run_json(run).unwrap().unwrap();
    for canary in canaries {
        assert!(!export.contains(canary));
    }
    drop(store);

    let reopened = SqliteEventSink::open(&path, TraceStoreConfig::default()).unwrap();
    let events = reopened.events_for_run(run, 10, 0).unwrap();
    assert_eq!(
        events
            .iter()
            .map(|event| event.record.sequence)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5]
    );
    assert!(matches!(
        events[0].record.event,
        RunEvent::ProgramLifecycle {
            outcome: ProgramLifecycleOutcome::Started,
            ..
        }
    ));
    assert!(matches!(
        events[1].record.event,
        RunEvent::ProgramValidated {
            statement_count: 3,
            instruction_count: 5,
            ..
        }
    ));
    assert!(matches!(
        events[2].record.event,
        RunEvent::ProgramExecutionCompleted {
            fuel_used: 9,
            fanout_batches: 1,
            ..
        }
    ));
    assert!(matches!(
        &events[3].record.event,
        RunEvent::ToolRejected { reason, .. } if reason == REDACTED_VALUE
    ));
    assert_eq!(
        events[0].raw_payload.as_ref().unwrap()["program"],
        REDACTED_VALUE
    );
    drop(reopened);
    fs::remove_file(&path).unwrap();
    let _ = fs::remove_file(path.with_extension("sqlite-wal"));
    let _ = fs::remove_file(path.with_extension("sqlite-shm"));
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
    let mut conflicting = first.clone();
    conflicting.timestamp_ms = 11;
    conflicting.event = RunEvent::ModelResponded { call_number: 1 };
    let conflict = store.append(&conflicting).unwrap_err();
    assert_eq!(
        conflict.to_string(),
        "conflicting event for execution test-execution:run-a:trace-a (run run-a) sequence 1"
    );
    assert!(matches!(
        conflict,
        TraceStoreError::Conflict { execution_id, run_id, sequence }
            if execution_id == "test-execution:run-a:trace-a" && run_id == "run-a" && sequence == 1
    ));
    let mut wrong_trace = first.clone();
    wrong_trace.trace_id = "trace-other".into();
    wrong_trace.sequence = 3;
    wrong_trace.timestamp_ms = 30;
    wrong_trace.event = RunEvent::ModelResponded { call_number: 2 };
    assert!(matches!(
        store.append(&wrong_trace),
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
fn empty_execution_ids_are_rejected_before_single_and_batch_writes() {
    let store = SqliteEventSink::open_in_memory(TraceStoreConfig::default()).unwrap();
    let mut single = started("run-empty-single", "trace-empty-single", 1);
    single.execution_id = " \t".into();
    assert!(matches!(
        store.append(&single),
        Err(TraceStoreError::InvalidRecord(message)) if message == "execution ID must not be empty"
    ));
    assert!(store.list_runs(RunListQuery::default()).unwrap().is_empty());

    let batch_first = started("run-empty-batch", "trace-empty-batch", 1);
    let mut batch_empty = started("run-empty-batch", "trace-empty-batch", 2);
    batch_empty.execution_id = "\n".into();
    assert!(matches!(
        store.append_batch(vec![(batch_first, None), (batch_empty, None)]),
        Err(TraceStoreError::InvalidRecord(message)) if message == "execution ID must not be empty"
    ));
    assert!(store.list_runs(RunListQuery::default()).unwrap().is_empty());
}

#[test]
fn legacy_migration_uses_collision_free_execution_ids_and_stable_roundtrips() {
    let path = temporary_database("legacy-execution-ids");
    let connection = rusqlite::Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY NOT NULL);
             INSERT INTO schema_migrations(version) VALUES (1);
             CREATE TABLE trace_events (
                 run_id TEXT NOT NULL,
                 trace_id TEXT NOT NULL,
                 sequence INTEGER NOT NULL,
                 timestamp_ms INTEGER NOT NULL,
                 event_kind TEXT NOT NULL,
                 status TEXT,
                 event_json TEXT NOT NULL,
                 raw_payload_json TEXT,
                 PRIMARY KEY (run_id, sequence)
             );",
        )
        .unwrap();
    for (run_id, trace_id) in [("a:b", "c"), ("a", "b:c")] {
        let legacy_record = json!({
            "run_id": run_id,
            "trace_id": trace_id,
            "sequence": 1,
            "timestamp_ms": 1,
            "event": {"type": "model_responded", "call_number": 1}
        })
        .to_string();
        connection
            .execute(
                "INSERT INTO trace_events
                 (run_id, trace_id, sequence, timestamp_ms, event_kind, status, event_json)
                 VALUES (?1, ?2, 1, 1, 'model.responded', NULL, ?3)",
                rusqlite::params![run_id, trace_id, legacy_record],
            )
            .unwrap();
    }
    drop(connection);

    let store = SqliteEventSink::open(&path, TraceStoreConfig::default()).unwrap();
    let runs = store
        .list_runs(RunListQuery {
            limit: 10,
            ..RunListQuery::default()
        })
        .unwrap();
    assert_eq!(runs.len(), 2);
    assert_ne!(runs[0].execution_id, runs[1].execution_id);
    for run in runs {
        let once = store
            .events_for_execution(&run.execution_id, 10, 0)
            .unwrap();
        let twice = store
            .events_for_execution(&run.execution_id, 10, 0)
            .unwrap();
        assert_eq!(once, twice);
        assert_eq!(once[0].record.execution_id, run.execution_id);
        assert_eq!(once[0].record.run_id, run.run_id);
        assert_eq!(once[0].record.trace_id, run.trace_id);
        assert_eq!(
            store
                .export_execution(&run.execution_id)
                .unwrap()
                .unwrap()
                .execution_id,
            run.execution_id
        );
    }
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn read_only_open_inspects_existing_traces_without_permitting_writes() {
    let path = temporary_database("read-only");
    let writer = SqliteEventSink::open(&path, TraceStoreConfig::default()).unwrap();
    writer
        .append(&started("run-read", "trace-read", 10))
        .unwrap();
    drop(writer);

    let reader = SqliteEventSink::open_read_only(&path).unwrap();
    assert_eq!(reader.events_for_run("run-read", 10, 0).unwrap().len(), 1);
    assert!(matches!(
        reader.append(&completed(
            "run-read",
            "trace-read",
            2,
            20,
            RunStatus::Completed
        )),
        Err(TraceStoreError::Sqlite(_))
    ));
    drop(reader);

    fs::remove_file(&path).unwrap();
    let _ = fs::remove_file(path.with_extension("sqlite-wal"));
    let _ = fs::remove_file(path.with_extension("sqlite-shm"));
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
fn repeated_public_run_ids_use_distinct_execution_keys_without_sequence_conflicts() {
    let store = SqliteEventSink::open_in_memory(TraceStoreConfig::default()).unwrap();
    let explicit_first = record(
        "public-run-id",
        "public-trace-id",
        1,
        1,
        RunEvent::ModelResponded { call_number: 1 },
    );
    let mut explicit_second = record(
        "public-run-id",
        "public-trace-id",
        1,
        2,
        RunEvent::ModelResponded { call_number: 1 },
    );
    explicit_second.execution_id = "test-execution:public-run-id:second".into();
    assert_ne!(explicit_first.execution_id, explicit_second.execution_id);
    assert_eq!(
        store.append(&explicit_first).unwrap(),
        AppendOutcome::Inserted
    );
    assert_eq!(
        store.append(&explicit_second).unwrap(),
        AppendOutcome::Inserted
    );

    let generated_first = record(
        "generated-run-id",
        "generated-trace-id",
        1,
        3,
        RunEvent::ModelResponded { call_number: 1 },
    );
    let mut generated_second = record(
        "generated-run-id",
        "generated-trace-id",
        1,
        4,
        RunEvent::ModelResponded { call_number: 1 },
    );
    generated_second.execution_id = "test-execution:generated-run-id:second".into();
    assert_eq!(
        store.append(&generated_first).unwrap(),
        AppendOutcome::Inserted
    );
    assert_eq!(
        store.append(&generated_second).unwrap(),
        AppendOutcome::Inserted
    );
    assert!(matches!(
        store.events_for_run("public-run-id", 10, 0),
        Err(TraceStoreError::AmbiguousRun { .. })
    ));
    assert!(matches!(
        store.events_for_run("generated-run-id", 10, 0),
        Err(TraceStoreError::AmbiguousRun { .. })
    ));
    assert_eq!(
        store
            .events_for_execution(&explicit_first.execution_id, 10, 0)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        store
            .events_for_execution(&generated_second.execution_id, 10, 0)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn default_redaction_matches_artifact_key_tokens_without_hiding_capabilities() {
    let redacted = RedactionConfig::default().redact(&json!({
        "program": "program-source",
        "program_source": "program-source",
        "programSource": "program-source",
        "programAST": "program-source",
        "programBytecode": "program-source",
        "vm-locals": "program-source",
        "vmLocals": "program-source",
        "accessToken": "api-secret",
        "authToken": "api-secret",
        "refreshToken": "api-secret",
        "passwordHash": "api-secret",
        "clientSecret": "api-secret",
        "openai_api_key": "api-secret",
        "openaiApiKey": "api-secret",
        "x-api-key": "api-secret",
        "api_keynote": "visible",
        "resource_api": "visible",
        "programmatic_conformance": "strict_json_ast_v1",
        "locale": "en-US",
        "resource": "programmatic-resource"
    }));
    assert_eq!(redacted["program"], REDACTED_VALUE);
    assert_eq!(redacted["program_source"], REDACTED_VALUE);
    assert_eq!(redacted["programSource"], REDACTED_VALUE);
    assert_eq!(redacted["programAST"], REDACTED_VALUE);
    assert_eq!(redacted["programBytecode"], REDACTED_VALUE);
    assert_eq!(redacted["vm-locals"], REDACTED_VALUE);
    assert_eq!(redacted["vmLocals"], REDACTED_VALUE);
    assert_eq!(redacted["accessToken"], REDACTED_VALUE);
    assert_eq!(redacted["authToken"], REDACTED_VALUE);
    assert_eq!(redacted["refreshToken"], REDACTED_VALUE);
    assert_eq!(redacted["passwordHash"], REDACTED_VALUE);
    assert_eq!(redacted["clientSecret"], REDACTED_VALUE);
    assert_eq!(redacted["openai_api_key"], REDACTED_VALUE);
    assert_eq!(redacted["openaiApiKey"], REDACTED_VALUE);
    assert_eq!(redacted["x-api-key"], REDACTED_VALUE);
    assert_eq!(redacted["api_keynote"], "visible");
    assert_eq!(redacted["resource_api"], "visible");
    assert_eq!(redacted["programmatic_conformance"], "strict_json_ast_v1");
    assert_eq!(redacted["locale"], "en-US");
    assert_eq!(redacted["resource"], "programmatic-resource");
}

#[test]
fn camel_case_redaction_persists_across_reopen_queries_and_exports() {
    let path = temporary_database("camel-case-redaction");
    let store = SqliteEventSink::open(
        &path,
        TraceStoreConfig {
            persist_raw_payloads: true,
            ..TraceStoreConfig::default()
        },
    )
    .unwrap();
    let canaries = [
        "ACCESS_TOKEN_CANARY",
        "AUTH_TOKEN_CANARY",
        "REFRESH_TOKEN_CANARY",
        "PASSWORD_HASH_CANARY",
        "CLIENT_SECRET_CANARY",
        "OPENAI_API_KEY_CANARY",
        "X_API_KEY_CANARY",
        "PROGRAM_SOURCE_CANARY",
        "PROGRAM_AST_CANARY",
        "PROGRAM_BYTECODE_CANARY",
        "VM_LOCALS_CANARY",
    ];
    let event = record(
        "camel-case-redaction",
        "camel-case-redaction-trace",
        1,
        10,
        RunEvent::ModelResponded { call_number: 1 },
    );
    let raw = json!({
        "accessToken": canaries[0],
        "authToken": canaries[1],
        "refreshToken": canaries[2],
        "passwordHash": canaries[3],
        "clientSecret": canaries[4],
        "openaiApiKey": canaries[5],
        "x-api-key": canaries[6],
        "programSource": canaries[7],
        "programAST": canaries[8],
        "programBytecode": canaries[9],
        "vmLocals": canaries[10],
        "programmatic_conformance": "strict_json_ast_v1",
        "locale": "en-US",
        "resource_api": "visible-resource-api",
        "resource": "visible-resource"
    });
    store.append_with_raw(&event, Some(&raw)).unwrap();
    let export = store
        .export_run_json("camel-case-redaction")
        .unwrap()
        .unwrap();
    for canary in canaries {
        assert!(!export.contains(canary));
    }
    drop(store);

    let reopened = SqliteEventSink::open(&path, TraceStoreConfig::default()).unwrap();
    let events = reopened
        .events_for_run("camel-case-redaction", 10, 0)
        .unwrap();
    assert!(matches!(
        events[0].record.event,
        RunEvent::ModelResponded { call_number: 1 }
    ));
    let raw = events[0].raw_payload.as_ref().unwrap();
    for key in [
        "accessToken",
        "authToken",
        "refreshToken",
        "passwordHash",
        "clientSecret",
        "openaiApiKey",
        "x-api-key",
        "programSource",
        "programAST",
        "programBytecode",
        "vmLocals",
    ] {
        assert_eq!(raw[key], REDACTED_VALUE);
    }
    assert_eq!(raw["programmatic_conformance"], "strict_json_ast_v1");
    assert_eq!(raw["locale"], "en-US");
    assert_eq!(raw["resource_api"], "visible-resource-api");
    assert_eq!(raw["resource"], "visible-resource");
    let export = reopened
        .export_run_json("camel-case-redaction")
        .unwrap()
        .unwrap();
    for canary in canaries {
        assert!(!export.contains(canary));
    }
    drop(reopened);
    fs::remove_file(&path).unwrap();
    let _ = fs::remove_file(path.with_extension("sqlite-wal"));
    let _ = fs::remove_file(path.with_extension("sqlite-shm"));
}

#[test]
fn raw_payload_capture_never_persists_program_artifacts_by_default() {
    const PROGRAM_CANARY: &str = "program-artifact-canary";
    let store = SqliteEventSink::open_in_memory(TraceStoreConfig {
        persist_raw_payloads: true,
        ..TraceStoreConfig::default()
    })
    .unwrap();
    let event = record(
        "run-program-artifact",
        "trace-program-artifact",
        1,
        10,
        RunEvent::ProgramLifecycle {
            attempt: 1,
            outcome: ProgramLifecycleOutcome::Started,
        },
    );
    let raw_payload = json!({
        "program": PROGRAM_CANARY,
        "bytecode": PROGRAM_CANARY,
        "nested": {"ast": PROGRAM_CANARY, "source": PROGRAM_CANARY}
    });

    store.append_with_raw(&event, Some(&raw_payload)).unwrap();
    let persisted = store.events_for_run("run-program-artifact", 10, 0).unwrap();
    let serialized = format!("{persisted:?}");
    let export = store
        .export_run_json("run-program-artifact")
        .unwrap()
        .unwrap();
    assert!(!serialized.contains(PROGRAM_CANARY));
    assert!(!export.contains(PROGRAM_CANARY));
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
    let atomic_first = started("run-atomic", "trace-c", 1);
    let mut atomic_conflict = atomic_first.clone();
    atomic_conflict.timestamp_ms = 2;
    atomic_conflict.event = RunEvent::ModelResponded { call_number: 1 };
    assert!(matches!(
        store.append_batch(vec![(atomic_first, None), (atomic_conflict, None),]),
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
