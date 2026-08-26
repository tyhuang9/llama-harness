use std::sync::Arc;

use llama_harness::{
    mock::{final_response, MockModelProvider},
    observability::{RedactionConfig, SqliteEventSink, TraceStoreConfig, REDACTED_VALUE},
    AgentDefinition, AgentRunner, RunEvent, RunRequest, RunStatus,
};

#[tokio::test]
async fn facade_observability_persists_a_redacted_trace_without_raw_payloads_by_default() {
    let store = Arc::new(
        SqliteEventSink::open_in_memory(TraceStoreConfig {
            redaction: RedactionConfig {
                secret_values: vec!["top-secret".into()],
                ..RedactionConfig::default()
            },
            ..TraceStoreConfig::default()
        })
        .unwrap(),
    );
    let runner = AgentRunner::builder(Arc::new(MockModelProvider::scripted([final_response(
        "done",
    )])))
    .event_sink(store.clone())
    .build();
    let request = RunRequest::new(
        AgentDefinition::new("agent", "Agent", "1", "model-top-secret"),
        "run",
    )
    .with_run_id("observability-run");

    let result = runner.run(request).await.unwrap();
    assert_eq!(result.status, RunStatus::Completed);
    let persisted = store.events_for_run("observability-run", 100, 0).unwrap();
    assert!(persisted.len() >= 4);
    assert!(persisted.iter().any(|event| matches!(
        &event.record.event,
        RunEvent::ModelRequested { model, .. }
            if model == &format!("model-{REDACTED_VALUE}")
    )));
    assert!(persisted.iter().all(|event| event.raw_payload.is_none()));
    assert!(!store
        .export_run_json("observability-run")
        .unwrap()
        .unwrap()
        .contains("top-secret"));
}
