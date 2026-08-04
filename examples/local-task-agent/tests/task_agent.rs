use llama_harness_core::{EventSink, InMemoryEventSink, ModelProvider, RunStatus};
use llama_harness_observability::{SqliteEventSink, TraceStoreConfig};
use llama_harness_ollama::OllamaProvider;
use local_task_agent::{
    build_runtime, default_tasks, scripted_provider, MockScenario, TaskStore, CREATE_TASK_TOOL,
    UPDATE_TASK_TOOL,
};
use serde_json::json;
use std::sync::Arc;

async fn run_with_memory_events(
    scenario: MockScenario,
    grant_approval: bool,
) -> (
    llama_harness_core::RunResult,
    Arc<TaskStore>,
    Arc<InMemoryEventSink>,
) {
    let provider = Arc::new(scripted_provider(scenario));
    let store = Arc::new(TaskStore::new(default_tasks()).unwrap());
    let events = Arc::new(InMemoryEventSink::default());
    let runtime = build_runtime(
        Arc::clone(&provider) as Arc<dyn ModelProvider>,
        Arc::clone(&store),
        "mock-model",
        grant_approval,
        Arc::clone(&events) as Arc<dyn EventSink>,
    )
    .unwrap();
    let result = runtime.run("exercise the task agent", None).await.unwrap();
    (result, store, events)
}

#[tokio::test]
async fn updates_existing_task_only_after_approval() {
    let (result, store, events) =
        run_with_memory_events(MockScenario::CompleteExisting, true).await;
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(store.snapshot().unwrap()[0].status, "completed");
    assert_eq!(result.approvals.len(), 1);
    assert!(result.approvals[0].granted);
    assert!(events.events().iter().any(|event| matches!(
        event.event,
        llama_harness_core::RunEvent::ApprovalRequested { ref tool_id, .. } if tool_id == UPDATE_TASK_TOOL
    )));
}

#[tokio::test]
async fn duplicate_work_is_checked_without_creating_a_second_task() {
    let (result, store, _) = run_with_memory_events(MockScenario::ListDuplicate, true).await;
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(store.snapshot().unwrap().len(), 1);
    assert!(result.approvals.is_empty());
    assert!(result
        .tool_calls
        .iter()
        .all(|call| call.tool_id != CREATE_TASK_TOOL));
}

#[tokio::test]
async fn new_tasks_require_approval_and_isolate_application_state() {
    let (result, store, _) = run_with_memory_events(MockScenario::CreateNew, true).await;
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(store.snapshot().unwrap().len(), 2);
    assert!(store
        .snapshot()
        .unwrap()
        .iter()
        .any(|task| task.title == "Schedule dentist appointment"));

    let (_, isolated_store, _) = run_with_memory_events(MockScenario::Ambiguous, true).await;
    assert_eq!(isolated_store.snapshot().unwrap(), default_tasks());
}

#[tokio::test]
async fn ambiguous_disallowed_and_malformed_requests_do_not_mutate_tasks() {
    for scenario in [
        MockScenario::Ambiguous,
        MockScenario::DisallowedTool,
        MockScenario::MalformedArguments,
    ] {
        let (result, store, _) = run_with_memory_events(scenario, true).await;
        assert_eq!(result.status, RunStatus::Completed);
        assert_eq!(store.snapshot().unwrap(), default_tasks());
    }
    let (disallowed, _, _) = run_with_memory_events(MockScenario::DisallowedTool, true).await;
    assert!(disallowed
        .errors
        .iter()
        .any(|error| error.code == "tool_rejected"));
    let (malformed, _, _) = run_with_memory_events(MockScenario::MalformedArguments, true).await;
    assert!(malformed
        .errors
        .iter()
        .any(|error| error.code == "tool_rejected"));
}

#[tokio::test]
async fn denied_approval_leaves_the_existing_task_unchanged() {
    let (result, store, _) = run_with_memory_events(MockScenario::CompleteExisting, false).await;
    assert_eq!(result.status, RunStatus::Completed);
    assert_eq!(store.snapshot().unwrap(), default_tasks());
    assert_eq!(result.approvals.len(), 1);
    assert!(!result.approvals[0].granted);
}

#[tokio::test]
async fn model_limit_stops_before_another_call_or_state_change() {
    let provider = Arc::new(scripted_provider(MockScenario::ListDuplicate));
    let store = Arc::new(TaskStore::new(default_tasks()).unwrap());
    let mut runtime = build_runtime(
        Arc::clone(&provider) as Arc<dyn ModelProvider>,
        Arc::clone(&store),
        "mock-model",
        true,
        Arc::new(InMemoryEventSink::default()),
    )
    .unwrap();
    runtime.agent.limits.max_model_calls = 1;
    let result = runtime.run("list then stop", None).await.unwrap();
    assert_eq!(result.status, RunStatus::LimitReached);
    assert!(result.model_call_limit_reached);
    assert_eq!(provider.requests().len(), 1);
    assert_eq!(store.snapshot().unwrap(), default_tasks());
}

#[tokio::test]
async fn sqlite_trace_is_ordered_queryable_and_does_not_store_raw_payloads() {
    let provider = Arc::new(scripted_provider(MockScenario::CompleteExisting));
    let store = Arc::new(TaskStore::new(default_tasks()).unwrap());
    let trace = Arc::new(SqliteEventSink::open_in_memory(TraceStoreConfig::default()).unwrap());
    let runtime = build_runtime(
        Arc::clone(&provider) as Arc<dyn ModelProvider>,
        Arc::clone(&store),
        "mock-model",
        true,
        Arc::clone(&trace) as Arc<dyn EventSink>,
    )
    .unwrap();
    let result = runtime.run("complete", None).await.unwrap();
    let events = trace.events_for_run(&result.id, 100, 0).unwrap();
    assert!(events.len() >= 7);
    assert_eq!(events.first().unwrap().record.sequence, 1);
    assert!(events
        .windows(2)
        .all(|events| events[0].record.sequence < events[1].record.sequence));
    assert!(events.iter().all(|event| event.raw_payload.is_none()));
    let export = trace.export_run_json(&result.id).unwrap().unwrap();
    assert!(!export.contains("authorization"));
    assert!(!export.contains("private credential"));
    assert_eq!(store.snapshot().unwrap()[0].status, "completed");
    assert_eq!(
        json!({"trace_id": result.trace_id})["trace_id"],
        result.trace_id
    );
}

#[tokio::test]
async fn real_ollama_task_agent_smoke_is_opt_in() {
    if std::env::var("LLAMA_HARNESS_TEST_OLLAMA").as_deref() != Ok("1") {
        return;
    }
    let provider = Arc::new(OllamaProvider::new().unwrap());
    let health = provider.health().await.unwrap();
    assert!(
        health.healthy,
        "Ollama was not healthy: {:?}",
        health.detail
    );
    let model = match std::env::var("LLAMA_HARNESS_TEST_OLLAMA_MODEL") {
        Ok(model) => model,
        Err(_) => provider
            .list_models()
            .await
            .unwrap()
            .into_iter()
            .next()
            .map(|model| model.id)
            .expect("an installed Ollama model is required when smoke testing is enabled"),
    };
    let store = Arc::new(TaskStore::new(default_tasks()).unwrap());
    let runtime = build_runtime(
        Arc::clone(&provider) as Arc<dyn ModelProvider>,
        Arc::clone(&store),
        model.clone(),
        false,
        Arc::new(InMemoryEventSink::default()),
    )
    .unwrap();
    let result = runtime
        .run(
            "Briefly summarize the current task without changing it.",
            Some(model),
        )
        .await
        .unwrap();
    assert!(matches!(
        result.status,
        RunStatus::Completed | RunStatus::LimitReached
    ));
    assert_eq!(store.snapshot().unwrap(), default_tasks());
}
