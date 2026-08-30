use llama_harness_core::{
    ExecutionPlan, HarnessError, PlanConcurrency, PlanNode, ResultBinding, ResultRef, RunOverrides,
    RunStrategy,
};
use serde_json::json;

fn node(id: &str) -> PlanNode {
    PlanNode::new(id, format!("tool.{id}"), json!({}))
}

fn error(plan: ExecutionPlan) -> String {
    match plan.validate(32).unwrap_err() {
        HarnessError::InvalidRequest(message) => message,
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn strategy_serde_and_override_default_are_stable() {
    let overrides: RunOverrides = serde_json::from_value(json!({})).unwrap();
    assert_eq!(overrides.strategy, None);
    assert_eq!(overrides.resolved_strategy(), RunStrategy::Adaptive);
    assert_eq!(
        serde_json::to_value(RunStrategy::DeclarativePlan).unwrap(),
        json!("declarative_plan")
    );

    let overrides: RunOverrides = serde_json::from_value(json!({
        "strategy": "programmatic"
    }))
    .unwrap();
    assert_eq!(overrides.resolved_strategy(), RunStrategy::Programmatic);
}

#[test]
fn plan_defaults_and_serde_round_trip() {
    let plan: ExecutionPlan = serde_json::from_value(json!({
        "nodes": [{"id": "lookup", "tool_id": "catalog.lookup"}]
    }))
    .unwrap();
    assert_eq!(plan.nodes[0].arguments, json!(null));
    assert_eq!(plan.nodes[0].concurrency, PlanConcurrency::ToolDefault);
    assert!(!plan.nodes[0].approval_barrier);
    assert_eq!(
        serde_json::from_value::<ExecutionPlan>(serde_json::to_value(&plan).unwrap()).unwrap(),
        plan
    );
}

#[test]
fn valid_dag_accepts_transitive_result_bindings() {
    let plan = ExecutionPlan::new(vec![
        node("account"),
        node("orders").with_dependency("account"),
        node("summary")
            .with_dependency("orders")
            .with_result_binding(ResultBinding::new(
                "/customer/id",
                ResultRef::new("account", "/id"),
            )),
    ]);
    assert_eq!(plan.validate(3), Ok(()));
}

#[test]
fn validation_rejects_duplicate_and_missing_ids() {
    assert_eq!(
        error(ExecutionPlan::new(vec![node("same"), node("same")])),
        "execution plan contains duplicate node ID 'same'"
    );
    assert_eq!(
        error(ExecutionPlan::new(vec![
            node("a").with_dependency("missing")
        ])),
        "execution plan node 'a' depends on missing node 'missing'"
    );
    assert_eq!(
        error(ExecutionPlan::new(vec![node("a").with_dependency("a")])),
        "execution plan node 'a' depends on itself"
    );
    assert_eq!(
        error(ExecutionPlan::new(vec![
            node("a"),
            node("b").with_dependency("a").with_dependency("a")
        ])),
        "execution plan node 'b' contains duplicate dependency 'a'"
    );
}

#[test]
fn validation_rejects_cycles_and_nondependency_bindings() {
    assert_eq!(
        error(ExecutionPlan::new(vec![
            node("a").with_dependency("b"),
            node("b").with_dependency("a"),
        ])),
        "execution plan contains a dependency cycle involving node 'a'"
    );
    assert_eq!(
        error(ExecutionPlan::new(vec![
            node("source"),
            node("target").with_result_binding(ResultBinding::new(
                "/value",
                ResultRef::new("source", ""),
            )),
        ])),
        "execution plan node 'target' binds from node 'source' which is not a transitive dependency"
    );
}

#[test]
fn validation_rejects_bad_pointers_and_node_limit() {
    assert_eq!(
        error(ExecutionPlan::new(vec![
            node("source"),
            node("target")
                .with_dependency("source")
                .with_result_binding(ResultBinding::new("value", ResultRef::new("source", ""),)),
        ])),
        "execution plan node 'target' has invalid binding target pointer 'value'"
    );
    assert_eq!(
        error(ExecutionPlan::new(vec![
            node("source"),
            node("target")
                .with_dependency("source")
                .with_result_binding(ResultBinding::new(
                    "/value",
                    ResultRef::new("source", "value"),
                )),
        ])),
        "execution plan node 'target' has invalid source output pointer 'value'"
    );
    assert_eq!(
        ExecutionPlan::new(vec![node("a")]).validate(0),
        Err(HarnessError::InvalidRequest(
            "execution plan has 1 nodes, exceeding maximum of 0".into()
        ))
    );
}

#[test]
fn validation_rejects_empty_node_and_tool_ids() {
    assert_eq!(
        error(ExecutionPlan::new(vec![node(" ")])),
        "execution plan node at index 0 has an empty ID"
    );
    assert_eq!(
        error(ExecutionPlan::new(vec![PlanNode::new("a", " ", json!({}))])),
        "execution plan node 'a' has an empty tool ID"
    );
}
