use llama_harness_core::plan::{
    MAX_EXECUTION_PLAN_NODES, MAX_PLAN_ARGUMENT_BYTES, MAX_PLAN_JSON_DEPTH,
};
use llama_harness_core::{
    ExecutionPlan, HarnessError, PlanConcurrency, PlanNode, ResultBinding, ResultRef, RunStrategy,
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
fn strategy_serde_and_default_are_stable() {
    assert_eq!(RunStrategy::default(), RunStrategy::Adaptive);
    assert_eq!(
        serde_json::to_value(RunStrategy::DeclarativePlan).unwrap(),
        json!("declarative_plan")
    );
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
                ResultRef::new("account", ""),
            )),
    ]);
    assert_eq!(plan.validate(3), Ok(()));
}

#[test]
fn plan_serde_rejects_unknown_fields_at_every_level() {
    for value in [
        json!({"nodes": [], "unknown": true}),
        json!({"nodes": [{"id": "a", "tool_id": "tool.a", "unknown": true}]}),
        json!({"nodes": [
            {"id": "a", "tool_id": "tool.a"},
            {"id": "b", "tool_id": "tool.b", "depends_on": ["a"], "result_bindings": [{
                "target_pointer": "/x", "source": {"node_id": "a"}, "unknown": true
            }]}
        ]}),
        json!({"nodes": [
            {"id": "a", "tool_id": "tool.a"},
            {"id": "b", "tool_id": "tool.b", "depends_on": ["a"], "result_bindings": [{
                "target_pointer": "/x", "source": {"node_id": "a", "unknown": true}
            }]}
        ]}),
    ] {
        assert!(serde_json::from_value::<ExecutionPlan>(value).is_err());
    }
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
fn validation_rejects_malformed_pointer_escapes_and_overlapping_targets() {
    for pointer in ["/value~2", "/value~"] {
        assert_eq!(
            error(ExecutionPlan::new(vec![
                node("source"),
                node("target")
                    .with_dependency("source")
                    .with_result_binding(
                        ResultBinding::new(pointer, ResultRef::new("source", ""),)
                    ),
            ])),
            format!("execution plan node 'target' has invalid binding target pointer '{pointer}'")
        );
    }
    assert_eq!(
        error(ExecutionPlan::new(vec![
            node("source"),
            node("target")
                .with_dependency("source")
                .with_result_binding(ResultBinding::new(
                    "/value",
                    ResultRef::new("source", "/bad~2"),
                )),
        ])),
        "execution plan node 'target' has invalid source output pointer '/bad~2'"
    );

    for targets in [["/value", "/value"], ["/value", "/value/nested"]] {
        let target = node("target")
            .with_dependency("source")
            .with_result_binding(ResultBinding::new(targets[0], ResultRef::new("source", "")))
            .with_result_binding(ResultBinding::new(targets[1], ResultRef::new("source", "")));
        assert!(error(ExecutionPlan::new(vec![node("source"), target]))
            .contains("overlapping binding target pointers"));
    }
}

#[test]
fn iterative_validation_accepts_a_chain_at_the_node_hard_cap() {
    let mut nodes = Vec::with_capacity(MAX_EXECUTION_PLAN_NODES);
    for index in 0..MAX_EXECUTION_PLAN_NODES {
        let current = format!("node-{index}");
        let mut current_node = node(&current);
        if index > 0 {
            current_node = current_node.with_dependency(format!("node-{}", index - 1));
        }
        nodes.push(current_node);
    }
    assert_eq!(
        ExecutionPlan::new(nodes).validate(MAX_EXECUTION_PLAN_NODES),
        Ok(())
    );
    assert_eq!(
        ExecutionPlan::default().validate(MAX_EXECUTION_PLAN_NODES + 1),
        Err(HarnessError::InvalidRequest(format!(
            "requested maximum node count {} exceeds library hard cap of {MAX_EXECUTION_PLAN_NODES}",
            MAX_EXECUTION_PLAN_NODES + 1
        )))
    );
}

#[test]
fn validation_enforces_argument_size_and_depth_hard_caps() {
    let oversized = "x".repeat(MAX_PLAN_ARGUMENT_BYTES + 1);
    assert!(error(ExecutionPlan::new(vec![PlanNode::new(
        "large",
        "tool.large",
        json!(oversized),
    )]))
    .contains("arguments are"));

    let mut deeply_nested = json!(null);
    for _ in 0..MAX_PLAN_JSON_DEPTH {
        deeply_nested = json!([deeply_nested]);
    }
    assert!(error(ExecutionPlan::new(vec![PlanNode::new(
        "deep",
        "tool.deep",
        deeply_nested,
    )]))
    .contains("arguments depth"));
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
