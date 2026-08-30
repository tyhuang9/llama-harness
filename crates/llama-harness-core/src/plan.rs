use crate::HarnessError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Maximum number of nodes accepted by the plan contract.
pub const MAX_EXECUTION_PLAN_NODES: usize = 1_024;
/// Maximum total number of dependency edges accepted by the plan contract.
pub const MAX_EXECUTION_PLAN_EDGES: usize = 8_192;
/// Maximum total number of result bindings accepted by the plan contract.
pub const MAX_EXECUTION_PLAN_BINDINGS: usize = 8_192;
/// Maximum serialized plan size in bytes.
pub const MAX_EXECUTION_PLAN_BYTES: usize = 1_048_576;
/// Maximum serialized arguments size for one node in bytes.
pub const MAX_PLAN_ARGUMENT_BYTES: usize = 262_144;
/// Maximum nesting depth accepted in node arguments.
pub const MAX_PLAN_JSON_DEPTH: usize = 64;
/// Maximum byte length for node and tool identifiers.
pub const MAX_PLAN_ID_LENGTH: usize = 256;
/// Maximum byte length for dependency references and JSON pointers.
pub const MAX_PLAN_POINTER_LENGTH: usize = 1_024;

const PLAN_JSON_ENVELOPE_BYTES: usize = b"{\"nodes\":[]}".len();

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// A declarative, dependency-ordered set of tool invocations.
pub struct ExecutionPlan {
    #[serde(default)]
    /// Nodes contained in this plan.
    pub nodes: Vec<PlanNode>,
}

impl ExecutionPlan {
    /// Creates an execution plan from an ordered list of nodes.
    pub fn new(nodes: Vec<PlanNode>) -> Self {
        Self { nodes }
    }

    /// Appends a node to the plan.
    pub fn with_node(mut self, node: PlanNode) -> Self {
        self.nodes.push(node);
        self
    }

    /// Validates bounded structural invariants without executing the plan.
    pub fn validate(&self, max_nodes: usize) -> Result<(), HarnessError> {
        if max_nodes > MAX_EXECUTION_PLAN_NODES {
            return invalid(format!(
                "requested maximum node count {max_nodes} exceeds library hard cap of {MAX_EXECUTION_PLAN_NODES}"
            ));
        }
        if self.nodes.len() > max_nodes {
            return invalid(format!(
                "execution plan has {} nodes, exceeding maximum of {max_nodes}",
                self.nodes.len()
            ));
        }

        let mut node_indexes = BTreeMap::new();
        let mut total_edges = 0usize;
        let mut total_bindings = 0usize;
        let mut serialized_plan_bytes = PLAN_JSON_ENVELOPE_BYTES;
        for (index, node) in self.nodes.iter().enumerate() {
            validate_node_header(index, node)?;
            if node_indexes.insert(node.id.as_str(), index).is_some() {
                return invalid(format!(
                    "execution plan contains duplicate node ID '{}'",
                    node.id
                ));
            }
            total_edges = total_edges
                .checked_add(node.depends_on.len())
                .ok_or_else(|| invalid_error("execution plan dependency count overflow"))?;
            total_bindings = total_bindings
                .checked_add(node.result_bindings.len())
                .ok_or_else(|| invalid_error("execution plan binding count overflow"))?;
            let serialized_node_bytes = serde_json::to_vec(node)
                .map_err(|error| {
                    invalid_error(format!(
                        "execution plan node '{}' cannot be serialized: {error}",
                        node.id
                    ))
                })?
                .len();
            serialized_plan_bytes = serialized_plan_bytes
                .checked_add(usize::from(index > 0))
                .and_then(|bytes| bytes.checked_add(serialized_node_bytes))
                .ok_or_else(|| invalid_error("execution plan serialized size overflow"))?;
            if serialized_plan_bytes > MAX_EXECUTION_PLAN_BYTES {
                return invalid(format!(
                    "execution plan serialized size exceeds library hard cap of {MAX_EXECUTION_PLAN_BYTES} bytes at node '{}'",
                    node.id
                ));
            }
        }
        if total_edges > MAX_EXECUTION_PLAN_EDGES {
            return invalid(format!(
                "execution plan has {total_edges} dependency edges, exceeding library hard cap of {MAX_EXECUTION_PLAN_EDGES}"
            ));
        }
        if total_bindings > MAX_EXECUTION_PLAN_BINDINGS {
            return invalid(format!(
                "execution plan has {total_bindings} result bindings, exceeding library hard cap of {MAX_EXECUTION_PLAN_BINDINGS}"
            ));
        }

        let mut indegrees = vec![0usize; self.nodes.len()];
        let mut dependents = vec![Vec::new(); self.nodes.len()];
        for (index, node) in self.nodes.iter().enumerate() {
            let mut dependencies = BTreeSet::new();
            for dependency in &node.depends_on {
                validate_reference_length("dependency node ID", &node.id, dependency)?;
                if dependency == &node.id {
                    return invalid(format!(
                        "execution plan node '{}' depends on itself",
                        node.id
                    ));
                }
                let Some(&dependency_index) = node_indexes.get(dependency.as_str()) else {
                    return invalid(format!(
                        "execution plan node '{}' depends on missing node '{dependency}'",
                        node.id
                    ));
                };
                if !dependencies.insert(dependency) {
                    return invalid(format!(
                        "execution plan node '{}' contains duplicate dependency '{dependency}'",
                        node.id
                    ));
                }
                indegrees[index] += 1;
                dependents[dependency_index].push(index);
            }
            validate_bindings(node, &node_indexes)?;
        }

        let topological_order = topological_order(&self.nodes, &dependents, &mut indegrees)?;
        let ancestors = compute_ancestors(&self.nodes, &node_indexes, &topological_order);
        for (index, node) in self.nodes.iter().enumerate() {
            for binding in &node.result_bindings {
                let source_index = node_indexes[binding.source.node_id.as_str()];
                if !ancestors[index][source_index] {
                    return invalid(format!(
                        "execution plan node '{}' binds from node '{}' which is not a transitive dependency",
                        node.id, binding.source.node_id
                    ));
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// One tool invocation within an execution plan.
pub struct PlanNode {
    /// Stable node identifier within the plan.
    pub id: String,
    /// Tool identifier to invoke.
    pub tool_id: String,
    #[serde(default)]
    /// JSON arguments supplied to the tool.
    pub arguments: Value,
    #[serde(default)]
    /// Node IDs that must complete before this node.
    pub depends_on: Vec<String>,
    #[serde(default)]
    /// Values copied from dependency outputs into this node's arguments.
    pub result_bindings: Vec<ResultBinding>,
    #[serde(default)]
    /// Concurrency constraint for this node.
    pub concurrency: PlanConcurrency,
    #[serde(default)]
    /// Whether host approval is required before this node executes.
    pub approval_barrier: bool,
    #[serde(default)]
    /// Whether this node forms an application-defined commit boundary.
    pub commit_boundary: bool,
}

impl PlanNode {
    /// Creates a plan node with no dependencies or barriers.
    pub fn new(id: impl Into<String>, tool_id: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            tool_id: tool_id.into(),
            arguments,
            depends_on: Vec::new(),
            result_bindings: Vec::new(),
            concurrency: PlanConcurrency::default(),
            approval_barrier: false,
            commit_boundary: false,
        }
    }

    /// Adds a dependency by node ID.
    pub fn with_dependency(mut self, node_id: impl Into<String>) -> Self {
        self.depends_on.push(node_id.into());
        self
    }

    /// Adds a result binding from a dependency output.
    pub fn with_result_binding(mut self, binding: ResultBinding) -> Self {
        self.result_bindings.push(binding);
        self
    }

    /// Sets the concurrency constraint.
    pub fn with_concurrency(mut self, concurrency: PlanConcurrency) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Sets whether host approval is required before execution.
    pub fn with_approval_barrier(mut self, enabled: bool) -> Self {
        self.approval_barrier = enabled;
        self
    }

    /// Sets whether this node is an application-defined commit boundary.
    pub fn with_commit_boundary(mut self, enabled: bool) -> Self {
        self.commit_boundary = enabled;
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// Copies one dependency output value into a node's arguments.
pub struct ResultBinding {
    /// Nonempty JSON pointer identifying the target within node arguments.
    pub target_pointer: String,
    /// Dependency output supplying the bound value.
    pub source: ResultRef,
}

impl ResultBinding {
    /// Creates a result binding.
    pub fn new(target_pointer: impl Into<String>, source: ResultRef) -> Self {
        Self {
            target_pointer: target_pointer.into(),
            source,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// Reference to a value in a plan node's output.
pub struct ResultRef {
    /// Source node identifier.
    pub node_id: String,
    #[serde(default)]
    /// JSON pointer within the source output; empty selects the entire output.
    pub output_pointer: String,
}

impl ResultRef {
    /// Creates a result reference.
    pub fn new(node_id: impl Into<String>, output_pointer: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            output_pointer: output_pointer.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Concurrency constraint applied to a plan node.
pub enum PlanConcurrency {
    /// Uses the referenced tool's default concurrency behavior.
    #[default]
    ToolDefault,
    /// Requires serialized execution relative to other serial nodes.
    Serial,
}

fn validate_node_header(index: usize, node: &PlanNode) -> Result<(), HarnessError> {
    if node.id.trim().is_empty() {
        return invalid(format!(
            "execution plan node at index {index} has an empty ID"
        ));
    }
    if node.id.len() > MAX_PLAN_ID_LENGTH {
        return invalid(format!(
            "execution plan node ID at index {index} exceeds {MAX_PLAN_ID_LENGTH} bytes"
        ));
    }
    if node.tool_id.trim().is_empty() {
        return invalid(format!(
            "execution plan node '{}' has an empty tool ID",
            node.id
        ));
    }
    if node.tool_id.len() > MAX_PLAN_ID_LENGTH {
        return invalid(format!(
            "execution plan node '{}' tool ID exceeds {MAX_PLAN_ID_LENGTH} bytes",
            node.id
        ));
    }
    let argument_size = serde_json::to_vec(&node.arguments)
        .map_err(|error| {
            invalid_error(format!(
                "node '{}' arguments cannot be serialized: {error}",
                node.id
            ))
        })?
        .len();
    if argument_size > MAX_PLAN_ARGUMENT_BYTES {
        return invalid(format!(
            "execution plan node '{}' arguments are {argument_size} bytes, exceeding library hard cap of {MAX_PLAN_ARGUMENT_BYTES} bytes",
            node.id
        ));
    }
    let depth = json_depth(&node.arguments);
    if depth > MAX_PLAN_JSON_DEPTH {
        return invalid(format!(
            "execution plan node '{}' arguments depth {depth} exceeds library hard cap of {MAX_PLAN_JSON_DEPTH}",
            node.id
        ));
    }
    Ok(())
}

fn validate_bindings(
    node: &PlanNode,
    node_indexes: &BTreeMap<&str, usize>,
) -> Result<(), HarnessError> {
    let mut targets = BTreeSet::new();
    for binding in &node.result_bindings {
        validate_pointer_length("binding target pointer", &node.id, &binding.target_pointer)?;
        if binding.target_pointer.is_empty() || !valid_json_pointer(&binding.target_pointer) {
            return invalid(format!(
                "execution plan node '{}' has invalid binding target pointer '{}'",
                node.id, binding.target_pointer
            ));
        }
        if !targets.insert(binding.target_pointer.as_str()) {
            return invalid(format!(
                "execution plan node '{}' has overlapping binding target pointers '{}' and '{}'",
                node.id, binding.target_pointer, binding.target_pointer
            ));
        }

        validate_reference_length("binding source node ID", &node.id, &binding.source.node_id)?;
        validate_pointer_length(
            "source output pointer",
            &node.id,
            &binding.source.output_pointer,
        )?;
        if !valid_json_pointer(&binding.source.output_pointer) {
            return invalid(format!(
                "execution plan node '{}' has invalid source output pointer '{}'",
                node.id, binding.source.output_pointer
            ));
        }
        if !node_indexes.contains_key(binding.source.node_id.as_str()) {
            return invalid(format!(
                "execution plan node '{}' binds from missing node '{}'",
                node.id, binding.source.node_id
            ));
        }
    }
    for target in &targets {
        for (index, byte) in target.bytes().enumerate().skip(1) {
            if byte == b'/' {
                let ancestor = &target[..index];
                if targets.contains(ancestor) {
                    return invalid(format!(
                        "execution plan node '{}' has overlapping binding target pointers '{}' and '{}'",
                        node.id, ancestor, target
                    ));
                }
            }
        }
    }
    Ok(())
}

fn topological_order(
    nodes: &[PlanNode],
    dependents: &[Vec<usize>],
    indegrees: &mut [usize],
) -> Result<Vec<usize>, HarnessError> {
    let mut ready: VecDeque<usize> = indegrees
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(index) = ready.pop_front() {
        order.push(index);
        for &dependent in &dependents[index] {
            indegrees[dependent] -= 1;
            if indegrees[dependent] == 0 {
                ready.push_back(dependent);
            }
        }
    }
    if order.len() != nodes.len() {
        let index = indegrees.iter().position(|degree| *degree > 0).unwrap_or(0);
        return invalid(format!(
            "execution plan contains a dependency cycle involving node '{}'",
            nodes[index].id
        ));
    }
    Ok(order)
}

fn compute_ancestors(
    nodes: &[PlanNode],
    node_indexes: &BTreeMap<&str, usize>,
    topological_order: &[usize],
) -> Vec<Vec<bool>> {
    let mut ancestors = vec![vec![false; nodes.len()]; nodes.len()];
    for &index in topological_order {
        for dependency in &nodes[index].depends_on {
            let dependency_index = node_indexes[dependency.as_str()];
            let (node_ancestors, dependency_ancestors) = if index < dependency_index {
                let (before_dependency, from_dependency) = ancestors.split_at_mut(dependency_index);
                (&mut before_dependency[index], &from_dependency[0])
            } else {
                let (before_node, from_node) = ancestors.split_at_mut(index);
                (&mut from_node[0], &before_node[dependency_index])
            };
            node_ancestors[dependency_index] = true;
            for (node_ancestor, dependency_ancestor) in
                node_ancestors.iter_mut().zip(dependency_ancestors)
            {
                *node_ancestor |= *dependency_ancestor;
            }
        }
    }
    ancestors
}

fn json_depth(value: &Value) -> usize {
    let mut maximum = 0usize;
    let mut pending = vec![(value, 1usize)];
    while let Some((value, depth)) = pending.pop() {
        maximum = maximum.max(depth);
        match value {
            Value::Array(values) => {
                pending.extend(values.iter().map(|value| (value, depth + 1)));
            }
            Value::Object(values) => {
                pending.extend(values.values().map(|value| (value, depth + 1)));
            }
            _ => {}
        }
    }
    maximum
}

fn valid_json_pointer(pointer: &str) -> bool {
    if pointer.is_empty() {
        return true;
    }
    if !pointer.starts_with('/') {
        return false;
    }
    let bytes = pointer.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'~' {
            if index + 1 >= bytes.len() || !matches!(bytes[index + 1], b'0' | b'1') {
                return false;
            }
            index += 2;
        } else {
            index += 1;
        }
    }
    true
}

fn validate_reference_length(label: &str, node_id: &str, value: &str) -> Result<(), HarnessError> {
    if value.len() > MAX_PLAN_ID_LENGTH {
        return invalid(format!(
            "execution plan node '{node_id}' {label} exceeds {MAX_PLAN_ID_LENGTH} bytes"
        ));
    }
    Ok(())
}

fn validate_pointer_length(label: &str, node_id: &str, value: &str) -> Result<(), HarnessError> {
    if value.len() > MAX_PLAN_POINTER_LENGTH {
        return invalid(format!(
            "execution plan node '{node_id}' {label} exceeds {MAX_PLAN_POINTER_LENGTH} bytes"
        ));
    }
    Ok(())
}

fn invalid<T>(message: String) -> Result<T, HarnessError> {
    Err(invalid_error(message))
}

fn invalid_error(message: impl Into<String>) -> HarnessError {
    HarnessError::InvalidRequest(message.into())
}
