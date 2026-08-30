use crate::HarnessError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
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

    /// Validates structural invariants without executing the plan.
    pub fn validate(&self, max_nodes: usize) -> Result<(), HarnessError> {
        if self.nodes.len() > max_nodes {
            return invalid(format!(
                "execution plan has {} nodes, exceeding maximum of {max_nodes}",
                self.nodes.len()
            ));
        }

        let mut node_indexes = BTreeMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            if node.id.trim().is_empty() {
                return invalid(format!(
                    "execution plan node at index {index} has an empty ID"
                ));
            }
            if node.tool_id.trim().is_empty() {
                return invalid(format!(
                    "execution plan node '{}' has an empty tool ID",
                    node.id
                ));
            }
            if node_indexes.insert(node.id.as_str(), index).is_some() {
                return invalid(format!(
                    "execution plan contains duplicate node ID '{}'",
                    node.id
                ));
            }
        }

        for node in &self.nodes {
            let mut dependencies = BTreeSet::new();
            for dependency in &node.depends_on {
                if dependency == &node.id {
                    return invalid(format!(
                        "execution plan node '{}' depends on itself",
                        node.id
                    ));
                }
                if !node_indexes.contains_key(dependency.as_str()) {
                    return invalid(format!(
                        "execution plan node '{}' depends on missing node '{dependency}'",
                        node.id
                    ));
                }
                if !dependencies.insert(dependency) {
                    return invalid(format!(
                        "execution plan node '{}' contains duplicate dependency '{dependency}'",
                        node.id
                    ));
                }
            }
            for binding in &node.result_bindings {
                if binding.target_pointer.is_empty() || !binding.target_pointer.starts_with('/') {
                    return invalid(format!(
                        "execution plan node '{}' has invalid binding target pointer '{}'",
                        node.id, binding.target_pointer
                    ));
                }
                if !valid_pointer(&binding.source.output_pointer) {
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
        }

        let mut states = vec![VisitState::Unvisited; self.nodes.len()];
        for index in 0..self.nodes.len() {
            visit(index, &self.nodes, &node_indexes, &mut states)?;
        }

        for node in &self.nodes {
            let dependencies = transitive_dependencies(node, &self.nodes, &node_indexes);
            for binding in &node.result_bindings {
                if !dependencies.contains(binding.source.node_id.as_str()) {
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Unvisited,
    Visiting,
    Visited,
}

fn visit(
    index: usize,
    nodes: &[PlanNode],
    node_indexes: &BTreeMap<&str, usize>,
    states: &mut [VisitState],
) -> Result<(), HarnessError> {
    match states[index] {
        VisitState::Visited => return Ok(()),
        VisitState::Visiting => {
            return invalid(format!(
                "execution plan contains a dependency cycle involving node '{}'",
                nodes[index].id
            ))
        }
        VisitState::Unvisited => {}
    }
    states[index] = VisitState::Visiting;
    for dependency in &nodes[index].depends_on {
        visit(
            node_indexes[dependency.as_str()],
            nodes,
            node_indexes,
            states,
        )?;
    }
    states[index] = VisitState::Visited;
    Ok(())
}

fn transitive_dependencies<'a>(
    node: &'a PlanNode,
    nodes: &'a [PlanNode],
    node_indexes: &BTreeMap<&str, usize>,
) -> BTreeSet<&'a str> {
    let mut dependencies = BTreeSet::new();
    let mut pending: Vec<&str> = node.depends_on.iter().map(String::as_str).collect();
    while let Some(dependency) = pending.pop() {
        if dependencies.insert(dependency) {
            pending.extend(
                nodes[node_indexes[dependency]]
                    .depends_on
                    .iter()
                    .map(String::as_str),
            );
        }
    }
    dependencies
}

fn valid_pointer(pointer: &str) -> bool {
    pointer.is_empty() || pointer.starts_with('/')
}

fn invalid<T>(message: String) -> Result<T, HarnessError> {
    Err(HarnessError::InvalidRequest(message))
}
