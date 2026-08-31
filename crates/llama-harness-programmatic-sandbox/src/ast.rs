use crate::{parser::parse_program, SandboxError, SandboxLimits};
use alloc::{string::String, vec::Vec};
use serde::{Deserialize, Serialize};

/// The only program format accepted by this crate.
pub const PROGRAM_VERSION_V1: u32 = 1;

/// A strictly versioned program syntax tree.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    /// Must equal [`PROGRAM_VERSION_V1`].
    pub version: u32,
    /// Statements executed in declared order.
    pub body: Vec<Statement>,
}

impl Program {
    /// Parses and structurally bounds an untrusted UTF-8 JSON program.
    pub fn from_json(input: &[u8], limits: &SandboxLimits) -> Result<Self, SandboxError> {
        parse_program(input, limits)
    }
}

/// One immutable program operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Statement {
    /// Binds one expression to a new immutable local.
    Let {
        /// New local name.
        name: String,
        /// Expression evaluated once.
        value: Expression,
    },
    /// Executes exactly one selected branch.
    Branch {
        /// Boolean branch condition.
        condition: Expression,
        /// Statements selected when true.
        then_body: Vec<Statement>,
        /// Statements selected when false.
        #[serde(default)]
        else_body: Vec<Statement>,
    },
    /// Iterates over a bounded collection.
    ForEach {
        /// Scoped item local.
        item: String,
        /// Collection expression.
        collection: Expression,
        /// Explicit iteration ceiling.
        max_iterations: u32,
        /// Loop body.
        body: Vec<Statement>,
    },
    /// Maps a bounded collection to a new immutable collection local.
    Map {
        /// Result local.
        name: String,
        /// Scoped item local.
        item: String,
        /// Source collection.
        collection: Expression,
        /// Explicit item ceiling.
        max_items: u32,
        /// Per-item result expression.
        value: Expression,
    },
    /// Filters a bounded collection to a new immutable collection local.
    Filter {
        /// Result local.
        name: String,
        /// Scoped item local.
        item: String,
        /// Source collection.
        collection: Expression,
        /// Explicit item ceiling.
        max_items: u32,
        /// Boolean predicate.
        predicate: Expression,
    },
    /// Reduces a bounded collection to a new immutable local.
    Reduce {
        /// Result local.
        name: String,
        /// Scoped item local.
        item: String,
        /// Scoped accumulator local.
        accumulator: String,
        /// Source collection.
        collection: Expression,
        /// Explicit item ceiling.
        max_items: u32,
        /// Initial accumulator value.
        initial: Expression,
        /// Per-item accumulator expression.
        value: Expression,
    },
    /// Yields one statically named tool call.
    Invoke {
        /// Result local populated after resume.
        name: String,
        /// Static registered tool identifier.
        tool_id: String,
        /// JSON-compatible argument expression.
        arguments: Expression,
    },
    /// Yields a bounded read-only fan-out over collection items.
    FanOut {
        /// Ordered result collection local.
        name: String,
        /// Static registered tool identifier.
        tool_id: String,
        /// Scoped item local.
        item: String,
        /// Source collection.
        collection: Expression,
        /// Explicit call ceiling, no greater than eight.
        max_calls: u8,
        /// Per-item JSON-compatible argument expression.
        arguments: Expression,
    },
    /// Completes execution with one JSON-compatible result.
    Return {
        /// Final result expression.
        value: Expression,
    },
}

/// A deterministic expression without dynamic calls or mutable state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Expression {
    /// JSON null.
    Null,
    /// Boolean literal.
    Boolean {
        /// Literal value.
        value: bool,
    },
    /// Checked signed 64-bit integer literal.
    Integer {
        /// Literal value.
        value: i64,
    },
    /// UTF-8 string literal.
    String {
        /// Literal value.
        value: String,
    },
    /// Immutable local lookup.
    Variable {
        /// Local name.
        name: String,
    },
    /// Strict RFC 6901 lookup into an expression result.
    Path {
        /// Source expression.
        value: alloc::boxed::Box<Expression>,
        /// Empty or slash-prefixed JSON pointer.
        pointer: String,
    },
    /// Ordered array construction.
    Array {
        /// Element expressions.
        items: Vec<Expression>,
    },
    /// Object construction from unique declared keys.
    Object {
        /// Object entries.
        entries: Vec<ObjectEntry>,
    },
    /// Checked binary operation.
    Binary {
        /// Operator.
        operator: BinaryOperator,
        /// Left operand.
        left: alloc::boxed::Box<Expression>,
        /// Right operand.
        right: alloc::boxed::Box<Expression>,
    },
    /// Unary operation.
    Unary {
        /// Operator.
        operator: UnaryOperator,
        /// Operand.
        value: alloc::boxed::Box<Expression>,
    },
}

/// One object-construction entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectEntry {
    /// Object key.
    pub key: String,
    /// Object value expression.
    pub value: Expression,
}

/// Deterministic binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOperator {
    /// Checked integer addition.
    Add,
    /// Checked integer subtraction.
    Subtract,
    /// Checked integer multiplication.
    Multiply,
    /// Checked integer division.
    Divide,
    /// Checked integer remainder.
    Remainder,
    /// Structural equality.
    Equal,
    /// Structural inequality.
    NotEqual,
    /// Integer less-than.
    LessThan,
    /// Integer less-than-or-equal.
    LessThanOrEqual,
    /// Integer greater-than.
    GreaterThan,
    /// Integer greater-than-or-equal.
    GreaterThanOrEqual,
    /// Boolean conjunction.
    And,
    /// Boolean disjunction.
    Or,
}

/// Deterministic unary and aggregation operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnaryOperator {
    /// Boolean negation.
    Not,
    /// Checked integer negation.
    Negate,
    /// Collection or object length.
    Count,
    /// Checked sum of integer collection entries.
    Sum,
    /// True when every boolean collection entry is true.
    All,
    /// True when any boolean collection entry is true.
    Any,
}
