use crate::{
    compiler::{compile_program, VerifiedProgram},
    parser::parse_program,
    SandboxError, SandboxLimits,
};
use alloc::{string::String, vec::Vec};
use serde::{Deserialize, Serialize};

/// The only program format accepted by this crate.
pub const PROGRAM_VERSION_V1: u32 = 1;

/// A strictly versioned program syntax tree.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    /// Must equal [`PROGRAM_VERSION_V1`].
    pub version: u32,
    /// Statements executed in declared order.
    pub body: Vec<Statement>,
}

impl core::fmt::Debug for Program {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Program")
            .field("version", &self.version)
            .field("statement_count", &self.body.len())
            .finish_non_exhaustive()
    }
}

impl Program {
    /// Parses and structurally bounds an untrusted UTF-8 JSON program.
    pub fn from_json(input: &[u8], limits: &SandboxLimits) -> Result<Self, SandboxError> {
        parse_program(input, limits)
    }

    /// Compiles and verifies this syntax tree into an opaque executable program.
    ///
    /// The private bytecode cannot be serialized or constructed by callers.
    pub fn compile(self, limits: &SandboxLimits) -> Result<VerifiedProgram, SandboxError> {
        compile_program(self, limits)
    }
}

/// One immutable program operation.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
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

impl core::fmt::Debug for Statement {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let kind = match self {
            Self::Let { .. } => "let",
            Self::Branch { .. } => "branch",
            Self::ForEach { .. } => "for_each",
            Self::Map { .. } => "map",
            Self::Filter { .. } => "filter",
            Self::Reduce { .. } => "reduce",
            Self::Invoke { .. } => "invoke",
            Self::FanOut { .. } => "fan_out",
            Self::Return { .. } => "return",
        };
        formatter
            .debug_struct("Statement")
            .field("kind", &kind)
            .finish_non_exhaustive()
    }
}

impl Drop for Statement {
    fn drop(&mut self) {
        let mut pending = Vec::new();
        detach_statement(self, &mut pending);
        while let Some(mut statement) = pending.pop() {
            detach_statement(&mut statement, &mut pending);
        }
    }
}

/// A deterministic expression without dynamic calls or mutable state.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
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

impl core::fmt::Debug for Expression {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let kind = match self {
            Self::Null => "null",
            Self::Boolean { .. } => "boolean",
            Self::Integer { .. } => "integer",
            Self::String { .. } => "string",
            Self::Variable { .. } => "variable",
            Self::Path { .. } => "path",
            Self::Array { .. } => "array",
            Self::Object { .. } => "object",
            Self::Binary { .. } => "binary",
            Self::Unary { .. } => "unary",
        };
        formatter
            .debug_struct("Expression")
            .field("kind", &kind)
            .finish_non_exhaustive()
    }
}

impl Drop for Expression {
    fn drop(&mut self) {
        let mut pending = Vec::new();
        detach_expression(self, &mut pending);
        while let Some(mut expression) = pending.pop() {
            detach_expression(&mut expression, &mut pending);
        }
    }
}

/// One object-construction entry.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectEntry {
    /// Object key.
    pub key: String,
    /// Object value expression.
    pub value: Expression,
}

impl core::fmt::Debug for ObjectEntry {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ObjectEntry")
            .field("key", &"<redacted>")
            .field("value", &self.value)
            .finish()
    }
}

fn detach_statement(statement: &mut Statement, pending: &mut Vec<Statement>) {
    match statement {
        Statement::Let { value, .. } | Statement::Return { value } => {
            drop(core::mem::replace(value, Expression::Null));
        }
        Statement::Branch {
            condition,
            then_body,
            else_body,
        } => {
            drop(core::mem::replace(condition, Expression::Null));
            pending.append(then_body);
            pending.append(else_body);
        }
        Statement::ForEach {
            collection, body, ..
        } => {
            drop(core::mem::replace(collection, Expression::Null));
            pending.append(body);
        }
        Statement::Map {
            collection, value, ..
        } => {
            drop(core::mem::replace(collection, Expression::Null));
            drop(core::mem::replace(value, Expression::Null));
        }
        Statement::Filter {
            collection,
            predicate,
            ..
        } => {
            drop(core::mem::replace(collection, Expression::Null));
            drop(core::mem::replace(predicate, Expression::Null));
        }
        Statement::Reduce {
            collection,
            initial,
            value,
            ..
        } => {
            drop(core::mem::replace(collection, Expression::Null));
            drop(core::mem::replace(initial, Expression::Null));
            drop(core::mem::replace(value, Expression::Null));
        }
        Statement::Invoke { arguments, .. } => {
            drop(core::mem::replace(arguments, Expression::Null));
        }
        Statement::FanOut {
            collection,
            arguments,
            ..
        } => {
            drop(core::mem::replace(collection, Expression::Null));
            drop(core::mem::replace(arguments, Expression::Null));
        }
    }
}

fn detach_expression(expression: &mut Expression, pending: &mut Vec<Expression>) {
    match expression {
        Expression::Path { value, .. } | Expression::Unary { value, .. } => {
            let child = core::mem::replace(value, alloc::boxed::Box::new(Expression::Null));
            pending.push(*child);
        }
        Expression::Array { items } => pending.append(items),
        Expression::Object { entries } => {
            let entries = core::mem::take(entries);
            pending.reserve(entries.len());
            for entry in entries {
                pending.push(entry.value);
            }
        }
        Expression::Binary { left, right, .. } => {
            let left = core::mem::replace(left, alloc::boxed::Box::new(Expression::Null));
            let right = core::mem::replace(right, alloc::boxed::Box::new(Expression::Null));
            pending.push(*left);
            pending.push(*right);
        }
        Expression::Null
        | Expression::Boolean { .. }
        | Expression::Integer { .. }
        | Expression::String { .. }
        | Expression::Variable { .. } => {}
    }
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
