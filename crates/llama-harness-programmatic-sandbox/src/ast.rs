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
///
/// Programs can only be created through [`Program::from_json`], which applies
/// the strict parser and hard structural limits before this opaque AST can be
/// retained by a caller.
///
/// ```compile_fail
/// use llama_harness_programmatic_sandbox::Program;
///
/// let _ = Program { version: 1, body: Vec::new() };
/// ```
///
/// ```compile_fail
/// use llama_harness_programmatic_sandbox::Program;
///
/// let _: Program = serde_json::from_str(r#"{"version":1,"body":[]}"#).unwrap();
/// ```
#[derive(Serialize)]
pub struct Program {
    /// Must equal [`PROGRAM_VERSION_V1`].
    pub(crate) version: u32,
    /// Statements executed in declared order.
    pub(crate) body: Vec<Statement>,
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

    /// Returns the validated top-level statement count without exposing the AST.
    pub fn statement_count(&self) -> usize {
        self.body.len()
    }

    /// Compiles and verifies this syntax tree into an opaque executable program.
    ///
    /// The private bytecode cannot be serialized or constructed by callers.
    pub fn compile(self, limits: &SandboxLimits) -> Result<VerifiedProgram, SandboxError> {
        compile_program(self, limits)
    }
}

/// Private serde wire form used only by the strict parser entry point.
///
/// Keeping this type private prevents callers from using a derived
/// `Deserialize` implementation to instantiate a [`Program`] before the
/// nesting, size, and structural checks in `parser::parse_program` run.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProgramWire {
    version: u32,
    body: Vec<Statement>,
}

pub(crate) fn deserialize_program(input: &[u8]) -> Result<Program, serde_json::Error> {
    let wire: ProgramWire = serde_json::from_slice(input)?;
    Ok(Program {
        version: wire.version,
        body: wire.body,
    })
}

/// One immutable program operation.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Statement {
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

/// A deterministic expression without dynamic calls or mutable state.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Expression {
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

/// One object-construction entry.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ObjectEntry {
    /// Object key.
    pub(crate) key: String,
    /// Object value expression.
    pub(crate) value: Expression,
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

/// Deterministic binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BinaryOperator {
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
pub(crate) enum UnaryOperator {
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
