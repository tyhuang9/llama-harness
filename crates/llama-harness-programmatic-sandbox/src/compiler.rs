use crate::{
    accounting::{
        checked_add, key_retained_bytes, measure_value, serialized_string_len,
        string_retained_bytes,
    },
    parser::validate_ast,
    BinaryOperator, Expression, Program, SandboxError, SandboxErrorCode, SandboxLimits, Statement,
    UnaryOperator, MAX_ATOMIC_KEY_BYTES, MAX_ATOMIC_STRING_BYTES,
};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    string::String,
    vec,
    vec::Vec,
};
use serde_json::Value;

/// An opaque, verified executable. Its bytecode is private and non-serializable.
pub struct VerifiedProgram {
    pub(crate) code: Vec<Instruction>,
    pub(crate) local_count: usize,
    pub(crate) limits: SandboxLimits,
}

impl core::fmt::Debug for VerifiedProgram {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("VerifiedProgram")
            .field("instruction_count", &self.code.len())
            .field("local_count", &self.local_count)
            .finish_non_exhaustive()
    }
}

impl VerifiedProgram {
    /// Returns the number of private verified top-level instructions.
    pub fn instruction_count(&self) -> usize {
        self.code.len()
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Instruction {
    Let {
        slot: usize,
        value: ExprCode,
    },
    Branch {
        condition: ExprCode,
        false_target: usize,
    },
    Jump {
        target: usize,
    },
    LoopStart {
        collection: ExprCode,
        item_slot: usize,
        max_iterations: usize,
        end_target: usize,
        body_slots: Vec<usize>,
    },
    LoopNext {
        body_target: usize,
    },
    Map {
        slot: usize,
        item_slot: usize,
        collection: ExprCode,
        max_items: usize,
        value: ExprCode,
    },
    Filter {
        slot: usize,
        item_slot: usize,
        collection: ExprCode,
        max_items: usize,
        predicate: ExprCode,
    },
    Reduce {
        slot: usize,
        item_slot: usize,
        accumulator_slot: usize,
        collection: ExprCode,
        max_items: usize,
        initial: ExprCode,
        value: ExprCode,
    },
    Invoke {
        slot: usize,
        tool_id: String,
        arguments: ExprCode,
        call_site: u32,
    },
    FanOut {
        slot: usize,
        tool_id: String,
        item_slot: usize,
        collection: ExprCode,
        max_calls: usize,
        arguments: ExprCode,
        call_site: u32,
    },
    Return {
        value: ExprCode,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct ExprCode(pub(crate) Vec<ExprInstruction>);

#[derive(Clone, Debug)]
pub(crate) enum ExprInstruction {
    Constant(Value),
    Load(usize),
    Path(String),
    Array(usize),
    Object(Vec<String>),
    Binary(BinaryOperator),
    Unary(UnaryOperator),
}

pub(crate) fn compile_program(
    program: Program,
    limits: &SandboxLimits,
) -> Result<VerifiedProgram, SandboxError> {
    limits.validate()?;
    validate_ast(&program, limits)?;
    let mut compiler = Compiler {
        code: Vec::new(),
        next_slot: 0,
        next_call_site: 0,
        expression_instructions: 0,
        limits: *limits,
    };
    compiler
        .code
        .try_reserve(program.body.len())
        .map_err(|_| resource("bytecode allocation failed"))?;
    let mut environment = BTreeMap::new();
    compiler.compile_block(&program.body, &mut environment, 1)?;
    if !matches!(compiler.code.last(), Some(Instruction::Return { .. })) {
        return Err(verify("program must end with a return statement"));
    }
    // Deliberately verify the completed private bytecode rather than relying on
    // the compiler's construction invariants. This is the last boundary before
    // the VM receives an executable program.
    verify_bytecode(&compiler.code, compiler.next_slot, limits)?;
    Ok(VerifiedProgram {
        code: compiler.code,
        local_count: compiler.next_slot,
        limits: *limits,
    })
}

struct Compiler {
    code: Vec<Instruction>,
    next_slot: usize,
    next_call_site: u32,
    expression_instructions: usize,
    limits: SandboxLimits,
}

impl Compiler {
    fn compile_block(
        &mut self,
        statements: &[Statement],
        environment: &mut BTreeMap<String, usize>,
        depth: usize,
    ) -> Result<(), SandboxError> {
        if depth > self.limits.max_control_stack {
            return Err(resource("control stack limit exceeded"));
        }
        for statement in statements {
            self.reserve_instruction()?;
            match statement {
                Statement::Let { name, value } => {
                    let value = self.compile_expression(value, environment, depth + 1)?;
                    let slot = self.bind(name, environment)?;
                    self.code.push(Instruction::Let { slot, value });
                }
                Statement::Branch {
                    condition,
                    then_body,
                    else_body,
                } => {
                    let condition = self.compile_expression(condition, environment, depth + 1)?;
                    let branch_index = self.code.len();
                    self.code.push(Instruction::Branch {
                        condition,
                        false_target: 0,
                    });
                    let mut then_environment = environment.clone();
                    self.compile_block(then_body, &mut then_environment, depth + 1)?;
                    self.reserve_instruction()?;
                    let jump_index = self.code.len();
                    self.code.push(Instruction::Jump { target: 0 });
                    let else_target = self.code.len();
                    let mut else_environment = environment.clone();
                    self.compile_block(else_body, &mut else_environment, depth + 1)?;
                    let end_target = self.code.len();
                    match &mut self.code[branch_index] {
                        Instruction::Branch { false_target, .. } => *false_target = else_target,
                        _ => return Err(verify("compiler emitted invalid branch")),
                    }
                    match &mut self.code[jump_index] {
                        Instruction::Jump { target } => *target = end_target,
                        _ => return Err(verify("compiler emitted invalid jump")),
                    }
                }
                Statement::ForEach {
                    item,
                    collection,
                    max_iterations,
                    body,
                } => {
                    let collection = self.compile_expression(collection, environment, depth + 1)?;
                    let item_slot = self.allocate_slot()?;
                    let start_index = self.code.len();
                    self.code.push(Instruction::LoopStart {
                        collection,
                        item_slot,
                        max_iterations: *max_iterations as usize,
                        end_target: 0,
                        body_slots: Vec::new(),
                    });
                    let body_target = self.code.len();
                    let mut body_environment = environment.clone();
                    body_environment.insert(item.clone(), item_slot);
                    self.compile_block(body, &mut body_environment, depth + 1)?;
                    self.reserve_instruction()?;
                    self.code.push(Instruction::LoopNext { body_target });
                    let end_target = self.code.len();
                    let mut body_slots = Vec::new();
                    body_slots
                        .try_reserve_exact(self.next_slot.saturating_sub(item_slot + 1))
                        .map_err(|_| resource("bytecode loop metadata allocation failed"))?;
                    body_slots.extend(item_slot + 1..self.next_slot);
                    match &mut self.code[start_index] {
                        Instruction::LoopStart {
                            end_target: target,
                            body_slots: slots,
                            ..
                        } => {
                            *target = end_target;
                            *slots = body_slots;
                        }
                        _ => return Err(verify("compiler emitted invalid loop")),
                    }
                }
                Statement::Map {
                    name,
                    item,
                    collection,
                    max_items,
                    value,
                } => {
                    let collection = self.compile_expression(collection, environment, depth + 1)?;
                    let item_slot = self.allocate_slot()?;
                    let mut scoped = environment.clone();
                    scoped.insert(item.clone(), item_slot);
                    let value = self.compile_expression(value, &scoped, depth + 1)?;
                    let slot = self.bind(name, environment)?;
                    self.code.push(Instruction::Map {
                        slot,
                        item_slot,
                        collection,
                        max_items: *max_items as usize,
                        value,
                    });
                }
                Statement::Filter {
                    name,
                    item,
                    collection,
                    max_items,
                    predicate,
                } => {
                    let collection = self.compile_expression(collection, environment, depth + 1)?;
                    let item_slot = self.allocate_slot()?;
                    let mut scoped = environment.clone();
                    scoped.insert(item.clone(), item_slot);
                    let predicate = self.compile_expression(predicate, &scoped, depth + 1)?;
                    let slot = self.bind(name, environment)?;
                    self.code.push(Instruction::Filter {
                        slot,
                        item_slot,
                        collection,
                        max_items: *max_items as usize,
                        predicate,
                    });
                }
                Statement::Reduce {
                    name,
                    item,
                    accumulator,
                    collection,
                    max_items,
                    initial,
                    value,
                } => {
                    let collection = self.compile_expression(collection, environment, depth + 1)?;
                    let initial = self.compile_expression(initial, environment, depth + 1)?;
                    let item_slot = self.allocate_slot()?;
                    let accumulator_slot = self.allocate_slot()?;
                    let mut scoped = environment.clone();
                    scoped.insert(item.clone(), item_slot);
                    scoped.insert(accumulator.clone(), accumulator_slot);
                    let value = self.compile_expression(value, &scoped, depth + 1)?;
                    let slot = self.bind(name, environment)?;
                    self.code.push(Instruction::Reduce {
                        slot,
                        item_slot,
                        accumulator_slot,
                        collection,
                        max_items: *max_items as usize,
                        initial,
                        value,
                    });
                }
                Statement::Invoke {
                    name,
                    tool_id,
                    arguments,
                } => {
                    let arguments = self.compile_expression(arguments, environment, depth + 1)?;
                    let slot = self.bind(name, environment)?;
                    let call_site = self.call_site()?;
                    self.code.push(Instruction::Invoke {
                        slot,
                        tool_id: tool_id.clone(),
                        arguments,
                        call_site,
                    });
                }
                Statement::FanOut {
                    name,
                    tool_id,
                    item,
                    collection,
                    max_calls,
                    arguments,
                } => {
                    let collection = self.compile_expression(collection, environment, depth + 1)?;
                    let item_slot = self.allocate_slot()?;
                    let mut scoped = environment.clone();
                    scoped.insert(item.clone(), item_slot);
                    let arguments = self.compile_expression(arguments, &scoped, depth + 1)?;
                    let slot = self.bind(name, environment)?;
                    let call_site = self.call_site()?;
                    self.code.push(Instruction::FanOut {
                        slot,
                        tool_id: tool_id.clone(),
                        item_slot,
                        collection,
                        max_calls: *max_calls as usize,
                        arguments,
                        call_site,
                    });
                }
                Statement::Return { value } => {
                    let value = self.compile_expression(value, environment, depth + 1)?;
                    self.code.push(Instruction::Return { value });
                }
            }
        }
        Ok(())
    }

    fn compile_expression(
        &mut self,
        expression: &Expression,
        environment: &BTreeMap<String, usize>,
        depth: usize,
    ) -> Result<ExprCode, SandboxError> {
        if depth > self.limits.max_nesting {
            return Err(resource("expression nesting limit exceeded"));
        }
        let mut code = Vec::new();
        self.emit_expression(expression, environment, depth, &mut code)?;
        verify_expression(&code, self.limits.max_operand_stack)?;
        self.expression_instructions = self
            .expression_instructions
            .checked_add(code.len())
            .ok_or_else(|| resource("bytecode instruction limit exceeded"))?;
        if self.expression_instructions + self.code.len() > self.limits.max_bytecode_instructions {
            return Err(resource("bytecode instruction limit exceeded"));
        }
        Ok(ExprCode(code))
    }

    fn emit_expression(
        &self,
        expression: &Expression,
        environment: &BTreeMap<String, usize>,
        depth: usize,
        code: &mut Vec<ExprInstruction>,
    ) -> Result<(), SandboxError> {
        if depth > self.limits.max_nesting {
            return Err(resource("expression nesting limit exceeded"));
        }
        if code.len() >= self.limits.max_bytecode_instructions {
            return Err(resource("bytecode instruction limit exceeded"));
        }
        match expression {
            Expression::Null => code.push(ExprInstruction::Constant(Value::Null)),
            Expression::Boolean { value } => {
                code.push(ExprInstruction::Constant(Value::Bool(*value)))
            }
            Expression::Integer { value } => {
                code.push(ExprInstruction::Constant(Value::Number((*value).into())))
            }
            Expression::String { value } => {
                code.push(ExprInstruction::Constant(Value::String(value.clone())))
            }
            Expression::Variable { name } => {
                code.push(ExprInstruction::Load(*environment.get(name).ok_or_else(
                    || verify("variable must reference an earlier binding in the same scope"),
                )?))
            }
            Expression::Path { value, pointer } => {
                self.emit_expression(value, environment, depth + 1, code)?;
                code.push(ExprInstruction::Path(pointer.clone()));
            }
            Expression::Array { items } => {
                for item in items {
                    self.emit_expression(item, environment, depth + 1, code)?;
                }
                code.push(ExprInstruction::Array(items.len()));
            }
            Expression::Object { entries } => {
                let mut keys = Vec::new();
                keys.try_reserve(entries.len())
                    .map_err(|_| resource("bytecode allocation failed"))?;
                for entry in entries {
                    self.emit_expression(&entry.value, environment, depth + 1, code)?;
                    keys.push(entry.key.clone());
                }
                code.push(ExprInstruction::Object(keys));
            }
            Expression::Binary {
                operator,
                left,
                right,
            } => {
                self.emit_expression(left, environment, depth + 1, code)?;
                self.emit_expression(right, environment, depth + 1, code)?;
                code.push(ExprInstruction::Binary(*operator));
            }
            Expression::Unary { operator, value } => {
                self.emit_expression(value, environment, depth + 1, code)?;
                code.push(ExprInstruction::Unary(*operator));
            }
        }
        if code.len() > self.limits.max_bytecode_instructions {
            return Err(resource("bytecode instruction limit exceeded"));
        }
        Ok(())
    }

    fn bind(
        &mut self,
        name: &str,
        environment: &mut BTreeMap<String, usize>,
    ) -> Result<usize, SandboxError> {
        if environment.contains_key(name) {
            return Err(verify("immutable local cannot be rebound"));
        }
        let slot = self.allocate_slot()?;
        environment.insert(name.into(), slot);
        Ok(slot)
    }

    fn allocate_slot(&mut self) -> Result<usize, SandboxError> {
        if self.next_slot >= self.limits.max_locals {
            return Err(resource("local binding limit exceeded"));
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        Ok(slot)
    }

    fn call_site(&mut self) -> Result<u32, SandboxError> {
        let site = self.next_call_site;
        self.next_call_site = self
            .next_call_site
            .checked_add(1)
            .ok_or_else(|| resource("call-site limit exceeded"))?;
        Ok(site)
    }

    fn reserve_instruction(&self) -> Result<(), SandboxError> {
        if self.code.len() + self.expression_instructions >= self.limits.max_bytecode_instructions {
            Err(resource("bytecode instruction limit exceeded"))
        } else {
            Ok(())
        }
    }
}

fn instruction_slots(instruction: &Instruction) -> Vec<usize> {
    match instruction {
        Instruction::Let { slot, .. } | Instruction::Invoke { slot, .. } => vec![*slot],
        Instruction::LoopStart { item_slot, .. } => vec![*item_slot],
        Instruction::Map {
            slot, item_slot, ..
        }
        | Instruction::Filter {
            slot, item_slot, ..
        }
        | Instruction::FanOut {
            slot, item_slot, ..
        } => vec![*slot, *item_slot],
        Instruction::Reduce {
            slot,
            item_slot,
            accumulator_slot,
            ..
        } => vec![*slot, *item_slot, *accumulator_slot],
        _ => Vec::new(),
    }
}

fn instruction_slots_in_region(
    region: &[Instruction],
    outer_item_slot: usize,
) -> Result<Vec<usize>, SandboxError> {
    let mut unique = BTreeSet::new();
    for instruction in region {
        for slot in instruction_slots(instruction) {
            if slot != outer_item_slot {
                unique.insert(slot);
            }
        }
    }
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(unique.len())
        .map_err(|_| resource("bytecode loop metadata allocation failed"))?;
    slots.extend(unique);
    Ok(slots)
}

/// Independently validates private compiler output before it is admitted to
/// the VM. The verifier intentionally receives only bytecode, its local
/// count, and immutable limits: it must not depend on compiler bookkeeping.
fn verify_bytecode(
    code: &[Instruction],
    local_count: usize,
    limits: &SandboxLimits,
) -> Result<(), SandboxError> {
    limits.validate()?;
    if code.is_empty() {
        return Err(verify("bytecode must not be empty"));
    }
    if local_count > limits.max_locals {
        return Err(resource("local binding limit exceeded"));
    }

    let mut call_sites = BTreeSet::new();
    let mut loop_bodies = BTreeMap::new();
    let mut expression_instructions = 0usize;
    let mut constant_bytes = 0usize;

    for (pc, instruction) in code.iter().enumerate() {
        for slot in instruction_slots(instruction) {
            if slot >= local_count {
                return Err(verify("bytecode local slot is invalid"));
            }
        }
        expression_instructions = expression_instructions
            .checked_add(verify_instruction_expressions(
                instruction,
                local_count,
                limits,
                &mut constant_bytes,
            )?)
            .ok_or_else(|| resource("bytecode instruction limit exceeded"))?;

        match instruction {
            Instruction::Branch { false_target, .. } => {
                if *false_target <= pc || *false_target > code.len() {
                    return Err(verify("bytecode branch target is invalid"));
                }
            }
            Instruction::Jump { target } => {
                if *target <= pc || *target > code.len() {
                    return Err(verify("bytecode jump target is invalid"));
                }
            }
            Instruction::LoopStart {
                item_slot,
                max_iterations,
                end_target,
                body_slots,
                ..
            } => {
                if *max_iterations == 0 || *max_iterations > limits.max_loop_iterations {
                    return Err(verify("bytecode loop bound is invalid"));
                }
                if *end_target <= pc + 1 || *end_target > code.len() {
                    return Err(verify("bytecode loop target is invalid"));
                }
                let loop_next = code
                    .get(end_target.saturating_sub(1))
                    .ok_or_else(|| verify("bytecode loop target is invalid"))?;
                if !matches!(loop_next, Instruction::LoopNext { body_target } if *body_target == pc + 1)
                {
                    return Err(verify("bytecode loop structure is invalid"));
                }
                if loop_bodies
                    .insert(pc + 1, (*item_slot, *end_target))
                    .is_some()
                {
                    return Err(verify("bytecode loop body is ambiguous"));
                }
                let expected_slots = instruction_slots_in_region(
                    &code[pc + 1..end_target.saturating_sub(1)],
                    *item_slot,
                )?;
                if *body_slots != expected_slots {
                    return Err(verify("bytecode loop local metadata is invalid"));
                }
            }
            Instruction::LoopNext { body_target } => {
                if *body_target >= pc || !loop_bodies.contains_key(body_target) {
                    return Err(verify("bytecode loop backedge is invalid"));
                }
            }
            Instruction::Map {
                slot,
                item_slot,
                max_items,
                ..
            }
            | Instruction::Filter {
                slot,
                item_slot,
                max_items,
                ..
            } => {
                if slot == item_slot {
                    return Err(verify("bytecode collection local slots must differ"));
                }
                if *max_items == 0 || *max_items > limits.max_collection_items {
                    return Err(verify("bytecode collection bound is invalid"));
                }
            }
            Instruction::Reduce {
                slot,
                item_slot,
                accumulator_slot,
                max_items,
                ..
            } => {
                if slot == item_slot || slot == accumulator_slot || item_slot == accumulator_slot {
                    return Err(verify("bytecode reduce local slots must differ"));
                }
                if *max_items == 0 || *max_items > limits.max_collection_items {
                    return Err(verify("bytecode collection bound is invalid"));
                }
            }
            Instruction::Invoke {
                tool_id, call_site, ..
            }
            | Instruction::FanOut {
                tool_id, call_site, ..
            } => {
                verify_tool_id(tool_id)?;
                if !call_sites.insert(*call_site) {
                    return Err(verify("bytecode call sites must be unique"));
                }
                if let Instruction::FanOut {
                    max_calls,
                    slot,
                    item_slot,
                    ..
                } = instruction
                {
                    if slot == item_slot {
                        return Err(verify("bytecode fan-out local slots must differ"));
                    }
                    if *max_calls == 0 || *max_calls > limits.max_fanout {
                        return Err(verify("bytecode fan-out bound is invalid"));
                    }
                }
            }
            Instruction::Let { .. } | Instruction::Return { .. } => {}
        }
    }
    if code
        .len()
        .checked_add(expression_instructions)
        .filter(|count| *count <= limits.max_bytecode_instructions)
        .is_none()
    {
        return Err(resource("bytecode instruction limit exceeded"));
    }

    verify_definite_local_initialization(code, local_count, &loop_bodies, limits)?;
    Ok(())
}

fn verify_instruction_expressions(
    instruction: &Instruction,
    local_count: usize,
    limits: &SandboxLimits,
    constant_bytes: &mut usize,
) -> Result<usize, SandboxError> {
    let mut total = 0usize;
    for expression in instruction_expressions(instruction) {
        verify_expression(&expression.0, limits.max_operand_stack)?;
        total = total
            .checked_add(expression.0.len())
            .ok_or_else(|| resource("bytecode instruction limit exceeded"))?;
        for opcode in &expression.0 {
            match opcode {
                ExprInstruction::Constant(value) => {
                    verify_constant_domain(value)?;
                    let measurement = measure_value(value, limits)?;
                    add_constant_bytes(constant_bytes, measurement.retained, limits)?;
                }
                ExprInstruction::Path(pointer) => {
                    verify_pointer(pointer)?;
                    add_constant_bytes(
                        constant_bytes,
                        string_retained_bytes(pointer.capacity())?,
                        limits,
                    )?;
                }
                ExprInstruction::Array(count) if *count > limits.max_collection_items => {
                    return Err(resource("bytecode array size exceeds the effective limit"));
                }
                ExprInstruction::Object(keys) => {
                    if keys.len() > limits.max_collection_items {
                        return Err(resource("bytecode object size exceeds the effective limit"));
                    }
                    let mut unique = BTreeSet::new();
                    for key in keys {
                        if key.len() > MAX_ATOMIC_KEY_BYTES || !unique.insert(key.as_str()) {
                            return Err(verify("bytecode object keys are invalid"));
                        }
                        add_constant_bytes(
                            constant_bytes,
                            key_retained_bytes(key.capacity())?,
                            limits,
                        )?;
                    }
                }
                ExprInstruction::Load(slot) if *slot >= local_count => {
                    return Err(verify("bytecode local load slot is invalid"));
                }
                ExprInstruction::Load(_)
                | ExprInstruction::Array(_)
                | ExprInstruction::Binary(_)
                | ExprInstruction::Unary(_) => {}
            }
        }
    }
    Ok(total)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AbstractValue {
    minimum_serialized: usize,
    structure: Option<Rc<AbstractStructure>>,
}

#[derive(Debug, PartialEq, Eq)]
enum AbstractStructure {
    Array(Vec<AbstractValue>),
    Object(BTreeMap<String, AbstractValue>),
}

impl AbstractValue {
    const fn unknown() -> Self {
        Self {
            minimum_serialized: 1,
            structure: None,
        }
    }

    const fn scalar(minimum_serialized: usize) -> Self {
        Self {
            minimum_serialized,
            structure: None,
        }
    }

    fn joined(&self, other: &Self) -> Self {
        if self == other {
            self.clone()
        } else {
            Self::scalar(self.minimum_serialized.min(other.minimum_serialized))
        }
    }
}

/// Computes a conservative abstract value for one verified postfix
/// expression. Dynamic values use the smallest valid JSON value. Statically
/// constructed arrays and objects retain only bounded shape and size facts so
/// later path operations can select a precise child without retaining values.
fn analyze_expression(
    code: &[ExprInstruction],
    limits: &SandboxLimits,
    locals: &BTreeMap<usize, AbstractValue>,
) -> Result<AbstractValue, SandboxError> {
    let mut stack = Vec::new();
    stack
        .try_reserve(code.len().min(limits.max_operand_stack))
        .map_err(|_| resource("bytecode output analysis allocation failed"))?;
    for instruction in code {
        let abstract_value = match instruction {
            ExprInstruction::Constant(value) => {
                AbstractValue::scalar(measure_value(value, limits)?.serialized)
            }
            ExprInstruction::Load(slot) => locals
                .get(slot)
                .cloned()
                .ok_or_else(|| verify("bytecode output local is unavailable"))?,
            ExprInstruction::Path(pointer) => {
                let source = stack
                    .pop()
                    .ok_or_else(|| verify("bytecode output expression stack is invalid"))?;
                resolve_static_path(source, pointer)?.unwrap_or_else(AbstractValue::unknown)
            }
            ExprInstruction::Unary(_) => {
                stack
                    .pop()
                    .ok_or_else(|| verify("bytecode output expression stack is invalid"))?;
                AbstractValue::unknown()
            }
            ExprInstruction::Binary(_) => {
                stack
                    .pop()
                    .ok_or_else(|| verify("bytecode output expression stack is invalid"))?;
                stack
                    .pop()
                    .ok_or_else(|| verify("bytecode output expression stack is invalid"))?;
                AbstractValue::unknown()
            }
            ExprInstruction::Array(count) => {
                let mut bytes = checked_add(2, count.saturating_sub(1))?;
                let mut items = Vec::new();
                items
                    .try_reserve_exact(*count)
                    .map_err(|_| resource("bytecode output analysis allocation failed"))?;
                for _ in 0..*count {
                    let item = stack
                        .pop()
                        .ok_or_else(|| verify("bytecode output expression stack is invalid"))?;
                    bytes = checked_add(bytes, item.minimum_serialized)?;
                    items.push(item);
                }
                items.reverse();
                AbstractValue {
                    minimum_serialized: bytes,
                    structure: Some(Rc::new(AbstractStructure::Array(items))),
                }
            }
            ExprInstruction::Object(keys) => {
                let mut bytes = checked_add(2, keys.len().saturating_sub(1))?;
                let mut entries = BTreeMap::new();
                for key in keys.iter().rev() {
                    let value = stack
                        .pop()
                        .ok_or_else(|| verify("bytecode output expression stack is invalid"))?;
                    bytes = checked_add(bytes, serialized_string_len(key)?)?;
                    bytes = checked_add(bytes, 1)?;
                    bytes = checked_add(bytes, value.minimum_serialized)?;
                    entries.insert(key.clone(), value);
                }
                AbstractValue {
                    minimum_serialized: bytes,
                    structure: Some(Rc::new(AbstractStructure::Object(entries))),
                }
            }
        };
        stack.push(abstract_value);
    }
    if stack.len() != 1 {
        return Err(verify("bytecode output expression stack is invalid"));
    }
    stack
        .pop()
        .ok_or_else(|| verify("bytecode output expression stack is invalid"))
}

fn resolve_static_path(
    mut value: AbstractValue,
    pointer: &str,
) -> Result<Option<AbstractValue>, SandboxError> {
    if pointer.is_empty() {
        return Ok(Some(value));
    }
    for encoded in pointer[1..].split('/') {
        let segment = decode_pointer_segment(encoded)?;
        let Some(structure) = value.structure.as_deref() else {
            return Ok(None);
        };
        value = match structure {
            AbstractStructure::Object(entries) => match entries.get(&segment) {
                Some(selected) => selected.clone(),
                None => return Ok(None),
            },
            AbstractStructure::Array(items) => {
                let Some(index) = parse_pointer_index(&segment) else {
                    return Ok(None);
                };
                match items.get(index) {
                    Some(selected) => selected.clone(),
                    None => return Ok(None),
                }
            }
        };
    }
    Ok(Some(value))
}

fn decode_pointer_segment(encoded: &str) -> Result<String, SandboxError> {
    let mut decoded = String::new();
    decoded
        .try_reserve_exact(encoded.len())
        .map_err(|_| resource("bytecode output analysis allocation failed"))?;
    let mut characters = encoded.chars();
    while let Some(character) = characters.next() {
        if character == '~' {
            decoded.push(match characters.next() {
                Some('0') => '~',
                Some('1') => '/',
                _ => return Err(verify("bytecode JSON pointer escape is invalid")),
            });
        } else {
            decoded.push(character);
        }
    }
    Ok(decoded)
}

fn parse_pointer_index(segment: &str) -> Option<usize> {
    if segment.is_empty() || (segment.len() > 1 && segment.starts_with('0')) {
        return None;
    }
    let mut value = 0usize;
    for byte in segment.bytes() {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add((byte - b'0') as usize)?;
    }
    Some(value)
}

fn verify_constant_domain(value: &Value) -> Result<(), SandboxError> {
    match value {
        Value::Number(number) if number.as_i64().is_none() => {
            Err(verify("bytecode constants must be signed integers"))
        }
        Value::Array(_) | Value::Object(_) => Err(verify(
            "bytecode constants must be scalar language literals",
        )),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => Ok(()),
    }
}

fn verify_definite_local_initialization(
    code: &[Instruction],
    local_count: usize,
    loop_bodies: &BTreeMap<usize, (usize, usize)>,
    limits: &SandboxLimits,
) -> Result<(), SandboxError> {
    let mut loop_locals = BTreeMap::new();
    for (body_target, (_, end_target)) in loop_bodies {
        let mut locals = BTreeSet::new();
        for instruction in &code[*body_target..end_target.saturating_sub(1)] {
            for slot in instruction_slots(instruction) {
                locals.insert(slot);
            }
        }
        loop_locals.insert(*body_target, locals);
    }

    let mut states: Vec<Option<BTreeMap<usize, AbstractValue>>> = vec![None; code.len()];
    states[0] = Some(BTreeMap::new());
    let mut pending = vec![0usize];
    while let Some(pc) = pending.pop() {
        let available = states[pc]
            .clone()
            .ok_or_else(|| verify("bytecode verifier state is invalid"))?;
        let instruction = &code[pc];
        match instruction {
            Instruction::Map {
                slot, item_slot, ..
            }
            | Instruction::Filter {
                slot, item_slot, ..
            }
            | Instruction::FanOut {
                slot, item_slot, ..
            } => {
                if available.contains_key(item_slot) {
                    return Err(verify("bytecode scoped local aliases a live binding"));
                }
                if slot == item_slot {
                    return Err(verify("bytecode scoped local slots must be distinct"));
                }
            }
            Instruction::Reduce {
                slot,
                item_slot,
                accumulator_slot,
                ..
            } => {
                if available.contains_key(item_slot) || available.contains_key(accumulator_slot) {
                    return Err(verify("bytecode scoped local aliases a live binding"));
                }
                if slot == item_slot || slot == accumulator_slot || item_slot == accumulator_slot {
                    return Err(verify("bytecode scoped local slots must be distinct"));
                }
            }
            _ => {}
        }
        verify_instruction_loads(instruction, &available, local_count)?;

        let mut propagate =
            |target: usize, state: BTreeMap<usize, AbstractValue>| -> Result<(), SandboxError> {
                if target >= code.len() {
                    return Err(verify("every reachable control path must return"));
                }
                match &mut states[target] {
                    Some(existing) => {
                        let intersection = existing
                            .iter()
                            .filter_map(|(slot, existing_value)| {
                                state.get(slot).map(|incoming_value| {
                                    (*slot, existing_value.joined(incoming_value))
                                })
                            })
                            .collect::<BTreeMap<_, _>>();
                        if *existing != intersection {
                            *existing = intersection;
                            pending.push(target);
                        }
                    }
                    None => {
                        states[target] = Some(state);
                        pending.push(target);
                    }
                }
                Ok(())
            };

        match instruction {
            Instruction::Return { value } => {
                let abstract_value = analyze_expression(&value.0, limits, &available)?;
                if abstract_value.minimum_serialized > limits.max_return_bytes {
                    return Err(SandboxError::output_limit());
                }
            }
            Instruction::Branch { false_target, .. } => {
                propagate(pc + 1, available.clone())?;
                propagate(*false_target, available)?;
            }
            Instruction::Jump { target } => propagate(*target, available)?,
            Instruction::LoopStart {
                item_slot,
                end_target,
                ..
            } => {
                if available.contains_key(item_slot) {
                    return Err(verify("bytecode loop local is already initialized"));
                }
                let mut body = available.clone();
                body.insert(*item_slot, AbstractValue::unknown());
                propagate(pc + 1, body)?;
                propagate(*end_target, available)?;
            }
            Instruction::LoopNext { body_target } => {
                let (item_slot, _) = loop_bodies
                    .get(body_target)
                    .ok_or_else(|| verify("bytecode loop backedge is invalid"))?;
                let locals = loop_locals
                    .get(body_target)
                    .ok_or_else(|| verify("bytecode loop locals are invalid"))?;
                let mut next_body = available.clone();
                for slot in locals {
                    next_body.remove(slot);
                }
                next_body.insert(*item_slot, AbstractValue::unknown());
                let mut after_loop = next_body.clone();
                after_loop.remove(item_slot);
                propagate(*body_target, next_body)?;
                propagate(pc + 1, after_loop)?;
            }
            Instruction::Let { slot, value } => {
                if available.contains_key(slot) {
                    return Err(verify("bytecode immutable local is rebound"));
                }
                let abstract_value = analyze_expression(&value.0, limits, &available)?;
                let mut next = available;
                next.insert(*slot, abstract_value);
                propagate(pc + 1, next)?;
            }
            Instruction::Map { slot, .. }
            | Instruction::Filter { slot, .. }
            | Instruction::FanOut { slot, .. } => {
                if available.contains_key(slot) {
                    return Err(verify("bytecode immutable local is rebound"));
                }
                let mut next = available;
                // These instructions always bind a JSON array, which has a
                // two-byte serialized lower bound even when it is empty.
                next.insert(*slot, AbstractValue::scalar(2));
                propagate(pc + 1, next)?;
            }
            Instruction::Reduce { slot, .. } | Instruction::Invoke { slot, .. } => {
                if available.contains_key(slot) {
                    return Err(verify("bytecode immutable local is rebound"));
                }
                let mut next = available;
                // A reduction or tool response can produce any accepted JSON
                // value, whose conservative serialized lower bound is one.
                next.insert(*slot, AbstractValue::unknown());
                propagate(pc + 1, next)?;
            }
        }
    }
    Ok(())
}

fn verify_instruction_loads(
    instruction: &Instruction,
    available: &BTreeMap<usize, AbstractValue>,
    local_count: usize,
) -> Result<(), SandboxError> {
    let verify_loads = |expression: &ExprCode, scope: &BTreeMap<usize, AbstractValue>| {
        for opcode in &expression.0 {
            if let ExprInstruction::Load(slot) = opcode {
                if *slot >= local_count || !scope.contains_key(slot) {
                    return Err(verify(
                        "bytecode local load is unavailable on this control-flow path",
                    ));
                }
            }
        }
        Ok(())
    };
    match instruction {
        Instruction::Map {
            item_slot,
            collection,
            value,
            ..
        } => {
            verify_loads(collection, available)?;
            let mut scoped = available.clone();
            scoped.insert(*item_slot, AbstractValue::unknown());
            verify_loads(value, &scoped)
        }
        Instruction::Filter {
            item_slot,
            collection,
            predicate,
            ..
        } => {
            verify_loads(collection, available)?;
            let mut scoped = available.clone();
            scoped.insert(*item_slot, AbstractValue::unknown());
            verify_loads(predicate, &scoped)
        }
        Instruction::Reduce {
            item_slot,
            accumulator_slot,
            collection,
            initial,
            value,
            ..
        } => {
            verify_loads(collection, available)?;
            verify_loads(initial, available)?;
            let mut scoped = available.clone();
            scoped.insert(*item_slot, AbstractValue::unknown());
            scoped.insert(*accumulator_slot, AbstractValue::unknown());
            verify_loads(value, &scoped)
        }
        Instruction::FanOut {
            item_slot,
            collection,
            arguments,
            ..
        } => {
            verify_loads(collection, available)?;
            let mut scoped = available.clone();
            scoped.insert(*item_slot, AbstractValue::unknown());
            verify_loads(arguments, &scoped)
        }
        _ => {
            for expression in instruction_expressions(instruction) {
                verify_loads(expression, available)?;
            }
            Ok(())
        }
    }
}

fn verify_pointer(pointer: &str) -> Result<(), SandboxError> {
    if pointer.len() > 1024 || (!pointer.is_empty() && !pointer.starts_with('/')) {
        return Err(verify("bytecode JSON pointer is invalid"));
    }
    let bytes = pointer.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'~' {
            index += 1;
            if index == bytes.len() || !matches!(bytes[index], b'0' | b'1') {
                return Err(verify("bytecode JSON pointer escape is invalid"));
            }
        }
        index += 1;
    }
    Ok(())
}

fn verify_tool_id(tool_id: &str) -> Result<(), SandboxError> {
    if tool_id.is_empty() || tool_id.len() > MAX_ATOMIC_STRING_BYTES {
        return Err(verify("bytecode tool identifier is invalid"));
    }
    Ok(())
}

fn add_constant_bytes(
    total: &mut usize,
    bytes: usize,
    limits: &SandboxLimits,
) -> Result<(), SandboxError> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| resource("constant byte limit exceeded"))?;
    if *total > limits.max_constant_bytes {
        return Err(resource("constant byte limit exceeded"));
    }
    Ok(())
}

fn instruction_expressions(instruction: &Instruction) -> Vec<&ExprCode> {
    match instruction {
        Instruction::Let { value, .. } | Instruction::Return { value } => vec![value],
        Instruction::Branch { condition, .. } => vec![condition],
        Instruction::LoopStart { collection, .. } => vec![collection],
        Instruction::Map {
            collection, value, ..
        } => vec![collection, value],
        Instruction::Filter {
            collection,
            predicate,
            ..
        } => vec![collection, predicate],
        Instruction::Reduce {
            collection,
            initial,
            value,
            ..
        } => vec![collection, initial, value],
        Instruction::Invoke { arguments, .. } => vec![arguments],
        Instruction::FanOut {
            collection,
            arguments,
            ..
        } => vec![collection, arguments],
        Instruction::Jump { .. } | Instruction::LoopNext { .. } => Vec::new(),
    }
}

fn verify_expression(code: &[ExprInstruction], max_stack: usize) -> Result<(), SandboxError> {
    let mut depth = 0usize;
    for instruction in code {
        let (pops, pushes) = match instruction {
            ExprInstruction::Constant(_) | ExprInstruction::Load(_) => (0, 1),
            ExprInstruction::Path(_) | ExprInstruction::Unary(_) => (1, 1),
            ExprInstruction::Array(count) => (*count, 1),
            ExprInstruction::Object(keys) => (keys.len(), 1),
            ExprInstruction::Binary(_) => (2, 1),
        };
        if depth < pops {
            return Err(verify("expression bytecode stack underflow"));
        }
        depth = depth - pops + pushes;
        if depth > max_stack {
            return Err(resource("operand stack limit exceeded"));
        }
    }
    if depth != 1 {
        return Err(verify("expression bytecode must produce exactly one value"));
    }
    Ok(())
}

fn verify(message: &str) -> SandboxError {
    SandboxError::new(SandboxErrorCode::Verification, message)
}
fn resource(message: &str) -> SandboxError {
    SandboxError::new(SandboxErrorCode::ResourceLimit, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile(raw: &[u8]) -> Result<VerifiedProgram, SandboxError> {
        Program::from_json(raw, &SandboxLimits::default())?.compile(&SandboxLimits::default())
    }

    #[test]
    fn rejects_forward_reference_and_missing_return() {
        assert_eq!(compile(br#"{"version":1,"body":[{"kind":"return","value":{"kind":"variable","name":"x"}}]}"#).unwrap_err().code(), SandboxErrorCode::Verification);
        assert_eq!(
            compile(br#"{"version":1,"body":[]}"#).unwrap_err().code(),
            SandboxErrorCode::Verification
        );
    }

    #[test]
    fn compiler_output_is_private_and_verified() {
        let program = compile(br#"{"version":1,"body":[{"kind":"let","name":"x","value":{"kind":"integer","value":1}},{"kind":"return","value":{"kind":"variable","name":"x"}}]}"#).unwrap();
        assert_eq!(program.local_count, 1);
        assert_eq!(program.code.len(), 2);
    }

    #[test]
    fn verifier_accepts_empty_object_keys_with_bounded_accounting() {
        let code = vec![Instruction::Return {
            value: ExprCode(vec![
                ExprInstruction::Constant(Value::Null),
                ExprInstruction::Object(vec![String::new()]),
            ]),
        }];
        assert!(verify_bytecode(&code, 0, &SandboxLimits::default()).is_ok());
    }

    #[test]
    fn verifier_rejects_a_provably_oversized_dynamic_return() {
        let payload = "x\"\\\n".repeat(16);
        let dynamic_key = "effect\"\\";
        let source = serde_json::to_vec(&serde_json::json!({"version":1,"body":[
            {"kind":"invoke","name":"effect","tool_id":"write","arguments":{"kind":"object","entries":[]}},
            {"kind":"return","value":{"kind":"object","entries":[
                {"key":dynamic_key,"value":{"kind":"variable","name":"effect"}},
                {"key":"payload","value":{"kind":"string","value":payload}}
            ]}}
        ]}))
        .unwrap();
        let minimum_value = serde_json::json!({
            dynamic_key: 0,
            "payload": payload,
        });
        let minimum = measure_value(&minimum_value, &SandboxLimits::default())
            .unwrap()
            .serialized;

        for (max_return_bytes, accepted) in [(minimum - 1, false), (minimum, true)] {
            let limits = SandboxLimits {
                max_return_bytes,
                ..SandboxLimits::default()
            };
            let compiled =
                Program::from_json(&source, &limits).and_then(|program| program.compile(&limits));
            assert_eq!(compiled.is_ok(), accepted);
            if let Err(error) = compiled {
                assert_eq!(error.code(), SandboxErrorCode::ResourceLimit);
                assert!(error.is_output_limit());
                assert_eq!(error.message(), "output byte limit exceeded");
            }
        }
    }

    #[test]
    fn verifier_propagates_local_bounds_and_ignores_unreachable_returns() {
        let local_return = serde_json::to_vec(&serde_json::json!({"version":1,"body":[
            {"kind":"let","name":"payload","value":{"kind":"string","value":"x".repeat(256)}},
            {"kind":"invoke","name":"effect","tool_id":"write","arguments":{"kind":"object","entries":[]}},
            {"kind":"return","value":{"kind":"object","entries":[
                {"key":"effect","value":{"kind":"variable","name":"effect"}},
                {"key":"payload","value":{"kind":"variable","name":"payload"}}
            ]}}
        ]}))
        .unwrap();
        let selected_path_return = serde_json::to_vec(&serde_json::json!({"version":1,"body":[
            {"kind":"let","name":"payload_container","value":{"kind":"object","entries":[
                {"key":"large","value":{"kind":"string","value":"x".repeat(256)}},
                {"key":"small","value":{"kind":"integer","value":0}}
            ]}},
            {"kind":"invoke","name":"effect","tool_id":"write","arguments":{"kind":"object","entries":[]}},
            {"kind":"return","value":{"kind":"path","value":{"kind":"variable","name":"payload_container"},"pointer":"/large"}}
        ]}))
        .unwrap();
        let limits = SandboxLimits {
            max_return_bytes: 128,
            ..SandboxLimits::default()
        };
        for source in [local_return, selected_path_return] {
            let error = Program::from_json(&source, &limits)
                .and_then(|program| program.compile(&limits))
                .unwrap_err();
            assert!(error.is_output_limit());
        }

        let unreachable = serde_json::to_vec(&serde_json::json!({"version":1,"body":[
            {"kind":"return","value":{"kind":"integer","value":0}},
            {"kind":"return","value":{"kind":"string","value":"x".repeat(256)}}
        ]}))
        .unwrap();
        let limits = SandboxLimits {
            max_return_bytes: 1,
            ..SandboxLimits::default()
        };
        Program::from_json(&unreachable, &limits)
            .and_then(|program| program.compile(&limits))
            .unwrap();
    }

    #[test]
    fn path_result_does_not_inherit_its_large_operand_lower_bound() {
        let source = serde_json::to_vec(&serde_json::json!({"version":1,"body":[
            {"kind":"return","value":{"kind":"path","value":{"kind":"object","entries":[
                {"key":"large","value":{"kind":"string","value":"x".repeat(4096)}},
                {"key":"selected","value":{"kind":"integer","value":0}}
            ]},"pointer":"/selected"}}
        ]}))
        .unwrap();
        let limits = SandboxLimits {
            max_return_bytes: 1,
            ..SandboxLimits::default()
        };

        Program::from_json(&source, &limits)
            .and_then(|program| program.compile(&limits))
            .unwrap();
    }

    #[test]
    fn branch_and_loop_bindings_do_not_escape_lexical_scope() {
        for raw in [
            br#"{"version":1,"body":[{"kind":"branch","condition":{"kind":"boolean","value":true},"then_body":[{"kind":"let","name":"x","value":{"kind":"integer","value":1}}]},{"kind":"return","value":{"kind":"variable","name":"x"}}]}"#.as_slice(),
            br#"{"version":1,"body":[{"kind":"for_each","item":"i","collection":{"kind":"array","items":[{"kind":"integer","value":1}]},"max_iterations":1,"body":[]},{"kind":"return","value":{"kind":"variable","name":"i"}}]}"#.as_slice(),
        ] {
            assert_eq!(compile(raw).unwrap_err().code(), SandboxErrorCode::Verification);
        }
    }

    #[test]
    fn scoped_loop_and_reduce_slots_consume_local_budget() {
        let parsed = Program::from_json(
            br#"{"version":1,"body":[{"kind":"reduce","name":"total","item":"item","accumulator":"acc","collection":{"kind":"array","items":[]},"max_items":1,"initial":{"kind":"integer","value":0},"value":{"kind":"variable","name":"acc"}},{"kind":"return","value":{"kind":"variable","name":"total"}}]}"#,
            &SandboxLimits::default(),
        )
        .unwrap();
        let limits = SandboxLimits {
            max_locals: 2,
            ..SandboxLimits::default()
        };
        assert_eq!(
            parsed.compile(&limits).unwrap_err().code(),
            SandboxErrorCode::ResourceLimit
        );
    }

    #[test]
    fn rejects_control_path_that_can_fall_through() {
        assert_eq!(
            compile(br#"{"version":1,"body":[{"kind":"branch","condition":{"kind":"boolean","value":true},"then_body":[{"kind":"return","value":{"kind":"null"}}],"else_body":[]}]}"#)
                .unwrap_err()
                .code(),
            SandboxErrorCode::Verification
        );
    }

    #[test]
    fn independent_verifier_rejects_private_tampering_without_public_bytecode() {
        let limits = SandboxLimits::default();
        let invalid_load = vec![Instruction::Return {
            value: ExprCode(vec![ExprInstruction::Load(1)]),
        }];
        assert_eq!(
            verify_bytecode(&invalid_load, 1, &limits)
                .unwrap_err()
                .code(),
            SandboxErrorCode::Verification
        );

        let arguments = ExprCode(vec![ExprInstruction::Constant(Value::Object(
            serde_json::Map::new(),
        ))]);
        let duplicate_sites = vec![
            Instruction::Invoke {
                slot: 0,
                tool_id: "read".into(),
                arguments: arguments.clone(),
                call_site: 0,
            },
            Instruction::Invoke {
                slot: 1,
                tool_id: "read".into(),
                arguments,
                call_site: 0,
            },
            Instruction::Return {
                value: ExprCode(vec![ExprInstruction::Load(1)]),
            },
        ];
        assert_eq!(
            verify_bytecode(&duplicate_sites, 2, &limits)
                .unwrap_err()
                .code(),
            SandboxErrorCode::Verification
        );

        let invalid_destination = vec![Instruction::Let {
            slot: 1,
            value: ExprCode(vec![ExprInstruction::Constant(Value::Null)]),
        }];
        assert_eq!(
            verify_bytecode(&invalid_destination, 1, &limits)
                .unwrap_err()
                .code(),
            SandboxErrorCode::Verification
        );

        let invalid_jump = vec![Instruction::Jump { target: 0 }];
        assert_eq!(
            verify_bytecode(&invalid_jump, 0, &limits)
                .unwrap_err()
                .code(),
            SandboxErrorCode::Verification
        );
    }

    #[test]
    fn independent_verifier_rejects_private_constant_stack_and_control_tampering() {
        let limits = SandboxLimits::default();
        let invalid_constant_type = vec![Instruction::Return {
            value: ExprCode(vec![ExprInstruction::Constant(serde_json::json!(1.5))]),
        }];
        assert_eq!(
            verify_bytecode(&invalid_constant_type, 0, &limits)
                .unwrap_err()
                .code(),
            SandboxErrorCode::Verification
        );

        let oversized_constant = vec![Instruction::Return {
            value: ExprCode(vec![ExprInstruction::Constant(serde_json::json!("abc"))]),
        }];
        assert_eq!(
            verify_bytecode(
                &oversized_constant,
                0,
                &SandboxLimits {
                    max_constant_bytes: 2,
                    ..SandboxLimits::default()
                },
            )
            .unwrap_err()
            .code(),
            SandboxErrorCode::ResourceLimit
        );

        let stack_underflow = vec![Instruction::Return {
            value: ExprCode(vec![ExprInstruction::Binary(BinaryOperator::Add)]),
        }];
        assert_eq!(
            verify_bytecode(&stack_underflow, 0, &limits)
                .unwrap_err()
                .code(),
            SandboxErrorCode::Verification
        );

        let stack_overflow = vec![Instruction::Return {
            value: ExprCode(vec![
                ExprInstruction::Constant(Value::Null),
                ExprInstruction::Constant(Value::Null),
                ExprInstruction::Array(2),
            ]),
        }];
        assert_eq!(
            verify_bytecode(
                &stack_overflow,
                0,
                &SandboxLimits {
                    max_operand_stack: 1,
                    ..SandboxLimits::default()
                },
            )
            .unwrap_err()
            .code(),
            SandboxErrorCode::ResourceLimit
        );

        let no_terminal_return = vec![Instruction::Let {
            slot: 0,
            value: ExprCode(vec![ExprInstruction::Constant(Value::Null)]),
        }];
        assert_eq!(
            verify_bytecode(&no_terminal_return, 1, &limits)
                .unwrap_err()
                .code(),
            SandboxErrorCode::Verification
        );

        let malformed_loop = vec![
            Instruction::LoopStart {
                collection: ExprCode(vec![ExprInstruction::Constant(Value::Array(Vec::new()))]),
                item_slot: 0,
                max_iterations: 1,
                end_target: 2,
                body_slots: Vec::new(),
            },
            Instruction::Return {
                value: ExprCode(vec![ExprInstruction::Constant(Value::Null)]),
            },
        ];
        assert_eq!(
            verify_bytecode(&malformed_loop, 1, &limits)
                .unwrap_err()
                .code(),
            SandboxErrorCode::Verification
        );
    }

    #[test]
    fn verifier_rejects_every_out_of_range_scoped_slot_before_vm_admission() {
        let limits = SandboxLimits::default();
        let empty = || ExprCode(vec![ExprInstruction::Array(0)]);
        let null = || ExprCode(vec![ExprInstruction::Constant(Value::Null)]);
        let programs = [
            vec![Instruction::LoopStart {
                collection: empty(),
                item_slot: 1,
                max_iterations: 1,
                end_target: 2,
                body_slots: Vec::new(),
            }],
            vec![Instruction::Map {
                slot: 0,
                item_slot: 1,
                collection: empty(),
                max_items: 1,
                value: null(),
            }],
            vec![Instruction::Filter {
                slot: 0,
                item_slot: 1,
                collection: empty(),
                max_items: 1,
                predicate: null(),
            }],
            vec![Instruction::FanOut {
                slot: 0,
                tool_id: "read".into(),
                item_slot: 1,
                collection: empty(),
                max_calls: 1,
                arguments: null(),
                call_site: 0,
            }],
            vec![Instruction::Reduce {
                slot: 0,
                item_slot: 1,
                accumulator_slot: 2,
                collection: empty(),
                max_items: 1,
                initial: null(),
                value: null(),
            }],
        ];
        for (index, code) in programs.iter().enumerate() {
            let local_count = if index == programs.len() - 1 { 2 } else { 1 };
            let error = verify_bytecode(code, local_count, &limits).unwrap_err();
            assert_eq!(error.code(), SandboxErrorCode::Verification);
            assert_eq!(error.message(), "bytecode local slot is invalid");
        }
    }

    #[test]
    fn verifier_rejects_scoped_slot_aliases_for_every_collection_instruction() {
        let limits = SandboxLimits::default();
        let null = || ExprCode(vec![ExprInstruction::Constant(Value::Null)]);
        let empty = || ExprCode(vec![ExprInstruction::Array(0)]);
        let prefix = || Instruction::Let {
            slot: 0,
            value: null(),
        };
        let return_slot = |slot| Instruction::Return {
            value: ExprCode(vec![ExprInstruction::Load(slot)]),
        };
        let programs = [
            vec![
                prefix(),
                Instruction::Map {
                    slot: 1,
                    item_slot: 0,
                    collection: empty(),
                    max_items: 1,
                    value: null(),
                },
                return_slot(1),
            ],
            vec![
                prefix(),
                Instruction::Filter {
                    slot: 1,
                    item_slot: 0,
                    collection: empty(),
                    max_items: 1,
                    predicate: null(),
                },
                return_slot(1),
            ],
            vec![
                prefix(),
                Instruction::FanOut {
                    slot: 1,
                    tool_id: "read".into(),
                    item_slot: 0,
                    collection: empty(),
                    max_calls: 1,
                    arguments: null(),
                    call_site: 0,
                },
                return_slot(1),
            ],
            vec![
                prefix(),
                Instruction::Reduce {
                    slot: 3,
                    item_slot: 0,
                    accumulator_slot: 2,
                    collection: empty(),
                    max_items: 1,
                    initial: null(),
                    value: null(),
                },
                return_slot(3),
            ],
        ];
        for code in programs {
            let error = verify_bytecode(&code, 4, &limits).unwrap_err();
            assert_eq!(error.code(), SandboxErrorCode::Verification);
            assert_eq!(
                error.message(),
                "bytecode scoped local aliases a live binding"
            );
        }

        let reduce_aliases_itself = vec![
            Instruction::Reduce {
                slot: 1,
                item_slot: 0,
                accumulator_slot: 0,
                collection: empty(),
                max_items: 1,
                initial: null(),
                value: null(),
            },
            return_slot(1),
        ];
        let error = verify_bytecode(&reduce_aliases_itself, 2, &limits).unwrap_err();
        assert_eq!(error.code(), SandboxErrorCode::Verification);
        assert_eq!(error.message(), "bytecode reduce local slots must differ");
    }

    #[test]
    fn verified_program_debug_output_never_contains_constants() {
        let program = compile(br#"{"version":1,"body":[{"kind":"return","value":{"kind":"string","value":"CANARY_SECRET"}}]}"#).unwrap();
        let debug = alloc::format!("{program:?}");
        assert!(!debug.contains("CANARY_SECRET"));
        assert!(debug.contains("instruction_count"));
    }
}
