use crate::{
    BinaryOperator, Expression, Program, SandboxError, SandboxErrorCode, SandboxLimits, Statement,
    UnaryOperator, PROGRAM_VERSION_V1,
};
use alloc::{collections::BTreeMap, string::String, vec::Vec};
use serde_json::Value;

/// An opaque, verified executable. Its bytecode is private and non-serializable.
#[derive(Clone, Debug)]
pub struct VerifiedProgram {
    pub(crate) code: Vec<Instruction>,
    pub(crate) local_count: usize,
    pub(crate) limits: SandboxLimits,
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
    if program.version != PROGRAM_VERSION_V1 {
        return Err(verify("unsupported program version"));
    }
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
    compiler.verify()?;
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
                    });
                    let body_target = self.code.len();
                    let mut body_environment = environment.clone();
                    body_environment.insert(item.clone(), item_slot);
                    self.compile_block(body, &mut body_environment, depth + 1)?;
                    self.reserve_instruction()?;
                    self.code.push(Instruction::LoopNext { body_target });
                    let end_target = self.code.len();
                    match &mut self.code[start_index] {
                        Instruction::LoopStart {
                            end_target: target, ..
                        } => *target = end_target,
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

    fn verify(&self) -> Result<(), SandboxError> {
        if self.code.len() + self.expression_instructions > self.limits.max_bytecode_instructions {
            return Err(resource("bytecode instruction limit exceeded"));
        }
        for instruction in &self.code {
            match instruction {
                Instruction::Branch { false_target, .. }
                | Instruction::Jump {
                    target: false_target,
                } if *false_target > self.code.len() => {
                    return Err(verify("bytecode jump target is invalid"))
                }
                Instruction::LoopStart { end_target, .. } if *end_target > self.code.len() => {
                    return Err(verify("bytecode loop target is invalid"))
                }
                Instruction::LoopNext { body_target } if *body_target >= self.code.len() => {
                    return Err(verify("bytecode loop backedge is invalid"))
                }
                _ => {}
            }
        }
        self.verify_all_paths_return()?;
        Ok(())
    }

    fn verify_all_paths_return(&self) -> Result<(), SandboxError> {
        let mut pending = Vec::from([0usize]);
        let mut visited = alloc::collections::BTreeSet::new();
        while let Some(pc) = pending.pop() {
            if pc == self.code.len() {
                return Err(verify("every reachable control path must return"));
            }
            if pc > self.code.len() || !visited.insert(pc) {
                continue;
            }
            match &self.code[pc] {
                Instruction::Return { .. } => {}
                Instruction::Branch { false_target, .. } => {
                    pending.push(pc + 1);
                    pending.push(*false_target);
                }
                Instruction::Jump { target } => pending.push(*target),
                Instruction::LoopStart { end_target, .. } => {
                    pending.push(pc + 1);
                    pending.push(*end_target);
                }
                Instruction::LoopNext { body_target } => {
                    pending.push(*body_target);
                    pending.push(pc + 1);
                }
                _ => pending.push(pc + 1),
            }
        }
        Ok(())
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
}
