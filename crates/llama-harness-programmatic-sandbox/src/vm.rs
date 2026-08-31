use crate::{
    compiler::{ExprCode, ExprInstruction, Instruction, VerifiedProgram},
    BinaryOperator, SandboxError, SandboxErrorCode, UnaryOperator,
};
use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};
use serde_json::{Map, Number, Value};

/// Host-supplied identifier that scopes resume tokens to one live execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExecutionId(pub u64);

/// One inert, statically named tool request yielded by the sandbox.
#[derive(Clone, PartialEq)]
pub struct ToolRequest {
    /// Host execution identity.
    pub execution_id: ExecutionId,
    /// Host-assigned repair/attempt ordinal.
    pub program_attempt: u32,
    /// Stable static call-site ordinal in the verified program.
    pub call_site: u32,
    /// Stable dynamic occurrence ordinal in this execution.
    pub dynamic_ordinal: u64,
    /// Static tool identifier embedded in the verified program.
    pub tool_id: String,
    /// Fully evaluated owned JSON arguments.
    pub arguments: Value,
}

impl core::fmt::Debug for ToolRequest {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ToolRequest")
            .field("execution_id", &self.execution_id)
            .field("program_attempt", &self.program_attempt)
            .field("call_site", &self.call_site)
            .field("dynamic_ordinal", &self.dynamic_ordinal)
            .field("tool_id", &"<redacted>")
            .field("arguments", &"<redacted>")
            .finish()
    }
}

/// One ordered batch of inert tool requests.
#[derive(Clone, PartialEq)]
pub struct ToolBatch {
    calls: Vec<ToolRequest>,
    read_only_fan_out: bool,
}

impl core::fmt::Debug for ToolBatch {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ToolBatch")
            .field("call_count", &self.calls.len())
            .field("read_only_fan_out", &self.read_only_fan_out)
            .finish()
    }
}

impl ToolBatch {
    /// Returns calls in deterministic source-collection order.
    pub fn calls(&self) -> &[ToolRequest] {
        &self.calls
    }

    /// Whether the program requested read-only fan-out semantics.
    ///
    /// The host must still independently verify each tool is read-only and
    /// parallel-safe before executing calls concurrently.
    pub const fn requests_read_only_fan_out(&self) -> bool {
        self.read_only_fan_out
    }
}

/// One host-produced response corresponding to a yielded request occurrence.
#[derive(Clone, PartialEq)]
pub struct ToolResponse {
    /// Static call-site ordinal copied from the request.
    pub call_site: u32,
    /// Dynamic occurrence ordinal copied from the request.
    pub dynamic_ordinal: u64,
    /// Semantic tool success flag.
    pub ok: bool,
    /// Owned JSON output. It remains inert data inside the VM.
    pub output: Value,
}

impl core::fmt::Debug for ToolResponse {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ToolResponse")
            .field("call_site", &self.call_site)
            .field("dynamic_ordinal", &self.dynamic_ordinal)
            .field("ok", &self.ok)
            .field("output", &"<redacted>")
            .finish()
    }
}

impl ToolResponse {
    /// Creates a successful response.
    pub fn success(request: &ToolRequest, output: Value) -> Self {
        Self {
            call_site: request.call_site,
            dynamic_ordinal: request.dynamic_ordinal,
            ok: true,
            output,
        }
    }

    /// Creates a failed response without accepting an executable error channel.
    pub fn failure(request: &ToolRequest) -> Self {
        Self {
            call_site: request.call_site,
            dynamic_ordinal: request.dynamic_ordinal,
            ok: false,
            output: Value::Null,
        }
    }
}

/// Opaque, single-use proof required to resume one suspended execution.
#[derive(Debug, PartialEq, Eq)]
pub struct ResumeToken {
    execution_id: ExecutionId,
    yield_ordinal: u32,
}

/// Result of one bounded scheduling slice.
#[derive(PartialEq)]
#[non_exhaustive]
pub enum StepOutcome {
    /// The slice budget was consumed without completing or yielding a tool batch.
    Sliced,
    /// Execution suspended with an inert batch and a single-use resume token.
    Yielded {
        /// Requests requiring host authorization and execution.
        batch: ToolBatch,
        /// Token consumed by [`Execution::resume`].
        resume: ResumeToken,
    },
    /// Execution returned a final owned JSON value.
    Complete(Value),
}

impl core::fmt::Debug for StepOutcome {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Sliced => formatter.write_str("Sliced"),
            Self::Yielded { batch, resume } => formatter
                .debug_struct("Yielded")
                .field("batch", batch)
                .field("resume", resume)
                .finish(),
            Self::Complete(_) => formatter.write_str("Complete(<redacted>)"),
        }
    }
}

/// Aggregate, value-free execution counters safe for host telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutionMetrics {
    /// Fuel charged before deterministic work.
    pub fuel_used: u64,
    /// Number of suspended tool batches.
    pub yields: u32,
    /// Number of branch decisions executed.
    pub branches: u64,
    /// Number of bounded loop iterations entered.
    pub loop_iterations: u64,
    /// Number of read-only fan-out batches yielded.
    pub fanout_batches: u32,
    /// Conservatively retained value bytes.
    pub retained_bytes: usize,
    /// Cumulative value bytes charged.
    pub cumulative_bytes: usize,
}

struct LoopFrame {
    items: Vec<Value>,
    index: usize,
    item_slot: usize,
    body_target: usize,
    body_slots: Vec<usize>,
}

enum Work {
    Map {
        slot: usize,
        item_slot: usize,
        items: Vec<Value>,
        index: usize,
        value: ExprCode,
        output: Vec<Value>,
    },
    Filter {
        slot: usize,
        item_slot: usize,
        items: Vec<Value>,
        index: usize,
        predicate: ExprCode,
        output: Vec<Value>,
    },
    Reduce {
        slot: usize,
        item_slot: usize,
        accumulator_slot: usize,
        items: Vec<Value>,
        index: usize,
        value: ExprCode,
        accumulator: Value,
    },
    FanOut {
        slot: usize,
        tool_id: String,
        item_slot: usize,
        items: Vec<Value>,
        index: usize,
        arguments: ExprCode,
        call_site: u32,
        calls: Vec<ToolRequest>,
    },
}

struct Pending {
    yield_ordinal: u32,
    slot: usize,
    requests: Vec<(u32, u64)>,
    fan_out: bool,
}

/// One live deterministic execution of a verified program.
pub struct Execution {
    program: VerifiedProgram,
    execution_id: ExecutionId,
    program_attempt: u32,
    pc: usize,
    locals: Vec<Option<Value>>,
    loops: Vec<LoopFrame>,
    work: Option<Work>,
    pending: Option<Pending>,
    fuel_used: u64,
    yields: u32,
    dynamic_ordinal: u64,
    live_bytes: usize,
    cumulative_bytes: usize,
    branches: u64,
    loop_iterations: u64,
    fanout_batches: u32,
    terminal: bool,
}

impl Execution {
    /// Starts a fresh execution. The host must keep `execution_id` unique among
    /// concurrently live executions and stable across this execution's resumes.
    pub fn new(program: VerifiedProgram, execution_id: ExecutionId) -> Result<Self, SandboxError> {
        Self::with_attempt(program, execution_id, 0)
    }

    /// Starts a fresh execution with an explicit host repair/attempt ordinal.
    pub fn with_attempt(
        program: VerifiedProgram,
        execution_id: ExecutionId,
        program_attempt: u32,
    ) -> Result<Self, SandboxError> {
        let mut locals = Vec::new();
        locals
            .try_reserve_exact(program.local_count)
            .map_err(|_| resource("local allocation failed"))?;
        locals.resize_with(program.local_count, || None);
        Ok(Self {
            program,
            execution_id,
            program_attempt,
            pc: 0,
            locals,
            loops: Vec::new(),
            work: None,
            pending: None,
            fuel_used: 0,
            yields: 0,
            dynamic_ordinal: 0,
            live_bytes: 0,
            cumulative_bytes: 0,
            branches: 0,
            loop_iterations: 0,
            fanout_batches: 0,
            terminal: false,
        })
    }

    /// Returns aggregate value-free counters for this live execution.
    pub const fn metrics(&self) -> ExecutionMetrics {
        ExecutionMetrics {
            fuel_used: self.fuel_used,
            yields: self.yields,
            branches: self.branches,
            loop_iterations: self.loop_iterations,
            fanout_batches: self.fanout_batches,
            retained_bytes: self.live_bytes,
            cumulative_bytes: self.cumulative_bytes,
        }
    }

    /// Advances execution by at most `slice_fuel` deterministic work units.
    pub fn step(&mut self, slice_fuel: u64) -> Result<StepOutcome, SandboxError> {
        if self.terminal {
            return Err(execution("execution is already terminal"));
        }
        if self.pending.is_some() {
            return Err(resume_error("execution must be resumed before stepping"));
        }
        if slice_fuel == 0 || slice_fuel > self.program.limits.max_slice_fuel {
            return Err(resource(
                "slice fuel must be nonzero and within the effective limit",
            ));
        }
        let mut remaining = slice_fuel;
        loop {
            if self.work.is_some() {
                if !self.advance_work(&mut remaining)? {
                    return Ok(StepOutcome::Sliced);
                }
                if let Some(outcome) = self.finish_fanout_if_ready()? {
                    return Ok(outcome);
                }
                continue;
            }
            if self.pc >= self.program.code.len() {
                self.terminal = true;
                return Err(execution("verified program reached no-return state"));
            }
            let instruction = self.program.code[self.pc].clone();
            let minimum = instruction_minimum_cost(&instruction);
            if !self.charge(&mut remaining, minimum)? {
                return Ok(StepOutcome::Sliced);
            }
            match instruction {
                Instruction::Let { slot, value } => {
                    let result = self.eval(&value)?;
                    self.store(slot, result)?;
                    self.pc += 1;
                }
                Instruction::Branch {
                    condition,
                    false_target,
                } => {
                    self.branches = self.branches.saturating_add(1);
                    let condition = expect_bool(self.eval(&condition)?, "branch condition")?;
                    self.pc = if condition { self.pc + 1 } else { false_target };
                }
                Instruction::Jump { target } => self.pc = target,
                Instruction::LoopStart {
                    collection,
                    item_slot,
                    max_iterations,
                    end_target,
                } => {
                    let items = expect_array(self.eval(&collection)?, "loop collection")?;
                    if items.len() > max_iterations {
                        return Err(resource("loop iteration limit exceeded"));
                    }
                    if items.is_empty() {
                        self.pc = end_target;
                        continue;
                    }
                    self.precharge_bytes(serialized_array_len(&items)?)?;
                    self.precharge_value(&items[0])?;
                    self.locals[item_slot] = Some(items[0].clone());
                    self.loop_iterations = self.loop_iterations.saturating_add(1);
                    self.loops
                        .try_reserve(1)
                        .map_err(|_| resource("control stack allocation failed"))?;
                    if self.loops.len() >= self.program.limits.max_control_stack {
                        return Err(resource("control stack limit exceeded"));
                    }
                    let body_slots = loop_body_slots(&self.program.code, self.pc + 1, end_target)?;
                    self.loops.push(LoopFrame {
                        items,
                        index: 0,
                        item_slot,
                        body_target: self.pc + 1,
                        body_slots,
                    });
                    self.pc += 1;
                }
                Instruction::LoopNext { body_target } => {
                    let next = {
                        let frame = self
                            .loops
                            .last_mut()
                            .ok_or_else(|| execution("loop stack underflow"))?;
                        if frame.body_target != body_target {
                            return Err(execution("loop frame target mismatch"));
                        }
                        frame.index += 1;
                        (frame.index < frame.items.len()).then(|| {
                            (
                                frame.item_slot,
                                frame.items[frame.index].clone(),
                                frame.body_slots.clone(),
                            )
                        })
                    };
                    if let Some((item_slot, item, body_slots)) = next {
                        clear_loop_body_slots(&mut self.locals, &body_slots);
                        self.loop_iterations = self.loop_iterations.saturating_add(1);
                        self.precharge_value(&item)?;
                        self.locals[item_slot] = Some(item);
                        self.pc = body_target;
                    } else {
                        let frame = self
                            .loops
                            .pop()
                            .ok_or_else(|| execution("loop stack underflow"))?;
                        clear_loop_body_slots(&mut self.locals, &frame.body_slots);
                        self.locals[frame.item_slot] = None;
                        self.pc += 1;
                    }
                }
                Instruction::Map {
                    slot,
                    item_slot,
                    collection,
                    max_items,
                    value,
                } => {
                    let items = expect_array(self.eval(&collection)?, "map collection")?;
                    validate_items(
                        items.len(),
                        max_items,
                        self.program.limits.max_collection_items,
                    )?;
                    self.precharge_bytes(serialized_array_syntax_len(items.len())?)?;
                    self.precharge_bytes(serialized_array_len(&items)?)?;
                    let mut output = Vec::new();
                    output
                        .try_reserve(items.len())
                        .map_err(|_| resource("collection allocation failed"))?;
                    self.work = Some(Work::Map {
                        slot,
                        item_slot,
                        items,
                        index: 0,
                        value,
                        output,
                    });
                }
                Instruction::Filter {
                    slot,
                    item_slot,
                    collection,
                    max_items,
                    predicate,
                } => {
                    let items = expect_array(self.eval(&collection)?, "filter collection")?;
                    validate_items(
                        items.len(),
                        max_items,
                        self.program.limits.max_collection_items,
                    )?;
                    // Reserve the worst-case serialized array framing before the
                    // output vector grows. A filtered result may retain fewer
                    // elements, so this is deliberately conservative.
                    self.precharge_bytes(serialized_array_syntax_len(items.len())?)?;
                    self.precharge_bytes(serialized_array_len(&items)?)?;
                    let mut output = Vec::new();
                    output
                        .try_reserve(items.len())
                        .map_err(|_| resource("collection allocation failed"))?;
                    self.work = Some(Work::Filter {
                        slot,
                        item_slot,
                        items,
                        index: 0,
                        predicate,
                        output,
                    });
                }
                Instruction::Reduce {
                    slot,
                    item_slot,
                    accumulator_slot,
                    collection,
                    max_items,
                    initial,
                    value,
                } => {
                    let items = expect_array(self.eval(&collection)?, "reduce collection")?;
                    validate_items(
                        items.len(),
                        max_items,
                        self.program.limits.max_collection_items,
                    )?;
                    self.precharge_bytes(serialized_array_len(&items)?)?;
                    let accumulator = self.eval(&initial)?;
                    self.precharge_value(&accumulator)?;
                    self.work = Some(Work::Reduce {
                        slot,
                        item_slot,
                        accumulator_slot,
                        items,
                        index: 0,
                        value,
                        accumulator,
                    });
                }
                Instruction::Invoke {
                    slot,
                    tool_id,
                    arguments,
                    call_site,
                } => {
                    let arguments = self.eval(&arguments)?;
                    require_object(&arguments)?;
                    let request = self.request(tool_id, call_site, arguments)?;
                    return self.suspend(slot, vec![request], false);
                }
                Instruction::FanOut {
                    slot,
                    tool_id,
                    item_slot,
                    collection,
                    max_calls,
                    arguments,
                    call_site,
                } => {
                    let items = expect_array(self.eval(&collection)?, "fan-out collection")?;
                    validate_items(items.len(), max_calls, self.program.limits.max_fanout)?;
                    self.precharge_bytes(serialized_array_len(&items)?)?;
                    let mut calls = Vec::new();
                    calls
                        .try_reserve(items.len())
                        .map_err(|_| resource("fan-out allocation failed"))?;
                    self.work = Some(Work::FanOut {
                        slot,
                        tool_id,
                        item_slot,
                        items,
                        index: 0,
                        arguments,
                        call_site,
                        calls,
                    });
                }
                Instruction::Return { value } => {
                    let output = self.eval(&value)?;
                    let serialized_bytes = serialized_value_len(&output)?;
                    if serialized_bytes > self.program.limits.max_output_bytes {
                        return Err(resource("output byte limit exceeded"));
                    }
                    self.precharge_bytes(serialized_bytes)?;
                    self.terminal = true;
                    return Ok(StepOutcome::Complete(output));
                }
            }
        }
    }

    /// Resumes exactly the currently suspended yield. The token is consumed.
    pub fn resume(
        &mut self,
        token: ResumeToken,
        responses: Vec<ToolResponse>,
    ) -> Result<(), SandboxError> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| resume_error("execution is not suspended"))?;
        if token.execution_id != self.execution_id || token.yield_ordinal != pending.yield_ordinal {
            self.pending = Some(pending);
            return Err(resume_error(
                "resume token does not match the suspended execution",
            ));
        }
        if responses.len() != pending.requests.len() {
            self.pending = Some(pending);
            return Err(resume_error(
                "response count does not match the yielded batch",
            ));
        }
        if !pending.fan_out && responses.len() != 1 {
            self.pending = Some(pending);
            return Err(resume_error("single-call response is missing"));
        }
        if responses
            .iter()
            .zip(pending.requests.iter())
            .any(|(response, expected)| (response.call_site, response.dynamic_ordinal) != *expected)
        {
            self.pending = Some(pending);
            return Err(resume_error(
                "response identity does not match yielded request order",
            ));
        }

        let mut values = Vec::new();
        if pending.fan_out {
            self.precharge_bytes(serialized_array_syntax_len(responses.len())?)?;
            values
                .try_reserve(responses.len())
                .map_err(|_| resource("response allocation failed"))?;
        }
        for (response, expected) in responses.into_iter().zip(pending.requests.iter()) {
            debug_assert_eq!(
                (response.call_site, response.dynamic_ordinal),
                *expected,
                "response identity was prevalidated"
            );
            self.precharge_value(&response.output)?;
            self.precharge_bytes(response_object_syntax_len(response.ok)?)?;
            let mut object = Map::new();
            object.insert("ok".into(), Value::Bool(response.ok));
            object.insert("output".into(), response.output);
            if pending.fan_out {
                values.push(Value::Object(object));
            } else {
                self.store_precharged(pending.slot, Value::Object(object))?;
            }
        }
        if pending.fan_out {
            self.store_precharged(pending.slot, Value::Array(values))?;
        }
        self.pc += 1;
        Ok(())
    }

    fn advance_work(&mut self, remaining: &mut u64) -> Result<bool, SandboxError> {
        let mut work = self
            .work
            .take()
            .ok_or_else(|| execution("work state is missing"))?;
        let mut advanced = true;
        let done = match &mut work {
            Work::Map {
                slot,
                item_slot,
                items,
                index,
                value,
                output,
            } => {
                if *index == items.len() {
                    self.store_precharged(*slot, Value::Array(core::mem::take(output)))?;
                    self.locals[*item_slot] = None;
                    self.pc += 1;
                    true
                } else if !self.charge(remaining, expression_cost(value))? {
                    advanced = false;
                    false
                } else {
                    self.precharge_value(&items[*index])?;
                    self.locals[*item_slot] = Some(items[*index].clone());
                    let mapped = self.eval(value)?;
                    self.precharge_value(&mapped)?;
                    output.push(mapped);
                    *index += 1;
                    false
                }
            }
            Work::Filter {
                slot,
                item_slot,
                items,
                index,
                predicate,
                output,
            } => {
                if *index == items.len() {
                    self.store_precharged(*slot, Value::Array(core::mem::take(output)))?;
                    self.locals[*item_slot] = None;
                    self.pc += 1;
                    true
                } else if !self.charge(remaining, expression_cost(predicate))? {
                    advanced = false;
                    false
                } else {
                    self.precharge_value(&items[*index])?;
                    self.locals[*item_slot] = Some(items[*index].clone());
                    if expect_bool(self.eval(predicate)?, "filter predicate")? {
                        self.precharge_value(&items[*index])?;
                        output.push(items[*index].clone());
                    }
                    *index += 1;
                    false
                }
            }
            Work::Reduce {
                slot,
                item_slot,
                accumulator_slot,
                items,
                index,
                value,
                accumulator,
            } => {
                if *index == items.len() {
                    self.store(*slot, accumulator.clone())?;
                    self.locals[*item_slot] = None;
                    self.locals[*accumulator_slot] = None;
                    self.pc += 1;
                    true
                } else if !self.charge(remaining, expression_cost(value))? {
                    advanced = false;
                    false
                } else {
                    self.precharge_value(&items[*index])?;
                    self.precharge_value(accumulator)?;
                    self.locals[*item_slot] = Some(items[*index].clone());
                    self.locals[*accumulator_slot] = Some(accumulator.clone());
                    *accumulator = self.eval(value)?;
                    self.precharge_value(accumulator)?;
                    *index += 1;
                    false
                }
            }
            Work::FanOut {
                item_slot,
                items,
                index,
                arguments,
                tool_id,
                call_site,
                calls,
                ..
            } => {
                if *index == items.len() {
                    self.locals[*item_slot] = None;
                    true
                } else if !self.charge(remaining, expression_cost(arguments))? {
                    advanced = false;
                    false
                } else {
                    self.precharge_value(&items[*index])?;
                    self.locals[*item_slot] = Some(items[*index].clone());
                    let args = self.eval(arguments)?;
                    require_object(&args)?;
                    calls.push(self.request(tool_id.clone(), *call_site, args)?);
                    *index += 1;
                    false
                }
            }
        };
        if !done {
            self.work = Some(work);
        } else if !matches!(work, Work::FanOut { .. }) {
            self.work = None;
        } else {
            self.work = Some(work);
        }
        Ok(advanced)
    }

    fn finish_fanout_if_ready(&mut self) -> Result<Option<StepOutcome>, SandboxError> {
        let ready =
            matches!(&self.work, Some(Work::FanOut { items, index, .. }) if *index == items.len());
        if !ready {
            return Ok(None);
        }
        let work = self
            .work
            .take()
            .ok_or_else(|| execution("fan-out work state missing"))?;
        if let Work::FanOut { slot, calls, .. } = work {
            self.suspend(slot, calls, true).map(Some)
        } else {
            Err(execution("fan-out work state changed"))
        }
    }

    fn suspend(
        &mut self,
        slot: usize,
        calls: Vec<ToolRequest>,
        fan_out: bool,
    ) -> Result<StepOutcome, SandboxError> {
        if calls.is_empty() {
            return Err(execution("tool batch cannot be empty"));
        }
        if self.yields as usize >= self.program.limits.max_yields {
            return Err(resource("yield limit exceeded"));
        }
        let yield_ordinal = self.yields;
        self.yields += 1;
        if fan_out {
            self.fanout_batches = self.fanout_batches.saturating_add(1);
        }
        let requests = calls
            .iter()
            .map(|call| (call.call_site, call.dynamic_ordinal))
            .collect();
        self.pending = Some(Pending {
            yield_ordinal,
            slot,
            requests,
            fan_out,
        });
        Ok(StepOutcome::Yielded {
            batch: ToolBatch {
                calls,
                read_only_fan_out: fan_out,
            },
            resume: ResumeToken {
                execution_id: self.execution_id,
                yield_ordinal,
            },
        })
    }

    fn request(
        &mut self,
        tool_id: String,
        call_site: u32,
        arguments: Value,
    ) -> Result<ToolRequest, SandboxError> {
        self.precharge_value(&arguments)?;
        self.precharge_bytes(tool_id.len())?;
        let dynamic_ordinal = self.dynamic_ordinal;
        self.dynamic_ordinal = self
            .dynamic_ordinal
            .checked_add(1)
            .ok_or_else(|| resource("dynamic call ordinal exhausted"))?;
        Ok(ToolRequest {
            execution_id: self.execution_id,
            program_attempt: self.program_attempt,
            call_site,
            dynamic_ordinal,
            tool_id,
            arguments,
        })
    }

    fn eval(&self, code: &ExprCode) -> Result<Value, SandboxError> {
        let mut stack = Vec::new();
        stack
            .try_reserve(self.program.limits.max_operand_stack.min(code.0.len()))
            .map_err(|_| resource("operand stack allocation failed"))?;
        for instruction in &code.0 {
            match instruction {
                ExprInstruction::Constant(value) => stack.push(value.clone()),
                ExprInstruction::Load(slot) => stack.push(
                    self.locals
                        .get(*slot)
                        .and_then(Option::as_ref)
                        .cloned()
                        .ok_or_else(|| {
                            execution("local is unavailable in this control-flow path")
                        })?,
                ),
                ExprInstruction::Path(pointer) => {
                    let value = pop(&mut stack)?;
                    stack.push(
                        value
                            .pointer(pointer)
                            .cloned()
                            .ok_or_else(|| execution("JSON pointer did not resolve"))?,
                    );
                }
                ExprInstruction::Array(count) => {
                    let start = stack
                        .len()
                        .checked_sub(*count)
                        .ok_or_else(|| execution("operand stack underflow"))?;
                    let items = stack.split_off(start);
                    stack.push(Value::Array(items));
                }
                ExprInstruction::Object(keys) => {
                    let start = stack
                        .len()
                        .checked_sub(keys.len())
                        .ok_or_else(|| execution("operand stack underflow"))?;
                    let values = stack.split_off(start);
                    let mut object = Map::new();
                    for (key, value) in keys.iter().cloned().zip(values) {
                        object.insert(key, value);
                    }
                    stack.push(Value::Object(object));
                }
                ExprInstruction::Binary(operator) => {
                    let right = pop(&mut stack)?;
                    let left = pop(&mut stack)?;
                    stack.push(binary(*operator, left, right)?);
                }
                ExprInstruction::Unary(operator) => {
                    let value = pop(&mut stack)?;
                    stack.push(unary(*operator, value)?);
                }
            }
        }
        if stack.len() != 1 {
            return Err(execution("expression did not produce one value"));
        }
        pop(&mut stack)
    }

    fn charge(&mut self, remaining: &mut u64, cost: u64) -> Result<bool, SandboxError> {
        if cost > *remaining {
            return Ok(false);
        }
        let next = self
            .fuel_used
            .checked_add(cost)
            .ok_or_else(|| resource("fuel limit exceeded"))?;
        if next > self.program.limits.max_fuel {
            return Err(resource("fuel limit exceeded"));
        }
        self.fuel_used = next;
        *remaining -= cost;
        Ok(true)
    }

    fn store(&mut self, slot: usize, value: Value) -> Result<(), SandboxError> {
        if self.locals.get(slot).and_then(Option::as_ref).is_some() {
            return Err(execution("immutable local was already initialized"));
        }
        self.precharge_value(&value)?;
        self.locals[slot] = Some(value);
        Ok(())
    }

    /// Stores a composite whose values and JSON framing were charged before its
    /// backing collection grew. This avoids charging host responses twice when
    /// they are wrapped as inert VM results.
    fn store_precharged(&mut self, slot: usize, value: Value) -> Result<(), SandboxError> {
        if self.locals.get(slot).and_then(Option::as_ref).is_some() {
            return Err(execution("immutable local was already initialized"));
        }
        self.locals[slot] = Some(value);
        Ok(())
    }

    fn precharge_value(&mut self, value: &Value) -> Result<(), SandboxError> {
        self.precharge_bytes(serialized_value_len(value)?)
    }

    fn precharge_bytes(&mut self, bytes: usize) -> Result<(), SandboxError> {
        let live_bytes = self
            .live_bytes
            .checked_add(bytes)
            .ok_or_else(|| resource("live byte limit exceeded"))?;
        let cumulative_bytes = self
            .cumulative_bytes
            .checked_add(bytes)
            .ok_or_else(|| resource("cumulative byte limit exceeded"))?;
        if live_bytes > self.program.limits.max_live_bytes {
            return Err(resource("live byte limit exceeded"));
        }
        if cumulative_bytes > self.program.limits.max_cumulative_bytes {
            return Err(resource("cumulative byte limit exceeded"));
        }
        self.live_bytes = live_bytes;
        self.cumulative_bytes = cumulative_bytes;
        Ok(())
    }
}

fn instruction_minimum_cost(instruction: &Instruction) -> u64 {
    let expression = match instruction {
        Instruction::Let { value, .. } | Instruction::Return { value } => expression_cost(value),
        Instruction::Branch { condition, .. } => expression_cost(condition),
        Instruction::LoopStart { collection, .. }
        | Instruction::Map { collection, .. }
        | Instruction::Filter { collection, .. }
        | Instruction::FanOut { collection, .. } => expression_cost(collection),
        Instruction::Reduce {
            collection,
            initial,
            ..
        } => expression_cost(collection).saturating_add(expression_cost(initial)),
        Instruction::Invoke { arguments, .. } => expression_cost(arguments),
        Instruction::Jump { .. } | Instruction::LoopNext { .. } => 1,
    };
    expression.max(1)
}

fn expression_cost(code: &ExprCode) -> u64 {
    (code.0.len() as u64).max(1)
}

fn pop(stack: &mut Vec<Value>) -> Result<Value, SandboxError> {
    stack
        .pop()
        .ok_or_else(|| execution("operand stack underflow"))
}
fn expect_bool(value: Value, label: &str) -> Result<bool, SandboxError> {
    value
        .as_bool()
        .ok_or_else(|| execution(alloc::format!("{label} must evaluate to a boolean")))
}
fn expect_array(value: Value, label: &str) -> Result<Vec<Value>, SandboxError> {
    match value {
        Value::Array(items) => Ok(items),
        _ => Err(execution(alloc::format!(
            "{label} must evaluate to an array"
        ))),
    }
}
fn require_object(value: &Value) -> Result<(), SandboxError> {
    if value.is_object() {
        Ok(())
    } else {
        Err(execution("tool arguments must evaluate to an object"))
    }
}
fn validate_items(actual: usize, declared: usize, effective: usize) -> Result<(), SandboxError> {
    if actual > declared || actual > effective {
        Err(resource("collection item limit exceeded"))
    } else {
        Ok(())
    }
}

fn loop_body_slots(
    code: &[Instruction],
    body_target: usize,
    end_target: usize,
) -> Result<Vec<usize>, SandboxError> {
    let end = end_target
        .checked_sub(1)
        .ok_or_else(|| execution("loop body target is invalid"))?;
    let body = code
        .get(body_target..end)
        .ok_or_else(|| execution("loop body target is invalid"))?;
    let mut slots = Vec::new();
    slots
        .try_reserve(body.len().saturating_mul(3))
        .map_err(|_| resource("loop local tracking allocation failed"))?;
    for instruction in body {
        match instruction {
            Instruction::Let { slot, .. }
            | Instruction::Map { slot, .. }
            | Instruction::Filter { slot, .. }
            | Instruction::Invoke { slot, .. }
            | Instruction::FanOut { slot, .. } => slots.push(*slot),
            Instruction::LoopStart { item_slot, .. } => slots.push(*item_slot),
            Instruction::Reduce {
                slot,
                item_slot,
                accumulator_slot,
                ..
            } => {
                slots.push(*slot);
                slots.push(*item_slot);
                slots.push(*accumulator_slot);
            }
            Instruction::Branch { .. }
            | Instruction::Jump { .. }
            | Instruction::LoopNext { .. }
            | Instruction::Return { .. } => {}
        }
    }
    slots.sort_unstable();
    slots.dedup();
    Ok(slots)
}

fn clear_loop_body_slots(locals: &mut [Option<Value>], slots: &[usize]) {
    for slot in slots {
        if let Some(local) = locals.get_mut(*slot) {
            *local = None;
        }
    }
}

fn integer(value: &Value) -> Result<i64, SandboxError> {
    value
        .as_i64()
        .ok_or_else(|| execution("integer operation requires i64 operands"))
}
fn binary(operator: BinaryOperator, left: Value, right: Value) -> Result<Value, SandboxError> {
    use BinaryOperator::*;
    let value = match operator {
        Add => Value::Number(Number::from(
            integer(&left)?
                .checked_add(integer(&right)?)
                .ok_or_else(|| execution("checked integer addition overflowed"))?,
        )),
        Subtract => Value::Number(Number::from(
            integer(&left)?
                .checked_sub(integer(&right)?)
                .ok_or_else(|| execution("checked integer subtraction overflowed"))?,
        )),
        Multiply => Value::Number(Number::from(
            integer(&left)?
                .checked_mul(integer(&right)?)
                .ok_or_else(|| execution("checked integer multiplication overflowed"))?,
        )),
        Divide => Value::Number(Number::from(
            integer(&left)?
                .checked_div(integer(&right)?)
                .ok_or_else(|| execution("checked integer division failed"))?,
        )),
        Remainder => Value::Number(Number::from(
            integer(&left)?
                .checked_rem(integer(&right)?)
                .ok_or_else(|| execution("checked integer remainder failed"))?,
        )),
        Equal => Value::Bool(left == right),
        NotEqual => Value::Bool(left != right),
        LessThan => Value::Bool(integer(&left)? < integer(&right)?),
        LessThanOrEqual => Value::Bool(integer(&left)? <= integer(&right)?),
        GreaterThan => Value::Bool(integer(&left)? > integer(&right)?),
        GreaterThanOrEqual => Value::Bool(integer(&left)? >= integer(&right)?),
        And => {
            Value::Bool(expect_bool(left, "left operand")? && expect_bool(right, "right operand")?)
        }
        Or => {
            Value::Bool(expect_bool(left, "left operand")? || expect_bool(right, "right operand")?)
        }
    };
    Ok(value)
}

fn unary(operator: UnaryOperator, value: Value) -> Result<Value, SandboxError> {
    use UnaryOperator::*;
    match operator {
        Not => Ok(Value::Bool(!expect_bool(value, "not operand")?)),
        Negate => Ok(Value::Number(Number::from(
            integer(&value)?
                .checked_neg()
                .ok_or_else(|| execution("checked integer negation overflowed"))?,
        ))),
        Count => match value {
            Value::Array(values) => Ok(Value::Number(Number::from(values.len() as u64))),
            Value::Object(values) => Ok(Value::Number(Number::from(values.len() as u64))),
            _ => Err(execution("count requires an array or object")),
        },
        Sum => {
            let values = expect_array(value, "sum operand")?;
            let mut total = 0i64;
            for value in values {
                total = total
                    .checked_add(integer(&value)?)
                    .ok_or_else(|| execution("checked integer sum overflowed"))?;
            }
            Ok(Value::Number(Number::from(total)))
        }
        All => {
            let values = expect_array(value, "all operand")?;
            let mut result = true;
            for value in values {
                result &= expect_bool(value, "all item")?;
            }
            Ok(Value::Bool(result))
        }
        Any => {
            let values = expect_array(value, "any operand")?;
            let mut result = false;
            for value in values {
                result |= expect_bool(value, "any item")?;
            }
            Ok(Value::Bool(result))
        }
    }
}

/// Exact compact JSON length without allocating a serialization buffer. This
/// is the unit used for final output and value-retention accounting.
fn serialized_value_len(value: &Value) -> Result<usize, SandboxError> {
    match value {
        Value::Null => Ok(4),
        Value::Bool(true) => Ok(4),
        Value::Bool(false) => Ok(5),
        Value::Number(number) => Ok(number.to_string().len()),
        Value::String(value) => serialized_string_len(value),
        Value::Array(values) => serialized_array_len(values),
        Value::Object(values) => {
            let mut total = 2usize;
            for (index, (key, value)) in values.iter().enumerate() {
                if index > 0 {
                    total = checked_bytes(total, 1)?;
                }
                total = checked_bytes(total, serialized_string_len(key)?)?;
                total = checked_bytes(total, 1)?;
                total = checked_bytes(total, serialized_value_len(value)?)?;
            }
            Ok(total)
        }
    }
}

fn serialized_array_len(values: &[Value]) -> Result<usize, SandboxError> {
    let mut total = serialized_array_syntax_len(values.len())?;
    for value in values {
        total = checked_bytes(total, serialized_value_len(value)?)?;
    }
    Ok(total)
}

fn serialized_array_syntax_len(count: usize) -> Result<usize, SandboxError> {
    checked_bytes(2, count.saturating_sub(1))
}

fn response_object_syntax_len(ok: bool) -> Result<usize, SandboxError> {
    let mut total = 2usize;
    total = checked_bytes(total, serialized_string_len("ok")?)?;
    total = checked_bytes(total, 1)?;
    total = checked_bytes(total, if ok { 4 } else { 5 })?;
    total = checked_bytes(total, 1)?;
    total = checked_bytes(total, serialized_string_len("output")?)?;
    checked_bytes(total, 1)
}

fn serialized_string_len(value: &str) -> Result<usize, SandboxError> {
    let mut total = 2usize;
    for byte in value.bytes() {
        total = checked_bytes(
            total,
            match byte {
                b'"' | b'\\' => 2,
                0x00..=0x1f => 6,
                _ => 1,
            },
        )?;
    }
    Ok(total)
}

fn checked_bytes(total: usize, additional: usize) -> Result<usize, SandboxError> {
    total
        .checked_add(additional)
        .ok_or_else(|| resource("value byte limit exceeded"))
}

fn resource(message: impl Into<String>) -> SandboxError {
    SandboxError::new(SandboxErrorCode::ResourceLimit, message)
}
fn execution(message: impl Into<String>) -> SandboxError {
    SandboxError::new(SandboxErrorCode::Execution, message)
}
fn resume_error(message: impl Into<String>) -> SandboxError {
    SandboxError::new(SandboxErrorCode::InvalidResume, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Program, SandboxLimits};
    use serde_json::json;

    fn execution(program: serde_json::Value, id: u64) -> Execution {
        let bytes = serde_json::to_vec(&program).unwrap();
        let limits = SandboxLimits::default();
        let verified = Program::from_json(&bytes, &limits)
            .unwrap()
            .compile(&limits)
            .unwrap();
        Execution::new(verified, ExecutionId(id)).unwrap()
    }

    fn run(mut execution: Execution) -> Value {
        loop {
            match execution.step(1_024).unwrap() {
                StepOutcome::Sliced => {}
                StepOutcome::Complete(value) => return value,
                StepOutcome::Yielded { .. } => panic!("unexpected tool yield"),
            }
        }
    }

    #[test]
    fn executes_branching_map_filter_and_reduce_deterministically() {
        let program = json!({"version":1,"body":[
            {"kind":"let","name":"xs","value":{"kind":"array","items":[
                {"kind":"integer","value":1},{"kind":"integer","value":2},{"kind":"integer","value":3}
            ]}},
            {"kind":"for_each","item":"loop_item","collection":{"kind":"variable","name":"xs"},"max_iterations":3,
             "body":[{"kind":"let","name":"per_iteration","value":{"kind":"variable","name":"loop_item"}}]},
            {"kind":"map","name":"doubled","item":"i","collection":{"kind":"variable","name":"xs"},"max_items":3,
             "value":{"kind":"binary","operator":"multiply","left":{"kind":"variable","name":"i"},"right":{"kind":"integer","value":2}}},
            {"kind":"filter","name":"even","item":"i","collection":{"kind":"variable","name":"doubled"},"max_items":3,
             "predicate":{"kind":"binary","operator":"equal","left":{"kind":"binary","operator":"remainder","left":{"kind":"variable","name":"i"},"right":{"kind":"integer","value":2}},"right":{"kind":"integer","value":0}}},
            {"kind":"reduce","name":"total","item":"i","accumulator":"acc","collection":{"kind":"variable","name":"even"},"max_items":3,
             "initial":{"kind":"integer","value":0},"value":{"kind":"binary","operator":"add","left":{"kind":"variable","name":"acc"},"right":{"kind":"variable","name":"i"}}},
            {"kind":"branch","condition":{"kind":"binary","operator":"equal","left":{"kind":"variable","name":"total"},"right":{"kind":"integer","value":12}},
             "then_body":[{"kind":"return","value":{"kind":"string","value":"ok"}}],
             "else_body":[{"kind":"return","value":{"kind":"string","value":"bad"}}]}
        ]});
        assert_eq!(run(execution(program, 1)), json!("ok"));
    }

    #[test]
    fn fanout_yields_ordered_occurrences_and_preserves_partial_failure() {
        let program = json!({"version":1,"body":[
            {"kind":"fan_out","name":"results","tool_id":"read.item","item":"i",
             "collection":{"kind":"array","items":[{"kind":"integer","value":2},{"kind":"integer","value":1}]},"max_calls":2,
             "arguments":{"kind":"object","entries":[{"key":"id","value":{"kind":"variable","name":"i"}}]}},
            {"kind":"return","value":{"kind":"variable","name":"results"}}
        ]});
        let mut vm = execution(program, 7);
        let (batch, token) = match vm.step(1_024).unwrap() {
            StepOutcome::Yielded { batch, resume } => (batch, resume),
            other => panic!("expected yield, got {other:?}"),
        };
        assert!(batch.requests_read_only_fan_out());
        let debug = alloc::format!("{batch:?}");
        assert!(!debug.contains("read.item"));
        assert!(!debug.contains("safe"));
        assert_eq!(batch.calls().len(), 2);
        assert_eq!(batch.calls()[0].arguments, json!({"id":2}));
        assert_eq!(batch.calls()[1].arguments, json!({"id":1}));
        assert_ne!(
            batch.calls()[0].dynamic_ordinal,
            batch.calls()[1].dynamic_ordinal
        );
        vm.resume(
            token,
            vec![
                ToolResponse::success(&batch.calls()[0], json!({"value":"safe"})),
                ToolResponse::failure(&batch.calls()[1]),
            ],
        )
        .unwrap();
        assert_eq!(
            vm.step(1_024).unwrap(),
            StepOutcome::Complete(json!([
                {"ok":true,"output":{"value":"safe"}},
                {"ok":false,"output":null}
            ]))
        );
    }

    #[test]
    fn resume_tokens_are_cross_run_and_order_bound() {
        let program = || {
            json!({"version":1,"body":[
                {"kind":"invoke","name":"result","tool_id":"read","arguments":{"kind":"object","entries":[]}},
                {"kind":"return","value":{"kind":"variable","name":"result"}}
            ]})
        };
        let mut first = execution(program(), 1);
        let mut second = execution(program(), 2);
        let (first_batch, first_token) = match first.step(100).unwrap() {
            StepOutcome::Yielded { batch, resume } => (batch, resume),
            _ => unreachable!(),
        };
        let (second_batch, second_token) = match second.step(100).unwrap() {
            StepOutcome::Yielded { batch, resume } => (batch, resume),
            _ => unreachable!(),
        };
        assert_eq!(
            second
                .resume(
                    first_token,
                    vec![ToolResponse::success(&second_batch.calls()[0], json!(1))]
                )
                .unwrap_err()
                .code(),
            SandboxErrorCode::InvalidResume
        );
        second
            .resume(
                second_token,
                vec![ToolResponse::success(&second_batch.calls()[0], json!(2))],
            )
            .unwrap();
        assert!(matches!(
            second.step(100).unwrap(),
            StepOutcome::Complete(_)
        ));
        first
            .resume(
                ResumeToken {
                    execution_id: ExecutionId(1),
                    yield_ordinal: 0,
                },
                vec![ToolResponse::success(&first_batch.calls()[0], json!(1))],
            )
            .unwrap();
        assert!(matches!(first.step(100).unwrap(), StepOutcome::Complete(_)));
    }

    #[test]
    fn resume_rejects_duplicate_count_and_response_order_without_consuming_the_token() {
        let program = json!({"version":1,"body":[
            {"kind":"fan_out","name":"results","tool_id":"read","item":"item",
             "collection":{"kind":"array","items":[{"kind":"integer","value":1},{"kind":"integer","value":2}]},"max_calls":2,
             "arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"variable","name":"item"}}]}},
            {"kind":"return","value":{"kind":"variable","name":"results"}}
        ]});
        let mut vm = execution(program, 42);
        let (batch, token) = match vm.step(100).unwrap() {
            StepOutcome::Yielded { batch, resume } => (batch, resume),
            other => panic!("expected yield, got {other:?}"),
        };
        let ordered = vec![
            ToolResponse::success(&batch.calls()[0], json!("first")),
            ToolResponse::success(&batch.calls()[1], json!("second")),
        ];
        assert_eq!(
            vm.resume(
                ResumeToken {
                    execution_id: ExecutionId(42),
                    yield_ordinal: 99,
                },
                ordered.clone(),
            )
            .unwrap_err()
            .code(),
            SandboxErrorCode::InvalidResume
        );
        assert_eq!(
            vm.resume(token, vec![ordered[0].clone()])
                .unwrap_err()
                .code(),
            SandboxErrorCode::InvalidResume
        );
        let valid_token = match vm.step(1).unwrap_err().code() {
            SandboxErrorCode::InvalidResume => ResumeToken {
                execution_id: ExecutionId(42),
                yield_ordinal: 0,
            },
            code => panic!("expected suspended execution, got {code:?}"),
        };
        assert_eq!(
            vm.resume(valid_token, vec![ordered[1].clone(), ordered[0].clone()])
                .unwrap_err()
                .code(),
            SandboxErrorCode::InvalidResume
        );
        vm.resume(
            ResumeToken {
                execution_id: ExecutionId(42),
                yield_ordinal: 0,
            },
            ordered,
        )
        .unwrap();
        assert!(matches!(vm.step(100).unwrap(), StepOutcome::Complete(_)));
        assert_eq!(
            vm.resume(
                ResumeToken {
                    execution_id: ExecutionId(42),
                    yield_ordinal: 0,
                },
                Vec::new(),
            )
            .unwrap_err()
            .code(),
            SandboxErrorCode::InvalidResume
        );
    }

    #[test]
    fn every_nonterminal_slice_charges_fuel_and_fuel_limit_is_terminal() {
        let program = json!({"version":1,"body":[
            {"kind":"let","name":"value","value":{"kind":"integer","value":1}},
            {"kind":"return","value":{"kind":"variable","name":"value"}}
        ]});
        let mut vm = execution(program, 43);
        assert_eq!(vm.step(1).unwrap(), StepOutcome::Sliced);
        assert_eq!(vm.metrics().fuel_used, 1);
        assert_eq!(vm.step(1).unwrap(), StepOutcome::Complete(json!(1)));
        assert_eq!(vm.metrics().fuel_used, 2);

        let limits = SandboxLimits {
            max_fuel: 1,
            max_slice_fuel: 1,
            ..SandboxLimits::default()
        };
        let bytes = serde_json::to_vec(&json!({"version":1,"body":[
            {"kind":"let","name":"value","value":{"kind":"integer","value":1}},
            {"kind":"return","value":{"kind":"variable","name":"value"}}
        ]}))
        .unwrap();
        let verified = Program::from_json(&bytes, &limits)
            .unwrap()
            .compile(&limits)
            .unwrap();
        let mut exhausted = Execution::new(verified, ExecutionId(44)).unwrap();
        assert_eq!(exhausted.step(1).unwrap(), StepOutcome::Sliced);
        assert_eq!(
            exhausted.step(1).unwrap_err().code(),
            SandboxErrorCode::ResourceLimit
        );
    }

    #[test]
    fn slice_and_yield_bounds_are_inclusive_and_refuse_the_next_unit_of_work() {
        let program = json!({"version":1,"body":[
            {"kind":"invoke","name":"first","tool_id":"read","arguments":{"kind":"object","entries":[]}},
            {"kind":"invoke","name":"second","tool_id":"read","arguments":{"kind":"object","entries":[]}},
            {"kind":"return","value":{"kind":"variable","name":"second"}}
        ]});
        let limits = SandboxLimits {
            max_yields: 1,
            max_slice_fuel: 1,
            ..SandboxLimits::default()
        };
        let verified = Program::from_json(&serde_json::to_vec(&program).unwrap(), &limits)
            .unwrap()
            .compile(&limits)
            .unwrap();
        let mut vm = Execution::new(verified, ExecutionId(45)).unwrap();
        assert_eq!(
            vm.step(0).unwrap_err().code(),
            SandboxErrorCode::ResourceLimit
        );
        assert_eq!(
            vm.step(2).unwrap_err().code(),
            SandboxErrorCode::ResourceLimit
        );
        let (batch, token) = match vm.step(1).unwrap() {
            StepOutcome::Yielded { batch, resume } => (batch, resume),
            other => panic!("expected first yield, got {other:?}"),
        };
        vm.resume(
            token,
            vec![ToolResponse::success(&batch.calls()[0], json!(1))],
        )
        .unwrap();
        assert_eq!(
            vm.step(1).unwrap_err().code(),
            SandboxErrorCode::ResourceLimit
        );
        assert_eq!(vm.metrics().yields, 1);
    }

    #[test]
    fn bounded_hostile_program_corpus_never_panics() {
        let limits = SandboxLimits {
            max_program_bytes: 128,
            max_ast_nodes: 16,
            max_nesting: 8,
            max_bytecode_instructions: 32,
            max_constant_bytes: 32,
            max_locals: 4,
            max_operand_stack: 8,
            max_control_stack: 4,
            max_fuel: 32,
            max_slice_fuel: 8,
            max_collection_items: 4,
            max_loop_iterations: 4,
            max_yields: 2,
            max_fanout: 2,
            max_live_bytes: 256,
            max_cumulative_bytes: 256,
            max_output_bytes: 128,
        };
        let corpus: &[&[u8]] = &[
            b"",
            b"\xff",
            br#"{"version":1,"body":[{"kind":"return","value":{"kind":"integer","value":1e999}}]}"#,
            br#"{"version":1,"body":[{"kind":"return","value":{"kind":"array","items":[[[[[[[[[0]]]]]]]]}}]}"#,
            br#"{"version":1,"body":[{"kind":"return","value":{"kind":"string","value":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}]}"#,
            br#"{"version":1,"body":[{"kind":"invoke","name":"x","tool_id":"read","arguments":{"kind":"null"}},{"kind":"return","value":{"kind":"variable","name":"x"}}]}"#,
        ];
        for raw in corpus {
            let result = std::panic::catch_unwind(|| {
                let _ = Program::from_json(raw, &limits)
                    .and_then(|program| program.compile(&limits))
                    .and_then(|verified| Execution::new(verified, ExecutionId(99)))
                    .and_then(|mut execution| execution.step(1));
            });
            assert!(
                result.is_ok(),
                "hostile input must be reported, never panic"
            );
        }
    }

    #[test]
    fn checked_integer_overflow_is_terminal_error() {
        let program = json!({"version":1,"body":[{"kind":"return","value":{"kind":"binary","operator":"add","left":{"kind":"integer","value":9223372036854775807i64},"right":{"kind":"integer","value":1}}} ]});
        let mut vm = execution(program, 9);
        assert_eq!(
            vm.step(100).unwrap_err().code(),
            SandboxErrorCode::Execution
        );
    }

    #[test]
    fn output_and_value_accounting_are_exact_at_boundaries() {
        fn compile_with(value: &str, max_output_bytes: usize) -> Execution {
            let program = json!({"version":1,"body":[{"kind":"return","value":{"kind":"string","value":value}}]});
            let limits = SandboxLimits {
                max_output_bytes,
                ..SandboxLimits::default()
            };
            let bytes = serde_json::to_vec(&program).unwrap();
            let verified = Program::from_json(&bytes, &limits)
                .unwrap()
                .compile(&limits)
                .unwrap();
            Execution::new(verified, ExecutionId(22)).unwrap()
        }

        assert_eq!(
            compile_with("1234", 7).step(100).unwrap(),
            StepOutcome::Complete(json!("1234"))
        );
        assert_eq!(
            compile_with("12345", 7).step(100).unwrap(),
            StepOutcome::Complete(json!("12345"))
        );
        assert_eq!(
            compile_with("123456", 7).step(100).unwrap_err().code(),
            SandboxErrorCode::ResourceLimit
        );

        let response_program = json!({"version":1,"body":[
            {"kind":"invoke","name":"result","tool_id":"read","arguments":{"kind":"object","entries":[]}},
            {"kind":"return","value":{"kind":"variable","name":"result"}}
        ]});
        let response_output = json!({"ok":true,"output":"x"});
        let expected = serialized_value_len(&json!({})).unwrap()
            + "read".len()
            + (serialized_value_len(&response_output).unwrap() * 2);
        let build_response_vm = |limit| {
            let limits = SandboxLimits {
                max_live_bytes: limit,
                max_cumulative_bytes: limit,
                ..SandboxLimits::default()
            };
            let verified =
                Program::from_json(&serde_json::to_vec(&response_program).unwrap(), &limits)
                    .unwrap()
                    .compile(&limits)
                    .unwrap();
            Execution::new(verified, ExecutionId(23)).unwrap()
        };
        let mut at_limit = build_response_vm(expected);
        let (batch, token) = match at_limit.step(100).unwrap() {
            StepOutcome::Yielded { batch, resume } => (batch, resume),
            _ => unreachable!(),
        };
        at_limit
            .resume(
                token,
                vec![ToolResponse::success(&batch.calls()[0], json!("x"))],
            )
            .unwrap();
        assert!(matches!(
            at_limit.step(100).unwrap(),
            StepOutcome::Complete(_)
        ));
        assert_eq!(at_limit.metrics().cumulative_bytes, expected);

        let mut below_limit = build_response_vm(expected - 1);
        let (batch, token) = match below_limit.step(100).unwrap() {
            StepOutcome::Yielded { batch, resume } => (batch, resume),
            _ => unreachable!(),
        };
        below_limit
            .resume(
                token,
                vec![ToolResponse::success(&batch.calls()[0], json!("x"))],
            )
            .unwrap();
        assert_eq!(
            below_limit.step(100).unwrap_err().code(),
            SandboxErrorCode::ResourceLimit
        );

        let intermediate_program = json!({"version":1,"body":[
            {"kind":"let","name":"xs","value":{"kind":"array","items":[{"kind":"string","value":"a"},{"kind":"string","value":"b"}]}},
            {"kind":"map","name":"mapped","item":"item","collection":{"kind":"variable","name":"xs"},"max_items":2,"value":{"kind":"variable","name":"item"}},
            {"kind":"return","value":{"kind":"variable","name":"mapped"}}
        ]});
        let build_intermediate_vm = |limit| {
            let limits = SandboxLimits {
                max_live_bytes: limit,
                max_cumulative_bytes: limit,
                ..SandboxLimits::default()
            };
            let verified =
                Program::from_json(&serde_json::to_vec(&intermediate_program).unwrap(), &limits)
                    .unwrap()
                    .compile(&limits)
                    .unwrap();
            Execution::new(verified, ExecutionId(24)).unwrap()
        };
        let mut baseline = build_intermediate_vm(4 * 1024);
        loop {
            match baseline.step(1_024).unwrap() {
                StepOutcome::Sliced => {}
                StepOutcome::Complete(value) => {
                    assert_eq!(value, json!(["a", "b"]));
                    break;
                }
                StepOutcome::Yielded { .. } => unreachable!(),
            }
        }
        let intermediate_limit = baseline.metrics().cumulative_bytes;
        assert!(matches!(
            run(build_intermediate_vm(intermediate_limit + 1)),
            Value::Array(_)
        ));
        assert!(matches!(
            run(build_intermediate_vm(intermediate_limit)),
            Value::Array(_)
        ));
        let mut oversized = build_intermediate_vm(intermediate_limit - 1);
        loop {
            match oversized.step(1_024) {
                Ok(StepOutcome::Sliced) => {}
                Ok(_) => panic!("expected an intermediate resource limit"),
                Err(error) => {
                    assert_eq!(error.code(), SandboxErrorCode::ResourceLimit);
                    break;
                }
            }
        }
    }
}
