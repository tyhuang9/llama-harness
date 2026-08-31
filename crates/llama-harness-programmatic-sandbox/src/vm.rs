use crate::{
    accounting::{
        array_framing_retained_bytes, checked_add, cloned_value_allocation_bytes,
        key_retained_bytes, measure_value, object_framing_retained_bytes, primitive_retained_bytes,
        serialized_string_len, string_retained_bytes, vector_allocation_bytes,
    },
    compiler::{ExprCode, ExprInstruction, Instruction, VerifiedProgram},
    BinaryOperator, SandboxError, SandboxErrorCode, UnaryOperator,
};
use alloc::{string::String, vec::Vec};
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
    program_attempt: u32,
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
        fuel_remaining: u64,
    },
    Filter {
        slot: usize,
        item_slot: usize,
        items: Vec<Value>,
        index: usize,
        predicate: ExprCode,
        output: Vec<Value>,
        fuel_remaining: u64,
    },
    Reduce {
        slot: usize,
        item_slot: usize,
        accumulator_slot: usize,
        items: Vec<Value>,
        index: usize,
        value: ExprCode,
        accumulator: Value,
        fuel_remaining: u64,
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
        fuel_remaining: u64,
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
    instruction_meter: Option<(usize, u64)>,
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
        let structural_bytes = checked_add(
            vector_allocation_bytes::<Option<Value>>(program.local_count)?,
            vector_allocation_bytes::<LoopFrame>(program.limits.max_control_stack)?,
        )?;
        if structural_bytes > program.limits.max_live_bytes {
            return Err(resource("live byte limit exceeded"));
        }
        if structural_bytes > program.limits.max_cumulative_bytes {
            return Err(resource("cumulative byte limit exceeded"));
        }
        let mut locals = Vec::new();
        locals
            .try_reserve_exact(program.local_count)
            .map_err(|_| resource("local allocation failed"))?;
        locals.resize_with(program.local_count, || None);
        let mut loops = Vec::new();
        loops
            .try_reserve_exact(program.limits.max_control_stack)
            .map_err(|_| resource("control stack allocation failed"))?;
        Ok(Self {
            program,
            execution_id,
            program_attempt,
            pc: 0,
            locals,
            loops,
            work: None,
            pending: None,
            fuel_used: 0,
            yields: 0,
            dynamic_ordinal: 0,
            live_bytes: structural_bytes,
            cumulative_bytes: structural_bytes,
            branches: 0,
            loop_iterations: 0,
            fanout_batches: 0,
            instruction_meter: None,
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
        let result = self.step_inner(slice_fuel);
        if result.is_err() {
            self.terminal = true;
            self.work = None;
            self.pending = None;
            self.instruction_meter = None;
        }
        result
    }

    fn step_inner(&mut self, slice_fuel: u64) -> Result<StepOutcome, SandboxError> {
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
            if self.instruction_meter.is_none() {
                let mut cost =
                    instruction_metered_cost(&instruction, &self.locals, &self.program.limits)?;
                match &instruction {
                    Instruction::LoopStart { end_target, .. } => {
                        cost = checked_fuel_add(
                            cost,
                            usize_to_fuel(end_target.saturating_sub(self.pc))?,
                        )?;
                    }
                    Instruction::LoopNext { body_target } => {
                        if let Some(frame) = self.loops.last() {
                            if frame.body_target != *body_target {
                                return Err(execution("loop frame target mismatch"));
                            }
                            if let Some(item) = frame.items.get(frame.index.saturating_add(1)) {
                                cost = checked_fuel_add(
                                    cost,
                                    usize_to_fuel(
                                        measure_value(item, &self.program.limits)?.retained,
                                    )?,
                                )?;
                                cost = checked_fuel_add(
                                    cost,
                                    usize_to_fuel(vector_allocation_bytes::<usize>(
                                        frame.body_slots.len(),
                                    )?)?,
                                )?;
                            }
                        }
                    }
                    _ => {}
                }
                self.instruction_meter = Some((self.pc, cost));
            }
            let (metered_pc, fuel_remaining) = self
                .instruction_meter
                .as_mut()
                .ok_or_else(|| execution("instruction meter is missing"))?;
            if *metered_pc != self.pc {
                return Err(execution("instruction meter program counter mismatch"));
            }
            if !burn_fuel(
                &mut self.fuel_used,
                self.program.limits.max_fuel,
                &mut remaining,
                fuel_remaining,
            )? {
                return Ok(StepOutcome::Sliced);
            }
            self.instruction_meter = None;
            match instruction {
                Instruction::Let { slot, value } => {
                    let result = self.eval(&value)?;
                    self.store_precharged(slot, result)?;
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
                    self.precharge_cloned_value(&items[0])?;
                    self.locals[item_slot] =
                        Some(clone_json_value(&items[0], &self.program.limits)?);
                    self.loop_iterations = self.loop_iterations.saturating_add(1);
                    if self.loops.len() >= self.program.limits.max_control_stack {
                        return Err(resource("control stack limit exceeded"));
                    }
                    let body_instruction_count = end_target
                        .checked_sub(self.pc.saturating_add(2))
                        .ok_or_else(|| execution("loop body target is invalid"))?;
                    let body_slot_capacity = body_instruction_count
                        .checked_mul(3)
                        .ok_or_else(|| resource("loop local tracking limit exceeded"))?;
                    self.precharge_bytes(vector_allocation_bytes::<usize>(body_slot_capacity)?)?;
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
                        (frame.index < frame.items.len()).then_some((
                            frame.item_slot,
                            frame.index,
                            frame.body_slots.len(),
                        ))
                    };
                    if let Some((item_slot, item_index, body_slot_count)) = next {
                        let item_retained = measure_value(
                            &self
                                .loops
                                .last()
                                .ok_or_else(|| execution("loop stack underflow"))?
                                .items[item_index],
                            &self.program.limits,
                        )?
                        .retained;
                        self.precharge_bytes(item_retained)?;
                        let clone_bytes = cloned_value_allocation_bytes(
                            &self
                                .loops
                                .last()
                                .ok_or_else(|| execution("loop stack underflow"))?
                                .items[item_index],
                            &self.program.limits,
                        )?
                        .checked_sub(item_retained)
                        .ok_or_else(|| resource("value clone accounting underflowed"))?;
                        self.precharge_bytes(clone_bytes)?;
                        self.precharge_bytes(vector_allocation_bytes::<usize>(body_slot_count)?)?;
                        let item = clone_json_value(
                            &self
                                .loops
                                .last()
                                .ok_or_else(|| execution("loop stack underflow"))?
                                .items[item_index],
                            &self.program.limits,
                        )?;
                        let mut body_slots = Vec::new();
                        body_slots
                            .try_reserve_exact(body_slot_count)
                            .map_err(|_| resource("loop local tracking allocation failed"))?;
                        body_slots.extend_from_slice(
                            &self
                                .loops
                                .last()
                                .ok_or_else(|| execution("loop stack underflow"))?
                                .body_slots,
                        );
                        clear_loop_body_slots(&mut self.locals, &body_slots);
                        self.loop_iterations = self.loop_iterations.saturating_add(1);
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
                    self.precharge_bytes(array_framing_retained_bytes()?)?;
                    self.precharge_bytes(vector_allocation_bytes::<Value>(items.len())?)?;
                    let mut output = Vec::new();
                    output
                        .try_reserve_exact(items.len())
                        .map_err(|_| resource("collection allocation failed"))?;
                    self.work = Some(Work::Map {
                        slot,
                        item_slot,
                        items,
                        index: 0,
                        value,
                        output,
                        fuel_remaining: 0,
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
                    self.precharge_bytes(array_framing_retained_bytes()?)?;
                    self.precharge_bytes(vector_allocation_bytes::<Value>(items.len())?)?;
                    let mut output = Vec::new();
                    output
                        .try_reserve_exact(items.len())
                        .map_err(|_| resource("collection allocation failed"))?;
                    self.work = Some(Work::Filter {
                        slot,
                        item_slot,
                        items,
                        index: 0,
                        predicate,
                        output,
                        fuel_remaining: 0,
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
                    let accumulator = self.eval(&initial)?;
                    self.work = Some(Work::Reduce {
                        slot,
                        item_slot,
                        accumulator_slot,
                        items,
                        index: 0,
                        value,
                        accumulator,
                        fuel_remaining: 0,
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
                    self.precharge_bytes(string_retained_bytes(tool_id.len())?)?;
                    let request = self.request(tool_id, call_site, arguments)?;
                    self.precharge_bytes(vector_allocation_bytes::<ToolRequest>(1)?)?;
                    let mut calls = Vec::new();
                    calls
                        .try_reserve_exact(1)
                        .map_err(|_| resource("tool batch allocation failed"))?;
                    calls.push(request);
                    return self.suspend(slot, calls, false);
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
                    self.precharge_bytes(vector_allocation_bytes::<ToolRequest>(items.len())?)?;
                    let mut calls = Vec::new();
                    calls
                        .try_reserve_exact(items.len())
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
                        fuel_remaining: 0,
                    });
                }
                Instruction::Return { value } => {
                    let output = self.eval(&value)?;
                    let measurement = measure_value(&output, &self.program.limits)?;
                    if measurement.serialized > self.program.limits.max_output_bytes {
                        return Err(resource("output byte limit exceeded"));
                    }
                    self.terminal = true;
                    return Ok(StepOutcome::Complete(output));
                }
            }
        }
    }

    /// Resumes exactly the currently suspended yield. The token is consumed.
    ///
    /// Any invalid resume attempt terminalizes the execution. This makes token
    /// loss explicit and prevents callers from replaying a yield after the
    /// non-clone token has been consumed. Once token, batch identity, and all
    /// response bounds are accepted, any later failure is also terminal.
    pub fn resume(
        &mut self,
        token: ResumeToken,
        responses: Vec<ToolResponse>,
    ) -> Result<(), SandboxError> {
        if self.terminal {
            discard_tool_responses(responses);
            return Err(execution("execution is already terminal"));
        }

        let validation = self.validate_resume(&token, &responses);
        let response_bytes = match validation {
            Ok(bytes) => bytes,
            Err(error) => {
                self.terminalize();
                discard_tool_responses(responses);
                return Err(error);
            }
        };

        // Acceptance point: all caller-controlled data has been checked while
        // the suspension was still intact. From here onward every failure is
        // terminal and the accepted effect can never be yielded again.
        let result = (|| {
            let pending = self
                .pending
                .take()
                .ok_or_else(|| resume_error("execution is not suspended"))?;
            self.live_bytes = self
                .live_bytes
                .checked_add(response_bytes)
                .ok_or_else(|| resource("live byte limit exceeded"))?;
            self.cumulative_bytes = self
                .cumulative_bytes
                .checked_add(response_bytes)
                .ok_or_else(|| resource("cumulative byte limit exceeded"))?;
            self.apply_responses(pending, responses)
        })();
        if result.is_err() {
            self.terminalize();
        }
        result
    }

    fn validate_resume(
        &self,
        token: &ResumeToken,
        responses: &[ToolResponse],
    ) -> Result<usize, SandboxError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| resume_error("execution is not suspended"))?;
        if token.execution_id != self.execution_id
            || token.program_attempt != self.program_attempt
            || token.yield_ordinal != pending.yield_ordinal
        {
            return Err(resume_error(
                "resume token does not match the suspended execution",
            ));
        }
        if responses.len() != pending.requests.len()
            || responses.len() > self.program.limits.max_collection_items
            || responses.len() > self.program.limits.max_fanout
        {
            return Err(resume_error(
                "response count does not match the yielded batch",
            ));
        }
        if !pending.fan_out && responses.len() != 1 {
            return Err(resume_error("single-call response is missing"));
        }
        if responses
            .iter()
            .zip(pending.requests.iter())
            .any(|(response, expected)| (response.call_site, response.dynamic_ordinal) != *expected)
        {
            return Err(resume_error(
                "response identity does not match yielded request order",
            ));
        }

        let mut retained = if pending.fan_out {
            checked_add(
                array_framing_retained_bytes()?,
                vector_allocation_bytes::<Value>(responses.len())?,
            )?
        } else {
            0
        };
        let mut serialized = if pending.fan_out {
            checked_add(2, responses.len().saturating_sub(1))?
        } else {
            0
        };
        for response in responses {
            let output = measure_value(&response.output, &self.program.limits)?;
            let wrapper_depth = output
                .max_depth
                .checked_add(if pending.fan_out { 2 } else { 1 })
                .ok_or_else(|| resource("response nesting limit exceeded"))?;
            if wrapper_depth > self.program.limits.max_nesting {
                return Err(resource("response nesting limit exceeded"));
            }
            let wrapper = response_wrapper_retained_bytes(output.retained)?;
            retained = checked_add(retained, wrapper)?;
            serialized = checked_add(
                serialized,
                response_wrapper_serialized_bytes(response.ok, output.serialized)?,
            )?;
        }
        if serialized > self.program.limits.max_output_bytes {
            return Err(resource("response serialized byte limit exceeded"));
        }
        let next_live = self
            .live_bytes
            .checked_add(retained)
            .ok_or_else(|| resource("live byte limit exceeded"))?;
        let next_cumulative = self
            .cumulative_bytes
            .checked_add(retained)
            .ok_or_else(|| resource("cumulative byte limit exceeded"))?;
        if next_live > self.program.limits.max_live_bytes {
            return Err(resource("live byte limit exceeded"));
        }
        if next_cumulative > self.program.limits.max_cumulative_bytes {
            return Err(resource("cumulative byte limit exceeded"));
        }
        Ok(retained)
    }

    fn apply_responses(
        &mut self,
        pending: Pending,
        responses: Vec<ToolResponse>,
    ) -> Result<(), SandboxError> {
        let mut values = Vec::new();
        if pending.fan_out {
            values
                .try_reserve_exact(responses.len())
                .map_err(|_| resource("response allocation failed"))?;
        }
        for response in responses {
            let mut object = Map::new();
            object.insert(clone_string("ok")?, Value::Bool(response.ok));
            object.insert(clone_string("output")?, response.output);
            if pending.fan_out {
                values.push(Value::Object(object));
            } else {
                self.store_precharged(pending.slot, Value::Object(object))?;
            }
        }
        if pending.fan_out {
            self.store_precharged(pending.slot, Value::Array(values))?;
        }
        self.pc = self
            .pc
            .checked_add(1)
            .ok_or_else(|| execution("program counter overflowed"))?;
        Ok(())
    }

    fn terminalize(&mut self) {
        self.terminal = true;
        self.pending = None;
        self.work = None;
        self.instruction_meter = None;
    }

    fn advance_work(&mut self, remaining: &mut u64) -> Result<bool, SandboxError> {
        let mut work = self
            .work
            .take()
            .ok_or_else(|| execution("work state is missing"))?;
        let done = match &mut work {
            Work::Map {
                slot,
                item_slot,
                items,
                index,
                value,
                output,
                fuel_remaining,
            } => {
                if *index == items.len() {
                    self.store_precharged(*slot, Value::Array(core::mem::take(output)))?;
                    self.locals[*item_slot] = None;
                    self.pc += 1;
                    true
                } else {
                    if *fuel_remaining == 0 {
                        *fuel_remaining = expression_metered_cost_with_overrides(
                            value,
                            &self.locals,
                            &self.program.limits,
                            &[(*item_slot, &items[*index])],
                        )?;
                        *fuel_remaining = checked_fuel_add(
                            *fuel_remaining,
                            usize_to_fuel(
                                measure_value(&items[*index], &self.program.limits)?.retained,
                            )?,
                        )?;
                    }
                    if !burn_fuel(
                        &mut self.fuel_used,
                        self.program.limits.max_fuel,
                        remaining,
                        fuel_remaining,
                    )? {
                        self.work = Some(work);
                        return Ok(false);
                    }
                    self.precharge_cloned_value(&items[*index])?;
                    self.locals[*item_slot] =
                        Some(clone_json_value(&items[*index], &self.program.limits)?);
                    let mapped = self.eval(value)?;
                    output.push(mapped);
                    *index += 1;
                    *fuel_remaining = 0;
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
                fuel_remaining,
            } => {
                if *index == items.len() {
                    self.store_precharged(*slot, Value::Array(core::mem::take(output)))?;
                    self.locals[*item_slot] = None;
                    self.pc += 1;
                    true
                } else {
                    if *fuel_remaining == 0 {
                        *fuel_remaining = expression_metered_cost_with_overrides(
                            predicate,
                            &self.locals,
                            &self.program.limits,
                            &[(*item_slot, &items[*index])],
                        )?;
                        *fuel_remaining = checked_fuel_add(
                            *fuel_remaining,
                            usize_to_fuel(
                                measure_value(&items[*index], &self.program.limits)?.retained,
                            )?,
                        )?;
                    }
                    if !burn_fuel(
                        &mut self.fuel_used,
                        self.program.limits.max_fuel,
                        remaining,
                        fuel_remaining,
                    )? {
                        self.work = Some(work);
                        return Ok(false);
                    }
                    self.precharge_cloned_value(&items[*index])?;
                    self.locals[*item_slot] =
                        Some(clone_json_value(&items[*index], &self.program.limits)?);
                    if expect_bool(self.eval(predicate)?, "filter predicate")? {
                        self.precharge_cloned_value(&items[*index])?;
                        output.push(clone_json_value(&items[*index], &self.program.limits)?);
                    }
                    *index += 1;
                    *fuel_remaining = 0;
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
                fuel_remaining,
            } => {
                if *index == items.len() {
                    let result = core::mem::replace(accumulator, Value::Null);
                    self.store_precharged(*slot, result)?;
                    self.locals[*item_slot] = None;
                    self.locals[*accumulator_slot] = None;
                    self.pc += 1;
                    true
                } else {
                    if *fuel_remaining == 0 {
                        *fuel_remaining = expression_metered_cost_with_overrides(
                            value,
                            &self.locals,
                            &self.program.limits,
                            &[
                                (*item_slot, &items[*index]),
                                (*accumulator_slot, accumulator),
                            ],
                        )?;
                        *fuel_remaining = checked_fuel_add(
                            *fuel_remaining,
                            checked_fuel_add(
                                usize_to_fuel(
                                    measure_value(&items[*index], &self.program.limits)?.retained,
                                )?,
                                usize_to_fuel(
                                    measure_value(accumulator, &self.program.limits)?.retained,
                                )?,
                            )?,
                        )?;
                    }
                    if !burn_fuel(
                        &mut self.fuel_used,
                        self.program.limits.max_fuel,
                        remaining,
                        fuel_remaining,
                    )? {
                        self.work = Some(work);
                        return Ok(false);
                    }
                    self.precharge_cloned_value(&items[*index])?;
                    self.precharge_cloned_value(accumulator)?;
                    self.locals[*item_slot] =
                        Some(clone_json_value(&items[*index], &self.program.limits)?);
                    self.locals[*accumulator_slot] =
                        Some(clone_json_value(accumulator, &self.program.limits)?);
                    *accumulator = self.eval(value)?;
                    *index += 1;
                    *fuel_remaining = 0;
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
                fuel_remaining,
                ..
            } => {
                if *index == items.len() {
                    self.locals[*item_slot] = None;
                    true
                } else {
                    if *fuel_remaining == 0 {
                        *fuel_remaining = expression_metered_cost_with_overrides(
                            arguments,
                            &self.locals,
                            &self.program.limits,
                            &[(*item_slot, &items[*index])],
                        )?;
                        *fuel_remaining = checked_fuel_add(
                            *fuel_remaining,
                            checked_fuel_add(
                                usize_to_fuel(
                                    measure_value(&items[*index], &self.program.limits)?.retained,
                                )?,
                                usize_to_fuel(string_retained_bytes(tool_id.len())?)?,
                            )?,
                        )?;
                    }
                    if !burn_fuel(
                        &mut self.fuel_used,
                        self.program.limits.max_fuel,
                        remaining,
                        fuel_remaining,
                    )? {
                        self.work = Some(work);
                        return Ok(false);
                    }
                    self.precharge_cloned_value(&items[*index])?;
                    self.locals[*item_slot] =
                        Some(clone_json_value(&items[*index], &self.program.limits)?);
                    let args = self.eval(arguments)?;
                    require_object(&args)?;
                    self.precharge_bytes(string_retained_bytes(tool_id.len())?)?;
                    calls.push(self.request(clone_string(tool_id)?, *call_site, args)?);
                    *index += 1;
                    *fuel_remaining = 0;
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
        Ok(true)
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
        self.precharge_bytes(vector_allocation_bytes::<(u32, u64)>(calls.len())?)?;
        let mut requests = Vec::new();
        requests
            .try_reserve_exact(calls.len())
            .map_err(|_| resource("pending request allocation failed"))?;
        for call in &calls {
            requests.push((call.call_site, call.dynamic_ordinal));
        }
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
                program_attempt: self.program_attempt,
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

    fn eval(&mut self, code: &ExprCode) -> Result<Value, SandboxError> {
        let mut stack = Vec::new();
        let capacity = self.program.limits.max_operand_stack.min(code.0.len());
        self.precharge_bytes(vector_allocation_bytes::<Value>(capacity)?)?;
        stack
            .try_reserve_exact(capacity)
            .map_err(|_| resource("operand stack allocation failed"))?;
        for instruction in &code.0 {
            match instruction {
                ExprInstruction::Constant(value) => {
                    self.precharge_cloned_value(value)?;
                    stack.push(clone_json_value(value, &self.program.limits)?);
                }
                ExprInstruction::Load(slot) => {
                    let retained = measure_value(
                        self.locals
                            .get(*slot)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| {
                                execution("local is unavailable in this control-flow path")
                            })?,
                        &self.program.limits,
                    )?
                    .retained;
                    self.precharge_bytes(retained)?;
                    let clone_bytes = cloned_value_allocation_bytes(
                        self.locals
                            .get(*slot)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| {
                                execution("local is unavailable in this control-flow path")
                            })?,
                        &self.program.limits,
                    )?
                    .checked_sub(retained)
                    .ok_or_else(|| resource("value clone accounting underflowed"))?;
                    self.precharge_bytes(clone_bytes)?;
                    let value =
                        self.locals
                            .get(*slot)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| {
                                execution("local is unavailable in this control-flow path")
                            })?;
                    stack.push(clone_json_value(value, &self.program.limits)?);
                }
                ExprInstruction::Path(pointer) => {
                    let value = pop(&mut stack)?;
                    let selected = value
                        .pointer(pointer)
                        .ok_or_else(|| execution("JSON pointer did not resolve"))?;
                    self.precharge_cloned_value(selected)?;
                    stack.push(clone_json_value(selected, &self.program.limits)?);
                }
                ExprInstruction::Array(count) => {
                    let start = stack
                        .len()
                        .checked_sub(*count)
                        .ok_or_else(|| execution("operand stack underflow"))?;
                    self.precharge_bytes(array_framing_retained_bytes()?)?;
                    self.precharge_bytes(vector_allocation_bytes::<Value>(*count)?)?;
                    let mut items = Vec::new();
                    items
                        .try_reserve_exact(*count)
                        .map_err(|_| resource("array allocation failed"))?;
                    while stack.len() > start {
                        items.push(pop(&mut stack)?);
                    }
                    items.reverse();
                    stack.push(Value::Array(items));
                }
                ExprInstruction::Object(keys) => {
                    let start = stack
                        .len()
                        .checked_sub(keys.len())
                        .ok_or_else(|| execution("operand stack underflow"))?;
                    self.precharge_bytes(object_framing_retained_bytes()?)?;
                    for key in keys {
                        self.precharge_bytes(key_retained_bytes(key.len())?)?;
                    }
                    let mut values = Vec::new();
                    self.precharge_bytes(vector_allocation_bytes::<Value>(keys.len())?)?;
                    values
                        .try_reserve_exact(keys.len())
                        .map_err(|_| resource("object staging allocation failed"))?;
                    while stack.len() > start {
                        values.push(pop(&mut stack)?);
                    }
                    values.reverse();
                    let mut object = Map::new();
                    for (key, value) in keys.iter().cloned().zip(values) {
                        object.insert(key, value);
                    }
                    stack.push(Value::Object(object));
                }
                ExprInstruction::Binary(operator) => {
                    let right = pop(&mut stack)?;
                    let left = pop(&mut stack)?;
                    let result = binary(*operator, left, right, &self.program.limits)?;
                    self.precharge_value(&result)?;
                    stack.push(result);
                }
                ExprInstruction::Unary(operator) => {
                    let value = pop(&mut stack)?;
                    let result = unary(*operator, value)?;
                    self.precharge_value(&result)?;
                    stack.push(result);
                }
            }
        }
        if stack.len() != 1 {
            return Err(execution("expression did not produce one value"));
        }
        pop(&mut stack)
    }

    /// Stores a composite whose values and JSON framing were charged before its
    /// backing collection grew. This avoids charging host responses twice when
    /// they are wrapped as inert VM results.
    fn store_precharged(&mut self, slot: usize, value: Value) -> Result<(), SandboxError> {
        let target = self
            .locals
            .get_mut(slot)
            .ok_or_else(|| execution("local slot is invalid"))?;
        if target.is_some() {
            return Err(execution("immutable local was already initialized"));
        }
        *target = Some(value);
        Ok(())
    }

    fn precharge_value(&mut self, value: &Value) -> Result<(), SandboxError> {
        self.precharge_bytes(measure_value(value, &self.program.limits)?.retained)
    }

    fn precharge_cloned_value(&mut self, value: &Value) -> Result<(), SandboxError> {
        self.precharge_bytes(cloned_value_allocation_bytes(value, &self.program.limits)?)
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

fn instruction_metered_cost(
    instruction: &Instruction,
    locals: &[Option<Value>],
    limits: &crate::SandboxLimits,
) -> Result<u64, SandboxError> {
    let mut cost = 1u64;
    let expression_cost = |expression: &ExprCode| {
        expression_metered_cost_with_overrides(expression, locals, limits, &[])
    };
    match instruction {
        Instruction::Let { value, .. }
        | Instruction::Branch {
            condition: value, ..
        } => {
            cost = checked_fuel_add(cost, expression_cost(value)?)?;
        }
        Instruction::Return { value } => {
            let evaluation = expression_metered_cost_with_overrides(value, locals, limits, &[])?;
            // Returning traverses the completed value again to enforce exact
            // compact-JSON output size, so reserve an equal conservative pass.
            cost = checked_fuel_add(cost, checked_fuel_add(evaluation, evaluation)?)?;
        }
        Instruction::LoopStart { collection, .. }
        | Instruction::Map { collection, .. }
        | Instruction::Filter { collection, .. } => {
            cost = checked_fuel_add(cost, expression_cost(collection)?)?;
        }
        Instruction::Reduce {
            collection,
            initial,
            ..
        } => {
            cost = checked_fuel_add(cost, expression_cost(collection)?)?;
            cost = checked_fuel_add(cost, expression_cost(initial)?)?;
        }
        Instruction::Invoke {
            arguments, tool_id, ..
        } => {
            cost = checked_fuel_add(cost, expression_cost(arguments)?)?;
            cost = checked_fuel_add(cost, usize_to_fuel(string_retained_bytes(tool_id.len())?)?)?;
        }
        Instruction::FanOut {
            collection,
            tool_id,
            ..
        } => {
            cost = checked_fuel_add(cost, expression_cost(collection)?)?;
            cost = checked_fuel_add(cost, usize_to_fuel(string_retained_bytes(tool_id.len())?)?)?;
        }
        Instruction::Jump { .. } | Instruction::LoopNext { .. } => {}
    }
    Ok(cost)
}

fn expression_metered_cost_with_overrides(
    code: &ExprCode,
    locals: &[Option<Value>],
    limits: &crate::SandboxLimits,
    overrides: &[(usize, &Value)],
) -> Result<u64, SandboxError> {
    #[derive(Clone, Copy)]
    struct Weight {
        retained: u64,
        clone_cost: u64,
    }

    let mut weights = Vec::new();
    weights
        .try_reserve_exact(code.0.len())
        .map_err(|_| resource("fuel metering allocation failed"))?;
    let mut cost = usize_to_fuel(code.0.len())?.max(1);
    for instruction in &code.0 {
        match instruction {
            ExprInstruction::Constant(value) => {
                let weight = Weight {
                    retained: usize_to_fuel(measure_value(value, limits)?.retained)?,
                    clone_cost: usize_to_fuel(cloned_value_allocation_bytes(value, limits)?)?,
                };
                cost = checked_fuel_add(cost, weight.clone_cost)?;
                weights.push(weight);
            }
            ExprInstruction::Load(slot) => {
                let value = overrides
                    .iter()
                    .find_map(|(candidate, value)| (*candidate == *slot).then_some(*value))
                    .or_else(|| locals.get(*slot).and_then(Option::as_ref))
                    .ok_or_else(|| execution("local is unavailable in this control-flow path"))?;
                let weight = Weight {
                    retained: usize_to_fuel(measure_value(value, limits)?.retained)?,
                    clone_cost: usize_to_fuel(cloned_value_allocation_bytes(value, limits)?)?,
                };
                cost = checked_fuel_add(cost, weight.clone_cost)?;
                weights.push(weight);
            }
            ExprInstruction::Path(pointer) => {
                let weight = weights
                    .pop()
                    .ok_or_else(|| execution("expression meter stack underflow"))?;
                cost = checked_fuel_add(cost, usize_to_fuel(pointer.len())?)?;
                cost = checked_fuel_add(cost, weight.clone_cost)?;
                weights.push(weight);
            }
            ExprInstruction::Array(count) => {
                let start = weights
                    .len()
                    .checked_sub(*count)
                    .ok_or_else(|| execution("expression meter stack underflow"))?;
                let mut children = Weight {
                    retained: 0,
                    clone_cost: 0,
                };
                for child in weights.drain(start..) {
                    children.retained = checked_fuel_add(children.retained, child.retained)?;
                    children.clone_cost = checked_fuel_add(children.clone_cost, child.clone_cost)?;
                }
                let framing = usize_to_fuel(array_framing_retained_bytes()?)?;
                let allocation = usize_to_fuel(vector_allocation_bytes::<Value>(*count)?)?;
                cost = checked_fuel_add(cost, usize_to_fuel(*count)?)?;
                cost = checked_fuel_add(cost, checked_fuel_add(framing, allocation)?)?;
                weights.push(Weight {
                    retained: checked_fuel_add(children.retained, framing)?,
                    clone_cost: checked_fuel_add(
                        children.clone_cost,
                        checked_fuel_add(framing, allocation)?,
                    )?,
                });
            }
            ExprInstruction::Object(keys) => {
                let start = weights
                    .len()
                    .checked_sub(keys.len())
                    .ok_or_else(|| execution("expression meter stack underflow"))?;
                let mut children = Weight {
                    retained: 0,
                    clone_cost: 0,
                };
                for child in weights.drain(start..) {
                    children.retained = checked_fuel_add(children.retained, child.retained)?;
                    children.clone_cost = checked_fuel_add(children.clone_cost, child.clone_cost)?;
                }
                let mut framing = usize_to_fuel(object_framing_retained_bytes()?)?;
                for key in keys {
                    framing =
                        checked_fuel_add(framing, usize_to_fuel(key_retained_bytes(key.len())?)?)?;
                }
                let staging = usize_to_fuel(vector_allocation_bytes::<Value>(keys.len())?)?;
                cost = checked_fuel_add(cost, checked_fuel_add(framing, staging)?)?;
                weights.push(Weight {
                    retained: checked_fuel_add(children.retained, framing)?,
                    clone_cost: checked_fuel_add(
                        children.clone_cost,
                        checked_fuel_add(framing, staging)?,
                    )?,
                });
            }
            ExprInstruction::Binary(operator) => {
                let right = weights
                    .pop()
                    .ok_or_else(|| execution("expression meter stack underflow"))?;
                let left = weights
                    .pop()
                    .ok_or_else(|| execution("expression meter stack underflow"))?;
                if matches!(operator, BinaryOperator::Equal | BinaryOperator::NotEqual) {
                    cost =
                        checked_fuel_add(cost, checked_fuel_add(left.retained, right.retained)?)?;
                }
                cost = checked_fuel_add(cost, usize_to_fuel(primitive_retained_bytes())?)?;
                weights.push(Weight {
                    retained: usize_to_fuel(primitive_retained_bytes())?,
                    clone_cost: usize_to_fuel(cloned_value_allocation_bytes(
                        &Value::Null,
                        limits,
                    )?)?,
                });
            }
            ExprInstruction::Unary(operator) => {
                let operand = weights
                    .pop()
                    .ok_or_else(|| execution("expression meter stack underflow"))?;
                if matches!(
                    operator,
                    UnaryOperator::Sum | UnaryOperator::All | UnaryOperator::Any
                ) {
                    cost = checked_fuel_add(cost, operand.retained)?;
                }
                cost = checked_fuel_add(cost, usize_to_fuel(primitive_retained_bytes())?)?;
                weights.push(Weight {
                    retained: usize_to_fuel(primitive_retained_bytes())?,
                    clone_cost: usize_to_fuel(cloned_value_allocation_bytes(
                        &Value::Null,
                        limits,
                    )?)?,
                });
            }
        }
    }
    if weights.len() != 1 {
        return Err(execution("expression meter did not produce one value"));
    }
    Ok(cost.max(1))
}

fn burn_fuel(
    fuel_used: &mut u64,
    max_fuel: u64,
    slice_remaining: &mut u64,
    work_remaining: &mut u64,
) -> Result<bool, SandboxError> {
    if *work_remaining == 0 {
        return Ok(true);
    }
    let available = max_fuel
        .checked_sub(*fuel_used)
        .ok_or_else(|| resource("fuel limit exceeded"))?;
    if available == 0 {
        return Err(resource("fuel limit exceeded"));
    }
    let charged = (*work_remaining).min(*slice_remaining).min(available);
    *fuel_used = fuel_used
        .checked_add(charged)
        .ok_or_else(|| resource("fuel limit exceeded"))?;
    *slice_remaining -= charged;
    *work_remaining -= charged;
    if *work_remaining > 0 && *slice_remaining > 0 && *fuel_used == max_fuel {
        return Err(resource("fuel limit exceeded"));
    }
    Ok(*work_remaining == 0)
}

fn checked_fuel_add(left: u64, right: u64) -> Result<u64, SandboxError> {
    left.checked_add(right)
        .ok_or_else(|| resource("fuel cost overflowed"))
}

fn usize_to_fuel(value: usize) -> Result<u64, SandboxError> {
    u64::try_from(value).map_err(|_| resource("fuel cost overflowed"))
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

fn response_wrapper_retained_bytes(output_retained: usize) -> Result<usize, SandboxError> {
    let mut total = object_framing_retained_bytes()?;
    total = checked_add(total, key_retained_bytes("ok".len())?)?;
    total = checked_add(total, primitive_retained_bytes())?;
    total = checked_add(total, key_retained_bytes("output".len())?)?;
    checked_add(total, output_retained)
}

fn response_wrapper_serialized_bytes(
    ok: bool,
    output_serialized: usize,
) -> Result<usize, SandboxError> {
    checked_add(response_object_syntax_len(ok)?, output_serialized)
}

fn discard_tool_responses(responses: Vec<ToolResponse>) {
    for response in responses {
        discard_json_value(response.output);
    }
}

fn discard_json_value(value: Value) {
    let mut pending = Vec::new();
    pending.push(value);
    while let Some(value) = pending.pop() {
        match value {
            Value::Array(mut children) => pending.append(&mut children),
            Value::Object(children) => {
                for (_, child) in children {
                    pending.push(child);
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
        }
    }
}

enum CloneOperation<'a> {
    Visit(&'a Value),
    FinishArray(usize),
    FinishObject(Vec<String>),
}

fn clone_json_value(source: &Value, limits: &crate::SandboxLimits) -> Result<Value, SandboxError> {
    let measurement = measure_value(source, limits)?;
    let mut operations = Vec::new();
    operations
        .try_reserve_exact(measurement.nodes.saturating_mul(2))
        .map_err(|_| resource("value clone allocation failed"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(measurement.nodes)
        .map_err(|_| resource("value clone allocation failed"))?;
    operations.push(CloneOperation::Visit(source));

    while let Some(operation) = operations.pop() {
        match operation {
            CloneOperation::Visit(value) => match value {
                Value::Null => values.push(Value::Null),
                Value::Bool(value) => values.push(Value::Bool(*value)),
                Value::Number(value) => values.push(Value::Number(value.clone())),
                Value::String(value) => values.push(Value::String(clone_string(value)?)),
                Value::Array(items) => {
                    operations.push(CloneOperation::FinishArray(items.len()));
                    for item in items.iter().rev() {
                        operations.push(CloneOperation::Visit(item));
                    }
                }
                Value::Object(entries) => {
                    let mut keys = Vec::new();
                    keys.try_reserve_exact(entries.len())
                        .map_err(|_| resource("value clone allocation failed"))?;
                    for key in entries.keys() {
                        keys.push(clone_string(key)?);
                    }
                    operations.push(CloneOperation::FinishObject(keys));
                    for value in entries.values().rev() {
                        operations.push(CloneOperation::Visit(value));
                    }
                }
            },
            CloneOperation::FinishArray(count) => {
                let start = values
                    .len()
                    .checked_sub(count)
                    .ok_or_else(|| execution("value clone stack underflow"))?;
                let mut array = Vec::new();
                array
                    .try_reserve_exact(count)
                    .map_err(|_| resource("value clone allocation failed"))?;
                while values.len() > start {
                    array.push(pop(&mut values)?);
                }
                array.reverse();
                values.push(Value::Array(array));
            }
            CloneOperation::FinishObject(keys) => {
                let start = values
                    .len()
                    .checked_sub(keys.len())
                    .ok_or_else(|| execution("value clone stack underflow"))?;
                let mut children = Vec::new();
                children
                    .try_reserve_exact(keys.len())
                    .map_err(|_| resource("value clone allocation failed"))?;
                while values.len() > start {
                    children.push(pop(&mut values)?);
                }
                children.reverse();
                let mut object = Map::new();
                for (key, value) in keys.into_iter().zip(children) {
                    object.insert(key, value);
                }
                values.push(Value::Object(object));
            }
        }
    }
    if values.len() != 1 {
        return Err(execution("value clone did not produce one value"));
    }
    pop(&mut values)
}

fn clone_string(source: &str) -> Result<String, SandboxError> {
    let mut cloned = String::new();
    cloned
        .try_reserve_exact(source.len())
        .map_err(|_| resource("string clone allocation failed"))?;
    cloned.push_str(source);
    Ok(cloned)
}

fn values_equal(
    left: &Value,
    right: &Value,
    limits: &crate::SandboxLimits,
) -> Result<bool, SandboxError> {
    let mut pending = Vec::new();
    pending
        .try_reserve_exact(1)
        .map_err(|_| resource("equality allocation failed"))?;
    pending.push((left, right));
    while let Some((left, right)) = pending.pop() {
        match (left, right) {
            (Value::Null, Value::Null) => {}
            (Value::Bool(left), Value::Bool(right)) if left == right => {}
            (Value::Number(left), Value::Number(right)) if left == right => {}
            (Value::String(left), Value::String(right)) if left == right => {}
            (Value::Array(left), Value::Array(right)) if left.len() == right.len() => {
                if left.len() > limits.max_collection_items {
                    return Err(resource("collection item limit exceeded"));
                }
                pending
                    .try_reserve(left.len())
                    .map_err(|_| resource("equality allocation failed"))?;
                for pair in left.iter().zip(right).rev() {
                    pending.push(pair);
                }
            }
            (Value::Object(left), Value::Object(right)) if left.len() == right.len() => {
                if left.len() > limits.max_collection_items {
                    return Err(resource("collection item limit exceeded"));
                }
                pending
                    .try_reserve(left.len())
                    .map_err(|_| resource("equality allocation failed"))?;
                for ((left_key, left_value), (right_key, right_value)) in
                    left.iter().zip(right).rev()
                {
                    if left_key != right_key {
                        return Ok(false);
                    }
                    pending.push((left_value, right_value));
                }
            }
            _ => return Ok(false),
        }
    }
    Ok(true)
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
    let capacity = body
        .len()
        .checked_mul(3)
        .ok_or_else(|| resource("loop local tracking limit exceeded"))?;
    slots
        .try_reserve_exact(capacity)
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
fn binary(
    operator: BinaryOperator,
    left: Value,
    right: Value,
    limits: &crate::SandboxLimits,
) -> Result<Value, SandboxError> {
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
        Equal => Value::Bool(values_equal(&left, &right, limits)?),
        NotEqual => Value::Bool(!values_equal(&left, &right, limits)?),
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

fn response_object_syntax_len(ok: bool) -> Result<usize, SandboxError> {
    let mut total = 2usize;
    total = checked_add(total, serialized_string_len("ok")?)?;
    total = checked_add(total, 1)?;
    total = checked_add(total, if ok { 4 } else { 5 })?;
    total = checked_add(total, 1)?;
    total = checked_add(total, serialized_string_len("output")?)?;
    checked_add(total, 1)
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
    use alloc::vec;
    use serde_json::json;

    fn execution(program: serde_json::Value, id: u64) -> Execution {
        let limits = SandboxLimits::default();
        execution_with_limits(program, id, limits)
    }

    fn execution_with_limits(
        program: serde_json::Value,
        id: u64,
        limits: SandboxLimits,
    ) -> Execution {
        let bytes = serde_json::to_vec(&program).unwrap();
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

    fn next_event(execution: &mut Execution, slice_fuel: u64) -> Result<StepOutcome, SandboxError> {
        loop {
            let before = execution.metrics().fuel_used;
            match execution.step(slice_fuel)? {
                StepOutcome::Sliced => {
                    assert!(
                        execution.metrics().fuel_used > before,
                        "every sliced transition must consume fuel"
                    );
                }
                outcome => return Ok(outcome),
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
        let (batch, token) = match next_event(&mut vm, 1).unwrap() {
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
            next_event(&mut vm, 1).unwrap(),
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
        let (first_batch, first_token) = match next_event(&mut first, 1).unwrap() {
            StepOutcome::Yielded { batch, resume } => (batch, resume),
            _ => unreachable!(),
        };
        let (second_batch, _second_token) = match next_event(&mut second, 1).unwrap() {
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
        assert_eq!(
            second.step(100).unwrap_err().code(),
            SandboxErrorCode::Execution
        );
        assert_eq!(
            first.step(100).unwrap_err().code(),
            SandboxErrorCode::InvalidResume
        );

        let mut valid = execution(program(), 3);
        let (valid_batch, valid_token) = match next_event(&mut valid, 1).unwrap() {
            StepOutcome::Yielded { batch, resume } => (batch, resume),
            _ => unreachable!(),
        };
        valid
            .resume(
                valid_token,
                vec![ToolResponse::success(&valid_batch.calls()[0], json!(3))],
            )
            .unwrap();
        assert!(matches!(
            next_event(&mut valid, 1).unwrap(),
            StepOutcome::Complete(_)
        ));
        drop(first_batch);
    }

    #[test]
    fn invalid_resume_terminalizes_without_reopening_or_replaying_the_yield() {
        let program = json!({"version":1,"body":[
            {"kind":"fan_out","name":"results","tool_id":"read","item":"item",
             "collection":{"kind":"array","items":[{"kind":"integer","value":1},{"kind":"integer","value":2}]},"max_calls":2,
             "arguments":{"kind":"object","entries":[{"key":"value","value":{"kind":"variable","name":"item"}}]}},
            {"kind":"return","value":{"kind":"variable","name":"results"}}
        ]});
        let suspend = |id| {
            let mut vm = execution(program.clone(), id);
            let yielded = next_event(&mut vm, 1).unwrap();
            match yielded {
                StepOutcome::Yielded { batch, resume } => (vm, batch, resume),
                other => panic!("expected yield, got {other:?}"),
            }
        };

        let (mut count_vm, count_batch, count_token) = suspend(42);
        let first = ToolResponse::success(&count_batch.calls()[0], json!("first"));
        assert_eq!(
            count_vm
                .resume(count_token, vec![first])
                .unwrap_err()
                .code(),
            SandboxErrorCode::InvalidResume
        );
        assert_eq!(
            count_vm.step(100).unwrap_err().code(),
            SandboxErrorCode::Execution
        );

        let (mut order_vm, order_batch, order_token) = suspend(43);
        let reversed = vec![
            ToolResponse::success(&order_batch.calls()[1], json!("second")),
            ToolResponse::success(&order_batch.calls()[0], json!("first")),
        ];
        assert_eq!(
            order_vm.resume(order_token, reversed).unwrap_err().code(),
            SandboxErrorCode::InvalidResume
        );
        assert_eq!(
            order_vm.step(100).unwrap_err().code(),
            SandboxErrorCode::Execution
        );

        let (mut valid_vm, valid_batch, valid_token) = suspend(44);
        let ordered = vec![
            ToolResponse::success(&valid_batch.calls()[0], json!("first")),
            ToolResponse::success(&valid_batch.calls()[1], json!("second")),
        ];
        valid_vm.resume(valid_token, ordered).unwrap();
        assert!(matches!(
            next_event(&mut valid_vm, 1).unwrap(),
            StepOutcome::Complete(_)
        ));
        assert_eq!(
            valid_vm.step(100).unwrap_err().code(),
            SandboxErrorCode::Execution
        );
    }

    #[test]
    fn tool_response_boundary_rejects_hostile_values_before_vm_mutation() {
        let program = json!({"version":1,"body":[
            {"kind":"invoke","name":"result","tool_id":"read","arguments":{"kind":"object","entries":[]}},
            {"kind":"return","value":{"kind":"variable","name":"result"}}
        ]});
        let limits = SandboxLimits {
            max_nesting: 16,
            max_collection_items: 2,
            ..SandboxLimits::default()
        };
        let attempt = |id: u64, output: Value| {
            let mut vm = execution_with_limits(program.clone(), id, limits);
            let (batch, token) = match next_event(&mut vm, 1).unwrap() {
                StepOutcome::Yielded { batch, resume } => (batch, resume),
                other => panic!("expected yield, got {other:?}"),
            };
            let before = vm.metrics();
            let result = vm.resume(
                token,
                vec![ToolResponse::success(&batch.calls()[0], output)],
            );
            (vm, before, result)
        };

        let mut too_deep = Value::Null;
        for _ in 0..limits.max_nesting {
            too_deep = Value::Array(vec![too_deep]);
        }
        let mut extremely_deep = Value::Null;
        for _ in 0..10_000 {
            extremely_deep = Value::Array(vec![extremely_deep]);
        }
        for (id, hostile) in [
            (51, too_deep),
            (52, Value::Array(vec![Value::Null; 3])),
            (53, json!(1.5)),
            (54, extremely_deep),
        ] {
            let (mut vm, before, result) = attempt(id, hostile);
            assert_eq!(result.unwrap_err().code(), SandboxErrorCode::ResourceLimit);
            assert_eq!(vm.metrics().retained_bytes, before.retained_bytes);
            assert_eq!(vm.metrics().cumulative_bytes, before.cumulative_bytes);
            assert_eq!(vm.metrics().yields, before.yields);
            assert_eq!(vm.step(1).unwrap_err().code(), SandboxErrorCode::Execution);
        }

        let inert = json!({"kind":"invoke","tool_id":"danger","arguments":{"kind":"return"}});
        let mut vm = execution(program, 55);
        let (batch, token) = match next_event(&mut vm, 1).unwrap() {
            StepOutcome::Yielded { batch, resume } => (batch, resume),
            other => panic!("expected yield, got {other:?}"),
        };
        vm.resume(
            token,
            vec![ToolResponse::success(&batch.calls()[0], inert.clone())],
        )
        .unwrap();
        let output = match next_event(&mut vm, 1).unwrap() {
            StepOutcome::Complete(output) => output,
            other => panic!("expected completion, got {other:?}"),
        };
        assert_eq!(output, json!({"ok":true,"output":inert}));
        assert_eq!(vm.metrics().yields, 1, "response-shaped code stays inert");
    }

    #[test]
    fn response_serialized_boundary_is_inclusive() {
        let program = json!({"version":1,"body":[
            {"kind":"invoke","name":"result","tool_id":"read","arguments":{"kind":"object","entries":[]}},
            {"kind":"return","value":{"kind":"variable","name":"result"}}
        ]});
        let output = json!("x");
        let output_bytes = measure_value(&output, &SandboxLimits::default())
            .unwrap()
            .serialized;
        let exact = response_wrapper_serialized_bytes(true, output_bytes).unwrap();
        let attempt = |limit, id| {
            let limits = SandboxLimits {
                max_output_bytes: limit,
                ..SandboxLimits::default()
            };
            let mut vm = execution_with_limits(program.clone(), id, limits);
            let (batch, token) = match next_event(&mut vm, 1).unwrap() {
                StepOutcome::Yielded { batch, resume } => (batch, resume),
                _ => unreachable!(),
            };
            vm.resume(
                token,
                vec![ToolResponse::success(&batch.calls()[0], output.clone())],
            )
        };
        assert_eq!(
            attempt(exact - 1, 55).unwrap_err().code(),
            SandboxErrorCode::ResourceLimit
        );
        assert!(attempt(exact, 56).is_ok());
        assert!(attempt(exact + 1, 57).is_ok());
    }

    #[test]
    fn every_nonterminal_slice_charges_fuel_and_fuel_limit_is_terminal() {
        let program = json!({"version":1,"body":[
            {"kind":"let","name":"value","value":{"kind":"integer","value":1}},
            {"kind":"return","value":{"kind":"variable","name":"value"}}
        ]});
        let mut vm = execution(program, 43);
        let mut slices = 0usize;
        loop {
            let before = vm.metrics().fuel_used;
            match vm.step(1).unwrap() {
                StepOutcome::Sliced => {
                    slices += 1;
                    assert_eq!(vm.metrics().fuel_used, before + 1);
                }
                StepOutcome::Complete(value) => {
                    assert_eq!(value, json!(1));
                    break;
                }
                StepOutcome::Yielded { .. } => unreachable!(),
            }
        }
        assert!(
            slices > 1,
            "expression work must resume across slice-one calls"
        );

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
    fn slice_one_completes_expression_larger_than_the_default_slice_deterministically() {
        let items = (0..64)
            .map(|value| json!({"kind":"integer","value":value}))
            .collect::<Vec<_>>();
        let program = json!({"version":1,"body":[{
            "kind":"return","value":{"kind":"array","items":items}
        }]});
        let limits = SandboxLimits {
            max_slice_fuel: 1,
            ..SandboxLimits::default()
        };
        let execute = |id| {
            let mut vm = execution_with_limits(program.clone(), id, limits);
            let mut slices = 0u64;
            loop {
                let before = vm.metrics().fuel_used;
                match vm.step(1).unwrap() {
                    StepOutcome::Sliced => {
                        slices += 1;
                        assert_eq!(vm.metrics().fuel_used, before + 1);
                    }
                    StepOutcome::Complete(output) => {
                        return (output, vm.metrics(), slices);
                    }
                    StepOutcome::Yielded { .. } => unreachable!(),
                }
            }
        };
        let first = execute(58);
        let second = execute(58);
        assert_eq!(first, second);
        assert!(
            first.2 > SandboxLimits::default().max_slice_fuel,
            "the expression must exceed the former atomic slice size"
        );
    }

    #[test]
    fn fuel_boundary_is_exact_at_n_minus_one_n_and_n_plus_one() {
        let program = json!({"version":1,"body":[{
            "kind":"return","value":{"kind":"null"}
        }]});
        let mut baseline = execution(program.clone(), 59);
        assert_eq!(
            next_event(&mut baseline, 1).unwrap(),
            StepOutcome::Complete(Value::Null)
        );
        let exact = baseline.metrics().fuel_used;
        let attempt = |max_fuel, id| {
            let limits = SandboxLimits {
                max_fuel,
                max_slice_fuel: 1,
                ..SandboxLimits::default()
            };
            let mut vm = execution_with_limits(program.clone(), id, limits);
            let outcome = next_event(&mut vm, 1);
            (outcome, vm.metrics())
        };
        assert_eq!(
            attempt(exact - 1, 60).0.unwrap_err().code(),
            SandboxErrorCode::ResourceLimit
        );
        for (limit, id) in [(exact, 61), (exact + 1, 62)] {
            let (outcome, metrics) = attempt(limit, id);
            assert_eq!(outcome.unwrap(), StepOutcome::Complete(Value::Null));
            assert_eq!(metrics.fuel_used, exact);
        }
    }

    #[test]
    fn identical_public_resume_sequences_are_deterministic() {
        let program = json!({"version":1,"body":[
            {"kind":"invoke","name":"result","tool_id":"read","arguments":{"kind":"object","entries":[
                {"key":"id","value":{"kind":"integer","value":7}}
            ]}},
            {"kind":"return","value":{"kind":"variable","name":"result"}}
        ]});
        let execute = || {
            let mut vm = execution(program.clone(), 63);
            let (batch, token) = match next_event(&mut vm, 1).unwrap() {
                StepOutcome::Yielded { batch, resume } => (batch, resume),
                other => panic!("expected yield, got {other:?}"),
            };
            let request = batch.calls()[0].clone();
            vm.resume(
                token,
                vec![ToolResponse::success(&request, json!({"value":9}))],
            )
            .unwrap();
            let output = match next_event(&mut vm, 1).unwrap() {
                StepOutcome::Complete(output) => output,
                other => panic!("expected completion, got {other:?}"),
            };
            (request, output, vm.metrics())
        };
        assert_eq!(execute(), execute());
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
        let (batch, token) = match next_event(&mut vm, 1).unwrap() {
            StepOutcome::Yielded { batch, resume } => (batch, resume),
            other => panic!("expected first yield, got {other:?}"),
        };
        vm.resume(
            token,
            vec![ToolResponse::success(&batch.calls()[0], json!(1))],
        )
        .unwrap();
        loop {
            match vm.step(1) {
                Ok(StepOutcome::Sliced) => {}
                Err(error) => {
                    assert_eq!(error.code(), SandboxErrorCode::ResourceLimit);
                    break;
                }
                Ok(other) => panic!("expected yield-limit failure, got {other:?}"),
            }
        }
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
            next_event(&mut vm, 1).unwrap_err().code(),
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
            next_event(&mut compile_with("1234", 7), 1).unwrap(),
            StepOutcome::Complete(json!("1234"))
        );
        assert_eq!(
            next_event(&mut compile_with("12345", 7), 1).unwrap(),
            StepOutcome::Complete(json!("12345"))
        );
        assert_eq!(
            next_event(&mut compile_with("123456", 7), 1)
                .unwrap_err()
                .code(),
            SandboxErrorCode::ResourceLimit
        );

        let response_program = json!({"version":1,"body":[
            {"kind":"invoke","name":"result","tool_id":"read","arguments":{"kind":"object","entries":[]}},
            {"kind":"return","value":{"kind":"variable","name":"result"}}
        ]});
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
        let execute_response = |limit| -> Result<usize, SandboxError> {
            let mut vm = build_response_vm(limit);
            let (batch, token) = match next_event(&mut vm, 1)? {
                StepOutcome::Yielded { batch, resume } => (batch, resume),
                _ => {
                    return Err(SandboxError::new(
                        SandboxErrorCode::Execution,
                        "expected response test yield",
                    ));
                }
            };
            vm.resume(
                token,
                vec![ToolResponse::success(&batch.calls()[0], json!("x"))],
            )?;
            if !matches!(next_event(&mut vm, 1)?, StepOutcome::Complete(_)) {
                return Err(SandboxError::new(
                    SandboxErrorCode::Execution,
                    "expected response test completion",
                ));
            }
            Ok(vm.metrics().cumulative_bytes)
        };
        let expected = execute_response(SandboxLimits::default().max_cumulative_bytes).unwrap();
        assert_eq!(execute_response(expected).unwrap(), expected);
        assert_eq!(
            execute_response(expected - 1).unwrap_err().code(),
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
        let mut baseline = build_intermediate_vm(64 * 1024);
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
