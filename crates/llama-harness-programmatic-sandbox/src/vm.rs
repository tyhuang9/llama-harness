use crate::{
    accounting::{
        array_framing_retained_bytes, checked_add, key_retained_bytes,
        object_framing_retained_bytes, primitive_retained_bytes, serialized_string_len,
        string_retained_bytes, vector_allocation_bytes, ValueMeasurement,
    },
    compiler::{ExprCode, ExprInstruction, Instruction, VerifiedProgram},
    value::{RuntimeNode, RuntimeValue},
    BinaryOperator, SandboxError, SandboxErrorCode, SandboxLimits, UnaryOperator,
};
use alloc::{string::String, vec, vec::Vec};
use serde_json::{Map, Number, Value};

/// Maximum string bytes copied or decoded by one fuel unit.
pub const MAX_ATOMIC_STRING_BYTES: usize = 64;
/// Maximum key bytes copied or compared by one fuel unit.
///
/// Keys use the same 64-byte quantum as every other runtime string so an
/// object construction, lookup, equality check, or materialization transition
/// never scans or copies more input string data than one atomic fuel unit.
pub const MAX_ATOMIC_KEY_BYTES: usize = MAX_ATOMIC_STRING_BYTES;

/// Host-supplied identifier that scopes resume tokens to one live execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExecutionId(pub u64);

/// One inert, statically named tool request yielded by the sandbox.
pub struct ToolRequest {
    /// Execution that issued this request.
    pub execution_id: ExecutionId,
    /// Program attempt that issued this request.
    pub program_attempt: u32,
    /// Static bytecode call site that issued this request.
    pub call_site: u32,
    /// Monotonic occurrence number within the execution.
    pub dynamic_ordinal: u64,
    /// Statically declared tool identifier.
    pub tool_id: String,
    /// Fully materialized, object-shaped request arguments.
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
    /// Returns the requests in their required response order.
    pub fn calls(&self) -> &[ToolRequest] {
        &self.calls
    }

    /// Reports whether every request is a static read-only fan-out call.
    pub const fn requests_read_only_fan_out(&self) -> bool {
        self.read_only_fan_out
    }
}

/// One validated inert response. Construction borrows the host's JSON value,
/// so rejected hostile trees remain owned and safely disposable by the host.
///
/// There is no unchecked constructor and no public field access, so a caller
/// cannot inject an unmeasured recursive value into a suspended execution.
///
/// ```compile_fail
/// use llama_harness_programmatic_sandbox::ToolResponse;
///
/// let _ = ToolResponse { call_site: 0 };
/// ```
pub struct ToolResponse {
    call_site: u32,
    dynamic_ordinal: u64,
    ok: bool,
    output: RuntimeValue,
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
    /// Validates and copies a successful response under host limits.
    /// Resume independently reapplies the execution's effective limits.
    pub fn success(
        request: &ToolRequest,
        output: &Value,
        limits: &SandboxLimits,
    ) -> Result<Self, SandboxError> {
        limits.validate()?;
        let output = RuntimeValue::from_json(output, limits)?;
        if output.measurement().serialized > limits.max_output_bytes {
            return Err(resource("response serialized byte limit exceeded"));
        }
        Ok(Self {
            call_site: request.call_site,
            dynamic_ordinal: request.dynamic_ordinal,
            ok: true,
            output,
        })
    }

    /// Creates an inert, checked failure response for a yielded request.
    pub fn failure(request: &ToolRequest) -> Self {
        Self {
            call_site: request.call_site,
            dynamic_ordinal: request.dynamic_ordinal,
            ok: false,
            output: RuntimeValue::null(),
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
#[non_exhaustive]
pub enum StepOutcome {
    /// The requested scheduling slice ended after consuming all supplied fuel.
    Sliced,
    /// Tool execution must be authorized and completed before resumption.
    Yielded {
        /// Ordered inert requests to execute.
        batch: ToolBatch,
        /// Single-use proof required by [`Execution::resume`].
        resume: ResumeToken,
    },
    /// The terminal JSON result of the program.
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Monotonic resource and control-flow counters for an execution.
pub struct ExecutionMetrics {
    /// Fuel units consumed so far.
    pub fuel_used: u64,
    /// Tool-yield batches emitted so far.
    pub yields: u32,
    /// Branch conditions evaluated so far.
    pub branches: u64,
    /// Loop items entered so far.
    pub loop_iterations: u64,
    /// Read-only fan-out batches emitted so far.
    pub fanout_batches: u32,
    /// Conservatively charged VM-owned bytes.
    pub retained_bytes: usize,
    /// Cumulatively charged VM-owned bytes.
    pub cumulative_bytes: usize,
}

struct LoopFrame {
    collection: RuntimeValue,
    index: usize,
    item_slot: usize,
    body_target: usize,
    start_pc: usize,
}

enum Work {
    Map {
        pc: usize,
        collection: RuntimeValue,
        index: usize,
        output: Vec<RuntimeValue>,
        measurement: ValueMeasurement,
    },
    Filter {
        pc: usize,
        collection: RuntimeValue,
        index: usize,
        output: Vec<RuntimeValue>,
        measurement: ValueMeasurement,
    },
    Reduce {
        pc: usize,
        collection: RuntimeValue,
        index: usize,
        accumulator: RuntimeValue,
    },
    FanOut {
        pc: usize,
        collection: RuntimeValue,
        index: usize,
        calls: Vec<ToolRequest>,
    },
}

struct Pending {
    yield_ordinal: u32,
    slot: usize,
    requests: Vec<(u32, u64)>,
    fan_out: bool,
}

#[derive(Clone, Copy)]
enum ExprLocation {
    Let(usize),
    Branch(usize),
    LoopCollection(usize),
    MapCollection(usize),
    MapValue(usize),
    FilterCollection(usize),
    FilterPredicate(usize),
    ReduceCollection(usize),
    ReduceInitial(usize),
    ReduceValue(usize),
    InvokeArguments(usize),
    FanOutCollection(usize),
    FanOutArguments(usize),
    Return(usize),
}

enum EvalPurpose {
    Let,
    Branch,
    LoopCollection,
    MapCollection,
    MapValue,
    FilterCollection,
    FilterPredicate,
    ReduceCollection,
    ReduceInitial { collection: RuntimeValue },
    ReduceValue,
    InvokeArguments,
    FanOutCollection,
    FanOutArguments,
    Return,
}

enum MaterializePurpose {
    Invoke,
    FanOut,
    Return,
}

enum Active {
    Eval {
        purpose: EvalPurpose,
        state: EvalState,
    },
    Materialize {
        purpose: MaterializePurpose,
        state: MaterializeState,
    },
    ClearLoop {
        frame_index: usize,
        slot_index: usize,
        next_item: RuntimeValue,
    },
}

struct EvalState {
    location: ExprLocation,
    ip: usize,
    stack: Vec<RuntimeValue>,
    operation: Option<EvalOperation>,
}

enum EvalOperation {
    LiteralString {
        opcode: usize,
        offset: usize,
        output: String,
        serialized: usize,
    },
    Array(BuildArray),
    Object(BuildObject),
    Equality(EqualityState),
    Aggregate(AggregateState),
    Path(PathState),
}

struct BuildArray {
    remaining: usize,
    values: Vec<RuntimeValue>,
    reverse_index: usize,
    reversing: bool,
    measurement: ValueMeasurement,
}

struct BuildObject {
    opcode: usize,
    remaining: usize,
    values: Vec<RuntimeValue>,
    reverse_index: usize,
    phase: ObjectPhase,
    entry_index: usize,
    entries: Vec<(String, RuntimeValue)>,
    measurement: ValueMeasurement,
}

enum ObjectPhase {
    Pop,
    Reverse,
    Entries,
}

struct EqualityState {
    negate: bool,
    frames: Vec<EqualityFrame>,
    string: Option<StringEquality>,
    equal: bool,
}

enum EqualityFrame {
    Values(RuntimeValue, RuntimeValue),
    Array(RuntimeValue, RuntimeValue, usize),
    Object(RuntimeValue, RuntimeValue, usize),
}

struct StringEquality {
    left: RuntimeValue,
    right: RuntimeValue,
    offset: usize,
}

struct AggregateState {
    operator: UnaryOperator,
    collection: RuntimeValue,
    index: usize,
    integer: i64,
    boolean: bool,
}

struct PathState {
    opcode: usize,
    current: RuntimeValue,
    pointer_offset: usize,
    segment: String,
    phase: PathPhase,
}

enum PathPhase {
    Decode,
    ObjectLookup(usize),
    ArrayIndex { offset: usize, value: usize },
}

struct MaterializeState {
    current: Option<RuntimeValue>,
    completed: Option<Value>,
    frames: Vec<MaterializeFrame>,
    string: Option<MaterializeString>,
}

enum MaterializeFrame {
    Array {
        source: RuntimeValue,
        index: usize,
        output: Vec<Value>,
    },
    Object {
        source: RuntimeValue,
        index: usize,
        output: Map<String, Value>,
    },
}

struct MaterializeString {
    source: RuntimeValue,
    offset: usize,
    output: String,
}

/// A single bounded execution of a [`VerifiedProgram`].
pub struct Execution {
    program: VerifiedProgram,
    execution_id: ExecutionId,
    program_attempt: u32,
    pc: usize,
    locals: Vec<Option<RuntimeValue>>,
    loops: Vec<LoopFrame>,
    work: Option<Work>,
    active: Option<Active>,
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

#[derive(Clone, Copy)]
enum WorkKind {
    Map,
    Filter,
    FanOut,
}

impl Execution {
    /// Starts a first attempt of a verified program under `execution_id`.
    pub fn new(program: VerifiedProgram, execution_id: ExecutionId) -> Result<Self, SandboxError> {
        Self::with_attempt(program, execution_id, 0)
    }

    /// Starts a specific host-assigned attempt of a verified program.
    pub fn with_attempt(
        program: VerifiedProgram,
        execution_id: ExecutionId,
        program_attempt: u32,
    ) -> Result<Self, SandboxError> {
        let structural_bytes = checked_add(
            vector_allocation_bytes::<Option<RuntimeValue>>(program.local_count)?,
            vector_allocation_bytes::<LoopFrame>(program.limits.max_control_stack)?,
        )?;
        ensure_byte_limits(structural_bytes, structural_bytes, &program.limits)?;
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
            active: None,
            pending: None,
            fuel_used: 0,
            yields: 0,
            dynamic_ordinal: 0,
            live_bytes: structural_bytes,
            cumulative_bytes: structural_bytes,
            branches: 0,
            loop_iterations: 0,
            fanout_batches: 0,
            terminal: false,
        })
    }

    /// Returns the execution's current monotonic metrics.
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

    /// Advances execution. One fuel unit performs at most one bytecode, value
    /// node, collection-item, or bounded string-chunk transition.
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
            self.terminalize();
        }
        result
    }

    fn step_inner(&mut self, slice_fuel: u64) -> Result<StepOutcome, SandboxError> {
        let mut remaining = slice_fuel;
        loop {
            if remaining == 0 {
                return Ok(StepOutcome::Sliced);
            }
            if self.active.is_some() {
                if let Some(outcome) = self.advance_active(&mut remaining)? {
                    return Ok(outcome);
                }
                continue;
            }
            if self.work.is_some() {
                if let Some(outcome) = self.advance_work(&mut remaining)? {
                    return Ok(outcome);
                }
                continue;
            }
            if self.pc >= self.program.code.len() {
                return Err(execution("verified program reached no-return state"));
            }
            self.start_instruction(&mut remaining)?;
        }
    }

    fn start_instruction(&mut self, remaining: &mut u64) -> Result<(), SandboxError> {
        charge_one(&mut self.fuel_used, self.program.limits.max_fuel, remaining)?;
        let pc = self.pc;
        match self
            .program
            .code
            .get(pc)
            .ok_or_else(|| execution("program counter is invalid"))?
        {
            Instruction::Let { .. } => self.start_eval(ExprLocation::Let(pc), EvalPurpose::Let)?,
            Instruction::Branch { .. } => {
                self.start_eval(ExprLocation::Branch(pc), EvalPurpose::Branch)?
            }
            Instruction::Jump { target } => self.pc = *target,
            Instruction::LoopStart { .. } => self.start_eval(
                ExprLocation::LoopCollection(pc),
                EvalPurpose::LoopCollection,
            )?,
            Instruction::LoopNext { .. } => self.start_loop_next()?,
            Instruction::Map { .. } => {
                self.start_eval(ExprLocation::MapCollection(pc), EvalPurpose::MapCollection)?
            }
            Instruction::Filter { .. } => self.start_eval(
                ExprLocation::FilterCollection(pc),
                EvalPurpose::FilterCollection,
            )?,
            Instruction::Reduce { .. } => self.start_eval(
                ExprLocation::ReduceCollection(pc),
                EvalPurpose::ReduceCollection,
            )?,
            Instruction::Invoke { .. } => self.start_eval(
                ExprLocation::InvokeArguments(pc),
                EvalPurpose::InvokeArguments,
            )?,
            Instruction::FanOut { .. } => self.start_eval(
                ExprLocation::FanOutCollection(pc),
                EvalPurpose::FanOutCollection,
            )?,
            Instruction::Return { .. } => {
                self.start_eval(ExprLocation::Return(pc), EvalPurpose::Return)?
            }
        }
        Ok(())
    }

    fn start_eval(
        &mut self,
        location: ExprLocation,
        purpose: EvalPurpose,
    ) -> Result<(), SandboxError> {
        let stack_bytes =
            vector_allocation_bytes::<RuntimeValue>(self.program.limits.max_operand_stack)?;
        precharge(
            &mut self.live_bytes,
            &mut self.cumulative_bytes,
            stack_bytes,
            &self.program.limits,
        )?;
        let mut stack = Vec::new();
        stack
            .try_reserve_exact(self.program.limits.max_operand_stack)
            .map_err(|_| resource("operand stack allocation failed"))?;
        self.active = Some(Active::Eval {
            purpose,
            state: EvalState {
                location,
                ip: 0,
                stack,
                operation: None,
            },
        });
        Ok(())
    }

    fn start_materialize(
        &mut self,
        value: RuntimeValue,
        purpose: MaterializePurpose,
    ) -> Result<(), SandboxError> {
        let measurement = value.measurement();
        let frame_capacity = measurement.max_depth.saturating_add(1);
        let bytes = checked_add(
            measurement.retained,
            vector_allocation_bytes::<MaterializeFrame>(frame_capacity)?,
        )?;
        precharge(
            &mut self.live_bytes,
            &mut self.cumulative_bytes,
            bytes,
            &self.program.limits,
        )?;
        let mut frames = Vec::new();
        frames
            .try_reserve_exact(frame_capacity)
            .map_err(|_| resource("materialization stack allocation failed"))?;
        self.active = Some(Active::Materialize {
            purpose,
            state: MaterializeState {
                current: Some(value),
                completed: None,
                frames,
                string: None,
            },
        });
        Ok(())
    }

    fn advance_active(&mut self, remaining: &mut u64) -> Result<Option<StepOutcome>, SandboxError> {
        let mut active = self
            .active
            .take()
            .ok_or_else(|| execution("active state is missing"))?;
        charge_one(&mut self.fuel_used, self.program.limits.max_fuel, remaining)?;
        match &mut active {
            Active::Eval { purpose, state } => {
                let result = advance_eval(
                    &self.program,
                    &self.locals,
                    state,
                    &mut self.live_bytes,
                    &mut self.cumulative_bytes,
                )?;
                if let Some(value) = result {
                    let purpose = core::mem::replace(purpose, EvalPurpose::Let);
                    self.finish_eval(purpose, value)
                } else {
                    self.active = Some(active);
                    Ok(None)
                }
            }
            Active::Materialize { purpose, state } => {
                if let Some(value) = advance_materialize(state)? {
                    let purpose = core::mem::replace(purpose, MaterializePurpose::Return);
                    self.finish_materialize(purpose, value)
                } else {
                    self.active = Some(active);
                    Ok(None)
                }
            }
            Active::ClearLoop {
                frame_index,
                slot_index,
                next_item,
            } => {
                let frame = self
                    .loops
                    .get(*frame_index)
                    .ok_or_else(|| execution("loop frame is invalid"))?;
                let body_slots = match self.program.code.get(frame.start_pc) {
                    Some(Instruction::LoopStart { body_slots, .. }) => body_slots,
                    _ => return Err(execution("loop metadata is invalid")),
                };
                if let Some(slot) = body_slots.get(*slot_index) {
                    let local = self
                        .locals
                        .get_mut(*slot)
                        .ok_or_else(|| execution("loop local slot is invalid"))?;
                    *local = None;
                    *slot_index += 1;
                    self.active = Some(active);
                    Ok(None)
                } else {
                    let frame = self
                        .loops
                        .get_mut(*frame_index)
                        .ok_or_else(|| execution("loop frame is invalid"))?;
                    precharge_cached_clone(
                        &mut self.live_bytes,
                        &mut self.cumulative_bytes,
                        next_item,
                        &self.program.limits,
                    )?;
                    self.locals[frame.item_slot] = Some(next_item.clone());
                    frame.index += 1;
                    self.loop_iterations = self.loop_iterations.saturating_add(1);
                    self.pc = frame.body_target;
                    Ok(None)
                }
            }
        }
    }

    fn finish_eval(
        &mut self,
        purpose: EvalPurpose,
        value: RuntimeValue,
    ) -> Result<Option<StepOutcome>, SandboxError> {
        match purpose {
            EvalPurpose::Let => {
                let slot = match self.program.code.get(self.pc) {
                    Some(Instruction::Let { slot, .. }) => *slot,
                    _ => return Err(execution("let state changed")),
                };
                self.store(slot, value)?;
                self.pc += 1;
            }
            EvalPurpose::Branch => {
                let false_target = match self.program.code.get(self.pc) {
                    Some(Instruction::Branch { false_target, .. }) => *false_target,
                    _ => return Err(execution("branch state changed")),
                };
                self.branches = self.branches.saturating_add(1);
                self.pc = if expect_bool(&value, "branch condition")? {
                    self.pc + 1
                } else {
                    false_target
                };
            }
            EvalPurpose::LoopCollection => self.finish_loop_collection(value)?,
            EvalPurpose::MapCollection => self.start_collection_work(value, WorkKind::Map)?,
            EvalPurpose::MapValue => self.finish_map_item(value)?,
            EvalPurpose::FilterCollection => self.start_collection_work(value, WorkKind::Filter)?,
            EvalPurpose::FilterPredicate => self.finish_filter_item(value)?,
            EvalPurpose::ReduceCollection => self.start_eval(
                ExprLocation::ReduceInitial(self.pc),
                EvalPurpose::ReduceInitial { collection: value },
            )?,
            EvalPurpose::ReduceInitial { collection } => {
                self.start_reduce_work(collection, value)?
            }
            EvalPurpose::ReduceValue => self.finish_reduce_item(value)?,
            EvalPurpose::InvokeArguments => {
                require_object(&value)?;
                self.start_materialize(value, MaterializePurpose::Invoke)?;
            }
            EvalPurpose::FanOutCollection => self.start_collection_work(value, WorkKind::FanOut)?,
            EvalPurpose::FanOutArguments => {
                require_object(&value)?;
                self.start_materialize(value, MaterializePurpose::FanOut)?;
            }
            EvalPurpose::Return => {
                if value.measurement().serialized > self.program.limits.max_output_bytes {
                    return Err(resource("output byte limit exceeded"));
                }
                self.start_materialize(value, MaterializePurpose::Return)?;
            }
        }
        Ok(None)
    }

    fn finish_materialize(
        &mut self,
        purpose: MaterializePurpose,
        value: Value,
    ) -> Result<Option<StepOutcome>, SandboxError> {
        match purpose {
            MaterializePurpose::Return => {
                self.terminal = true;
                Ok(Some(StepOutcome::Complete(value)))
            }
            MaterializePurpose::Invoke => {
                let (slot, tool_id, call_site) = match self.program.code.get(self.pc) {
                    Some(Instruction::Invoke {
                        slot,
                        tool_id,
                        call_site,
                        ..
                    }) => (*slot, clone_string(tool_id)?, *call_site),
                    _ => return Err(execution("invoke state changed")),
                };
                precharge(
                    &mut self.live_bytes,
                    &mut self.cumulative_bytes,
                    string_retained_bytes(tool_id.capacity())?,
                    &self.program.limits,
                )?;
                let call = self.request(tool_id, call_site, value)?;
                Ok(Some(self.suspend(slot, vec![call], false)?))
            }
            MaterializePurpose::FanOut => {
                let (tool_id, call_site) = match self.program.code.get(self.pc) {
                    Some(Instruction::FanOut {
                        tool_id, call_site, ..
                    }) => (clone_string(tool_id)?, *call_site),
                    _ => return Err(execution("fan-out state changed")),
                };
                precharge(
                    &mut self.live_bytes,
                    &mut self.cumulative_bytes,
                    string_retained_bytes(tool_id.capacity())?,
                    &self.program.limits,
                )?;
                let call = self.request(tool_id, call_site, value)?;
                match self.work.as_mut() {
                    Some(Work::FanOut { index, calls, .. }) => {
                        calls.push(call);
                        *index += 1;
                    }
                    _ => return Err(execution("fan-out work state changed")),
                }
                Ok(None)
            }
        }
    }

    fn finish_loop_collection(&mut self, value: RuntimeValue) -> Result<(), SandboxError> {
        let items = expect_array(&value, "loop collection")?;
        let (item_slot, max_iterations, end_target) = match self.program.code.get(self.pc) {
            Some(Instruction::LoopStart {
                item_slot,
                max_iterations,
                end_target,
                ..
            }) => (*item_slot, *max_iterations, *end_target),
            _ => return Err(execution("loop state changed")),
        };
        if items.len() > max_iterations {
            return Err(resource("loop iteration limit exceeded"));
        }
        if items.is_empty() {
            self.pc = end_target;
            return Ok(());
        }
        if self.loops.len() >= self.program.limits.max_control_stack {
            return Err(resource("control stack limit exceeded"));
        }
        let first = items[0].clone();
        precharge_cached_clone(
            &mut self.live_bytes,
            &mut self.cumulative_bytes,
            &first,
            &self.program.limits,
        )?;
        self.locals[item_slot] = Some(first);
        self.loops.push(LoopFrame {
            collection: value,
            index: 0,
            item_slot,
            body_target: self.pc + 1,
            start_pc: self.pc,
        });
        self.loop_iterations = self.loop_iterations.saturating_add(1);
        self.pc += 1;
        Ok(())
    }

    fn start_loop_next(&mut self) -> Result<(), SandboxError> {
        let body_target = match self.program.code.get(self.pc) {
            Some(Instruction::LoopNext { body_target }) => *body_target,
            _ => return Err(execution("loop-next state changed")),
        };
        let frame_index = self
            .loops
            .len()
            .checked_sub(1)
            .ok_or_else(|| execution("loop frame is missing"))?;
        let frame = &self.loops[frame_index];
        if frame.body_target != body_target {
            return Err(execution("loop frame target mismatch"));
        }
        let items = expect_array(&frame.collection, "loop collection")?;
        if frame.index + 1 >= items.len() {
            let frame = self
                .loops
                .pop()
                .ok_or_else(|| execution("loop frame is missing"))?;
            self.locals[frame.item_slot] = None;
            self.pc += 1;
            return Ok(());
        }
        self.active = Some(Active::ClearLoop {
            frame_index,
            slot_index: 0,
            next_item: items[frame.index + 1].clone(),
        });
        Ok(())
    }

    fn advance_work(&mut self, remaining: &mut u64) -> Result<Option<StepOutcome>, SandboxError> {
        charge_one(&mut self.fuel_used, self.program.limits.max_fuel, remaining)?;
        let mut work = self
            .work
            .take()
            .ok_or_else(|| execution("work state is missing"))?;
        match &mut work {
            Work::Map {
                pc,
                collection,
                index,
                output,
                measurement,
            } => {
                let items = expect_array(collection, "map collection")?;
                if *index == items.len() {
                    let value = RuntimeValue::array_measured(core::mem::take(output), *measurement);
                    let (slot, item_slot) = match self.program.code.get(*pc) {
                        Some(Instruction::Map {
                            slot, item_slot, ..
                        }) => (*slot, *item_slot),
                        _ => return Err(execution("map state changed")),
                    };
                    self.locals[item_slot] = None;
                    self.store(slot, value)?;
                    self.pc += 1;
                } else {
                    let item_slot = match self.program.code.get(*pc) {
                        Some(Instruction::Map { item_slot, .. }) => *item_slot,
                        _ => return Err(execution("map state changed")),
                    };
                    assign_scoped(
                        &mut self.locals,
                        item_slot,
                        &items[*index],
                        &mut self.live_bytes,
                        &mut self.cumulative_bytes,
                        &self.program.limits,
                    )?;
                    let location = ExprLocation::MapValue(*pc);
                    self.work = Some(work);
                    self.start_eval(location, EvalPurpose::MapValue)?;
                    return Ok(None);
                }
            }
            Work::Filter {
                pc,
                collection,
                index,
                output,
                measurement,
            } => {
                let items = expect_array(collection, "filter collection")?;
                if *index == items.len() {
                    let value = RuntimeValue::array_measured(core::mem::take(output), *measurement);
                    let (slot, item_slot) = match self.program.code.get(*pc) {
                        Some(Instruction::Filter {
                            slot, item_slot, ..
                        }) => (*slot, *item_slot),
                        _ => return Err(execution("filter state changed")),
                    };
                    self.locals[item_slot] = None;
                    self.store(slot, value)?;
                    self.pc += 1;
                } else {
                    let item_slot = match self.program.code.get(*pc) {
                        Some(Instruction::Filter { item_slot, .. }) => *item_slot,
                        _ => return Err(execution("filter state changed")),
                    };
                    assign_scoped(
                        &mut self.locals,
                        item_slot,
                        &items[*index],
                        &mut self.live_bytes,
                        &mut self.cumulative_bytes,
                        &self.program.limits,
                    )?;
                    let location = ExprLocation::FilterPredicate(*pc);
                    self.work = Some(work);
                    self.start_eval(location, EvalPurpose::FilterPredicate)?;
                    return Ok(None);
                }
            }
            Work::Reduce {
                pc,
                collection,
                index,
                accumulator,
            } => {
                let items = expect_array(collection, "reduce collection")?;
                if *index == items.len() {
                    let (slot, item_slot, accumulator_slot) = match self.program.code.get(*pc) {
                        Some(Instruction::Reduce {
                            slot,
                            item_slot,
                            accumulator_slot,
                            ..
                        }) => (*slot, *item_slot, *accumulator_slot),
                        _ => return Err(execution("reduce state changed")),
                    };
                    self.locals[item_slot] = None;
                    self.locals[accumulator_slot] = None;
                    self.store(slot, accumulator.clone())?;
                    self.pc += 1;
                } else {
                    let (item_slot, accumulator_slot) = match self.program.code.get(*pc) {
                        Some(Instruction::Reduce {
                            item_slot,
                            accumulator_slot,
                            ..
                        }) => (*item_slot, *accumulator_slot),
                        _ => return Err(execution("reduce state changed")),
                    };
                    assign_scoped(
                        &mut self.locals,
                        item_slot,
                        &items[*index],
                        &mut self.live_bytes,
                        &mut self.cumulative_bytes,
                        &self.program.limits,
                    )?;
                    assign_scoped(
                        &mut self.locals,
                        accumulator_slot,
                        accumulator,
                        &mut self.live_bytes,
                        &mut self.cumulative_bytes,
                        &self.program.limits,
                    )?;
                    let location = ExprLocation::ReduceValue(*pc);
                    self.work = Some(work);
                    self.start_eval(location, EvalPurpose::ReduceValue)?;
                    return Ok(None);
                }
            }
            Work::FanOut {
                pc,
                collection,
                index,
                calls,
            } => {
                let items = expect_array(collection, "fan-out collection")?;
                if *index == items.len() {
                    let (slot, item_slot) = match self.program.code.get(*pc) {
                        Some(Instruction::FanOut {
                            slot, item_slot, ..
                        }) => (*slot, *item_slot),
                        _ => return Err(execution("fan-out state changed")),
                    };
                    self.locals[item_slot] = None;
                    return Ok(Some(self.suspend(slot, core::mem::take(calls), true)?));
                }
                let item_slot = match self.program.code.get(*pc) {
                    Some(Instruction::FanOut { item_slot, .. }) => *item_slot,
                    _ => return Err(execution("fan-out state changed")),
                };
                assign_scoped(
                    &mut self.locals,
                    item_slot,
                    &items[*index],
                    &mut self.live_bytes,
                    &mut self.cumulative_bytes,
                    &self.program.limits,
                )?;
                let location = ExprLocation::FanOutArguments(*pc);
                self.work = Some(work);
                self.start_eval(location, EvalPurpose::FanOutArguments)?;
                return Ok(None);
            }
        }
        Ok(None)
    }

    fn start_collection_work(
        &mut self,
        collection: RuntimeValue,
        kind: WorkKind,
    ) -> Result<(), SandboxError> {
        let items = expect_array(&collection, "collection operation")?;
        let pc = self.pc;
        let bound = match (kind, self.program.code.get(pc)) {
            (WorkKind::Map, Some(Instruction::Map { max_items, .. }))
            | (WorkKind::Filter, Some(Instruction::Filter { max_items, .. })) => *max_items,
            (WorkKind::FanOut, Some(Instruction::FanOut { max_calls, .. })) => *max_calls,
            _ => return Err(execution("collection work state changed")),
        };
        if items.len() > bound {
            return Err(resource("collection operation limit exceeded"));
        }
        match kind {
            WorkKind::Map | WorkKind::Filter => {
                let allocation = vector_allocation_bytes::<RuntimeValue>(items.len())?;
                precharge(
                    &mut self.live_bytes,
                    &mut self.cumulative_bytes,
                    checked_add(array_framing_retained_bytes()?, allocation)?,
                    &self.program.limits,
                )?;
                let mut output = Vec::new();
                output
                    .try_reserve_exact(items.len())
                    .map_err(|_| resource("collection output allocation failed"))?;
                let measurement = ValueMeasurement {
                    retained: array_framing_retained_bytes()?,
                    serialized: 2,
                    nodes: 1,
                    collection_items: 0,
                    max_depth: 1,
                };
                self.work = Some(if matches!(kind, WorkKind::Map) {
                    Work::Map {
                        pc,
                        collection,
                        index: 0,
                        output,
                        measurement,
                    }
                } else {
                    Work::Filter {
                        pc,
                        collection,
                        index: 0,
                        output,
                        measurement,
                    }
                });
            }
            WorkKind::FanOut => {
                let allocation = vector_allocation_bytes::<ToolRequest>(items.len())?;
                precharge(
                    &mut self.live_bytes,
                    &mut self.cumulative_bytes,
                    allocation,
                    &self.program.limits,
                )?;
                let mut calls = Vec::new();
                calls
                    .try_reserve_exact(items.len())
                    .map_err(|_| resource("fan-out request allocation failed"))?;
                self.work = Some(Work::FanOut {
                    pc,
                    collection,
                    index: 0,
                    calls,
                });
            }
        }
        Ok(())
    }

    fn start_reduce_work(
        &mut self,
        collection: RuntimeValue,
        accumulator: RuntimeValue,
    ) -> Result<(), SandboxError> {
        let items = expect_array(&collection, "reduce collection")?;
        let max_items = match self.program.code.get(self.pc) {
            Some(Instruction::Reduce { max_items, .. }) => *max_items,
            _ => return Err(execution("reduce state changed")),
        };
        if items.len() > max_items {
            return Err(resource("collection operation limit exceeded"));
        }
        self.work = Some(Work::Reduce {
            pc: self.pc,
            collection,
            index: 0,
            accumulator,
        });
        Ok(())
    }

    fn finish_map_item(&mut self, value: RuntimeValue) -> Result<(), SandboxError> {
        match self.work.as_mut() {
            Some(Work::Map {
                index,
                output,
                measurement,
                ..
            }) => {
                append_array_measurement(measurement, value.measurement(), output.len())?;
                output.push(value);
                *index += 1;
                Ok(())
            }
            _ => Err(execution("map work state changed")),
        }
    }

    fn finish_filter_item(&mut self, value: RuntimeValue) -> Result<(), SandboxError> {
        let include = expect_bool(&value, "filter predicate")?;
        match self.work.as_mut() {
            Some(Work::Filter {
                collection,
                index,
                output,
                measurement,
                ..
            }) => {
                if include {
                    let item = expect_array(collection, "filter collection")?[*index].clone();
                    precharge_cached_clone(
                        &mut self.live_bytes,
                        &mut self.cumulative_bytes,
                        &item,
                        &self.program.limits,
                    )?;
                    append_array_measurement(measurement, item.measurement(), output.len())?;
                    output.push(item);
                }
                *index += 1;
                Ok(())
            }
            _ => Err(execution("filter work state changed")),
        }
    }

    fn finish_reduce_item(&mut self, value: RuntimeValue) -> Result<(), SandboxError> {
        match self.work.as_mut() {
            Some(Work::Reduce {
                index, accumulator, ..
            }) => {
                *accumulator = value;
                *index += 1;
                Ok(())
            }
            _ => Err(execution("reduce work state changed")),
        }
    }

    /// Resumes exactly the current suspension. Every invalid attempt is terminal.
    pub fn resume(
        &mut self,
        token: ResumeToken,
        responses: Vec<ToolResponse>,
    ) -> Result<(), SandboxError> {
        if self.terminal {
            return Err(execution("execution is already terminal"));
        }
        let retained = match self.validate_resume(&token, &responses) {
            Ok(retained) => retained,
            Err(error) => {
                self.terminalize();
                return Err(error);
            }
        };
        let result = (|| {
            let pending = self
                .pending
                .take()
                .ok_or_else(|| resume_error("execution is not suspended"))?;
            precharge(
                &mut self.live_bytes,
                &mut self.cumulative_bytes,
                retained,
                &self.program.limits,
            )?;
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
        let mut retained = if pending.fan_out {
            checked_add(
                array_framing_retained_bytes()?,
                vector_allocation_bytes::<RuntimeValue>(responses.len())?,
            )?
        } else {
            0
        };
        let mut serialized = if pending.fan_out {
            checked_add(2, responses.len().saturating_sub(1))?
        } else {
            0
        };
        for (response, expected) in responses.iter().zip(&pending.requests) {
            if (response.call_site, response.dynamic_ordinal) != *expected {
                return Err(resume_error(
                    "response identity does not match yielded request order",
                ));
            }
            let output = response.output.measurement();
            let wrapper_depth = output
                .max_depth
                .checked_add(if pending.fan_out { 2 } else { 1 })
                .ok_or_else(|| resource("response nesting limit exceeded"))?;
            if wrapper_depth > self.program.limits.max_nesting
                || output.collection_items > self.program.limits.max_collection_items
                || output.retained > self.program.limits.max_live_bytes
                || output.retained > self.program.limits.max_cumulative_bytes
            {
                return Err(resource("response value exceeds effective limits"));
            }
            retained = checked_add(retained, response_wrapper_retained(output.retained)?)?;
            serialized = checked_add(
                serialized,
                response_wrapper_serialized(response.ok, output.serialized)?,
            )?;
        }
        if serialized > self.program.limits.max_output_bytes {
            return Err(resource("response serialized byte limit exceeded"));
        }
        ensure_byte_limits(
            self.live_bytes
                .checked_add(retained)
                .ok_or_else(|| resource("live byte limit exceeded"))?,
            self.cumulative_bytes
                .checked_add(retained)
                .ok_or_else(|| resource("cumulative byte limit exceeded"))?,
            &self.program.limits,
        )?;
        Ok(retained)
    }

    fn apply_responses(
        &mut self,
        pending: Pending,
        responses: Vec<ToolResponse>,
    ) -> Result<(), SandboxError> {
        if pending.fan_out {
            let mut values = Vec::new();
            values
                .try_reserve_exact(responses.len())
                .map_err(|_| resource("response allocation failed"))?;
            for response in responses {
                values.push(RuntimeValue::response(response.ok, response.output)?);
            }
            self.store(pending.slot, RuntimeValue::array(values)?)?;
        } else {
            let response = responses
                .into_iter()
                .next()
                .ok_or_else(|| resume_error("single-call response is missing"))?;
            self.store(
                pending.slot,
                RuntimeValue::response(response.ok, response.output)?,
            )?;
        }
        self.pc += 1;
        Ok(())
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
        let allocation = vector_allocation_bytes::<(u32, u64)>(calls.len())?;
        precharge(
            &mut self.live_bytes,
            &mut self.cumulative_bytes,
            allocation,
            &self.program.limits,
        )?;
        let mut requests = Vec::new();
        requests
            .try_reserve_exact(calls.len())
            .map_err(|_| resource("pending request allocation failed"))?;
        requests.extend(
            calls
                .iter()
                .map(|call| (call.call_site, call.dynamic_ordinal)),
        );
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

    fn store(&mut self, slot: usize, value: RuntimeValue) -> Result<(), SandboxError> {
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

    fn terminalize(&mut self) {
        self.terminal = true;
        self.pending = None;
        self.active = None;
        self.work = None;
    }
}

fn expression(
    program: &VerifiedProgram,
    location: ExprLocation,
) -> Result<&ExprCode, SandboxError> {
    let (pc, selector) = match location {
        ExprLocation::Let(pc) => (pc, 0),
        ExprLocation::Branch(pc) => (pc, 0),
        ExprLocation::LoopCollection(pc) => (pc, 0),
        ExprLocation::MapCollection(pc) => (pc, 0),
        ExprLocation::MapValue(pc) => (pc, 1),
        ExprLocation::FilterCollection(pc) => (pc, 0),
        ExprLocation::FilterPredicate(pc) => (pc, 1),
        ExprLocation::ReduceCollection(pc) => (pc, 0),
        ExprLocation::ReduceInitial(pc) => (pc, 1),
        ExprLocation::ReduceValue(pc) => (pc, 2),
        ExprLocation::InvokeArguments(pc) => (pc, 0),
        ExprLocation::FanOutCollection(pc) => (pc, 0),
        ExprLocation::FanOutArguments(pc) => (pc, 1),
        ExprLocation::Return(pc) => (pc, 0),
    };
    let instruction = program
        .code
        .get(pc)
        .ok_or_else(|| execution("expression program counter is invalid"))?;
    match (instruction, selector) {
        (Instruction::Let { value, .. }, 0)
        | (
            Instruction::Branch {
                condition: value, ..
            },
            0,
        )
        | (
            Instruction::LoopStart {
                collection: value, ..
            },
            0,
        )
        | (
            Instruction::Map {
                collection: value, ..
            },
            0,
        )
        | (
            Instruction::Filter {
                collection: value, ..
            },
            0,
        )
        | (
            Instruction::Reduce {
                collection: value, ..
            },
            0,
        )
        | (
            Instruction::Invoke {
                arguments: value, ..
            },
            0,
        )
        | (
            Instruction::FanOut {
                collection: value, ..
            },
            0,
        )
        | (Instruction::Return { value }, 0) => Ok(value),
        (Instruction::Map { value, .. }, 1)
        | (
            Instruction::Filter {
                predicate: value, ..
            },
            1,
        )
        | (Instruction::Reduce { initial: value, .. }, 1)
        | (
            Instruction::FanOut {
                arguments: value, ..
            },
            1,
        )
        | (Instruction::Reduce { value, .. }, 2) => Ok(value),
        _ => Err(execution("expression selector is invalid")),
    }
}

fn advance_eval(
    program: &VerifiedProgram,
    locals: &[Option<RuntimeValue>],
    state: &mut EvalState,
    live: &mut usize,
    cumulative: &mut usize,
) -> Result<Option<RuntimeValue>, SandboxError> {
    if let Some(mut operation) = state.operation.take() {
        if !advance_eval_operation(program, state, &mut operation, live, cumulative)? {
            state.operation = Some(operation);
            return Ok(None);
        }
        if state.ip == expression(program, state.location)?.0.len() {
            return finish_eval_state(state);
        }
        return Ok(None);
    }

    let code = expression(program, state.location)?;
    let opcode_index = state.ip;
    let opcode = code
        .0
        .get(opcode_index)
        .ok_or_else(|| execution("expression instruction is missing"))?;
    state.ip += 1;
    match opcode {
        ExprInstruction::Constant(Value::Null) => state.stack.push(RuntimeValue::null()),
        ExprInstruction::Constant(Value::Bool(value)) => {
            state.stack.push(RuntimeValue::boolean(*value));
        }
        ExprInstruction::Constant(Value::Number(value)) => {
            state.stack.push(RuntimeValue::integer(
                value
                    .as_i64()
                    .ok_or_else(|| execution("constant integer is invalid"))?,
            ));
        }
        ExprInstruction::Constant(Value::String(value)) => {
            precharge(
                live,
                cumulative,
                string_retained_bytes(value.len())?,
                &program.limits,
            )?;
            let mut output = String::new();
            output
                .try_reserve_exact(value.len())
                .map_err(|_| resource("literal string allocation failed"))?;
            state.operation = Some(EvalOperation::LiteralString {
                opcode: opcode_index,
                offset: 0,
                output,
                serialized: 2,
            });
        }
        ExprInstruction::Constant(_) => {
            return Err(execution("composite bytecode constant is invalid"));
        }
        ExprInstruction::Load(slot) => {
            let value = locals
                .get(*slot)
                .and_then(Option::as_ref)
                .ok_or_else(|| execution("local is unavailable in this control-flow path"))?;
            precharge_cached_clone(live, cumulative, value, &program.limits)?;
            state.stack.push(value.clone());
        }
        ExprInstruction::Path(pointer) => {
            let current = pop_runtime(&mut state.stack)?;
            if pointer.is_empty() {
                state.stack.push(current);
            } else {
                precharge(
                    live,
                    cumulative,
                    string_retained_bytes(pointer.len())?,
                    &program.limits,
                )?;
                let mut segment = String::new();
                segment
                    .try_reserve_exact(pointer.len())
                    .map_err(|_| resource("JSON pointer allocation failed"))?;
                state.operation = Some(EvalOperation::Path(PathState {
                    opcode: opcode_index,
                    current,
                    pointer_offset: 1,
                    segment,
                    phase: PathPhase::Decode,
                }));
            }
        }
        ExprInstruction::Array(count) => {
            if state.stack.len() < *count {
                return Err(execution("operand stack underflow"));
            }
            let allocation = vector_allocation_bytes::<RuntimeValue>(*count)?;
            precharge(
                live,
                cumulative,
                checked_add(array_framing_retained_bytes()?, allocation)?,
                &program.limits,
            )?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(*count)
                .map_err(|_| resource("array allocation failed"))?;
            state.operation = Some(EvalOperation::Array(BuildArray {
                remaining: *count,
                values,
                reverse_index: 0,
                reversing: false,
                measurement: ValueMeasurement {
                    retained: array_framing_retained_bytes()?,
                    serialized: 2,
                    nodes: 1,
                    collection_items: 0,
                    max_depth: 1,
                },
            }));
        }
        ExprInstruction::Object(keys) => {
            if state.stack.len() < keys.len() {
                return Err(execution("operand stack underflow"));
            }
            let allocations = checked_add(
                vector_allocation_bytes::<RuntimeValue>(keys.len())?,
                vector_allocation_bytes::<(String, RuntimeValue)>(keys.len())?,
            )?;
            precharge(
                live,
                cumulative,
                checked_add(object_framing_retained_bytes()?, allocations)?,
                &program.limits,
            )?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(keys.len())
                .map_err(|_| resource("object value allocation failed"))?;
            let mut entries = Vec::new();
            entries
                .try_reserve_exact(keys.len())
                .map_err(|_| resource("object entry allocation failed"))?;
            state.operation = Some(EvalOperation::Object(BuildObject {
                opcode: opcode_index,
                remaining: keys.len(),
                values,
                reverse_index: 0,
                phase: ObjectPhase::Pop,
                entry_index: 0,
                entries,
                measurement: ValueMeasurement {
                    retained: object_framing_retained_bytes()?,
                    serialized: 2,
                    nodes: 1,
                    collection_items: 0,
                    max_depth: 1,
                },
            }));
        }
        ExprInstruction::Binary(operator) => {
            let right = pop_runtime(&mut state.stack)?;
            let left = pop_runtime(&mut state.stack)?;
            if matches!(operator, BinaryOperator::Equal | BinaryOperator::NotEqual) {
                let depth = left
                    .measurement()
                    .max_depth
                    .checked_add(right.measurement().max_depth)
                    .and_then(|value| value.checked_add(2))
                    .ok_or_else(|| resource("equality work limit exceeded"))?;
                precharge(
                    live,
                    cumulative,
                    vector_allocation_bytes::<EqualityFrame>(depth)?,
                    &program.limits,
                )?;
                let mut frames = Vec::new();
                frames
                    .try_reserve_exact(depth)
                    .map_err(|_| resource("equality allocation failed"))?;
                frames.push(EqualityFrame::Values(left, right));
                state.operation = Some(EvalOperation::Equality(EqualityState {
                    negate: matches!(operator, BinaryOperator::NotEqual),
                    frames,
                    string: None,
                    equal: true,
                }));
            } else {
                precharge(
                    live,
                    cumulative,
                    primitive_retained_bytes(),
                    &program.limits,
                )?;
                state.stack.push(binary_scalar(*operator, &left, &right)?);
            }
        }
        ExprInstruction::Unary(operator) => {
            let value = pop_runtime(&mut state.stack)?;
            if matches!(
                operator,
                UnaryOperator::Sum | UnaryOperator::All | UnaryOperator::Any
            ) {
                expect_array(&value, "aggregate operand")?;
                state.operation = Some(EvalOperation::Aggregate(AggregateState {
                    operator: *operator,
                    collection: value,
                    index: 0,
                    integer: 0,
                    boolean: matches!(operator, UnaryOperator::All),
                }));
            } else {
                precharge(
                    live,
                    cumulative,
                    primitive_retained_bytes(),
                    &program.limits,
                )?;
                state.stack.push(unary_scalar(*operator, &value)?);
            }
        }
    }
    if state.operation.is_none() && state.ip == code.0.len() {
        finish_eval_state(state)
    } else {
        Ok(None)
    }
}

fn finish_eval_state(state: &mut EvalState) -> Result<Option<RuntimeValue>, SandboxError> {
    if state.stack.len() != 1 {
        return Err(execution("expression did not produce exactly one value"));
    }
    Ok(Some(pop_runtime(&mut state.stack)?))
}

fn advance_eval_operation(
    program: &VerifiedProgram,
    state: &mut EvalState,
    operation: &mut EvalOperation,
    live: &mut usize,
    cumulative: &mut usize,
) -> Result<bool, SandboxError> {
    match operation {
        EvalOperation::LiteralString {
            opcode,
            offset,
            output,
            serialized,
        } => {
            let source = match expression(program, state.location)?.0.get(*opcode) {
                Some(ExprInstruction::Constant(Value::String(source))) => source,
                _ => return Err(execution("literal string state changed")),
            };
            let end = utf8_chunk_end(source, *offset, MAX_ATOMIC_STRING_BYTES);
            for byte in source.as_bytes().get(*offset..end).unwrap_or_default() {
                *serialized = checked_add(*serialized, escaped_byte_len(*byte))?;
            }
            output.push_str(source.get(*offset..end).unwrap_or_default());
            *offset = end;
            if *offset == source.len() {
                state.stack.push(RuntimeValue::string_measured(
                    core::mem::take(output),
                    *serialized,
                )?);
                Ok(true)
            } else {
                Ok(false)
            }
        }
        EvalOperation::Array(build) => advance_array_build(build, &mut state.stack),
        EvalOperation::Object(build) => advance_object_build(
            program,
            state.location,
            build,
            &mut state.stack,
            live,
            cumulative,
        ),
        EvalOperation::Equality(equality) => {
            if advance_equality(equality)? {
                precharge(
                    live,
                    cumulative,
                    primitive_retained_bytes(),
                    &program.limits,
                )?;
                state.stack.push(RuntimeValue::boolean(if equality.negate {
                    !equality.equal
                } else {
                    equality.equal
                }));
                Ok(true)
            } else {
                Ok(false)
            }
        }
        EvalOperation::Aggregate(aggregate) => {
            if let Some(value) = advance_aggregate(aggregate)? {
                precharge(
                    live,
                    cumulative,
                    primitive_retained_bytes(),
                    &program.limits,
                )?;
                state.stack.push(value);
                Ok(true)
            } else {
                Ok(false)
            }
        }
        EvalOperation::Path(path) => {
            if let Some(value) = advance_path(program, state.location, path)? {
                precharge_cached_clone(live, cumulative, &value, &program.limits)?;
                state.stack.push(value);
                Ok(true)
            } else {
                Ok(false)
            }
        }
    }
}

fn advance_array_build(
    build: &mut BuildArray,
    stack: &mut Vec<RuntimeValue>,
) -> Result<bool, SandboxError> {
    if !build.reversing {
        if build.remaining > 0 {
            let value = pop_runtime(stack)?;
            append_array_measurement(
                &mut build.measurement,
                value.measurement(),
                build.values.len(),
            )?;
            build.values.push(value);
            build.remaining -= 1;
            return Ok(false);
        }
        build.reversing = true;
    }
    let len = build.values.len();
    if build.reverse_index < len / 2 {
        build
            .values
            .swap(build.reverse_index, len - build.reverse_index - 1);
        build.reverse_index += 1;
        return Ok(false);
    }
    stack.push(RuntimeValue::array_measured(
        core::mem::take(&mut build.values),
        build.measurement,
    ));
    Ok(true)
}

fn advance_object_build(
    program: &VerifiedProgram,
    location: ExprLocation,
    build: &mut BuildObject,
    stack: &mut Vec<RuntimeValue>,
    live: &mut usize,
    cumulative: &mut usize,
) -> Result<bool, SandboxError> {
    let keys = match expression(program, location)?.0.get(build.opcode) {
        Some(ExprInstruction::Object(keys)) => keys,
        _ => return Err(execution("object construction state changed")),
    };
    match build.phase {
        ObjectPhase::Pop => {
            if build.remaining > 0 {
                build.values.push(pop_runtime(stack)?);
                build.remaining -= 1;
            } else {
                build.phase = ObjectPhase::Reverse;
            }
            Ok(false)
        }
        ObjectPhase::Reverse => {
            let len = build.values.len();
            if build.reverse_index < len / 2 {
                build
                    .values
                    .swap(build.reverse_index, len - build.reverse_index - 1);
                build.reverse_index += 1;
            } else {
                build.phase = ObjectPhase::Entries;
            }
            Ok(false)
        }
        ObjectPhase::Entries => {
            if build.entry_index < keys.len() {
                let key = &keys[build.entry_index];
                debug_assert!(key.len() <= MAX_ATOMIC_KEY_BYTES);
                precharge(
                    live,
                    cumulative,
                    key_retained_bytes(key.len())?,
                    &program.limits,
                )?;
                let value = build
                    .values
                    .get(build.entry_index)
                    .ok_or_else(|| execution("object construction value is missing"))?
                    .clone();
                append_object_measurement(
                    &mut build.measurement,
                    key,
                    value.measurement(),
                    build.entry_index,
                )?;
                build.entries.push((clone_string(key)?, value));
                build.entry_index += 1;
                Ok(false)
            } else {
                stack.push(RuntimeValue::object_measured(
                    core::mem::take(&mut build.entries),
                    build.measurement,
                ));
                Ok(true)
            }
        }
    }
}

fn advance_equality(state: &mut EqualityState) -> Result<bool, SandboxError> {
    if !state.equal {
        state.frames.clear();
        state.string = None;
        return Ok(true);
    }
    if let Some(string) = state.string.as_mut() {
        let (left, right) = match (string.left.node(), string.right.node()) {
            (RuntimeNode::String(left), RuntimeNode::String(right)) => (left, right),
            _ => return Err(execution("string equality state changed")),
        };
        if left.len() != right.len() {
            state.equal = false;
            state.string = None;
            return Ok(true);
        }
        let end = string
            .offset
            .saturating_add(MAX_ATOMIC_STRING_BYTES)
            .min(left.len());
        if left.as_bytes().get(string.offset..end) != right.as_bytes().get(string.offset..end) {
            state.equal = false;
            state.string = None;
            return Ok(true);
        }
        string.offset = end;
        if end == left.len() {
            state.string = None;
        }
        return Ok(state.string.is_none() && state.frames.is_empty());
    }
    let Some(frame) = state.frames.pop() else {
        return Ok(true);
    };
    match frame {
        EqualityFrame::Values(left, right) => match (left.node(), right.node()) {
            (RuntimeNode::Null, RuntimeNode::Null) => {}
            (RuntimeNode::Bool(left), RuntimeNode::Bool(right)) => state.equal = left == right,
            (RuntimeNode::Integer(left), RuntimeNode::Integer(right)) => {
                state.equal = left == right;
            }
            (RuntimeNode::String(left_value), RuntimeNode::String(right_value)) => {
                if left_value.len() != right_value.len() {
                    state.equal = false;
                } else if left_value.is_empty() {
                } else {
                    state.string = Some(StringEquality {
                        left,
                        right,
                        offset: 0,
                    });
                }
            }
            (RuntimeNode::Array(left_values), RuntimeNode::Array(right_values)) => {
                if left_values.len() != right_values.len() {
                    state.equal = false;
                } else if !left_values.is_empty() {
                    state.frames.push(EqualityFrame::Array(left, right, 0));
                }
            }
            (RuntimeNode::Object(left_values), RuntimeNode::Object(right_values)) => {
                if left_values.len() != right_values.len() {
                    state.equal = false;
                } else if !left_values.is_empty() {
                    state.frames.push(EqualityFrame::Object(left, right, 0));
                }
            }
            _ => state.equal = false,
        },
        EqualityFrame::Array(left, right, index) => {
            let (left_values, right_values) = match (left.node(), right.node()) {
                (RuntimeNode::Array(left), RuntimeNode::Array(right)) => (left, right),
                _ => return Err(execution("array equality state changed")),
            };
            if index < left_values.len() {
                state
                    .frames
                    .push(EqualityFrame::Array(left.clone(), right.clone(), index + 1));
                state.frames.push(EqualityFrame::Values(
                    left_values[index].clone(),
                    right_values[index].clone(),
                ));
            }
        }
        EqualityFrame::Object(left, right, index) => {
            let (left_values, right_values) = match (left.node(), right.node()) {
                (RuntimeNode::Object(left), RuntimeNode::Object(right)) => (left, right),
                _ => return Err(execution("object equality state changed")),
            };
            if index < left_values.len() {
                let (left_key, left_value) = &left_values[index];
                let (right_key, right_value) = &right_values[index];
                debug_assert!(left_key.len() <= MAX_ATOMIC_KEY_BYTES);
                if left_key != right_key {
                    state.equal = false;
                } else {
                    state.frames.push(EqualityFrame::Object(
                        left.clone(),
                        right.clone(),
                        index + 1,
                    ));
                    state.frames.push(EqualityFrame::Values(
                        left_value.clone(),
                        right_value.clone(),
                    ));
                }
            }
        }
    }
    Ok((!state.equal || state.frames.is_empty()) && state.string.is_none())
}

fn advance_aggregate(state: &mut AggregateState) -> Result<Option<RuntimeValue>, SandboxError> {
    let items = expect_array(&state.collection, "aggregate operand")?;
    if state.index == items.len() {
        return Ok(Some(match state.operator {
            UnaryOperator::Sum => RuntimeValue::integer(state.integer),
            UnaryOperator::All | UnaryOperator::Any => RuntimeValue::boolean(state.boolean),
            _ => return Err(execution("aggregate operator is invalid")),
        }));
    }
    match state.operator {
        UnaryOperator::Sum => {
            state.integer = state
                .integer
                .checked_add(integer(&items[state.index])?)
                .ok_or_else(|| execution("checked integer sum overflowed"))?;
        }
        UnaryOperator::All => {
            state.boolean &= expect_bool(&items[state.index], "all item")?;
        }
        UnaryOperator::Any => {
            state.boolean |= expect_bool(&items[state.index], "any item")?;
        }
        _ => return Err(execution("aggregate operator is invalid")),
    }
    state.index += 1;
    Ok(None)
}

fn advance_path(
    program: &VerifiedProgram,
    location: ExprLocation,
    state: &mut PathState,
) -> Result<Option<RuntimeValue>, SandboxError> {
    let pointer = match expression(program, location)?.0.get(state.opcode) {
        Some(ExprInstruction::Path(pointer)) => pointer,
        _ => return Err(execution("JSON pointer state changed")),
    };
    match state.phase {
        PathPhase::Decode => {
            let bytes = pointer.as_bytes();
            let mut traversed = 0usize;
            while state.pointer_offset < bytes.len() && traversed < MAX_ATOMIC_STRING_BYTES {
                let byte = bytes[state.pointer_offset];
                if byte == b'/' {
                    state.pointer_offset += 1;
                    break;
                }
                if byte == b'~' {
                    let escaped = *bytes
                        .get(state.pointer_offset + 1)
                        .ok_or_else(|| execution("JSON pointer escape is incomplete"))?;
                    state.segment.push(match escaped {
                        b'0' => '~',
                        b'1' => '/',
                        _ => return Err(execution("JSON pointer escape is invalid")),
                    });
                    state.pointer_offset += 2;
                    traversed += 2;
                } else {
                    let character = pointer[state.pointer_offset..]
                        .chars()
                        .next()
                        .ok_or_else(|| execution("JSON pointer character is invalid"))?;
                    state.segment.push(character);
                    state.pointer_offset += character.len_utf8();
                    traversed += character.len_utf8();
                }
            }
            let segment_complete = state.pointer_offset == pointer.len()
                || bytes.get(state.pointer_offset.saturating_sub(1)) == Some(&b'/');
            if segment_complete {
                state.phase = match state.current.node() {
                    RuntimeNode::Object(_) => PathPhase::ObjectLookup(0),
                    RuntimeNode::Array(_) => PathPhase::ArrayIndex {
                        offset: 0,
                        value: 0,
                    },
                    _ => return Err(execution("JSON pointer did not resolve")),
                };
            }
            Ok(None)
        }
        PathPhase::ObjectLookup(index) => {
            let entries = state
                .current
                .as_object()
                .ok_or_else(|| execution("JSON pointer object state changed"))?;
            let Some((key, value)) = entries.get(index) else {
                return Err(execution("JSON pointer did not resolve"));
            };
            debug_assert!(key.len() <= MAX_ATOMIC_KEY_BYTES);
            if key == &state.segment {
                let selected = value.clone();
                if state.pointer_offset == pointer.len() {
                    Ok(Some(selected))
                } else {
                    state.current = selected;
                    state.segment.clear();
                    state.phase = PathPhase::Decode;
                    Ok(None)
                }
            } else {
                state.phase = PathPhase::ObjectLookup(index + 1);
                Ok(None)
            }
        }
        PathPhase::ArrayIndex {
            mut offset,
            mut value,
        } => {
            let bytes = state.segment.as_bytes();
            let end = offset
                .saturating_add(MAX_ATOMIC_STRING_BYTES)
                .min(bytes.len());
            while offset < end {
                let digit = bytes[offset];
                if !digit.is_ascii_digit() || (offset == 0 && bytes.len() > 1 && digit == b'0') {
                    return Err(execution("JSON pointer array index is invalid"));
                }
                value = value
                    .checked_mul(10)
                    .and_then(|current| current.checked_add((digit - b'0') as usize))
                    .ok_or_else(|| execution("JSON pointer array index overflowed"))?;
                offset += 1;
            }
            if offset < bytes.len() {
                state.phase = PathPhase::ArrayIndex { offset, value };
                return Ok(None);
            }
            let selected = state
                .current
                .as_array()
                .and_then(|items| items.get(value))
                .cloned()
                .ok_or_else(|| execution("JSON pointer did not resolve"))?;
            if state.pointer_offset == pointer.len() {
                Ok(Some(selected))
            } else {
                state.current = selected;
                state.segment.clear();
                state.phase = PathPhase::Decode;
                Ok(None)
            }
        }
    }
}

fn advance_materialize(state: &mut MaterializeState) -> Result<Option<Value>, SandboxError> {
    if let Some(string) = state.string.as_mut() {
        let source = match string.source.node() {
            RuntimeNode::String(source) => source,
            _ => return Err(execution("string materialization state changed")),
        };
        let end = utf8_chunk_end(source, string.offset, MAX_ATOMIC_STRING_BYTES);
        string
            .output
            .push_str(source.get(string.offset..end).unwrap_or_default());
        string.offset = end;
        if end == source.len() {
            let string = state
                .string
                .take()
                .ok_or_else(|| execution("string materialization is missing"))?;
            state.completed = Some(Value::String(string.output));
        }
        return Ok(None);
    }

    if let Some(value) = state.completed.take() {
        if let Some(frame) = state.frames.last_mut() {
            match frame {
                MaterializeFrame::Array { index, output, .. } => {
                    output.push(value);
                    *index += 1;
                }
                MaterializeFrame::Object {
                    source,
                    index,
                    output,
                } => {
                    let entries = source
                        .as_object()
                        .ok_or_else(|| execution("object materialization state changed"))?;
                    let key = entries
                        .get(*index)
                        .map(|(key, _)| key)
                        .ok_or_else(|| execution("object materialization key is missing"))?;
                    debug_assert!(key.len() <= MAX_ATOMIC_KEY_BYTES);
                    output.insert(clone_string(key)?, value);
                    *index += 1;
                }
            }
            return Ok(None);
        }
        return Ok(Some(value));
    }

    if let Some(current) = state.current.take() {
        match current.node() {
            RuntimeNode::Null => state.completed = Some(Value::Null),
            RuntimeNode::Bool(value) => state.completed = Some(Value::Bool(*value)),
            RuntimeNode::Integer(value) => {
                state.completed = Some(Value::Number(Number::from(*value)));
            }
            RuntimeNode::String(source) => {
                let mut output = String::new();
                output
                    .try_reserve_exact(source.len())
                    .map_err(|_| resource("string materialization allocation failed"))?;
                if source.is_empty() {
                    state.completed = Some(Value::String(output));
                } else {
                    state.string = Some(MaterializeString {
                        source: current,
                        offset: 0,
                        output,
                    });
                }
            }
            RuntimeNode::Array(values) => {
                let mut output = Vec::new();
                output
                    .try_reserve_exact(values.len())
                    .map_err(|_| resource("array materialization allocation failed"))?;
                state.frames.push(MaterializeFrame::Array {
                    source: current,
                    index: 0,
                    output,
                });
            }
            RuntimeNode::Object(_) => {
                state.frames.push(MaterializeFrame::Object {
                    source: current,
                    index: 0,
                    output: Map::new(),
                });
            }
        }
        return Ok(None);
    }

    let frame = state
        .frames
        .last_mut()
        .ok_or_else(|| execution("materialization state is empty"))?;
    match frame {
        MaterializeFrame::Array {
            source,
            index,
            output: _,
        } => {
            let values = source
                .as_array()
                .ok_or_else(|| execution("array materialization state changed"))?;
            if *index < values.len() {
                state.current = Some(values[*index].clone());
            } else {
                let frame = state
                    .frames
                    .pop()
                    .ok_or_else(|| execution("array materialization frame is missing"))?;
                if let MaterializeFrame::Array { output, .. } = frame {
                    state.completed = Some(Value::Array(output));
                }
            }
        }
        MaterializeFrame::Object {
            source,
            index,
            output: _,
        } => {
            let values = source
                .as_object()
                .ok_or_else(|| execution("object materialization state changed"))?;
            if *index < values.len() {
                state.current = Some(values[*index].1.clone());
            } else {
                let frame = state
                    .frames
                    .pop()
                    .ok_or_else(|| execution("object materialization frame is missing"))?;
                if let MaterializeFrame::Object { output, .. } = frame {
                    state.completed = Some(Value::Object(output));
                }
            }
        }
    }
    Ok(None)
}

fn binary_scalar(
    operator: BinaryOperator,
    left: &RuntimeValue,
    right: &RuntimeValue,
) -> Result<RuntimeValue, SandboxError> {
    use BinaryOperator::*;
    Ok(match operator {
        Add => RuntimeValue::integer(
            integer(left)?
                .checked_add(integer(right)?)
                .ok_or_else(|| execution("checked integer addition overflowed"))?,
        ),
        Subtract => RuntimeValue::integer(
            integer(left)?
                .checked_sub(integer(right)?)
                .ok_or_else(|| execution("checked integer subtraction overflowed"))?,
        ),
        Multiply => RuntimeValue::integer(
            integer(left)?
                .checked_mul(integer(right)?)
                .ok_or_else(|| execution("checked integer multiplication overflowed"))?,
        ),
        Divide => RuntimeValue::integer(
            integer(left)?
                .checked_div(integer(right)?)
                .ok_or_else(|| execution("checked integer division failed"))?,
        ),
        Remainder => RuntimeValue::integer(
            integer(left)?
                .checked_rem(integer(right)?)
                .ok_or_else(|| execution("checked integer remainder failed"))?,
        ),
        LessThan => RuntimeValue::boolean(integer(left)? < integer(right)?),
        LessThanOrEqual => RuntimeValue::boolean(integer(left)? <= integer(right)?),
        GreaterThan => RuntimeValue::boolean(integer(left)? > integer(right)?),
        GreaterThanOrEqual => RuntimeValue::boolean(integer(left)? >= integer(right)?),
        And => RuntimeValue::boolean(
            expect_bool(left, "left operand")? && expect_bool(right, "right operand")?,
        ),
        Or => RuntimeValue::boolean(
            expect_bool(left, "left operand")? || expect_bool(right, "right operand")?,
        ),
        Equal | NotEqual => return Err(execution("equality must use its continuation")),
    })
}

fn unary_scalar(
    operator: UnaryOperator,
    value: &RuntimeValue,
) -> Result<RuntimeValue, SandboxError> {
    use UnaryOperator::*;
    Ok(match operator {
        Not => RuntimeValue::boolean(!expect_bool(value, "not operand")?),
        Negate => RuntimeValue::integer(
            integer(value)?
                .checked_neg()
                .ok_or_else(|| execution("checked integer negation overflowed"))?,
        ),
        Count => {
            let count = match value.node() {
                RuntimeNode::Array(values) => values.len(),
                RuntimeNode::Object(values) => values.len(),
                _ => return Err(execution("count requires an array or object")),
            };
            RuntimeValue::integer(
                i64::try_from(count).map_err(|_| execution("count result overflowed"))?,
            )
        }
        Sum | All | Any => return Err(execution("aggregate must use its continuation")),
    })
}

fn expect_bool(value: &RuntimeValue, context: &'static str) -> Result<bool, SandboxError> {
    value
        .as_bool()
        .ok_or_else(|| execution(alloc::format!("{context} must be boolean")))
}

fn integer(value: &RuntimeValue) -> Result<i64, SandboxError> {
    value
        .as_integer()
        .ok_or_else(|| execution("integer operation requires i64 operands"))
}

fn expect_array<'a>(
    value: &'a RuntimeValue,
    context: &'static str,
) -> Result<&'a [RuntimeValue], SandboxError> {
    value
        .as_array()
        .ok_or_else(|| execution(alloc::format!("{context} must be an array")))
}

fn require_object(value: &RuntimeValue) -> Result<(), SandboxError> {
    if value.as_object().is_some() {
        Ok(())
    } else {
        Err(execution("tool arguments must be an object"))
    }
}

fn pop_runtime(values: &mut Vec<RuntimeValue>) -> Result<RuntimeValue, SandboxError> {
    values
        .pop()
        .ok_or_else(|| execution("operand stack underflow"))
}

fn assign_scoped(
    locals: &mut [Option<RuntimeValue>],
    slot: usize,
    value: &RuntimeValue,
    live: &mut usize,
    cumulative: &mut usize,
    limits: &SandboxLimits,
) -> Result<(), SandboxError> {
    precharge_cached_clone(live, cumulative, value, limits)?;
    let local = locals
        .get_mut(slot)
        .ok_or_else(|| execution("scoped local slot is invalid"))?;
    *local = Some(value.clone());
    Ok(())
}

fn append_array_measurement(
    measurement: &mut ValueMeasurement,
    child: ValueMeasurement,
    existing_items: usize,
) -> Result<(), SandboxError> {
    if existing_items > 0 {
        measurement.serialized = checked_add(measurement.serialized, 1)?;
    }
    accumulate_measurement(measurement, child)?;
    measurement.collection_items = checked_add(measurement.collection_items, 1)?;
    Ok(())
}

fn append_object_measurement(
    measurement: &mut ValueMeasurement,
    key: &str,
    child: ValueMeasurement,
    existing_items: usize,
) -> Result<(), SandboxError> {
    if existing_items > 0 {
        measurement.serialized = checked_add(measurement.serialized, 1)?;
    }
    measurement.retained = checked_add(measurement.retained, key_retained_bytes(key.len())?)?;
    measurement.serialized = checked_add(
        measurement.serialized,
        checked_add(serialized_string_len(key)?, 1)?,
    )?;
    accumulate_measurement(measurement, child)?;
    measurement.collection_items = checked_add(measurement.collection_items, 1)?;
    Ok(())
}

fn accumulate_measurement(
    measurement: &mut ValueMeasurement,
    child: ValueMeasurement,
) -> Result<(), SandboxError> {
    measurement.retained = checked_add(measurement.retained, child.retained)?;
    measurement.serialized = checked_add(measurement.serialized, child.serialized)?;
    measurement.nodes = checked_add(measurement.nodes, child.nodes)?;
    measurement.collection_items =
        checked_add(measurement.collection_items, child.collection_items)?;
    measurement.max_depth = measurement.max_depth.max(
        child
            .max_depth
            .checked_add(1)
            .ok_or_else(|| resource("value nesting limit exceeded"))?,
    );
    Ok(())
}

fn precharge_cached_clone(
    live: &mut usize,
    cumulative: &mut usize,
    value: &RuntimeValue,
    limits: &SandboxLimits,
) -> Result<(), SandboxError> {
    precharge(live, cumulative, value.measurement().retained, limits)
}

fn precharge(
    live: &mut usize,
    cumulative: &mut usize,
    bytes: usize,
    limits: &SandboxLimits,
) -> Result<(), SandboxError> {
    let next_live = live
        .checked_add(bytes)
        .ok_or_else(|| resource("live byte limit exceeded"))?;
    let next_cumulative = cumulative
        .checked_add(bytes)
        .ok_or_else(|| resource("cumulative byte limit exceeded"))?;
    ensure_byte_limits(next_live, next_cumulative, limits)?;
    *live = next_live;
    *cumulative = next_cumulative;
    Ok(())
}

fn ensure_byte_limits(
    live: usize,
    cumulative: usize,
    limits: &SandboxLimits,
) -> Result<(), SandboxError> {
    if live > limits.max_live_bytes {
        return Err(resource("live byte limit exceeded"));
    }
    if cumulative > limits.max_cumulative_bytes {
        return Err(resource("cumulative byte limit exceeded"));
    }
    Ok(())
}

fn charge_one(fuel_used: &mut u64, max_fuel: u64, remaining: &mut u64) -> Result<(), SandboxError> {
    if *remaining == 0 {
        return Err(execution("scheduler attempted work without slice fuel"));
    }
    if *fuel_used >= max_fuel {
        return Err(resource("fuel limit exceeded"));
    }
    *fuel_used = fuel_used
        .checked_add(1)
        .ok_or_else(|| resource("fuel limit exceeded"))?;
    *remaining -= 1;
    Ok(())
}

fn response_wrapper_retained(output: usize) -> Result<usize, SandboxError> {
    let mut total = object_framing_retained_bytes()?;
    total = checked_add(total, key_retained_bytes(2)?)?;
    total = checked_add(total, primitive_retained_bytes())?;
    total = checked_add(total, key_retained_bytes(6)?)?;
    checked_add(total, output)
}

fn response_wrapper_serialized(ok: bool, output: usize) -> Result<usize, SandboxError> {
    let mut total = 2usize;
    total = checked_add(total, serialized_string_len("ok")?)?;
    total = checked_add(total, 1)?;
    total = checked_add(total, if ok { 4 } else { 5 })?;
    total = checked_add(total, 1)?;
    total = checked_add(total, serialized_string_len("output")?)?;
    total = checked_add(total, 1)?;
    checked_add(total, output)
}

fn clone_string(source: &str) -> Result<String, SandboxError> {
    let mut output = String::new();
    output
        .try_reserve_exact(source.len())
        .map_err(|_| resource("string allocation failed"))?;
    output.push_str(source);
    Ok(output)
}

fn utf8_chunk_end(source: &str, offset: usize, maximum: usize) -> usize {
    let mut end = offset.saturating_add(maximum).min(source.len());
    while end > offset && !source.is_char_boundary(end) {
        end -= 1;
    }
    if end == offset && offset < source.len() {
        end = offset
            + source[offset..]
                .chars()
                .next()
                .map(char::len_utf8)
                .unwrap_or(0);
    }
    end
}

const fn escaped_byte_len(byte: u8) -> usize {
    match byte {
        b'"' | b'\\' => 2,
        0x00..=0x1f => 6,
        _ => 1,
    }
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
    use crate::Program;
    use alloc::{format, vec};
    use serde_json::json;

    fn execution(raw: &str, id: u64, limits: SandboxLimits) -> Execution {
        let program = Program::from_json(raw.as_bytes(), &limits)
            .unwrap()
            .compile(&limits)
            .unwrap();
        Execution::new(program, ExecutionId(id)).unwrap()
    }

    fn complete(vm: &mut Execution, slice: u64) -> Value {
        loop {
            match vm.step(slice).unwrap() {
                StepOutcome::Sliced => {}
                StepOutcome::Complete(value) => return value,
                StepOutcome::Yielded { .. } => panic!("unexpected yield"),
            }
        }
    }

    fn yield_once(vm: &mut Execution) -> (ToolBatch, ResumeToken) {
        loop {
            match vm.step(1).unwrap() {
                StepOutcome::Sliced => {}
                StepOutcome::Yielded { batch, resume } => return (batch, resume),
                StepOutcome::Complete(_) => panic!("expected yield"),
            }
        }
    }

    #[test]
    fn slice_one_completes_long_literal_and_every_slice_charges_one() {
        let literal = "x".repeat(MAX_ATOMIC_STRING_BYTES * 8 + 3);
        let raw = format!(
            "{{\"version\":1,\"body\":[{{\"kind\":\"return\",\"value\":{{\"kind\":\"string\",\"value\":{}}}}}]}}",
            serde_json::to_string(&literal).unwrap()
        );
        let limits = SandboxLimits {
            max_slice_fuel: 1,
            ..SandboxLimits::default()
        };
        let mut vm = execution(&raw, 1, limits);
        loop {
            let before = vm.metrics().fuel_used;
            match vm.step(1).unwrap() {
                StepOutcome::Sliced => assert_eq!(vm.metrics().fuel_used, before + 1),
                StepOutcome::Complete(Value::String(value)) => {
                    assert_eq!(value, literal);
                    break;
                }
                other => panic!("unexpected outcome: {other:?}"),
            }
        }
    }

    #[test]
    fn equality_and_aggregates_advance_per_chunk_or_item() {
        let string = "q".repeat(MAX_ATOMIC_STRING_BYTES * 12);
        let values = (0..128).map(|_| json!(1)).collect::<Vec<_>>();
        let raw = format!(
            "{{\"version\":1,\"body\":[
              {{\"kind\":\"let\",\"name\":\"same\",\"value\":{{\"kind\":\"binary\",\"operator\":\"equal\",\"left\":{{\"kind\":\"string\",\"value\":{0}}},\"right\":{{\"kind\":\"string\",\"value\":{0}}}}}}},
              {{\"kind\":\"let\",\"name\":\"total\",\"value\":{{\"kind\":\"unary\",\"operator\":\"sum\",\"value\":{{\"kind\":\"array\",\"items\":{1}}}}}}},
              {{\"kind\":\"return\",\"value\":{{\"kind\":\"array\",\"items\":[{{\"kind\":\"variable\",\"name\":\"same\"}},{{\"kind\":\"variable\",\"name\":\"total\"}}]}}}}
            ]}}",
            serde_json::to_string(&string).unwrap(),
            serde_json::to_string(&values.iter().map(|value| json!({"kind":"integer","value":value})).collect::<Vec<_>>()).unwrap()
        );
        let mut vm = execution(&raw, 2, SandboxLimits::default());
        assert_eq!(complete(&mut vm, 1), json!([true, 128]));
        assert!(vm.metrics().fuel_used > 128 + (string.len() / MAX_ATOMIC_STRING_BYTES) as u64);
    }

    #[test]
    fn map_filter_reduce_remain_deterministic_at_slice_one() {
        let raw = r#"{"version":1,"body":[
          {"kind":"map","name":"mapped","item":"item","collection":{"kind":"array","items":[{"kind":"integer","value":1},{"kind":"integer","value":2},{"kind":"integer","value":3}]},"max_items":3,"value":{"kind":"binary","operator":"add","left":{"kind":"variable","name":"item"},"right":{"kind":"integer","value":1}}},
          {"kind":"filter","name":"filtered","item":"item","collection":{"kind":"variable","name":"mapped"},"max_items":3,"predicate":{"kind":"binary","operator":"greater_than","left":{"kind":"variable","name":"item"},"right":{"kind":"integer","value":2}}},
          {"kind":"reduce","name":"total","item":"item","accumulator":"acc","collection":{"kind":"variable","name":"filtered"},"max_items":3,"initial":{"kind":"integer","value":0},"value":{"kind":"binary","operator":"add","left":{"kind":"variable","name":"acc"},"right":{"kind":"variable","name":"item"}}},
          {"kind":"return","value":{"kind":"variable","name":"total"}}
        ]}"#;
        let mut vm = execution(raw, 3, SandboxLimits::default());
        assert_eq!(complete(&mut vm, 1), json!(7));
    }

    #[test]
    fn loop_local_clearing_is_metered_at_slice_one() {
        let raw = r#"{"version":1,"body":[
          {"kind":"for_each","item":"item","collection":{"kind":"array","items":[{"kind":"integer","value":1},{"kind":"integer","value":2}]},"max_iterations":2,"body":[
            {"kind":"let","name":"first","value":{"kind":"variable","name":"item"}},
            {"kind":"let","name":"second","value":{"kind":"variable","name":"first"}},
            {"kind":"let","name":"third","value":{"kind":"variable","name":"second"}}
          ]},
          {"kind":"return","value":{"kind":"null"}}
        ]}"#;
        let limits = SandboxLimits {
            max_slice_fuel: 1,
            ..SandboxLimits::default()
        };
        let mut vm = execution(raw, 15, limits);
        let mut slices = 0usize;
        loop {
            let before = vm.metrics().fuel_used;
            match vm.step(1).unwrap() {
                StepOutcome::Sliced => {
                    assert_eq!(vm.metrics().fuel_used, before + 1);
                    slices += 1;
                }
                StepOutcome::Complete(Value::Null) => break,
                other => panic!("unexpected outcome: {other:?}"),
            }
        }
        assert!(
            slices >= 3,
            "the second iteration must clear three body slots"
        );
    }

    #[test]
    fn path_lookup_and_materialization_are_resumable() {
        let key = "k".repeat(MAX_ATOMIC_KEY_BYTES);
        let raw = format!(
            "{{\"version\":1,\"body\":[{{\"kind\":\"return\",\"value\":{{\"kind\":\"path\",\"value\":{{\"kind\":\"object\",\"entries\":[{{\"key\":{0},\"value\":{{\"kind\":\"string\",\"value\":\"found\"}}}}]}},\"pointer\":{1}}}}}]}}",
            serde_json::to_string(&key).unwrap(),
            serde_json::to_string(&format!("/{key}")).unwrap()
        );
        let mut vm = execution(&raw, 4, SandboxLimits::default());
        assert_eq!(complete(&mut vm, 1), json!("found"));
    }

    #[test]
    fn checked_response_admission_is_inert_and_resume_is_atomic() {
        let raw = r#"{"version":1,"body":[
          {"kind":"invoke","name":"result","tool_id":"read","arguments":{"kind":"object","entries":[]}},
          {"kind":"return","value":{"kind":"path","value":{"kind":"variable","name":"result"},"pointer":"/output/value"}}
        ]}"#;
        let limits = SandboxLimits::default();
        let mut vm = execution(raw, 5, limits);
        let (batch, token) = yield_once(&mut vm);
        let output = json!({"value":{"kind":"invoke","tool_id":"danger"}});
        let response = ToolResponse::success(&batch.calls()[0], &output, &limits).unwrap();
        vm.resume(token, vec![response]).unwrap();
        assert_eq!(complete(&mut vm, 1), output["value"]);
        assert!(vm.step(1).is_err());
    }

    #[test]
    fn cross_execution_token_terminalizes_without_replay() {
        let raw = r#"{"version":1,"body":[{"kind":"invoke","name":"r","tool_id":"read","arguments":{"kind":"object","entries":[]}},{"kind":"return","value":{"kind":"null"}}]}"#;
        let limits = SandboxLimits::default();
        let mut first = execution(raw, 6, limits);
        let mut second = execution(raw, 7, limits);
        let (_first_batch, first_token) = yield_once(&mut first);
        let (second_batch, _second_token) = yield_once(&mut second);
        let response = ToolResponse::failure(&second_batch.calls()[0]);
        assert_eq!(
            second
                .resume(first_token, vec![response])
                .unwrap_err()
                .code(),
            SandboxErrorCode::InvalidResume
        );
        assert!(second.step(1).is_err());
    }

    #[test]
    fn hostile_borrowed_response_is_rejected_without_recursive_cleanup() {
        let raw = r#"{"version":1,"body":[{"kind":"invoke","name":"r","tool_id":"read","arguments":{"kind":"object","entries":[]}},{"kind":"return","value":{"kind":"null"}}]}"#;
        let limits = SandboxLimits::default();
        let mut vm = execution(raw, 8, limits);
        let (batch, _token) = yield_once(&mut vm);
        let mut hostile = Value::Null;
        for _ in 0..10_000 {
            hostile = Value::Array(vec![hostile]);
        }
        let result = std::panic::catch_unwind(|| {
            ToolResponse::success(&batch.calls()[0], &hostile, &limits)
        });
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().unwrap_err().code(),
            SandboxErrorCode::ResourceLimit
        );
        discard_host_value(hostile);
    }

    #[test]
    fn float_and_collection_boundaries_are_rejected_at_constructor() {
        let raw = r#"{"version":1,"body":[{"kind":"invoke","name":"r","tool_id":"read","arguments":{"kind":"object","entries":[]}},{"kind":"return","value":{"kind":"null"}}]}"#;
        let limits = SandboxLimits {
            max_collection_items: 2,
            ..SandboxLimits::default()
        };
        let mut vm = execution(raw, 9, limits);
        let (batch, _) = yield_once(&mut vm);
        assert!(ToolResponse::success(&batch.calls()[0], &json!(1.5), &limits).is_err());
        assert!(ToolResponse::success(&batch.calls()[0], &json!([1, 2]), &limits).is_ok());
        assert!(ToolResponse::success(&batch.calls()[0], &json!([1, 2, 3]), &limits).is_err());
    }

    #[test]
    fn response_constructor_enforces_serialized_limit_at_n_minus_one_n_and_n_plus_one() {
        let raw = r#"{"version":1,"body":[{"kind":"invoke","name":"r","tool_id":"read","arguments":{"kind":"object","entries":[]}},{"kind":"return","value":{"kind":"null"}}]}"#;
        let mut vm = execution(raw, 10, SandboxLimits::default());
        let (batch, _) = yield_once(&mut vm);
        let output = json!("abc");
        let serialized = 5;
        for (limit, accepted) in [
            (serialized - 1, false),
            (serialized, true),
            (serialized + 1, true),
        ] {
            let limits = SandboxLimits {
                max_output_bytes: limit,
                ..SandboxLimits::default()
            };
            assert_eq!(
                ToolResponse::success(&batch.calls()[0], &output, &limits).is_ok(),
                accepted
            );
        }
    }

    #[test]
    fn response_keys_and_program_tool_ids_respect_the_string_quantum() {
        let raw = r#"{"version":1,"body":[{"kind":"invoke","name":"r","tool_id":"read","arguments":{"kind":"object","entries":[]}},{"kind":"return","value":{"kind":"null"}}]}"#;
        let mut vm = execution(raw, 11, SandboxLimits::default());
        let (batch, _) = yield_once(&mut vm);
        let mut object = Map::new();
        object.insert("k".repeat(MAX_ATOMIC_KEY_BYTES + 1), Value::Null);
        assert_eq!(
            ToolResponse::success(
                &batch.calls()[0],
                &Value::Object(object),
                &SandboxLimits::default(),
            )
            .unwrap_err()
            .code(),
            SandboxErrorCode::ResourceLimit
        );

        let too_long_tool = "t".repeat(MAX_ATOMIC_STRING_BYTES + 1);
        let raw = format!(
            r#"{{"version":1,"body":[{{"kind":"invoke","name":"r","tool_id":"{too_long_tool}","arguments":{{"kind":"object","entries":[]}}}},{{"kind":"return","value":{{"kind":"null"}}}}]}}"#
        );
        assert_eq!(
            Program::from_json(raw.as_bytes(), &SandboxLimits::default())
                .unwrap_err()
                .code(),
            SandboxErrorCode::InvalidProgram
        );
    }

    #[test]
    fn fuel_exhaustion_is_terminal_at_exact_boundary() {
        let raw = r#"{"version":1,"body":[{"kind":"return","value":{"kind":"array","items":[{"kind":"integer","value":1},{"kind":"integer","value":2}]}}]}"#;
        let mut baseline = execution(raw, 12, SandboxLimits::default());
        let expected = complete(&mut baseline, 1);
        assert_eq!(expected, json!([1, 2]));
        let used = baseline.metrics().fuel_used;
        let limits = SandboxLimits {
            max_fuel: used - 1,
            max_slice_fuel: 1,
            ..SandboxLimits::default()
        };
        let mut exhausted = execution(raw, 13, limits);
        while matches!(exhausted.step(1), Ok(StepOutcome::Sliced)) {}
        assert!(exhausted.step(1).is_err());
    }

    #[test]
    fn public_debug_is_value_redacted() {
        let raw = r#"{"version":1,"body":[{"kind":"invoke","name":"r","tool_id":"CANARY","arguments":{"kind":"object","entries":[]}},{"kind":"return","value":{"kind":"null"}}]}"#;
        let mut vm = execution(raw, 14, SandboxLimits::default());
        let (batch, _) = yield_once(&mut vm);
        assert!(!format!("{batch:?}").contains("CANARY"));
        let response = ToolResponse::success(
            &batch.calls()[0],
            &json!({"secret":"CANARY"}),
            &SandboxLimits::default(),
        )
        .unwrap();
        assert!(!format!("{response:?}").contains("CANARY"));
    }

    fn discard_host_value(value: Value) {
        let mut pending = Vec::with_capacity(10_001);
        pending.push(value);
        while let Some(value) = pending.pop() {
            match value {
                Value::Array(mut values) => pending.append(&mut values),
                Value::Object(values) => pending.extend(values.into_iter().map(|(_, value)| value)),
                _ => {}
            }
        }
    }
}
