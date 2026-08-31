# llama-harness-programmatic-sandbox

Deterministic, resource-bounded `no_std` execution contracts for the optional
llama-harness programmatic tool-calling strategy.

This crate contains no provider, tool, policy, approval, registry, runtime, or
host handles. Programs yield inert owned-data tool batches; the embedding host
remains responsible for every external effect.

The crate is `#![no_std]` with `alloc`, forbids unsafe code, and has no build
script or build dependencies. Its normal dependencies are only `serde` and
`serde_json`, both with default features disabled. It is a deterministic
same-process language runtime, not an operating-system isolation boundary:
hosts must keep every yielded request behind their own broker, policy,
approval, deadline, and cancellation gates.

## Execution contract

- Parsing and direct construction of the public AST use the same structural
  validator before private bytecode is emitted and independently verified.
- Fuel is resumable at instruction and expression-work boundaries. A valid
  program can make progress with `max_slice_fuel = 1`; every `Sliced` outcome
  consumes fuel, and no expression is refused forever because its total cost
  is larger than one scheduling slice.
- Retained and cumulative byte limits use one conservative logical allocation
  model for primitive values, strings, keys, collection storage, and VM
  capacity. The counters may retain charges until execution ends. Output and
  response serialization limits use exact compact-JSON lengths without
  allocating a serialization buffer.
- Tool responses are inert JSON data. Before a suspended state is consumed,
  the VM iteratively validates response identity, order, signed-integer
  domain, depth, collection size, retained bytes, cumulative bytes, and
  serialized bytes.
- `ResumeToken` is non-clone and single use. Because `Execution::resume`
  consumes it, any invalid resume attempt terminalizes that execution. After a
  valid token and response batch are accepted, any later error is also
  terminal; a yielded effect cannot be reopened or replayed.

Public debug output for programs, statements, expressions, requests,
responses, outcomes, and verified programs is redacted and never includes
program constants, tool identifiers, arguments, or tool output values.
