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
