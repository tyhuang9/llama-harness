# llama-harness-programmatic-sandbox

Deterministic, resource-bounded `no_std` execution contracts for the optional
llama-harness programmatic tool-calling strategy.

This crate contains no provider, tool, policy, approval, registry, runtime, or
host handles. Programs yield inert owned-data tool batches; the embedding host
remains responsible for every external effect.
