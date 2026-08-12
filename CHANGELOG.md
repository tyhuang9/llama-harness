# Changelog

All notable changes to this project are documented here.

## 0.1.0 — Unreleased

- Establish the public Rust facade and release-preparation work for the embedded agent engine.
- Introduce the versioned local sidecar protocol and SDK distribution work in subsequent tracked phases.

## Versioning policy

The first-party Rust framework crates use a unified version during the pre-1.0 series. Until 1.0, minor releases may include breaking API changes; patch releases are reserved for compatible fixes. The sidecar protocol has its own major/minor version and is not coupled to Cargo, npm, or PyPI package versions.
