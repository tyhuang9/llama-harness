# Distribution

Rust users depend on the `llama-harness` facade and explicitly enable only the
features they need. The facade's core exports are always available; `ollama`,
`observability`, `evals`, `tauri`, `programmatic`, and default-off `mcp` are
optional named modules. These eight Rust-facing crates are the 0.2.0
publication set: `llama-harness-programmatic-sandbox`, `llama-harness-core`,
`llama-harness-ollama`, `llama-harness-observability`, `llama-harness-tauri`,
`llama-harness-evals`, `llama-harness-mcp`, and `llama-harness`. The CLI,
protocol/runtime sidecar, Promptfoo adapter, examples, console, and scripted
test sidecar are not Rust crates.io packages.

The unified Rust workspace and both SDK package manifests use version `0.2.0`.
The Rust crates.io release requires Rust 1.88. Its registry authorization is
separate from runtime-artifact and SDK registry authorization.

Node distribution is two artifacts: `@llama-harness/sdk` plus exactly one
matching `@llama-harness/runtime-<platform>-<arch>` package containing the
runtime executable. Python distribution is a matching platform-tagged
`llama-harness` wheel containing `llama_harness/runtime/llama-harness-runtime`.
The release workflow stages these packages only after copying the output of the
same Rust build matrix. It requires the Cargo workspace, runtime hello, npm
package, Python project, and SDK client-hello identities to agree on the
requested version. No empty platform package is published, and staging never
rewrites package metadata to make a mismatch appear valid. The workflow
intentionally produces no Python source distribution: a source archive cannot
honestly contain one platform's runtime. Developers building from source set
`LLAMA_HARNESS_RUNTIME_PATH` explicitly.

Supported release targets are Windows x64, macOS arm64, and Linux x64. Other
platforms must set `LLAMA_HARNESS_RUNTIME_PATH` to a reviewed local executable
until a matching artifact is released. Artifact provenance consists of the
workflow run, `checksums.sha256`, and `release-manifest.json`; verify all three
before installation or registry upload.

Linux runtime packages and wheels declare a glibc 2.35 minimum. They are built
and executed on the pinned Ubuntu 22.04 runner, and `readelf` must show no
required GLIBC symbol newer than 2.35 before the wheel is tagged
`manylinux_2_35_x86_64`. This proves the declared floor for that build; it does
not claim compatibility with older glibc releases or non-glibc Linux systems.

The protocol version is independent from package versioning. The 0.2.0 SDKs
offer protocol 1.1 and retain the documented 1.0 compatibility fallback. A
package-version mismatch is nevertheless a release failure: stop before upload
and rebuild from the reviewed source rather than editing staged artifacts.
