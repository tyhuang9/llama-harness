# Distribution

Rust users depend on the `llama-harness` facade and explicitly enable only the
features they need. `core` is always embedded; `ollama`, `observability`,
`evals`, `protocol`, and `tauri` are optional. The CLI, Promptfoo adapter,
examples, console, and scripted test sidecar are deliberately not published.

Node distribution is two artifacts: `@llama-harness/sdk` plus exactly one
matching `@llama-harness/runtime-<platform>-<arch>` package containing the
runtime executable. Python distribution is a matching platform-tagged
`llama-harness` wheel containing `llama_harness/runtime/llama-harness-runtime`.
The release workflow stages these packages only after copying the output of the
same Rust build matrix. No empty platform package is published. The release
workflow intentionally produces no Python source distribution: a source archive
cannot honestly contain one platform's runtime. Developers building from source
set `LLAMA_HARNESS_RUNTIME_PATH` explicitly.

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
