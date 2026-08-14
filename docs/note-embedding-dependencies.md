# Note embedding dependency evidence

This is reproducible pre-integration evidence for embedding the facade-only
Tauri host into Note. Note was inspected read-only; this work does not modify
Note or align package versions.

## Reviewed inputs

- llama-harness revision: `d5f7c0ed5691a974bef1150dbe3b9e57c86ab454`
- llama-harness `Cargo.lock` SHA-256: `5ff6bb7126cd08809c24aa37137b57ec3a21eaaf25b43aa22f1725b3950ed9a3`
- Note revision: `855695adad798a82a768207b6c8e19651cfd0234`
- Note `backend/src-tauri/Cargo.lock` SHA-256: `2d07ff9c8327bbb925f722cbee6fa023003aedbe89c2d9915231ae57055c0653`
- target used for comparison: `x86_64-pc-windows-msvc`

Set repository locations explicitly; no personal absolute path is part of the
procedure:

```powershell
$HARNESS_REPO = Resolve-Path <llama-harness-checkout>
$NOTE_REPO = Resolve-Path <note-checkout>
$NOTE_MANIFEST = Join-Path $NOTE_REPO "backend/src-tauri/Cargo.toml"

git -C $HARNESS_REPO rev-parse HEAD
Get-FileHash -Algorithm SHA256 (Join-Path $HARNESS_REPO "Cargo.lock")
git -C $NOTE_REPO rev-parse HEAD
Get-FileHash -Algorithm SHA256 (Join-Path $NOTE_REPO "backend/src-tauri/Cargo.lock")

cargo tree --manifest-path $NOTE_MANIFEST --locked --target x86_64-pc-windows-msvc -p note -i reqwest@0.13.4
cargo tree --manifest-path $NOTE_MANIFEST --locked --target x86_64-pc-windows-msvc -p note -e features -i reqwest@0.13.4
cargo tree --manifest-path $NOTE_MANIFEST --locked --target x86_64-pc-windows-msvc -p note -i rustls@0.23.42
cargo tree --manifest-path (Join-Path $HARNESS_REPO "Cargo.toml") --locked --target x86_64-pc-windows-msvc -p llama-harness-ollama -i reqwest@0.12.28
cargo tree --manifest-path (Join-Path $HARNESS_REPO "Cargo.toml") --locked --target x86_64-pc-windows-msvc -p llama-harness-ollama -e features -i reqwest@0.12.28
cargo tree --manifest-path (Join-Path $HARNESS_REPO "Cargo.toml") --locked --target x86_64-pc-windows-msvc -p llama-harness-ollama -i rustls@0.23.43

$noteTree = cargo tree --manifest-path $NOTE_MANIFEST --locked --target x86_64-pc-windows-msvc -p note --all-features --prefix none --format '{p}'
$harnessTree = cargo tree --manifest-path (Join-Path $HARNESS_REPO "Cargo.toml") --locked --target x86_64-pc-windows-msvc -p llama-harness --all-features --prefix none --format '{p}'
[pscustomobject]@{
  TreeEntries = @($noteTree).Count
  UniquePackages = @($noteTree | Sort-Object -Unique).Count
}
[pscustomobject]@{
  TreeEntries = @($harnessTree).Count
  UniquePackages = @($harnessTree | Sort-Object -Unique).Count
}
```

## Recorded result and decision

Recorded on 2026-08-13: Note directly selects `reqwest 0.13.4` with `json`,
`rustls`, and `stream`, resolving `rustls 0.23.42`. llama-harness Ollama
directly selects `reqwest 0.12.28` with `json`, `rustls-tls`, and `stream`,
resolving `rustls 0.23.43`. The package-rooted, all-feature target trees had
459 unique Note packages (1,233 tree entries) and 405 unique facade packages
(953 tree entries) before embedding.

Cargo permits the two reqwest minor lines, so both reqwest facades and their
version-specific dependencies would coexist when harness is embedded in Note.
This branch does not claim a combined binary or target-size measurement because
it does not build a modified Note. Aligning reqwest safely requires an Ollama
client API and behavior review, plus a combined Note build; that dependency
upgrade is proposed as a separate follow-up rather than broad churn in the
Tauri boundary branch. Re-run this evidence after either recorded revision,
lockfile, direct networking dependency, target, or feature set changes.
