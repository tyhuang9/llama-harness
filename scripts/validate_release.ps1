param(
  [Parameter(Mandatory = $true)][string]$Platform,
  [Parameter(Mandatory = $true)][string]$Target,
  [Parameter(Mandatory = $true)][string]$Executable,
  [string]$Version = "0.2.0",
  [string]$Output = "release-stage/local"
)

$ErrorActionPreference = "Stop"
python scripts/validate_release_version.py $Version
if ($LASTEXITCODE) { exit $LASTEXITCODE }
if (Test-Path -LiteralPath $Output) { throw "refusing to overwrite existing release output: $Output" }
$cargoMetadata = cargo metadata --locked --format-version 1 --no-deps | ConvertFrom-Json
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$workspacePackages = @($cargoMetadata.packages | Where-Object { $cargoMetadata.workspace_members -contains $_.id })
$wrongCargoVersions = @($workspacePackages | Where-Object { $_.version -ne $Version } | ForEach-Object { "$($_.name)=$($_.version)" })
if ($wrongCargoVersions.Count -ne 0) {
  throw "Cargo workspace package versions do not match release version $Version: $($wrongCargoVersions -join ', ')"
}
$runtimePackage = @($workspacePackages | Where-Object { $_.name -eq "llama-harness-runtime" })
if ($runtimePackage.Count -ne 1) { throw "expected exactly one llama-harness-runtime package in Cargo workspace metadata" }
$npmVersion = node -p "require('./sdks/typescript/packages/sdk/package.json').version"
if ($npmVersion -ne $Version) { throw "npm SDK version $npmVersion does not match release version" }
$pythonVersion = (Select-String -Path "sdks/python/pyproject.toml" -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
if ($pythonVersion -ne $Version) { throw "Python SDK version $pythonVersion does not match release version" }
$work = Join-Path $Output "work"
$artifacts = Join-Path $Output "artifacts"
New-Item -ItemType Directory -Force -Path $work, $artifacts | Out-Null
$sdkStage = Join-Path (Resolve-Path $work) "sdk-stage"
$artifacts = (Resolve-Path $artifacts).Path

cargo build --release --locked --target $Target -p llama-harness-runtime
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$runtimeName = "llama-harness-runtime-$Platform"
if ($Executable.EndsWith(".exe")) { $runtimeName += ".exe" }
$runtime = Join-Path $artifacts $runtimeName
Copy-Item "target/$Target/release/$Executable" $runtime
& $runtime --help | Out-Null
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$processInfo = [System.Diagnostics.ProcessStartInfo]::new()
$processInfo.FileName = (Resolve-Path -LiteralPath $runtime).Path
$processInfo.UseShellExecute = $false
$processInfo.CreateNoWindow = $true
$processInfo.RedirectStandardInput = $true
$processInfo.RedirectStandardOutput = $true
$processInfo.RedirectStandardError = $true
$runtimeProcess = [System.Diagnostics.Process]::new()
$runtimeProcess.StartInfo = $processInfo
if (-not $runtimeProcess.Start()) { throw "failed to start runtime for release identity validation" }
$clientHello = [ordered]@{
  protocol_version = "1.1"
  request_id = "release-runtime-identity"
  type = "client_hello"
  payload = [ordered]@{
    sdk = [ordered]@{ name = "llama-harness-release-validation"; version = $Version }
    capabilities = @()
  }
} | ConvertTo-Json -Compress -Depth 8
$runtimeProcess.StandardInput.WriteLine($clientHello)
$runtimeProcess.StandardInput.Close()
$runtimeStdout = $runtimeProcess.StandardOutput.ReadToEnd()
$runtimeStderr = $runtimeProcess.StandardError.ReadToEnd()
$runtimeProcess.WaitForExit()
if ($runtimeProcess.ExitCode -ne 0) {
  throw "runtime hello validation exited with $($runtimeProcess.ExitCode): $runtimeStderr"
}
$runtimeLines = @($runtimeStdout -split "`r?`n" | Where-Object { $_.Trim() })
if ($runtimeLines.Count -ne 1) { throw "runtime hello validation expected one JSONL response, received $($runtimeLines.Count)" }
try { $runtimeHello = $runtimeLines[0] | ConvertFrom-Json -Depth 32 } catch { throw "runtime hello response is not JSON: $($_.Exception.Message)" }
if ($runtimeHello.type -ne "runtime_hello" -or $runtimeHello.request_id -ne "release-runtime-identity") {
  throw "runtime did not return the expected runtime_hello response"
}
if ($runtimeHello.payload.runtime_version -ne $Version) {
  throw "runtime hello version $($runtimeHello.payload.runtime_version) does not match release version $Version"
}
if ($Platform -eq "linux-x64") {
  python scripts/verify_elf_compatibility.py --binary $runtime --max-glibc 2.35
  if ($LASTEXITCODE) { exit $LASTEXITCODE }
}

python scripts/prepare_sdk_runtime_packages.py --runtime $runtime --platform $Platform --version $Version --out $sdkStage
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$npmPackage = Join-Path $sdkStage "npm/runtime-$Platform"
Push-Location $artifacts
try { npm pack $npmPackage | Out-Null; if ($LASTEXITCODE) { exit $LASTEXITCODE } } finally { Pop-Location }
python scripts/inspect_npm_package.py --package (Join-Path $artifacts "llama-harness-runtime-$Platform-$Version.tgz") --name "@llama-harness/runtime-$Platform" --version $Version --require-runtime
if ($LASTEXITCODE) { exit $LASTEXITCODE }

python -m build --wheel --outdir $artifacts (Join-Path $sdkStage "python-source")
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$tag = (Get-Content (Join-Path $sdkStage "platform.json") | ConvertFrom-Json).python_platform_tag
$wheel = (Get-ChildItem -LiteralPath $artifacts -Filter "*.whl" | Select-Object -First 1).FullName
python -m wheel tags --platform-tag $tag --remove $wheel
if ($LASTEXITCODE) { exit $LASTEXITCODE }
python scripts/inspect_python_packages.py --dist $artifacts --version $Version --require-runtime --platform-tag $tag
if ($LASTEXITCODE) { exit $LASTEXITCODE }
python scripts/write_release_manifest.py --artifacts $artifacts --version $Version --output (Join-Path $artifacts "release-manifest.json")
if ($LASTEXITCODE) { exit $LASTEXITCODE }
python scripts/inspect_release_artifacts.py --artifacts $artifacts --version $Version --platform $Platform
if ($LASTEXITCODE) { exit $LASTEXITCODE }
