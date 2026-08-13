param(
  [Parameter(Mandatory = $true)][string]$Platform,
  [Parameter(Mandatory = $true)][string]$Target,
  [Parameter(Mandatory = $true)][string]$Executable,
  [string]$Version = "0.1.0",
  [string]$Output = "release-stage/local"
)

$ErrorActionPreference = "Stop"
if (Test-Path -LiteralPath $Output) { throw "refusing to overwrite existing release output: $Output" }
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
python scripts/inspect_python_packages.py --dist $artifacts --require-runtime
if ($LASTEXITCODE) { exit $LASTEXITCODE }
python scripts/write_release_manifest.py --artifacts $artifacts --version $Version --output (Join-Path $artifacts "release-manifest.json")
if ($LASTEXITCODE) { exit $LASTEXITCODE }
python scripts/inspect_release_artifacts.py --artifacts $artifacts --version $Version --platform $Platform
if ($LASTEXITCODE) { exit $LASTEXITCODE }
