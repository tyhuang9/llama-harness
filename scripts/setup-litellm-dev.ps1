$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$VenvDir = Join-Path $RepoRoot ".venv-litellm"
$RequirementsFile = Join-Path $RepoRoot "requirements-litellm.txt"

function Resolve-HostPython {
    if ($env:PYTHON) {
        return @{ Exe = $env:PYTHON; Args = @() }
    }

    foreach ($candidate in @(
        @{ Exe = "py"; Args = @("-3") },
        @{ Exe = "python"; Args = @() },
        @{ Exe = "python3"; Args = @() }
    )) {
        try {
            & $candidate.Exe @($candidate.Args) --version *> $null
            return $candidate
        } catch {
            continue
        }
    }

    throw "Python 3.10-3.13 is required but was not found on PATH."
}

$HostPython = Resolve-HostPython
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    & $HostPython.Exe @($HostPython.Args) -m venv $VenvDir
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r $RequirementsFile

Write-Host -NoNewline "LiteLLM dev Python: "
& $VenvPython -c "import sys; print(sys.executable)"

Write-Host -NoNewline "LiteLLM version: "
& $VenvPython -c "import importlib.metadata as metadata; import litellm; print(metadata.version('litellm'))"
