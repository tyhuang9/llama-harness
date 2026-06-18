$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$RuntimeDir = Join-Path $RepoRoot "bundled\litellm-runtime"
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

if (Test-Path $RuntimeDir) {
    Remove-Item -Recurse -Force $RuntimeDir
}

New-Item -ItemType Directory -Force $RuntimeDir | Out-Null
& $HostPython.Exe @($HostPython.Args) -m venv $RuntimeDir

$RuntimePython = Join-Path $RuntimeDir "Scripts\python.exe"

& $RuntimePython -m pip install --upgrade pip
& $RuntimePython -m pip install -r $RequirementsFile

Write-Host -NoNewline "Bundled LiteLLM runtime Python: "
& $RuntimePython -c "import sys; print(sys.executable)"

Write-Host -NoNewline "LiteLLM version: "
& $RuntimePython -c "import importlib.metadata as metadata; import litellm; print(metadata.version('litellm'))"
