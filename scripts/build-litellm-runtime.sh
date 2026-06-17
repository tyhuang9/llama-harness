#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="$REPO_ROOT/bundled/litellm-runtime"
REQUIREMENTS_FILE="$REPO_ROOT/requirements-litellm.txt"

find_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "$PYTHON"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  printf 'Python 3.10-3.13 is required but was not found on PATH.\n' >&2
  return 1
}

HOST_PYTHON="$(find_python)"

rm -rf "$RUNTIME_DIR"
mkdir -p "$RUNTIME_DIR"

"$HOST_PYTHON" -m venv "$RUNTIME_DIR"

RUNTIME_PYTHON="$RUNTIME_DIR/bin/python"

"$RUNTIME_PYTHON" -m pip install --upgrade pip
"$RUNTIME_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"

printf 'Bundled LiteLLM runtime Python: '
"$RUNTIME_PYTHON" -c "import sys; print(sys.executable)"

printf 'LiteLLM version: '
"$RUNTIME_PYTHON" -c "import importlib.metadata as metadata; import litellm; print(metadata.version('litellm'))"
