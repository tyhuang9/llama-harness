#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv-litellm"
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

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$HOST_PYTHON" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"

printf 'LiteLLM dev Python: '
"$VENV_PYTHON" -c "import sys; print(sys.executable)"

printf 'LiteLLM version: '
"$VENV_PYTHON" -c "import importlib.metadata as metadata; import litellm; print(metadata.version('litellm'))"
