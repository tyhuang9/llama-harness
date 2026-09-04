"""Stage platform-specific SDK payloads from one verified runtime executable.

The script is release infrastructure, not an installer: it never downloads a
runtime and only copies the executable supplied by the build matrix.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 release hosts use the strict fallback below.
    tomllib = None


PYTHON_TAGS = {
    "win32-x64": "win_amd64",
    "darwin-arm64": "macosx_11_0_arm64",
    "linux-x64": "manylinux_2_35_x86_64",
}


def python_project_version(project: Path) -> str:
    content = project.read_text(encoding="utf-8")
    if tomllib is not None:
        try:
            version = tomllib.loads(content)["project"]["version"]
        except (KeyError, tomllib.TOMLDecodeError) as error:
            raise ValueError(f"cannot read Python SDK version: {error}") from error
        if not isinstance(version, str):
            raise ValueError("Python SDK version must be a string")
        return version

    in_project = False
    for line in content.splitlines():
        if re.fullmatch(r"\s*\[project\]\s*(?:#.*)?", line):
            in_project = True
            continue
        if re.match(r"\s*\[", line):
            in_project = False
        if in_project:
            match = re.fullmatch(r'\s*version\s*=\s*"([^"]+)"\s*(?:#.*)?', line)
            if match:
                return match.group(1)
    raise ValueError("cannot read Python SDK version on Python 3.10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--platform", choices=sorted(PYTHON_TAGS), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    runtime = args.runtime.resolve()
    if not runtime.is_file():
        raise SystemExit(f"runtime does not exist: {runtime}")
    python_project = repository / "sdks" / "python" / "pyproject.toml"
    try:
        python_version = python_project_version(python_project)
    except ValueError as error:
        raise SystemExit(f"cannot read Python SDK version from {python_project}: {error}") from error
    if python_version != args.version:
        raise SystemExit(
            f"Python SDK version {python_version!r} does not match requested release version {args.version!r}"
        )
    output = args.out.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)

    extension = ".exe" if args.platform.startswith("win32") else ""
    runtime_name = f"llama-harness-runtime{extension}"
    npm_root = output / "npm" / f"runtime-{args.platform}"
    npm_bin = npm_root / "bin"
    npm_bin.mkdir(parents=True)
    shutil.copy2(runtime, npm_bin / runtime_name)
    (npm_root / "package.json").write_text(
        json.dumps(
            {
                "name": f"@llama-harness/runtime-{args.platform}",
                "version": args.version,
                "description": "Platform runtime for @llama-harness/sdk",
                "license": "MIT",
                "os": [args.platform.split("-")[0]],
                "cpu": [args.platform.rsplit("-", 1)[1]],
                "files": [f"bin/{runtime_name}", "README.md", "LICENSE"],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(repository / "LICENSE", npm_root / "LICENSE")
    (npm_root / "README.md").write_text(
        "# llama-harness platform runtime\n\nInstalled by `@llama-harness/sdk`; do not execute it as a network service.\n",
        encoding="utf-8",
    )

    python_root = output / "python-source"
    shutil.copytree(repository / "sdks" / "python", python_root, ignore=shutil.ignore_patterns("dist", "__pycache__"))
    staged_runtime = python_root / "src" / "llama_harness" / "runtime" / runtime_name
    staged_runtime.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runtime, staged_runtime)
    (output / "platform.json").write_text(
        json.dumps({"platform": args.platform, "python_platform_tag": PYTHON_TAGS[args.platform], "runtime": runtime_name}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
