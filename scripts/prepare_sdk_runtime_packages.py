"""Stage platform-specific SDK payloads from one verified runtime executable.

The script is release infrastructure, not an installer: it never downloads a
runtime and only copies the executable supplied by the build matrix.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PYTHON_TAGS = {
    "win32-x64": "win_amd64",
    "darwin-arm64": "macosx_11_0_arm64",
    "linux-x64": "manylinux_2_17_x86_64",
}


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
    pyproject = python_root / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8").replace('version = "0.1.0"', f'version = "{args.version}"', 1)
    pyproject.write_text(content, encoding="utf-8")
    (output / "platform.json").write_text(
        json.dumps({"platform": args.platform, "python_platform_tag": PYTHON_TAGS[args.platform], "runtime": runtime_name}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
