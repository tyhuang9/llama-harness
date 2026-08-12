"""Reject unexpected payloads in built Python SDK distributions."""

from __future__ import annotations

import sys
import zipfile
import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path("sdks/python/dist"))
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    dist = args.dist
    wheels = list(dist.glob("*.whl"))
    if not wheels:
        raise SystemExit("no Python wheel found; run `python -m build sdks/python` first")
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
        forbidden = [name for name in names if ".env" in name or name.endswith(".pyc")]
        if forbidden:
            raise SystemExit(f"{wheel}: forbidden package contents: {forbidden}")
        if args.require_runtime and not any(
            name.endswith("llama_harness/runtime/llama-harness-runtime")
            or name.endswith("llama_harness/runtime/llama-harness-runtime.exe")
            for name in names
        ):
            raise SystemExit(f"{wheel}: missing packaged runtime")
        print(f"{wheel}: {len(names)} files; no secret-like or bytecode payloads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
