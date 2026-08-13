"""Reject unexpected payloads in packed npm archives."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    with tarfile.open(args.package, "r:gz") as archive:
        names = archive.getnames()
        manifest = archive.extractfile("package/package.json")
        if manifest is None:
            raise SystemExit(f"{args.package}: missing package.json")
        package = json.load(manifest)
    if package.get("name") != args.name or package.get("version") != args.version:
        raise SystemExit(f"{args.package}: package metadata does not match {args.name}@{args.version}")
    forbidden = [name for name in names if not name.startswith("package/") or ".env" in name or name.endswith(".pyc") or "/node_modules/" in name]
    if forbidden:
        raise SystemExit(f"{args.package}: forbidden package contents: {forbidden}")
    if args.require_runtime and not any(name.startswith("package/bin/llama-harness-runtime") for name in names):
        raise SystemExit(f"{args.package}: missing packaged runtime")
    print(f"{args.package}: {len(names)} files; package contents verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
