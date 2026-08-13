"""Verify release artifact checksums and manifest entries before publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", action="append", required=True)
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    manifest = json.loads((artifacts / "release-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != args.version:
        raise SystemExit("release manifest version does not match requested version")
    files = sorted(path for path in artifacts.rglob("*") if path.is_file() and path.name not in {"checksums.sha256", "release-manifest.json"})
    expected = {path.relative_to(artifacts).as_posix(): {"bytes": path.stat().st_size, "sha256": digest(path)} for path in files}
    actual = {entry["path"]: {"bytes": entry["bytes"], "sha256": entry["sha256"]} for entry in manifest.get("artifacts", [])}
    if actual != expected:
        raise SystemExit("release manifest does not exactly describe the staged artifacts")
    checksums = {path: value for value, path in (line.split("  ", 1) for line in (artifacts / "checksums.sha256").read_text(encoding="utf-8").splitlines())}
    if checksums != {path: metadata["sha256"] for path, metadata in expected.items()}:
        raise SystemExit("checksums.sha256 does not exactly describe the staged artifacts")
    for platform in args.platform:
        runtime = [path for path in artifacts.glob(f"llama-harness-runtime-{platform}*") if path.suffix != ".tgz"]
        npm = list(artifacts.glob(f"llama-harness-runtime-{platform}-{args.version}.tgz"))
        wheel = list(artifacts.glob(f"llama_harness-{args.version}-*" + ".whl"))
        if len(runtime) != 1 or len(npm) != 1 or not wheel:
            raise SystemExit(f"missing staged artifacts for {platform}")
    print(f"{artifacts}: {len(files)} artifacts, manifest, and checksums verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
