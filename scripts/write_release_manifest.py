"""Write checksums and a machine-readable manifest for release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .validate_release_version import validate_release_version
except ImportError:
    from validate_release_version import validate_release_version


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate_release_version(args.version)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        artifacts = args.artifacts.resolve()
        if not artifacts.exists():
            raise FileNotFoundError(f"release artifacts directory does not exist: {artifacts}")
        if not artifacts.is_dir():
            raise NotADirectoryError(f"release artifacts path is not a directory: {artifacts}")
        files = sorted(
            path
            for path in artifacts.iterdir()
            if path.is_file() and path.name not in {"checksums.sha256", "release-manifest.json"}
        )
        nested_files = [path for path in artifacts.rglob("*") if path.is_file() and path.parent != artifacts]
    except OSError as error:
        raise SystemExit(f"cannot scan release artifacts: {error}") from error
    if nested_files:
        raise SystemExit(f"release artifacts must be root files: {nested_files}")
    if not files:
        raise SystemExit("no release artifacts found")
    entries = []
    checksum_lines = []
    for path in files:
        relative = path.relative_to(artifacts).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
        checksum_lines.append(f"{digest}  {relative}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "version": args.version, "artifacts": entries}, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
