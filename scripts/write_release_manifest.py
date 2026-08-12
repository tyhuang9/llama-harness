"""Write checksums and a machine-readable manifest for release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    files = sorted(path for path in artifacts.rglob("*") if path.is_file() and path.name not in {"checksums.sha256", "release-manifest.json"})
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
