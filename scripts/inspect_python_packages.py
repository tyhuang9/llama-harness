"""Reject unexpected payloads in built Python SDK distributions."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def main() -> int:
    dist = Path("sdks/python/dist")
    wheels = list(dist.glob("*.whl"))
    if not wheels:
        raise SystemExit("no Python wheel found; run `python -m build sdks/python` first")
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
        forbidden = [name for name in names if ".env" in name or name.endswith(".pyc")]
        if forbidden:
            raise SystemExit(f"{wheel}: forbidden package contents: {forbidden}")
        print(f"{wheel}: {len(names)} files; no secret-like or bytecode payloads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
