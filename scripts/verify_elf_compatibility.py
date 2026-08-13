"""Fail closed when an ELF binary requires a newer glibc than declared."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


GLIBC_SYMBOL = re.compile(r"\bGLIBC_(\d+)\.(\d+)\b")


def required_glibc_versions(readelf_output: str) -> set[tuple[int, int]]:
    return {(int(major), int(minor)) for major, minor in GLIBC_SYMBOL.findall(readelf_output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--max-glibc", required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"ELF binary does not exist: {binary}")
    try:
        allowed = tuple(int(part) for part in args.max_glibc.split("."))
        if len(allowed) != 2:
            raise ValueError
    except ValueError as error:
        raise SystemExit("--max-glibc must be MAJOR.MINOR") from error
    header = subprocess.run(
        ["readelf", "--file-header", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "ELF" not in header:
        raise SystemExit(f"{binary}: readelf did not identify an ELF binary")
    version_info = subprocess.run(
        ["readelf", "--wide", "--version-info", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    required = required_glibc_versions(version_info)
    newer = sorted(version for version in required if version > allowed)
    if newer:
        formatted = ", ".join(f"{major}.{minor}" for major, minor in newer)
        raise SystemExit(f"{binary}: requires GLIBC newer than {args.max_glibc}: {formatted}")
    maximum = max(required, default=None)
    detail = "static/no GLIBC symbols" if maximum is None else f"maximum GLIBC_{maximum[0]}.{maximum[1]}"
    print(f"{binary}: {detail}; compatible with glibc {args.max_glibc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
