"""Validate the deliberately narrow release version accepted by package tooling."""

from __future__ import annotations

import re
import sys


RELEASE_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def validate_release_version(version: str) -> str:
    if not RELEASE_VERSION.fullmatch(version):
        raise ValueError("release version must be an exact stable SemVer (for example, 1.2.3)")
    return version


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_release_version.py VERSION")
    try:
        validate_release_version(sys.argv[1])
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
