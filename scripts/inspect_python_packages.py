"""Reject unsafe or unexpected payloads in built Python SDK wheels."""

from __future__ import annotations

import argparse
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath


PACKAGE_FILES = {
    "llama_harness/__init__.py",
    "llama_harness/client.py",
    "llama_harness/runtime/.gitkeep",
}


def _safe_member_name(name: str) -> str:
    if not name or "\\" in name:
        raise ValueError(f"unsafe wheel member path: {name!r}")
    path = PurePosixPath(name)
    canonical = path.as_posix()
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe wheel member path: {name!r}")
    if name.rstrip("/") != canonical:
        raise ValueError(f"non-canonical wheel member path: {name!r}")
    return canonical


def inspect_wheel(wheel: Path, version: str, require_runtime: bool, platform_tag: str | None) -> int:
    expected_filename = f"llama_harness-{version}-py3-none-{platform_tag or 'any'}.whl"
    if wheel.name != expected_filename:
        raise ValueError(f"{wheel}: expected wheel filename {expected_filename}")
    dist_info = f"llama_harness-{version}.dist-info"
    expected = PACKAGE_FILES | {
        f"{dist_info}/METADATA",
        f"{dist_info}/WHEEL",
        f"{dist_info}/RECORD",
        f"{dist_info}/licenses/LICENSE",
    }
    if require_runtime:
        if platform_tag is None:
            raise ValueError("--platform-tag is required with --require-runtime")
        extension = ".exe" if platform_tag == "win_amd64" else ""
        expected.add(f"llama_harness/runtime/llama-harness-runtime{extension}")

    seen: set[str] = set()
    files: set[str] = set()
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            name = _safe_member_name(member.filename)
            if name in seen:
                raise ValueError(f"{wheel}: duplicate wheel member: {name}")
            seen.add(name)
            unix_type = (member.external_attr >> 16) & 0o170000
            if member.is_dir():
                continue
            if unix_type not in {0, stat.S_IFREG}:
                raise ValueError(f"{wheel}: links and special wheel members are forbidden: {name}")
            files.add(name)
    if files != expected:
        missing = sorted(expected - files)
        unexpected = sorted(files - expected)
        raise ValueError(f"{wheel}: wheel contents differ; missing={missing}, unexpected={unexpected}")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path("sdks/python/dist"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--require-runtime", action="store_true")
    parser.add_argument("--platform-tag")
    args = parser.parse_args()
    wheels = list(args.dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one Python wheel in {args.dist}; found {len(wheels)}")
    try:
        count = inspect_wheel(wheels[0], args.version, args.require_runtime, args.platform_tag)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error)) from error
    print(f"{wheels[0]}: {count} files; wheel contents verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
