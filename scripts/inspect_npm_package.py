"""Reject unsafe or unexpected payloads in packed npm archives."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath


SDK_FILES = {
    "package/package.json",
    "package/LICENSE",
    "package/README.md",
    "package/dist/index.js",
    "package/dist/index.d.ts",
}


def _safe_member_name(name: str) -> str:
    if not name or "\\" in name:
        raise ValueError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    canonical = path.as_posix()
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    if name.rstrip("/") != canonical:
        raise ValueError(f"non-canonical archive member path: {name!r}")
    return canonical


def inspect_npm_package(
    package_path: Path,
    expected_name: str,
    expected_version: str,
    require_runtime: bool = False,
) -> int:
    members: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(package_path, "r:gz") as archive:
        for member in archive.getmembers():
            name = _safe_member_name(member.name)
            if name in members:
                raise ValueError(f"{package_path}: duplicate archive member: {name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"{package_path}: links and special archive members are forbidden: {name}")
            members[name] = member

        files = {name for name, member in members.items() if member.isfile()}
        if require_runtime:
            extension = ".exe" if expected_name.endswith("win32-x64") else ""
            expected_files = {
                "package/package.json",
                "package/LICENSE",
                "package/README.md",
                f"package/bin/llama-harness-runtime{extension}",
            }
        else:
            expected_files = SDK_FILES
        allowed_directories = {
            str(parent)
            for name in expected_files
            for parent in PurePosixPath(name).parents
            if str(parent) != "."
        }
        directories = {name for name, member in members.items() if member.isdir()}
        if files != expected_files or not directories.issubset(allowed_directories):
            missing = sorted(expected_files - files)
            unexpected = sorted((files - expected_files) | (directories - allowed_directories))
            raise ValueError(f"{package_path}: package contents differ; missing={missing}, unexpected={unexpected}")

        manifest_member = members.get("package/package.json")
        if manifest_member is None or not manifest_member.isfile():
            raise ValueError(f"{package_path}: missing package.json")
        manifest = archive.extractfile(manifest_member)
        if manifest is None:
            raise ValueError(f"{package_path}: package.json is unreadable")
        metadata = json.load(manifest)
    if metadata.get("name") != expected_name or metadata.get("version") != expected_version:
        raise ValueError(f"{package_path}: package metadata does not match {expected_name}@{expected_version}")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    try:
        count = inspect_npm_package(args.package, args.name, args.version, args.require_runtime)
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as error:
        raise SystemExit(str(error)) from error
    print(f"{args.package}: {count} files; package contents verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
