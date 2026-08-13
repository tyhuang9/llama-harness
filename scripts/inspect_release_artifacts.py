"""Verify the exact release artifact set, package contents, and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

try:
    from .inspect_npm_package import inspect_npm_package
    from .inspect_python_packages import inspect_wheel
    from .validate_release_version import validate_release_version
except ImportError:
    from inspect_npm_package import inspect_npm_package
    from inspect_python_packages import inspect_wheel
    from validate_release_version import validate_release_version


PLATFORMS = {
    "win32-x64": ("llama-harness-runtime-win32-x64.exe", "win_amd64"),
    "darwin-arm64": ("llama-harness-runtime-darwin-arm64", "macosx_11_0_arm64"),
    "linux-x64": ("llama-harness-runtime-linux-x64", "manylinux_2_35_x86_64"),
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_artifacts(version: str, platforms: list[str], require_sdk: bool) -> dict[str, tuple[str, str] | None]:
    expected: dict[str, tuple[str, str] | None] = {}
    for platform in platforms:
        if platform not in PLATFORMS:
            raise ValueError(f"unsupported release platform: {platform}")
        runtime, wheel_tag = PLATFORMS[platform]
        expected[runtime] = None
        expected[f"llama-harness-runtime-{platform}-{version}.tgz"] = ("npm-runtime", platform)
        expected[f"llama_harness-{version}-py3-none-{wheel_tag}.whl"] = ("wheel", wheel_tag)
    if require_sdk:
        expected[f"llama-harness-sdk-{version}.tgz"] = ("npm-sdk", "")
    return expected


def _load_manifest(path: Path, version: str) -> list[dict[str, object]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "version", "artifacts"}:
        raise ValueError("release manifest has unknown or missing fields")
    if manifest["schema_version"] != 1:
        raise ValueError("release manifest schema_version must be 1")
    if manifest["version"] != version:
        raise ValueError("release manifest version does not match requested version")
    entries = manifest["artifacts"]
    if not isinstance(entries, list):
        raise ValueError("release manifest artifacts must be a list")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("release manifest artifact has unknown or missing fields")
        path_value = entry["path"]
        if not isinstance(path_value, str) or not path_value or path_value in seen:
            raise ValueError(f"release manifest has invalid or duplicate path: {path_value!r}")
        path = PurePosixPath(path_value)
        if path.is_absolute() or len(path.parts) != 1 or path.as_posix() != path_value:
            raise ValueError(f"release manifest path must be a root file: {path_value!r}")
        if not isinstance(entry["bytes"], int) or entry["bytes"] < 0:
            raise ValueError(f"release manifest has invalid size for {path_value}")
        if not isinstance(entry["sha256"], str) or not SHA256.fullmatch(entry["sha256"]):
            raise ValueError(f"release manifest has invalid SHA-256 for {path_value}")
        seen.add(path_value)
    return entries


def _load_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or not SHA256.fullmatch(parts[0]):
            raise ValueError(f"malformed checksum line: {line!r}")
        name = parts[1]
        member = PurePosixPath(name)
        if not name or member.is_absolute() or len(member.parts) != 1 or member.as_posix() != name:
            raise ValueError(f"checksum path must be a root file: {name!r}")
        if name in checksums:
            raise ValueError(f"duplicate checksum path: {name}")
        checksums[name] = parts[0]
    return checksums


def inspect_release_artifacts(artifacts: Path, version: str, platforms: list[str], require_sdk: bool) -> int:
    validate_release_version(version)
    if len(set(platforms)) != len(platforms):
        raise ValueError("release platform arguments must be unique")
    root = artifacts.resolve()
    expected_contract = expected_artifacts(version, platforms, require_sdk)
    actual_files = {
        path.name: path
        for path in root.iterdir()
        if path.is_file() and path.name not in {"checksums.sha256", "release-manifest.json"}
    }
    if set(actual_files) != set(expected_contract):
        missing = sorted(set(expected_contract) - set(actual_files))
        unexpected = sorted(set(actual_files) - set(expected_contract))
        raise ValueError(f"release artifact set differs; missing={missing}, unexpected={unexpected}")
    nested_files = [path for path in root.rglob("*") if path.is_file() and path.parent != root]
    if nested_files:
        raise ValueError(f"release artifacts must be root files: {nested_files}")

    entries = _load_manifest(root / "release-manifest.json", version)
    measured = {name: {"bytes": path.stat().st_size, "sha256": digest(path)} for name, path in actual_files.items()}
    described = {entry["path"]: {"bytes": entry["bytes"], "sha256": entry["sha256"]} for entry in entries}
    if described != measured:
        raise ValueError("release manifest does not exactly describe the staged artifacts")
    checksums = _load_checksums(root / "checksums.sha256")
    if checksums != {name: metadata["sha256"] for name, metadata in measured.items()}:
        raise ValueError("checksums.sha256 does not exactly describe the staged artifacts")

    for name, contract in expected_contract.items():
        if contract is None:
            continue
        kind, detail = contract
        path = actual_files[name]
        if kind == "npm-runtime":
            inspect_npm_package(path, f"@llama-harness/runtime-{detail}", version, True)
        elif kind == "npm-sdk":
            inspect_npm_package(path, "@llama-harness/sdk", version)
        elif kind == "wheel":
            inspect_wheel(path, version, True, detail)
    return len(actual_files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", action="append", required=True)
    parser.add_argument("--require-sdk", action="store_true")
    args = parser.parse_args()
    try:
        count = inspect_release_artifacts(args.artifacts, args.version, args.platform, args.require_sdk)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(f"{args.artifacts}: {count} artifacts, package contents, manifest, and checksums verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
