from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
import re
from unittest import mock

from scripts.inspect_npm_package import SDK_FILES, inspect_npm_package
from scripts.inspect_python_packages import PACKAGE_FILES, inspect_wheel
from scripts.inspect_release_artifacts import PLATFORMS, inspect_release_artifacts
from scripts.validate_release_version import validate_release_version
from scripts.verify_elf_compatibility import required_glibc_versions
from scripts.write_release_manifest import main as write_release_manifest


VERSION = "1.2.3"


def write_tar(path: Path, files: dict[str, bytes], extra_members: list[tarfile.TarInfo] | None = None) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        for member in extra_members or []:
            archive.addfile(member, io.BytesIO(b"x") if member.size else None)


def npm_files(name: str, runtime: str | None = None) -> dict[str, bytes]:
    files = {
        "package/package.json": json.dumps({"name": name, "version": VERSION}).encode(),
        "package/LICENSE": b"license",
        "package/README.md": b"readme",
    }
    if runtime:
        files[f"package/bin/{runtime}"] = b"runtime"
    else:
        files["package/dist/index.js"] = b"export {}"
        files["package/dist/index.d.ts"] = b"export {}"
    return files


def wheel_files(runtime: str | None = None) -> dict[str, bytes]:
    dist_info = f"llama_harness-{VERSION}.dist-info"
    files = {name: b"payload" for name in PACKAGE_FILES}
    files.update(
        {
            f"{dist_info}/METADATA": b"metadata",
            f"{dist_info}/WHEEL": b"wheel",
            f"{dist_info}/RECORD": b"record",
            f"{dist_info}/licenses/LICENSE": b"license",
        }
    )
    if runtime:
        files[f"llama_harness/runtime/{runtime}"] = b"runtime"
    return files


def write_wheel(path: Path, files: dict[str, bytes], duplicate: str | None = None, symlink: str | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(duplicate, b"duplicate")
        if symlink:
            member = zipfile.ZipInfo(symlink)
            member.create_system = 3
            member.external_attr = 0o120777 << 16
            archive.writestr(member, b"target")


def write_release_metadata(root: Path, schema_version: int = 1) -> None:
    files = sorted(path for path in root.iterdir() if path.is_file() and path.name not in {"checksums.sha256", "release-manifest.json"})
    entries = []
    checksums = []
    for path in files:
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256})
        checksums.append(f"{sha256}  {path.name}")
    (root / "release-manifest.json").write_text(
        json.dumps({"schema_version": schema_version, "version": VERSION, "artifacts": entries}), encoding="utf-8"
    )
    (root / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")


def write_complete_release(root: Path) -> None:
    for platform, (runtime_artifact, wheel_tag) in PLATFORMS.items():
        (root / runtime_artifact).write_bytes(b"runtime")
        runtime_name = "llama-harness-runtime.exe" if platform == "win32-x64" else "llama-harness-runtime"
        write_tar(
            root / f"llama-harness-runtime-{platform}-{VERSION}.tgz",
            npm_files(f"@llama-harness/runtime-{platform}", runtime_name),
        )
        write_wheel(
            root / f"llama_harness-{VERSION}-py3-none-{wheel_tag}.whl",
            wheel_files(runtime_name),
        )
    write_tar(root / f"llama-harness-sdk-{VERSION}.tgz", npm_files("@llama-harness/sdk"))
    write_release_metadata(root)


class NpmArchiveTests(unittest.TestCase):
    def test_accepts_exact_sdk_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "sdk.tgz"
            files = npm_files("@llama-harness/sdk")
            self.assertEqual(set(files), SDK_FILES)
            write_tar(package, files)
            self.assertEqual(inspect_npm_package(package, "@llama-harness/sdk", VERSION), 5)

    def test_rejects_traversal_links_duplicates_and_unexpected_files(self) -> None:
        cases: list[tuple[str, dict[str, bytes], list[tarfile.TarInfo]]] = []
        traversal = npm_files("@llama-harness/sdk")
        traversal["package/../escape"] = b"bad"
        cases.append(("traversal", traversal, []))
        symlink = tarfile.TarInfo("package/link")
        symlink.type = tarfile.SYMTYPE
        symlink.linkname = "package/package.json"
        cases.append(("link", npm_files("@llama-harness/sdk"), [symlink]))
        duplicate = tarfile.TarInfo("package/package.json")
        duplicate.size = 1
        cases.append(("duplicate", npm_files("@llama-harness/sdk"), [duplicate]))
        unexpected = npm_files("@llama-harness/sdk")
        unexpected["package/.env"] = b"secret"
        cases.append(("unexpected", unexpected, []))
        for name, files, members in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                package = Path(directory) / "sdk.tgz"
                write_tar(package, files, members)
                with self.assertRaises(ValueError):
                    inspect_npm_package(package, "@llama-harness/sdk", VERSION)


class WheelArchiveTests(unittest.TestCase):
    def test_accepts_exact_platform_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / f"llama_harness-{VERSION}-py3-none-win_amd64.whl"
            write_wheel(wheel, wheel_files("llama-harness-runtime.exe"))
            self.assertEqual(inspect_wheel(wheel, VERSION, True, "win_amd64"), 8)

    def test_rejects_traversal_links_duplicates_and_unexpected_files(self) -> None:
        base = wheel_files()
        cases = [
            ("traversal", {**base, "../escape": b"bad"}, None, None),
            ("symlink", base, None, "llama_harness/link"),
            ("duplicate", base, "llama_harness/client.py", None),
            ("unexpected", {**base, "llama_harness/__pycache__/client.pyc": b"bad"}, None, None),
        ]
        for name, files, duplicate, symlink in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                wheel = Path(directory) / f"llama_harness-{VERSION}-py3-none-any.whl"
                write_wheel(wheel, files, duplicate, symlink)
                with self.assertRaises(ValueError):
                    inspect_wheel(wheel, VERSION, False, None)


class ReleaseSetTests(unittest.TestCase):
    def test_accepts_exact_combined_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_complete_release(root)
            self.assertEqual(inspect_release_artifacts(root, VERSION, list(PLATFORMS), True), 10)

    def test_rejects_missing_extra_duplicate_and_unknown_schema(self) -> None:
        mutations = ("missing-wheel", "extra", "duplicate-manifest", "duplicate-checksum", "schema")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_complete_release(root)
                if mutation == "missing-wheel":
                    (root / f"llama_harness-{VERSION}-py3-none-macosx_11_0_arm64.whl").unlink()
                elif mutation == "extra":
                    (root / "unknown.bin").write_bytes(b"extra")
                elif mutation == "duplicate-manifest":
                    manifest_path = root / "release-manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["artifacts"].append(manifest["artifacts"][0])
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                elif mutation == "duplicate-checksum":
                    checksum_path = root / "checksums.sha256"
                    first = checksum_path.read_text(encoding="utf-8").splitlines()[0]
                    checksum_path.write_text(checksum_path.read_text(encoding="utf-8") + first + "\n", encoding="utf-8")
                elif mutation == "schema":
                    write_release_metadata(root, schema_version=2)
                with self.assertRaises(ValueError):
                    inspect_release_artifacts(root, VERSION, list(PLATFORMS), True)


class InputAndAbiTests(unittest.TestCase):
    def test_manifest_writer_rejects_missing_and_non_directory_artifact_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for artifacts in (root / "missing", root / "artifact-file"):
                if artifacts.name == "artifact-file":
                    artifacts.write_text("not a directory", encoding="utf-8")
                output = root / f"{artifacts.name}.json"
                with self.subTest(artifacts=artifacts), self.assertRaisesRegex(
                    SystemExit, "cannot scan release artifacts"
                ), mock.patch.object(
                    sys,
                    "argv",
                    [
                        "write_release_manifest.py",
                        "--artifacts",
                        str(artifacts),
                        "--version",
                        VERSION,
                        "--output",
                        str(output),
                    ],
                ):
                    write_release_manifest()

    def test_manifest_writer_reports_artifact_scan_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            with self.assertRaisesRegex(SystemExit, "cannot scan release artifacts"), mock.patch.object(
                Path, "iterdir", side_effect=PermissionError("access denied")
            ), mock.patch.object(
                sys,
                "argv",
                [
                    "write_release_manifest.py",
                    "--artifacts",
                    str(artifacts),
                    "--version",
                    VERSION,
                    "--output",
                    str(root / "release-manifest.json"),
                ],
            ):
                write_release_manifest()

    def test_release_version_rejects_shell_and_package_metacharacters(self) -> None:
        for value in ("1.2.3; whoami", "1.2", "01.2.3", "1.2.3-alpha", "$(whoami)", "1.2.3/path"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_release_version(value)
        self.assertEqual(validate_release_version(VERSION), VERSION)

    def test_extracts_glibc_symbol_versions(self) -> None:
        output = "Name: GLIBC_2.17 Flags: none  Name: GLIBC_2.35 Name: GLIBCXX_3.4"
        self.assertEqual(required_glibc_versions(output), {(2, 17), (2, 35)})

    def test_workflows_pin_actions_and_do_not_interpolate_dispatch_version_in_scripts(self) -> None:
        root = Path(__file__).resolve().parents[2]
        workflows = list((root / ".github" / "workflows").glob("*.yml"))
        uses = re.compile(r"^\s*(?:-\s*)?uses:\s*[^\s@]+@([^\s#]+)", re.MULTILINE)
        for workflow in workflows:
            text = workflow.read_text(encoding="utf-8")
            with self.subTest(workflow=workflow.name):
                refs = uses.findall(text)
                self.assertTrue(refs)
                self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in refs), refs)
                for line in text.splitlines():
                    if "${{ inputs.version }}" in line:
                        self.assertEqual(line.strip(), "RELEASE_VERSION: ${{ inputs.version }}")


if __name__ == "__main__":
    unittest.main()
