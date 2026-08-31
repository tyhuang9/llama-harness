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
import subprocess
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

    def test_rust_release_workflow_is_validation_only_and_independent(self) -> None:
        root = Path(__file__).resolve().parents[2]
        workflow = (root / ".github" / "workflows" / "release-rust.yml").read_text(encoding="utf-8")
        preflight = (root / "scripts" / "release" / "check-rust-release.ps1").read_text(encoding="utf-8")

        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertIn('refs/heads/main', workflow)
        self.assertIn("toolchain: 1.88.0", workflow)
        self.assertIn("cargo run --locked --package xtask -- release-check", workflow)
        self.assertNotIn("cargo publish", workflow)
        self.assertNotIn("cargo owner", workflow)
        self.assertNotIn("gh release", workflow)
        self.assertNotIn("secrets.", workflow)
        self.assertNotIn("upload-artifact", workflow)
        self.assertNotIn("llama-harness-runtime", workflow)
        self.assertNotIn("llama-harness-protocol", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("SOURCE_COMMIT: ${{ github.sha }}", workflow)
        self.assertIn("REVIEWED_SOURCE_COMMIT: ${{ inputs.source_commit }}", workflow)

        self.assertNotIn("AllowDirty", preflight)
        self.assertIn("git status --porcelain=v1 --untracked-files=all", preflight)
        self.assertIn('$null -eq $_.publish', preflight)
        self.assertEqual(preflight.count('"llama-harness"'), 1)
        for crate in (
            "llama-harness-core",
            "llama-harness-evals",
            "llama-harness-observability",
            "llama-harness-ollama",
            "llama-harness-programmatic-sandbox",
            "llama-harness-tauri",
            "llama-harness-mcp",
        ):
            self.assertIn(f'"{crate}"', preflight)


class RustReleaseWorkflowTests(unittest.TestCase):
    @staticmethod
    def run_preflight(root: Path, source_commit: str, version: str = "0.1.0") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "pwsh",
                "-NoProfile",
                "-File",
                str(root / "scripts" / "release" / "check-rust-release.ps1"),
                "-Version",
                version,
                "-SourceCommit",
                source_commit,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    @staticmethod
    def clone_repository(root: Path, destination: Path) -> str:
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(root), str(destination)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        source_preflight = root / "scripts" / "release" / "check-rust-release.ps1"
        cloned_preflight = destination / "scripts" / "release" / "check-rust-release.ps1"
        cloned_preflight.write_text(source_preflight.read_text(encoding="utf-8"), encoding="utf-8")
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=destination,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    def test_rust_release_workflow_is_read_only_and_complete(self) -> None:
        root = Path(__file__).resolve().parents[2]
        workflow = (root / ".github" / "workflows" / "release-rust.yml").read_text(encoding="utf-8")

        self.assertIn("pull_request:", workflow)
        self.assertRegex(workflow, r"workflow_dispatch:\s+inputs:\s+version:")
        self.assertRegex(
            workflow,
            r"version:\s+description:[^\n]+\s+required: true[\s\S]*?\s+type: string",
        )
        self.assertRegex(workflow, r"permissions:\s+contents: read")
        self.assertIn("concurrency:", workflow)
        self.assertIn("cancel-in-progress: true", workflow)
        self.assertGreaterEqual(workflow.count("refs/heads/main"), 1)
        self.assertEqual(workflow.count("${{ inputs.version }}"), 1)
        self.assertEqual(workflow.count("${{ inputs.source_commit }}"), 1)
        self.assertEqual(workflow.count("${{ github.sha }}"), 1)
        self.assertIn("toolchain: stable", workflow)
        self.assertIn("toolchain: 1.88.0", workflow)

        required_commands = (
            "scripts/release/check-rust-release.ps1",
            "cargo deny check advisories licenses sources",
            "cargo run --locked --package xtask -- release-check",
            "scripts/semver/check-rust-semver.ps1",
            "cargo generate-lockfile",
            "cargo check --all-targets --all-features",
        )
        for command in required_commands:
            with self.subTest(command=command):
                self.assertIn(command, workflow)

        documented_crates = set(
            re.findall(r"--package (llama-harness(?:-[a-z]+)*)", workflow)
        )
        self.assertEqual(
            documented_crates,
            {
                "llama-harness-core",
                "llama-harness-ollama",
                "llama-harness-observability",
                "llama-harness-evals",
                "llama-harness-programmatic-sandbox",
                "llama-harness-tauri",
                "llama-harness-mcp",
                "llama-harness",
            },
        )

        forbidden = (
            "cargo publish",
            "secrets.",
            "contents: write",
            "packages: write",
            "id-token: write",
            "upload-artifact",
            "download-artifact",
            "git push",
            "gh release",
            "actions/setup-node",
            "actions/setup-python",
            "sdks/",
            "apps/harness-console",
            "llama-harness-runtime",
        )
        for value in forbidden:
            with self.subTest(forbidden=value):
                self.assertNotIn(value, workflow)

    def test_rust_release_preflight_has_exact_metadata_contract(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script = (root / "scripts" / "release" / "check-rust-release.ps1").read_text(encoding="utf-8")
        crate_block = re.search(
            r"\$expectedCrates = @\((.*?)\)\s*\|\s*Sort-Object\s*\$stableVersionPattern",
            script,
            re.DOTALL,
        )
        self.assertIsNotNone(crate_block)
        crates = set(re.findall(r'"(llama-harness(?:-[a-z]+)*)"', crate_block.group(1)))
        self.assertEqual(
            crates,
            {
                "llama-harness-core",
                "llama-harness-ollama",
                "llama-harness-observability",
                "llama-harness-evals",
                "llama-harness-programmatic-sandbox",
                "llama-harness-tauri",
                "llama-harness-mcp",
                "llama-harness",
            },
        )
        self.assertIn("cargo metadata --locked --format-version 1 --no-deps", script)
        self.assertIn('$expectedRustVersion = "1.88"', script)
        self.assertIn("exact stable SemVer", script)
        self.assertIn("Unreleased", script)
        self.assertIn("yyyy-MM-dd", script)
        self.assertIn("must not be empty", script)
        self.assertIn("git status --porcelain=v1 --untracked-files=all", script)
        self.assertIn('$null -eq $_.publish', script)
        self.assertIn("does not match reviewed source commit", script)
        self.assertNotIn("AllowDirty", script)

    def test_rust_release_preflight_rejects_a_different_source_commit(self) -> None:
        root = Path(__file__).resolve().parents[2]
        result = self.run_preflight(root, "0" * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match reviewed source commit", result.stdout + result.stderr)

    def test_rust_release_preflight_rejects_metadata_changelog_and_tree_drift(self) -> None:
        root = Path(__file__).resolve().parents[2]
        cases = (
            (
                "default-publish crate",
                lambda clone: (clone / "crates" / "llama-harness-protocol" / "Cargo.toml").write_text(
                    (clone / "crates" / "llama-harness-protocol" / "Cargo.toml")
                    .read_text(encoding="utf-8")
                    .replace("publish = false\n", ""),
                    encoding="utf-8",
                ),
                "Expected exactly the eight supported crates.io packages",
            ),
            (
                "MSRV mismatch",
                lambda clone: (clone / "Cargo.toml").write_text(
                    (clone / "Cargo.toml").read_text(encoding="utf-8").replace(
                        'rust-version = "1.88"', 'rust-version = "1.89"'
                    ),
                    encoding="utf-8",
                ),
                "Every published crate must declare Rust 1.88",
            ),
            (
                "duplicate changelog",
                lambda clone: (clone / "CHANGELOG.md").write_text(
                    (clone / "CHANGELOG.md").read_text(encoding="utf-8")
                    + "\n## 0.1.0 — Unreleased\n\n- Duplicate.\n",
                    encoding="utf-8",
                ),
                "exactly one level-two heading",
            ),
            (
                "empty changelog",
                lambda clone: (clone / "CHANGELOG.md").write_text(
                    "# Changelog\n\n## 0.1.0 — Unreleased\n\n## Versioning policy\n\nPolicy.\n",
                    encoding="utf-8",
                ),
                "must not be empty",
            ),
            (
                "comment-only changelog",
                lambda clone: (clone / "CHANGELOG.md").write_text(
                    "# Changelog\n\n## 0.1.0 — Unreleased\n\n<!--\nplaceholder\n-->\n\n"
                    "## Versioning policy\n\nPolicy.\n",
                    encoding="utf-8",
                ),
                "must not be empty",
            ),
            (
                "dirty tree",
                lambda clone: (clone / "untracked-release-file").write_text("dirty", encoding="utf-8"),
                "requires a clean Git working tree",
            ),
        )

        for name, mutate, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                clone = Path(directory) / "repository"
                source_commit = self.clone_repository(root, clone)
                mutate(clone)
                result = self.run_preflight(clone, source_commit)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stdout + result.stderr)

    def test_semver_gate_uses_the_latest_reachable_stable_release(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script = (root / "scripts" / "semver" / "check-rust-semver.ps1").read_text(encoding="utf-8")

        self.assertIn('git tag --merged HEAD --list "v[0-9]*.[0-9]*.[0-9]*" --sort=-version:refname', script)
        self.assertIn("--baseline-rev $comparisonTag", script)
        self.assertIn('$null -eq $_.publish', script)
        self.assertNotIn("--baseline-rev $expectedTag", script)


if __name__ == "__main__":
    unittest.main()
