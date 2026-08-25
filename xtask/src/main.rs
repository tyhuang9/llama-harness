use std::{
    collections::HashSet,
    env,
    ffi::{OsStr, OsString},
    fs,
    path::{Component, Path, PathBuf},
    process::{Command, ExitCode, Output},
    time::{SystemTime, UNIX_EPOCH},
};

const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");
const MAX_ARCHIVE_BYTES: u64 = 2 * 1024 * 1024;
const MAX_ENTRY_BYTES: u64 = 1024 * 1024;
const MAX_UNPACKED_BYTES: u64 = 8 * 1024 * 1024;
const MAX_ARCHIVE_ENTRIES: usize = 256;
const FACADE_FEATURES: &[&str] = &["ollama", "observability", "evals", "tauri"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PublishableCrate {
    name: &'static str,
}

const PUBLISHABLE_CRATES: &[PublishableCrate] = &[
    PublishableCrate {
        name: "llama-harness-core",
    },
    PublishableCrate {
        name: "llama-harness-ollama",
    },
    PublishableCrate {
        name: "llama-harness-observability",
    },
    PublishableCrate {
        name: "llama-harness-tauri",
    },
    PublishableCrate {
        name: "llama-harness-evals",
    },
    PublishableCrate {
        name: "llama-harness",
    },
];

type CheckResult<T = ()> = Result<T, String>;

fn main() -> ExitCode {
    let mut arguments = env::args().skip(1);
    let command = arguments.next();
    if arguments.next().is_some() {
        return usage();
    }

    let root = workspace_root();
    let result = match command.as_deref() {
        Some("release-check") => release_check(&root),
        Some("package-check") => package_check(&root),
        Some("protocol-check") => protocol_check(&root),
        _ => return usage(),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn usage() -> ExitCode {
    eprintln!("usage: cargo run -p xtask -- <release-check|package-check|protocol-check>");
    ExitCode::FAILURE
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be directly inside the workspace root")
        .to_path_buf()
}

fn protocol_check(root: &Path) -> CheckResult {
    run_cargo(
        root,
        ["test", "--locked", "--package", "llama-harness-protocol"],
    )
}

fn release_check(root: &Path) -> CheckResult {
    ensure_clean_tree(root)?;
    run_cargo(root, ["fmt", "--check", "--all"])?;
    run_cargo(
        root,
        [
            "clippy",
            "--locked",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ],
    )?;
    run_cargo(root, ["test", "--locked", "--workspace", "--all-features"])?;
    run_cargo_with_env(
        root,
        [
            "doc",
            "--locked",
            "--workspace",
            "--all-features",
            "--no-deps",
        ],
        [("RUSTDOCFLAGS", "-D warnings")],
    )?;
    package_check_after_clean(root)
}

fn package_check(root: &Path) -> CheckResult {
    ensure_clean_tree(root)?;
    package_check_after_clean(root)
}

fn package_check_after_clean(root: &Path) -> CheckResult {
    let temporary = TemporaryDirectory::new("llama-harness-package-check")?;
    let package_target = temporary.path().join("package-target");
    let extraction_root = temporary.path().join("extracted");
    fs::create_dir_all(&package_target)
        .map_err(|error| format!("failed to create {}: {error}", package_target.display()))?;
    fs::create_dir_all(&extraction_root)
        .map_err(|error| format!("failed to create {}: {error}", extraction_root.display()))?;

    let mut package_arguments = vec![
        OsString::from("package"),
        OsString::from("--locked"),
        OsString::from("--no-verify"),
        OsString::from("--target-dir"),
        package_target.as_os_str().to_owned(),
    ];
    for package in PUBLISHABLE_CRATES {
        package_arguments.push(OsString::from("--package"));
        package_arguments.push(OsString::from(package.name));
    }
    run_cargo(root, package_arguments)?;

    let mut extracted_packages = Vec::with_capacity(PUBLISHABLE_CRATES.len());
    for package in PUBLISHABLE_CRATES {
        let archive = package_target
            .join("package")
            .join(format!("{}-{PACKAGE_VERSION}.crate", package.name));
        let extracted = inspect_and_extract_archive(package, &archive, &extraction_root)?;
        extracted_packages.push((*package, extracted));
    }

    if extracted_packages.len() != PUBLISHABLE_CRATES.len() {
        return Err(format!(
            "expected exactly {} packaged crates, found {}",
            PUBLISHABLE_CRATES.len(),
            extracted_packages.len()
        ));
    }
    check_packaged_consumer(root, temporary.path(), &extracted_packages)
}

fn toml_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .replace('"', "\\\"")
}

fn ensure_clean_tree(root: &Path) -> CheckResult {
    let output = run_capture(
        root,
        "git",
        ["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    if output.stdout.is_empty() {
        return Ok(());
    }

    let changes = String::from_utf8_lossy(&output.stdout);
    let preview = changes.lines().take(20).collect::<Vec<_>>().join("\n");
    let suffix = if changes.lines().count() > 20 {
        "\n... additional changes omitted"
    } else {
        ""
    };
    Err(format!(
        "package and release checks require a clean Git tree; commit or remove all tracked and untracked changes first:\n{preview}{suffix}"
    ))
}

fn inspect_and_extract_archive(
    package: &PublishableCrate,
    archive: &Path,
    extraction_root: &Path,
) -> CheckResult<PathBuf> {
    let archive_size = fs::metadata(archive)
        .map_err(|error| format!("missing package archive {}: {error}", archive.display()))?
        .len();
    if archive_size == 0 || archive_size > MAX_ARCHIVE_BYTES {
        return Err(format!(
            "{} archive size {archive_size} is outside the permitted range 1..={MAX_ARCHIVE_BYTES} bytes",
            package.name
        ));
    }

    let names_output = run_capture(
        extraction_root,
        "tar",
        [OsString::from("-tzf"), archive.as_os_str().to_owned()],
    )?;
    let verbose_output = run_capture(
        extraction_root,
        "tar",
        [OsString::from("-tvzf"), archive.as_os_str().to_owned()],
    )?;
    let names = output_lines(&names_output.stdout, archive)?;
    let verbose = output_lines(&verbose_output.stdout, archive)?;
    validate_archive_entries(package, &names, &verbose)?;

    run_command(
        extraction_root,
        "tar",
        [
            OsString::from("-xzf"),
            archive.as_os_str().to_owned(),
            OsString::from("-C"),
            extraction_root.as_os_str().to_owned(),
        ],
    )?;

    let extracted = extraction_root.join(format!("{}-{PACKAGE_VERSION}", package.name));
    let unpacked_size = inspect_extracted_package(package, &extracted)?;
    eprintln!(
        "verified {}: {} entries, {} compressed bytes, {} unpacked bytes",
        package.name,
        names.len(),
        archive_size,
        unpacked_size
    );
    Ok(extracted)
}

fn output_lines(bytes: &[u8], archive: &Path) -> CheckResult<Vec<String>> {
    let text = std::str::from_utf8(bytes).map_err(|error| {
        format!(
            "{} emitted non-UTF-8 tar output: {error}",
            archive.display()
        )
    })?;
    let lines = text
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return Err(format!("{} contains no archive entries", archive.display()));
    }
    Ok(lines)
}

fn validate_archive_entries(
    package: &PublishableCrate,
    names: &[String],
    verbose: &[String],
) -> CheckResult {
    if names.len() != verbose.len() {
        return Err(format!(
            "{} archive listing was inconsistent ({} names, {} typed entries)",
            package.name,
            names.len(),
            verbose.len()
        ));
    }
    if names.len() > MAX_ARCHIVE_ENTRIES {
        return Err(format!(
            "{} archive has {} entries; limit is {MAX_ARCHIVE_ENTRIES}",
            package.name,
            names.len()
        ));
    }

    let prefix = format!("{}-{PACKAGE_VERSION}/", package.name);
    let mut seen: HashSet<&str> = HashSet::with_capacity(names.len());
    for (name, detail) in names.iter().zip(verbose) {
        let entry_type = detail
            .trim_start()
            .chars()
            .next()
            .ok_or_else(|| format!("{} has an untyped archive entry", package.name))?;
        if !matches!(entry_type, '-' | 'd') {
            return Err(format!(
                "{} archive contains a link or special entry: {name}",
                package.name
            ));
        }
        validate_archive_path(name)?;
        if !seen.insert(name.as_str()) {
            return Err(format!(
                "{} archive contains a duplicate entry: {name}",
                package.name
            ));
        }
        let relative = name.strip_prefix(&prefix).ok_or_else(|| {
            format!(
                "{} archive entry is outside its canonical root {prefix}: {name}",
                package.name
            )
        })?;
        if relative.is_empty() || !allowed_package_path(relative) {
            return Err(format!(
                "{} archive contains an unexpected path: {relative}",
                package.name
            ));
        }
    }

    for required in [
        "Cargo.toml",
        "Cargo.toml.orig",
        "Cargo.lock",
        "README.md",
        "LICENSE",
        "src/lib.rs",
    ] {
        let required = format!("{prefix}{required}");
        if !seen.contains(required.as_str()) {
            return Err(format!(
                "{} archive is missing required file {required}",
                package.name
            ));
        }
    }
    Ok(())
}

fn validate_archive_path(name: &str) -> CheckResult {
    if name.contains('\\') || name.chars().any(char::is_control) {
        return Err(format!("archive contains a non-canonical path: {name:?}"));
    }
    let path = Path::new(name);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("archive contains an unsafe path: {name:?}"));
    }
    Ok(())
}

fn allowed_package_path(relative: &str) -> bool {
    let normalized = relative.trim_end_matches('/');
    matches!(
        normalized,
        ".cargo_vcs_info.json"
            | "Cargo.toml"
            | "Cargo.toml.orig"
            | "Cargo.lock"
            | "README.md"
            | "LICENSE"
    ) || ["src", "tests", "examples"].iter().any(|directory| {
        normalized == *directory || normalized.starts_with(&format!("{directory}/"))
    })
}

fn inspect_extracted_package(package: &PublishableCrate, extracted: &Path) -> CheckResult<u64> {
    let mut file_count = 0;
    let mut total_size = 0;
    inspect_tree(extracted, &mut file_count, &mut total_size)?;
    if file_count > MAX_ARCHIVE_ENTRIES {
        return Err(format!(
            "{} extracted package has {file_count} files; limit is {MAX_ARCHIVE_ENTRIES}",
            package.name
        ));
    }
    if total_size > MAX_UNPACKED_BYTES {
        return Err(format!(
            "{} extracted package is {total_size} bytes; limit is {MAX_UNPACKED_BYTES}",
            package.name
        ));
    }

    let readme = read_bounded_text(&extracted.join("README.md"), 128 * 1024)?;
    let expected_heading = format!("# {}", package.name);
    if !readme.starts_with(&expected_heading)
        || readme.len() < 100
        || readme.contains("TODO")
        || readme.contains("TBD")
    {
        return Err(format!(
            "{} README must start with {expected_heading:?}, contain substantive crate-specific material, and have no placeholders",
            package.name
        ));
    }

    let license = read_bounded_text(&extracted.join("LICENSE"), 64 * 1024)?;
    for marker in [
        "MIT License",
        "Permission is hereby granted, free of charge",
        "THE SOFTWARE IS PROVIDED \"AS IS\"",
    ] {
        if !license.contains(marker) {
            return Err(format!(
                "{} LICENSE is missing required MIT license text: {marker:?}",
                package.name
            ));
        }
    }

    let manifest = read_bounded_text(&extracted.join("Cargo.toml"), 256 * 1024)?;
    for marker in [
        format!("name = \"{}\"", package.name),
        format!("version = \"{PACKAGE_VERSION}\""),
        "license = \"MIT\"".to_owned(),
        "readme = \"README.md\"".to_owned(),
    ] {
        if !manifest.contains(&marker) {
            return Err(format!(
                "{} normalized Cargo.toml is missing {marker:?}",
                package.name
            ));
        }
    }
    Ok(total_size)
}

fn inspect_tree(path: &Path, file_count: &mut usize, total_size: &mut u64) -> CheckResult {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "extracted package contains a symbolic link: {}",
            path.display()
        ));
    }
    if metadata.is_file() {
        if metadata.len() > MAX_ENTRY_BYTES {
            return Err(format!(
                "extracted file {} is {} bytes; per-file limit is {MAX_ENTRY_BYTES}",
                path.display(),
                metadata.len()
            ));
        }
        *file_count += 1;
        *total_size = total_size
            .checked_add(metadata.len())
            .ok_or_else(|| "extracted package size overflowed u64".to_owned())?;
        return Ok(());
    }
    if !metadata.is_dir() {
        return Err(format!(
            "extracted package contains a special file: {}",
            path.display()
        ));
    }
    for entry in
        fs::read_dir(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?
    {
        let entry = entry.map_err(|error| format!("failed to read directory entry: {error}"))?;
        inspect_tree(&entry.path(), file_count, total_size)?;
    }
    Ok(())
}

fn read_bounded_text(path: &Path, maximum: u64) -> CheckResult<String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("missing required file {}: {error}", path.display()))?;
    if metadata.len() == 0 || metadata.len() > maximum {
        return Err(format!(
            "{} size {} is outside the permitted range 1..={maximum} bytes",
            path.display(),
            metadata.len()
        ));
    }
    fs::read_to_string(path)
        .map_err(|error| format!("{} is not valid UTF-8 text: {error}", path.display()))
}

fn check_packaged_consumer(
    root: &Path,
    temporary_root: &Path,
    extracted_packages: &[(PublishableCrate, PathBuf)],
) -> CheckResult {
    let fixture = root.join("tests/packaged-consumer");
    let consumer = temporary_root.join("standalone-consumer");
    copy_tree(&fixture, &consumer)?;

    let manifest_path = consumer.join("Cargo.toml");
    let manifest = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("failed to read {}: {error}", manifest_path.display()))?;
    let patched_manifest = manifest_with_patches(&manifest, extracted_packages)?;
    fs::write(&manifest_path, patched_manifest)
        .map_err(|error| format!("failed to write {}: {error}", manifest_path.display()))?;

    run_cargo(&consumer, ["generate-lockfile"])?;
    let target = temporary_root.join("consumer-target");
    run_consumer_check(&consumer, &target, &[])?;
    run_consumer_check(&consumer, &target, &["--no-default-features"])?;
    for feature in FACADE_FEATURES {
        run_consumer_check(
            &consumer,
            &target,
            &["--no-default-features", "--features", feature],
        )?;
    }
    run_consumer_check(&consumer, &target, &["--all-features"])?;
    run_consumer_check(
        &consumer,
        &target,
        &["--all-features", "--example", "realistic"],
    )?;
    Ok(())
}

fn manifest_with_patches(
    manifest: &str,
    extracted_packages: &[(PublishableCrate, PathBuf)],
) -> CheckResult<String> {
    if manifest.contains("[patch.crates-io]") {
        return Err(
            "packaged-consumer fixture must not contain a pre-generated patch table".into(),
        );
    }
    if extracted_packages.len() != PUBLISHABLE_CRATES.len() {
        return Err("cannot generate patches without all six extracted crates".into());
    }

    let mut patched = manifest.trim_end().to_owned();
    patched.push_str("\n\n[patch.crates-io]\n");
    for (package, path) in extracted_packages {
        let path = toml_path(path);
        patched.push_str(&format!("{} = {{ path = \"{path}\" }}\n", package.name));
    }
    Ok(patched)
}

fn run_consumer_check(consumer: &Path, target: &Path, extra: &[&str]) -> CheckResult {
    let mut arguments = vec![
        OsString::from("check"),
        OsString::from("--locked"),
        OsString::from("--target-dir"),
        target.as_os_str().to_owned(),
    ];
    arguments.extend(extra.iter().map(OsString::from));
    run_cargo(consumer, arguments)
}

fn copy_tree(source: &Path, destination: &Path) -> CheckResult {
    let metadata = fs::symlink_metadata(source)
        .map_err(|error| format!("failed to inspect fixture {}: {error}", source.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "fixture contains a symbolic link: {}",
            source.display()
        ));
    }
    if metadata.is_file() {
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        }
        fs::copy(source, destination).map_err(|error| {
            format!(
                "failed to copy {} to {}: {error}",
                source.display(),
                destination.display()
            )
        })?;
        return Ok(());
    }
    if !metadata.is_dir() {
        return Err(format!(
            "fixture contains a special file: {}",
            source.display()
        ));
    }
    fs::create_dir_all(destination)
        .map_err(|error| format!("failed to create {}: {error}", destination.display()))?;
    for entry in fs::read_dir(source)
        .map_err(|error| format!("failed to read {}: {error}", source.display()))?
    {
        let entry = entry.map_err(|error| format!("failed to read fixture entry: {error}"))?;
        copy_tree(&entry.path(), &destination.join(entry.file_name()))?;
    }
    Ok(())
}

fn run_cargo<I, S>(working_directory: &Path, arguments: I) -> CheckResult
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    run_command(working_directory, "cargo", arguments)
}

fn run_cargo_with_env<I, S, E, K, V>(
    working_directory: &Path,
    arguments: I,
    environment: E,
) -> CheckResult
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
    E: IntoIterator<Item = (K, V)>,
    K: AsRef<OsStr>,
    V: AsRef<OsStr>,
{
    let arguments = arguments
        .into_iter()
        .map(|argument| argument.as_ref().to_owned())
        .collect::<Vec<_>>();
    eprintln!(
        "+ cargo {}",
        arguments
            .iter()
            .map(|argument| argument.to_string_lossy())
            .collect::<Vec<_>>()
            .join(" ")
    );
    let status = Command::new("cargo")
        .args(&arguments)
        .envs(environment)
        .current_dir(working_directory)
        .status()
        .map_err(|error| format!("failed to start cargo: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("cargo exited with status {status}"))
    }
}

fn run_command<I, S>(working_directory: &Path, program: &str, arguments: I) -> CheckResult
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let arguments = arguments
        .into_iter()
        .map(|argument| argument.as_ref().to_owned())
        .collect::<Vec<_>>();
    eprintln!(
        "+ {program} {}",
        arguments
            .iter()
            .map(|argument| argument.to_string_lossy())
            .collect::<Vec<_>>()
            .join(" ")
    );
    let status = Command::new(program)
        .args(&arguments)
        .current_dir(working_directory)
        .status()
        .map_err(|error| format!("failed to start {program}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{program} exited with status {status}"))
    }
}

fn run_capture<I, S>(working_directory: &Path, program: &str, arguments: I) -> CheckResult<Output>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let output = Command::new(program)
        .args(arguments)
        .current_dir(working_directory)
        .output()
        .map_err(|error| format!("failed to start {program}: {error}"))?;
    if output.status.success() {
        Ok(output)
    } else {
        Err(format!(
            "{program} exited with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    fn new(prefix: &str) -> CheckResult<Self> {
        let base = env::temp_dir();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock is before Unix epoch: {error}"))?
            .as_nanos();
        for attempt in 0..100 {
            let path = base.join(format!(
                "{prefix}-{}-{timestamp}-{attempt}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(format!("failed to create {}: {error}", path.display()));
                }
            }
        }
        Err(format!(
            "failed to create a unique temporary directory under {}",
            base.display()
        ))
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path) {
            eprintln!(
                "warning: failed to remove temporary directory {}: {error}",
                self.path.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publishable_set_is_exact_and_unique() {
        assert_eq!(PUBLISHABLE_CRATES.len(), 6);
        let names = PUBLISHABLE_CRATES
            .iter()
            .map(|package| package.name)
            .collect::<HashSet<_>>();
        assert_eq!(names.len(), 6);
        assert!(names.contains("llama-harness"));
        assert!(names.contains("llama-harness-core"));
        assert_eq!(PUBLISHABLE_CRATES[0].name, "llama-harness-core");
        assert_eq!(PUBLISHABLE_CRATES[3].name, "llama-harness-tauri");
        assert_eq!(PUBLISHABLE_CRATES[4].name, "llama-harness-evals");
        assert_eq!(PUBLISHABLE_CRATES[5].name, "llama-harness");
    }

    #[test]
    fn archive_paths_reject_traversal_and_non_canonical_separators() {
        assert!(validate_archive_path("llama-harness-0.1.0/src/lib.rs").is_ok());
        assert!(validate_archive_path("../outside").is_err());
        assert!(validate_archive_path("llama-harness-0.1.0/../outside").is_err());
        assert!(validate_archive_path("llama-harness-0.1.0\\src\\lib.rs").is_err());
    }

    #[test]
    fn package_paths_are_allowlisted() {
        assert!(allowed_package_path("Cargo.toml"));
        assert!(allowed_package_path("Cargo.lock"));
        assert!(allowed_package_path("src/lib.rs"));
        assert!(allowed_package_path("tests/fixtures/suite.yaml"));
        assert!(!allowed_package_path(".env"));
        assert!(!allowed_package_path("target/debug/output"));
        assert!(!allowed_package_path("build.rs"));
    }

    #[test]
    fn generated_manifest_patches_all_extracted_crates() {
        let packages = PUBLISHABLE_CRATES
            .iter()
            .map(|package| (*package, PathBuf::from(format!("/tmp/{}", package.name))))
            .collect::<Vec<_>>();
        let manifest = manifest_with_patches("[package]\nname = \"consumer\"\n", &packages)
            .expect("patch generation should succeed");
        assert_eq!(manifest.matches(" = { path = ").count(), 6);
        assert!(manifest.contains("[patch.crates-io]"));
        assert!(manifest.contains("llama-harness = { path = \"/tmp/llama-harness\" }"));
    }
}
