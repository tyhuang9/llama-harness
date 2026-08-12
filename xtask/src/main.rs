use std::{
    env,
    process::{Command, ExitCode},
};

const PUBLISHABLE_CRATES: &[&str] = &[
    "llama-harness-core",
    "llama-harness-ollama",
    "llama-harness-observability",
    "llama-harness-evals",
    "llama-harness-protocol",
    "llama-harness",
];

fn main() -> ExitCode {
    match env::args().nth(1).as_deref() {
        Some("release-check") => release_check(),
        Some("package-list") => package_list(),
        Some("protocol-check") => protocol_check(),
        _ => {
            eprintln!("usage: cargo run -p xtask -- <release-check|package-list|protocol-check>");
            ExitCode::FAILURE
        }
    }
}

fn protocol_check() -> ExitCode {
    if run_cargo(&["test", "--package", "llama-harness-protocol"]) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn release_check() -> ExitCode {
    for arguments in [
        vec!["fmt", "--check", "--all"],
        vec![
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ],
        vec!["test", "--workspace", "--all-features"],
        vec!["doc", "--workspace", "--all-features", "--no-deps"],
    ] {
        if !run_cargo(&arguments) {
            return ExitCode::FAILURE;
        }
    }
    package_list()
}

fn package_list() -> ExitCode {
    for package in PUBLISHABLE_CRATES {
        if !run_cargo(&["package", "--list", "--allow-dirty", "--package", package]) {
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

fn run_cargo(arguments: &[&str]) -> bool {
    eprintln!("+ cargo {}", arguments.join(" "));
    Command::new("cargo")
        .args(arguments)
        .status()
        .is_ok_and(|status| status.success())
}
