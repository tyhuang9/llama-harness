fn main() {
    ensure_optional_litellm_resource_exists();
    tauri_build::build()
}

fn ensure_optional_litellm_resource_exists() {
    let manifest_dir = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"),
    );
    let runtime_dir = manifest_dir.join("../../bundled/litellm-runtime");
    let has_files = runtime_dir
        .read_dir()
        .ok()
        .and_then(|mut entries| entries.next())
        .is_some();

    if has_files {
        return;
    }

    if let Err(error) = std::fs::create_dir_all(&runtime_dir) {
        panic!(
            "failed to create optional LiteLLM resource directory {}: {error}",
            runtime_dir.display()
        );
    }

    let placeholder = runtime_dir.join(".tauri-resource-placeholder");
    if let Err(error) = std::fs::write(
        &placeholder,
        "Generated placeholder for dev-only Tauri checks.\n",
    ) {
        panic!(
            "failed to create optional LiteLLM resource placeholder {}: {error}",
            placeholder.display()
        );
    }
}
