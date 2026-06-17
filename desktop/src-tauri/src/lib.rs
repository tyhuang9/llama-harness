use tauri::{path::BaseDirectory, Manager};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            for resource_path in ["litellm-runtime", "bundled/litellm-runtime"] {
                if let Ok(runtime_dir) = app.path().resolve(resource_path, BaseDirectory::Resource)
                {
                    if runtime_dir.exists() {
                        std::env::set_var("LLAMA_HARNESS_LITELLM_RUNTIME_DIR", runtime_dir);
                        break;
                    }
                }
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running llama-harness desktop application");
}
