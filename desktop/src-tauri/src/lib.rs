use std::env;

use tauri::{path::BaseDirectory, Manager};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    normalize_wslg_window_scaling();

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

fn normalize_wslg_window_scaling() {
    if !is_wslg_session() || env_flag_disabled("LLAMA_HARNESS_WSLG_SCALING_FIX") {
        return;
    }

    // WSLg can expose Windows fractional display scaling through GTK while the
    // host still presents the Linux window as physical pixels. That makes a
    // maximized Tauri window stop at the scaled work area instead of the screen.
    set_env_default("GDK_SCALE", "1");
    set_env_default("GDK_DPI_SCALE", "1");
}

fn is_wslg_session() -> bool {
    env::var_os("WSL2_GUI_APPS_ENABLED").is_some()
        && (env::var_os("WAYLAND_DISPLAY").is_some() || env::var_os("DISPLAY").is_some())
}

fn env_flag_disabled(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(false)
}

fn set_env_default(name: &str, value: &str) {
    if env::var_os(name).is_none() {
        env::set_var(name, value);
    }
}
