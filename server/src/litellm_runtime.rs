use crate::{
    config::{AppConfig, REDACTED_SECRET},
    secrets::LITELLM_MASTER_KEY_ENV,
};
use anyhow::{anyhow, Context, Result};
use reqwest::{Client, RequestBuilder, StatusCode};
use serde_json::Value;
use std::{
    collections::HashMap,
    env,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    net::TcpStream,
    process::{Child, Command},
    sync::RwLock,
    time::sleep,
};

const LITELLM_PYTHON_ENV: &str = "LLAMA_HARNESS_LITELLM_PYTHON";
const LITELLM_RUNTIME_DIR_ENV: &str = "LLAMA_HARNESS_LITELLM_RUNTIME_DIR";
const LITELLM_COMMAND_ENV: &str = "LLAMA_HARNESS_LITELLM_COMMAND";
const MIN_STARTUP_TIMEOUT_MS: u64 = 30_000;
const MAX_STARTUP_TIMEOUT_MS: u64 = 180_000;
const STARTUP_POLL_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Clone)]
pub struct LiteLlmRuntimeManager {
    process: Arc<RwLock<Option<ManagedLiteLlmProcess>>>,
    http: Client,
}

pub struct ManagedLiteLlmProcess {
    child: Child,
}

pub struct LiteLlmStart {
    pub pid: u32,
    pub command: String,
}

impl LiteLlmRuntimeManager {
    pub fn new() -> Self {
        Self {
            process: Arc::new(RwLock::new(None)),
            http: Client::new(),
        }
    }

    pub async fn ensure_after_apply(
        &self,
        config: &AppConfig,
        output_path: &Path,
        app_env: &HashMap<String, String>,
    ) -> Result<(bool, Option<String>)> {
        if self.managed_is_running().await {
            self.stop_managed().await?;
            let _start = self.start_managed(config, output_path, app_env).await?;
            let ready = self.wait_for_ready(config).await;
            return Ok((
                ready,
                (!ready).then(|| startup_timeout_message(config, "LiteLLM was restarted")),
            ));
        }

        if self.port_open(&config.litellm.base_url).await {
            let ready = self.readiness_healthy(config).await;
            return Ok((
                ready,
                Some(if ready {
                    "LiteLLM is already running outside Llama Harness. Saved keys and config are ready, but restart that proxy if new providers do not appear.".to_string()
                } else {
                    "A process is already using the LiteLLM port, and it did not respond as a ready LiteLLM proxy. Llama Harness did not stop it.".to_string()
                }),
            ));
        }

        let _start = self.start_managed(config, output_path, app_env).await?;
        let ready = self.wait_for_ready(config).await;
        Ok((
            ready,
            (!ready).then(|| startup_timeout_message(config, "LiteLLM started")),
        ))
    }

    pub async fn ensure_started(
        &self,
        config: &AppConfig,
        output_path: &Path,
        app_env: &HashMap<String, String>,
    ) -> Result<Option<LiteLlmStart>> {
        if self.readiness_healthy(config).await {
            return Ok(None);
        }

        if self.managed_is_running().await {
            self.stop_managed().await?;
        } else if self.port_open(&config.litellm.base_url).await {
            return Err(anyhow!(
                "a process is already using the LiteLLM port; Llama Harness will not stop an external process"
            ));
        }

        let start = self.start_managed(config, output_path, app_env).await?;
        if !self.wait_for_ready(config).await {
            if !self.managed_is_running().await {
                return Err(anyhow!(
                    "LiteLLM exited before it became ready. Check the server logs for LiteLLM stderr."
                ));
            }
            return Err(anyhow!(startup_timeout_message(config, "LiteLLM started")));
        }

        Ok(Some(start))
    }

    pub async fn managed_is_running(&self) -> bool {
        let mut process = self.process.write().await;
        let should_clear = match process.as_mut() {
            Some(managed) => match managed.child.try_wait() {
                Ok(Some(status)) => {
                    tracing::info!(%status, "managed LiteLLM process exited");
                    true
                }
                Ok(None) => false,
                Err(err) => {
                    tracing::warn!(error = %err, "failed to inspect managed LiteLLM process");
                    true
                }
            },
            None => return false,
        };

        if should_clear {
            *process = None;
            false
        } else {
            true
        }
    }

    pub async fn stop_managed(&self) -> Result<()> {
        let mut managed = {
            let mut process = self.process.write().await;
            process.take()
        };

        if let Some(mut process) = managed.take() {
            process
                .child
                .kill()
                .await
                .context("failed to stop managed LiteLLM process")?;
            let _ = process.child.wait().await;
        }

        Ok(())
    }

    pub async fn start_managed(
        &self,
        config: &AppConfig,
        output_path: &Path,
        app_env: &HashMap<String, String>,
    ) -> Result<LiteLlmStart> {
        let command = resolve_litellm_command()?;
        let (host, port) = litellm_host_port(&config.litellm.base_url);
        let command_summary =
            litellm_start_command_summary_with_command(&command, config, output_path);
        let mut child_command = Command::new(&command.executable);
        child_command
            .args(&command.prefix_args)
            .arg("--config")
            .arg(output_path)
            .arg("--host")
            .arg(&host)
            .arg("--port")
            .arg(&port)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_remove("DEBUG")
            .envs(litellm_child_env(config, app_env));
        let mut child = child_command.spawn().with_context(|| {
            format!(
                "failed to start LiteLLM using {}",
                command.executable.display()
            )
        })?;

        pipe_child_output("stdout", child.stdout.take());
        pipe_child_output("stderr", child.stderr.take());

        let pid = child.id().unwrap_or(0);
        tracing::info!(pid, command = %command_summary, "managed LiteLLM process started");

        let mut process = self.process.write().await;
        *process = Some(ManagedLiteLlmProcess { child });

        Ok(LiteLlmStart {
            pid,
            command: command_summary,
        })
    }

    pub async fn wait_for_ready(&self, config: &AppConfig) -> bool {
        let started = std::time::Instant::now();
        let timeout = litellm_startup_timeout(config);
        while started.elapsed() < timeout {
            if self.readiness_healthy(config).await {
                return true;
            }
            if !self.managed_is_running().await {
                return false;
            }
            sleep(STARTUP_POLL_INTERVAL).await;
        }
        false
    }

    pub async fn readiness_healthy(&self, config: &AppConfig) -> bool {
        if !config.litellm.enabled {
            return false;
        }

        for path in ["/health/readiness", "/v1/models", "/health"] {
            if self.health_endpoint(config, path).await {
                return true;
            }
        }
        false
    }

    async fn health_endpoint(&self, config: &AppConfig, path: &str) -> bool {
        self.with_auth(config, self.http.get(url(&config.litellm.base_url, path)))
            .send()
            .await
            .map(|response| {
                let status = response.status();
                let ready = status.is_success();
                if !ready {
                    tracing::debug!(path, %status, "LiteLLM health endpoint is not ready");
                }
                ready
            })
            .unwrap_or(false)
    }

    pub async fn port_open(&self, base_url: &str) -> bool {
        let (host, port) = litellm_host_port(base_url);
        let Ok(port) = port.parse::<u16>() else {
            return false;
        };
        TcpStream::connect((host.as_str(), port)).await.is_ok()
    }

    pub async fn forward_chat_completions(&self, config: &AppConfig, body: Value) -> Result<Value> {
        let mut last_not_found = None;
        for path in ["/chat/completions", "/v1/chat/completions"] {
            let response = self
                .with_auth(config, self.http.post(url(&config.litellm.base_url, path)))
                .json(&body)
                .send()
                .await
                .map_err(|err| {
                    if err.is_connect() || err.is_timeout() {
                        anyhow!(
                            "LiteLLM proxy is not reachable at {}.",
                            config.litellm.base_url.trim_end_matches('/')
                        )
                    } else {
                        anyhow!("failed to call LiteLLM proxy: {err}")
                    }
                })?;

            let status = response.status();
            let text = response
                .text()
                .await
                .context("failed to read LiteLLM chat completion response")?;
            let value =
                serde_json::from_str::<Value>(&text).unwrap_or_else(|_| Value::String(text));
            if status.is_success() {
                return Ok(value);
            }
            if status == StatusCode::NOT_FOUND {
                last_not_found = Some(anyhow!("LiteLLM returned {status}: {value}"));
                continue;
            }
            return Err(anyhow!("LiteLLM returned {status}: {value}"));
        }

        Err(last_not_found
            .unwrap_or_else(|| anyhow!("LiteLLM chat completions endpoint was not found.")))
    }

    fn with_auth(&self, config: &AppConfig, request: RequestBuilder) -> RequestBuilder {
        match config.litellm.api_key.as_deref().map(str::trim) {
            Some(api_key) if !api_key.is_empty() && api_key != REDACTED_SECRET => {
                request.bearer_auth(api_key)
            }
            _ => request,
        }
    }
}

impl Default for LiteLlmRuntimeManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ManagedLiteLlmProcess {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

pub fn litellm_start_command_summary(config: &AppConfig, config_path: &Path) -> String {
    match resolve_litellm_command() {
        Ok(command) => litellm_start_command_summary_with_command(&command, config, config_path),
        Err(err) => {
            let (host, port) = litellm_host_port(&config.litellm.base_url);
            format!(
                "<unresolved LiteLLM command: {err}> --config {} --host {} --port {}",
                config_path.display(),
                host,
                port
            )
        }
    }
}

struct LiteLlmCommand {
    executable: PathBuf,
    prefix_args: Vec<String>,
}

pub(crate) fn litellm_child_env(
    config: &AppConfig,
    app_env: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut child_env = app_env.clone();
    if let Some(api_key) = config
        .litellm
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != REDACTED_SECRET)
    {
        child_env.insert(LITELLM_MASTER_KEY_ENV.to_string(), api_key.to_string());
    }
    child_env
}

pub(crate) fn litellm_host_port(base_url: &str) -> (String, String) {
    let trimmed = base_url.trim().trim_end_matches('/');
    let without_scheme = trimmed
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(trimmed);
    let authority = without_scheme
        .split('/')
        .next()
        .unwrap_or(without_scheme)
        .rsplit('@')
        .next()
        .unwrap_or(without_scheme);

    if let Some(rest) = authority.strip_prefix('[') {
        if let Some((host, after_host)) = rest.split_once(']') {
            let port = after_host.strip_prefix(':').unwrap_or("4000");
            return (host.to_string(), port.to_string());
        }
    }

    let (host, port) = authority
        .rsplit_once(':')
        .map(|(host, port)| (host, port))
        .unwrap_or((authority, "4000"));
    let host = if host.is_empty() { "127.0.0.1" } else { host };
    let port = if port.is_empty() { "4000" } else { port };
    (host.to_string(), port.to_string())
}

fn resolve_litellm_python() -> Result<PathBuf> {
    if let Some(path) = env_path(LITELLM_PYTHON_ENV) {
        return Ok(path);
    }

    if let Some(runtime_dir) = env_path(LITELLM_RUNTIME_DIR_ENV) {
        return python_in_runtime(&runtime_dir);
    }

    for root in repo_root_candidates() {
        let candidate = root.join(".venv-litellm");
        if candidate.exists() {
            return python_in_runtime(&candidate);
        }
    }

    Err(anyhow!(
        "LiteLLM Python was not found. Run scripts/setup-litellm-dev.sh or set {LITELLM_PYTHON_ENV}."
    ))
}

fn resolve_litellm_command() -> Result<LiteLlmCommand> {
    if let Some(path) = env_path(LITELLM_COMMAND_ENV) {
        return Ok(LiteLlmCommand {
            executable: path,
            prefix_args: Vec::new(),
        });
    }

    if let Some(runtime_dir) = env_path(LITELLM_RUNTIME_DIR_ENV) {
        return command_in_runtime(&runtime_dir);
    }

    for root in repo_root_candidates() {
        let candidate = root.join(".venv-litellm");
        if candidate.exists() {
            return command_in_runtime(&candidate);
        }
    }

    let python = resolve_litellm_python()?;
    Ok(python_litellm_shim(python))
}

fn command_in_runtime(runtime_dir: &Path) -> Result<LiteLlmCommand> {
    let script = runtime_dir.join(platform_litellm_relative_path());
    if script.exists() {
        return Ok(LiteLlmCommand {
            executable: script,
            prefix_args: Vec::new(),
        });
    }

    let python = python_in_runtime(runtime_dir)?;
    Ok(python_litellm_shim(python))
}

fn python_litellm_shim(python: PathBuf) -> LiteLlmCommand {
    LiteLlmCommand {
        executable: python,
        prefix_args: vec![
            "-c".to_string(),
            "from litellm import run_server; raise SystemExit(run_server())".to_string(),
        ],
    }
}

fn env_path(key: &str) -> Option<PathBuf> {
    env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn python_in_runtime(runtime_dir: &Path) -> Result<PathBuf> {
    let python = runtime_dir.join(platform_python_relative_path());
    if python.exists() {
        Ok(python)
    } else {
        Err(anyhow!(
            "LiteLLM runtime Python does not exist at {}",
            python.display()
        ))
    }
}

fn platform_python_relative_path() -> PathBuf {
    if cfg!(windows) {
        PathBuf::from("Scripts").join("python.exe")
    } else {
        PathBuf::from("bin").join("python")
    }
}

fn platform_litellm_relative_path() -> PathBuf {
    if cfg!(windows) {
        PathBuf::from("Scripts").join("litellm.exe")
    } else {
        PathBuf::from("bin").join("litellm")
    }
}

fn repo_root_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(current_dir) = env::current_dir() {
        candidates.push(current_dir.clone());
        if let Some(parent) = current_dir.parent() {
            candidates.push(parent.to_path_buf());
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(repo_root) = manifest_dir.parent() {
        candidates.push(repo_root.to_path_buf());
    }

    candidates.sort();
    candidates.dedup();
    candidates
}

fn litellm_start_command_summary_with_command(
    command: &LiteLlmCommand,
    config: &AppConfig,
    config_path: &Path,
) -> String {
    let (host, port) = litellm_host_port(&config.litellm.base_url);
    let prefix_args = if command.prefix_args.is_empty() {
        String::new()
    } else {
        format!("{} ", command.prefix_args.join(" "))
    };
    format!(
        "{} {}--config {} --host {} --port {}",
        command.executable.display(),
        prefix_args,
        config_path.display(),
        host,
        port
    )
}

fn litellm_startup_timeout(config: &AppConfig) -> Duration {
    Duration::from_millis(
        config
            .litellm
            .timeout_ms
            .clamp(MIN_STARTUP_TIMEOUT_MS, MAX_STARTUP_TIMEOUT_MS),
    )
}

fn startup_timeout_message(config: &AppConfig, prefix: &str) -> String {
    let timeout = litellm_startup_timeout(config).as_secs();
    format!(
        "{prefix}, but it did not become ready at {} within {timeout}s. Check the server logs for LiteLLM stdout/stderr.",
        url(&config.litellm.base_url, "/health/readiness")
    )
}

fn pipe_child_output(
    label: &'static str,
    pipe: Option<impl tokio::io::AsyncRead + Unpin + Send + 'static>,
) {
    let Some(pipe) = pipe else {
        return;
    };

    tokio::spawn(async move {
        let mut lines = BufReader::new(pipe).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    tracing::info!(target: "litellm_runtime", stream = label, "{}", line);
                }
                Ok(None) => break,
                Err(err) => {
                    tracing::warn!(error = %err, stream = label, "failed to read LiteLLM output");
                    break;
                }
            }
        }
    });
}

fn url(base_url: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}
