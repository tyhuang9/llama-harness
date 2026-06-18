mod app_policy;
mod config;
mod litellm;
mod litellm_runtime;
mod ollama;
mod providers;
mod routes;
mod runs;
mod secrets;

use crate::{
    config::AppConfig, litellm_runtime::LiteLlmRuntimeManager, ollama::OllamaClient,
    providers::ProviderRegistry, routes::AppState,
};
use anyhow::Context;
use axum::Router;
use chrono::Utc;
use std::{collections::VecDeque, env, net::SocketAddr, sync::Arc};
use tokio::sync::RwLock;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "llama_harness_server=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config_path = config::default_config_path();
    let runs_path = config::default_runs_path();
    let audit_path = config::default_audit_path();
    let config: AppConfig = config::load_config(&config_path).await?;
    let catalog_dir = config::default_catalog_dir(&config_path);
    let catalog = app_policy::load_domain_catalog(&catalog_dir, &config).await?;
    let run_history = runs::load_runs(&runs_path, 100)
        .await
        .unwrap_or_else(|err| {
            tracing::warn!(error = %err, "failed to load run history");
            VecDeque::new()
        });
    let audit_history = runs::load_audit(&audit_path, 100)
        .await
        .unwrap_or_else(|err| {
            tracing::warn!(error = %err, "failed to load audit history");
            VecDeque::new()
        });

    let state = AppState {
        config: Arc::new(RwLock::new(config)),
        config_path,
        catalog: Arc::new(RwLock::new(catalog)),
        catalog_dir,
        runs_path,
        runs: Arc::new(RwLock::new(run_history)),
        audit_path,
        audit: Arc::new(RwLock::new(audit_history)),
        providers: ProviderRegistry::new(OllamaClient::new()),
        litellm_runtime: LiteLlmRuntimeManager::new(),
        started_at: Utc::now(),
    };

    let app: Router = routes::router(state)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = env::var("LLAMA_HARNESS_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8787".to_string())
        .parse()
        .context("LLAMA_HARNESS_ADDR must be a socket address")?;

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "llama-harness server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
