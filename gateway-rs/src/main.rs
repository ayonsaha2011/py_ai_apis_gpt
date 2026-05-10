mod auth;
mod config;
mod db;
mod error;
mod gpu;
mod models;
mod queue;
mod routes;
mod services;
mod validation;

use std::sync::Arc;

use axum::Router;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::{
    config::Config,
    db::Db,
    queue::Admission,
    routes::router,
    services::{DockerBackend, NativeBackend, RuntimeBackend},
};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub db: Db,
    pub http: reqwest::Client,
    pub admission: Admission,
    pub runtime: Arc<dyn RuntimeBackend>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Arc::new(Config::from_env()?);
    let db = Db::connect(&config).await?;
    db.migrate().await?;
    db.ensure_model_registry(&config).await?;

    let runtime: Arc<dyn RuntimeBackend> = match config.runtime_backend.as_str() {
        "docker" => Arc::new(DockerBackend::new(config.compose_project.clone())),
        "native" => Arc::new(NativeBackend::new((*config).clone())),
        _ => Arc::new(NativeBackend::new((*config).clone())),
    };

    let state = AppState {
        config: config.clone(),
        db,
        http: reqwest::Client::builder()
            .pool_max_idle_per_host(64)
            .build()?,
        admission: Admission::new(
            config.local_max_heavy_jobs,
            config.text_max_waiting,
            config.ltx_max_waiting,
        ),
        runtime,
    };

    state.db.recover_interrupted_video_jobs().await?;
    routes::spawn_video_dispatcher(state.clone());

    let app: Router = router(state)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(config.bind_addr).await?;
    tracing::info!("gateway listening on {}", config.bind_addr);
    axum::serve(listener, app).await?;
    Ok(())
}
