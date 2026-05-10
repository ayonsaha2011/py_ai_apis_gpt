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

use axum::{
    http::{header, HeaderName, HeaderValue, Method},
    Router,
};
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
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
            config.text_heavy_jobs,
            config.video_heavy_jobs,
            config.text_max_waiting,
            config.ltx_max_waiting,
        ),
        runtime,
    };

    state.db.recover_interrupted_video_jobs().await?;
    routes::spawn_video_dispatcher(state.clone());

    let app: Router = router(state)
        .layer(cors_layer(&config))
        .layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(config.bind_addr).await?;
    tracing::info!("gateway listening on {}", config.bind_addr);
    axum::serve(listener, app).await?;
    Ok(())
}

fn cors_layer(config: &Config) -> CorsLayer {
    if config.cors_permissive {
        return CorsLayer::permissive();
    }

    let layer = CorsLayer::new()
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers([
            header::AUTHORIZATION,
            header::CONTENT_TYPE,
            HeaderName::from_static("x-admin-key"),
            HeaderName::from_static("x-service-key"),
        ]);

    if config
        .cors_allowed_origins
        .iter()
        .any(|origin| origin == "*")
    {
        return layer.allow_origin(Any);
    }

    let origins = config
        .cors_allowed_origins
        .iter()
        .filter_map(|origin| origin.parse::<HeaderValue>().ok())
        .collect::<Vec<_>>();

    layer.allow_origin(origins)
}

#[cfg(test)]
mod tests {
    use std::{net::SocketAddr, path::PathBuf, time::Duration};

    use crate::config::Config;

    fn config_with_cors(cors_permissive: bool, cors_allowed_origins: Vec<String>) -> Config {
        Config {
            bind_addr: "127.0.0.1:0".parse::<SocketAddr>().unwrap(),
            profile: "cloud_h200".into(),
            runtime_backend: "native".into(),
            compose_project: "test".into(),
            turso_db_url: "file:storage/gateway.db".into(),
            turso_auth_token: String::new(),
            admin_api_key: "admin-key".into(),
            admin_emails: vec![],
            cors_permissive,
            cors_allowed_origins,
            service_api_key: "service-key".into(),
            session_ttl: Duration::from_secs(3600),
            text_worker_url: "http://127.0.0.1:8101".into(),
            ltx_worker_url: "http://127.0.0.1:8102".into(),
            text_model_registry: vec!["test-model".into()],
            text_max_waiting: 8,
            ltx_max_waiting: 8,
            local_max_heavy_jobs: 1,
            text_heavy_jobs: 1,
            video_heavy_jobs: 1,
            qdrant_url: "http://127.0.0.1:6333".into(),
            r2_public_base_url: String::new(),
            storage_dir: PathBuf::from("storage"),
            frontend_dir: PathBuf::from("frontend/dist"),
            log_dir: PathBuf::from("runtime/logs"),
            native_text_command: String::new(),
            native_ltx_command: String::new(),
        }
    }

    #[test]
    fn cors_wildcard_origin_does_not_panic() {
        let config = config_with_cors(false, vec!["*".into()]);
        let _ = super::cors_layer(&config);
    }
}
