use std::{env, net::SocketAddr, path::PathBuf, time::Duration};

use anyhow::Context;

#[derive(Clone, Debug)]
pub struct Config {
    pub bind_addr: SocketAddr,
    pub profile: String,
    pub runtime_backend: String,
    pub compose_project: String,
    pub turso_db_url: String,
    pub turso_auth_token: String,
    pub admin_api_key: String,
    pub admin_emails: Vec<String>,
    pub service_api_key: String,
    pub session_ttl: Duration,
    pub text_worker_url: String,
    pub ltx_worker_url: String,
    pub text_model_registry: Vec<String>,
    pub text_max_waiting: usize,
    pub ltx_max_waiting: usize,
    pub local_max_heavy_jobs: usize,
    pub qdrant_url: String,
    pub r2_public_base_url: String,
    pub storage_dir: PathBuf,
    pub frontend_dir: PathBuf,
    pub log_dir: PathBuf,
    pub native_text_command: String,
    pub native_ltx_command: String,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let bind_addr = env_or("GATEWAY_BIND", "127.0.0.1:8080")
            .parse()
            .context("GATEWAY_BIND must be host:port")?;
        let registry = env_or(
            "TEXT_MODEL_REGISTRY",
            &env_or(
                "TEXT_MODEL_ID",
                "google/gemma-3-12b-it-qat-q4_0-unquantized",
            ),
        )
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
        let admin_emails = env_or("ADMIN_EMAILS", "")
            .split(',')
            .map(|email| email.trim().to_ascii_lowercase())
            .filter(|email| !email.is_empty())
            .collect::<Vec<_>>();
        let default_log_dir = env::var("LOG_DIR").unwrap_or_else(|_| "runtime/logs".to_owned());
        Ok(Self {
            bind_addr,
            profile: env_or("GATEWAY_PROFILE", "local_rtx_5090"),
            runtime_backend: env_or("RUNTIME_BACKEND", "native"),
            compose_project: env_or("COMPOSE_PROJECT", "py_ai_apis_gpt"),
            turso_db_url: env_or("TURSO_DB_URL", "file:storage/gateway.db"),
            turso_auth_token: env_or("TURSO_AUTH_TOKEN", ""),
            admin_api_key: env_or("ADMIN_API_KEY", ""),
            admin_emails,
            service_api_key: env_or("SERVICE_API_KEY", ""),
            session_ttl: Duration::from_secs(env_or("SESSION_TTL_SECONDS", "86400").parse()?),
            text_worker_url: env_or("TEXT_WORKER_URL", "http://127.0.0.1:8101"),
            ltx_worker_url: env_or("LTX_WORKER_URL", "http://127.0.0.1:8102"),
            text_model_registry: registry,
            text_max_waiting: env_or("TEXT_MAX_WAITING", "512").parse()?,
            ltx_max_waiting: env_or("LTX_MAX_WAITING", "512").parse()?,
            local_max_heavy_jobs: env_or("LTX_LOCAL_MAX_HEAVY_JOBS", "1").parse()?,
            qdrant_url: env_or("QDRANT_URL", "http://127.0.0.1:6333"),
            r2_public_base_url: env_or("R2_PUBLIC_BASE_URL", ""),
            storage_dir: PathBuf::from(env_or("LOCAL_STORAGE_DIR", "storage")),
            frontend_dir: PathBuf::from(env_or("FRONTEND_DIR", "frontend/dist")),
            log_dir: PathBuf::from(env_or("SERVICE_LOG_DIR", &default_log_dir)),
            native_text_command: env_or(
                "NATIVE_TEXT_COMMAND",
                "uv run uvicorn app.main:app --host 127.0.0.1 --port 8101",
            ),
            native_ltx_command: env_or(
                "NATIVE_LTX_COMMAND",
                "uv run uvicorn app.main:app --host 127.0.0.1 --port 8102",
            ),
        })
    }
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_owned())
}
