use std::{env, net::SocketAddr, path::PathBuf, time::Duration};

use anyhow::{bail, Context};

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
    pub cors_permissive: bool,
    pub cors_allowed_origins: Vec<String>,
    pub service_api_key: String,
    pub session_ttl: Duration,
    pub text_worker_url: String,
    pub ltx_worker_url: String,
    pub text_model_registry: Vec<String>,
    pub text_max_waiting: usize,
    pub ltx_max_waiting: usize,
    pub local_max_heavy_jobs: usize,
    pub text_heavy_jobs: usize,
    pub video_heavy_jobs: usize,
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
        let profile = env_or("GATEWAY_PROFILE", "local_rtx_5090");
        let cors_allowed_origins = env_or("CORS_ALLOWED_ORIGINS", "")
            .split(',')
            .map(str::trim)
            .filter(|origin| !origin.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        let cors_permissive = env_or(
            "CORS_PERMISSIVE",
            if profile.starts_with("local") {
                "true"
            } else {
                "false"
            },
        )
        .parse()?;
        let local_heavy_jobs: usize = env_or("LTX_LOCAL_MAX_HEAVY_JOBS", "1").parse()?;
        let turso_db_url = env_or("TURSO_DB_URL", "file:storage/gateway.db");
        let turso_auth_token = env_or("TURSO_AUTH_TOKEN", "");
        validate_database_env(&turso_db_url, &turso_auth_token)?;
        let default_log_dir = env::var("LOG_DIR").unwrap_or_else(|_| "runtime/logs".to_owned());
        Ok(Self {
            bind_addr,
            profile,
            runtime_backend: env_or("RUNTIME_BACKEND", "native"),
            compose_project: env_or("COMPOSE_PROJECT", "py_ai_apis_gpt"),
            turso_db_url,
            turso_auth_token,
            admin_api_key: env_or("ADMIN_API_KEY", ""),
            admin_emails,
            cors_permissive,
            cors_allowed_origins,
            service_api_key: env_or("SERVICE_API_KEY", ""),
            session_ttl: Duration::from_secs(env_or("SESSION_TTL_SECONDS", "86400").parse()?),
            text_worker_url: env_or("TEXT_WORKER_URL", "http://127.0.0.1:8101"),
            ltx_worker_url: env_or("LTX_WORKER_URL", "http://127.0.0.1:8102"),
            text_model_registry: registry,
            text_max_waiting: env_or("TEXT_MAX_WAITING", "512").parse()?,
            ltx_max_waiting: env_or("LTX_MAX_WAITING", "512").parse()?,
            local_max_heavy_jobs: local_heavy_jobs,
            text_heavy_jobs: env_or("TEXT_HEAVY_JOBS", &local_heavy_jobs.to_string()).parse()?,
            video_heavy_jobs: env_or("VIDEO_HEAVY_JOBS", &local_heavy_jobs.to_string()).parse()?,
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

fn validate_database_env(url: &str, token: &str) -> anyhow::Result<()> {
    if url.contains("replace-") {
        bail!(
            "TURSO_DB_URL is still a placeholder. Set a real Turso libsql:// URL, or use TURSO_DB_URL=file:storage/gateway.db for a local smoke run."
        );
    }
    if url.starts_with("libsql://") && (token.is_empty() || token.contains("replace-")) {
        bail!(
            "TURSO_AUTH_TOKEN is required when TURSO_DB_URL uses libsql://. For local testing, set TURSO_DB_URL=file:storage/gateway.db."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_database_env;

    #[test]
    fn rejects_placeholder_remote_database_url() {
        let err = validate_database_env("libsql://replace-with-database-url", "").unwrap_err();
        assert!(err.to_string().contains("placeholder"));
    }

    #[test]
    fn remote_database_requires_token() {
        let err = validate_database_env("libsql://db.example.turso.io", "").unwrap_err();
        assert!(err.to_string().contains("TURSO_AUTH_TOKEN"));
    }

    #[test]
    fn local_file_database_does_not_require_token() {
        validate_database_env("file:storage/gateway.db", "").unwrap();
    }
}
