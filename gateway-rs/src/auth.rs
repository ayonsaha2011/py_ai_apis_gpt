use argon2::{
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use axum::http::HeaderMap;
use rand_core::{OsRng, RngCore};
use sha2::{Digest, Sha256};

use crate::{
    error::{AppError, Result},
    models::User,
    AppState,
};

pub fn hash_password(password: &str) -> Result<String> {
    if password.len() < 12 {
        return Err(AppError::BadRequest(
            "password must be at least 12 characters".into(),
        ));
    }
    let salt = SaltString::generate(&mut OsRng);
    Ok(Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .to_string())
}

pub fn verify_password(password: &str, hash: &str) -> Result<bool> {
    let parsed = PasswordHash::new(hash).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    Ok(Argon2::default()
        .verify_password(password.as_bytes(), &parsed)
        .is_ok())
}

pub fn new_token() -> String {
    let mut bytes = [0_u8; 32];
    OsRng.fill_bytes(&mut bytes);
    format!("pat_{}", hex::encode(bytes))
}

pub fn hash_token(token: &str) -> String {
    hex::encode(Sha256::digest(token.as_bytes()))
}

pub async fn require_user(headers: &HeaderMap, state: &AppState) -> Result<User> {
    let token = bearer(headers).ok_or(AppError::Unauthorized)?;
    let token_hash = hash_token(token);
    state
        .db
        .user_by_session(&token_hash)
        .await?
        .ok_or(AppError::Unauthorized)
}

pub async fn require_admin(headers: &HeaderMap, state: &AppState) -> Result<()> {
    let supplied = headers
        .get("x-admin-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !state.config.admin_api_key.is_empty() && supplied == state.config.admin_api_key {
        return Ok(());
    }
    let user = require_user(headers, state).await?;
    if user.role == "admin" {
        Ok(())
    } else {
        Err(AppError::Forbidden)
    }
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    let value = headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?;
    value.strip_prefix("Bearer ")
}

#[cfg(test)]
mod tests {
    use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};

    use async_trait::async_trait;
    use axum::http::{HeaderMap, HeaderValue};
    use tempfile::tempdir;

    use super::*;
    use crate::{
        config::Config,
        db::Db,
        queue::Admission,
        services::{RuntimeBackend, ServiceStatus},
        AppState,
    };

    struct NoopRuntime;

    #[async_trait]
    impl RuntimeBackend for NoopRuntime {
        async fn start(&self, name: &str) -> anyhow::Result<ServiceStatus> {
            Ok(ServiceStatus {
                name: name.into(),
                status: "ok".into(),
                detail: None,
            })
        }

        async fn stop(&self, name: &str) -> anyhow::Result<ServiceStatus> {
            Ok(ServiceStatus {
                name: name.into(),
                status: "ok".into(),
                detail: None,
            })
        }

        async fn list(&self) -> anyhow::Result<Vec<ServiceStatus>> {
            Ok(vec![])
        }
    }

    fn config_for(path: &std::path::Path) -> Config {
        Config {
            bind_addr: "127.0.0.1:0".parse::<SocketAddr>().unwrap(),
            profile: "local_rtx_5090".into(),
            runtime_backend: "native".into(),
            compose_project: "test".into(),
            turso_db_url: format!("file:{}", path.display()),
            turso_auth_token: String::new(),
            admin_api_key: "automation-admin-key".into(),
            admin_emails: vec![],
            cors_permissive: true,
            cors_allowed_origins: vec![],
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

    async fn state() -> anyhow::Result<(tempfile::TempDir, AppState)> {
        let dir = tempdir()?;
        let config = Arc::new(config_for(&dir.path().join("gateway.db")));
        let db = Db::connect(&config).await?;
        db.migrate().await?;
        Ok((
            dir,
            AppState {
                config,
                db,
                http: reqwest::Client::new(),
                admission: Admission::new(1, 1, 8, 8),
                runtime: Arc::new(NoopRuntime),
            },
        ))
    }

    fn bearer_headers(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        headers
    }

    #[tokio::test]
    async fn password_minimum_matches_frontend_contract() {
        assert!(matches!(
            hash_password("short"),
            Err(AppError::BadRequest(_))
        ));
        assert!(hash_password("long-enough-password").is_ok());
    }

    #[tokio::test]
    async fn require_admin_accepts_admin_bearer_or_admin_key_only() -> anyhow::Result<()> {
        let (_dir, state) = state().await?;
        let admin = state
            .db
            .create_user("admin@example.com", "hash", "admin")
            .await?;
        let user = state
            .db
            .create_user("user@example.com", "hash", "user")
            .await?;
        let admin_token = "admin-token";
        let user_token = "user-token";
        state
            .db
            .create_session(&hash_token(admin_token), &admin.id, 3600)
            .await?;
        state
            .db
            .create_session(&hash_token(user_token), &user.id, 3600)
            .await?;

        assert!(require_admin(&bearer_headers(admin_token), &state)
            .await
            .is_ok());
        assert!(matches!(
            require_admin(&bearer_headers(user_token), &state).await,
            Err(AppError::Forbidden)
        ));

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-admin-key",
            HeaderValue::from_static("automation-admin-key"),
        );
        assert!(require_admin(&headers, &state).await.is_ok());
        Ok(())
    }
}
