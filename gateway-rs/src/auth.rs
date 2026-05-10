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

pub fn require_admin(headers: &HeaderMap, state: &AppState) -> Result<()> {
    let supplied = headers
        .get("x-admin-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !state.config.admin_api_key.is_empty() && supplied == state.config.admin_api_key {
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
