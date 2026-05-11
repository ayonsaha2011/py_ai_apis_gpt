use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("unauthorized")]
    Unauthorized,
    #[error("forbidden")]
    Forbidden,
    #[error("not found")]
    NotFound,
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("service unavailable: {0}")]
    Unavailable(String),
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),
    #[error(transparent)]
    Libsql(#[from] libsql::Error),
}

pub type Result<T> = std::result::Result<T, AppError>;

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self {
            AppError::Unauthorized => StatusCode::UNAUTHORIZED,
            AppError::Forbidden => StatusCode::FORBIDDEN,
            AppError::NotFound => StatusCode::NOT_FOUND,
            AppError::Conflict(_) => StatusCode::CONFLICT,
            AppError::BadRequest(_) => StatusCode::UNPROCESSABLE_ENTITY,
            AppError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            AppError::Anyhow(_) | AppError::Reqwest(_) | AppError::Libsql(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        let error_id = Uuid::now_v7();
        if status.is_server_error() {
            tracing::error!(
                error_id = %error_id,
                status = status.as_u16(),
                error = %self,
                error_debug = ?self,
                "request failed"
            );
        } else if status != StatusCode::UNAUTHORIZED {
            tracing::warn!(
                error_id = %error_id,
                status = status.as_u16(),
                error = %self,
                "request rejected"
            );
        }
        let body = Json(json!({
            "error": {
                "id": error_id,
                "status": status.as_u16(),
                "code": status.canonical_reason().unwrap_or("error"),
                "message": self.to_string()
            }
        }));
        (status, body).into_response()
    }
}
