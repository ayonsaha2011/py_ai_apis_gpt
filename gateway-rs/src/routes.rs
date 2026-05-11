use std::{convert::Infallible, io::SeekFrom, time::Duration};

use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response, Sse},
    routing::{delete, get, patch, post},
    Json, Router,
};
use futures_util::{StreamExt, TryStreamExt};
use regex::Regex;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio_stream::wrappers::IntervalStream;
use tower_http::services::{ServeDir, ServeFile};
use uuid::Uuid;

use crate::{
    auth,
    db::now_ts,
    error::{AppError, Result},
    gpu,
    models::{
        ChatRequest, CreateRagCollectionRequest, LoginRequest, MeResponse, RagIngestRequest,
        RegisterRequest, TokenResponse, VideoJobRequest,
    },
    validation::{effective_seed, validate_video_request, video_r2_key},
    AppState,
};

pub fn router(state: AppState) -> Router {
    let frontend_dir = state.config.frontend_dir.clone();
    let app = Router::new()
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/auth/register", post(register))
        .route("/auth/login", post(login))
        .route("/auth/logout", post(logout))
        .route("/auth/me", get(me))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/history/chat", get(history_chat))
        .route("/history/sessions", get(history_sessions))
        .route("/history/videos", get(history_videos))
        .route(
            "/rag/collections",
            post(create_rag_collection).get(list_rag_collections),
        )
        .route("/rag/ingest", post(rag_ingest))
        .route("/rag/documents/:id", delete(delete_rag_document))
        .route("/v1/video/jobs", post(create_video_job))
        .route(
            "/v1/video/jobs/:id",
            get(get_video_job).delete(cancel_video_job),
        )
        .route("/v1/video/jobs/:id/events", get(video_job_events))
        .route("/admin/gpus", get(admin_gpus))
        .route("/admin/services", get(admin_services))
        .route("/admin/logs/:service", get(admin_logs))
        .route("/admin/users", get(admin_users))
        .route("/admin/users/:id/role", patch(admin_update_user_role))
        .route("/admin/services/:name/start", post(admin_start_service))
        .route("/admin/services/:name/stop", post(admin_stop_service))
        .route("/admin/models/:model_id/start", post(admin_start_model))
        .with_state(state);

    if frontend_dir.join("index.html").is_file() {
        tracing::info!(
            "serving frontend static assets from {}",
            frontend_dir.display()
        );
        app.fallback_service(
            ServeDir::new(&frontend_dir).fallback(ServeFile::new(frontend_dir.join("index.html"))),
        )
    } else {
        tracing::warn!(
            "frontend static assets disabled; missing {}",
            frontend_dir.join("index.html").display()
        );
        app
    }
}

#[derive(Debug, Deserialize)]
struct HistoryQuery {
    session_id: Option<String>,
    limit: Option<i64>,
    offset: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct LogsQuery {
    lines: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct UpdateUserRoleRequest {
    role: String,
}

pub fn spawn_video_dispatcher(state: AppState) {
    tokio::spawn(async move {
        loop {
            if let Err(err) = dispatch_one_video_job(state.clone()).await {
                tracing::warn!("video dispatcher: {err}");
                tokio::time::sleep(Duration::from_secs(2)).await;
            } else {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    });
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok", "service": "gateway-rs"}))
}

async fn status(State(state): State<AppState>) -> Result<Json<Value>> {
    Ok(Json(json!({
        "status": "ok",
        "profile": state.config.profile,
        "ltx_full_dev_only": true,
        "queues": state.admission.stats(),
        "gpus": gpu::list_gpus().unwrap_or_default(),
    })))
}

async fn register(
    State(state): State<AppState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<TokenResponse>> {
    let email = req.email.trim().to_ascii_lowercase();
    if !email.contains('@') {
        return Err(AppError::BadRequest("valid email is required".into()));
    }
    if state.db.user_by_email(&email).await?.is_some() {
        return Err(AppError::Conflict("email already registered".into()));
    }
    let role = if state.db.user_count().await? == 0
        || state
            .config
            .admin_emails
            .iter()
            .any(|value| value == &email)
    {
        "admin"
    } else {
        "user"
    };
    let password_hash = auth::hash_password(&req.password)?;
    let user = state.db.create_user(&email, &password_hash, role).await?;
    let token = auth::new_token();
    state
        .db
        .create_session(
            &auth::hash_token(&token),
            &user.id,
            state.config.session_ttl.as_secs(),
        )
        .await?;
    state
        .db
        .create_audit_event(
            Some(&user.id),
            "auth.register",
            "user",
            &json!({"email": email, "role": user.role.clone()}),
        )
        .await?;
    Ok(Json(TokenResponse {
        access_token: token,
        token_type: "bearer",
        expires_in: state.config.session_ttl.as_secs(),
        role: user.role,
    }))
}

async fn login(
    State(state): State<AppState>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<TokenResponse>> {
    let email = req.email.trim().to_ascii_lowercase();
    let Some((user, password_hash)) = state.db.user_by_email(&email).await? else {
        return Err(AppError::Unauthorized);
    };
    if !auth::verify_password(&req.password, &password_hash)? {
        return Err(AppError::Unauthorized);
    }
    let token = auth::new_token();
    state
        .db
        .create_session(
            &auth::hash_token(&token),
            &user.id,
            state.config.session_ttl.as_secs(),
        )
        .await?;
    state
        .db
        .create_audit_event(
            Some(&user.id),
            "auth.login",
            "session",
            &json!({"email": user.email.clone()}),
        )
        .await?;
    Ok(Json(TokenResponse {
        access_token: token,
        token_type: "bearer",
        expires_in: state.config.session_ttl.as_secs(),
        role: user.role,
    }))
}

async fn logout(State(state): State<AppState>, headers: HeaderMap) -> Result<StatusCode> {
    let token = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .ok_or(AppError::Unauthorized)?;
    let token_hash = auth::hash_token(token);
    let user = state.db.user_by_session(&token_hash).await?;
    state.db.revoke_session(&token_hash).await?;
    state
        .db
        .create_audit_event(
            user.as_ref().map(|u| u.id.as_str()),
            "auth.logout",
            "session",
            &json!({}),
        )
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn me(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<MeResponse>> {
    let user = auth::require_user(&headers, &state).await?;
    Ok(Json(MeResponse {
        user_id: user.id,
        email: user.email,
        role: user.role,
        created_at: user.created_at,
    }))
}

async fn history_sessions(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>> {
    let user = auth::require_user(&headers, &state).await?;
    Ok(Json(json!({
        "sessions": state.db.list_chat_sessions(&user.id).await?
    })))
}

async fn history_chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HistoryQuery>,
) -> Result<Json<Value>> {
    let user = auth::require_user(&headers, &state).await?;
    let limit = query.limit.unwrap_or(50).clamp(1, 200);
    let offset = query.offset.unwrap_or(0).max(0);
    Ok(Json(json!({
        "messages": state
            .db
            .list_chat_messages(&user.id, query.session_id.as_deref(), limit, offset)
            .await?
    })))
}

async fn history_videos(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HistoryQuery>,
) -> Result<Json<Value>> {
    let user = auth::require_user(&headers, &state).await?;
    let limit = query.limit.unwrap_or(50).clamp(1, 200);
    let offset = query.offset.unwrap_or(0).max(0);
    Ok(Json(json!({
        "videos": state.db.list_video_jobs(&user.id, limit, offset).await?
    })))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<ChatRequest>,
) -> Result<Response> {
    let user = auth::require_user(&headers, &state).await?;
    let _wait = state.admission.admit_text_waiter()?;
    let _heavy = state.admission.acquire_text_heavy().await?;

    let session_id = req
        .session_id
        .clone()
        .unwrap_or_else(|| Uuid::now_v7().to_string());
    let model_id = req
        .model
        .clone()
        .unwrap_or_else(|| state.config.text_model_registry[0].clone());
    req.model = Some(model_id.clone());
    req.session_id = Some(session_id.clone());

    if let Some(last_user) = req.messages.iter().rev().find(|m| m.role == "user") {
        state
            .db
            .store_chat_message(
                &user.id,
                &session_id,
                "user",
                &last_user.content,
                Some(&model_id),
            )
            .await?;
    }

    let mut worker_req = state
        .http
        .post(format!(
            "{}/v1/chat/completions",
            state.config.text_worker_url
        ))
        .header("x-service-key", &state.config.service_api_key)
        .header("x-user-id", &user.id)
        .json(&req);

    if let Some(collection) = &req.rag_collection {
        if let Some(qdrant_name) = state.db.rag_qdrant_name(&user.id, collection).await? {
            worker_req = worker_req.header("x-rag-qdrant-collection", qdrant_name);
        }
    }

    let worker_resp = worker_req.send().await?;
    let status =
        StatusCode::from_u16(worker_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    if let Some(audit) = worker_resp
        .headers()
        .get("x-rag-audit")
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
    {
        state
            .db
            .create_audit_event(Some(&user.id), "rag.retrieve", "chat", &audit)
            .await?;
    }

    if req.stream.unwrap_or(false) {
        return stream_response(worker_resp, status);
    }

    let bytes = worker_resp.bytes().await?;
    if status.is_success() {
        if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
            if let Some(text) = value["choices"][0]["message"]["content"].as_str() {
                state
                    .db
                    .store_chat_message(&user.id, &session_id, "assistant", text, Some(&model_id))
                    .await?;
            }
            return Ok((status, Json(value)).into_response());
        }
    }
    Ok((status, Body::from(bytes)).into_response())
}

async fn create_rag_collection(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateRagCollectionRequest>,
) -> Result<Json<Value>> {
    let user = auth::require_user(&headers, &state).await?;
    let name = req.name.trim();
    if name.is_empty() {
        return Err(AppError::BadRequest("collection name is required".into()));
    }
    let (id, qdrant_name) = state.db.create_rag_collection(&user.id, name).await?;
    let worker_resp = state
        .http
        .post(format!("{}/rag/collections", state.config.text_worker_url))
        .header("x-service-key", &state.config.service_api_key)
        .json(&json!({"collection": qdrant_name}))
        .send()
        .await?;
    if !worker_resp.status().is_success() {
        return Err(AppError::Unavailable(format!(
            "text worker returned {}",
            worker_resp.status()
        )));
    }
    state
        .db
        .create_audit_event(
            Some(&user.id),
            "rag.collection.create",
            name,
            &json!({"collection_id": id, "qdrant_name": qdrant_name.clone()}),
        )
        .await?;
    Ok(Json(
        json!({"id": id, "name": name, "qdrant_name": qdrant_name}),
    ))
}

async fn list_rag_collections(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>> {
    let user = auth::require_user(&headers, &state).await?;
    Ok(Json(
        json!({"collections": state.db.list_rag_collections(&user.id).await?}),
    ))
}

async fn rag_ingest(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RagIngestRequest>,
) -> Result<Response> {
    let user = auth::require_user(&headers, &state).await?;
    let Some(qdrant_name) = state.db.rag_qdrant_name(&user.id, &req.collection).await? else {
        return Err(AppError::NotFound);
    };
    let document_id = Uuid::now_v7().to_string();
    let source = req.source_name.clone().unwrap_or_else(|| "api".into());
    let metadata_value = req.metadata.clone().unwrap_or_else(|| json!({}));
    let resp = state
        .http
        .post(format!("{}/rag/ingest", state.config.text_worker_url))
        .header("x-service-key", &state.config.service_api_key)
        .header("x-user-id", &user.id)
        .json(&json!({
            "collection": qdrant_name,
            "document_id": document_id,
            "texts": req.texts.clone(),
            "source_name": source.clone(),
            "metadata": metadata_value.clone()
        }))
        .send()
        .await?;
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let value = resp.json::<Value>().await.unwrap_or_else(|_| json!({}));
    if status.is_success() {
        let count = value
            .get("ingested_chunks")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let metadata = serde_json::to_string(&metadata_value).unwrap_or_else(|_| "{}".into());
        let document_id = state
            .db
            .create_rag_document(
                &document_id,
                &user.id,
                &req.collection,
                &source,
                &metadata,
                count,
            )
            .await?;
        state
            .db
            .create_audit_event(
                Some(&user.id),
                "rag.ingest",
                &req.collection,
                &json!({"document_id": document_id, "chunks": count, "source_name": source.clone()}),
            )
            .await?;
        return Ok((
            status,
            Json(json!({"document_id": document_id, "ingested_chunks": count})),
        )
            .into_response());
    }
    Ok((status, Json(value)).into_response())
}

async fn delete_rag_document(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Response> {
    let user = auth::require_user(&headers, &state).await?;
    let Some(collection) = state.db.rag_document_collection(&user.id, &id).await? else {
        return Err(AppError::NotFound);
    };
    let user_id = user.id.clone();
    let resp = state
        .http
        .delete(format!(
            "{}/rag/documents/{}",
            state.config.text_worker_url, id
        ))
        .header("x-service-key", &state.config.service_api_key)
        .header("x-rag-qdrant-collection", collection.clone())
        .header("x-user-id", user_id.clone())
        .send()
        .await?;
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    if status.is_success() {
        state.db.mark_rag_document_deleted(&user_id, &id).await?;
        state
            .db
            .create_audit_event(
                Some(&user_id),
                "rag.document.delete",
                &id,
                &json!({"qdrant_collection": collection}),
            )
            .await?;
    }
    proxy_complete_response(resp).await
}

async fn create_video_job(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<VideoJobRequest>,
) -> Result<(StatusCode, Json<Value>)> {
    let user = auth::require_user(&headers, &state).await?;
    validate_video_request(&req, &state.config.profile)?;
    let _wait = state.admission.admit_video_waiter()?;
    if state.db.count_open_video_jobs().await? >= state.config.ltx_max_waiting as i64 {
        return Err(AppError::Unavailable("video queue is full".into()));
    }
    let job_id = Uuid::now_v7();
    let seed = effective_seed(&user.id, &job_id, req.seed_hint, &req.prompt);
    let r2_key = video_r2_key(&user.id, &job_id);
    state
        .db
        .create_video_job(&user.id, &job_id, &req, seed, &r2_key)
        .await?;
    state
        .db
        .create_audit_event(
            Some(&user.id),
            "video.submit",
            &job_id.to_string(),
            &json!({
                "mode": req.mode.as_str(),
                "effective_seed": seed,
                "r2_key": r2_key.clone(),
                "width": req.width,
                "height": req.height,
                "num_frames": req.num_frames
            }),
        )
        .await?;
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({
            "job_id": job_id,
            "status": "queued",
            "effective_seed": seed,
            "r2_key": r2_key,
            "created_at": now_ts()
        })),
    ))
}

async fn get_video_job(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<Value>> {
    let user = auth::require_user(&headers, &state).await?;
    let Some(job) = state.db.video_job(&user.id, &id).await? else {
        return Err(AppError::NotFound);
    };
    Ok(Json(json!(job)))
}

async fn cancel_video_job(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<Value>> {
    let user = auth::require_user(&headers, &state).await?;
    if !state.db.cancel_video_job(&user.id, &id).await? {
        return Err(AppError::NotFound);
    }
    let _ = state
        .http
        .post(format!(
            "{}/internal/ltx/jobs/{}/cancel",
            state.config.ltx_worker_url, id
        ))
        .header("x-service-key", &state.config.service_api_key)
        .send()
        .await;
    state
        .db
        .create_audit_event(Some(&user.id), "video.cancel", &id, &json!({}))
        .await?;
    Ok(Json(json!({"job_id": id, "status": "cancelling"})))
}

async fn video_job_events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<
    Sse<
        impl futures_util::Stream<Item = std::result::Result<axum::response::sse::Event, Infallible>>,
    >,
> {
    let user = auth::require_user(&headers, &state).await?;
    let stream =
        IntervalStream::new(tokio::time::interval(Duration::from_secs(2))).then(move |_| {
            let state = state.clone();
            let user_id = user.id.clone();
            let id = id.clone();
            async move {
                let event = match state.db.video_job(&user_id, &id).await {
                    Ok(Some(job)) => axum::response::sse::Event::default()
                        .json_data(job)
                        .unwrap_or_else(|_| axum::response::sse::Event::default().data("{}")),
                    Ok(None) => axum::response::sse::Event::default()
                        .event("error")
                        .data("not_found"),
                    Err(err) => axum::response::sse::Event::default()
                        .event("error")
                        .data(err.to_string()),
                };
                Ok(event)
            }
        });
    Ok(Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()))
}

async fn admin_gpus(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<Value>> {
    auth::require_admin(&headers, &state).await?;
    Ok(Json(json!({"gpus": gpu::list_gpus().unwrap_or_default()})))
}

async fn admin_services(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<Value>> {
    auth::require_admin(&headers, &state).await?;
    Ok(Json(json!({"services": state.runtime.list().await?})))
}

async fn admin_users(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<Value>> {
    auth::require_admin(&headers, &state).await?;
    Ok(Json(json!({"users": state.db.list_users().await?})))
}

async fn admin_update_user_role(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<UpdateUserRoleRequest>,
) -> Result<Json<Value>> {
    auth::require_admin(&headers, &state).await?;
    let role = req.role.trim();
    if !matches!(role, "admin" | "user") {
        return Err(AppError::BadRequest("role must be admin or user".into()));
    }
    if !state.db.update_user_role(&id, role).await? {
        return Err(AppError::NotFound);
    }
    let actor = auth::require_user(&headers, &state).await.ok();
    state
        .db
        .create_audit_event(
            actor.as_ref().map(|user| user.id.as_str()),
            "admin.user.role",
            &id,
            &json!({"role": role}),
        )
        .await?;
    Ok(Json(json!({"user_id": id, "role": role})))
}

async fn admin_logs(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(service): Path<String>,
    Query(query): Query<LogsQuery>,
) -> Result<Json<Value>> {
    auth::require_admin(&headers, &state).await?;
    let Some(name) = log_name(&service) else {
        return Err(AppError::BadRequest("unknown service log".into()));
    };
    let lines = query.lines.unwrap_or(200).clamp(1, 2000);
    let path = state.config.log_dir.join(format!("{name}.log"));
    let content = redact_log_secrets(&tail_file(&path, lines).await?);
    Ok(Json(json!({
        "service": name,
        "path": path,
        "lines": lines,
        "content": content,
    })))
}

async fn admin_start_service(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<Json<Value>> {
    auth::require_admin(&headers, &state).await?;
    let actor = auth::require_user(&headers, &state).await.ok();
    let result = state.runtime.start(&name).await?;
    state
        .db
        .create_audit_event(
            actor.as_ref().map(|user| user.id.as_str()),
            "admin.service.start",
            &name,
            &json!({}),
        )
        .await?;
    Ok(Json(json!(result)))
}

async fn admin_stop_service(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<Json<Value>> {
    auth::require_admin(&headers, &state).await?;
    let actor = auth::require_user(&headers, &state).await.ok();
    let result = state.runtime.stop(&name).await?;
    state
        .db
        .create_audit_event(
            actor.as_ref().map(|user| user.id.as_str()),
            "admin.service.stop",
            &name,
            &json!({}),
        )
        .await?;
    Ok(Json(json!(result)))
}

async fn admin_start_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(model_id): Path<String>,
) -> Result<Json<Value>> {
    auth::require_admin(&headers, &state).await?;
    let actor = auth::require_user(&headers, &state).await.ok();
    let resp = state
        .http
        .post(format!(
            "{}/admin/models/start",
            state.config.text_worker_url
        ))
        .header("x-service-key", &state.config.service_api_key)
        .json(&json!({"model_id": model_id.clone()}))
        .send()
        .await?;
    let value = proxy_json(resp).await?;
    state
        .db
        .create_audit_event(
            actor.as_ref().map(|user| user.id.as_str()),
            "admin.model.start",
            &model_id,
            &json!({}),
        )
        .await?;
    Ok(value)
}

async fn dispatch_one_video_job(state: AppState) -> Result<()> {
    let Some((job_id, params_json, user_id, seed, r2_key)) =
        state.db.next_queued_video_job().await?
    else {
        return Ok(());
    };
    let req: VideoJobRequest = match serde_json::from_str(&params_json) {
        Ok(req) => req,
        Err(err) => {
            tracing::error!(
                job_id = %job_id,
                user_id = %user_id,
                error = %err,
                params_json = %params_json,
                "video job contains invalid params_json"
            );
            state
                .db
                .update_video_job(
                    &job_id,
                    "failed",
                    0.0,
                    None,
                    Some(&format!("invalid video job params_json: {err}")),
                    None,
                )
                .await?;
            return Ok(());
        }
    };
    let _permit = state.admission.acquire_video_heavy().await?;
    state
        .db
        .mark_video_stage(&job_id, "materializing_inputs", 0.05)
        .await?;
    let body = json!({
        "job_id": job_id,
        "request": req,
        "effective_seed": seed,
        "r2_key": r2_key.clone(),
        "user_id": user_id.clone(),
    });
    state
        .db
        .mark_video_stage(&job_id, "generating", 0.10)
        .await?;
    let result = state
        .http
        .post(format!("{}/internal/ltx/jobs", state.config.ltx_worker_url))
        .header("x-service-key", &state.config.service_api_key)
        .json(&body)
        .send()
        .await;
    match result {
        Ok(resp) if resp.status().is_success() => {
            let value = resp.json::<Value>().await.unwrap_or_else(|_| json!({}));
            let url = value.get("result_url").and_then(Value::as_str);
            let metadata = value.get("metadata");
            state
                .db
                .update_video_job(&job_id, "complete", 1.0, url, None, metadata)
                .await?;
            state
                .db
                .create_audit_event(
                    Some(&user_id),
                    "video.complete",
                    &job_id,
                    &json!({"result_url": url, "r2_key": r2_key}),
                )
                .await?;
        }
        Ok(resp) => {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            tracing::error!(
                job_id = %job_id,
                user_id = %user_id,
                status = status.as_u16(),
                mode = req.mode.as_str(),
                width = req.width,
                height = req.height,
                num_frames = req.num_frames,
                worker_error = %text,
                "LTX worker returned a failed video job response"
            );
            state
                .db
                .update_video_job(
                    &job_id,
                    "failed",
                    0.0,
                    None,
                    Some(&format!("{status}: {text}")),
                    None,
                )
                .await?;
        }
        Err(err) => {
            tracing::error!(
                job_id = %job_id,
                user_id = %user_id,
                mode = req.mode.as_str(),
                width = req.width,
                height = req.height,
                num_frames = req.num_frames,
                error = %err,
                error_debug = ?err,
                "failed to call LTX worker for video job"
            );
            state
                .db
                .update_video_job(&job_id, "failed", 0.0, None, Some(&err.to_string()), None)
                .await?;
        }
    }
    Ok(())
}

fn stream_response(resp: reqwest::Response, status: StatusCode) -> Result<Response> {
    let mut builder = Response::builder().status(status);
    if let Some(headers) = builder.headers_mut() {
        headers.insert(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        headers.insert(
            axum::http::header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        );
    }
    let stream = resp.bytes_stream().map_err(std::io::Error::other);
    Ok(builder
        .body(Body::from_stream(stream))
        .map_err(|e| AppError::BadRequest(e.to_string()))?)
}

async fn proxy_complete_response(resp: reqwest::Response) -> Result<Response> {
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let bytes = resp.bytes().await?;
    Ok((status, Body::from(bytes)).into_response())
}

async fn proxy_json(resp: reqwest::Response) -> Result<Json<Value>> {
    let status = resp.status();
    let value = resp
        .json::<Value>()
        .await
        .unwrap_or_else(|_| json!({"status": status.as_u16()}));
    Ok(Json(value))
}

fn log_name(service: &str) -> Option<&'static str> {
    match service {
        "gateway" => Some("gateway"),
        "text" | "text-worker" => Some("text"),
        "ltx" | "ltx-worker" => Some("ltx"),
        "qdrant" => Some("qdrant"),
        _ => None,
    }
}

async fn tail_file(path: &std::path::Path, lines: usize) -> Result<String> {
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|_| AppError::NotFound)?;
    let len = file
        .metadata()
        .await
        .map_err(|e| AppError::BadRequest(e.to_string()))?
        .len();
    let start = len.saturating_sub(1024 * 1024);
    file.seek(SeekFrom::Start(start))
        .await
        .map_err(|e| AppError::BadRequest(e.to_string()))?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)
        .await
        .map_err(|e| AppError::BadRequest(e.to_string()))?;
    let selected = buf
        .lines()
        .rev()
        .take(lines)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");
    Ok(selected)
}

fn redact_log_secrets(input: &str) -> String {
    let re = Regex::new(
        r"(?i)\b(token|secret|password|authorization|api[_-]?key)(\s*[:=]\s*)([^\s,;]+)",
    )
    .expect("valid secret redaction regex");
    re.replace_all(input, "$1$2[REDACTED]").into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_name_allows_only_known_services() {
        assert_eq!(log_name("gateway"), Some("gateway"));
        assert_eq!(log_name("text-worker"), Some("text"));
        assert_eq!(log_name("ltx-worker"), Some("ltx"));
        assert_eq!(log_name("../gateway"), None);
        assert_eq!(log_name("gateway/../../secret"), None);
    }

    #[test]
    fn log_redaction_masks_common_secret_shapes() {
        let redacted = redact_log_secrets(
            "token=abc123 password: hunter2 api_key=key123 authorization: Bearer xyz",
        );
        assert!(redacted.contains("token=[REDACTED]"));
        assert!(redacted.contains("password: [REDACTED]"));
        assert!(redacted.contains("api_key=[REDACTED]"));
        assert!(redacted.contains("authorization: [REDACTED]"));
        assert!(!redacted.contains("hunter2"));
        assert!(!redacted.contains("key123"));
    }
}
