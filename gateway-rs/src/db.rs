use std::sync::Arc;

use chrono::Utc;
use libsql::{params, Builder, Connection};
use serde_json::Value;
use uuid::Uuid;

use crate::{
    config::Config,
    models::{User, VideoJobRequest, VideoJobStatus},
};

#[derive(Clone)]
pub struct Db {
    conn: Arc<Connection>,
}

impl Db {
    pub async fn connect(config: &Config) -> anyhow::Result<Self> {
        let database = if config.turso_db_url.starts_with("libsql://") {
            Builder::new_remote(config.turso_db_url.clone(), config.turso_auth_token.clone())
                .build()
                .await?
        } else {
            let path = config
                .turso_db_url
                .strip_prefix("file:")
                .unwrap_or(&config.turso_db_url);
            if let Some(parent) = std::path::Path::new(path).parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            Builder::new_local(path).build().await?
        };
        Ok(Self {
            conn: Arc::new(database.connect()?),
        })
    }

    pub async fn migrate(&self) -> anyhow::Result<()> {
        self.conn
            .execute_batch(include_str!("../migrations/001_init.sql"))
            .await?;
        self.record_migration_if_missing("001_init").await?;
        if !self.migration_applied("002_user_roles").await? {
            self.ensure_user_role_column().await?;
            self.ensure_bootstrap_admin().await?;
            self.record_migration("002_user_roles").await?;
        }
        if !self.migration_applied("003_video_metadata").await? {
            self.add_column_if_missing("video_jobs", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
                .await?;
            self.record_migration("003_video_metadata").await?;
        }
        if !self
            .migration_applied("004_video_job_runtime_columns")
            .await?
        {
            self.ensure_video_job_runtime_columns().await?;
            self.record_migration("004_video_job_runtime_columns")
                .await?;
        }
        if !self.migration_applied("005_video_job_core_columns").await? {
            self.ensure_video_job_core_columns().await?;
            self.record_migration("005_video_job_core_columns").await?;
        }
        Ok(())
    }

    async fn ensure_user_role_column(&self) -> anyhow::Result<()> {
        self.add_column_if_missing("users", "role", "TEXT NOT NULL DEFAULT 'user'")
            .await
    }

    async fn ensure_video_job_runtime_columns(&self) -> anyhow::Result<()> {
        self.add_column_if_missing("video_jobs", "result_url", "TEXT")
            .await?;
        self.add_column_if_missing("video_jobs", "error", "TEXT")
            .await?;
        self.add_column_if_missing("video_jobs", "progress", "REAL NOT NULL DEFAULT 0.0")
            .await?;
        self.add_column_if_missing(
            "video_jobs",
            "cancel_requested",
            "INTEGER NOT NULL DEFAULT 0",
        )
        .await?;
        self.add_column_if_missing("video_jobs", "started_at", "INTEGER")
            .await?;
        self.add_column_if_missing("video_jobs", "updated_at", "INTEGER NOT NULL DEFAULT 0")
            .await?;
        self.add_column_if_missing("video_jobs", "completed_at", "INTEGER")
            .await?;
        self.add_column_if_missing("video_jobs", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
            .await?;
        Ok(())
    }

    async fn ensure_video_job_core_columns(&self) -> anyhow::Result<()> {
        self.add_column_if_missing("video_jobs", "id", "TEXT NOT NULL DEFAULT ''")
            .await?;
        self.add_column_if_missing("video_jobs", "user_id", "TEXT NOT NULL DEFAULT ''")
            .await?;
        self.add_column_if_missing(
            "video_jobs",
            "mode",
            "TEXT NOT NULL DEFAULT 'text_to_video'",
        )
        .await?;
        self.add_column_if_missing("video_jobs", "status", "TEXT NOT NULL DEFAULT 'failed'")
            .await?;
        self.add_column_if_missing("video_jobs", "prompt", "TEXT NOT NULL DEFAULT ''")
            .await?;
        self.add_column_if_missing("video_jobs", "params_json", "TEXT NOT NULL DEFAULT '{}'")
            .await?;
        self.add_column_if_missing("video_jobs", "effective_seed", "INTEGER NOT NULL DEFAULT 0")
            .await?;
        self.add_column_if_missing("video_jobs", "r2_key", "TEXT NOT NULL DEFAULT ''")
            .await?;
        self.add_column_if_missing("video_jobs", "created_at", "INTEGER NOT NULL DEFAULT 0")
            .await?;
        self.fail_legacy_video_jobs_without_params().await?;
        Ok(())
    }

    async fn fail_legacy_video_jobs_without_params(&self) -> anyhow::Result<()> {
        let now = now_ts();
        self.conn
            .execute(
                "UPDATE video_jobs SET status = 'failed', progress = 0.0, error = COALESCE(error, 'legacy video job is missing params_json; resubmit the request'), updated_at = CASE WHEN updated_at = 0 THEN ? ELSE updated_at END, completed_at = COALESCE(completed_at, ?) WHERE params_json = '{}' AND status IN ('queued','running','materializing_inputs','generating','encoding','upscaling','uploading')",
                params![now, now],
            )
            .await?;
        Ok(())
    }

    async fn add_column_if_missing(
        &self,
        table: &str,
        column: &str,
        definition: &str,
    ) -> anyhow::Result<()> {
        if !is_identifier(table) || !is_identifier(column) {
            anyhow::bail!("invalid migration identifier")
        }
        if !self.column_exists(table, column).await? {
            self.conn
                .execute(
                    &format!("ALTER TABLE {table} ADD COLUMN {column} {definition}"),
                    (),
                )
                .await?;
        }
        Ok(())
    }

    async fn column_exists(&self, table: &str, column: &str) -> anyhow::Result<bool> {
        let mut rows = self
            .conn
            .query(&format!("PRAGMA table_info({table})"), ())
            .await?;
        while let Some(row) = rows.next().await? {
            let name: String = row.get(1)?;
            if name == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn migration_applied(&self, version: &str) -> anyhow::Result<bool> {
        let mut rows = self
            .conn
            .query(
                "SELECT 1 FROM schema_migrations WHERE version = ?",
                params![version],
            )
            .await?;
        Ok(rows.next().await?.is_some())
    }

    async fn record_migration_if_missing(&self, version: &str) -> anyhow::Result<()> {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                params![version, now_ts()],
            )
            .await?;
        Ok(())
    }

    async fn record_migration(&self, version: &str) -> anyhow::Result<()> {
        self.conn
            .execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                params![version, now_ts()],
            )
            .await?;
        Ok(())
    }

    async fn ensure_bootstrap_admin(&self) -> anyhow::Result<()> {
        let mut rows = self
            .conn
            .query("SELECT COUNT(*) FROM users WHERE role = 'admin'", ())
            .await?;
        let admin_count = if let Some(row) = rows.next().await? {
            row.get::<i64>(0)?
        } else {
            0
        };
        if admin_count == 0 {
            self.conn
                .execute(
                    "UPDATE users SET role = 'admin' WHERE id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1)",
                    (),
                )
                .await?;
        }
        Ok(())
    }

    pub async fn ensure_model_registry(&self, config: &Config) -> anyhow::Result<()> {
        let now = now_ts();
        for model in &config.text_model_registry {
            self.conn
                .execute(
                    "INSERT OR IGNORE INTO model_registry (model_id, worker_kind, enabled, config_json, created_at) VALUES (?, 'text', 1, '{}', ?)",
                    params![model.clone(), now],
                )
                .await?;
        }
        Ok(())
    }

    pub async fn user_count(&self) -> anyhow::Result<i64> {
        let mut rows = self.conn.query("SELECT COUNT(*) FROM users", ()).await?;
        Ok(if let Some(row) = rows.next().await? {
            row.get(0)?
        } else {
            0
        })
    }

    pub async fn create_user(
        &self,
        email: &str,
        password_hash: &str,
        role: &str,
    ) -> anyhow::Result<User> {
        let user = User {
            id: Uuid::now_v7().to_string(),
            email: email.to_owned(),
            role: role.to_owned(),
            created_at: now_ts(),
        };
        self.conn
            .execute(
                "INSERT INTO users (id, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
                params![
                    user.id.clone(),
                    user.email.clone(),
                    password_hash,
                    user.role.clone(),
                    user.created_at
                ],
            )
            .await?;
        Ok(user)
    }

    pub async fn user_by_email(&self, email: &str) -> anyhow::Result<Option<(User, String)>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, email, password_hash, role, created_at FROM users WHERE email = ?",
                params![email],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            let user = User {
                id: row.get(0)?,
                email: row.get(1)?,
                role: row.get(3)?,
                created_at: row.get(4)?,
            };
            let hash: String = row.get(2)?;
            Ok(Some((user, hash)))
        } else {
            Ok(None)
        }
    }

    pub async fn user_by_id(&self, id: &str) -> anyhow::Result<Option<User>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, email, role, created_at FROM users WHERE id = ?",
                params![id],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            Ok(Some(User {
                id: row.get(0)?,
                email: row.get(1)?,
                role: row.get(2)?,
                created_at: row.get(3)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn list_users(&self) -> anyhow::Result<Vec<Value>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, email, role, created_at FROM users ORDER BY created_at DESC LIMIT 500",
                (),
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(serde_json::json!({
                "user_id": row.get::<String>(0)?,
                "email": row.get::<String>(1)?,
                "role": row.get::<String>(2)?,
                "created_at": row.get::<i64>(3)?,
            }));
        }
        Ok(out)
    }

    pub async fn update_user_role(&self, user_id: &str, role: &str) -> anyhow::Result<bool> {
        let changed = self
            .conn
            .execute(
                "UPDATE users SET role = ? WHERE id = ?",
                params![role, user_id],
            )
            .await?;
        Ok(changed > 0)
    }

    pub async fn create_session(
        &self,
        token_hash: &str,
        user_id: &str,
        ttl_seconds: u64,
    ) -> anyhow::Result<()> {
        let now = now_ts();
        let expires = now + ttl_seconds as i64;
        self.conn
            .execute(
                "INSERT INTO sessions (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
                params![token_hash, user_id, expires, now],
            )
            .await?;
        Ok(())
    }

    pub async fn revoke_session(&self, token_hash: &str) -> anyhow::Result<()> {
        self.conn
            .execute(
                "UPDATE sessions SET revoked_at = ? WHERE token_hash = ?",
                params![now_ts(), token_hash],
            )
            .await?;
        Ok(())
    }

    pub async fn user_by_session(&self, token_hash: &str) -> anyhow::Result<Option<User>> {
        let mut rows = self
            .conn
            .query(
                "SELECT u.id, u.email, u.role, u.created_at FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ? AND s.expires_at > ? AND s.revoked_at IS NULL",
                params![token_hash, now_ts()],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            Ok(Some(User {
                id: row.get(0)?,
                email: row.get(1)?,
                role: row.get(2)?,
                created_at: row.get(3)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn create_audit_event(
        &self,
        user_id: Option<&str>,
        event_type: &str,
        target: &str,
        metadata: &Value,
    ) -> anyhow::Result<()> {
        self.conn
            .execute(
                "INSERT INTO audit_events (id, user_id, event_type, target, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                params![
                    Uuid::now_v7().to_string(),
                    user_id,
                    event_type,
                    target,
                    serde_json::to_string(metadata)?,
                    now_ts()
                ],
            )
            .await?;
        Ok(())
    }

    pub async fn create_chat_session_if_missing(
        &self,
        user_id: &str,
        session_id: &str,
    ) -> anyhow::Result<()> {
        let now = now_ts();
        self.conn
            .execute(
                "INSERT OR IGNORE INTO chat_sessions (id, user_id, title, created_at, updated_at) VALUES (?, ?, NULL, ?, ?)",
                params![session_id, user_id, now, now],
            )
            .await?;
        Ok(())
    }

    pub async fn store_chat_message(
        &self,
        user_id: &str,
        session_id: &str,
        role: &str,
        content: &str,
        model_id: Option<&str>,
    ) -> anyhow::Result<()> {
        self.create_chat_session_if_missing(user_id, session_id)
            .await?;
        let id = Uuid::now_v7().to_string();
        let now = now_ts();
        self.conn
            .execute(
                "INSERT INTO chat_messages (id, session_id, user_id, role, content, model_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![id, session_id, user_id, role, content, model_id, now],
            )
            .await?;
        self.conn
            .execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ? AND user_id = ?",
                params![now, session_id, user_id],
            )
            .await?;
        Ok(())
    }

    pub async fn list_chat_sessions(
        &self,
        user_id: &str,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT 100",
                params![user_id],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(serde_json::json!({
                "session_id": row.get::<String>(0)?,
                "title": row.get::<Option<String>>(1)?,
                "created_at": row.get::<i64>(2)?,
                "updated_at": row.get::<i64>(3)?,
            }));
        }
        Ok(out)
    }

    pub async fn list_chat_messages(
        &self,
        user_id: &str,
        session_id: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut out = Vec::new();
        if let Some(session_id) = session_id {
            let mut rows = self
                .conn
                .query(
                    "SELECT id, session_id, role, content, model_id, created_at FROM chat_messages WHERE user_id = ? AND session_id = ? ORDER BY created_at ASC LIMIT ? OFFSET ?",
                    params![user_id, session_id, limit, offset],
                )
                .await?;
            while let Some(row) = rows.next().await? {
                out.push(chat_message_json(&row)?);
            }
        } else {
            let mut rows = self
                .conn
                .query(
                    "SELECT id, session_id, role, content, model_id, created_at FROM chat_messages WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    params![user_id, limit, offset],
                )
                .await?;
            while let Some(row) = rows.next().await? {
                out.push(chat_message_json(&row)?);
            }
        }
        Ok(out)
    }

    pub async fn create_rag_collection(
        &self,
        user_id: &str,
        name: &str,
    ) -> anyhow::Result<(String, String)> {
        let id = Uuid::now_v7().to_string();
        let qdrant_name = format!("u_{}_{}", user_id.replace('-', ""), name.replace('-', "_"));
        self.conn
            .execute(
                "INSERT INTO rag_collections (id, user_id, name, qdrant_name, created_at) VALUES (?, ?, ?, ?, ?)",
                params![id.clone(), user_id, name, qdrant_name.clone(), now_ts()],
            )
            .await?;
        Ok((id, qdrant_name))
    }

    pub async fn rag_qdrant_name(
        &self,
        user_id: &str,
        name: &str,
    ) -> anyhow::Result<Option<String>> {
        let mut rows = self
            .conn
            .query(
                "SELECT qdrant_name FROM rag_collections WHERE user_id = ? AND name = ?",
                params![user_id, name],
            )
            .await?;
        Ok(if let Some(row) = rows.next().await? {
            Some(row.get(0)?)
        } else {
            None
        })
    }

    pub async fn list_rag_collections(
        &self,
        user_id: &str,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, name, qdrant_name, created_at FROM rag_collections WHERE user_id = ? ORDER BY created_at DESC",
                params![user_id],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(serde_json::json!({
                "id": row.get::<String>(0)?,
                "name": row.get::<String>(1)?,
                "qdrant_name": row.get::<String>(2)?,
                "created_at": row.get::<i64>(3)?,
            }));
        }
        Ok(out)
    }

    pub async fn create_rag_document(
        &self,
        document_id: &str,
        user_id: &str,
        collection_name: &str,
        source_name: &str,
        metadata_json: &str,
        chunk_count: i64,
    ) -> anyhow::Result<String> {
        let mut rows = self
            .conn
            .query(
                "SELECT id FROM rag_collections WHERE user_id = ? AND name = ?",
                params![user_id, collection_name],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            anyhow::bail!("rag collection not found")
        };
        let collection_id: String = row.get(0)?;
        self.conn
            .execute(
                "INSERT INTO rag_documents (id, collection_id, user_id, source_name, metadata_json, chunk_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![document_id, collection_id, user_id, source_name, metadata_json, chunk_count, now_ts()],
            )
            .await?;
        Ok(document_id.to_owned())
    }

    pub async fn rag_document_collection(
        &self,
        user_id: &str,
        document_id: &str,
    ) -> anyhow::Result<Option<String>> {
        let mut rows = self
            .conn
            .query(
                "SELECT c.qdrant_name FROM rag_documents d JOIN rag_collections c ON c.id = d.collection_id WHERE d.id = ? AND d.user_id = ? AND d.deleted_at IS NULL",
                params![document_id, user_id],
            )
            .await?;
        Ok(if let Some(row) = rows.next().await? {
            Some(row.get(0)?)
        } else {
            None
        })
    }

    pub async fn mark_rag_document_deleted(
        &self,
        user_id: &str,
        document_id: &str,
    ) -> anyhow::Result<()> {
        self.conn
            .execute(
                "UPDATE rag_documents SET deleted_at = ? WHERE id = ? AND user_id = ?",
                params![now_ts(), document_id, user_id],
            )
            .await?;
        Ok(())
    }

    pub async fn create_video_job(
        &self,
        user_id: &str,
        job_id: &Uuid,
        req: &VideoJobRequest,
        effective_seed: i64,
        r2_key: &str,
    ) -> anyhow::Result<()> {
        let now = now_ts();
        let metadata = serde_json::json!({
            "upscaled": false,
            "render_width": req.width,
            "render_height": req.height,
            "output_width": req.width,
            "output_height": req.height,
        });
        self.conn
            .execute(
                "INSERT INTO video_jobs (id, user_id, mode, status, prompt, params_json, effective_seed, r2_key, metadata_json, created_at, updated_at) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)",
                params![
                    job_id.to_string(),
                    user_id,
                    req.mode.as_str(),
                    req.prompt.clone(),
                    serde_json::to_string(req)?,
                    effective_seed,
                    r2_key,
                    serde_json::to_string(&metadata)?,
                    now,
                    now
                ],
            )
            .await?;
        Ok(())
    }

    pub async fn count_open_video_jobs(&self) -> anyhow::Result<i64> {
        let mut rows = self
            .conn
            .query(
                "SELECT COUNT(*) FROM video_jobs WHERE status IN ('queued','running','materializing_inputs','generating','encoding','upscaling','uploading')",
                (),
            )
            .await?;
        Ok(if let Some(row) = rows.next().await? {
            row.get(0)?
        } else {
            0
        })
    }

    pub async fn update_video_job(
        &self,
        job_id: &str,
        status: &str,
        progress: f32,
        result_url: Option<&str>,
        error: Option<&str>,
        metadata: Option<&Value>,
    ) -> anyhow::Result<()> {
        let now = now_ts();
        let completed = if matches!(status, "complete" | "failed" | "cancelled") {
            Some(now)
        } else {
            None
        };
        let metadata_json = metadata.map(serde_json::to_string).transpose()?;
        self.conn
            .execute(
                "UPDATE video_jobs SET status = ?, progress = ?, result_url = COALESCE(?, result_url), error = ?, metadata_json = COALESCE(?, metadata_json), updated_at = ?, completed_at = COALESCE(?, completed_at) WHERE id = ?",
                params![status, progress, result_url, error, metadata_json, now, completed, job_id],
            )
            .await?;
        Ok(())
    }

    pub async fn mark_video_stage(
        &self,
        job_id: &str,
        status: &str,
        progress: f32,
    ) -> anyhow::Result<()> {
        let now = now_ts();
        self.conn
            .execute(
                "UPDATE video_jobs SET status = ?, progress = ?, started_at = COALESCE(started_at, ?), updated_at = ? WHERE id = ? AND cancel_requested = 0",
                params![status, progress, now, now, job_id],
            )
            .await?;
        Ok(())
    }

    pub async fn cancel_video_job(&self, user_id: &str, job_id: &str) -> anyhow::Result<bool> {
        let changed = self.conn
            .execute(
                "UPDATE video_jobs SET cancel_requested = 1, status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END, updated_at = ? WHERE id = ? AND user_id = ? AND status NOT IN ('complete','failed','cancelled')",
                params![now_ts(), job_id, user_id],
            )
            .await?;
        Ok(changed > 0)
    }

    pub async fn video_job(
        &self,
        user_id: &str,
        job_id: &str,
    ) -> anyhow::Result<Option<VideoJobStatus>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, status, progress, result_url, error, created_at, updated_at, effective_seed, r2_key, metadata_json FROM video_jobs WHERE id = ? AND user_id = ?",
                params![job_id, user_id],
            )
            .await?;
        Ok(if let Some(row) = rows.next().await? {
            Some(VideoJobStatus {
                job_id: row.get(0)?,
                status: row.get(1)?,
                progress: row.get::<f64>(2)? as f32,
                result_url: row.get(3)?,
                error: row.get(4)?,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
                effective_seed: row.get(7)?,
                r2_key: row.get(8)?,
                metadata: parse_json_value(row.get::<String>(9)?)?,
            })
        } else {
            None
        })
    }

    pub async fn list_video_jobs(
        &self,
        user_id: &str,
        limit: i64,
        offset: i64,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, mode, status, prompt, progress, result_url, error, created_at, updated_at, effective_seed, r2_key, metadata_json FROM video_jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params![user_id, limit, offset],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(serde_json::json!({
                "job_id": row.get::<String>(0)?,
                "mode": row.get::<String>(1)?,
                "status": row.get::<String>(2)?,
                "prompt": row.get::<String>(3)?,
                "progress": row.get::<f64>(4)?,
                "result_url": row.get::<Option<String>>(5)?,
                "error": row.get::<Option<String>>(6)?,
                "created_at": row.get::<i64>(7)?,
                "updated_at": row.get::<i64>(8)?,
                "effective_seed": row.get::<i64>(9)?,
                "r2_key": row.get::<String>(10)?,
                "metadata": parse_json_value(row.get::<String>(11)?)?,
            }));
        }
        Ok(out)
    }

    pub async fn recover_interrupted_video_jobs(&self) -> anyhow::Result<()> {
        self.conn
            .execute(
                "UPDATE video_jobs SET status = 'queued', progress = 0.0, updated_at = ? WHERE status IN ('running','materializing_inputs','generating','encoding','upscaling','uploading')",
                params![now_ts()],
            )
            .await?;
        Ok(())
    }

    pub async fn next_queued_video_job(
        &self,
    ) -> anyhow::Result<Option<(String, String, String, i64, String)>> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, params_json, user_id, effective_seed, r2_key FROM video_jobs WHERE status = 'queued' AND cancel_requested = 0 ORDER BY created_at ASC LIMIT 1",
                (),
            )
            .await?;
        Ok(if let Some(row) = rows.next().await? {
            Some((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
            ))
        } else {
            None
        })
    }
}

pub fn now_ts() -> i64 {
    Utc::now().timestamp()
}

fn chat_message_json(row: &libsql::Row) -> anyhow::Result<serde_json::Value> {
    Ok(serde_json::json!({
        "id": row.get::<String>(0)?,
        "session_id": row.get::<String>(1)?,
        "role": row.get::<String>(2)?,
        "content": row.get::<String>(3)?,
        "model_id": row.get::<Option<String>>(4)?,
        "created_at": row.get::<i64>(5)?,
    }))
}

fn is_identifier(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn parse_json_value(raw: String) -> anyhow::Result<Value> {
    Ok(serde_json::from_str(&raw).unwrap_or_else(|_| serde_json::json!({})))
}

#[cfg(test)]
mod tests {
    use std::{net::SocketAddr, path::PathBuf, time::Duration};

    use tempfile::tempdir;

    use super::*;
    use crate::models::VideoMode;

    fn config_for(path: &std::path::Path) -> Config {
        Config {
            bind_addr: "127.0.0.1:0".parse::<SocketAddr>().unwrap(),
            profile: "cloud_h200".into(),
            runtime_backend: "native".into(),
            compose_project: "test".into(),
            turso_db_url: format!("file:{}", path.display()),
            turso_auth_token: String::new(),
            admin_api_key: "admin-key".into(),
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

    async fn migrated_db() -> anyhow::Result<(tempfile::TempDir, Db)> {
        let dir = tempdir()?;
        let config = config_for(&dir.path().join("gateway.db"));
        let db = Db::connect(&config).await?;
        db.migrate().await?;
        Ok((dir, db))
    }

    #[tokio::test]
    async fn migration_table_is_idempotent() -> anyhow::Result<()> {
        let (_dir, db) = migrated_db().await?;
        db.migrate().await?;
        assert!(db.column_exists("users", "role").await?);
        assert!(db.column_exists("video_jobs", "metadata_json").await?);
        assert!(db.column_exists("video_jobs", "progress").await?);
        assert!(db.migration_applied("001_init").await?);
        assert!(db.migration_applied("002_user_roles").await?);
        assert!(db.migration_applied("003_video_metadata").await?);
        assert!(
            db.migration_applied("004_video_job_runtime_columns")
                .await?
        );
        assert!(db.migration_applied("005_video_job_core_columns").await?);
        Ok(())
    }

    #[tokio::test]
    async fn migration_repairs_legacy_video_jobs_without_progress() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let config = config_for(&dir.path().join("legacy.db"));
        let db = Db::connect(&config).await?;
        db.conn
            .execute_batch(
                "
                CREATE TABLE users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE video_jobs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    effective_seed INTEGER NOT NULL,
                    r2_key TEXT NOT NULL UNIQUE,
                    created_at INTEGER NOT NULL
                );
                ",
            )
            .await?;
        db.migrate().await?;
        for column in [
            "result_url",
            "error",
            "progress",
            "cancel_requested",
            "started_at",
            "updated_at",
            "completed_at",
            "metadata_json",
        ] {
            assert!(db.column_exists("video_jobs", column).await?, "{column}");
        }
        db.recover_interrupted_video_jobs().await?;
        Ok(())
    }

    #[tokio::test]
    async fn migration_repairs_legacy_video_jobs_without_params_json() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let config = config_for(&dir.path().join("legacy-no-params.db"));
        let db = Db::connect(&config).await?;
        db.conn
            .execute_batch(
                "
                CREATE TABLE schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at INTEGER NOT NULL
                );
                INSERT INTO schema_migrations (version, applied_at) VALUES
                    ('001_init', 1),
                    ('002_user_roles', 1),
                    ('003_video_metadata', 1),
                    ('004_video_job_runtime_columns', 1);
                CREATE TABLE users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE video_jobs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    effective_seed INTEGER NOT NULL,
                    r2_key TEXT NOT NULL UNIQUE,
                    result_url TEXT,
                    error TEXT,
                    progress REAL NOT NULL DEFAULT 0.0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    started_at INTEGER,
                    updated_at INTEGER NOT NULL,
                    completed_at INTEGER
                );
                INSERT INTO users (id, email, password_hash, role, created_at)
                    VALUES ('user-1', 'user@example.com', 'hash', 'user', 1);
                INSERT INTO video_jobs (id, user_id, mode, status, prompt, effective_seed, r2_key, progress, metadata_json, cancel_requested, created_at, updated_at)
                    VALUES ('job-1', 'user-1', 'text_to_video', 'queued', 'legacy prompt', 7, 'users/user-1/videos/job-1/output.mp4', 0.0, '{}', 0, 1, 1);
                ",
            )
            .await?;
        db.migrate().await?;
        assert!(db.column_exists("video_jobs", "params_json").await?);
        assert!(db.migration_applied("005_video_job_core_columns").await?);
        assert!(db.next_queued_video_job().await?.is_none());
        let job = db.video_job("user-1", "job-1").await?.unwrap();
        assert_eq!(job.status, "failed");
        assert!(job
            .error
            .unwrap_or_default()
            .contains("missing params_json"));
        Ok(())
    }

    #[tokio::test]
    async fn bootstrap_admin_promotes_first_existing_user() -> anyhow::Result<()> {
        let (_dir, db) = migrated_db().await?;
        let user = db.create_user("user@example.com", "hash", "user").await?;
        db.conn
            .execute(
                "DELETE FROM schema_migrations WHERE version = '002_user_roles'",
                (),
            )
            .await?;
        db.migrate().await?;
        let promoted = db.user_by_id(&user.id).await?.expect("user exists");
        assert_eq!(promoted.role, "admin");
        Ok(())
    }

    #[tokio::test]
    async fn user_role_update_and_list_are_consistent() -> anyhow::Result<()> {
        let (_dir, db) = migrated_db().await?;
        let user = db.create_user("person@example.com", "hash", "user").await?;
        assert!(db.update_user_role(&user.id, "admin").await?);
        assert!(!db.update_user_role("missing", "admin").await?);
        let listed = db.list_users().await?;
        assert!(listed
            .iter()
            .any(|item| { item["user_id"] == user.id && item["role"] == "admin" }));
        Ok(())
    }

    #[tokio::test]
    async fn video_job_persists_seed_r2_key_and_metadata() -> anyhow::Result<()> {
        let (_dir, db) = migrated_db().await?;
        let user = db.create_user("video@example.com", "hash", "user").await?;
        let job_id = Uuid::now_v7();
        let req = VideoJobRequest {
            mode: VideoMode::TextToVideo,
            prompt: "test prompt".into(),
            negative_prompt: None,
            width: 3840,
            height: 2160,
            num_frames: 121,
            frame_rate: Some(24.0),
            guidance_scale: Some(7.5),
            num_inference_steps: Some(40),
            seed_hint: None,
            image_url: None,
            video_url: None,
            audio_url: None,
            keyframe_urls: None,
            retake_start_time: None,
            retake_end_time: None,
            enhance_prompt: None,
            extra: None,
        };
        db.create_video_job(&user.id, &job_id, &req, 42, "users/u/videos/j/output.mp4")
            .await?;
        db.mark_video_stage(&job_id.to_string(), "generating", 0.25)
            .await?;
        let metadata = serde_json::json!({
            "upscaled": true,
            "render_width": 1024,
            "render_height": 576,
            "output_width": 3840,
            "output_height": 2160
        });
        db.update_video_job(
            &job_id.to_string(),
            "complete",
            1.0,
            Some("https://cdn.example/video.mp4"),
            None,
            Some(&metadata),
        )
        .await?;
        let status = db
            .video_job(&user.id, &job_id.to_string())
            .await?
            .expect("job exists");
        assert_eq!(status.effective_seed, 42);
        assert_eq!(status.r2_key, "users/u/videos/j/output.mp4");
        assert_eq!(status.metadata["upscaled"], true);
        assert_eq!(status.metadata["render_width"], 1024);
        Ok(())
    }
}
