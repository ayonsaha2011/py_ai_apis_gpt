use std::sync::Arc;

use chrono::Utc;
use libsql::{params, Builder, Connection};
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
        self.ensure_user_role_column().await?;
        self.ensure_bootstrap_admin().await?;
        Ok(())
    }

    async fn ensure_user_role_column(&self) -> anyhow::Result<()> {
        let mut rows = self.conn.query("PRAGMA table_info(users)", ()).await?;
        let mut has_role = false;
        while let Some(row) = rows.next().await? {
            let name: String = row.get(1)?;
            if name == "role" {
                has_role = true;
                break;
            }
        }
        if !has_role {
            self.conn
                .execute(
                    "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'",
                    (),
                )
                .await?;
        }
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
        let id = Uuid::now_v7().to_string();
        self.conn
            .execute(
                "INSERT INTO rag_documents (id, collection_id, user_id, source_name, metadata_json, chunk_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![id.clone(), collection_id, user_id, source_name, metadata_json, chunk_count, now_ts()],
            )
            .await?;
        Ok(id)
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
        self.conn
            .execute(
                "INSERT INTO video_jobs (id, user_id, mode, status, prompt, params_json, effective_seed, r2_key, created_at, updated_at) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?)",
                params![
                    job_id.to_string(),
                    user_id,
                    req.mode.as_str(),
                    req.prompt.clone(),
                    serde_json::to_string(req)?,
                    effective_seed,
                    r2_key,
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
                "SELECT COUNT(*) FROM video_jobs WHERE status IN ('queued','running')",
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
    ) -> anyhow::Result<()> {
        let now = now_ts();
        let completed = if matches!(status, "complete" | "failed" | "cancelled") {
            Some(now)
        } else {
            None
        };
        self.conn
            .execute(
                "UPDATE video_jobs SET status = ?, progress = ?, result_url = COALESCE(?, result_url), error = ?, updated_at = ?, completed_at = COALESCE(?, completed_at) WHERE id = ?",
                params![status, progress, result_url, error, now, completed, job_id],
            )
            .await?;
        Ok(())
    }

    pub async fn mark_video_started(&self, job_id: &str) -> anyhow::Result<()> {
        let now = now_ts();
        self.conn
            .execute(
                "UPDATE video_jobs SET status = 'running', started_at = COALESCE(started_at, ?), updated_at = ? WHERE id = ? AND cancel_requested = 0",
                params![now, now, job_id],
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
                "SELECT id, status, progress, result_url, error, created_at, updated_at FROM video_jobs WHERE id = ? AND user_id = ?",
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
                "SELECT id, mode, status, prompt, progress, result_url, error, created_at, updated_at FROM video_jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
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
            }));
        }
        Ok(out)
    }

    pub async fn recover_interrupted_video_jobs(&self) -> anyhow::Result<()> {
        self.conn
            .execute(
                "UPDATE video_jobs SET status = 'queued', progress = 0.0, updated_at = ? WHERE status = 'running'",
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
