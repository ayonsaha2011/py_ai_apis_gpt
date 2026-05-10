CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    expires_at INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    revoked_at INTEGER,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    revoked_at INTEGER,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_id TEXT PRIMARY KEY,
    worker_kind TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS service_instances (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    url TEXT NOT NULL,
    gpu_id INTEGER,
    status TEXT NOT NULL,
    started_at INTEGER,
    stopped_at INTEGER,
    last_heartbeat_at INTEGER,
    config_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS gpu_leases (
    id TEXT PRIMARY KEY,
    gpu_id INTEGER NOT NULL,
    holder TEXT NOT NULL,
    lease_kind TEXT NOT NULL,
    acquired_at INTEGER NOT NULL,
    expires_at INTEGER,
    released_at INTEGER
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model_id TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY(session_id) REFERENCES chat_sessions(id),
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS rag_collections (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    qdrant_name TEXT NOT NULL UNIQUE,
    created_at INTEGER NOT NULL,
    UNIQUE(user_id, name),
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS rag_documents (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    deleted_at INTEGER,
    FOREIGN KEY(collection_id) REFERENCES rag_collections(id),
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS video_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    prompt TEXT NOT NULL,
    params_json TEXT NOT NULL,
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
    completed_at INTEGER,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    job_id TEXT,
    kind TEXT NOT NULL,
    r2_key TEXT NOT NULL UNIQUE,
    public_url TEXT,
    content_type TEXT,
    bytes INTEGER,
    created_at INTEGER NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(job_id) REFERENCES video_jobs(id)
);

CREATE TABLE IF NOT EXISTS audit_events (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    event_type TEXT NOT NULL,
    target TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_user_session ON chat_messages(user_id, session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_video_user_created ON video_jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_video_status ON video_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_events(created_at DESC);
