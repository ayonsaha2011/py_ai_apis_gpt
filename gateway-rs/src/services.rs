use std::{collections::HashMap, fs::OpenOptions, process::Stdio, sync::Arc};

use async_trait::async_trait;
use tokio::{
    process::{Child, Command},
    sync::Mutex,
};

use crate::config::Config;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ServiceStatus {
    pub name: String,
    pub status: String,
    pub detail: Option<String>,
}

#[async_trait]
pub trait RuntimeBackend: Send + Sync {
    async fn start(&self, name: &str) -> anyhow::Result<ServiceStatus>;
    async fn stop(&self, name: &str) -> anyhow::Result<ServiceStatus>;
    async fn list(&self) -> anyhow::Result<Vec<ServiceStatus>>;
}

pub struct NativeBackend {
    config: Config,
    children: Arc<Mutex<HashMap<String, Child>>>,
}

impl NativeBackend {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            children: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn command_for(&self, name: &str) -> Option<&str> {
        match name {
            "text-worker" | "text" => Some(&self.config.native_text_command),
            "ltx-worker" | "ltx" => Some(&self.config.native_ltx_command),
            _ => None,
        }
    }

    fn canonical_name(name: &str) -> Option<&'static str> {
        match name {
            "text-worker" | "text" => Some("text"),
            "ltx-worker" | "ltx" => Some("ltx"),
            _ => None,
        }
    }

    fn workdir_for(&self, name: &str) -> Option<&'static str> {
        match name {
            "text-worker" | "text" => Some("services/text-worker"),
            "ltx-worker" | "ltx" => Some("services/ltx-worker"),
            _ => None,
        }
    }
}

#[async_trait]
impl RuntimeBackend for NativeBackend {
    async fn start(&self, name: &str) -> anyhow::Result<ServiceStatus> {
        let Some(command) = self.command_for(name) else {
            return Ok(ServiceStatus {
                name: name.into(),
                status: "unknown".into(),
                detail: Some("no native command configured".into()),
            });
        };
        let canonical_name = Self::canonical_name(name).unwrap_or(name);
        let mut guard = self.children.lock().await;
        if guard.contains_key(name) {
            return Ok(ServiceStatus {
                name: name.into(),
                status: "running".into(),
                detail: None,
            });
        }
        let root = std::env::current_dir()?;
        let service_dir = self.workdir_for(name).map(|d| root.join(d));
        let cache_dir = root.join(".uv-cache");
        tokio::fs::create_dir_all(&cache_dir).await?;
        tokio::fs::create_dir_all(&self.config.log_dir).await?;
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.config.log_dir.join(format!("{canonical_name}.log")))?;
        let child = if cfg!(windows) {
            let mut cmd = Command::new("powershell");
            cmd.args(["-NoProfile", "-Command", command])
                .env("UV_CACHE_DIR", &cache_dir)
                .stdin(Stdio::null())
                .stdout(Stdio::from(log_file.try_clone()?))
                .stderr(Stdio::from(log_file));
            if let Some(dir) = service_dir {
                cmd.current_dir(dir);
            }
            cmd.spawn()?
        } else {
            let mut cmd = Command::new("sh");
            cmd.args(["-lc", command])
                .env("UV_CACHE_DIR", &cache_dir)
                .stdin(Stdio::null())
                .stdout(Stdio::from(log_file.try_clone()?))
                .stderr(Stdio::from(log_file));
            if let Some(dir) = service_dir {
                cmd.current_dir(dir);
            }
            cmd.spawn()?
        };
        guard.insert(name.into(), child);
        Ok(ServiceStatus {
            name: name.into(),
            status: "running".into(),
            detail: None,
        })
    }

    async fn stop(&self, name: &str) -> anyhow::Result<ServiceStatus> {
        let mut guard = self.children.lock().await;
        if let Some(mut child) = guard.remove(name) {
            child.kill().await?;
            Ok(ServiceStatus {
                name: name.into(),
                status: "stopped".into(),
                detail: None,
            })
        } else {
            Ok(ServiceStatus {
                name: name.into(),
                status: "not_running".into(),
                detail: None,
            })
        }
    }

    async fn list(&self) -> anyhow::Result<Vec<ServiceStatus>> {
        let guard = self.children.lock().await;
        Ok(["text-worker", "ltx-worker"]
            .iter()
            .map(|name| ServiceStatus {
                name: (*name).into(),
                status: if guard.contains_key(*name) {
                    "running"
                } else {
                    "not_running"
                }
                .into(),
                detail: None,
            })
            .collect())
    }
}

pub struct DockerBackend {
    project: String,
}

impl DockerBackend {
    pub fn new(project: String) -> Self {
        Self { project }
    }

    async fn compose(&self, args: &[&str]) -> anyhow::Result<ServiceStatus> {
        let output = Command::new("docker")
            .args(["compose", "-p", &self.project])
            .args(args)
            .output()
            .await?;
        Ok(ServiceStatus {
            name: args.last().copied().unwrap_or("compose").into(),
            status: if output.status.success() {
                "ok"
            } else {
                "failed"
            }
            .into(),
            detail: Some(
                String::from_utf8_lossy(if output.status.success() {
                    &output.stdout
                } else {
                    &output.stderr
                })
                .to_string(),
            ),
        })
    }
}

#[async_trait]
impl RuntimeBackend for DockerBackend {
    async fn start(&self, name: &str) -> anyhow::Result<ServiceStatus> {
        self.compose(&["up", "-d", name]).await
    }

    async fn stop(&self, name: &str) -> anyhow::Result<ServiceStatus> {
        self.compose(&["stop", name]).await
    }

    async fn list(&self) -> anyhow::Result<Vec<ServiceStatus>> {
        let status = self.compose(&["ps"]).await?;
        Ok(vec![status])
    }
}
