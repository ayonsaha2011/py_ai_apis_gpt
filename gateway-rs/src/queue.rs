use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::{AppError, Result};

#[derive(Clone)]
pub struct Admission {
    text_heavy: Arc<Semaphore>,
    video_heavy: Arc<Semaphore>,
    text_waiting: Arc<AtomicUsize>,
    video_waiting: Arc<AtomicUsize>,
    text_max_waiting: usize,
    video_max_waiting: usize,
}

impl Admission {
    pub fn new(
        text_heavy_jobs: usize,
        video_heavy_jobs: usize,
        text_max_waiting: usize,
        video_max_waiting: usize,
    ) -> Self {
        Self {
            text_heavy: Arc::new(Semaphore::new(text_heavy_jobs.max(1))),
            video_heavy: Arc::new(Semaphore::new(video_heavy_jobs.max(1))),
            text_waiting: Arc::new(AtomicUsize::new(0)),
            video_waiting: Arc::new(AtomicUsize::new(0)),
            text_max_waiting,
            video_max_waiting,
        }
    }

    pub fn admit_text_waiter(&self) -> Result<WaitGuard> {
        self.admit(
            &self.text_waiting,
            self.text_max_waiting,
            "text queue is full",
        )
    }

    pub fn admit_video_waiter(&self) -> Result<WaitGuard> {
        self.admit(
            &self.video_waiting,
            self.video_max_waiting,
            "video queue is full",
        )
    }

    pub async fn acquire_text_heavy(&self) -> Result<OwnedSemaphorePermit> {
        Ok(self
            .text_heavy
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| AppError::Unavailable("text workload gate closed".into()))?)
    }

    pub async fn acquire_video_heavy(&self) -> Result<OwnedSemaphorePermit> {
        Ok(self
            .video_heavy
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| AppError::Unavailable("video workload gate closed".into()))?)
    }

    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "text_waiting": self.text_waiting.load(Ordering::SeqCst),
            "video_waiting": self.video_waiting.load(Ordering::SeqCst),
            "text_heavy_available": self.text_heavy.available_permits(),
            "video_heavy_available": self.video_heavy.available_permits(),
        })
    }

    fn admit(&self, counter: &Arc<AtomicUsize>, max: usize, message: &str) -> Result<WaitGuard> {
        let prev = counter.fetch_add(1, Ordering::SeqCst);
        if prev >= max {
            counter.fetch_sub(1, Ordering::SeqCst);
            return Err(AppError::Unavailable(message.into()));
        }
        Ok(WaitGuard {
            counter: counter.clone(),
        })
    }
}

pub struct WaitGuard {
    counter: Arc<AtomicUsize>,
}

impl Drop for WaitGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
    }
}
