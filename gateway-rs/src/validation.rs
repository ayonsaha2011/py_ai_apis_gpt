use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{
    error::{AppError, Result},
    models::{VideoJobRequest, VideoMode},
};

pub fn validate_video_request(req: &VideoJobRequest, profile: &str) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(AppError::BadRequest("prompt is required".into()));
    }
    if req.width % 32 != 0 || req.height % 32 != 0 {
        return Err(AppError::BadRequest(
            "width and height must be divisible by 32".into(),
        ));
    }
    if (req.num_frames - 1) % 8 != 0 {
        return Err(AppError::BadRequest("num_frames must satisfy 8k+1".into()));
    }
    let max_side = if profile.contains("h200") { 1536 } else { 1024 };
    if req.width > max_side || req.height > max_side {
        return Err(AppError::BadRequest(format!(
            "width/height exceed profile limit {max_side}"
        )));
    }
    if req.num_frames > 257 {
        return Err(AppError::BadRequest("num_frames exceeds 257".into()));
    }
    match req.mode {
        VideoMode::TextToVideo | VideoMode::Distilled => {}
        VideoMode::ImageToVideo => require_url("image_url", req.image_url.as_deref())?,
        VideoMode::VideoToVideo | VideoMode::Hdr => {
            require_url("video_url", req.video_url.as_deref())?
        }
        VideoMode::AudioToVideo => require_url("audio_url", req.audio_url.as_deref())?,
        VideoMode::KeyframeInterpolation => {
            if req.keyframe_urls.as_ref().map_or(0, Vec::len) < 2 {
                return Err(AppError::BadRequest(
                    "keyframe_interpolation requires at least two keyframe_urls".into(),
                ));
            }
        }
        VideoMode::Retake => {
            require_url("video_url", req.video_url.as_deref())?;
            let start = req
                .retake_start_time
                .ok_or_else(|| AppError::BadRequest("retake_start_time is required".into()))?;
            let end = req
                .retake_end_time
                .ok_or_else(|| AppError::BadRequest("retake_end_time is required".into()))?;
            if end <= start {
                return Err(AppError::BadRequest(
                    "retake_end_time must be greater than retake_start_time".into(),
                ));
            }
        }
    }
    for url in req
        .image_url
        .iter()
        .chain(req.video_url.iter())
        .chain(req.audio_url.iter())
    {
        reject_private_url(url)?;
    }
    if let Some(urls) = &req.keyframe_urls {
        for url in urls {
            reject_private_url(url)?;
        }
    }
    Ok(())
}

pub fn effective_seed(user_id: &str, job_id: &Uuid, seed_hint: Option<u64>, prompt: &str) -> i64 {
    let mut h = Sha256::new();
    h.update(user_id.as_bytes());
    h.update(job_id.as_bytes());
    h.update(seed_hint.unwrap_or(0).to_le_bytes());
    h.update(prompt.as_bytes());
    let bytes = h.finalize();
    i64::from_le_bytes(bytes[0..8].try_into().expect("slice length")) & 0x7fff_ffff
}

pub fn video_r2_key(user_id: &str, job_id: &Uuid) -> String {
    format!("users/{user_id}/videos/{job_id}/output.mp4")
}

fn require_url(name: &str, value: Option<&str>) -> Result<()> {
    if value.is_some_and(|v| !v.trim().is_empty()) {
        Ok(())
    } else {
        Err(AppError::BadRequest(format!("{name} is required")))
    }
}

fn reject_private_url(url: &str) -> Result<()> {
    let lower = url.to_ascii_lowercase();
    let blocked = [
        "localhost",
        "127.",
        "10.",
        "172.16.",
        "172.17.",
        "172.18.",
        "172.19.",
        "172.20.",
        "192.168.",
        "[::1]",
    ];
    if blocked.iter().any(|needle| lower.contains(needle)) {
        return Err(AppError::BadRequest(
            "private network media URLs are not accepted".into(),
        ));
    }
    if !(lower.starts_with("https://") || lower.starts_with("s3://") || lower.starts_with("r2://"))
    {
        return Err(AppError::BadRequest(
            "media URLs must use https, s3, or r2".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::VideoMode;

    #[test]
    fn rejects_bad_frame_count() {
        let req = VideoJobRequest {
            mode: VideoMode::TextToVideo,
            prompt: "x".into(),
            negative_prompt: None,
            width: 768,
            height: 512,
            num_frames: 100,
            frame_rate: None,
            guidance_scale: None,
            num_inference_steps: None,
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
        assert!(validate_video_request(&req, "local_rtx_5090").is_err());
    }
}
