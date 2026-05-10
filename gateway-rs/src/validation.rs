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
    let budget = ltx_budget(req, profile);
    let max_side = budget.max_side;
    if req.width > max_side || req.height > max_side {
        return Err(AppError::BadRequest(format!(
            "width/height exceed profile limit {max_side}"
        )));
    }
    if let Some(steps) = req.num_inference_steps {
        if !(1..=40).contains(&steps) {
            return Err(AppError::BadRequest(
                "num_inference_steps must be between 1 and 40".into(),
            ));
        }
    }
    if req.num_frames > budget.max_frames {
        return Err(AppError::BadRequest(format!(
            "num_frames exceeds {} for {}; {}",
            budget.max_frames, budget.label, budget.guidance
        )));
    }
    let pixel_frames = req.width as u64 * req.height as u64 * req.num_frames as u64;
    if pixel_frames > budget.max_pixel_frames {
        return Err(AppError::BadRequest(format!(
            "request exceeds {} memory budget ({} pixel-frames > {}); {}",
            budget.label, pixel_frames, budget.max_pixel_frames, budget.guidance
        )));
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

struct LtxBudget {
    max_side: u32,
    max_frames: u32,
    max_pixel_frames: u64,
    label: &'static str,
    guidance: &'static str,
}

fn ltx_budget(req: &VideoJobRequest, profile: &str) -> LtxBudget {
    let profile = profile.to_ascii_lowercase();
    let h200 = profile.contains("h200");
    let h100 = profile.contains("h100");
    let distilled_like = matches!(
        req.mode,
        VideoMode::Distilled | VideoMode::VideoToVideo | VideoMode::Hdr
    );

    if h200 && distilled_like {
        return LtxBudget {
            max_side: 1536,
            max_frames: 241,
            max_pixel_frames: 1024 * 576 * 241,
            label: "H200 distilled LTX",
            guidance: "use up to 10 seconds at 1024x576, or reduce resolution for longer clips",
        };
    }
    if h200 {
        return LtxBudget {
            max_side: 1536,
            max_frames: 121,
            max_pixel_frames: 1024 * 576 * 121,
            label: "H200 full 22B bf16 LTX",
            guidance: "use 5 seconds at 1024x576 for full 22B bf16; use distilled or reduce resolution for longer clips",
        };
    }
    if h100 && distilled_like {
        return LtxBudget {
            max_side: 1024,
            max_frames: 121,
            max_pixel_frames: 1024 * 576 * 121,
            label: "H100 distilled LTX",
            guidance: "use 5 seconds at 1024x576, or switch to H200 for longer HD clips",
        };
    }
    if h100 {
        return LtxBudget {
            max_side: 1024,
            max_frames: 121,
            max_pixel_frames: 768 * 448 * 121,
            label: "H100 full 22B bf16 LTX",
            guidance:
                "use 5 seconds at 768x448 for full 22B bf16; use distilled or H200 for larger clips",
        };
    }
    if distilled_like {
        return LtxBudget {
            max_side: 1024,
            max_frames: 121,
            max_pixel_frames: 1024 * 576 * 121,
            label: "local distilled LTX",
            guidance: "use 5 seconds at 1024x576, or reduce resolution for longer clips",
        };
    }
    LtxBudget {
        max_side: 1024,
        max_frames: 121,
        max_pixel_frames: 768 * 448 * 121,
        label: "local full 22B LTX",
        guidance: "use 5 seconds at 768x448, or switch to the H200 profile for larger jobs",
    }
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

    fn base_request(mode: VideoMode, width: u32, height: u32, frames: u32) -> VideoJobRequest {
        VideoJobRequest {
            mode,
            prompt: "x".into(),
            negative_prompt: None,
            width,
            height,
            num_frames: frames,
            frame_rate: None,
            guidance_scale: None,
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
        }
    }

    #[test]
    fn accepts_h200_full_22b_5s_hd_budget() {
        let req = base_request(VideoMode::TextToVideo, 1024, 576, 121);
        assert!(validate_video_request(&req, "cloud_h200").is_ok());
    }

    #[test]
    fn rejects_h200_full_22b_10s_hd_budget() {
        let req = base_request(VideoMode::TextToVideo, 1024, 576, 241);
        let err = validate_video_request(&req, "cloud_h200").unwrap_err();
        assert!(err.to_string().contains("H200 full 22B bf16"));
    }

    #[test]
    fn accepts_h200_distilled_10s_hd_budget() {
        let req = base_request(VideoMode::Distilled, 1024, 576, 241);
        assert!(validate_video_request(&req, "cloud_h200").is_ok());
    }

    #[test]
    fn accepts_h100_full_22b_5s_sd_budget() {
        let req = base_request(VideoMode::TextToVideo, 768, 448, 121);
        assert!(validate_video_request(&req, "cloud_h100").is_ok());
    }

    #[test]
    fn rejects_h100_full_22b_5s_hd_budget() {
        let req = base_request(VideoMode::TextToVideo, 1024, 576, 121);
        let err = validate_video_request(&req, "cloud_h100").unwrap_err();
        assert!(err.to_string().contains("H100 full 22B bf16"));
    }

    #[test]
    fn accepts_h100_distilled_5s_hd_budget() {
        let req = base_request(VideoMode::Distilled, 1024, 576, 121);
        assert!(validate_video_request(&req, "cloud_h100").is_ok());
    }
}
