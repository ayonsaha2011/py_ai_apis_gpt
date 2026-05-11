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
    if !matches!(req.mode, VideoMode::TextToVideo | VideoMode::ImageToVideo) {
        return Err(AppError::BadRequest(
            "this deployment allows only text_to_video and image_to_video so the LTX worker keeps exactly one full dev model on GPU".into(),
        ));
    }
    if (req.num_frames - 1) % 8 != 0 {
        return Err(AppError::BadRequest("num_frames must satisfy 8k+1".into()));
    }
    let budget = ltx_budget(profile);
    let output_pixels = req.width as u64 * req.height as u64;
    let pixel_frames = output_pixels * req.num_frames as u64;
    let native_request = req.width <= budget.max_native_side
        && req.height <= budget.max_native_side
        && req.num_frames <= budget.max_frames
        && pixel_frames <= budget.max_pixel_frames
        && req.width % 32 == 0
        && req.height % 32 == 0;
    let upscaled_request = !native_request
        && budget.max_output_side > budget.max_native_side
        && req.num_frames <= budget.max_upscaled_frames;
    if req.width % 32 != 0 || req.height % 32 != 0 {
        if !upscaled_request || req.width % 2 != 0 || req.height % 2 != 0 {
            return Err(AppError::BadRequest(
                "width and height must be divisible by 32 for native generation, or even for upscaled output".into(),
            ));
        }
    }
    let max_side = budget.max_output_side;
    if req.width > max_side || req.height > max_side {
        return Err(AppError::BadRequest(format!(
            "width/height exceed 4K output limit {max_side}"
        )));
    }
    if output_pixels > budget.max_output_pixels {
        return Err(AppError::BadRequest(format!(
            "width/height exceed 4K output pixel limit {}",
            budget.max_output_pixels
        )));
    }
    if let Some(steps) = req.num_inference_steps {
        if !(1..=40).contains(&steps) {
            return Err(AppError::BadRequest(
                "num_inference_steps must be between 1 and 40".into(),
            ));
        }
    }
    let max_frame_limit = budget.max_frames.max(budget.max_upscaled_frames);
    if req.num_frames > max_frame_limit {
        let requested_seconds = frame_seconds(req.num_frames, req.frame_rate);
        let allowed_seconds = frame_seconds(max_frame_limit, req.frame_rate);
        return Err(AppError::BadRequest(format!(
            "num_frames exceeds {max_frame_limit} for {}; requested {} frames (~{requested_seconds:.1}s), allowed {max_frame_limit} frames (~{allowed_seconds:.1}s). {}",
            budget.label, req.num_frames, budget.guidance
        )));
    }
    if !native_request {
        if req.num_frames > budget.max_upscaled_frames
            || budget.max_output_side <= budget.max_native_side
        {
            return Err(AppError::BadRequest(format!(
                "request exceeds {} memory budget ({} pixel-frames > {}); {}",
                budget.label, pixel_frames, budget.max_pixel_frames, budget.guidance
            )));
        }
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
    max_native_side: u32,
    max_frames: u32,
    max_pixel_frames: u64,
    max_output_side: u32,
    max_output_pixels: u64,
    max_upscaled_frames: u32,
    label: &'static str,
    guidance: &'static str,
}

fn ltx_budget(profile: &str) -> LtxBudget {
    let profile = profile.to_ascii_lowercase();
    let h200 = profile.contains("h200");
    let h100 = profile.contains("h100");

    if h200 {
        return LtxBudget {
            max_native_side: 1536,
            max_frames: 121,
            max_pixel_frames: 1024 * 576 * 121,
            max_output_side: 4096,
            max_output_pixels: 4096 * 2160,
            max_upscaled_frames: 121,
            label: "H200 full 22B bf16 LTX",
            guidance:
                "use 5 seconds at 1024x576 per job; 20s HD requires chunked/stitch generation, which is not enabled while the worker keeps only one full dev model on GPU",
        };
    }
    if h100 {
        return LtxBudget {
            max_native_side: 1024,
            max_frames: 121,
            max_pixel_frames: 768 * 448 * 121,
            max_output_side: 4096,
            max_output_pixels: 4096 * 2160,
            max_upscaled_frames: 121,
            label: "H100 full 22B bf16 LTX",
            guidance: "use 5 seconds at 768x448 per job, or use H200 for larger 5-second jobs",
        };
    }
    LtxBudget {
        max_native_side: 1024,
        max_frames: 121,
        max_pixel_frames: 768 * 448 * 121,
        max_output_side: 1024,
        max_output_pixels: 1024 * 1024,
        max_upscaled_frames: 121,
        label: "local full 22B LTX",
        guidance:
            "use 5 seconds at 768x448, or switch to the H200 profile for larger 5-second jobs",
    }
}

fn frame_seconds(frames: u32, frame_rate: Option<f32>) -> f32 {
    let fps = frame_rate.unwrap_or(24.0).max(1.0);
    frames.saturating_sub(1) as f32 / fps
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
    fn rejects_h200_distilled_mode() {
        let req = base_request(VideoMode::Distilled, 1024, 576, 241);
        let err = validate_video_request(&req, "cloud_h200").unwrap_err();
        assert!(err.to_string().contains("exactly one full dev model"));
    }

    #[test]
    fn accepts_h100_full_22b_5s_sd_budget() {
        let req = base_request(VideoMode::TextToVideo, 768, 448, 121);
        assert!(validate_video_request(&req, "cloud_h100").is_ok());
    }

    #[test]
    fn accepts_h100_full_22b_5s_hd_as_upscaled_budget() {
        let req = base_request(VideoMode::TextToVideo, 1024, 576, 121);
        assert!(validate_video_request(&req, "cloud_h100").is_ok());
    }

    #[test]
    fn rejects_h100_distilled_mode() {
        let req = base_request(VideoMode::Distilled, 1024, 576, 121);
        let err = validate_video_request(&req, "cloud_h100").unwrap_err();
        assert!(err.to_string().contains("exactly one full dev model"));
    }

    #[test]
    fn accepts_h200_4k_upscaled_5s_budget() {
        let req = base_request(VideoMode::TextToVideo, 3840, 2160, 121);
        assert!(validate_video_request(&req, "cloud_h200").is_ok());
    }

    #[test]
    fn rejects_h200_4k_upscaled_10s_full_budget() {
        let req = base_request(VideoMode::TextToVideo, 3840, 2160, 241);
        assert!(validate_video_request(&req, "cloud_h200").is_err());
    }

    #[test]
    fn accepts_h100_4k_upscaled_5s_budget() {
        let req = base_request(VideoMode::TextToVideo, 3840, 2160, 121);
        assert!(validate_video_request(&req, "cloud_h100").is_ok());
    }

    #[test]
    fn rejects_square_4096_output_over_4k_pixels() {
        let req = base_request(VideoMode::TextToVideo, 4096, 4096, 121);
        assert!(validate_video_request(&req, "cloud_h200").is_err());
    }

    #[test]
    fn rejects_distilled_when_full_dev_only() {
        let req = base_request(VideoMode::Distilled, 768, 448, 121);
        let err = validate_video_request(&req, "cloud_h200").unwrap_err();
        assert!(err.to_string().contains("exactly one full dev model"));
    }
}
