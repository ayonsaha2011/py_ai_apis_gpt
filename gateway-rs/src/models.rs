use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub email: String,
    pub role: String,
    pub created_at: i64,
}

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: &'static str,
    pub expires_in: u64,
    pub role: String,
}

#[derive(Debug, Serialize)]
pub struct MeResponse {
    pub user_id: String,
    pub email: String,
    pub role: String,
    pub created_at: i64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
    pub session_id: Option<String>,
    pub use_rag: Option<bool>,
    pub rag_collection: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateRagCollectionRequest {
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RagIngestRequest {
    pub collection: String,
    pub texts: Vec<String>,
    pub source_name: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VideoMode {
    TextToVideo,
    ImageToVideo,
    VideoToVideo,
    AudioToVideo,
    KeyframeInterpolation,
    Retake,
    Distilled,
    Hdr,
}

impl VideoMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TextToVideo => "text_to_video",
            Self::ImageToVideo => "image_to_video",
            Self::VideoToVideo => "video_to_video",
            Self::AudioToVideo => "audio_to_video",
            Self::KeyframeInterpolation => "keyframe_interpolation",
            Self::Retake => "retake",
            Self::Distilled => "distilled",
            Self::Hdr => "hdr",
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct VideoJobRequest {
    pub mode: VideoMode,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    pub num_frames: u32,
    pub frame_rate: Option<f32>,
    pub guidance_scale: Option<f32>,
    pub num_inference_steps: Option<u32>,
    pub seed_hint: Option<u64>,
    pub image_url: Option<String>,
    pub video_url: Option<String>,
    pub audio_url: Option<String>,
    pub keyframe_urls: Option<Vec<String>>,
    pub retake_start_time: Option<f32>,
    pub retake_end_time: Option<f32>,
    pub enhance_prompt: Option<bool>,
    pub extra: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VideoJobStatus {
    pub job_id: String,
    pub status: String,
    pub progress: f32,
    pub result_url: Option<String>,
    pub error: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Serialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub total_mb: u64,
    pub used_mb: u64,
    pub free_mb: u64,
    pub utilization_pct: u32,
}
