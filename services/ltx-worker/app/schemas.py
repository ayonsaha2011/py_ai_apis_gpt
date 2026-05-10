from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class VideoMode(StrEnum):
    text_to_video = "text_to_video"
    image_to_video = "image_to_video"
    video_to_video = "video_to_video"
    audio_to_video = "audio_to_video"
    keyframe_interpolation = "keyframe_interpolation"
    retake = "retake"
    distilled = "distilled"
    hdr = "hdr"


class VideoRequest(BaseModel):
    mode: VideoMode
    prompt: str
    negative_prompt: str | None = ""
    width: int
    height: int
    num_frames: int
    frame_rate: float | None = 24.0
    guidance_scale: float | None = 7.5
    num_inference_steps: int | None = 40
    seed_hint: int | None = None
    image_url: str | None = None
    video_url: str | None = None
    audio_url: str | None = None
    keyframe_urls: list[str] | None = None
    retake_start_time: float | None = None
    retake_end_time: float | None = None
    enhance_prompt: bool | None = False
    extra: dict[str, Any] | None = None


class InternalJobRequest(BaseModel):
    job_id: str
    request: VideoRequest
    effective_seed: int
    r2_key: str

