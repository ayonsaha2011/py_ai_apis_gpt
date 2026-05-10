from __future__ import annotations

import asyncio
import gc
import math
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from .config import settings
from .media import fetch_media
from .schemas import VideoMode, VideoRequest


def _files() -> dict[str, str]:
    return {p.name: str(p) for p in settings.ltx_model_dir.glob("*.safetensors")}


def _pick(*names: str) -> str:
    files = _files()
    for name in names:
        if name in files:
            return files[name]
    raise FileNotFoundError(f"missing required LTX asset; tried: {', '.join(names)} in {settings.ltx_model_dir}")


def _gemma_root() -> str:
    return str(settings.gemma_root or (settings.ltx_model_dir / "gemma-3-12b"))


def _quantization() -> dict[str, Any]:
    if settings.quantization == "none":
        return {}
    from ltx_core.quantization import QuantizationPolicy

    if settings.quantization == "fp8_cast":
        return {"quantization": QuantizationPolicy.fp8_cast()}
    if settings.quantization == "fp8_scaled_mm":
        return {"quantization": QuantizationPolicy.fp8_scaled_mm()}
    raise ValueError(f"unsupported quantization {settings.quantization}")


def _distilled_lora() -> list[Any]:
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

    path = _pick("ltx-2.3-22b-distilled-lora-384-1.1.safetensors", "ltx-2.3-22b-distilled-lora-384.safetensors")
    return [LoraPathStrengthAndSDOps(path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]


def _device() -> torch.device:
    ensure_runtime_ready()
    return torch.device(settings.cuda_device)


def runtime_status() -> dict[str, Any]:
    return {
        "cuda_requested": settings.cuda_device.startswith("cuda"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": settings.cuda_device,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "CUDA out of memory" in str(exc)


def release_cuda_memory(clear_pipelines: bool = False) -> None:
    if clear_pipelines:
        _two_stage_pipeline.cache_clear()
        _distilled_pipeline.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def ensure_runtime_ready() -> None:
    if settings.cuda_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for LTX generation but this worker loaded a CPU-only PyTorch runtime. "
            "Install the locked CUDA PyTorch environment with Python 3.12, then restart the worker."
    )


def validate_ltx_budget(req: VideoRequest) -> None:
    if (req.num_frames - 1) % 8 != 0:
        raise ValueError("num_frames must satisfy 8k+1")
    if req.num_inference_steps is not None and not 1 <= req.num_inference_steps <= 40:
        raise ValueError("num_inference_steps must be between 1 and 40")

    budget = _ltx_budget(req)
    native_request = _native_request_fits(req, budget)
    upscaled_request = (
        not native_request
        and budget["max_output_side"] > budget["max_native_side"]
        and req.num_frames <= budget["max_upscaled_frames"]
    )
    if req.width % 32 != 0 or req.height % 32 != 0:
        if not upscaled_request or req.width % 2 != 0 or req.height % 2 != 0:
            raise ValueError(
                "width and height must be divisible by 32 for native generation, or even for 4K upscaled output"
            )
    if req.width > budget["max_output_side"] or req.height > budget["max_output_side"]:
        raise ValueError(f"width/height exceed 4K output limit {budget['max_output_side']}")
    output_pixels = req.width * req.height
    if output_pixels > budget["max_output_pixels"]:
        raise ValueError(f"width/height exceed 4K output pixel limit {budget['max_output_pixels']}")
    max_frame_limit = max(budget["max_frames"], budget["max_upscaled_frames"])
    if req.num_frames > max_frame_limit:
        raise ValueError(f"num_frames exceeds {max_frame_limit} for {budget['label']}; {budget['guidance']}")
    pixel_frames = req.width * req.height * req.num_frames
    if not native_request and (
        req.num_frames > budget["max_upscaled_frames"] or budget["max_output_side"] <= budget["max_native_side"]
    ):
        raise ValueError(
            f"request exceeds {budget['label']} memory budget ({pixel_frames} pixel-frames > {budget['max_pixel_frames']}); {budget['guidance']}"
        )


def _ltx_budget(req: VideoRequest) -> dict[str, Any]:
    profile = settings.gpu_profile.lower()
    h200 = "h200" in profile
    h100 = "h100" in profile
    distilled_like = req.mode in {VideoMode.distilled, VideoMode.video_to_video, VideoMode.hdr}
    if h200 and distilled_like:
        return {
            "max_native_side": 1536,
            "max_frames": 241,
            "max_pixel_frames": 1024 * 576 * 241,
            "max_output_side": 4096,
            "max_output_pixels": 4096 * 2160,
            "max_upscaled_frames": 121,
            "label": "H200 distilled LTX",
            "guidance": "use up to 10 seconds at 1024x576, or reduce resolution for longer clips",
        }
    if h200:
        return {
            "max_native_side": 1536,
            "max_frames": 121,
            "max_pixel_frames": 1024 * 576 * 121,
            "max_output_side": 4096,
            "max_output_pixels": 4096 * 2160,
            "max_upscaled_frames": 121,
            "label": "H200 full 22B bf16 LTX",
            "guidance": "use 5 seconds at 1024x576 for full 22B bf16; use distilled or reduce resolution for longer clips",
        }
    if h100 and distilled_like:
        return {
            "max_native_side": 1024,
            "max_frames": 121,
            "max_pixel_frames": 1024 * 576 * 121,
            "max_output_side": 4096,
            "max_output_pixels": 4096 * 2160,
            "max_upscaled_frames": 121,
            "label": "H100 distilled LTX",
            "guidance": "use 5 seconds at 1024x576, or switch to H200 for longer HD clips",
        }
    if h100:
        return {
            "max_native_side": 1024,
            "max_frames": 121,
            "max_pixel_frames": 768 * 448 * 121,
            "max_output_side": 4096,
            "max_output_pixels": 4096 * 2160,
            "max_upscaled_frames": 121,
            "label": "H100 full 22B bf16 LTX",
            "guidance": "use 5 seconds at 768x448 for full 22B bf16; use distilled or H200 for larger clips",
        }
    if distilled_like:
        return {
            "max_native_side": 1024,
            "max_frames": 121,
            "max_pixel_frames": 1024 * 576 * 121,
            "max_output_side": 1024,
            "max_output_pixels": 1024 * 1024,
            "max_upscaled_frames": 121,
            "label": "local distilled LTX",
            "guidance": "use 5 seconds at 1024x576, or reduce resolution for longer clips",
        }
    return {
        "max_native_side": 1024,
        "max_frames": 121,
        "max_pixel_frames": 768 * 448 * 121,
        "max_output_side": 1024,
        "max_output_pixels": 1024 * 1024,
        "max_upscaled_frames": 121,
        "label": "local full 22B LTX",
        "guidance": "use 5 seconds at 768x448, or switch to the H200 profile for larger jobs",
    }


def _native_request_fits(req: VideoRequest, budget: dict[str, Any]) -> bool:
    pixel_frames = req.width * req.height * req.num_frames
    return (
        req.width <= budget["max_native_side"]
        and req.height <= budget["max_native_side"]
        and req.num_frames <= budget["max_frames"]
        and pixel_frames <= budget["max_pixel_frames"]
        and req.width % 32 == 0
        and req.height % 32 == 0
    )


@lru_cache(maxsize=1)
def _two_stage_pipeline() -> Any:
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

    return TI2VidTwoStagesPipeline(
        checkpoint_path=_pick("ltx-2.3-22b-dev.safetensors", "ltx-2.3-22b-distilled-1.1.safetensors"),
        distilled_lora=_distilled_lora(),
        spatial_upsampler_path=_pick("ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"),
        gemma_root=_gemma_root(),
        loras=[],
        device=_device(),
        torch_compile=settings.torch_compile_ltx,
        **_quantization(),
    )


@lru_cache(maxsize=1)
def _distilled_pipeline() -> Any:
    from ltx_pipelines.distilled import DistilledPipeline

    return DistilledPipeline(
        distilled_checkpoint_path=_pick("ltx-2.3-22b-distilled-1.1.safetensors", "ltx-2.3-22b-distilled.safetensors"),
        spatial_upsampler_path=_pick("ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"),
        gemma_root=_gemma_root(),
        loras=[],
        device=_device(),
        torch_compile=settings.torch_compile_ltx,
        **_quantization(),
    )


async def run_ltx_job(job_id: str, req: VideoRequest, seed: int, output_path: Path) -> tuple[Path, dict[str, Any]]:
    job_dir = output_path.parent
    job_dir.mkdir(parents=True, exist_ok=True)
    return await asyncio.to_thread(_run_sync, job_id, req, seed, output_path)


def _run_sync(job_id: str, req: VideoRequest, seed: int, output_path: Path) -> tuple[Path, dict[str, Any]]:
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video

    generation_req, upscale_target = _generation_plan(req)
    encode_path = output_path.parent / "native_output.mp4" if upscale_target else output_path

    if generation_req.mode == VideoMode.distilled:
        video, audio = _distilled_pipeline()(
            prompt=generation_req.prompt,
            seed=seed,
            height=generation_req.height,
            width=generation_req.width,
            num_frames=generation_req.num_frames,
            frame_rate=generation_req.frame_rate or 24.0,
            images=[],
            tiling_config=TilingConfig.default(),
            enhance_prompt=bool(generation_req.enhance_prompt),
        )
    elif generation_req.mode in {VideoMode.text_to_video, VideoMode.image_to_video}:
        images = _image_conditionings(generation_req)
        video, audio = _two_stage_pipeline()(
            prompt=generation_req.prompt,
            negative_prompt=generation_req.negative_prompt or "",
            seed=seed,
            height=generation_req.height,
            width=generation_req.width,
            num_frames=generation_req.num_frames,
            frame_rate=generation_req.frame_rate or 24.0,
            num_inference_steps=generation_req.num_inference_steps or 40,
            video_guider_params=MultiModalGuiderParams(cfg_scale=generation_req.guidance_scale or 7.5),
            audio_guider_params=MultiModalGuiderParams(),
            images=images,
            tiling_config=TilingConfig.default(),
            enhance_prompt=bool(generation_req.enhance_prompt),
        )
    else:
        video, audio = _run_specialized_pipeline(generation_req, seed)

    chunks = get_video_chunks_number(generation_req.num_frames, TilingConfig.default())
    encode_video(
        video=video,
        fps=int(generation_req.frame_rate or 24),
        audio=audio,
        output_path=str(encode_path),
        video_chunks_number=chunks,
    )
    if upscale_target:
        _upscale_video(encode_path, output_path, upscale_target[0], upscale_target[1])
    metadata = {
        "upscaled": bool(upscale_target),
        "render_width": generation_req.width,
        "render_height": generation_req.height,
        "output_width": req.width,
        "output_height": req.height,
        "num_frames": req.num_frames,
        "frame_rate": req.frame_rate or 24.0,
    }
    return output_path, metadata


def _generation_plan(req: VideoRequest) -> tuple[VideoRequest, tuple[int, int] | None]:
    budget = _ltx_budget(req)
    if _native_request_fits(req, budget):
        return req, None
    native_pixels = max(32 * 32, budget["max_pixel_frames"] // max(req.num_frames, 1))
    width, height = _fit_native_size(req.width, req.height, budget["max_native_side"], native_pixels)
    data = req.model_dump()
    data["width"] = width
    data["height"] = height
    return VideoRequest.model_validate(data), (req.width, req.height)


def _fit_native_size(target_width: int, target_height: int, max_side: int, max_pixels: int) -> tuple[int, int]:
    aspect = max(target_width, 1) / max(target_height, 1)
    width = min(target_width, max_side, int(math.sqrt(max_pixels * aspect)))
    height = min(target_height, max_side, int(width / aspect))
    width = _floor_to_32(width)
    height = _floor_to_32(height)
    while width * height > max_pixels and (width > 256 or height > 256):
        if width >= height and width > 256:
            width -= 32
        elif height > 256:
            height -= 32
        else:
            break
    return max(256, width), max(256, height)


def _floor_to_32(value: int) -> int:
    return max(256, (max(value, 256) // 32) * 32)


def _upscale_video(input_path: Path, output_path: Path, width: int, height: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for 4K video output but was not found on PATH")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"scale={width}:{height}:flags=lanczos",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 4K upscale failed: {result.stderr[-2000:]}")


def _image_conditionings(req: VideoRequest) -> list[Any]:
    if not req.image_url:
        return []
    image_path = Path(req.extra.get("image_path")) if req.extra and req.extra.get("image_path") else None
    return [(str(image_path or req.image_url), 0)]


def _run_specialized_pipeline(req: VideoRequest, seed: int) -> tuple[Any, Any]:
    if req.mode == VideoMode.video_to_video:
        from ltx_pipelines.ic_lora import ICLoraPipeline

        pipeline = ICLoraPipeline(
            distilled_checkpoint_path=_pick("ltx-2.3-22b-distilled-1.1.safetensors"),
            gemma_root=_gemma_root(),
            spatial_upsampler_path=_pick("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            loras=[],
            device=_device(),
            **_quantization(),
        )
        return pipeline(prompt=req.prompt, seed=seed, height=req.height, width=req.width, num_frames=req.num_frames, frame_rate=req.frame_rate or 24.0, images=[], videos=[req.video_url])
    if req.mode == VideoMode.audio_to_video:
        from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage

        pipeline = A2VidPipelineTwoStage(
            checkpoint_path=_pick("ltx-2.3-22b-dev.safetensors", "ltx-2.3-22b-distilled-1.1.safetensors"),
            distilled_lora=_distilled_lora(),
            spatial_upsampler_path=_pick("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            gemma_root=_gemma_root(),
            loras=[],
            device=_device(),
            **_quantization(),
        )
        return pipeline(prompt=req.prompt, audio_path=req.audio_url, seed=seed, height=req.height, width=req.width, num_frames=req.num_frames, frame_rate=req.frame_rate or 24.0)
    if req.mode == VideoMode.keyframe_interpolation:
        from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline

        pipeline = KeyframeInterpolationPipeline(
            checkpoint_path=_pick("ltx-2.3-22b-dev.safetensors", "ltx-2.3-22b-distilled-1.1.safetensors"),
            distilled_lora=_distilled_lora(),
            spatial_upsampler_path=_pick("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            gemma_root=_gemma_root(),
            loras=[],
            device=_device(),
            **_quantization(),
        )
        return pipeline(prompt=req.prompt, keyframes=req.keyframe_urls or [], seed=seed, height=req.height, width=req.width, num_frames=req.num_frames, frame_rate=req.frame_rate or 24.0)
    if req.mode == VideoMode.retake:
        from ltx_pipelines.retake import RetakePipeline

        pipeline = RetakePipeline(
            checkpoint_path=_pick("ltx-2.3-22b-dev.safetensors", "ltx-2.3-22b-distilled-1.1.safetensors"),
            gemma_root=_gemma_root(),
            device=_device(),
            **_quantization(),
        )
        return pipeline(prompt=req.prompt, video_path=req.video_url, start_time=req.retake_start_time, end_time=req.retake_end_time, seed=seed)
    if req.mode == VideoMode.hdr:
        from ltx_pipelines.hdr_ic_lora import HDRICLoraPipeline

        pipeline = HDRICLoraPipeline(
            distilled_checkpoint_path=_pick("ltx-2.3-22b-distilled-1.1.safetensors"),
            gemma_root=_gemma_root(),
            spatial_upsampler_path=_pick("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            device=_device(),
            **_quantization(),
        )
        return pipeline(input=req.video_url, prompt=req.prompt, seed=seed, height=req.height, width=req.width, num_frames=req.num_frames)
    raise ValueError(f"unsupported mode {req.mode}")


async def materialize_inputs(req: VideoRequest, job_dir: Path) -> VideoRequest:
    data = req.model_dump()
    extra = dict(data.get("extra") or {})
    if req.image_url:
        extra["image_path"] = str(await fetch_media(req.image_url, job_dir, "image"))
    if req.video_url:
        extra["video_path"] = str(await fetch_media(req.video_url, job_dir, "video"))
        data["video_url"] = extra["video_path"]
    if req.audio_url:
        extra["audio_path"] = str(await fetch_media(req.audio_url, job_dir, "audio"))
        data["audio_url"] = extra["audio_path"]
    if req.keyframe_urls:
        paths = []
        for idx, url in enumerate(req.keyframe_urls):
            paths.append(str(await fetch_media(url, job_dir, f"keyframe_{idx}")))
        data["keyframe_urls"] = paths
    data["extra"] = extra
    return VideoRequest.model_validate(data)
