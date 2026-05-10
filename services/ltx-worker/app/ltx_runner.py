from __future__ import annotations

import asyncio
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


def ensure_runtime_ready() -> None:
    if settings.cuda_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for LTX generation but this worker loaded a CPU-only PyTorch runtime. "
            "Install the locked CUDA PyTorch environment with Python 3.12, then restart the worker."
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


async def run_ltx_job(job_id: str, req: VideoRequest, seed: int, output_path: Path) -> Path:
    job_dir = output_path.parent
    job_dir.mkdir(parents=True, exist_ok=True)
    return await asyncio.to_thread(_run_sync, job_id, req, seed, output_path)


def _run_sync(job_id: str, req: VideoRequest, seed: int, output_path: Path) -> Path:
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video

    if req.mode == VideoMode.distilled:
        video, audio = _distilled_pipeline()(
            prompt=req.prompt,
            seed=seed,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            frame_rate=req.frame_rate or 24.0,
            images=[],
            tiling_config=TilingConfig.default(),
            enhance_prompt=bool(req.enhance_prompt),
        )
    elif req.mode in {VideoMode.text_to_video, VideoMode.image_to_video}:
        images = _image_conditionings(req)
        video, audio = _two_stage_pipeline()(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "",
            seed=seed,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            frame_rate=req.frame_rate or 24.0,
            num_inference_steps=req.num_inference_steps or 40,
            video_guider_params=MultiModalGuiderParams(cfg_scale=req.guidance_scale or 7.5),
            audio_guider_params=MultiModalGuiderParams(),
            images=images,
            tiling_config=TilingConfig.default(),
            enhance_prompt=bool(req.enhance_prompt),
        )
    else:
        video, audio = _run_specialized_pipeline(req, seed)

    chunks = get_video_chunks_number(req.num_frames, TilingConfig.default())
    encode_video(video=video, fps=int(req.frame_rate or 24), audio=audio, output_path=str(output_path), video_chunks_number=chunks)
    return output_path


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
