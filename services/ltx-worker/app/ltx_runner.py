from __future__ import annotations

import asyncio
import gc
import logging
import os
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import httpx
import torch

from .config import settings
from .media import fetch_media
from .schemas import VideoMode, VideoRequest

logger = logging.getLogger(__name__)

WEIGHT_BYTES = 44 * 1024**3
PER_TOKEN_ACT_BYTES = 48 * 1024
B200_PER_TOKEN_ACT_BYTES = 32 * 1024
OVERHEAD_BYTES = 12 * 1024**3
B200_OVERHEAD_BYTES = 10 * 1024**3
SAFETY_BYTES = 6 * 1024**3

_PRECISION_READY = False
_PATCH_APPLIED = False
_PIPELINE_LOAD_LOCK = threading.Lock()

# One active pipeline at a time
_active_pipeline: Any | None = None
_active_variant: str | None = None


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


def _quantization_for(name: str) -> dict[str, Any]:
    if name == "none":
        return {}
    from ltx_core.quantization import QuantizationPolicy

    if name == "fp8_cast":
        return {"quantization": QuantizationPolicy.fp8_cast()}
    if name == "fp8_scaled_mm":
        return {"quantization": QuantizationPolicy.fp8_scaled_mm()}
    raise ValueError(f"unsupported quantization {name}")


def _device() -> torch.device:
    ensure_runtime_ready()
    return torch.device(settings.cuda_device)


def runtime_status() -> dict[str, Any]:
    vram_free_gib = 0.0
    vram_total_gib = 0.0
    gpu_name = ""
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            vram_free_gib = free / 1024**3
            vram_total_gib = total / 1024**3
            gpu_name = torch.cuda.get_device_name(0)
        except RuntimeError:
            pass
    return {
        "cuda_requested": settings.cuda_device.startswith("cuda"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": settings.cuda_device,
        "gpu_profile": settings.gpu_profile,
        "gpu_name": gpu_name,
        "vram_free_gib": round(vram_free_gib, 2),
        "vram_total_gib": round(vram_total_gib, 2),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "active_variant": _active_variant,
        "pipeline_loaded": _active_pipeline is not None,
        "preload_on_start": settings.preload_on_start,
        "max_num_frames": settings.max_num_frames,
        "max_tokens": _max_tokens(),
    }


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "CUDA out of memory" in str(exc)


def release_cuda_memory(clear_pipelines: bool = False) -> None:
    global _active_pipeline, _active_variant
    if clear_pipelines:
        _active_pipeline = None
        _active_variant = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _patch_image_processor_fast() -> None:
    try:
        import transformers

        original = transformers.AutoImageProcessor.from_pretrained.__func__

        def _use_fast(cls: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("use_fast", True)
            return original(cls, *args, **kwargs)

        transformers.AutoImageProcessor.from_pretrained = classmethod(_use_fast)
    except Exception:
        logger.debug("Could not patch AutoImageProcessor.from_pretrained", exc_info=True)


def _apply_patch_once() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    from .ltx_patch.patch import apply_patch

    apply_patch(settings.text_worker_url, torch.device(settings.cuda_device))
    _PATCH_APPLIED = True


def ensure_runtime_ready() -> None:
    global _PRECISION_READY
    if settings.cuda_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for LTX generation but this worker loaded a CPU-only PyTorch runtime. "
            "Install the locked CUDA PyTorch environment with Python 3.12, then restart the worker."
        )
    if not _PRECISION_READY and torch.cuda.is_available():
        _apply_patch_once()
        _patch_image_processor_fast()
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        _PRECISION_READY = True


def _max_tokens() -> int:
    profile = settings.gpu_profile.lower()
    if "b200" in profile:
        return settings.max_tokens_b200
    if "h200" in profile:
        return settings.max_tokens_h200
    if "h100" in profile:
        return settings.max_tokens_h100
    return settings.max_tokens_local


def _token_count(width: int, height: int, num_frames: int) -> int:
    return ((num_frames - 1) // 8 + 1) * (height // 8) * (width // 8)


def _required_bytes(tokens: int) -> int:
    weight_bytes = 0 if _active_pipeline is not None else WEIGHT_BYTES
    if "b200" in settings.gpu_profile.lower():
        return weight_bytes + tokens * B200_PER_TOKEN_ACT_BYTES + B200_OVERHEAD_BYTES
    return weight_bytes + tokens * PER_TOKEN_ACT_BYTES + OVERHEAD_BYTES


def _suggest_native(num_frames: int, cap: int, aspect: float = 16 / 9) -> tuple[int, int]:
    f_factor = (num_frames - 1) // 8 + 1
    if f_factor <= 0:
        return 768, 448
    spatial_cap = cap // f_factor
    if spatial_cap <= 0:
        return 768, 448
    width_latent = int((spatial_cap * aspect) ** 0.5)
    height_latent = int(width_latent / aspect)
    width = max(256, (width_latent * 8 // 32) * 32)
    height = max(256, (height_latent * 8 // 32) * 32)
    while _token_count(width, height, num_frames) > cap and (width > 256 or height > 256):
        if width >= height and width > 256:
            width -= 32
        elif height > 256:
            height -= 32
        else:
            break
    return width, height


def current_budget_hint() -> dict[str, Any]:
    profile = settings.gpu_profile.lower()
    cap = _max_tokens()
    examples = {
        "b200": "1920x1088@481f (20s), 2560x1440@241f (10s), 3840x2160@121f (5s output)",
        "h200": "1408x768@481f (20s), 1664x928@241f (10s), 1920x1088@121f (5s)",
        "h100": "768x448@481f (20s), 1024x576@241f (10s), 1280x704@121f (5s)",
    }
    if "b200" in profile:
        example = examples["b200"]
    elif "h200" in profile:
        example = examples["h200"]
    elif "h100" in profile:
        example = examples["h100"]
    else:
        example = "768x448 up to 81f"
    return {
        "profile": settings.gpu_profile,
        "max_tokens": cap,
        "max_num_frames": settings.max_num_frames,
        "examples": example,
    }


def validate_ltx_budget(req: VideoRequest) -> None:
    if req.mode not in {VideoMode.text_to_video, VideoMode.image_to_video}:
        raise ValueError(
            "this LTX worker allows only text_to_video and image_to_video; "
            "use model_variant to select distilled/full_fp8/full_bf16"
        )
    if req.num_frames < 1 or (req.num_frames - 1) % 8 != 0:
        raise ValueError("num_frames must satisfy 8k+1")
    if req.num_inference_steps is not None and not 1 <= req.num_inference_steps <= 40:
        raise ValueError("num_inference_steps must be between 1 and 40")
    if req.width % 32 != 0 or req.height % 32 != 0:
        raise ValueError("width and height must be divisible by 32")
    if req.width < 256 or req.height < 256:
        raise ValueError("width and height must be >= 256")
    if req.num_frames > settings.max_num_frames:
        raise ValueError(
            f"num_frames {req.num_frames} exceeds max {settings.max_num_frames} "
            f"(~{(settings.max_num_frames - 1) / 24:.1f}s @24fps)"
        )

    tokens = _token_count(req.width, req.height, req.num_frames)
    cap = _max_tokens()
    if tokens > cap:
        suggested_w, suggested_h = _suggest_native(req.num_frames, cap)
        raise ValueError(
            f"{req.width}x{req.height}@{req.num_frames}f exceeds {settings.gpu_profile} "
            f"single-shot capacity ({tokens} > {cap} tokens). "
            f"Suggested native at this duration: {suggested_w}x{suggested_h}@{req.num_frames}f. "
            f"For other durations on B200: 1920x1088@481f, 2560x1440@241f. "
            f"For H200: 1408x768@481f, 1920x1088@121f."
        )

    if torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info(_device())
        except RuntimeError:
            return
        required = _required_bytes(tokens) + SAFETY_BYTES
        if free < required:
            raise ValueError(
                f"GPU currently has only {free >> 30} GiB free; "
                f"{req.width}x{req.height}@{req.num_frames}f needs ~{required >> 30} GiB. "
                f"Retry shortly or reduce size."
            )


def _resolve_variant(req: VideoRequest | None) -> str:
    raw = (req.model_variant if req is not None else None) or settings.ltx_default_variant or "distilled"
    return raw if raw in {"distilled", "full_fp8", "full_bf16"} else "distilled"


def _build_pipeline(variant: str) -> Any:
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

    if variant == "distilled":
        try:
            checkpoint = _pick(settings.ltx_distilled_checkpoint, "ltx-2.3-22b-distilled.safetensors")
        except FileNotFoundError:
            logger.warning(
                "distilled checkpoint not found (tried %s); falling back to full_bf16",
                settings.ltx_distilled_checkpoint,
            )
            return _build_pipeline("full_bf16")
        quant: dict[str, Any] = {}
    elif variant == "full_fp8":
        checkpoint = _pick("ltx-2.3-22b-dev.safetensors")
        quant = _quantization_for("fp8_cast")
    else:  # full_bf16
        checkpoint = _pick("ltx-2.3-22b-dev.safetensors")
        quant = {}

    logger.info("ltx_runner loading pipeline variant=%s checkpoint=%s", variant, checkpoint)
    return TI2VidOneStagePipeline(
        checkpoint_path=checkpoint,
        gemma_root=_gemma_root(),  # intercepted by ltx_patch; value unused
        loras=[],
        device=_device(),
        torch_compile=settings.torch_compile_ltx,
        **quant,
    )


def _switch_pipeline(variant: str) -> tuple[Any, str]:
    global _active_pipeline, _active_variant
    with _PIPELINE_LOAD_LOCK:
        if _active_variant == variant and _active_pipeline is not None:
            return _active_pipeline, variant
        if _active_pipeline is not None:
            logger.info("ltx_runner unloading pipeline variant=%s to load variant=%s", _active_variant, variant)
            _active_pipeline = None
            _active_variant = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pipeline = _build_pipeline(variant)
        _active_pipeline = pipeline
        _active_variant = variant
        return pipeline, variant


def _select_pipeline(req: VideoRequest | None = None) -> tuple[Any, str]:
    return _switch_pipeline(_resolve_variant(req))


def preload_ltx_models() -> dict[str, Any]:
    ensure_runtime_ready()
    variant = _resolve_variant(None)
    pipeline, pipeline_kind = _switch_pipeline(variant)
    status = runtime_status()
    logger.info(
        "LTX pipeline ready variant=%s loaded=%s gpu_free_gib=%s gpu_total_gib=%s",
        pipeline_kind,
        pipeline is not None,
        status["vram_free_gib"],
        status["vram_total_gib"],
    )
    return {"pipeline": pipeline_kind, "runtime": status}


def _tiling_for(width: int, height: int, frames: int) -> Any:
    from ltx_core.model.video_vae import TilingConfig
    from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig

    pixels = width * height * frames
    if pixels > 200_000_000:
        return TilingConfig(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=24, tile_overlap_in_frames=8),
        )
    if pixels > 60_000_000:
        return TilingConfig(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=320, tile_overlap_in_pixels=64),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=24, tile_overlap_in_frames=8),
        )
    return TilingConfig.default()


def _enhance_prompt(prompt: str) -> str:
    system = (
        "You are a video prompt enhancer. Rewrite the user's prompt to be more vivid, "
        "cinematic, and descriptive for AI video generation. "
        "Return only the enhanced prompt, no commentary or explanation."
    )
    try:
        resp = httpx.post(
            f"{settings.text_worker_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 512,
                "temperature": 0.7,
                "stream": False,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        enhanced = resp.json()["choices"][0]["message"]["content"].strip()
        logger.info("enhance_prompt original=%r enhanced=%r", prompt[:80], enhanced[:80])
        return enhanced
    except Exception:
        logger.warning("enhance_prompt failed; using original prompt", exc_info=True)
        return prompt


async def run_ltx_job(job_id: str, req: VideoRequest, seed: int, output_path: Path) -> tuple[Path, dict[str, Any]]:
    job_dir = output_path.parent
    job_dir.mkdir(parents=True, exist_ok=True)
    return await asyncio.to_thread(_run_sync, job_id, req, seed, output_path)


def _run_sync(job_id: str, req: VideoRequest, seed: int, output_path: Path) -> tuple[Path, dict[str, Any]]:
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video

    pipeline, pipeline_kind = _select_pipeline(req)
    tiling = _tiling_for(req.width, req.height, req.num_frames)
    images = _image_conditionings(req)
    fps = req.frame_rate or 24.0

    # Pre-enhance prompt via text-worker before calling pipeline
    prompt = req.prompt
    if req.enhance_prompt:
        prompt = _enhance_prompt(prompt)

    logger.info(
        "ltx_runner job=%s variant=%s %sx%s@%sf tokens=%s",
        job_id, pipeline_kind, req.width, req.height, req.num_frames,
        _token_count(req.width, req.height, req.num_frames),
    )

    try:
        with torch.inference_mode():
            video, audio = pipeline(
                prompt=prompt,
                negative_prompt=req.negative_prompt or "",
                seed=seed,
                height=req.height,
                width=req.width,
                num_frames=req.num_frames,
                frame_rate=fps,
                num_inference_steps=req.num_inference_steps or 40,
                video_guider_params=MultiModalGuiderParams(cfg_scale=req.guidance_scale or 7.5),
                audio_guider_params=MultiModalGuiderParams(),
                images=images,
                tiling_config=tiling,
                enhance_prompt=False,  # already enhanced above
            )

        release_cuda_memory()
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(_device())
                logger.info(
                    "ltx_runner job=%s pre_encode_free_gib=%.2f total_gib=%.2f",
                    job_id,
                    free / 1024**3,
                    total / 1024**3,
                )
            except RuntimeError:
                logger.debug("ltx_runner job=%s could not read pre-encode CUDA memory", job_id, exc_info=True)

        chunks = get_video_chunks_number(req.num_frames, tiling)
        with torch.inference_mode():
            encode_video(
                video=video,
                fps=int(fps),
                audio=audio,
                output_path=str(output_path),
                video_chunks_number=chunks,
            )
    finally:
        if "video" in locals():
            del video
        if "audio" in locals():
            del audio
        release_cuda_memory()

    metadata = {
        "pipeline": pipeline_kind,
        "variant": pipeline_kind,
        "render_width": req.width,
        "render_height": req.height,
        "output_width": req.width,
        "output_height": req.height,
        "num_frames": req.num_frames,
        "frame_rate": fps,
        "tokens": _token_count(req.width, req.height, req.num_frames),
        "upscaled": False,
    }
    return output_path, metadata


def _image_conditionings(req: VideoRequest) -> list[Any]:
    if not req.image_url:
        return []
    image_path = Path(req.extra.get("image_path")) if req.extra and req.extra.get("image_path") else None
    return [(str(image_path or req.image_url), 0)]


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
