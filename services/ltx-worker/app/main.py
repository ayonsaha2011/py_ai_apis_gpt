from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

import torch
from fastapi import FastAPI, Header, HTTPException

from .config import settings
from .ltx_runner import (
    current_budget_hint,
    ensure_runtime_ready,
    is_cuda_oom,
    materialize_inputs,
    release_cuda_memory,
    run_ltx_job,
    runtime_status,
    validate_ltx_budget,
)
from .schemas import InternalJobRequest
from .storage import store_video

logger = logging.getLogger(__name__)

_jobs: dict[str, asyncio.Event] = {}
_gpu_lock = asyncio.Lock()


def _check_service_key(x_service_key: str | None) -> None:
    if settings.service_api_key and x_service_key != settings.service_api_key:
        raise HTTPException(status_code=401, detail="invalid service key")


app = FastAPI(title="ltx-worker")


@app.get("/health")
async def health() -> dict:
    status = runtime_status()
    return {
        "status": "ok" if status["cuda_available"] else "degraded",
        "model_dir": str(settings.ltx_model_dir),
        "runtime": status,
        "budget": current_budget_hint(),
    }


@app.post("/internal/ltx/jobs")
async def run_job(body: InternalJobRequest, x_service_key: str | None = Header(default=None)) -> dict:
    _check_service_key(x_service_key)
    try:
        validate_ltx_budget(body.request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    cancel_event = asyncio.Event()
    _jobs[body.job_id] = cancel_event
    job_dir = settings.local_storage_dir / "work" / body.job_id
    output_path = job_dir / "output.mp4"
    try:
        try:
            ensure_runtime_ready()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        async with _gpu_lock:
            if cancel_event.is_set():
                raise HTTPException(status_code=409, detail="job cancelled before start")
            try:
                req = await materialize_inputs(body.request, job_dir)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if cancel_event.is_set():
                raise HTTPException(status_code=409, detail="job cancelled after input materialization")
            output_path, metadata = await run_ltx_job(body.job_id, req, body.effective_seed, output_path)
            if cancel_event.is_set():
                raise HTTPException(status_code=409, detail="job cancelled after generation")
            result_url = await store_video(output_path, body.r2_key)
            return {"job_id": body.job_id, "status": "complete", "result_url": result_url, "metadata": metadata}
    except HTTPException:
        raise
    except Exception as exc:
        if is_cuda_oom(exc):
            free_gib = total_gib = 0
            try:
                free, total = torch.cuda.mem_get_info()
                free_gib, total_gib = free >> 30, total >> 30
                logger.error(
                    "LTX OOM. free=%dGiB/%dGiB profile=%s\n%s",
                    free_gib, total_gib, settings.gpu_profile,
                    torch.cuda.memory_summary(abbreviated=True),
                )
            except RuntimeError:
                logger.error("LTX OOM; mem_get_info unavailable", exc_info=exc)
            release_cuda_memory(clear_pipelines=True)
            hint = current_budget_hint()
            raise HTTPException(
                status_code=507,
                detail=(
                    f"LTX OOM on {hint['profile']} (bf16 dev). "
                    f"Free {free_gib} GiB / {total_gib} GiB. "
                    f"Max single-shot budget {hint['max_tokens']} tokens; "
                    f"recommended sizes: {hint['examples']}."
                ),
            ) from exc
        raise
    finally:
        release_cuda_memory()
        _jobs.pop(body.job_id, None)
        shutil.rmtree(job_dir, ignore_errors=True)


@app.post("/internal/ltx/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, x_service_key: str | None = Header(default=None)) -> dict:
    _check_service_key(x_service_key)
    event = _jobs.get(job_id)
    if event is not None:
        event.set()
        return {"job_id": job_id, "status": "cancelling"}
    return {"job_id": job_id, "status": "not_running"}
