from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException

from .config import settings
from .ltx_runner import ensure_runtime_ready, materialize_inputs, run_ltx_job, runtime_status
from .schemas import InternalJobRequest
from .storage import store_video

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
    }


@app.post("/internal/ltx/jobs")
async def run_job(body: InternalJobRequest, x_service_key: str | None = Header(default=None)) -> dict:
    _check_service_key(x_service_key)
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
            req = await materialize_inputs(body.request, job_dir)
            if cancel_event.is_set():
                raise HTTPException(status_code=409, detail="job cancelled after input materialization")
            await run_ltx_job(body.job_id, req, body.effective_seed, output_path)
            if cancel_event.is_set():
                raise HTTPException(status_code=409, detail="job cancelled after generation")
            result_url = await store_video(output_path, body.r2_key)
            return {"job_id": body.job_id, "status": "complete", "result_url": result_url}
    finally:
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
