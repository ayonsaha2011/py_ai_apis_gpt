from __future__ import annotations

import pytest

from app.ltx_runner import validate_ltx_budget
from app.schemas import InternalJobRequest, VideoMode


def test_internal_job_schema_accepts_all_modes() -> None:
    for mode in VideoMode:
        job = InternalJobRequest.model_validate(
            {
                "job_id": "job",
                "effective_seed": 123,
                "r2_key": "users/u/videos/job/output.mp4",
                "request": {
                    "mode": mode.value,
                    "prompt": "test",
                    "width": 512,
                    "height": 512,
                    "num_frames": 97,
                },
            }
        )
        assert job.request.mode == mode


def test_ltx_worker_rejects_distilled_mode() -> None:
    job = InternalJobRequest.model_validate(
        {
            "job_id": "job",
            "effective_seed": 123,
            "r2_key": "users/u/videos/job/output.mp4",
            "request": {
                "mode": VideoMode.distilled.value,
                "prompt": "test",
                "width": 512,
                "height": 512,
                "num_frames": 97,
            },
        }
    )
    with pytest.raises(ValueError, match="one full dev model"):
        validate_ltx_budget(job.request)
