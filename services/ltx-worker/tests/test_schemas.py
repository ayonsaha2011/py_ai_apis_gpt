from __future__ import annotations

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

