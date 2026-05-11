from __future__ import annotations

import pytest
import torch

from app import ltx_runner
from app.config import settings
from app.ltx_runner import (
    _max_tokens,
    _select_pipeline,
    _suggest_native,
    _tiling_for,
    _token_count,
    _use_one_stage,
    validate_ltx_budget,
)
from app.schemas import VideoMode, VideoRequest


@pytest.fixture(autouse=True)
def _stub_vram_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _req(width: int, height: int, num_frames: int, mode: VideoMode = VideoMode.text_to_video) -> VideoRequest:
    return VideoRequest(mode=mode, prompt="x", width=width, height=height, num_frames=num_frames)


@pytest.fixture
def h200_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "gpu_profile", "cloud_h200")


@pytest.fixture
def h100_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "gpu_profile", "cloud_h100")


@pytest.fixture
def local_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "gpu_profile", "local_rtx_5090")


def test_max_tokens_per_profile(h200_profile: None) -> None:
    assert _max_tokens() == settings.max_tokens_h200


def test_max_tokens_h100(h100_profile: None) -> None:
    assert _max_tokens() == settings.max_tokens_h100


def test_max_tokens_local(local_profile: None) -> None:
    assert _max_tokens() == settings.max_tokens_local


def test_token_count_formula() -> None:
    assert _token_count(1408, 768, 481) == 61 * 96 * 176
    assert _token_count(1920, 1088, 121) == 16 * 136 * 240
    assert _token_count(768, 448, 121) == 16 * 56 * 96


def test_h200_accepts_20s_at_1408x768(h200_profile: None) -> None:
    validate_ltx_budget(_req(1408, 768, 481))


def test_h200_accepts_5s_full_hd(h200_profile: None) -> None:
    validate_ltx_budget(_req(1920, 1088, 121))


def test_h200_rejects_1920_1088_at_481f(h200_profile: None) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        validate_ltx_budget(_req(1920, 1088, 481))


def test_h200_reject_message_suggests_native(h200_profile: None) -> None:
    with pytest.raises(ValueError) as exc:
        validate_ltx_budget(_req(1920, 1088, 481))
    assert "Suggested native" in str(exc.value)
    assert "@481f" in str(exc.value)


def test_h100_accepts_20s_at_768x448(h100_profile: None) -> None:
    validate_ltx_budget(_req(768, 448, 481))


def test_h100_rejects_20s_at_1024x576(h100_profile: None) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        validate_ltx_budget(_req(1024, 576, 481))


def test_rejects_non_multiple_of_32() -> None:
    with pytest.raises(ValueError, match="divisible by 32"):
        validate_ltx_budget(_req(500, 500, 49))


def test_rejects_bad_frame_count() -> None:
    with pytest.raises(ValueError, match="8k\\+1"):
        validate_ltx_budget(_req(512, 512, 100))


def test_rejects_above_max_num_frames(h200_profile: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_num_frames", 481)
    with pytest.raises(ValueError, match="exceeds max 481"):
        validate_ltx_budget(_req(768, 448, 489))


def test_rejects_specialized_modes() -> None:
    with pytest.raises(ValueError, match="one full dev model"):
        validate_ltx_budget(_req(512, 512, 49, mode=VideoMode.distilled))


def test_suggest_native_under_cap_h200() -> None:
    cap = settings.max_tokens_h200
    width, height = _suggest_native(481, cap)
    assert _token_count(width, height, 481) <= cap
    assert width % 32 == 0
    assert height % 32 == 0


def test_use_one_stage_long() -> None:
    assert _use_one_stage(_req(768, 448, 241)) is True


def test_use_one_stage_short_small() -> None:
    assert _use_one_stage(_req(1024, 576, 121)) is True


def test_use_one_stage_short_large() -> None:
    assert _use_one_stage(_req(1920, 1088, 121)) is True


def test_tiling_for_small() -> None:
    tiling = _tiling_for(512, 512, 25)
    assert tiling.spatial_config.tile_size_in_pixels == 512
    assert tiling.temporal_config.tile_size_in_frames == 64


def test_tiling_for_mid() -> None:
    tiling = _tiling_for(1024, 576, 121)
    assert tiling.spatial_config.tile_size_in_pixels == 448
    assert tiling.temporal_config.tile_size_in_frames == 48


def test_tiling_for_large() -> None:
    tiling = _tiling_for(1408, 768, 481)
    assert tiling.spatial_config.tile_size_in_pixels == 384
    assert tiling.temporal_config.tile_size_in_frames == 32


def test_vram_probe_rejects_when_pool_too_small(monkeypatch: pytest.MonkeyPatch, h200_profile: None) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **kw: (4 * 1024**3, 141 * 1024**3))
    monkeypatch.setattr(ltx_runner, "ensure_runtime_ready", lambda: None)
    with pytest.raises(ValueError, match="only 4 GiB free"):
        validate_ltx_budget(_req(1408, 768, 481))


def test_vram_probe_passes_when_pool_ample(monkeypatch: pytest.MonkeyPatch, h200_profile: None) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **kw: (130 * 1024**3, 141 * 1024**3))
    monkeypatch.setattr(ltx_runner, "ensure_runtime_ready", lambda: None)
    validate_ltx_budget(_req(1408, 768, 481))
