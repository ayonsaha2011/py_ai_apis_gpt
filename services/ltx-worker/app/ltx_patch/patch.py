from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import torch

logger = logging.getLogger(__name__)


def apply_patch(text_worker_url: str, device: torch.device) -> None:
    """
    Monkey-patch ltx_core.components.prompt_encoder.PromptEncoder so it uses
    RemoteGemmaTextEncoder instead of loading a local Gemma model.

    Must be called before any ltx_pipelines import that triggers PromptEncoder init.
    """
    from .remote_encoder import RemoteGemmaTextEncoder

    try:
        from ltx_core.components import prompt_encoder as _pe_mod
    except ImportError:
        logger.warning("ltx_core.components.prompt_encoder not found; skipping patch")
        return

    _remote = RemoteGemmaTextEncoder(text_worker_url, device)

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        self._text_encoder = _remote
        # Stub out any attributes the pipeline may read without using
        self._gemma_root = None

    @contextmanager  # type: ignore[arg-type]
    def _patched_ctx(self: Any):  # type: ignore[no-untyped-def]
        yield self._text_encoder

    _pe_mod.PromptEncoder.__init__ = _patched_init  # type: ignore[method-assign]

    # Patch _text_encoder_ctx if it exists; fall back gracefully if the method name differs
    if hasattr(_pe_mod.PromptEncoder, "_text_encoder_ctx"):
        _pe_mod.PromptEncoder._text_encoder_ctx = _patched_ctx  # type: ignore[method-assign]
    else:
        logger.warning(
            "PromptEncoder._text_encoder_ctx not found; LTX may call encode() directly. "
            "Check ltx_core version if text conditioning fails."
        )

    logger.info("ltx_patch applied: PromptEncoder → RemoteGemmaTextEncoder at %s", _remote._url)
