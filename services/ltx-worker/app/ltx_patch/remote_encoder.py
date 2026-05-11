from __future__ import annotations

import base64
import logging

import httpx
import numpy as np
import torch

logger = logging.getLogger(__name__)


class RemoteGemmaTextEncoder:
    """Calls text-worker /encode instead of running a local Gemma instance."""

    def __init__(self, text_worker_url: str, device: torch.device) -> None:
        self._url = text_worker_url.rstrip("/") + "/encode"
        self._device = device

    def encode(self, text: str, padding_side: str = "left") -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        resp = httpx.post(self._url, json={"texts": [text]}, timeout=60.0)
        resp.raise_for_status()
        emb = resp.json()["embeddings"][0]
        shape = emb["shape"]  # [L, seq_len, D]
        arr = np.frombuffer(base64.b64decode(emb["hidden_states"]), dtype=np.float32).reshape(shape)
        mask_arr = np.frombuffer(base64.b64decode(emb["attention_mask"]), dtype=np.int64)
        hidden = tuple(torch.from_numpy(arr[i].copy()).to(self._device) for i in range(shape[0]))
        mask = torch.from_numpy(mask_arr.copy()).unsqueeze(0).to(self._device)
        return hidden, mask

    def enhance_t2v(self, prompt: str, **_: object) -> str:
        return prompt

    def enhance_i2v(self, prompt: str, **_: object) -> str:
        return prompt
