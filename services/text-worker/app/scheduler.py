from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import settings
from .schemas import ChatMessage, ChatRequest


Past = tuple[tuple[torch.Tensor, ...], ...] | tuple[Any, ...]


@dataclass
class CacheEntry:
    input_ids: tuple[int, ...]
    past: Past
    last_used: float
    bytes_used: int


class KVCacheManager:
    def __init__(self, max_bytes: int, ttl_seconds: int) -> None:
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.bytes_used = 0

    def get_prefix(self, session_id: str, input_ids: list[int]) -> tuple[Past | None, int]:
        self.evict_expired()
        entry = self.entries.get(session_id)
        if entry is None:
            return None, 0
        prefix = list(entry.input_ids)
        if len(prefix) <= len(input_ids) and input_ids[: len(prefix)] == prefix:
            entry.last_used = time.monotonic()
            self.entries.move_to_end(session_id)
            return _detach_past(entry.past), len(prefix)
        return None, 0

    def put(self, session_id: str | None, input_ids: list[int], past: Past) -> None:
        if not session_id:
            return
        bytes_used = _past_nbytes(past)
        if bytes_used > self.max_bytes:
            return
        old = self.entries.pop(session_id, None)
        if old is not None:
            self.bytes_used -= old.bytes_used
        self.entries[session_id] = CacheEntry(
            input_ids=tuple(input_ids),
            past=_detach_past(past),
            last_used=time.monotonic(),
            bytes_used=bytes_used,
        )
        self.bytes_used += bytes_used
        self.evict_to_budget()

    def evict_expired(self) -> None:
        cutoff = time.monotonic() - self.ttl_seconds
        for key in list(self.entries):
            if self.entries[key].last_used < cutoff:
                entry = self.entries.pop(key)
                self.bytes_used -= entry.bytes_used

    def evict_to_budget(self) -> None:
        while self.bytes_used > self.max_bytes and self.entries:
            _, entry = self.entries.popitem(last=False)
            self.bytes_used -= entry.bytes_used


@dataclass
class GenerationTicket:
    request_id: str
    request: ChatRequest
    prompt_ids: list[int]
    max_new_tokens: int
    future: asyncio.Future[str]
    stream: asyncio.Queue[str | None] | None
    cancelled: bool = False


@dataclass
class ActiveState:
    ticket: GenerationTicket
    generated: list[int] = field(default_factory=list)
    past: Past | None = None
    past_len: int = 0
    next_input_id: int | None = None
    finished: bool = False


class GenerationScheduler:
    def __init__(self) -> None:
        self.model_id = settings.model_id
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
        self.waiting: asyncio.Queue[GenerationTicket] = asyncio.Queue(maxsize=settings.max_waiting)
        self.kv_cache = KVCacheManager(settings.kv_cache_bytes, settings.kv_cache_ttl_seconds)
        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._model_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._model_lock:
            await asyncio.to_thread(self._load_model, self.model_id)
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._shutdown.set()
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def switch_model(self, model_id: str) -> None:
        async with self._model_lock:
            self.model_id = model_id
            await asyncio.to_thread(self._load_model, model_id)
            self.kv_cache = KVCacheManager(settings.kv_cache_bytes, settings.kv_cache_ttl_seconds)

    async def submit(self, request: ChatRequest) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        async with self._model_lock:
            ticket = self._ticket(request, future, None)
        await self.waiting.put(ticket)
        return await future

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)
        async with self._model_lock:
            ticket = self._ticket(request, future, queue)
        await self.waiting.put(ticket)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
            await future
        except asyncio.CancelledError:
            ticket.cancelled = True
            if not future.done():
                future.cancel()
            raise

    def _ticket(self, request: ChatRequest, future: asyncio.Future[str], queue: asyncio.Queue[str | None] | None) -> GenerationTicket:
        assert self.tokenizer is not None
        prompt = self._messages_to_prompt(request.messages)
        tokenized = self.tokenizer(prompt, return_tensors=None, add_special_tokens=True)
        ids = list(tokenized["input_ids"])[-settings.max_input_tokens :]
        return GenerationTicket(
            request_id=f"chatcmpl-{uuid.uuid4().hex}",
            request=request,
            prompt_ids=ids,
            max_new_tokens=min(request.max_tokens or settings.max_new_tokens, settings.max_new_tokens),
            future=future,
            stream=queue,
        )

    def _load_model(self, model_id: str) -> None:
        source = self._model_source(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        load_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "attn_implementation": settings.attn_implementation,
        }
        if self.device.type == "cuda":
            load_kwargs["device_map"] = {"": self.device.index or 0}
        self.model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
        self.model.eval()
        if self.device.type == "cpu":
            self.model.to(self.device)

    def _model_source(self, model_id: str) -> str:
        model_dir = Path(settings.text_model_dir)
        if model_id == settings.model_id and (model_dir / "config.json").is_file():
            return str(model_dir)
        return model_id

    def _messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        assert self.tokenizer is not None
        raw = [m.model_dump() for m in messages]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(raw, tokenize=False, add_generation_prompt=True)
        return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"

    async def _run(self) -> None:
        active: list[ActiveState] = []
        while not self._shutdown.is_set():
            await self._admit(active)
            if not active:
                await asyncio.sleep(settings.scheduler_tick_ms / 1000)
                continue
            try:
                async with self._model_lock:
                    await asyncio.to_thread(self._decode_one_tick, active)
            except Exception as exc:
                self._fail_states(active, exc)
            active[:] = [state for state in active if not state.finished]

    async def _admit(self, active: list[ActiveState]) -> None:
        while len(active) < settings.max_active and not self.waiting.empty():
            ticket = await self.waiting.get()
            if ticket.future.cancelled() or ticket.cancelled:
                continue
            if len(ticket.prompt_ids) > settings.max_batch_tokens:
                ticket.future.set_exception(RuntimeError("prompt exceeds TEXT_MAX_BATCH_TOKENS"))
                if ticket.stream is not None:
                    try:
                        ticket.stream.put_nowait(None)
                    except asyncio.QueueFull:
                        pass
                continue
            if active and self._active_token_budget(active) + len(ticket.prompt_ids) > settings.max_batch_tokens:
                await self.waiting.put(ticket)
                break
            active.append(ActiveState(ticket=ticket))
        prefill = [state for state in active if state.past is None and not state.finished]
        if prefill:
            try:
                async with self._model_lock:
                    await asyncio.to_thread(self._prefill, prefill)
            except Exception as exc:
                self._fail_states(prefill, exc)

    def _active_token_budget(self, active: list[ActiveState]) -> int:
        return sum(len(state.ticket.prompt_ids) + len(state.generated) for state in active if not state.finished)

    def _prefill(self, states: list[ActiveState]) -> None:
        assert self.model is not None and self.tokenizer is not None
        uncached: list[ActiveState] = []
        for state in states:
            past, consumed = self.kv_cache.get_prefix(state.ticket.request.session_id or "", state.ticket.prompt_ids)
            if past is not None and consumed < len(state.ticket.prompt_ids):
                remaining = state.ticket.prompt_ids[consumed:]
                input_ids = torch.tensor([remaining], device=self.device)
                attention_mask = torch.ones((1, consumed + len(remaining)), device=self.device, dtype=torch.long)
                with torch.inference_mode():
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
                state.past = out.past_key_values
                state.past_len = consumed + len(remaining)
                state.next_input_id = _sample_next(out.logits[:, -1, :], state.ticket.request.temperature, state.ticket.request.top_p)[0]
                self.kv_cache.put(state.ticket.request.session_id, state.ticket.prompt_ids, state.past)
            elif past is not None:
                state.past = past
                state.past_len = len(state.ticket.prompt_ids)
                state.next_input_id = state.ticket.prompt_ids[-1]
            else:
                uncached.append(state)

        if not uncached:
            return
        max_len = max(len(s.ticket.prompt_ids) for s in uncached)
        pad = self.tokenizer.pad_token_id
        input_ids = []
        attention = []
        for state in uncached:
            ids = state.ticket.prompt_ids
            input_ids.append([pad] * (max_len - len(ids)) + ids)
            attention.append([0] * (max_len - len(ids)) + [1] * len(ids))
        ids_t = torch.tensor(input_ids, device=self.device)
        mask_t = torch.tensor(attention, device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=ids_t, attention_mask=mask_t, use_cache=True)
        next_ids = _sample_next(out.logits[:, -1, :], uncached[0].ticket.request.temperature, uncached[0].ticket.request.top_p)
        for idx, state in enumerate(uncached):
            state.past = _split_past(out.past_key_values, idx)
            state.past_len = max_len
            state.next_input_id = next_ids[idx]
            self.kv_cache.put(state.ticket.request.session_id, state.ticket.prompt_ids, state.past)

    def _decode_one_tick(self, active: list[ActiveState]) -> None:
        grouped: dict[int, list[ActiveState]] = defaultdict(list)
        for state in active:
            if state.ticket.cancelled or state.ticket.future.cancelled():
                state.finished = True
                continue
            if not state.finished and state.past is not None and state.next_input_id is not None:
                grouped[state.past_len].append(state)
        for states in grouped.values():
            self._decode_group(states)

    def _decode_group(self, states: list[ActiveState]) -> None:
        assert self.model is not None and self.tokenizer is not None
        input_ids = torch.tensor([[s.next_input_id] for s in states], device=self.device)
        past = _merge_past([s.past for s in states if s.past is not None])
        attention = torch.ones((len(states), states[0].past_len + 1), device=self.device, dtype=torch.long)
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, attention_mask=attention, past_key_values=past, use_cache=True)
        next_ids = _sample_next(out.logits[:, -1, :], states[0].ticket.request.temperature, states[0].ticket.request.top_p)
        for idx, state in enumerate(states):
            emitted = int(input_ids[idx, 0].item())
            state.generated.append(emitted)
            token = self.tokenizer.decode([emitted], skip_special_tokens=True)
            if token and state.ticket.stream is not None:
                if not self._emit_stream(state, token):
                    continue
            state.past = _split_past(out.past_key_values, idx)
            state.past_len += 1
            state.next_input_id = next_ids[idx]
            if emitted == self.tokenizer.eos_token_id or len(state.generated) >= state.ticket.max_new_tokens:
                self._finish_state(state)

    def _emit_stream(self, state: ActiveState, item: str | None) -> bool:
        if state.ticket.stream is None:
            return True
        try:
            state.ticket.stream.put_nowait(item)
            return True
        except asyncio.QueueFull:
            state.finished = True
            if not state.ticket.future.done():
                state.ticket.future.set_exception(RuntimeError("stream client is not reading fast enough"))
            return False

    def _finish_state(self, state: ActiveState) -> None:
        assert self.tokenizer is not None
        state.finished = True
        text = self.tokenizer.decode(state.generated, skip_special_tokens=True)
        if not state.ticket.future.done():
            state.ticket.future.set_result(text)
        if state.ticket.stream is not None:
            self._emit_stream(state, None)

    def _fail_states(self, states: list[ActiveState], exc: BaseException) -> None:
        for state in states:
            state.finished = True
            if not state.ticket.future.done():
                state.ticket.future.set_exception(exc)
            if state.ticket.stream is not None:
                try:
                    state.ticket.stream.put_nowait(None)
                except asyncio.QueueFull:
                    pass


def _sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> list[int]:
    if temperature <= 0:
        return logits.argmax(dim=-1).tolist()
    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(sorted_probs, 1).squeeze(-1)
        return sorted_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).tolist()
    return torch.multinomial(probs, 1).squeeze(-1).tolist()


def _split_past(past: Past, idx: int) -> Past:
    out = []
    for layer in past:
        if isinstance(layer, torch.Tensor):
            out.append(layer[idx : idx + 1].contiguous())
        else:
            out.append(tuple(t[idx : idx + 1].contiguous() if isinstance(t, torch.Tensor) else t for t in layer))
    return tuple(out)


def _merge_past(pasts: list[Past]) -> Past:
    merged = []
    for layer_items in zip(*pasts, strict=True):
        first = layer_items[0]
        if isinstance(first, torch.Tensor):
            merged.append(torch.cat(layer_items, dim=0))
        else:
            merged.append(tuple(torch.cat([item[i] for item in layer_items], dim=0) for i in range(len(first))))
    return tuple(merged)


def _detach_past(past: Past) -> Past:
    detached = []
    for layer in past:
        if isinstance(layer, torch.Tensor):
            detached.append(layer.detach())
        else:
            detached.append(tuple(t.detach() if isinstance(t, torch.Tensor) else t for t in layer))
    return tuple(detached)


def _past_nbytes(past: Past) -> int:
    total = 0
    for layer in past:
        tensors = [layer] if isinstance(layer, torch.Tensor) else [t for t in layer if isinstance(t, torch.Tensor)]
        total += sum(t.nelement() * t.element_size() for t in tensors)
    return total
