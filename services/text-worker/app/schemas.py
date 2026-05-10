from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False
    session_id: str | None = None
    use_rag: bool = False
    rag_collection: str | None = None


class SchedulerResult(BaseModel):
    text: str
    finish_reason: str = "stop"


class RagCollectionRequest(BaseModel):
    collection: str


class RagIngestRequest(BaseModel):
    collection: str
    texts: list[str]
    source_name: str = "api"
    metadata: dict = Field(default_factory=dict)

