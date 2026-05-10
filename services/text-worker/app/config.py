from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "gateway-rs").is_dir() and (parent / "services").is_dir():
            return parent
    return Path.cwd()


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (_repo_root() / path).resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[_repo_root() / ".env", _repo_root() / ".env.local", ".env", ".env.local"],
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8101
    service_api_key: str = ""
    model_id: str = Field(
        default="google/gemma-3-12b-it-qat-q4_0-unquantized",
        validation_alias=AliasChoices("TEXT_MODEL_ID", "MODEL_ID"),
    )
    text_model_dir: Path = Field(
        default=Path("models/text/gemma-3-12b-it-qat-q4_0-unquantized"),
        validation_alias=AliasChoices("TEXT_MODEL_DIR", "MODEL_DIR"),
    )
    device: str = Field(default="cuda", validation_alias=AliasChoices("TEXT_DEVICE", "DEVICE"))
    dtype: str = Field(default="bfloat16", validation_alias=AliasChoices("TEXT_DTYPE", "DTYPE"))
    attn_implementation: str = Field(
        default="sdpa",
        validation_alias=AliasChoices("TEXT_ATTN_IMPLEMENTATION", "ATTN_IMPLEMENTATION"),
    )
    max_waiting: int = Field(
        default=512,
        validation_alias=AliasChoices("TEXT_MAX_WAITING", "MAX_WAITING"),
    )
    max_active: int = Field(default=64, validation_alias=AliasChoices("TEXT_MAX_ACTIVE", "MAX_ACTIVE"))
    max_batch_tokens: int = Field(
        default=4096,
        validation_alias=AliasChoices("TEXT_MAX_BATCH_TOKENS", "MAX_BATCH_TOKENS"),
    )
    max_new_tokens: int = Field(
        default=2048,
        validation_alias=AliasChoices("TEXT_MAX_NEW_TOKENS", "MAX_NEW_TOKENS"),
    )
    max_input_tokens: int = Field(
        default=8192,
        validation_alias=AliasChoices("TEXT_MAX_INPUT_TOKENS", "MAX_INPUT_TOKENS"),
    )
    scheduler_tick_ms: int = Field(
        default=2,
        validation_alias=AliasChoices("TEXT_SCHEDULER_TICK_MS", "SCHEDULER_TICK_MS"),
    )
    kv_cache_bytes: int = Field(
        default=8 * 1024 * 1024 * 1024,
        validation_alias=AliasChoices("TEXT_KV_CACHE_BYTES", "KV_CACHE_BYTES"),
    )
    kv_cache_ttl_seconds: int = Field(
        default=600,
        validation_alias=AliasChoices("TEXT_KV_CACHE_TTL_SECONDS", "KV_CACHE_TTL_SECONDS"),
    )
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str = ""
    rag_embed_model: str = "BAAI/bge-small-en-v1.5"
    rag_vector_size: int = 384
    rag_top_k: int = 5
    rag_score_threshold: float = 0.3
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64

    def model_post_init(self, __context: object) -> None:
        self.text_model_dir = _resolve_path(self.text_model_dir)


settings = Settings()
