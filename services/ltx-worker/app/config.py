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
    port: int = 8102
    service_api_key: str = ""
    ltx_model_dir: Path = Field(
        default=Path("models/ltx-2.3"),
        validation_alias=AliasChoices("LTX_MODEL_DIR", "MODEL_DIR"),
    )
    gemma_root: Path | None = Field(default=None, validation_alias=AliasChoices("LTX_GEMMA_ROOT", "GEMMA_ROOT"))
    cuda_device: str = Field(default="cuda:0", validation_alias=AliasChoices("LTX_CUDA_DEVICE", "CUDA_DEVICE"))
    quantization: str = Field(default="fp8_cast", validation_alias=AliasChoices("LTX_QUANTIZATION", "QUANTIZATION"))
    torch_compile_ltx: bool = Field(
        default=False,
        validation_alias=AliasChoices("LTX_TORCH_COMPILE", "TORCH_COMPILE_LTX"),
    )
    local_storage_dir: Path = Path("storage/videos")
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""
    r2_public_base_url: str = ""
    max_media_bytes: int = 512 * 1024 * 1024

    def model_post_init(self, __context: object) -> None:
        self.ltx_model_dir = _resolve_path(self.ltx_model_dir)
        self.local_storage_dir = _resolve_path(self.local_storage_dir)
        if self.gemma_root is not None:
            self.gemma_root = _resolve_path(self.gemma_root)


settings = Settings()
