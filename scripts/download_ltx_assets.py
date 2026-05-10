from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


LTX_ALLOW = [
    "ltx-2.3-22b-dev.safetensors",
    "ltx-2.3-22b-distilled-1.1.safetensors",
    "ltx-2.3-22b-distilled.safetensors",
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    "ltx-2.3-22b-distilled-lora-384.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
    "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
]

DEFAULT_TEXT_MODEL_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"
BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class DownloadSpec:
    label: str
    repo_id: str
    local_dir: Path
    allow_patterns: list[str] | None = None


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def configure_hf_cache(cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    return Path(os.environ["HF_HUB_CACHE"]).expanduser().resolve()


def check_free_space(paths: list[Path], required_gb: float) -> None:
    if required_gb <= 0:
        return
    checked: set[Path] = set()
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(path)
        free_gb = usage.free / BYTES_PER_GB
        anchor = path.anchor
        key = Path(anchor) if anchor else path
        if key in checked:
            continue
        checked.add(key)
        if free_gb < required_gb:
            raise SystemExit(
                f"Not enough free disk for model downloads at {path}: "
                f"{free_gb:.1f} GiB free, require at least {required_gb:.1f} GiB. "
                "Move MODEL_DIR/TEXT_MODEL_DIR/HF_HOME to a larger volume or pass --min-free-gb 0 to override."
            )


def add_unique(specs: list[DownloadSpec], spec: DownloadSpec) -> None:
    for existing in specs:
        if existing.repo_id == spec.repo_id and existing.local_dir == spec.local_dir:
            print(f"Reusing {existing.local_dir} for {spec.label}; already scheduled as {existing.label}.")
            return
    specs.append(spec)


def download_snapshot(spec: DownloadSpec, cache_dir: Path | None, max_workers: int) -> None:
    from huggingface_hub import snapshot_download

    spec.local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {spec.label}: {spec.repo_id} -> {spec.local_dir}")
    snapshot_download(
        spec.repo_id,
        local_dir=spec.local_dir,
        allow_patterns=spec.allow_patterns,
        cache_dir=cache_dir,
        max_workers=max_workers,
        token=hf_token(),
    )


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and token.startswith("replace-with"):
        raise SystemExit(
            "HF_TOKEN is still a placeholder. Set HF_TOKEN to a Hugging Face token that has accepted access "
            "to google/gemma-3-12b-it-qat-q4_0-unquantized, or run `huggingface-cli login`."
        )
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def preflight_repo_access(specs: list[DownloadSpec]) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    token = hf_token()
    api = HfApi(token=token)
    for spec in specs:
        try:
            api.model_info(spec.repo_id, token=token)
        except GatedRepoError as exc:
            raise SystemExit(_gated_repo_message(spec.repo_id)) from exc
        except RepositoryNotFoundError as exc:
            raise SystemExit(f"Cannot find Hugging Face repo {spec.repo_id}. Check the model id and HF_TOKEN access.") from exc
        except Exception as exc:
            if "401" in str(exc) or "Unauthorized" in str(exc):
                raise SystemExit(_gated_repo_message(spec.repo_id)) from exc
            raise


def _gated_repo_message(repo_id: str) -> str:
    return (
        f"Cannot access gated Hugging Face repo {repo_id}.\n"
        "Fix:\n"
        f"  1. Open https://huggingface.co/{repo_id} and accept the model access terms with the account that owns your token.\n"
        "  2. Put a read token in the deployment env file, for example:\n"
        "       HF_TOKEN=hf_your_read_token\n"
        "     or login on the server with:\n"
        "       huggingface-cli login\n"
        "  3. Re-run the deploy command.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned LTX-2.3 and Gemma assets into the model cache.")
    parser.add_argument("--model-dir", default="models/ltx-2.3", help="LTX_MODEL_DIR")
    parser.add_argument("--text-model-id", default=DEFAULT_TEXT_MODEL_ID, help="TEXT_MODEL_ID")
    parser.add_argument(
        "--text-model-dir",
        default="models/text/gemma-3-12b-it-qat-q4_0-unquantized",
        help="TEXT_MODEL_DIR",
    )
    parser.add_argument(
        "--ltx-gemma-root",
        default=None,
        help="LTX_GEMMA_ROOT. Defaults to <model-dir>/gemma-3-12b unless set; point it at TEXT_MODEL_DIR to avoid duplicates.",
    )
    parser.add_argument("--hf-cache-dir", default=os.environ.get("HF_HOME"), help="HF_HOME/HF cache root on a large volume.")
    parser.add_argument("--min-free-gb", type=float, default=float(os.environ.get("MODEL_MIN_FREE_GB", "120")))
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("HF_HUB_DOWNLOAD_THREADS", "4")))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    parser.add_argument("--skip-gemma", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    model_dir = resolve_path(args.model_dir)
    text_model_dir = resolve_path(args.text_model_dir)
    ltx_gemma_root = resolve_path(args.ltx_gemma_root) if args.ltx_gemma_root else model_dir / "gemma-3-12b"
    hf_cache = configure_hf_cache(resolve_path(args.hf_cache_dir) if args.hf_cache_dir else None)

    specs: list[DownloadSpec] = []
    add_unique(specs, DownloadSpec("LTX-2.3 checkpoints", "Lightricks/LTX-2.3", model_dir, LTX_ALLOW))
    if not args.skip_gemma:
        add_unique(specs, DownloadSpec("LTX Gemma text encoder", args.text_model_id, ltx_gemma_root))
    if not args.skip_text:
        add_unique(specs, DownloadSpec("text worker model", args.text_model_id, text_model_dir))

    free_space_paths = [spec.local_dir for spec in specs]
    if hf_cache is not None:
        free_space_paths.append(hf_cache)
    check_free_space(free_space_paths, args.min_free_gb)
    preflight_repo_access(specs)

    for spec in specs:
        download_snapshot(spec, hf_cache, args.max_workers)

    print(f"LTX assets ready in {model_dir}")
    if not args.skip_text:
        print(f"Text model assets ready in {text_model_dir}")
    if not args.skip_gemma:
        print(f"LTX Gemma assets ready in {ltx_gemma_root}")


if __name__ == "__main__":
    main()
