from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned LTX-2.3 and text model assets into the local model cache.")
    parser.add_argument("--model-dir", default="models/ltx-2.3", help="LTX_MODEL_DIR")
    parser.add_argument("--text-model-id", default=DEFAULT_TEXT_MODEL_ID, help="TEXT_MODEL_ID")
    parser.add_argument(
        "--text-model-dir",
        default="models/text/gemma-3-12b-it-qat-q4_0-unquantized",
        help="TEXT_MODEL_DIR",
    )
    parser.add_argument("--skip-gemma", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download("Lightricks/LTX-2.3", local_dir=model_dir, allow_patterns=LTX_ALLOW)
    if not args.skip_gemma:
        snapshot_download(
            args.text_model_id,
            local_dir=model_dir / "gemma-3-12b",
        )
    if not args.skip_text:
        text_model_dir = Path(args.text_model_dir)
        text_model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(args.text_model_id, local_dir=text_model_dir)
        print(f"Text model assets ready in {text_model_dir.resolve()}")
    print(f"LTX assets ready in {model_dir.resolve()}")


if __name__ == "__main__":
    main()
