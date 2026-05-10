# Ubuntu 24.04 H100 Deployment

This profile is for a production H100 server running the full LTX-2.3 22B BF16 pipeline, 30 to 40 inference steps, and the text model `google/gemma-3-12b-it-qat-q4_0-unquantized`.

It runs native processes instead of Docker for the Python workers so a preinstalled CUDA PyTorch environment can be used. Qdrant is installed as a native binary by `scripts/deploy_h100.sh`.

## Runtime Contract

- OS: Ubuntu 24.04 x86_64.
- GPU: NVIDIA H100 only. The service script fails on non-H100 GPUs unless `ALLOW_NON_H100=1` is set for explicit testing.
- NVIDIA driver: must expose CUDA 12.8 or newer in `nvidia-smi`.
- Python: 3.12.
- PyTorch: CUDA build with `torch.cuda.is_available() == True`. The deploy script creates `.venv-h100` with `--system-site-packages` so a preinstalled torch 2.8.0+cu128 runtime remains visible.
- LTX: full `ltx-2.3-22b-dev.safetensors`, `LTX_QUANTIZATION=none`, BF16 model execution, one heavy LTX job per H100.
- Text: `google/gemma-3-12b-it-qat-q4_0-unquantized` under `TEXT_MODEL_DIR`.
- LTX Gemma root defaults to `TEXT_MODEL_DIR` on H100 so the same 24GB Gemma assets are reused.

PyTorch's [official local install page](https://docs.pytorch.org/get-started/locally/) documents CUDA-enabled pip installs and CUDA availability verification. The [previous-version page](https://docs.pytorch.org/get-started/previous-versions/) also lists CUDA 12.8 wheel indexes for PyTorch 2.7.x; use your image's validated torch 2.8.0+cu128 build only if it passes the checks below.

## One-Time Server Setup

The deploy script performs the one-time setup, Qdrant installation, dependency installation, model checks, React frontend build, and gateway release build:

```bash
bash scripts/deploy_h100.sh deploy
```

Use these flags when needed:

- `--skip-apt`: skip apt package installation.
- `--skip-python-deps`: skip Python dependency installation.
- `--skip-models`: skip Hugging Face model downloads.
- `--install-pytorch`: install CUDA 12.8 PyTorch into the deployment venv instead of relying only on the preinstalled torch package.

The script creates `.venv-h100` with `--system-site-packages`, so preinstalled CUDA PyTorch from the base image remains visible.

To manually verify the preinstalled PyTorch H100 image:

```bash
python3.12 - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

If PyTorch is not already installed, install the CUDA 12.8 wheel set into the Python 3.12 environment:

```bash
python3.12 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Manual Qdrant installation, if needed outside the script:

```bash
sudo mkdir -p /opt/qdrant
curl -fsSL https://github.com/qdrant/qdrant/releases/download/v1.11.0/qdrant-x86_64-unknown-linux-musl.tar.gz \
  | sudo tar -xz -C /opt/qdrant
sudo chmod +x /opt/qdrant/qdrant
```

## Models

Authenticate to Hugging Face if the Gemma model requires approval in your account:

```bash
huggingface-cli login
```

Download the full LTX assets and text model:

```bash
python3.12 scripts/download_ltx_assets.py \
  --model-dir models/ltx-2.3 \
  --text-model-id google/gemma-3-12b-it-qat-q4_0-unquantized \
  --text-model-dir models/text/gemma-3-12b-it-qat-q4_0-unquantized \
  --ltx-gemma-root models/text/gemma-3-12b-it-qat-q4_0-unquantized \
  --hf-cache-dir /mnt/models/.cache/huggingface \
  --min-free-gb 120
```

Required LTX files:

- `models/ltx-2.3/ltx-2.3-22b-dev.safetensors`
- `models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
- `models/ltx-2.3/ltx-2.3-temporal-upscaler-x2-1.0.safetensors`
- `models/text/gemma-3-12b-it-qat-q4_0-unquantized/config.json` for both `TEXT_MODEL_DIR` and `LTX_GEMMA_ROOT`

## Environment

Create the H100 env file:

```bash
cp .env.h100.example .env.h100
chmod 600 .env.h100
```

Edit `.env.h100` and set real values for:

- `ADMIN_API_KEY` and `SERVICE_API_KEY` if you want to replace the generated values
- `TURSO_DB_URL` and `TURSO_AUTH_TOKEN` if using hosted Turso instead of local libSQL
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`, and `R2_PUBLIC_BASE_URL` for production artifact storage

Keep these H100 LTX settings:

```bash
GATEWAY_PROFILE=cloud_h100
LTX_CUDA_DEVICE=cuda:0
LTX_QUANTIZATION=none
LTX_LOCAL_MAX_HEAVY_JOBS=1
AI_PYTHON_BIN=python3.12
```

Use `--install-pytorch` only when the base image does not already provide a CUDA 12.8 PyTorch package.

## Service Control

Make the deployment script executable:

```bash
chmod +x scripts/deploy_h100.sh
```

Run the production readiness check:

```bash
scripts/deploy_h100.sh check
```

Start all services:

```bash
scripts/deploy_h100.sh start all
```

Restart all services:

```bash
scripts/deploy_h100.sh restart all
```

Stop all services:

```bash
scripts/deploy_h100.sh stop all
```

Check status and health:

```bash
scripts/deploy_h100.sh status all
```

Follow logs:

```bash
scripts/deploy_h100.sh logs gateway -f
scripts/deploy_h100.sh logs text -f
scripts/deploy_h100.sh logs ltx -f
scripts/deploy_h100.sh logs qdrant -f
```

Logs and PID files are stored under `runtime/h100/`.

## Video Request Defaults

For the full 22B BF16 H100 profile, submit text-to-video jobs with:

```json
{
  "mode": "text_to_video",
  "prompt": "A cinematic product shot with controlled lighting and realistic motion",
  "negative_prompt": "blur, low quality, distorted",
  "width": 1024,
  "height": 576,
  "num_frames": 97,
  "frame_rate": 24,
  "guidance_scale": 7.5,
  "num_inference_steps": 40
}
```

Use 30 steps for faster previews and 40 steps for production output. The gateway enforces dimensions divisible by 32 and frame counts in `8k+1` form.

## Operational Notes

- On a single H100, keep `LTX_LOCAL_MAX_HEAVY_JOBS=1`. Hundreds of clients can connect, but GPU-heavy work must be queued.
- The text worker and LTX worker both use CUDA. For maximum video throughput on a one-GPU server, stop the text worker while running sustained full 22B video batches.
- Video outputs are never response-cached. Each request receives a unique job ID, R2 key, and effective seed.
- Use `scripts/deploy_h100.sh stop ltx` before changing model files, then `scripts/deploy_h100.sh start ltx`.
