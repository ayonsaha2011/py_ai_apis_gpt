# Ubuntu 24.04 H100 Deployment

This profile is for production H100 servers running LTX-2.3 22B BF16 with one heavy GPU job at a time and the text model `google/gemma-3-12b-it-qat-q4_0-unquantized`.

It runs native processes instead of Docker for the Python workers so a preinstalled CUDA PyTorch environment can be reused. Qdrant is installed as a native binary by `scripts/deploy_h100.sh`.

## Runtime Contract

- OS: Ubuntu 24.04 x86_64.
- GPU: NVIDIA H100. Use `scripts/deploy_h200.sh` for H200 servers.
- NVIDIA driver: must expose CUDA 12.8 or newer in `nvidia-smi`.
- Python: 3.12.
- PyTorch: CUDA build with `torch.cuda.is_available() == True`. The deploy script creates `.venv-h100` with `--system-site-packages` so a preinstalled torch 2.8.0+cu128 runtime remains visible.
- LTX: full `ltx-2.3-22b-dev.safetensors`, `LTX_QUANTIZATION=none`, BF16 model execution, one heavy LTX job per H100.
- Text: `google/gemma-3-12b-it-qat-q4_0-unquantized` under `TEXT_MODEL_DIR`.
- Profile: `GATEWAY_PROFILE=cloud_h100`.

## One-Time Server Setup

```bash
bash scripts/deploy_h100.sh deploy
```

Useful flags:

- `--skip-apt`: skip apt package installation.
- `--skip-python-deps`: skip Python dependency installation.
- `--skip-models`: skip Hugging Face model downloads.
- `--install-pytorch`: install CUDA 12.8 PyTorch into the deployment venv instead of relying only on preinstalled torch.
- `--hf-token hf_...`: pass a Hugging Face token for gated model downloads without editing `.env.h100`.

Example with a gated-model token:

```bash
bash scripts/deploy_h100.sh deploy --hf-token hf_your_read_token
```

## Environment

```bash
cp .env.h100.example .env.h100
chmod 600 .env.h100
```

Keep these H100 profile settings:

```bash
GATEWAY_PROFILE=cloud_h100
LTX_CUDA_DEVICE=cuda:0
LTX_QUANTIZATION=none
LTX_LOCAL_MAX_HEAVY_JOBS=1
AI_PYTHON_BIN=python3.12
```

Set real values for `ADMIN_API_KEY`, `SERVICE_API_KEY`, Turso, and R2 before exposing the service.

Gemma is a gated Hugging Face model. The deploy script installs and checks `huggingface-cli`/`hf` inside `.venv-h100`. Before model download, authenticate with one of these methods.

Option 1: pass the token directly to deploy:

```bash
bash scripts/deploy_h100.sh deploy --hf-token hf_your_read_token
```

Option 2: put a token in `.env.h100`:

```bash
HF_TOKEN=hf_your_read_token
```

Option 3: login from the deployment venv:

```bash
/workspace/py_ai_apis_gpt/.venv-h100/bin/huggingface-cli login
bash scripts/deploy_h100.sh deploy
```

The token account must have accepted access for `google/gemma-3-12b-it-qat-q4_0-unquantized`. If `HF_TOKEN` in `.env.h100` is still a placeholder, either replace it or remove that line before using CLI login.

For a local smoke run without Turso, set:

```bash
TURSO_DB_URL=file:storage/gateway.db
TURSO_AUTH_TOKEN=
```

For production Turso, both `TURSO_DB_URL=libsql://...` and `TURSO_AUTH_TOKEN=...` must be real values. The deploy script now rejects placeholders before starting services.

## Service Control

```bash
bash scripts/deploy_h100.sh check
bash scripts/deploy_h100.sh deploy --hf-token hf_your_read_token
bash scripts/deploy_h100.sh start all
bash scripts/deploy_h100.sh restart all
bash scripts/deploy_h100.sh status all
bash scripts/deploy_h100.sh logs ltx -f
bash scripts/deploy_h100.sh stop all
```

Logs and PID files are stored under `runtime/h100/`.

## Video Request Defaults

For the full 22B BF16 H100 profile, use the safer SD budget:

```json
{
  "mode": "text_to_video",
  "prompt": "A cinematic product shot with controlled lighting and realistic motion",
  "negative_prompt": "blur, low quality, distorted",
  "width": 768,
  "height": 448,
  "num_frames": 121,
  "frame_rate": 24,
  "guidance_scale": 7.5,
  "num_inference_steps": 40
}
```

For HD on H100, use `mode=distilled` with `1024x576` and `121` frames. For 10-second HD clips, use the H200 profile.

4K support on H100 is output-only upscaling. The request can use `3840x2160` for 5 seconds, but the worker records a safe native render size in job metadata:

- `metadata.upscaled=true`
- `metadata.render_width` and `metadata.render_height`: native GPU render size
- `metadata.output_width=3840`
- `metadata.output_height=2160`

## Systemd Mode

The native script is the default operator interface. For systemd-managed production, copy the prepared units after `deploy` succeeds:

```bash
sudo cp infra/systemd/py-ai-*.service /etc/systemd/system/
sudo cp infra/systemd/py-ai-apis-gpt.logrotate /etc/logrotate.d/py-ai-apis-gpt
sudo systemctl daemon-reload
sudo systemctl enable --now py-ai-text-worker py-ai-ltx-worker py-ai-gateway
```

The units expect the repo at `/workspace/py_ai_apis_gpt`; edit `WorkingDirectory`, `EnvironmentFile`, and `ExecStart` if the checkout path differs.

## Operational Notes

- Keep `LTX_LOCAL_MAX_HEAVY_JOBS=1` on a single H100.
- The gateway can accept many clients concurrently, but video work is queued and admitted by the GPU budget.
- Video outputs are never response-cached. Each request receives a unique job ID, R2 key, and effective seed.
- Supported H100 outcomes:
  - Full 22B BF16: native SD 5 seconds, 4K output via upscale 5 seconds.
  - Distilled/specialized: native HD 5 seconds, 4K output via upscale 5 seconds.
