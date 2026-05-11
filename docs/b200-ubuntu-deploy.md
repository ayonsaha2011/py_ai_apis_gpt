# Ubuntu 24.04 B200 Deployment

Use `scripts/deploy_b200.sh` for B200 or GB200-class servers. It wraps the shared cloud deployment script with B200 defaults:

```bash
bash scripts/deploy_b200.sh deploy
```

For gated Hugging Face model access without editing `.env.b200`:

```bash
bash scripts/deploy_b200.sh deploy --hf-token hf_your_read_token
```

The B200 defaults are:

- `AI_ENV_FILE=.env.b200`
- `AI_RUNTIME_DIR=runtime/b200`
- `AI_VENV_DIR=.venv-b200`
- `GATEWAY_PROFILE=cloud_b200`
- `LTX_QUANTIZATION=none`
- `LTX_MAX_TOKENS_B200=2100000`
- `LTX_MAX_NUM_FRAMES=481`

The B200 profile keeps one LTX-2.3 full-dev BF16 pipeline on GPU and preloads it at service startup (`LTX_PRELOAD_ON_START=true`). The text worker also stays on CUDA by default with `TEXT_DEVICE=cuda:0`; use `TEXT_DEVICE=cuda:1` only on multi-GPU hosts where the LTX worker should own `cuda:0`.

Default B200 video target:

```json
{
  "mode": "text_to_video",
  "prompt": "A cinematic full-HD continuous shot...",
  "width": 1920,
  "height": 1088,
  "num_frames": 481,
  "num_inference_steps": 40,
  "guidance_scale": 7.5
}
```

`1920x1088` is used instead of `1920x1080` because LTX native dimensions must be divisible by 32. At 24fps, `481` frames is 20 seconds in the required `8k+1` frame-count form.

Gemma downloads require gated Hugging Face access. Authenticate with one of these methods:

```bash
bash scripts/deploy_b200.sh deploy --hf-token hf_your_read_token
```

or put a real read token in `.env.b200`:

```bash
HF_TOKEN=hf_your_read_token
```

For a local smoke run, use `TURSO_DB_URL=file:storage/gateway.db`. For production Turso, set both `TURSO_DB_URL=libsql://...` and `TURSO_AUTH_TOKEN=...`; placeholders are rejected before services start.

Operational commands:

```bash
bash scripts/deploy_b200.sh check
bash scripts/deploy_b200.sh start all
bash scripts/deploy_b200.sh restart gateway
bash scripts/deploy_b200.sh logs ltx -f
bash scripts/deploy_b200.sh stop all
```

Supported B200 outcomes:

- Full 22B BF16: native Full-HD 20 seconds at `1920x1088@481f`.
- 4K output: supported via upscale for 5-second jobs.
- Distilled/specialized: disabled to avoid loading a second LTX model.
