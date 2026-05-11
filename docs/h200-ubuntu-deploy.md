# Ubuntu 24.04 H200 Deployment

Use `scripts/deploy_h200.sh` for H200 servers. It wraps the shared deployment script with H200 defaults:

```bash
bash scripts/deploy_h200.sh deploy
```

For gated Hugging Face model access without editing `.env.h200`:

```bash
bash scripts/deploy_h200.sh deploy --hf-token hf_your_read_token
```

The H200 defaults are:

- `AI_ENV_FILE=.env.h200`
- `AI_RUNTIME_DIR=runtime/h200`
- `AI_VENV_DIR=.venv-h200`
- `GATEWAY_PROFILE=cloud_h200`

The H200 wrapper calls the shared H100/H200 deployment implementation through `bash`, so it does not require executable bits on `scripts/deploy_h100.sh`.

Gemma downloads require gated Hugging Face access. The deploy script installs and checks `huggingface-cli`/`hf` inside `.venv-h200`. Authenticate with one of these methods.

Option 1: pass the token directly:

```bash
bash scripts/deploy_h200.sh deploy --hf-token hf_your_read_token
```

Option 2: put a real read token in `.env.h200`:

```bash
HF_TOKEN=hf_your_read_token
```

Option 3: login from the deployment venv:

```bash
/workspace/py_ai_apis_gpt/.venv-h200/bin/huggingface-cli login
# Newer Hugging Face CLI also supports:
/workspace/py_ai_apis_gpt/.venv-h200/bin/hf auth login
bash scripts/deploy_h200.sh deploy
```

The token account must have accepted access for `google/gemma-3-12b-it-qat-q4_0-unquantized`. If `HF_TOKEN` in `.env.h200` is still a placeholder, either replace it or remove that line before using CLI login.
The deploy script installs `huggingface_hub[cli]` and `hf_transfer` before model downloads, including when `HF_HUB_ENABLE_HF_TRANSFER=1`.

For a local smoke run, use `TURSO_DB_URL=file:storage/gateway.db`. For production Turso, set both `TURSO_DB_URL=libsql://...` and `TURSO_AUTH_TOKEN=...`; placeholders are rejected before services start.

For H100 servers, use `docs/h100-ubuntu-deploy.md` and `scripts/deploy_h100.sh`.

After `git pull`, restart the gateway to pick up Rust gateway changes. The H200 wrapper rebuilds the release binary automatically when Rust source, migrations, or Cargo metadata are newer than the current binary:

```bash
bash scripts/deploy_h200.sh restart gateway
```

For full 22B BF16 H200 text-to-video, the gateway admits 5-second HD jobs at `1024x576` and `121` frames. The LTX worker keeps one full dev pipeline on GPU, so distilled and specialized modes are rejected before GPU allocation. A single 20-second HD job is not supported by this full-dev-only profile; it needs a future chunk-and-stitch workflow that runs multiple 5-second segments and joins them after generation.

4K and Full HD output sizes above the native render budget are supported as output-only upscaling for 5-second jobs. The API response and history include `metadata.upscaled`, `render_width`, `render_height`, `output_width`, and `output_height` so operators can distinguish native render cost from final artifact size.

Systemd units are shared with the H100 path:

```bash
sudo cp infra/systemd/py-ai-*.service /etc/systemd/system/
sudo cp infra/systemd/py-ai-apis-gpt.logrotate /etc/logrotate.d/py-ai-apis-gpt
sudo systemctl daemon-reload
sudo systemctl enable --now py-ai-text-worker py-ai-ltx-worker py-ai-gateway
```

Supported H200 outcomes:

- Full 22B BF16: native HD 5 seconds, 4K output via upscale 5 seconds.
- Distilled/specialized: disabled to avoid loading a second model.
