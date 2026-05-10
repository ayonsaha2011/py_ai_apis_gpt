# Ubuntu 24.04 H200 Deployment

Use `scripts/deploy_h200.sh` for H200 servers. It wraps the shared deployment script with H200 defaults:

```bash
bash scripts/deploy_h200.sh deploy
```

The H200 defaults are:

- `AI_ENV_FILE=.env.h200`
- `AI_RUNTIME_DIR=runtime/h200`
- `AI_VENV_DIR=.venv-h200`
- `GATEWAY_PROFILE=cloud_h200`

The H200 wrapper calls the shared H100/H200 deployment implementation through `bash`, so it does not require executable bits on `scripts/deploy_h100.sh`.

Gemma downloads require gated Hugging Face access. Put a real read token in `.env.h200`:

```bash
HF_TOKEN=hf_your_read_token
```

For a local smoke run, use `TURSO_DB_URL=file:storage/gateway.db`. For production Turso, set both `TURSO_DB_URL=libsql://...` and `TURSO_AUTH_TOKEN=...`; placeholders are rejected before services start.

For H100 servers, use `docs/h100-ubuntu-deploy.md` and `scripts/deploy_h100.sh`.

For full 22B BF16 H200 text-to-video, the gateway admits 5-second HD jobs at `1024x576` and `121` frames. Distilled and compatible specialized modes admit 10-second HD jobs at `1024x576` and `241` frames. Larger requests are rejected before GPU allocation.

4K is supported as output-only upscaling for 5-second jobs. The API response and history include `metadata.upscaled`, `render_width`, `render_height`, `output_width`, and `output_height` so operators can distinguish native render cost from final artifact size.

Systemd units are shared with the H100 path:

```bash
sudo cp infra/systemd/py-ai-*.service /etc/systemd/system/
sudo cp infra/systemd/py-ai-apis-gpt.logrotate /etc/logrotate.d/py-ai-apis-gpt
sudo systemctl daemon-reload
sudo systemctl enable --now py-ai-text-worker py-ai-ltx-worker py-ai-gateway
```

Supported H200 outcomes:

- Full 22B BF16: native HD 5 seconds, 4K output via upscale 5 seconds.
- Distilled/specialized: native HD 10 seconds, 4K output via upscale 5 seconds.
