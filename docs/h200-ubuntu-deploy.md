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

For H100 servers, use `docs/h100-ubuntu-deploy.md` and `scripts/deploy_h100.sh`.

For full 22B BF16 H200 text-to-video, the gateway admits 5-second HD jobs at `1024x576` and `121` frames. Distilled and compatible specialized modes admit 10-second HD jobs at `1024x576` and `241` frames. Larger requests are rejected before GPU allocation.
