# py_ai_apis_gpt

Production-oriented multi-model inference stack with a Rust gateway, custom PyTorch text worker, and LTX-2.3 video worker.

## Components

- `gateway-rs`: Axum gateway for auth, Turso/libSQL state, routing, fair queues, GPU/service administration, RAG ownership, video job lifecycle, and SSE.
- `services/text-worker`: FastAPI custom PyTorch text server with request admission, continuous decode batching, managed KV cache, per-session prefix reuse, streaming, and Qdrant-backed RAG.
- `services/ltx-worker`: FastAPI LTX-2.3 worker using the official `Lightricks/LTX-2` package at pinned commit `41d924371612b692c0fd1e4d9d94c3dfb3c02cb3`.
- `frontend`: Static operations UI for auth, chat, RAG, video jobs, history, and admin controls. The Rust gateway serves it from `FRONTEND_DIR`.
- `postman`: Postman collection and local/cloud environments for all public gateway routes.
- `infra`: Docker Compose profiles for local and cloud GPU deployments.
- `scripts`: setup, model download, and local start scripts.

## Local Start

1. Copy `.env.example` to `.env` and set strong `ADMIN_API_KEY` and `SERVICE_API_KEY`.
2. Install Python 3.12 for LTX. The current machine has Python 3.11.9, so install 3.12 before running the LTX worker.
3. Download LTX and text model assets:

```powershell
python scripts\download_ltx_assets.py --model-dir models\ltx-2.3
```

The default text model is `google/gemma-3-12b-it-qat-q4_0-unquantized`, cached at `TEXT_MODEL_DIR=models/text/gemma-3-12b-it-qat-q4_0-unquantized`. Point `LTX_GEMMA_ROOT` at the same directory to avoid storing a second 24GB Gemma copy for LTX prompt conditioning.

4. Start services in separate terminals:

```powershell
scripts\dev-local.ps1 gateway
scripts\dev-local.ps1 text
scripts\dev-local.ps1 ltx
scripts\dev-local.ps1 frontend
```

Open `http://127.0.0.1:5173` for the React frontend during development. To serve the built React app directly from the Rust gateway, run `scripts\dev-local.ps1 gateway-static` and open `http://127.0.0.1:8080`. The local RTX 5090 profile admits one GPU-heavy workload at a time and queues excess work with SSE progress.

## Postman

Import:

- `postman/gateway.postman_collection.json`
- `postman/local.postman_environment.json`
- `postman/cloud.postman_environment.json`

Run `Auth / Register` or `Auth / Login` first; the collection stores `access_token` automatically. Video job creation stores `video_job_id`, and RAG ingest stores `document_id`.

## Verification

```powershell
cargo test --manifest-path gateway-rs\Cargo.toml
uv run --project services/text-worker pytest
uv run --project services/ltx-worker pytest
```

GPU integration tests require the real model assets and are marked separately.

## Cloud GPU Deployment

For Ubuntu 24.04 H100 servers, use:

- `docs/h100-ubuntu-deploy.md`
- `.env.h100.example`
- `scripts/deploy_h100.sh`

For Ubuntu 24.04 H200 servers, use:

- `docs/h200-ubuntu-deploy.md`
- `.env.h200.example`
- `scripts/deploy_h200.sh`

For Ubuntu 24.04 B200/GB200-class servers, use:

- `docs/b200-ubuntu-deploy.md`
- `.env.b200.example`
- `scripts/deploy_b200.sh`

The profiles are separate. H100 defaults to `GATEWAY_PROFILE=cloud_h100`, `.env.h100`, `.venv-h100`, and `runtime/h100`. H200 defaults to `GATEWAY_PROFILE=cloud_h200`, `.env.h200`, `.venv-h200`, and `runtime/h200`. B200 defaults to `GATEWAY_PROFILE=cloud_b200`, `.env.b200`, `.venv-b200`, and `runtime/b200`, with native `1920x1088@481f` as the default 20-second Full-HD LTX-2.3 dev target.

The first registered user is promoted to `admin` automatically. Additional bootstrap admins can be configured with `ADMIN_EMAILS=email1@example.com,email2@example.com` before registration.
