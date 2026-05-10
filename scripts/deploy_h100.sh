#!/usr/bin/env bash
# H100 Ubuntu 24.04 deployment and operations script.
# Target image: preinstalled CUDA PyTorch, for example torch 2.8.0+cu128 on CUDA 12.8.
# Usage:
#   bash scripts/deploy_h100.sh deploy [--skip-models] [--skip-apt] [--skip-python-deps] [--install-pytorch]
#   bash scripts/deploy_h100.sh check
#   bash scripts/deploy_h100.sh start|stop|restart|status|logs [all|gateway|text|ltx|qdrant]

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${AI_ENV_FILE:-"$ROOT_DIR/.env.h100"}"
RUNTIME_DIR="${AI_RUNTIME_DIR:-"$ROOT_DIR/runtime/h100"}"
PID_DIR="$RUNTIME_DIR/pids"
LOG_DIR="$RUNTIME_DIR/logs"
VENV_DIR="${AI_VENV_DIR:-"$ROOT_DIR/.venv-h100"}"
PYTHON_BIN="${AI_PYTHON_BIN:-python3.12}"
QDRANT_VERSION="${QDRANT_VERSION:-1.11.0}"
QDRANT_BIN="${QDRANT_BIN:-/opt/qdrant/qdrant}"
QDRANT_STORAGE="${QDRANT_STORAGE:-"$ROOT_DIR/storage/qdrant"}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${GREEN}[deploy]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
fail() { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage:
  scripts/deploy_h100.sh deploy [--skip-models] [--skip-apt] [--skip-python-deps] [--install-pytorch]
  scripts/deploy_h100.sh setup  [--skip-models] [--skip-apt] [--skip-python-deps] [--install-pytorch]
  scripts/deploy_h100.sh check
  scripts/deploy_h100.sh start   [all|gateway|text|ltx|qdrant]
  scripts/deploy_h100.sh stop    [all|gateway|text|ltx|qdrant]
  scripts/deploy_h100.sh restart [all|gateway|text|ltx|qdrant]
  scripts/deploy_h100.sh status  [all|gateway|text|ltx|qdrant]
  scripts/deploy_h100.sh logs    [all|gateway|text|ltx|qdrant] [-f] [-n lines]

Default deploy action:
  setup OS packages, Rust, uv, native Qdrant, Python deps, env file,
  model cache, frontend build, production gateway build, full checks.

Environment:
  AI_ENV_FILE       Defaults to .env.h100
  AI_VENV_DIR       Defaults to .venv-h100
  AI_PYTHON_BIN     Defaults to python3.12
  QDRANT_BIN        Defaults to /opt/qdrant/qdrant
  QDRANT_VERSION    Defaults to 1.11.0
USAGE
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

sudo_cmd() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

load_env_if_present() {
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi

  export GATEWAY_BIND="${GATEWAY_BIND:-0.0.0.0:8080}"
  export GATEWAY_PROFILE="${GATEWAY_PROFILE:-cloud_h100}"
  export RUNTIME_BACKEND="${RUNTIME_BACKEND:-native}"
  export SERVICE_LOG_DIR="${SERVICE_LOG_DIR:-$LOG_DIR}"
  export LOCAL_STORAGE_DIR="${LOCAL_STORAGE_DIR:-storage}"
  export FRONTEND_DIR="${FRONTEND_DIR:-frontend/dist}"
  export TEXT_WORKER_URL="${TEXT_WORKER_URL:-http://127.0.0.1:8101}"
  export LTX_WORKER_URL="${LTX_WORKER_URL:-http://127.0.0.1:8102}"
  export TEXT_MODEL_ID="${TEXT_MODEL_ID:-google/gemma-3-12b-it-qat-q4_0-unquantized}"
  export TEXT_MODEL_DIR="${TEXT_MODEL_DIR:-models/text/gemma-3-12b-it-qat-q4_0-unquantized}"
  export TEXT_MODEL_REGISTRY="${TEXT_MODEL_REGISTRY:-google/gemma-3-12b-it-qat-q4_0-unquantized}"
  export LTX_MODEL_DIR="${LTX_MODEL_DIR:-models/ltx-2.3}"
  export LTX_GEMMA_ROOT="${LTX_GEMMA_ROOT:-$TEXT_MODEL_DIR}"
  export HF_HOME="${HF_HOME:-$ROOT_DIR/models/.cache/huggingface}"
  export MODEL_MIN_FREE_GB="${MODEL_MIN_FREE_GB:-120}"
  export HF_HUB_DOWNLOAD_THREADS="${HF_HUB_DOWNLOAD_THREADS:-4}"
  export LTX_CUDA_DEVICE="${LTX_CUDA_DEVICE:-cuda:0}"
  export LTX_QUANTIZATION="${LTX_QUANTIZATION:-none}"
  export LTX_TORCH_COMPILE="${LTX_TORCH_COMPILE:-false}"
  export LTX_LOCAL_MAX_HEAVY_JOBS="${LTX_LOCAL_MAX_HEAVY_JOBS:-1}"
  export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
  export QDRANT_API_KEY="${QDRANT_API_KEY:-}"
  export QDRANT_MANAGED="${QDRANT_MANAGED:-native}"
  export QDRANT_VERSION="${QDRANT_VERSION:-1.11.0}"
  export QDRANT_BIN="${QDRANT_BIN:-/opt/qdrant/qdrant}"
  if [[ "${QDRANT_STORAGE:-storage/qdrant}" = /* ]]; then
    export QDRANT_STORAGE="${QDRANT_STORAGE:-storage/qdrant}"
  else
    export QDRANT_STORAGE="$ROOT_DIR/${QDRANT_STORAGE:-storage/qdrant}"
  fi
  export AI_PYTHON_MODE="system"
  export AI_PYTHON_BIN="$VENV_DIR/bin/python"
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

require_env_file() {
  [[ -f "$ENV_FILE" ]] || fail "Missing $ENV_FILE. Run: bash scripts/deploy_h100.sh setup"
}

ensure_dirs() {
  mkdir -p "$PID_DIR" "$LOG_DIR" "$ROOT_DIR/storage" "$ROOT_DIR/models" "$QDRANT_STORAGE"
}

repo_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$ROOT_DIR" "$1" ;;
  esac
}

install_apt_packages() {
  info "Installing Ubuntu packages..."
  sudo_cmd apt-get update -qq
  sudo_cmd apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl ffmpeg git git-lfs jq libffi-dev libssl-dev \
    pkg-config python3-pip python3.12 python3.12-venv tmux unzip
  git lfs install
}

install_rust() {
  if command -v cargo >/dev/null 2>&1; then
    info "Rust already installed: $(cargo --version)"
    return
  fi
  info "Installing Rust toolchain..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    info "uv already installed: $(uv --version)"
    return
  fi
  info "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

install_node() {
  if command -v node >/dev/null 2>&1; then
    local major
    major="$(node --version | sed 's/^v//' | cut -d. -f1)"
    if (( major >= 20 )); then
      info "Node.js already installed: $(node --version)"
      return
    fi
  fi
  info "Installing Node.js 22..."
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo_cmd bash -
  sudo_cmd apt-get install -y --no-install-recommends nodejs
}

install_qdrant_binary() {
  if [[ -x "$QDRANT_BIN" ]]; then
    info "Qdrant already installed at $QDRANT_BIN"
    return
  fi
  info "Installing native Qdrant $QDRANT_VERSION..."
  local tmp
  tmp="$(mktemp -d)"
  local url="https://github.com/qdrant/qdrant/releases/download/v${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-musl.tar.gz"
  curl -fsSL "$url" -o "$tmp/qdrant.tar.gz"
  sudo_cmd mkdir -p "$(dirname "$QDRANT_BIN")"
  sudo_cmd tar -xzf "$tmp/qdrant.tar.gz" -C "$(dirname "$QDRANT_BIN")"
  sudo_cmd chmod +x "$QDRANT_BIN"
  rm -rf "$tmp"
}

create_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    warn "$ENV_FILE already exists; not overwriting."
    return
  fi
  local template="$ROOT_DIR/.env.h100.example"
  if [[ "$ENV_FILE" == *".env.h200" && -f "$ROOT_DIR/.env.h200.example" ]]; then
    template="$ROOT_DIR/.env.h200.example"
  fi
  info "Creating $ENV_FILE from $(basename "$template")..."
  cp "$template" "$ENV_FILE"
  local admin_key service_key
  admin_key="$(make_secret)"
  service_key="$(make_secret)"
  sed -i "s|^ADMIN_API_KEY=.*|ADMIN_API_KEY=$admin_key|" "$ENV_FILE"
  sed -i "s|^SERVICE_API_KEY=.*|SERVICE_API_KEY=$service_key|" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  info "Generated ADMIN_API_KEY and SERVICE_API_KEY in $ENV_FILE"
}

make_secret() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -base64 48 | tr -d '\n=' | tr '+/' '-_'
  else
    "$PYTHON_BIN" - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
  fi
}

create_venv() {
  info "Creating Python venv at $VENV_DIR with system site packages..."
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR" --system-site-packages
  fi
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
}

install_pytorch_if_requested() {
  local install_pytorch="$1"
  if [[ "$install_pytorch" != "1" ]]; then
    return
  fi
  info "Installing CUDA 12.8 PyTorch into $VENV_DIR..."
  "$VENV_DIR/bin/python" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
}

install_python_deps() {
  info "Installing Python dependencies without replacing system CUDA PyTorch..."
  "$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/infra/requirements-h100-system.txt"
}

download_models() {
  require_huggingface_access
  info "Downloading LTX-2.3 and Gemma model assets..."
  local hf_args=()
  if [[ -n "${HF_TOKEN:-}" && "${HF_TOKEN:-}" != replace-* ]]; then
    hf_args+=(--hf-token "$HF_TOKEN")
  elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" && "${HUGGING_FACE_HUB_TOKEN:-}" != replace-* ]]; then
    hf_args+=(--hf-token "$HUGGING_FACE_HUB_TOKEN")
  fi
  "$VENV_DIR/bin/python" "$ROOT_DIR/scripts/download_ltx_assets.py" \
    --model-dir "$(repo_path "${LTX_MODEL_DIR:-models/ltx-2.3}")" \
    --text-model-id "${TEXT_MODEL_ID:-google/gemma-3-12b-it-qat-q4_0-unquantized}" \
    --text-model-dir "$(repo_path "${TEXT_MODEL_DIR:-models/text/gemma-3-12b-it-qat-q4_0-unquantized}")" \
    --ltx-gemma-root "$(repo_path "${LTX_GEMMA_ROOT:-${TEXT_MODEL_DIR:-models/text/gemma-3-12b-it-qat-q4_0-unquantized}}")" \
    --hf-cache-dir "${HF_HOME:-$ROOT_DIR/models/.cache/huggingface}" \
    --min-free-gb "${MODEL_MIN_FREE_GB:-120}" \
    --max-workers "${HF_HUB_DOWNLOAD_THREADS:-4}" \
    "${hf_args[@]}"
}

build_gateway() {
  require_cmd cargo
  info "Building Rust gateway release binary..."
  cargo build --manifest-path "$ROOT_DIR/gateway-rs/Cargo.toml" --release
}

build_frontend() {
  require_cmd npm
  info "Building React frontend..."
  if [[ -f "$ROOT_DIR/frontend/package-lock.json" ]]; then
    npm --prefix "$ROOT_DIR/frontend" ci
  else
    npm --prefix "$ROOT_DIR/frontend" install
  fi
  npm --prefix "$ROOT_DIR/frontend" run build
}

require_secret() {
  local key="$1"
  local value="${!key:-}"
  [[ -n "$value" && "$value" != replace-* ]] || fail "$key must be set to a real value in $ENV_FILE"
}

require_huggingface_access() {
  local token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
  if [[ -n "$token" && "$token" == replace-* ]]; then
    fail "HF_TOKEN is still a placeholder in $ENV_FILE. Set HF_TOKEN to a Hugging Face read token with access to google/gemma-3-12b-it-qat-q4_0-unquantized, or remove it and run huggingface-cli login."
  fi
  if [[ -z "$token" ]]; then
    warn "HF_TOKEN is not set. The downloader will use any token from huggingface-cli login; Gemma downloads will fail if no cached token has accepted access."
  fi
}

check_gateway_database_env() {
  local url="${TURSO_DB_URL:-}"
  local token="${TURSO_AUTH_TOKEN:-}"
  [[ -n "$url" ]] || fail "TURSO_DB_URL must be set. Use a real libsql:// URL with TURSO_AUTH_TOKEN, or file:storage/gateway.db for a local smoke run."
  if [[ "$url" == *replace-* ]]; then
    fail "TURSO_DB_URL is still a placeholder in $ENV_FILE. Set a real Turso URL or use TURSO_DB_URL=file:storage/gateway.db for local testing."
  fi
  if [[ "$url" == libsql://* ]]; then
    [[ -n "$token" && "$token" != replace-* ]] || fail "TURSO_AUTH_TOKEN must be set for TURSO_DB_URL=$url. For local testing, set TURSO_DB_URL=file:storage/gateway.db and leave TURSO_AUTH_TOKEN empty."
  fi
}

check_os() {
  [[ -f /etc/os-release ]] || fail "Cannot detect OS"
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]] || fail "Expected Ubuntu 24.04, got ${PRETTY_NAME:-unknown}"
}

check_gpu() {
  require_cmd nvidia-smi
  local names
  names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  local expected="H100|H200"
  if [[ "${GATEWAY_PROFILE:-}" == *"h200"* ]]; then
    expected="H200"
  elif [[ "${GATEWAY_PROFILE:-}" == *"h100"* ]]; then
    expected="H100"
  fi
  grep -Eqi "$expected" <<<"$names" || fail "Expected $expected GPU for GATEWAY_PROFILE=${GATEWAY_PROFILE:-unset}, detected: ${names:-none}"
  local cuda
  cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1)"
  [[ -n "$cuda" ]] || fail "Could not read CUDA version from nvidia-smi"
  local major minor
  major="${cuda%%.*}"
  minor="${cuda#*.}"
  minor="${minor%%.*}"
  if (( major < 12 || (major == 12 && minor < 8) )); then
    fail "Need NVIDIA driver exposing CUDA 12.8+, got $cuda"
  fi
  info "GPU OK: ${names//$'\n'/, } CUDA driver $cuda"
}

check_torch() {
  local py="$VENV_DIR/bin/python"
  [[ -x "$py" ]] || fail "Missing venv python at $py"
  "$py" - <<'PY'
import os
import sys
import torch

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
name = torch.cuda.get_device_name(0)
print(f"gpu0={name}")
profile = os.environ.get("GATEWAY_PROFILE", "cloud_h100").lower()
expected = ("H200",) if "h200" in profile else ("H100",) if "h100" in profile else ("H100", "H200")
if not any(item in name for item in expected):
    raise SystemExit(f"Expected {' or '.join(expected)} GPU for GATEWAY_PROFILE={profile}, got {name}")
if not str(torch.version.cuda or "").startswith("12.8"):
    raise SystemExit(f"Expected CUDA 12.8 PyTorch build, got {torch.version.cuda}")
PY
}

check_imports() {
  "$VENV_DIR/bin/python" - <<'PY'
import fastapi
import qdrant_client
import transformers
import uvicorn
import ltx_core
import ltx_pipelines

major = int(transformers.__version__.split(".", 1)[0])
if major >= 5:
    raise SystemExit(f"transformers<5 required by LTX Gemma adapter, got {transformers.__version__}")
print(f"transformers={transformers.__version__}")
print("python imports OK")
PY
}

check_models() {
  local ltx_dir
  local text_dir
  local gemma_dir
  ltx_dir="$(repo_path "${LTX_MODEL_DIR:-models/ltx-2.3}")"
  text_dir="$(repo_path "${TEXT_MODEL_DIR:-models/text/gemma-3-12b-it-qat-q4_0-unquantized}")"
  gemma_dir="$(repo_path "${LTX_GEMMA_ROOT:-${TEXT_MODEL_DIR:-models/text/gemma-3-12b-it-qat-q4_0-unquantized}}")"
  [[ -f "$ltx_dir/ltx-2.3-22b-dev.safetensors" ]] || fail "Missing full LTX 22B dev checkpoint: $ltx_dir/ltx-2.3-22b-dev.safetensors"
  [[ -f "$gemma_dir/config.json" ]] || fail "Missing LTX Gemma root: $gemma_dir/config.json"
  [[ -f "$text_dir/config.json" ]] || fail "Missing text model: $text_dir/config.json"
  info "Model cache OK."
}

check_env() {
  require_env_file
  require_secret ADMIN_API_KEY
  require_secret SERVICE_API_KEY
  check_gateway_database_env
  [[ "${LTX_QUANTIZATION:-none}" == "none" ]] || fail "Full 22B BF16 profile requires LTX_QUANTIZATION=none"
  [[ "${TEXT_MODEL_ID:-}" == "google/gemma-3-12b-it-qat-q4_0-unquantized" ]] || fail "Expected TEXT_MODEL_ID=google/gemma-3-12b-it-qat-q4_0-unquantized"
}

full_check() {
  load_env_if_present
  check_os
  check_env
  check_gpu
  check_torch
  check_imports
  check_models
  [[ -x "$QDRANT_BIN" ]] || fail "Missing Qdrant binary at $QDRANT_BIN"
  [[ -x "$ROOT_DIR/gateway-rs/target/release/gateway-rs" ]] || fail "Missing gateway release binary; run deploy/setup"
  info "Full ${GATEWAY_PROFILE:-cloud_h100} system check passed."
}

setup_all() {
  local skip_models="0"
  local skip_apt="0"
  local skip_python_deps="0"
  local install_pytorch="0"

  shift || true
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --skip-models) skip_models="1" ;;
      --skip-apt) skip_apt="1" ;;
      --skip-python-deps) skip_python_deps="1" ;;
      --install-pytorch) install_pytorch="1" ;;
      -h|--help) usage; exit 0 ;;
      *) fail "Unknown setup option: $1" ;;
    esac
    shift
  done

  cd "$ROOT_DIR"
  ensure_dirs
  load_env_if_present
  if [[ "$skip_apt" != "1" ]]; then
    install_apt_packages
  fi
  install_rust
  install_uv
  install_node
  install_qdrant_binary
  create_env_file
  create_venv
  install_pytorch_if_requested "$install_pytorch"
  if [[ "$skip_python_deps" != "1" ]]; then
    install_python_deps
  fi
  load_env_if_present
  if [[ "$skip_models" != "1" ]]; then
    download_models
  else
    warn "Skipping model download."
  fi
  build_frontend
  build_gateway
  full_check
  info "Deployment setup complete."
}

pid_file() {
  echo "$PID_DIR/$1.pid"
}

log_file() {
  echo "$LOG_DIR/$1.log"
}

is_running() {
  local pidfile pid
  pidfile="$(pid_file "$1")"
  [[ -f "$pidfile" ]] || return 1
  pid="$(cat "$pidfile")"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

start_process() {
  local name="$1"
  local cwd="$2"
  local command="$3"
  local pidfile logfile
  pidfile="$(pid_file "$name")"
  logfile="$(log_file "$name")"
  if is_running "$name"; then
    info "$name already running pid=$(cat "$pidfile")"
    return
  fi
  (
    cd "$cwd"
    nohup bash -lc "$command" >>"$logfile" 2>&1 &
    echo $! >"$pidfile"
  )
  sleep 1
  if ! is_running "$name"; then
    tail -n 80 "$logfile" >&2 || true
    fail "$name failed to start"
  fi
  info "started $name pid=$(cat "$pidfile") log=$logfile"
}

stop_process() {
  local name="$1"
  local pidfile pid
  pidfile="$(pid_file "$name")"
  if ! is_running "$name"; then
    rm -f "$pidfile"
    info "$name not running"
    return
  fi
  pid="$(cat "$pidfile")"
  kill "$pid"
  for _ in {1..30}; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      rm -f "$pidfile"
      info "stopped $name"
      return
    fi
    sleep 1
  done
  kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$pidfile"
  warn "force-stopped $name"
}

gateway_command() {
  local bin="$ROOT_DIR/gateway-rs/target/release/gateway-rs"
  [[ -x "$bin" ]] || cargo build --manifest-path "$ROOT_DIR/gateway-rs/Cargo.toml" --release >&2
  printf 'exec %q' "$bin"
}

python_command() {
  local port="$1"
  printf 'exec %q -m uvicorn app.main:app --host 127.0.0.1 --port %q' "$VENV_DIR/bin/python" "$port"
}

start_qdrant() {
  [[ -x "$QDRANT_BIN" ]] || fail "Qdrant is not installed at $QDRANT_BIN. Run deploy/setup first."
  mkdir -p "$QDRANT_STORAGE"
  start_process qdrant "$ROOT_DIR" "export QDRANT__STORAGE__STORAGE_PATH=$(printf '%q' "$QDRANT_STORAGE"); export QDRANT__SERVICE__HTTP_PORT=6333; export QDRANT__SERVICE__GRPC_PORT=6334; exec $(printf '%q' "$QDRANT_BIN")"
}

stop_qdrant() {
  stop_process qdrant
}

start_one() {
  case "$1" in
    qdrant) start_qdrant ;;
    text) start_process text "$ROOT_DIR/services/text-worker" "$(python_command 8101)" ;;
    ltx) start_process ltx "$ROOT_DIR/services/ltx-worker" "$(python_command 8102)" ;;
    gateway) start_process gateway "$ROOT_DIR" "$(gateway_command)" ;;
    *) fail "Unknown service: $1" ;;
  esac
}

stop_one() {
  case "$1" in
    qdrant) stop_qdrant ;;
    text|ltx|gateway) stop_process "$1" ;;
    *) fail "Unknown service: $1" ;;
  esac
}

health_url() {
  case "$1" in
    gateway) echo "http://127.0.0.1:${GATEWAY_BIND##*:}/health" ;;
    text) echo "$TEXT_WORKER_URL/health" ;;
    ltx) echo "$LTX_WORKER_URL/health" ;;
    qdrant) echo "$QDRANT_URL/healthz" ;;
    *) fail "Unknown service: $1" ;;
  esac
}

status_one() {
  local name="$1"
  if is_running "$name"; then
    echo "$name running pid=$(cat "$(pid_file "$name")")"
  else
    echo "$name stopped"
  fi
  local url
  url="$(health_url "$name")"
  if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
    echo "$name health ok $url"
  else
    echo "$name health unavailable $url"
  fi
}

logs_one() {
  local name="$1"
  local follow="$2"
  local lines="$3"
  local logfile
  logfile="$(log_file "$name")"
  [[ -f "$logfile" ]] || fail "No log file for $name at $logfile"
  if [[ "$follow" == "1" ]]; then
    tail -n "$lines" -f "$logfile"
  else
    tail -n "$lines" "$logfile"
  fi
}

target_order_start() {
  if [[ "$1" == "all" ]]; then
    echo "qdrant text ltx gateway"
  else
    echo "$1"
  fi
}

target_order_stop() {
  if [[ "$1" == "all" ]]; then
    echo "gateway ltx text qdrant"
  else
    echo "$1"
  fi
}

service_action() {
  local action="$1"
  local target="${2:-all}"
  shift || true
  shift || true
  require_env_file
  load_env_if_present
  ensure_dirs
  cd "$ROOT_DIR"

  case "$action" in
    start)
      check_env
      for svc in $(target_order_start "$target"); do start_one "$svc"; done
      ;;
    stop)
      for svc in $(target_order_stop "$target"); do stop_one "$svc"; done
      ;;
    restart)
      for svc in $(target_order_stop "$target"); do stop_one "$svc"; done
      for svc in $(target_order_start "$target"); do start_one "$svc"; done
      ;;
    status)
      for svc in $(target_order_start "$target"); do status_one "$svc"; done
      ;;
    logs)
      local follow="0"
      local lines="200"
      while [[ $# -gt 0 ]]; do
        case "$1" in
          -f|--follow) follow="1" ;;
          -n|--lines)
            shift
            lines="${1:-200}"
            ;;
          *) fail "Unknown logs option: $1" ;;
        esac
        shift
      done
      if [[ "$target" == "all" ]]; then
        for svc in gateway text ltx qdrant; do
          echo "===== $svc ====="
          logs_one "$svc" "0" "$lines" || true
        done
      else
        logs_one "$target" "$follow" "$lines"
      fi
      ;;
    *) fail "Unknown action: $action" ;;
  esac
}

main() {
  local action="${1:-}"
  case "$action" in
    ""|-h|--help) usage ;;
    deploy|setup) setup_all "$@" ;;
    check) full_check ;;
    start|stop|restart|status|logs) service_action "$@" ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
