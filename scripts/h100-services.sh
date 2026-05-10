#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${AI_ENV_FILE:-"$ROOT_DIR/.env.h100"}"
RUNTIME_DIR="${AI_RUNTIME_DIR:-"$ROOT_DIR/runtime/h100"}"
PID_DIR="$RUNTIME_DIR/pids"
LOG_DIR="$RUNTIME_DIR/logs"

usage() {
  cat <<'USAGE'
Usage:
  scripts/h100-services.sh check
  scripts/h100-services.sh start   [all|gateway|text|ltx|qdrant]
  scripts/h100-services.sh stop    [all|gateway|text|ltx|qdrant]
  scripts/h100-services.sh restart [all|gateway|text|ltx|qdrant]
  scripts/h100-services.sh status  [all|gateway|text|ltx|qdrant]
  scripts/h100-services.sh logs    [all|gateway|text|ltx|qdrant] [-f] [-n lines]

Environment:
  AI_ENV_FILE       Defaults to .env.h100
  AI_PYTHON_MODE    system or uv. system uses AI_PYTHON_BIN.
  AI_PYTHON_BIN     Defaults to python3.12 when AI_PYTHON_MODE=system.
USAGE
}

load_env() {
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing env file: $ENV_FILE" >&2
    echo "Create it from .env.h100.example and replace the secrets before starting services." >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a

  export GATEWAY_BIND="${GATEWAY_BIND:-0.0.0.0:8080}"
  export GATEWAY_PROFILE="${GATEWAY_PROFILE:-cloud_h100}"
  export RUNTIME_BACKEND="${RUNTIME_BACKEND:-native}"
  export SERVICE_LOG_DIR="${SERVICE_LOG_DIR:-$LOG_DIR}"
  export LOCAL_STORAGE_DIR="${LOCAL_STORAGE_DIR:-storage}"
  export FRONTEND_DIR="${FRONTEND_DIR:-frontend/dist}"
  export TEXT_WORKER_URL="${TEXT_WORKER_URL:-http://127.0.0.1:8101}"
  export LTX_WORKER_URL="${LTX_WORKER_URL:-http://127.0.0.1:8102}"
  export TEXT_MODEL_DIR="${TEXT_MODEL_DIR:-models/text/gemma-3-12b-it-qat-q4_0-unquantized}"
  export LTX_MODEL_DIR="${LTX_MODEL_DIR:-models/ltx-2.3}"
  export LTX_GEMMA_ROOT="${LTX_GEMMA_ROOT:-$TEXT_MODEL_DIR}"
  export LTX_CUDA_DEVICE="${LTX_CUDA_DEVICE:-cuda:0}"
  export LTX_QUANTIZATION="${LTX_QUANTIZATION:-none}"
  export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
  export QDRANT_MANAGED="${QDRANT_MANAGED:-1}"
  export QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:v1.11.0}"
  export QDRANT_CONTAINER="${QDRANT_CONTAINER:-py-ai-apis-qdrant}"
  export AI_PYTHON_MODE="${AI_PYTHON_MODE:-system}"
  export AI_PYTHON_BIN="${AI_PYTHON_BIN:-python3.12}"
}

ensure_dirs() {
  mkdir -p "$PID_DIR" "$LOG_DIR" "$ROOT_DIR/storage" "$ROOT_DIR/models"
}

repo_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$ROOT_DIR" "$1" ;;
  esac
}

require_secret() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" || "$value" == replace-* ]]; then
    echo "$name must be set to a real secret in $ENV_FILE" >&2
    exit 1
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_h100_or_h200() {
  require_cmd nvidia-smi
  local names
  names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  local expected="H100|H200"
  if [[ "${GATEWAY_PROFILE:-}" == *"h200"* ]]; then
    expected="H200"
  elif [[ "${GATEWAY_PROFILE:-}" == *"h100"* ]]; then
    expected="H100"
  fi
  if ! grep -Eqi "$expected" <<<"$names"; then
    if [[ "${ALLOW_NON_CLOUD_GPU:-${ALLOW_NON_H100:-0}}" != "1" ]]; then
      echo "This deployment profile requires $expected for GATEWAY_PROFILE=${GATEWAY_PROFILE:-unset}. Detected GPU(s): ${names:-none}" >&2
      echo "Set ALLOW_NON_CLOUD_GPU=1 only for explicit non-production testing." >&2
      exit 1
    fi
  fi
}

check_torch() {
  local py
  if [[ "$AI_PYTHON_MODE" == "system" ]]; then
    py="$AI_PYTHON_BIN"
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
PY
  else
    uv run --directory services/ltx-worker python - <<'PY'
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
PY
  fi
}

check_models() {
  local ltx_dir
  local text_dir
  local gemma_dir
  ltx_dir="$(repo_path "${LTX_MODEL_DIR:-models/ltx-2.3}")"
  text_dir="$(repo_path "${TEXT_MODEL_DIR:-models/text/gemma-3-12b-it-qat-q4_0-unquantized}")"
  gemma_dir="$(repo_path "${LTX_GEMMA_ROOT:-${TEXT_MODEL_DIR:-models/text/gemma-3-12b-it-qat-q4_0-unquantized}}")"
  [[ -f "$ltx_dir/ltx-2.3-22b-dev.safetensors" ]] || {
    echo "Missing full LTX 22B dev checkpoint in $ltx_dir" >&2
    exit 1
  }
  [[ -f "$gemma_dir/config.json" ]] || {
    echo "Missing LTX Gemma assets in $gemma_dir" >&2
    exit 1
  }
  [[ -f "$text_dir/config.json" ]] || {
    echo "Missing text model config in $text_dir" >&2
    exit 1
  }
}

check_all() {
  require_cmd curl
  require_cmd git
  require_cmd cargo
  if [[ "$AI_PYTHON_MODE" == "uv" ]]; then
    require_cmd uv
  else
    require_cmd "$AI_PYTHON_BIN"
  fi
  require_secret ADMIN_API_KEY
  require_secret SERVICE_API_KEY
  require_h100_or_h200
  check_torch
  check_models
  echo "${GATEWAY_PROFILE:-cloud_h100} deployment checks passed."
}

pid_file() {
  echo "$PID_DIR/$1.pid"
}

log_file() {
  echo "$LOG_DIR/$1.log"
}

is_running() {
  local pidfile
  pidfile="$(pid_file "$1")"
  [[ -f "$pidfile" ]] || return 1
  local pid
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
    echo "$name already running pid=$(cat "$pidfile")"
    return
  fi

  (
    cd "$cwd"
    nohup bash -lc "$command" >>"$logfile" 2>&1 &
    echo $! >"$pidfile"
  )

  sleep 1
  if ! is_running "$name"; then
    echo "$name failed to start. Last log lines:" >&2
    tail -n 80 "$logfile" >&2 || true
    exit 1
  fi
  echo "started $name pid=$(cat "$pidfile") log=$logfile"
}

stop_process() {
  local name="$1"
  local pidfile
  pidfile="$(pid_file "$name")"
  if ! is_running "$name"; then
    rm -f "$pidfile"
    echo "$name not running"
    return
  fi
  local pid
  pid="$(cat "$pidfile")"
  kill "$pid"
  for _ in {1..30}; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      rm -f "$pidfile"
      echo "stopped $name"
      return
    fi
    sleep 1
  done
  kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$pidfile"
  echo "force-stopped $name"
}

gateway_command() {
  local bin="$ROOT_DIR/gateway-rs/target/release/gateway-rs"
  if [[ ! -x "$bin" ]]; then
    cargo build --manifest-path "$ROOT_DIR/gateway-rs/Cargo.toml" --release >&2
  fi
  printf 'exec %q' "$bin"
}

python_service_command() {
  local service_dir="$1"
  local port="$2"
  if [[ "$AI_PYTHON_MODE" == "system" ]]; then
    printf 'exec %q -m uvicorn app.main:app --host 127.0.0.1 --port %q' "$AI_PYTHON_BIN" "$port"
  else
    printf 'exec uv run --directory %q uvicorn app.main:app --host 127.0.0.1 --port %q' "$service_dir" "$port"
  fi
}

start_qdrant() {
  if [[ "$QDRANT_MANAGED" != "1" ]]; then
    echo "qdrant is externally managed at $QDRANT_URL"
    return
  fi
  require_cmd docker
  mkdir -p "$ROOT_DIR/storage/qdrant"
  if docker ps --format '{{.Names}}' | grep -qx "$QDRANT_CONTAINER"; then
    echo "qdrant already running container=$QDRANT_CONTAINER"
    return
  fi
  if docker ps -a --format '{{.Names}}' | grep -qx "$QDRANT_CONTAINER"; then
    docker start "$QDRANT_CONTAINER" >/dev/null
  else
    docker run -d \
      --name "$QDRANT_CONTAINER" \
      --restart unless-stopped \
      -p 6333:6333 -p 6334:6334 \
      -v "$ROOT_DIR/storage/qdrant:/qdrant/storage" \
      "$QDRANT_IMAGE" >/dev/null
  fi
  echo "started qdrant container=$QDRANT_CONTAINER"
}

stop_qdrant() {
  if [[ "$QDRANT_MANAGED" != "1" ]]; then
    echo "qdrant is externally managed"
    return
  fi
  if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -qx "$QDRANT_CONTAINER"; then
    docker stop "$QDRANT_CONTAINER" >/dev/null
    echo "stopped qdrant"
  else
    echo "qdrant not running"
  fi
}

start_one() {
  case "$1" in
    qdrant) start_qdrant ;;
    text)
      local cmd
      cmd="$(python_service_command services/text-worker 8101)"
      if [[ "$AI_PYTHON_MODE" == "system" ]]; then
        start_process text "$ROOT_DIR/services/text-worker" "$cmd"
      else
        start_process text "$ROOT_DIR" "$cmd"
      fi
      ;;
    ltx)
      local cmd
      cmd="$(python_service_command services/ltx-worker 8102)"
      if [[ "$AI_PYTHON_MODE" == "system" ]]; then
        start_process ltx "$ROOT_DIR/services/ltx-worker" "$cmd"
      else
        start_process ltx "$ROOT_DIR" "$cmd"
      fi
      ;;
    gateway)
      start_process gateway "$ROOT_DIR" "$(gateway_command)"
      ;;
    *) echo "Unknown service: $1" >&2; exit 1 ;;
  esac
}

stop_one() {
  case "$1" in
    qdrant) stop_qdrant ;;
    text|ltx|gateway) stop_process "$1" ;;
    *) echo "Unknown service: $1" >&2; exit 1 ;;
  esac
}

health_url() {
  case "$1" in
    gateway) echo "http://127.0.0.1:${GATEWAY_BIND##*:}/health" ;;
    text) echo "$TEXT_WORKER_URL/health" ;;
    ltx) echo "$LTX_WORKER_URL/health" ;;
    qdrant) echo "$QDRANT_URL/healthz" ;;
  esac
}

status_one() {
  local name="$1"
  if [[ "$name" == "qdrant" ]]; then
    if [[ "$QDRANT_MANAGED" == "1" ]] && command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -qx "$QDRANT_CONTAINER"; then
      echo "qdrant running container=$QDRANT_CONTAINER"
    else
      echo "qdrant external-or-stopped url=$QDRANT_URL"
    fi
  elif is_running "$name"; then
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
  if [[ "$name" == "qdrant" && "$QDRANT_MANAGED" == "1" ]]; then
    if [[ "$follow" == "1" ]]; then
      docker logs -f --tail "$lines" "$QDRANT_CONTAINER"
    else
      docker logs --tail "$lines" "$QDRANT_CONTAINER"
    fi
    return
  fi
  local logfile
  logfile="$(log_file "$name")"
  [[ -f "$logfile" ]] || {
    echo "No log file for $name at $logfile" >&2
    return 1
  }
  if [[ "$follow" == "1" ]]; then
    tail -n "$lines" -f "$logfile"
  else
    tail -n "$lines" "$logfile"
  fi
}

for_target() {
  local target="$1"
  shift
  if [[ "$target" == "all" ]]; then
    "$@" qdrant
    "$@" text
    "$@" ltx
    "$@" gateway
  else
    "$@" "$target"
  fi
}

for_target_stop_order() {
  local target="$1"
  shift
  if [[ "$target" == "all" ]]; then
    "$@" gateway
    "$@" ltx
    "$@" text
    "$@" qdrant
  else
    "$@" "$target"
  fi
}

main() {
  local action="${1:-}"
  local target="${2:-all}"
  if [[ -z "$action" || "$action" == "-h" || "$action" == "--help" ]]; then
    usage
    exit 0
  fi

  load_env
  ensure_dirs
  cd "$ROOT_DIR"

  case "$action" in
    check) check_all ;;
    start)
      require_secret ADMIN_API_KEY
      require_secret SERVICE_API_KEY
      for_target "$target" start_one
      ;;
    stop) for_target_stop_order "$target" stop_one ;;
    restart)
      for_target_stop_order "$target" stop_one
      for_target "$target" start_one
      ;;
    status) for_target "$target" status_one ;;
    logs)
      local follow="0"
      local lines="200"
      if [[ $# -ge 2 ]]; then
        shift 2
      else
        shift 1
      fi
      while [[ $# -gt 0 ]]; do
        case "$1" in
          -f|--follow) follow="1" ;;
          -n|--lines)
            shift
            lines="${1:-200}"
            ;;
          *) echo "Unknown logs option: $1" >&2; exit 1 ;;
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
    *) usage; exit 1 ;;
  esac
}

main "$@"
