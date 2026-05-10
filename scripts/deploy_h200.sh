#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export AI_ENV_FILE="${AI_ENV_FILE:-"$ROOT_DIR/.env.h200"}"
export AI_RUNTIME_DIR="${AI_RUNTIME_DIR:-"$ROOT_DIR/runtime/h200"}"
export AI_VENV_DIR="${AI_VENV_DIR:-"$ROOT_DIR/.venv-h200"}"
export GATEWAY_PROFILE="${GATEWAY_PROFILE:-cloud_h200}"

exec "$ROOT_DIR/scripts/deploy_h100.sh" "$@"
