#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${REPO_ROOT}/docker/.env"
VENV_DIR="${REPO_ROOT}/.venv"
SKIP_INSTALL=0

usage() {
  cat <<'EOF'
Usage: docker/start_gateway.sh [--env-file PATH] [--skip-install]

Options:
  --env-file PATH   Path to env file (default: docker/.env)
  --skip-install    Skip dependency install into the shared .venv
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      [[ $# -ge 2 ]] || { echo "error: --env-file requires a path" >&2; exit 2; }
      ENV_FILE="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "error: env file not found: ${ENV_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${GATEWAY_PORT:=11457}"
: "${VLLM_SERVICE_PORT:=11451}"
: "${GATEWAY_OTEL_SERVICE_NAME:=vllm-gateway}"
: "${GATEWAY_OTEL_EXPORTER_OTLP_TRACES_INSECURE:=true}"
: "${GATEWAY_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:=grpc://127.0.0.1:4317}"
: "${GATEWAY_JAEGER_API_BASE_URL:=http://127.0.0.1:16686/api/traces}"
: "${GATEWAY_REQUEST_TIMEOUT_SECONDS:=120}"
: "${GATEWAY_ARTIFACT_COMPRESSION:=none}"
: "${GATEWAY_JOB_END_TRACE_WAIT_SECONDS:=10}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ "${SKIP_INSTALL}" -eq 0 ]]; then
  "${VENV_PYTHON}" -m pip install -r "${REPO_ROOT}/gateway/requirements.txt"
fi

cd "${REPO_ROOT}"

exec env \
  OTEL_SERVICE_NAME="${GATEWAY_OTEL_SERVICE_NAME}" \
  OTEL_EXPORTER_OTLP_TRACES_INSECURE="${GATEWAY_OTEL_EXPORTER_OTLP_TRACES_INSECURE}" \
  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${GATEWAY_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}" \
  VLLM_BASE_URL="http://127.0.0.1:${VLLM_SERVICE_PORT}" \
  JAEGER_API_BASE_URL="${GATEWAY_JAEGER_API_BASE_URL}" \
  GATEWAY_REQUEST_TIMEOUT_SECONDS="${GATEWAY_REQUEST_TIMEOUT_SECONDS}" \
  GATEWAY_ARTIFACT_COMPRESSION="${GATEWAY_ARTIFACT_COMPRESSION}" \
  GATEWAY_JOB_END_TRACE_WAIT_SECONDS="${GATEWAY_JOB_END_TRACE_WAIT_SECONDS}" \
  "${VENV_PYTHON}" -m uvicorn gateway.app:app --host 0.0.0.0 --port "${GATEWAY_PORT}"
