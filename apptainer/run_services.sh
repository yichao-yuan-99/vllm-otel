#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

RUN_DIR="${SCRIPT_DIR}/run"
LOG_DIR="${SCRIPT_DIR}/logs"
JAEGER_PID_FILE="${RUN_DIR}/jaeger.pid"
VLLM_PID_FILE="${RUN_DIR}/vllm.pid"

usage() {
  cat <<'USAGE'
Usage: apptainer/run_services.sh [--env-file PATH] <command>

Commands:
  pull      Pull Jaeger + vLLM images and convert to SIF under APPTAINER_IMGS
  start     Start Jaeger and vLLM services in background
  stop      Stop Jaeger and vLLM services
  status    Show service status
  logs      Tail both service logs
  test      Run OTEL + force-sequence smoke tests

Examples:
  bash apptainer/run_services.sh --env-file apptainer/.env pull
  bash apptainer/run_services.sh start
  bash apptainer/run_services.sh test
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      [[ $# -ge 2 ]] || { echo "error: --env-file requires a path" >&2; exit 2; }
      ENV_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

COMMAND="${1:-}"
if [[ -z "${COMMAND}" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "error: env file not found: ${ENV_FILE}" >&2
  echo "hint: create ${SCRIPT_DIR}/.env and define required variables (see README)." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${APPTAINER_IMGS:=${HOME}/apptainer-images}"
: "${JAEGER_IMAGE:=docker://jaegertracing/all-in-one:1.57}"
: "${VLLM_IMAGE:=docker://yichaoyuan/vllm-openai-otel:v0.16.0-otel-lp-rocm}"
: "${VLLM_SERVICE_PORT:=11451}"
: "${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:=grpc://127.0.0.1:4317}"
: "${OTEL_EXPORTER_OTLP_TRACES_INSECURE:=true}"
: "${OTEL_SERVICE_NAME:=vllm-server}"
: "${OTEL_TEST_SERVICE_NAME:=otel-smoke-client}"
: "${VLLM_COLLECT_DETAILED_TRACES:=all}"
: "${VLLM_LOGITS_PROCESSORS:=forceSeq.force_sequence_logits_processor:ForceSequenceAdapter}"
: "${VLLM_FORCE_SEQ_TRUST_REMOTE_CODE:=false}"
: "${VLLM_STARTUP_TIMEOUT:=900}"
: "${VLLM_FORCE_SEQUENCE_TEST_TEXT:=FORCE_SEQ_SMOKE_OK}"
: "${VLLM_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${VLLM_TENSOR_PARALLEL_SIZE:=8}"
: "${JAEGER_API_BASE_URL:=http://127.0.0.1:16686/api/traces}"
: "${AITER_JIT_DIR:=${TMPDIR:-/tmp}/vllm-aiter-jit-${USER:-user}}"
: "${VLLM_RUNTIME_ROOT:=${TMPDIR:-/tmp}/vllm-runtime-${USER:-user}}"
: "${XDG_CACHE_HOME:=${VLLM_RUNTIME_ROOT}/xdg-cache}"
: "${VLLM_CACHE_ROOT:=${XDG_CACHE_HOME}/vllm}"

: "${VLLM_MODEL_NAME:?Set VLLM_MODEL_NAME in ${ENV_FILE}}"
: "${VLLM_SERVED_MODEL_NAME:?Set VLLM_SERVED_MODEL_NAME in ${ENV_FILE}}"
: "${HF_HOME:?Set HF_HOME in your shell environment (or ${ENV_FILE})}"
: "${HF_HUB_CACHE:?Set HF_HUB_CACHE in your shell environment (or ${ENV_FILE})}"

JAEGER_SIF_DEFAULT="${APPTAINER_IMGS}/jaeger-all-in-one-1.57.sif"
VLLM_SIF_DEFAULT="${APPTAINER_IMGS}/vllm-openai-otel-v0.16.0-otel-lp-rocm.sif"
JAEGER_SIF="${JAEGER_SIF:-${JAEGER_SIF_DEFAULT}}"
VLLM_SIF="${VLLM_SIF:-${VLLM_SIF_DEFAULT}}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${APPTAINER_IMGS}"
mkdir -p "${AITER_JIT_DIR}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: required command not found: $1" >&2
    exit 1
  }
}

is_running() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 1
  fi
  local pid
  pid="$(<"${pid_file}")"
  kill -0 "${pid}" >/dev/null 2>&1
}

wait_for_http_ok() {
  local url="$1"
  local timeout="$2"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if (( $(date +%s) - start >= timeout )); then
      return 1
    fi
    sleep 3
  done
}

pull_images() {
  if [[ ! -f "${JAEGER_SIF}" ]]; then
    echo "[pull] ${JAEGER_IMAGE} -> ${JAEGER_SIF}"
    apptainer pull "${JAEGER_SIF}" "${JAEGER_IMAGE}"
  else
    echo "[pull] found existing ${JAEGER_SIF}"
  fi

  if [[ ! -f "${VLLM_SIF}" ]]; then
    echo "[pull] ${VLLM_IMAGE} -> ${VLLM_SIF}"
    apptainer pull "${VLLM_SIF}" "${VLLM_IMAGE}"
  else
    echo "[pull] found existing ${VLLM_SIF}"
  fi
}

require_sif_files() {
  local missing=0

  if [[ ! -f "${JAEGER_SIF}" ]]; then
    echo "error: missing Jaeger SIF: ${JAEGER_SIF}" >&2
    missing=1
  fi

  if [[ ! -f "${VLLM_SIF}" ]]; then
    echo "error: missing vLLM SIF: ${VLLM_SIF}" >&2
    missing=1
  fi

  if [[ "${missing}" -ne 0 ]]; then
    echo "hint: run 'bash apptainer/run_services.sh --env-file ${ENV_FILE} pull' first" >&2
    exit 1
  fi
}

start_jaeger() {
  if is_running "${JAEGER_PID_FILE}"; then
    echo "[start] jaeger already running (pid $(<"${JAEGER_PID_FILE}"))"
    return
  fi
  echo "[start] launching jaeger"
  apptainer run \
    --cleanenv \
    --env COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
    "${JAEGER_SIF}" \
    >"${LOG_DIR}/jaeger.log" 2>&1 &
  echo "$!" > "${JAEGER_PID_FILE}"

  if wait_for_http_ok "http://127.0.0.1:16686" 60; then
    echo "[start] jaeger ready at http://127.0.0.1:16686"
  else
    echo "error: jaeger failed readiness check; see ${LOG_DIR}/jaeger.log" >&2
    exit 1
  fi
}

start_vllm() {
  if is_running "${VLLM_PID_FILE}"; then
    echo "[start] vllm already running (pid $(<"${VLLM_PID_FILE}"))"
    return
  fi

  echo "[start] launching vllm on port ${VLLM_SERVICE_PORT}"
  apptainer exec \
    --rocm \
    --cleanenv \
    --bind "${HF_HOME}:${HF_HOME}" \
    --bind "${HF_HUB_CACHE}:${HF_HUB_CACHE}" \
    --env PYTHONNOUSERSITE=1 \
    --env AITER_JIT_DIR="${AITER_JIT_DIR}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    --env VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME}" \
    --env OTEL_EXPORTER_OTLP_TRACES_INSECURE="${OTEL_EXPORTER_OTLP_TRACES_INSECURE}" \
    --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}" \
    --env HIP_VISIBLE_DEVICES="${VLLM_VISIBLE_DEVICES}" \
    --env ROCR_VISIBLE_DEVICES="${VLLM_VISIBLE_DEVICES}" \
    --env VLLM_MODEL_NAME="${VLLM_MODEL_NAME}" \
    --env VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${VLLM_FORCE_SEQ_TRUST_REMOTE_CODE}" \
    "${VLLM_SIF}" \
    /opt/vllm-plugins/vllm_entrypoint.sh \
    --model "${VLLM_MODEL_NAME}" \
    --served-model-name "${VLLM_SERVED_MODEL_NAME}" \
    --port "${VLLM_SERVICE_PORT}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --otlp-traces-endpoint "${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}" \
    --collect-detailed-traces "${VLLM_COLLECT_DETAILED_TRACES}" \
    --enable-prompt-tokens-details \
    --logits-processors "${VLLM_LOGITS_PROCESSORS}" \
    >"${LOG_DIR}/vllm.log" 2>&1 &
  echo "$!" > "${VLLM_PID_FILE}"

  if wait_for_http_ok "http://127.0.0.1:${VLLM_SERVICE_PORT}/v1/models" "${VLLM_STARTUP_TIMEOUT}"; then
    echo "[start] vllm ready at http://127.0.0.1:${VLLM_SERVICE_PORT}"
  else
    echo "error: vllm failed readiness check; see ${LOG_DIR}/vllm.log" >&2
    exit 1
  fi
}

stop_service() {
  local name="$1"
  local pid_file="$2"

  if ! is_running "${pid_file}"; then
    echo "[stop] ${name} not running"
    rm -f "${pid_file}"
    return
  fi

  local pid
  pid="$(<"${pid_file}")"
  echo "[stop] stopping ${name} (pid ${pid})"
  kill "${pid}" >/dev/null 2>&1 || true

  for _ in {1..20}; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      sleep 1
    else
      rm -f "${pid_file}"
      echo "[stop] ${name} stopped"
      return
    fi
  done

  echo "[stop] force killing ${name} (pid ${pid})"
  kill -9 "${pid}" >/dev/null 2>&1 || true
  rm -f "${pid_file}"
}

show_status() {
  if is_running "${JAEGER_PID_FILE}"; then
    echo "jaeger: running (pid $(<"${JAEGER_PID_FILE}"))"
  else
    echo "jaeger: stopped"
  fi

  if is_running "${VLLM_PID_FILE}"; then
    echo "vllm: running (pid $(<"${VLLM_PID_FILE}"))"
  else
    echo "vllm: stopped"
  fi
}

run_tests() {
  if ! is_running "${VLLM_PID_FILE}"; then
    echo "error: vllm is not running. Start services first." >&2
    exit 1
  fi

  echo "[test] running OTEL smoke test"
  apptainer exec \
    --cleanenv \
    --bind "${REPO_ROOT}:/workspace:ro" \
    --env PYTHONNOUSERSITE=1 \
    --env AITER_JIT_DIR="${AITER_JIT_DIR}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    --env VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}" \
    --env OTEL_SERVICE_NAME="${OTEL_TEST_SERVICE_NAME}" \
    --env OTEL_EXPORTER_OTLP_TRACES_INSECURE="true" \
    --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="grpc://127.0.0.1:4317" \
    --env VLLM_BASE_URL="http://127.0.0.1:${VLLM_SERVICE_PORT}" \
    --env VLLM_TEST_MODEL_NAME="${VLLM_SERVED_MODEL_NAME}" \
    --env VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT}" \
    "${VLLM_SIF}" \
    python3 /workspace/docker/tests/dummy_otel_client.py

  echo "[test] running force-sequence smoke test"
  apptainer exec \
    --cleanenv \
    --bind "${REPO_ROOT}:/workspace:ro" \
    --env PYTHONNOUSERSITE=1 \
    --env AITER_JIT_DIR="${AITER_JIT_DIR}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    --env VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}" \
    --env VLLM_BASE_URL="http://127.0.0.1:${VLLM_SERVICE_PORT}" \
    --env VLLM_TEST_MODEL_NAME="${VLLM_SERVED_MODEL_NAME}" \
    --env VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT}" \
    --env VLLM_FORCE_SEQUENCE_TEST_TEXT="${VLLM_FORCE_SEQUENCE_TEST_TEXT}" \
    "${VLLM_SIF}" \
    python3 /workspace/docker/tests/force_sequence_smoke_test.py

  echo "[test] both smoke tests passed"
}

require_cmd apptainer
require_cmd curl

case "${COMMAND}" in
  pull)
    pull_images
    ;;
  start)
    require_sif_files
    start_jaeger
    start_vllm
    ;;
  stop)
    stop_service vllm "${VLLM_PID_FILE}"
    stop_service jaeger "${JAEGER_PID_FILE}"
    ;;
  status)
    show_status
    ;;
  logs)
    tail -n 200 -f "${LOG_DIR}/jaeger.log" "${LOG_DIR}/vllm.log"
    ;;
  test)
    run_tests
    ;;
  *)
    echo "error: unknown command: ${COMMAND}" >&2
    usage >&2
    exit 2
    ;;
esac
