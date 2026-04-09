#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-embedded-$(date -u +%Y%m%dT%H%M%SZ)}}"
PROFILE_ID=0
ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

: "${EXPERIMENT_SCRIPT:?EXPERIMENT_SCRIPT must be set to an experiment script path}"

SERVICE_READY_TIMEOUT_SECONDS="${SERVICE_READY_TIMEOUT_SECONDS:-900}"
SERVICE_READY_POLL_INTERVAL_SECONDS="${SERVICE_READY_POLL_INTERVAL_SECONDS:-2.0}"

JAEGER_OTLP_LOCAL_PORT=4317
JAEGER_UI_LOCAL_PORT=16686
VLLM_PORT=11451
GATEWAY_PORT=11457
GATEWAY_PARSE_PORT=18171
LMCACHE_PORT=29411

JAEGER_SIF="${JAEGER_SIF:-/work1/talati/yichaoy/apptainer_imgs/all-in-one-1.57.sif}"
VLLM_SIF="${VLLM_SIF:-/work1/talati/yichaoy/apptainer_imgs/vllm-vllm-openai-rocm:v0.17.1-otel-lp-rocm-lmcache-gfx942.sif}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${REPO_ROOT}/configs/model_config.toml}"
VLLM_MODEL_KEY="${VLLM_MODEL_KEY:-}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-}"
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MODEL_KEY_RESOLVED=""

VLLM_APPTAINER_HOME="${VLLM_APPTAINER_HOME:-}"
HF_HOME="${HF_HOME:-/work1/talati/yichaoy/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-/work1/talati/yichaoy/huggingface/hub}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

USER_NAME="${USER:-user}"
TMP_ROOT="${TMPDIR:-/tmp}"
AITER_JIT_DIR="${AITER_JIT_DIR:-${TMP_ROOT}/vllm-aiter-jit-${USER_NAME}}"
VLLM_RUNTIME_ROOT="${VLLM_RUNTIME_ROOT:-${TMP_ROOT}/vllm-runtime-${USER_NAME}}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${VLLM_RUNTIME_ROOT}/xdg-cache}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${XDG_CACHE_HOME}/vllm}"

OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-vllm-server}"
OTEL_EXPORTER_OTLP_TRACES_INSECURE="${OTEL_EXPORTER_OTLP_TRACES_INSECURE:-true}"
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="grpc://127.0.0.1:${JAEGER_OTLP_LOCAL_PORT}"
VLLM_COLLECT_DETAILED_TRACES="${VLLM_COLLECT_DETAILED_TRACES:-all}"
VLLM_LOGITS_PROCESSORS="${VLLM_LOGITS_PROCESSORS:-forceSeq.force_sequence_logits_processor:ForceSequenceAdapter}"
VLLM_MODEL_EXTRA_ARGS_B64="${VLLM_MODEL_EXTRA_ARGS_B64:-}"
VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${VLLM_FORCE_SEQ_TRUST_REMOTE_CODE:-true}"
VLLM_EXTRA_ENV_B64="${VLLM_EXTRA_ENV_B64:-}"

GATEWAY_CONFIG_DEFAULT="${GATEWAY_CONFIG_DEFAULT:-${REPO_ROOT}/gateway_ctx/config.toml}"
GATEWAY_CONFIG_FALLBACK="${GATEWAY_CONFIG_FALLBACK:-${REPO_ROOT}/gateway_ctx/config.example.toml}"
GATEWAY_VENV_DIR="${GATEWAY_VENV_DIR:-${REPO_ROOT}/.venv}"
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_SKIP_INSTALL="${GATEWAY_SKIP_INSTALL:-1}"

EXPERIMENT_RUNNER="${EXPERIMENT_RUNNER:-bash}"

JOB_LOG_DIR="${JOB_LOG_DIR:-${REPO_ROOT}/servers/servers-amdhpc-mi3008x-embedded-TP1-0/logs}"
mkdir -p "${JOB_LOG_DIR}" "${AITER_JIT_DIR}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

JAEGER_LOG_SHARED="${JOB_LOG_DIR}/jaeger.${RUN_ID}.shared.log"
AMD_SMI_POWER_DAEMON_BIN="${AMD_SMI_POWER_DAEMON_BIN:-amd-smi-power-daemon}"
AMD_SMI_POWER_SOCKET_PATH="${AMD_SMI_POWER_SOCKET_PATH:-/tmp/amdsmi-power-reader.${RUN_ID}.sock}"
AMD_SMI_POWER_DAEMON_LOG="${JOB_LOG_DIR}/amd-smi-power-daemon.${RUN_ID}.log"
VLLM_LOG="${JOB_LOG_DIR}/vllm.${RUN_ID}.p${PROFILE_ID}.log"
GATEWAY_LOG="${JOB_LOG_DIR}/gateway-ctx.${RUN_ID}.p${PROFILE_ID}.log"
EXPERIMENT_LOG="${JOB_LOG_DIR}/experiment.${RUN_ID}.p${PROFILE_ID}.log"

AMD_SMI_POWER_DAEMON_PID=""
JAEGER_PID=""
VLLM_PID=""
GATEWAY_PID=""
EXPERIMENT_PID=""
VLLM_EXTRA_ENV_ARGS=()

command -v "${AMD_SMI_POWER_DAEMON_BIN}" >/dev/null 2>&1 || {
  echo "Required command not found: ${AMD_SMI_POWER_DAEMON_BIN}" >&2
  exit 127
}
command -v apptainer >/dev/null 2>&1 || {
  echo "Required command not found: apptainer" >&2
  exit 127
}
command -v python3 >/dev/null 2>&1 || {
  echo "Required command not found: python3" >&2
  exit 127
}
if [[ ! -f "${EXPERIMENT_SCRIPT}" ]]; then
  echo "Experiment script not found: ${EXPERIMENT_SCRIPT}" >&2
  exit 66
fi

resolve_model_launch_settings() {
  local selector="${VLLM_MODEL_KEY:-${VLLM_MODEL_NAME:-qwen3_coder_30b_fp8}}"
  local resolver_path="${SCRIPT_DIR}/model_resolver.py"
  if [[ ! -f "${resolver_path}" ]]; then
    echo "Required model resolver not found: ${resolver_path}" >&2
    exit 66
  fi

  local resolve_cmd=(
    python3
    "${resolver_path}"
    --config "${MODEL_CONFIG_PATH}"
    --selector "${selector}"
  )
  if [[ -n "${VLLM_SERVED_MODEL_NAME}" ]]; then
    resolve_cmd+=(--served-model-name "${VLLM_SERVED_MODEL_NAME}")
  fi
  if [[ -n "${VLLM_MODEL_EXTRA_ARGS_B64}" ]]; then
    resolve_cmd+=(--extra-args-b64 "${VLLM_MODEL_EXTRA_ARGS_B64}")
  fi

  local resolved_output=""
  if ! resolved_output="$("${resolve_cmd[@]}")"; then
    echo "Failed to resolve model settings for selector '${selector}'" >&2
    exit 64
  fi

  local resolved_fields=()
  mapfile -t resolved_fields <<<"${resolved_output}"
  if [[ "${#resolved_fields[@]}" -ne 4 ]]; then
    echo "Unexpected model resolver output for selector '${selector}'" >&2
    exit 65
  fi

  VLLM_MODEL_KEY_RESOLVED="${resolved_fields[0]}"
  VLLM_MODEL_NAME="${resolved_fields[1]}"
  VLLM_SERVED_MODEL_NAME="${resolved_fields[2]}"
  VLLM_MODEL_EXTRA_ARGS_B64="${resolved_fields[3]}"
}

load_vllm_extra_env_args() {
  local encoded="${VLLM_EXTRA_ENV_B64:-}"
  if [[ -z "${encoded}" ]]; then
    return 0
  fi

  local tmp_file=""
  tmp_file="$(mktemp)"

  VLLM_EXTRA_ENV_B64="${encoded}" python3 - "${tmp_file}" <<'PY'
import base64
import json
import os
import sys

encoded = os.environ["VLLM_EXTRA_ENV_B64"]
try:
    decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"error: invalid VLLM_EXTRA_ENV_B64 payload: {exc}")

try:
    extra_env = json.loads(decoded)
except json.JSONDecodeError as exc:
    raise SystemExit(f"error: VLLM_EXTRA_ENV_B64 did not decode to valid JSON: {exc}")

if not isinstance(extra_env, dict) or not all(
    isinstance(key, str) and isinstance(value, str)
    for key, value in extra_env.items()
):
    raise SystemExit("error: decoded VLLM_EXTRA_ENV_B64 payload must be a JSON string map")

with open(sys.argv[1], "wb") as handle:
    for key in sorted(extra_env):
        handle.write(b"--env")
        handle.write(b"\0")
        handle.write(f"{key}={extra_env[key]}".encode("utf-8"))
        handle.write(b"\0")
PY

  mapfile -d '' -t VLLM_EXTRA_ENV_ARGS < "${tmp_file}"
  rm -f "${tmp_file}"
}

APPTAINER_HOME_ARGS=()
if [[ -n "${VLLM_APPTAINER_HOME}" ]]; then
  mkdir -p "${VLLM_APPTAINER_HOME}"
  APPTAINER_HOME_ARGS=(-H "${VLLM_APPTAINER_HOME}")
fi

BIND_ARGS=()
if [[ -n "${HF_HOME}" ]]; then
  BIND_ARGS+=(--bind "${HF_HOME}:${HF_HOME}")
fi
if [[ -n "${HF_HUB_CACHE}" ]]; then
  BIND_ARGS+=(--bind "${HF_HUB_CACHE}:${HF_HUB_CACHE}")
fi

probe_http_url() {
  local url="$1"
  python3 -c "import sys, urllib.request; req = urllib.request.Request(sys.argv[1], method='GET'); resp = urllib.request.urlopen(req, timeout=3); sys.exit(0 if int(resp.status) == 200 else 1)" "$url" >/dev/null 2>&1
}

probe_tcp_port() {
  local host="$1"
  local port="$2"
  python3 -c "import socket, sys; sock = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2); sock.close()" "$host" "$port" >/dev/null 2>&1
}

probe_unix_socket() {
  local socket_path="$1"
  python3 -c "import socket, sys; sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); sock.settimeout(2); sock.connect(sys.argv[1]); sock.close()" "$socket_path" >/dev/null 2>&1
}

terminate_process() {
  local name="$1"
  local pid="$2"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    return 0
  fi
  echo "Stopping ${name} (pid=${pid})"
  kill "${pid}" >/dev/null 2>&1 || true
  local deadline=$((SECONDS + 20))
  while kill -0 "${pid}" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      kill -9 "${pid}" >/dev/null 2>&1 || true
      break
    fi
    sleep 1
  done
  wait "${pid}" >/dev/null 2>&1 || true
}

cleanup() {
  set +e
  terminate_process "experiment profile=${PROFILE_ID}" "${EXPERIMENT_PID:-}"
  terminate_process "gateway-ctx profile=${PROFILE_ID}" "${GATEWAY_PID:-}"
  terminate_process "vllm profile=${PROFILE_ID}" "${VLLM_PID:-}"
  terminate_process "amd-smi-power-daemon" "${AMD_SMI_POWER_DAEMON_PID:-}"
  terminate_process "jaeger" "${JAEGER_PID:-}"
}
trap cleanup EXIT INT TERM

start_amd_smi_power_daemon() {
  if [[ -e "${AMD_SMI_POWER_SOCKET_PATH}" && ! -S "${AMD_SMI_POWER_SOCKET_PATH}" ]]; then
    echo "AMD SMI power daemon socket path exists and is not a socket: ${AMD_SMI_POWER_SOCKET_PATH}" >&2
    return 1
  fi

  echo "Launching AMD SMI power daemon socket=${AMD_SMI_POWER_SOCKET_PATH}"
  "${AMD_SMI_POWER_DAEMON_BIN}" \
    --socket-path "${AMD_SMI_POWER_SOCKET_PATH}" \
    >"${AMD_SMI_POWER_DAEMON_LOG}" 2>&1 &
  AMD_SMI_POWER_DAEMON_PID=$!

  local deadline=$((SECONDS + SERVICE_READY_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${AMD_SMI_POWER_DAEMON_PID}" >/dev/null 2>&1; then
      wait "${AMD_SMI_POWER_DAEMON_PID}" >/dev/null 2>&1 || true
      echo "AMD SMI power daemon exited before readiness. See ${AMD_SMI_POWER_DAEMON_LOG}" >&2
      return 1
    fi
    if [[ -S "${AMD_SMI_POWER_SOCKET_PATH}" ]] && probe_unix_socket "${AMD_SMI_POWER_SOCKET_PATH}"; then
      return 0
    fi
    sleep "${SERVICE_READY_POLL_INTERVAL_SECONDS}"
  done

  echo "Timed out waiting for AMD SMI power daemon readiness. See ${AMD_SMI_POWER_DAEMON_LOG}" >&2
  return 1
}

start_shared_jaeger() {
  echo "Launching shared Jaeger on ui=${JAEGER_UI_LOCAL_PORT} otlp=${JAEGER_OTLP_LOCAL_PORT}"
  apptainer run \
    --cleanenv \
    "${APPTAINER_HOME_ARGS[@]}" \
    --env COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
    "${JAEGER_SIF}" \
    >"${JAEGER_LOG_SHARED}" 2>&1 &
  JAEGER_PID=$!
  sleep 1
  if ! kill -0 "${JAEGER_PID}" >/dev/null 2>&1; then
    wait "${JAEGER_PID}" >/dev/null 2>&1 || true
    echo "Shared Jaeger failed to start. See ${JAEGER_LOG_SHARED}" >&2
    exit 71
  fi
}

launch_vllm() {
  local otel_service_name_worker="${OTEL_SERVICE_NAME}-p${PROFILE_ID}"
  local vllm_cmd=(
    /opt/vllm-plugins/vllm_entrypoint.sh
    --model "${VLLM_MODEL_NAME}"
    --served-model-name "${VLLM_SERVED_MODEL_NAME}"
    --port "${VLLM_PORT}"
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --otlp-traces-endpoint "${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}"
    --collect-detailed-traces "${VLLM_COLLECT_DETAILED_TRACES}"
    --enable-prompt-tokens-details
    --logits-processors "${VLLM_LOGITS_PROCESSORS}"
  )

  echo "Launching vLLM profile=${PROFILE_ID} port=${VLLM_PORT} gpu=${ROCR_VISIBLE_DEVICES}"
  apptainer exec \
    --rocm \
    --cleanenv \
    "${APPTAINER_HOME_ARGS[@]}" \
    "${BIND_ARGS[@]}" \
    "${VLLM_EXTRA_ENV_ARGS[@]}" \
    --env PYTHONNOUSERSITE=1 \
    --env AITER_JIT_DIR="${AITER_JIT_DIR}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    --env VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    --env TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env OTEL_SERVICE_NAME="${otel_service_name_worker}" \
    --env OTEL_EXPORTER_OTLP_TRACES_INSECURE="${OTEL_EXPORTER_OTLP_TRACES_INSECURE}" \
    --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}" \
    --env HIP_VISIBLE_DEVICE=0 \
    --env HIP_VISIBLE_DEVICES=0 \
    --env ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}" \
    --env VLLM_MODEL_NAME="${VLLM_MODEL_NAME}" \
    --env VLLM_MODEL_EXTRA_ARGS_B64="${VLLM_MODEL_EXTRA_ARGS_B64}" \
    --env VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${VLLM_FORCE_SEQ_TRUST_REMOTE_CODE}" \
    --env LMCACHE_INTERNAL_API_SERVER_ENABLED=1 \
    --env PYTHONHASHSEED=0 \
    --env LMCACHE_INTERNAL_API_SERVER_PORT_START="${LMCACHE_PORT}" \
    "${VLLM_SIF}" \
    "${vllm_cmd[@]}" \
    >"${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
}

wait_for_vllm_ready() {
  local deadline=$((SECONDS + SERVICE_READY_TIMEOUT_SECONDS))
  local url="http://127.0.0.1:${VLLM_PORT}/v1/models"
  while (( SECONDS < deadline )); do
    if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
      echo "vLLM exited before readiness. See ${VLLM_LOG}" >&2
      return 1
    fi
    if probe_http_url "${url}"; then
      return 0
    fi
    sleep "${SERVICE_READY_POLL_INTERVAL_SECONDS}"
  done
  echo "Timed out waiting for vLLM readiness. See ${VLLM_LOG}" >&2
  return 1
}

launch_gateway() {
  local gateway_config_path=""
  local gateway_python="python3"
  if [[ -n "${GATEWAY_CONFIG:-}" ]]; then
    gateway_config_path="${GATEWAY_CONFIG}"
  elif [[ -f "${GATEWAY_CONFIG_DEFAULT}" ]]; then
    gateway_config_path="${GATEWAY_CONFIG_DEFAULT}"
  else
    gateway_config_path="${GATEWAY_CONFIG_FALLBACK}"
  fi
  if [[ -x "${GATEWAY_VENV_DIR}/bin/python" ]]; then
    gateway_python="${GATEWAY_VENV_DIR}/bin/python"
  fi

  local gateway_cmd=(
    "${gateway_python}"
    -m gateway_ctx
    start
    --config "${gateway_config_path}"
    --host "${GATEWAY_HOST}"
    --venv-dir "${GATEWAY_VENV_DIR}"
    --port-profile-id "${PROFILE_ID}"
  )
  if [[ "${GATEWAY_SKIP_INSTALL}" == "1" ]]; then
    gateway_cmd+=(--skip-install)
  fi

  echo "Launching gateway-ctx profile=${PROFILE_ID} raw=${GATEWAY_PORT} parsed=${GATEWAY_PARSE_PORT}"
  GATEWAY_JAEGER_API_BASE_URL_OVERRIDE="http://127.0.0.1:${JAEGER_UI_LOCAL_PORT}/api/traces" \
  GATEWAY_OTLP_TRACES_ENDPOINT_OVERRIDE="grpc://127.0.0.1:${JAEGER_OTLP_LOCAL_PORT}" \
    "${gateway_cmd[@]}" >"${GATEWAY_LOG}" 2>&1 &
  GATEWAY_PID=$!
}

wait_for_gateway_ready() {
  local deadline=$((SECONDS + SERVICE_READY_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${GATEWAY_PID}" >/dev/null 2>&1; then
      echo "Gateway-ctx exited before readiness. See ${GATEWAY_LOG}" >&2
      return 1
    fi
    if [[ "${GATEWAY_PORT}" -eq "${GATEWAY_PARSE_PORT}" ]]; then
      if probe_tcp_port "127.0.0.1" "${GATEWAY_PORT}"; then
        return 0
      fi
    else
      if probe_tcp_port "127.0.0.1" "${GATEWAY_PORT}" && probe_tcp_port "127.0.0.1" "${GATEWAY_PARSE_PORT}"; then
        return 0
      fi
    fi
    sleep 1
  done
  echo "Timed out waiting for gateway-ctx readiness. See ${GATEWAY_LOG}" >&2
  return 1
}

launch_experiment() {
  echo "Launching experiment profile=${PROFILE_ID} script=${EXPERIMENT_SCRIPT}"
  (
    export PORT_PROFILE_ID="${PROFILE_ID}"
    export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}"
    export GATEWAY_BASE_URL="http://127.0.0.1:${GATEWAY_PORT}"
    export GATEWAY_PARSE_BASE_URL="http://127.0.0.1:${GATEWAY_PARSE_PORT}"
    export JAEGER_BASE_URL="http://127.0.0.1:${JAEGER_UI_LOCAL_PORT}"
    export AMD_SMI_POWER_SOCKET_PATH="${AMD_SMI_POWER_SOCKET_PATH}"
    "${EXPERIMENT_RUNNER}" "${EXPERIMENT_SCRIPT}" "${PROFILE_ID}"
  ) >"${EXPERIMENT_LOG}" 2>&1 &
  EXPERIMENT_PID=$!
}

wait_for_experiment_phase() {
  while true; do
    if ! kill -0 "${JAEGER_PID}" >/dev/null 2>&1; then
      wait "${JAEGER_PID}" >/dev/null 2>&1 || true
      echo "Shared Jaeger exited during experiment phase. See ${JAEGER_LOG_SHARED}" >&2
      return 91
    fi
    if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
      wait "${VLLM_PID}" >/dev/null 2>&1 || true
      echo "vLLM exited during experiment phase. See ${VLLM_LOG}" >&2
      return 92
    fi
    if ! kill -0 "${GATEWAY_PID}" >/dev/null 2>&1; then
      wait "${GATEWAY_PID}" >/dev/null 2>&1 || true
      echo "Gateway-ctx exited during experiment phase. See ${GATEWAY_LOG}" >&2
      return 93
    fi
    if ! kill -0 "${EXPERIMENT_PID}" >/dev/null 2>&1; then
      if wait "${EXPERIMENT_PID}"; then
        return 0
      fi
      return $?
    fi
    sleep 2
  done
}

echo "MI3008X embedded TP1 profile 0 service stack starting on $(hostname) at $(date)"
echo "Run ID: ${RUN_ID}"
resolve_model_launch_settings
load_vllm_extra_env_args
if [[ -n "${VLLM_MODEL_KEY_RESOLVED}" ]]; then
  echo "Resolved model key '${VLLM_MODEL_KEY_RESOLVED}' -> '${VLLM_MODEL_NAME}' (served as '${VLLM_SERVED_MODEL_NAME}')"
else
  echo "Using raw model '${VLLM_MODEL_NAME}' (served as '${VLLM_SERVED_MODEL_NAME}')"
fi

start_amd_smi_power_daemon
start_shared_jaeger
launch_vllm
wait_for_vllm_ready
launch_gateway
wait_for_gateway_ready

echo "Services are ready."
echo "  amd-smi-power-socket: ${AMD_SMI_POWER_SOCKET_PATH}"
echo "  vLLM:   http://127.0.0.1:${VLLM_PORT}"
echo "  gateway-ctx: http://${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "  gateway-ctx-parse: http://${GATEWAY_HOST}:${GATEWAY_PARSE_PORT}"
echo "  jaeger: http://127.0.0.1:${JAEGER_UI_LOCAL_PORT}"
echo "Logs:"
echo "  ${AMD_SMI_POWER_DAEMON_LOG}"
echo "  ${JAEGER_LOG_SHARED}"
echo "  ${VLLM_LOG}"
echo "  ${GATEWAY_LOG}"
echo "  ${EXPERIMENT_LOG}"

launch_experiment

EXPERIMENT_PHASE_EXIT_CODE=0
if ! wait_for_experiment_phase; then
  EXPERIMENT_PHASE_EXIT_CODE=$?
fi

echo "Experiment phase finished with exit code ${EXPERIMENT_PHASE_EXIT_CODE} at $(date)"
exit "${EXPERIMENT_PHASE_EXIT_CODE}"
