#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-embedded-$(date -u +%Y%m%dT%H%M%SZ)}}"
SERVICE_READY_TIMEOUT_SECONDS="${SERVICE_READY_TIMEOUT_SECONDS:-900}"
SERVICE_READY_POLL_INTERVAL_SECONDS="${SERVICE_READY_POLL_INTERVAL_SECONDS:-2.0}"

: "${EXPERIMENT_SCRIPT:?EXPERIMENT_SCRIPT must be set to an experiment script path}"

PROFILE_IDS=(0 1 2 3)

JAEGER_OTLP_LOCAL_PORT=4317
JAEGER_UI_LOCAL_PORT=16686

declare -A PROFILE_GPU_IDS=(
  [0]=0
  [1]=1
  [2]=2
  [3]=3
)
declare -A PROFILE_VLLM_PORTS=(
  [0]=11451
  [1]=24123
  [2]=31987
  [3]=40823
)
declare -A PROFILE_GATEWAY_PORTS=(
  [0]=11457
  [1]=24157
  [2]=31955
  [3]=40857
)
declare -A PROFILE_GATEWAY_PARSE_PORTS=(
  [0]=18171
  [1]=28171
  [2]=38171
  [3]=48171
)
declare -A PROFILE_LMCACHE_PORTS=(
  [0]=29411
  [1]=29437
  [2]=29459
  [3]=29483
)

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

JOB_LOG_DIR="${JOB_LOG_DIR:-${REPO_ROOT}/servers/servers-amdhpc-mi2104x-embedded-TP1/logs}"
mkdir -p "${JOB_LOG_DIR}" "${AITER_JIT_DIR}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

JAEGER_LOG_SHARED="${JOB_LOG_DIR}/jaeger.${RUN_ID}.shared.log"
AMD_SMI_POWER_DAEMON_BIN="${AMD_SMI_POWER_DAEMON_BIN:-amd-smi-power-daemon}"
AMD_SMI_POWER_SOCKET_PATH="${AMD_SMI_POWER_SOCKET_PATH:-/tmp/amdsmi-power-reader.${RUN_ID}.sock}"
AMD_SMI_POWER_DAEMON_LOG="${JOB_LOG_DIR}/amd-smi-power-daemon.${RUN_ID}.log"

AMD_SMI_POWER_DAEMON_PID=""
JAEGER_PID=""
VLLM_EXTRA_ENV_ARGS=()
declare -A PROFILE_VLLM_PID=()
declare -A PROFILE_GATEWAY_PID=()
declare -A PROFILE_EXPERIMENT_PID=()
declare -A PROFILE_EXPERIMENT_EXIT_CODE=()
declare -A PROFILE_VLLM_LOG=()
declare -A PROFILE_GATEWAY_LOG=()
declare -A PROFILE_EXPERIMENT_LOG=()

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
  for profile_id in "${PROFILE_IDS[@]}"; do
    terminate_process "experiment profile=${profile_id}" "${PROFILE_EXPERIMENT_PID[$profile_id]:-}"
  done
  for profile_id in "${PROFILE_IDS[@]}"; do
    terminate_process "gateway-ctx profile=${profile_id}" "${PROFILE_GATEWAY_PID[$profile_id]:-}"
  done
  for profile_id in "${PROFILE_IDS[@]}"; do
    terminate_process "vllm profile=${profile_id}" "${PROFILE_VLLM_PID[$profile_id]:-}"
  done
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
  local profile_id="$1"
  local vllm_port="$2"
  local lmcache_port="$3"
  local rocr_visible_devices="$4"
  local vllm_log="${JOB_LOG_DIR}/vllm.${RUN_ID}.p${profile_id}.log"
  local otel_service_name_worker="${OTEL_SERVICE_NAME}-p${profile_id}"
  local vllm_cmd=(
    /opt/vllm-plugins/vllm_entrypoint.sh
    --model "${VLLM_MODEL_NAME}"
    --served-model-name "${VLLM_SERVED_MODEL_NAME}"
    --port "${vllm_port}"
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --otlp-traces-endpoint "${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}"
    --collect-detailed-traces "${VLLM_COLLECT_DETAILED_TRACES}"
    --enable-prompt-tokens-details
    --logits-processors "${VLLM_LOGITS_PROCESSORS}"
  )

  PROFILE_VLLM_LOG["${profile_id}"]="${vllm_log}"

  echo "Launching vLLM profile=${profile_id} port=${vllm_port} gpu=${rocr_visible_devices}"
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
    --env ROCR_VISIBLE_DEVICES="${rocr_visible_devices}" \
    --env VLLM_MODEL_NAME="${VLLM_MODEL_NAME}" \
    --env VLLM_MODEL_EXTRA_ARGS_B64="${VLLM_MODEL_EXTRA_ARGS_B64}" \
    --env VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${VLLM_FORCE_SEQ_TRUST_REMOTE_CODE}" \
    --env LMCACHE_INTERNAL_API_SERVER_ENABLED=1 \
    --env PYTHONHASHSEED=0 \
    --env LMCACHE_INTERNAL_API_SERVER_PORT_START="${lmcache_port}" \
    "${VLLM_SIF}" \
    "${vllm_cmd[@]}" \
    >"${vllm_log}" 2>&1 &
  PROFILE_VLLM_PID["${profile_id}"]=$!
}

wait_for_vllm_ready() {
  local profile_id="$1"
  local vllm_port="$2"
  local pid="${PROFILE_VLLM_PID[$profile_id]}"
  local deadline=$((SECONDS + SERVICE_READY_TIMEOUT_SECONDS))
  local url="http://127.0.0.1:${vllm_port}/v1/models"
  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "vLLM for profile ${profile_id} exited before readiness. See ${PROFILE_VLLM_LOG[$profile_id]}" >&2
      return 1
    fi
    if probe_http_url "${url}"; then
      return 0
    fi
    sleep "${SERVICE_READY_POLL_INTERVAL_SECONDS}"
  done
  echo "Timed out waiting for vLLM readiness for profile ${profile_id}. See ${PROFILE_VLLM_LOG[$profile_id]}" >&2
  return 1
}

launch_gateway() {
  local profile_id="$1"
  local gateway_port="$2"
  local gateway_parse_port="$3"
  local gateway_log="${JOB_LOG_DIR}/gateway-ctx.${RUN_ID}.p${profile_id}.log"
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

  PROFILE_GATEWAY_LOG["${profile_id}"]="${gateway_log}"

  local gateway_cmd=(
    "${gateway_python}"
    -m gateway_ctx
    start
    --config "${gateway_config_path}"
    --host "${GATEWAY_HOST}"
    --venv-dir "${GATEWAY_VENV_DIR}"
    --port-profile-id "${profile_id}"
  )
  if [[ "${GATEWAY_SKIP_INSTALL}" == "1" ]]; then
    gateway_cmd+=(--skip-install)
  fi

  echo "Launching gateway-ctx profile=${profile_id} raw=${gateway_port} parsed=${gateway_parse_port}"
  GATEWAY_JAEGER_API_BASE_URL_OVERRIDE="http://127.0.0.1:${JAEGER_UI_LOCAL_PORT}/api/traces" \
  GATEWAY_OTLP_TRACES_ENDPOINT_OVERRIDE="grpc://127.0.0.1:${JAEGER_OTLP_LOCAL_PORT}" \
    "${gateway_cmd[@]}" >"${gateway_log}" 2>&1 &
  PROFILE_GATEWAY_PID["${profile_id}"]=$!
}

wait_for_gateway_ready() {
  local profile_id="$1"
  local gateway_port="$2"
  local gateway_parse_port="$3"
  local pid="${PROFILE_GATEWAY_PID[$profile_id]}"
  local deadline=$((SECONDS + SERVICE_READY_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "Gateway-ctx for profile ${profile_id} exited before readiness. See ${PROFILE_GATEWAY_LOG[$profile_id]}" >&2
      return 1
    fi
    if [[ "${gateway_port}" -eq "${gateway_parse_port}" ]]; then
      if probe_tcp_port "127.0.0.1" "${gateway_port}"; then
        return 0
      fi
    else
      if probe_tcp_port "127.0.0.1" "${gateway_port}" && probe_tcp_port "127.0.0.1" "${gateway_parse_port}"; then
        return 0
      fi
    fi
    sleep 1
  done
  echo "Timed out waiting for gateway-ctx readiness for profile ${profile_id}. See ${PROFILE_GATEWAY_LOG[$profile_id]}" >&2
  return 1
}

launch_experiment() {
  local profile_id="$1"
  local experiment_log="${JOB_LOG_DIR}/experiment.${RUN_ID}.p${profile_id}.log"
  PROFILE_EXPERIMENT_LOG["${profile_id}"]="${experiment_log}"
  echo "Launching experiment profile=${profile_id} script=${EXPERIMENT_SCRIPT}"
  (
    export PORT_PROFILE_ID="${profile_id}"
    export VLLM_BASE_URL="http://127.0.0.1:${PROFILE_VLLM_PORTS[$profile_id]}"
    export GATEWAY_BASE_URL="http://127.0.0.1:${PROFILE_GATEWAY_PORTS[$profile_id]}"
    export GATEWAY_PARSE_BASE_URL="http://127.0.0.1:${PROFILE_GATEWAY_PARSE_PORTS[$profile_id]}"
    export JAEGER_BASE_URL="http://127.0.0.1:${JAEGER_UI_LOCAL_PORT}"
    export AMD_SMI_POWER_SOCKET_PATH="${AMD_SMI_POWER_SOCKET_PATH}"
    "${EXPERIMENT_RUNNER}" "${EXPERIMENT_SCRIPT}" "${profile_id}"
  ) >"${experiment_log}" 2>&1 &
  PROFILE_EXPERIMENT_PID["${profile_id}"]=$!
}

wait_for_experiment_phase() {
  local first_failure=0
  local profile_id=""
  while true; do
    local active_experiments=0
    if ! kill -0 "${JAEGER_PID}" >/dev/null 2>&1; then
      echo "Shared Jaeger exited during experiment phase. See ${JAEGER_LOG_SHARED}" >&2
      return 91
    fi
    for profile_id in "${PROFILE_IDS[@]}"; do
      local vllm_pid="${PROFILE_VLLM_PID[$profile_id]:-}"
      local gateway_pid="${PROFILE_GATEWAY_PID[$profile_id]:-}"
      local experiment_pid="${PROFILE_EXPERIMENT_PID[$profile_id]:-}"
      if [[ -n "${vllm_pid}" ]] && ! kill -0 "${vllm_pid}" >/dev/null 2>&1; then
        echo "vLLM for profile ${profile_id} exited during experiment phase. See ${PROFILE_VLLM_LOG[$profile_id]}" >&2
        return 92
      fi
      if [[ -n "${gateway_pid}" ]] && ! kill -0 "${gateway_pid}" >/dev/null 2>&1; then
        echo "Gateway-ctx for profile ${profile_id} exited during experiment phase. See ${PROFILE_GATEWAY_LOG[$profile_id]}" >&2
        return 93
      fi
      if [[ -n "${experiment_pid}" ]]; then
        if kill -0 "${experiment_pid}" >/dev/null 2>&1; then
          active_experiments=1
        elif [[ -z "${PROFILE_EXPERIMENT_EXIT_CODE[$profile_id]:-}" ]]; then
          if wait "${experiment_pid}"; then
            PROFILE_EXPERIMENT_EXIT_CODE["${profile_id}"]=0
          else
            PROFILE_EXPERIMENT_EXIT_CODE["${profile_id}"]=$?
          fi
          if [[ "${PROFILE_EXPERIMENT_EXIT_CODE[$profile_id]}" -ne 0 && "${first_failure}" -eq 0 ]]; then
            first_failure="${PROFILE_EXPERIMENT_EXIT_CODE[$profile_id]}"
          fi
        fi
      fi
    done
    if [[ "${active_experiments}" -eq 0 ]]; then
      return "${first_failure}"
    fi
    sleep 2
  done
}

echo "MI2104X embedded TP1 service stack starting on $(hostname) at $(date)"
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
launch_vllm 0 "${PROFILE_VLLM_PORTS[0]}" "${PROFILE_LMCACHE_PORTS[0]}" "${PROFILE_GPU_IDS[0]}"
launch_vllm 1 "${PROFILE_VLLM_PORTS[1]}" "${PROFILE_LMCACHE_PORTS[1]}" "${PROFILE_GPU_IDS[1]}"
launch_vllm 2 "${PROFILE_VLLM_PORTS[2]}" "${PROFILE_LMCACHE_PORTS[2]}" "${PROFILE_GPU_IDS[2]}"
launch_vllm 3 "${PROFILE_VLLM_PORTS[3]}" "${PROFILE_LMCACHE_PORTS[3]}" "${PROFILE_GPU_IDS[3]}"
wait_for_vllm_ready 0 "${PROFILE_VLLM_PORTS[0]}"
wait_for_vllm_ready 1 "${PROFILE_VLLM_PORTS[1]}"
wait_for_vllm_ready 2 "${PROFILE_VLLM_PORTS[2]}"
wait_for_vllm_ready 3 "${PROFILE_VLLM_PORTS[3]}"
launch_gateway 0 "${PROFILE_GATEWAY_PORTS[0]}" "${PROFILE_GATEWAY_PARSE_PORTS[0]}"
launch_gateway 1 "${PROFILE_GATEWAY_PORTS[1]}" "${PROFILE_GATEWAY_PARSE_PORTS[1]}"
launch_gateway 2 "${PROFILE_GATEWAY_PORTS[2]}" "${PROFILE_GATEWAY_PARSE_PORTS[2]}"
launch_gateway 3 "${PROFILE_GATEWAY_PORTS[3]}" "${PROFILE_GATEWAY_PARSE_PORTS[3]}"
wait_for_gateway_ready 0 "${PROFILE_GATEWAY_PORTS[0]}" "${PROFILE_GATEWAY_PARSE_PORTS[0]}"
wait_for_gateway_ready 1 "${PROFILE_GATEWAY_PORTS[1]}" "${PROFILE_GATEWAY_PARSE_PORTS[1]}"
wait_for_gateway_ready 2 "${PROFILE_GATEWAY_PORTS[2]}" "${PROFILE_GATEWAY_PARSE_PORTS[2]}"
wait_for_gateway_ready 3 "${PROFILE_GATEWAY_PORTS[3]}" "${PROFILE_GATEWAY_PARSE_PORTS[3]}"

echo "Services are ready."
echo "  amd-smi-power-socket: ${AMD_SMI_POWER_SOCKET_PATH}"
echo "  jaeger: http://127.0.0.1:${JAEGER_UI_LOCAL_PORT}"
for profile_id in "${PROFILE_IDS[@]}"; do
  echo "  profile ${profile_id}: gpu=${PROFILE_GPU_IDS[$profile_id]} vLLM=http://127.0.0.1:${PROFILE_VLLM_PORTS[$profile_id]} gateway-ctx=http://${GATEWAY_HOST}:${PROFILE_GATEWAY_PORTS[$profile_id]} gateway-ctx-parse=http://${GATEWAY_HOST}:${PROFILE_GATEWAY_PARSE_PORTS[$profile_id]}"
done
echo "Logs:"
echo "  ${AMD_SMI_POWER_DAEMON_LOG}"
echo "  ${JAEGER_LOG_SHARED}"
for profile_id in "${PROFILE_IDS[@]}"; do
  echo "  ${PROFILE_VLLM_LOG[$profile_id]}"
  echo "  ${PROFILE_GATEWAY_LOG[$profile_id]}"
done

launch_experiment 0
launch_experiment 1
launch_experiment 2
launch_experiment 3

EXPERIMENT_PHASE_EXIT_CODE=0
if wait_for_experiment_phase; then
  EXPERIMENT_PHASE_EXIT_CODE=0
else
  EXPERIMENT_PHASE_EXIT_CODE=$?
fi

echo "Experiment phase finished with exit code ${EXPERIMENT_PHASE_EXIT_CODE} at $(date)"
exit "${EXPERIMENT_PHASE_EXIT_CODE}"
