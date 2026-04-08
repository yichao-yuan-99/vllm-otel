#!/usr/bin/env bash
#SBATCH --job-name=vllm_embedded_tp1_mi3001x_tp1
#SBATCH --output=/work1/talati/yichaoy/vllm-otel/servers/servers-amdhpc-embedded-TP1/logs/slurm.%j.out
#SBATCH --error=/work1/talati/yichaoy/vllm-otel/servers/servers-amdhpc-embedded-TP1/logs/slurm.%j.err
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --partition=mi3001x

set -euo pipefail

echo "Embedded TP1 job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"

REPO_ROOT=/work1/talati/yichaoy/vllm-otel
cd "${REPO_ROOT}"

EXPERIMENT_SCRIPT=/work1/talati/yichaoy/vllm-otel/experiments/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-amd/generated/20260407T220804Z/run_replay.sh
EXPERIMENT_RUNNER="${EXPERIMENT_RUNNER:-bash}"
SERVICE_READY_TIMEOUT_SECONDS=900
SERVICE_READY_POLL_INTERVAL_SECONDS=2.0
JAEGER_OTLP_LOCAL_PORT=4317
JAEGER_UI_LOCAL_PORT=16686

JAEGER_SIF=/work1/talati/yichaoy/apptainer_imgs/all-in-one-1.57.sif
VLLM_SIF=/work1/talati/yichaoy/apptainer_imgs/vllm-vllm-openai-rocm:v0.17.1-otel-lp-rocm-lmcache-gfx942.sif
VLLM_MODEL_NAME=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
VLLM_SERVED_MODEL_NAME=Qwen3-Coder-30B-A3B-Instruct-FP8
VLLM_TENSOR_PARALLEL_SIZE=1

VLLM_APPTAINER_HOME=''
HF_HOME=/work1/talati/yichaoy/huggingface
HF_HUB_CACHE=/work1/talati/yichaoy/huggingface/hub
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

AITER_JIT_DIR=/tmp/vllm-aiter-jit-yichaoy
XDG_CACHE_HOME=/tmp/vllm-runtime-yichaoy/xdg-cache
VLLM_CACHE_ROOT=/tmp/vllm-runtime-yichaoy/xdg-cache/vllm

OTEL_SERVICE_NAME=vllm-server
OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="grpc://127.0.0.1:${JAEGER_OTLP_LOCAL_PORT}"
VLLM_COLLECT_DETAILED_TRACES=all
VLLM_LOGITS_PROCESSORS=forceSeq.force_sequence_logits_processor:ForceSequenceAdapter
VLLM_MODEL_EXTRA_ARGS_B64=WyItLXRydXN0LXJlbW90ZS1jb2RlIl0=
VLLM_FORCE_SEQ_TRUST_REMOTE_CODE=true

GATEWAY_CONFIG_DEFAULT=/work1/talati/yichaoy/vllm-otel/gateway/config.toml
GATEWAY_CONFIG_FALLBACK=/work1/talati/yichaoy/vllm-otel/gateway/config.example.toml
GATEWAY_VENV_DIR="${GATEWAY_VENV_DIR:-/work1/talati/yichaoy/vllm-otel/.venv}"
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_SKIP_INSTALL="${GATEWAY_SKIP_INSTALL:-1}"

JOB_LOG_DIR=/work1/talati/yichaoy/vllm-otel/servers/servers-amdhpc-embedded-TP1/logs
mkdir -p "${JOB_LOG_DIR}" "${AITER_JIT_DIR}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

JAEGER_LOG_SHARED="${JOB_LOG_DIR}/jaeger.${SLURM_JOB_ID}.shared.log"
AMD_SMI_POWER_DAEMON_BIN="${AMD_SMI_POWER_DAEMON_BIN:-amd-smi-power-daemon}"
AMD_SMI_POWER_SOCKET_PATH="${AMD_SMI_POWER_SOCKET_PATH:-/tmp/amdsmi-power-reader.${SLURM_JOB_ID}.sock}"
AMD_SMI_POWER_DAEMON_LOG="${JOB_LOG_DIR}/amd-smi-power-daemon.${SLURM_JOB_ID}.log"

AMD_SMI_POWER_DAEMON_PID=""
declare -A PROFILE_VLLM_PID=()
declare -A PROFILE_GATEWAY_PID=()
declare -A PROFILE_EXPERIMENT_PID=()
declare -A PROFILE_EXPERIMENT_EXIT_CODE=()
declare -A PROFILE_VLLM_PORT=()
declare -A PROFILE_GATEWAY_PORT=()
declare -A PROFILE_GATEWAY_PARSE_PORT=()
declare -A PROFILE_VLLM_LOG=()
declare -A PROFILE_GATEWAY_LOG=()
declare -A PROFILE_EXPERIMENT_LOG=()

PROFILE_IDS=(0)

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
    terminate_process "gateway profile=${profile_id}" "${PROFILE_GATEWAY_PID[$profile_id]:-}"
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
  "${AMD_SMI_POWER_DAEMON_BIN}"                 --socket-path "${AMD_SMI_POWER_SOCKET_PATH}"                 >"${AMD_SMI_POWER_DAEMON_LOG}" 2>&1 &
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
  apptainer run                 --cleanenv                 "${APPTAINER_HOME_ARGS[@]}"                 --env COLLECTOR_ZIPKIN_HOST_PORT=:9411                 "${JAEGER_SIF}"                 >"${JAEGER_LOG_SHARED}" 2>&1 &
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
  local vllm_log="${JOB_LOG_DIR}/vllm.${SLURM_JOB_ID}.p${profile_id}.log"
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

  PROFILE_VLLM_PORT["${profile_id}"]="${vllm_port}"
  PROFILE_VLLM_LOG["${profile_id}"]="${vllm_log}"

  echo "Launching vLLM profile=${profile_id} port=${vllm_port} gpu=${rocr_visible_devices}"
  apptainer exec                 --rocm                 --cleanenv                 "${APPTAINER_HOME_ARGS[@]}"                 "${BIND_ARGS[@]}"                 --env PYTHONNOUSERSITE=1                 --env AITER_JIT_DIR="${AITER_JIT_DIR}"                 --env XDG_CACHE_HOME="${XDG_CACHE_HOME}"                 --env VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}"                 --env HF_HOME="${HF_HOME}"                 --env HF_HUB_CACHE="${HF_HUB_CACHE}"                 --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}"                 --env TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}"                 --env HF_TOKEN="${HF_TOKEN}"                 --env OTEL_SERVICE_NAME="${otel_service_name_worker}"                 --env OTEL_EXPORTER_OTLP_TRACES_INSECURE="${OTEL_EXPORTER_OTLP_TRACES_INSECURE}"                 --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}"                 --env HIP_VISIBLE_DEVICE=0                 --env HIP_VISIBLE_DEVICES=0                 --env ROCR_VISIBLE_DEVICES="${rocr_visible_devices}"                 --env VLLM_MODEL_NAME="${VLLM_MODEL_NAME}"                 --env VLLM_MODEL_EXTRA_ARGS_B64="${VLLM_MODEL_EXTRA_ARGS_B64}"                 --env VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${VLLM_FORCE_SEQ_TRUST_REMOTE_CODE}"                 --env LMCACHE_INTERNAL_API_SERVER_ENABLED=1                 --env PYTHONHASHSEED=0                 --env LMCACHE_INTERNAL_API_SERVER_PORT_START="${lmcache_port}"                 "${VLLM_SIF}"                 "${vllm_cmd[@]}"                 >"${vllm_log}" 2>&1 &
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
  echo "Timed out waiting for vLLM readiness for profile ${profile_id}." >&2
  return 1
}

launch_gateway() {
  local profile_id="$1"
  local gateway_port="$2"
  local gateway_parse_port="$3"
  local gateway_log="${JOB_LOG_DIR}/gateway.${SLURM_JOB_ID}.p${profile_id}.log"
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

  PROFILE_GATEWAY_PORT["${profile_id}"]="${gateway_port}"
  PROFILE_GATEWAY_PARSE_PORT["${profile_id}"]="${gateway_parse_port}"
  PROFILE_GATEWAY_LOG["${profile_id}"]="${gateway_log}"

  local gateway_cmd=(
    "${gateway_python}"
    -m gateway
    start
    --config "${gateway_config_path}"
    --host "${GATEWAY_HOST}"
    --venv-dir "${GATEWAY_VENV_DIR}"
    --port-profile-id "${profile_id}"
  )
  if [[ "${GATEWAY_SKIP_INSTALL}" == "1" ]]; then
    gateway_cmd+=(--skip-install)
  fi

  echo "Launching gateway profile=${profile_id} raw=${gateway_port} parsed=${gateway_parse_port}"
  GATEWAY_JAEGER_API_BASE_URL_OVERRIDE="http://127.0.0.1:${JAEGER_UI_LOCAL_PORT}/api/traces"               GATEWAY_OTLP_TRACES_ENDPOINT_OVERRIDE="grpc://127.0.0.1:${JAEGER_OTLP_LOCAL_PORT}"                 "${gateway_cmd[@]}" >"${gateway_log}" 2>&1 &
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
      echo "Gateway for profile ${profile_id} exited before readiness. See ${PROFILE_GATEWAY_LOG[$profile_id]}" >&2
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
  echo "Timed out waiting for gateway readiness for profile ${profile_id}." >&2
  return 1
}

launch_experiment() {
  local profile_id="$1"
  local experiment_log="${JOB_LOG_DIR}/experiment.${SLURM_JOB_ID}.p${profile_id}.log"
  PROFILE_EXPERIMENT_LOG["${profile_id}"]="${experiment_log}"
  echo "Launching experiment profile=${profile_id} script=${EXPERIMENT_SCRIPT}"
  (
    export PORT_PROFILE_ID="${profile_id}"
    export VLLM_BASE_URL="http://127.0.0.1:${PROFILE_VLLM_PORT[$profile_id]}"
    export GATEWAY_BASE_URL="http://127.0.0.1:${PROFILE_GATEWAY_PORT[$profile_id]}"
    export GATEWAY_PARSE_BASE_URL="http://127.0.0.1:${PROFILE_GATEWAY_PARSE_PORT[$profile_id]}"
    export JAEGER_BASE_URL="http://127.0.0.1:${JAEGER_UI_LOCAL_PORT}"
    export AMD_SMI_POWER_SOCKET_PATH="${AMD_SMI_POWER_SOCKET_PATH}"
    "${EXPERIMENT_RUNNER}" "${EXPERIMENT_SCRIPT}" "${profile_id}"
  ) >"${experiment_log}" 2>&1 &
  PROFILE_EXPERIMENT_PID["${profile_id}"]=$!
}

wait_for_experiment_phase() {
  local first_failure=0
  while true; do
    local active_experiments=0
    if ! kill -0 "${JAEGER_PID}" >/dev/null 2>&1; then
      echo "Shared Jaeger exited during experiment phase." >&2
      return 91
    fi
    for profile_id in "${PROFILE_IDS[@]}"; do
      local vllm_pid="${PROFILE_VLLM_PID[$profile_id]:-}"
      local gateway_pid="${PROFILE_GATEWAY_PID[$profile_id]:-}"
      local experiment_pid="${PROFILE_EXPERIMENT_PID[$profile_id]:-}"
      if [[ -n "${vllm_pid}" ]] && ! kill -0 "${vllm_pid}" >/dev/null 2>&1; then
        echo "vLLM for profile ${profile_id} exited during experiment phase." >&2
        return 92
      fi
      if [[ -n "${gateway_pid}" ]] && ! kill -0 "${gateway_pid}" >/dev/null 2>&1; then
        echo "Gateway for profile ${profile_id} exited during experiment phase." >&2
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

start_amd_smi_power_daemon
start_shared_jaeger
launch_vllm 0 11451 29411 0
wait_for_vllm_ready 0 11451
launch_gateway 0 11457 18171
wait_for_gateway_ready 0 11457 18171
launch_experiment 0

echo "All services are ready; waiting for experiment completion."
EXPERIMENT_PHASE_EXIT_CODE=0
if ! wait_for_experiment_phase; then
  EXPERIMENT_PHASE_EXIT_CODE=$?
fi
echo "Experiment phase finished with exit code ${EXPERIMENT_PHASE_EXIT_CODE} at $(date)"
exit "${EXPERIMENT_PHASE_EXIT_CODE}"
