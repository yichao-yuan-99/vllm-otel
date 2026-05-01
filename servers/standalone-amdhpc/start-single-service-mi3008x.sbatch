#!/usr/bin/env bash
#SBATCH --job-name=vllm_standalone_mi3008x_tp8
#SBATCH --output=logs/slurm.%j.out
#SBATCH --error=logs/slurm.%j.err
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=mi3008x

set -euo pipefail

REPO_ROOT="/home1/yichaoy/vllm-otel"
SCRIPT_DIR="${REPO_ROOT}/servers/standalone-amdhpc"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-standalone-$(date -u +%Y%m%dT%H%M%SZ)}}"
SERVICE_READY_TIMEOUT_SECONDS="${SERVICE_READY_TIMEOUT_SECONDS:-900}"
SERVICE_READY_POLL_INTERVAL_SECONDS="${SERVICE_READY_POLL_INTERVAL_SECONDS:-2.0}"

MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${REPO_ROOT}/configs/model_config.toml}"
PORT_PROFILE_CONFIG_PATH="${PORT_PROFILE_CONFIG_PATH:-${REPO_ROOT}/configs/port_profiles.toml}"
LOGIN_HOST="${LOGIN_HOST:-${SSH_TARGET:-${REMOTE_HOST:-login}}}"
SSH_EXTRA_OPTIONS="${SSH_EXTRA_OPTIONS:-}"

PORT_PROFILE_ID="${PORT_PROFILE_ID:-0}"
VLLM_PORT="${VLLM_PORT:-}"
VLLM_MODEL_KEY="${VLLM_MODEL_KEY:-}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-}"
VLLM_MODEL_EXTRA_ARGS_B64="${VLLM_MODEL_EXTRA_ARGS_B64:-}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-8}"
VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${VLLM_FORCE_SEQ_TRUST_REMOTE_CODE:-true}"
VLLM_MODEL_KEY_RESOLVED=""

VLLM_SIF="${VLLM_SIF:-/work1/talati/yichaoy/apptainer_imgs/vllm-vllm-openai-rocm:v0.20.0-otel-lp-rocm-lmcache-gfx942.sif}"
VLLM_APPTAINER_HOME="${VLLM_APPTAINER_HOME:-}"

ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
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

JOB_LOG_DIR="${JOB_LOG_DIR:-${REPO_ROOT}/servers/standalone-amdhpc/logs}"
mkdir -p "${JOB_LOG_DIR}" "${AITER_JIT_DIR}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

VLLM_LOG="${JOB_LOG_DIR}/vllm.${RUN_ID}.log"
VLLM_PID=""
TUNNEL_PIDS=()

command -v ssh >/dev/null 2>&1 || {
  echo "Required command not found: ssh" >&2
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

resolve_vllm_port() {
  if [[ -n "${VLLM_PORT}" ]]; then
    return 0
  fi

  local resolved_port=""
  local resolver_path="${SCRIPT_DIR}/resolve_vllm_port.py"
  if [[ ! -f "${resolver_path}" ]]; then
    echo "Required port resolver not found: ${resolver_path}" >&2
    exit 63
  fi

  if ! resolved_port="$(
    python3 "${resolver_path}" \
      --config "${PORT_PROFILE_CONFIG_PATH}" \
      --profile-id "${PORT_PROFILE_ID}"
  )"; then
    echo "Failed to resolve vLLM port for PORT_PROFILE_ID='${PORT_PROFILE_ID}'" >&2
    exit 64
  fi

  VLLM_PORT="${resolved_port}"
}

resolve_model_launch_settings() {
  local resolved_output=""
  local resolver_path="${SCRIPT_DIR}/model_resolver.py"
  if [[ ! -f "${resolver_path}" ]]; then
    echo "Required model resolver not found: ${resolver_path}" >&2
    exit 65
  fi

  if ! resolved_output="$(
    python3 "${resolver_path}" \
      --config "${MODEL_CONFIG_PATH}" \
      --selector "${VLLM_MODEL_KEY}" \
      --model-name "${VLLM_MODEL_NAME}" \
      --served-model-name "${VLLM_SERVED_MODEL_NAME}" \
      --extra-args-b64 "${VLLM_MODEL_EXTRA_ARGS_B64}"
  )"; then
    echo "Failed to resolve model launch settings" >&2
    exit 66
  fi

  local resolved_fields=()
  mapfile -t resolved_fields <<<"${resolved_output}"
  if [[ "${#resolved_fields[@]}" -ne 4 ]]; then
    echo "Unexpected model resolver output" >&2
    exit 67
  fi

  VLLM_MODEL_KEY_RESOLVED="${resolved_fields[0]}"
  VLLM_MODEL_NAME="${resolved_fields[1]}"
  VLLM_SERVED_MODEL_NAME="${resolved_fields[2]}"
  VLLM_MODEL_EXTRA_ARGS_B64="${resolved_fields[3]}"
}

probe_http_url() {
  local url="$1"
  python3 -c "import sys, urllib.request; req = urllib.request.Request(sys.argv[1], method='GET'); resp = urllib.request.urlopen(req, timeout=3); sys.exit(0 if int(resp.status) == 200 else 1)" "$url" >/dev/null 2>&1
}

start_tunnel() {
  local remote_port="$1"
  local local_port="$2"
  local retries=3
  local attempt=0
  while true; do
    ssh "${SSH_OPTIONS[@]}" -N -R "${remote_port}:127.0.0.1:${local_port}" "${LOGIN_HOST}" &
    local tunnel_pid=$!
    sleep 1
    if kill -0 "${tunnel_pid}" >/dev/null 2>&1; then
      TUNNEL_PIDS+=("${tunnel_pid}")
      return 0
    fi

    wait "${tunnel_pid}" >/dev/null 2>&1 || true
    attempt=$((attempt + 1))
    if (( attempt >= retries )); then
      echo "Reverse tunnel failed after ${retries} attempts (login_host=${LOGIN_HOST} remote=${remote_port} local=${local_port})." >&2
      return 1
    fi
    echo "Reverse tunnel startup failed (attempt ${attempt}/${retries}) for login_host=${LOGIN_HOST}; retrying." >&2
    sleep 1
  done
}

verify_tunnels_alive() {
  local failed=0
  local pid=""
  for pid in "${TUNNEL_PIDS[@]}"; do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "Reverse tunnel process exited before startup completed (pid=${pid})." >&2
      failed=1
    fi
  done
  return "${failed}"
}

compute_hip_visible_devices() {
  local rocr_visible_devices="$1"
  local hip_visible_devices="0"
  if [[ "${rocr_visible_devices}" == *,* ]]; then
    IFS=',' read -r -a visible_device_list <<<"${rocr_visible_devices}"
    local mapped_devices=()
    local i=""
    for i in "${!visible_device_list[@]}"; do
      mapped_devices+=("${i}")
    done
    IFS=',' hip_visible_devices="${mapped_devices[*]}"
  fi
  printf '%s\n' "${hip_visible_devices}"
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
  terminate_process "vllm" "${VLLM_PID:-}"
  local pid=""
  for pid in "${TUNNEL_PIDS[@]}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}

handle_signal() {
  local signal_name="$1"
  echo "Received ${signal_name}; shutting down."
  exit 0
}

trap cleanup EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

APPTAINER_HOME_ARGS=()
if [[ -n "${VLLM_APPTAINER_HOME}" ]]; then
  mkdir -p "${VLLM_APPTAINER_HOME}"
  APPTAINER_HOME_ARGS=(-H "${VLLM_APPTAINER_HOME}")
fi

SSH_OPTIONS=(
  -o ExitOnForwardFailure=yes
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
)
if [[ -n "${SSH_EXTRA_OPTIONS}" ]]; then
  read -r -a SSH_EXTRA_OPTIONS_ARRAY <<<"${SSH_EXTRA_OPTIONS}"
  SSH_OPTIONS+=("${SSH_EXTRA_OPTIONS_ARRAY[@]}")
fi

BIND_ARGS=()
if [[ -n "${HF_HOME}" ]]; then
  BIND_ARGS+=(--bind "${HF_HOME}:${HF_HOME}")
fi
if [[ -n "${HF_HUB_CACHE}" ]]; then
  BIND_ARGS+=(--bind "${HF_HUB_CACHE}:${HF_HUB_CACHE}")
fi

launch_vllm() {
  local hip_visible_devices=""
  hip_visible_devices="$(compute_hip_visible_devices "${ROCR_VISIBLE_DEVICES}")"
  local vllm_cmd=(
    /opt/vllm-plugins/vllm_entrypoint.sh
    --model "${VLLM_MODEL_NAME}"
    --served-model-name "${VLLM_SERVED_MODEL_NAME}"
    --port "${VLLM_PORT}"
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
  )

  echo "Launching single vLLM service port=${VLLM_PORT} tp=${VLLM_TENSOR_PARALLEL_SIZE} gpus=${ROCR_VISIBLE_DEVICES}"
  apptainer exec \
    --rocm \
    --cleanenv \
    "${APPTAINER_HOME_ARGS[@]}" \
    "${BIND_ARGS[@]}" \
    --env PYTHONNOUSERSITE=1 \
    --env AITER_JIT_DIR="${AITER_JIT_DIR}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    --env VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    --env TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env HIP_VISIBLE_DEVICES="${hip_visible_devices}" \
    --env ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}" \
    --env VLLM_MODEL_NAME="${VLLM_MODEL_NAME}" \
    --env VLLM_MODEL_EXTRA_ARGS_B64="${VLLM_MODEL_EXTRA_ARGS_B64}" \
    --env VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${VLLM_FORCE_SEQ_TRUST_REMOTE_CODE}" \
    --env PYTHONHASHSEED=0 \
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

wait_for_service_phase() {
  while true; do
    if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
      echo "vLLM exited during service phase. See ${VLLM_LOG}" >&2
      return 91
    fi
    if ! verify_tunnels_alive; then
      echo "One or more reverse tunnels exited during service phase." >&2
      return 92
    fi
    sleep 5
  done
}

echo "Standalone MI3008X single-service vLLM starting on $(hostname) at $(date)"
echo "Run ID: ${RUN_ID}"

resolve_vllm_port
resolve_model_launch_settings
if [[ -n "${VLLM_MODEL_KEY_RESOLVED}" ]]; then
  echo "Resolved model key '${VLLM_MODEL_KEY_RESOLVED}' -> '${VLLM_MODEL_NAME}' (served as '${VLLM_SERVED_MODEL_NAME}')"
else
  echo "Using raw model '${VLLM_MODEL_NAME}' (served as '${VLLM_SERVED_MODEL_NAME}')"
fi

echo "Opening reverse tunnel(s) to login host ${LOGIN_HOST}"
start_tunnel "${VLLM_PORT}" "${VLLM_PORT}"
sleep 1
if ! verify_tunnels_alive; then
  echo "One or more reverse tunnels failed to establish. Aborting startup." >&2
  exit 72
fi

launch_vllm
wait_for_vllm_ready

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}"

echo "Service is ready."
echo "  LOGIN_HOST=${LOGIN_HOST}"
echo "  PORT_PROFILE_ID=${PORT_PROFILE_ID}"
echo "  VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}"
echo "  ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES}"
echo "  VLLM_BASE_URL=${VLLM_BASE_URL}"
echo "  login-node-vllm: ssh ${LOGIN_HOST} 'curl http://127.0.0.1:${VLLM_PORT}/v1/models'"
echo "Logs:"
echo "  ${VLLM_LOG}"
echo "Waiting for service exit or job cancellation."

SERVICE_PHASE_EXIT_CODE=0
if wait_for_service_phase; then
  SERVICE_PHASE_EXIT_CODE=0
else
  SERVICE_PHASE_EXIT_CODE=$?
fi

echo "Service phase finished with exit code ${SERVICE_PHASE_EXIT_CODE} at $(date)"
exit "${SERVICE_PHASE_EXIT_CODE}"
