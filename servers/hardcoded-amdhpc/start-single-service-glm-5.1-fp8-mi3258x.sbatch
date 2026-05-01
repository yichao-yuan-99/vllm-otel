#!/usr/bin/env bash
#SBATCH --job-name=vllm_glm_5_1_fp8_mi3258x_tp8
#SBATCH --output=logs/slurm.%j.out
#SBATCH --error=logs/slurm.%j.err
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=mi3258x

set -euo pipefail

REPO_ROOT="/home1/yichaoy/vllm-otel"
SCRIPT_DIR="${REPO_ROOT}/servers/hardcoded-amdhpc"
cd "${REPO_ROOT}"

RUN_ID="${SLURM_JOB_ID:-hardcoded-$(date -u +%Y%m%dT%H%M%SZ)}"
PORT_PROFILE_ID="${PORT_PROFILE_ID:-0}"

PORT_PROFILE_CONFIG_PATH="${REPO_ROOT}/configs/port_profiles.toml"

LOGIN_HOST="login"
JAEGER_SIF="/work1/talati/yichaoy/apptainer_imgs/all-in-one-1.57.sif"
VLLM_SIF="/work1/talati/yichaoy/apptainer_imgs/vllm-vllm-openai-rocm:v0.20.0-otel-lp-rocm-lmcache-gfx942.sif"
ROCR_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

HF_HOME="/work1/talati/yichaoy/huggingface"
HF_HUB_CACHE="/work1/talati/yichaoy/huggingface/hub"
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_TOKEN="${HF_TOKEN:-}"

USER_NAME="${USER:-user}"
TMP_ROOT="${TMPDIR:-/tmp}"
AITER_JIT_DIR="${TMP_ROOT}/vllm-aiter-jit-${USER_NAME}"
VLLM_RUNTIME_ROOT="${TMP_ROOT}/vllm-runtime-${USER_NAME}"
XDG_CACHE_HOME="${VLLM_RUNTIME_ROOT}/xdg-cache"
VLLM_CACHE_ROOT="${XDG_CACHE_HOME}/vllm"
PYTHON_OVERLAY_ROOT="${VLLM_RUNTIME_ROOT}/python-overlay"
VLLM_REGEX_SPEC="${VLLM_REGEX_SPEC:-regex>=2025.10.22}"

JAEGER_OTLP_LOCAL_PORT=4317
JAEGER_UI_LOCAL_PORT=16686
OTEL_SERVICE_NAME="vllm-server-p${PORT_PROFILE_ID}"
OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="grpc://127.0.0.1:${JAEGER_OTLP_LOCAL_PORT}"
VLLM_COLLECT_DETAILED_TRACES=all

SERVICE_READY_TIMEOUT_SECONDS=1200
SERVICE_READY_POLL_INTERVAL_SECONDS=2

JOB_LOG_DIR="${REPO_ROOT}/servers/hardcoded-amdhpc/logs"
mkdir -p "${JOB_LOG_DIR}" "${AITER_JIT_DIR}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${PYTHON_OVERLAY_ROOT}"

JAEGER_LOG="${JOB_LOG_DIR}/jaeger.${RUN_ID}.log"
VLLM_LOG="${JOB_LOG_DIR}/vllm.${RUN_ID}.log"
JAEGER_PID=""
VLLM_PID=""
TUNNEL_PID=""
VLLM_PORT=""
JAEGER_OTLP_LOGIN_PORT=""
JAEGER_UI_LOGIN_PORT=""

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

resolve_profile_ports() {
  local resolved_output=""
  if ! resolved_output="$(
    python3 - "${PORT_PROFILE_CONFIG_PATH}" "${PORT_PROFILE_ID}" <<'PY'
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

config_path = Path(sys.argv[1])
profile_id = sys.argv[2].strip()
if not profile_id.isdigit():
    raise SystemExit(f"invalid PORT_PROFILE_ID: {profile_id!r}")

payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
profiles = payload.get("profiles")
if not isinstance(profiles, dict):
    raise SystemExit("configs/port_profiles.toml must define [profiles]")

profile = profiles.get(profile_id)
if not isinstance(profile, dict):
    raise SystemExit(f"unknown port profile id: {profile_id}")

for key in ("vllm_port", "jaeger_otlp_port", "jaeger_api_port"):
    value = profile.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"invalid {key} for profile {profile_id}")
    print(value)
PY
  )"; then
    echo "Failed to resolve ports for PORT_PROFILE_ID='${PORT_PROFILE_ID}'" >&2
    exit 64
  fi

  local resolved_fields=()
  mapfile -t resolved_fields <<<"${resolved_output}"
  if [[ "${#resolved_fields[@]}" -ne 3 ]]; then
    echo "Unexpected port resolver output" >&2
    exit 65
  fi

  VLLM_PORT="${resolved_fields[0]}"
  JAEGER_OTLP_LOGIN_PORT="${resolved_fields[1]}"
  JAEGER_UI_LOGIN_PORT="${resolved_fields[2]}"
}

probe_http_url() {
  local url="$1"
  python3 -c "import sys, urllib.request; req = urllib.request.Request(sys.argv[1], method='GET'); resp = urllib.request.urlopen(req, timeout=3); sys.exit(0 if int(resp.status) == 200 else 1)" "$url" >/dev/null 2>&1
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
  terminate_process "jaeger" "${JAEGER_PID:-}"
  terminate_process "vllm" "${VLLM_PID:-}"
  terminate_process "ssh reverse tunnel" "${TUNNEL_PID:-}"
}

trap cleanup EXIT
trap 'exit 0' INT TERM

start_tunnel() {
  ssh \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -N \
    -R "${VLLM_PORT}:127.0.0.1:${VLLM_PORT}" \
    -R "${JAEGER_OTLP_LOGIN_PORT}:127.0.0.1:${JAEGER_OTLP_LOCAL_PORT}" \
    -R "${JAEGER_UI_LOGIN_PORT}:127.0.0.1:${JAEGER_UI_LOCAL_PORT}" \
    "${LOGIN_HOST}" &
  TUNNEL_PID=$!

  sleep 1
  if ! kill -0 "${TUNNEL_PID}" >/dev/null 2>&1; then
    wait "${TUNNEL_PID}" >/dev/null 2>&1 || true
    echo "Failed to establish reverse tunnels to ${LOGIN_HOST} for vLLM and Jaeger profile ports" >&2
    exit 72
  fi
}

launch_jaeger() {
  apptainer run \
    --cleanenv \
    --env COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
    "${JAEGER_SIF}" >"${JAEGER_LOG}" 2>&1 &
  JAEGER_PID=$!
}

wait_for_jaeger_ready() {
  local deadline=$((SECONDS + SERVICE_READY_TIMEOUT_SECONDS))
  local url="http://127.0.0.1:${JAEGER_UI_LOCAL_PORT}"
  while (( SECONDS < deadline )); do
    if ! kill -0 "${JAEGER_PID}" >/dev/null 2>&1; then
      echo "Jaeger exited before readiness. See ${JAEGER_LOG}" >&2
      return 1
    fi
    if probe_http_url "${url}"; then
      return 0
    fi
    sleep "${SERVICE_READY_POLL_INTERVAL_SECONDS}"
  done
  echo "Timed out waiting for Jaeger readiness. See ${JAEGER_LOG}" >&2
  return 1
}

prepare_python_overlay() {
  python3 -m pip install \
    --disable-pip-version-check \
    --no-compile \
    --target "${PYTHON_OVERLAY_ROOT}" \
    --upgrade \
    "${VLLM_REGEX_SPEC}" >/dev/null

  # The ROCm image bundles a CUDA-only nixl_ep extension. Shadow it with a
  # minimal stub so import-time type references do not require libcuda.so.1.
  cat > "${PYTHON_OVERLAY_ROOT}/nixl_ep.py" <<'PY'
class Buffer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "nixl_ep is unavailable in this environment; the shim only exists "
            "to avoid import-time failures on ROCm."
        )
PY
}

launch_vllm() {
  apptainer exec \
    --rocm \
    --cleanenv \
    --bind "${HF_HOME}:${HF_HOME}" \
    --bind "${HF_HUB_CACHE}:${HF_HUB_CACHE}" \
    --bind "${PYTHON_OVERLAY_ROOT}:${PYTHON_OVERLAY_ROOT}" \
    --env PYTHONNOUSERSITE=1 \
    --env PYTHONPATH="${PYTHON_OVERLAY_ROOT}" \
    --env AITER_JIT_DIR="${AITER_JIT_DIR}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    --env VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    --env TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME}" \
    --env OTEL_EXPORTER_OTLP_TRACES_INSECURE="${OTEL_EXPORTER_OTLP_TRACES_INSECURE}" \
    --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}" \
    --env VLLM_COLLECT_DETAILED_TRACES="${VLLM_COLLECT_DETAILED_TRACES}" \
    --env HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
    --env ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}" \
    --env VLLM_PORT="${VLLM_PORT}" \
    --env PYTHONHASHSEED=0 \
    "${VLLM_SIF}" \
    bash --noprofile --norc -c '
      export VLLM_ROCM_USE_AITER=1
      export VLLM_ROCM_USE_AITER_RMSNORM=0

      exec vllm serve zai-org/GLM-5.1-FP8 \
        --trust-remote-code \
        --chat-template-content-format=string \
        --tensor-parallel-size 8 \
        --tool-call-parser glm47 \
        --enable-auto-tool-choice \
        --reasoning-parser glm45 \
        --otlp-traces-endpoint "${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}" \
        --collect-detailed-traces "${VLLM_COLLECT_DETAILED_TRACES}" \
        --enable-prompt-tokens-details \
        --disable-custom-all-reduce \
        --enforce-eager \
        --port "${VLLM_PORT}"
    ' >"${VLLM_LOG}" 2>&1 &
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
    if ! kill -0 "${JAEGER_PID}" >/dev/null 2>&1; then
      echo "Jaeger exited during service phase. See ${JAEGER_LOG}" >&2
      return 90
    fi
    if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
      echo "vLLM exited during service phase. See ${VLLM_LOG}" >&2
      return 91
    fi
    if ! kill -0 "${TUNNEL_PID}" >/dev/null 2>&1; then
      echo "Reverse tunnel exited during service phase." >&2
      return 92
    fi
    sleep 5
  done
}

echo "Hardcoded MI3258X GLM 5.1 FP8 service starting on $(hostname) at $(date)"
echo "Run ID: ${RUN_ID}"

resolve_profile_ports
echo "Resolved PORT_PROFILE_ID=${PORT_PROFILE_ID} to VLLM_PORT=${VLLM_PORT}, JAEGER_OTLP_LOGIN_PORT=${JAEGER_OTLP_LOGIN_PORT}, JAEGER_UI_LOGIN_PORT=${JAEGER_UI_LOGIN_PORT}"

echo "Opening reverse tunnels to ${LOGIN_HOST}"
start_tunnel

echo "Launching Jaeger"
launch_jaeger
wait_for_jaeger_ready

echo "Preparing Python overlay for the vLLM image"
prepare_python_overlay

echo "Launching vLLM"
launch_vllm
wait_for_vllm_ready

echo "Service is ready."
echo "  LOGIN_HOST=${LOGIN_HOST}"
echo "  PORT_PROFILE_ID=${PORT_PROFILE_ID}"
echo "  VLLM_BASE_URL=http://127.0.0.1:${VLLM_PORT}"
echo "  JAEGER_BASE_URL=http://127.0.0.1:${JAEGER_UI_LOGIN_PORT}"
echo "  login-node-vllm: ssh ${LOGIN_HOST} 'curl http://127.0.0.1:${VLLM_PORT}/v1/models'"
echo "  login-node-jaeger: ssh ${LOGIN_HOST} 'curl http://127.0.0.1:${JAEGER_UI_LOGIN_PORT}'"
echo "  jaeger log: ${JAEGER_LOG}"
echo "  vllm log: ${VLLM_LOG}"
echo "Waiting for service exit or job cancellation."

wait_for_service_phase
