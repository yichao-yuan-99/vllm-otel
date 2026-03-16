#!/usr/bin/env bash
#SBATCH --job-name=vllm_job_g_bench_orch
#SBATCH --output=/work1/talati/yichaoy/vllm-otel/servers/servers-amdhpc/logs/slurm.%j.out
#SBATCH --error=/work1/talati/yichaoy/vllm-otel/servers/servers-amdhpc/logs/slurm.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#SBATCH --partition=mi3008x

set -euo pipefail

echo "Grouped job ${SLURM_JOB_ID} (group=bench_orch) starting at $(date)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unknown}"

LOGIN_HOST=login
GROUP_NAME=bench_orch
GROUP_SIZE=8
GROUP_PROFILE_IDS_CSV=0,1,2,3,4,5,6,7
GROUP_VLLM_PORTS_CSV=11451,24123,31987,40823,52341,59231,60231,61231
GROUP_GATEWAY_PORTS_CSV=11457,24157,31955,40857,52357,59257,60257,61257
GROUP_GATEWAY_PARSE_PORTS_CSV=18171,28171,38171,48171,58171,59171,60171,61171
GROUP_JAEGER_OTLP_LOGIN_PORTS_CSV=4317,24831,32719,41735,53477,60319,61319,62319
GROUP_JAEGER_UI_LOGIN_PORTS_CSV=16686,27544,35264,44612,56198,61784,62784,63784
GROUP_LMCACHE_INTERNAL_API_SERVER_PORT_START_CSV=29411,29437,29459,29483,29507,29531,29557,29579
GROUP_VLLM_VISIBLE_DEVICES_SEMICOLON='0;1;2;3;4;5;6;7'
GROUP_VLLM_TENSOR_PARALLEL_SIZE=1
JAEGER_OTLP_LOCAL_PORT=4317
JAEGER_UI_LOCAL_PORT=16686

JAEGER_SIF=/work1/talati/yichaoy/apptainer_imgs/all-in-one-1.57.sif
VLLM_SIF=/work1/talati/yichaoy/apptainer_imgs/vllm-vllm-openai-rocm:v0.17.1-otel-lp-rocm-lmcache-gfx942.sif

VLLM_MODEL_NAME=Qwen/Qwen3-Coder-30B-A3B-Instruct
VLLM_SERVED_MODEL_NAME=Qwen3-Coder-30B-A3B-Instruct

VLLM_APPTAINER_HOME=''
HF_HOME=/work1/talati/yichaoy/huggingface
HF_HUB_CACHE=/work1/talati/yichaoy/huggingface/hub
HF_TOKEN="${HF_TOKEN:-}"

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
GROUP_GATEWAY_CONFIG="${GROUP_GATEWAY_CONFIG:-}"

REPO_ROOT=/work1/talati/yichaoy/vllm-otel
JOB_LOG_DIR=/work1/talati/yichaoy/vllm-otel/servers/servers-amdhpc/logs
mkdir -p "${JOB_LOG_DIR}" "${AITER_JIT_DIR}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"
GROUP_LOCAL_MODE_SCRIPT=/work1/talati/yichaoy/vllm-otel/sbatch-orchestrator/entrypoint.sh
GROUP_LOCAL_MODE_LOG="${JOB_LOG_DIR}/group-local-mode.${SLURM_JOB_ID}.log"
JAEGER_LOG_SHARED="${JOB_LOG_DIR}/jaeger.${SLURM_JOB_ID}.shared.log"

run_group_worker() {
  set -euo pipefail
  local SSH_OPTIONS=(-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3)

  local index="${SLURM_PROCID:-${SLURM_NODEID:-}}"
  if [[ -z "${index}" ]]; then
    echo "Missing SLURM_PROCID/SLURM_NODEID for grouped worker." >&2
    exit 92
  fi

  IFS=',' read -r -a PROFILE_IDS <<<"${GROUP_PROFILE_IDS_CSV}"
  IFS=',' read -r -a VLLM_PORTS <<<"${GROUP_VLLM_PORTS_CSV}"
  IFS=',' read -r -a GATEWAY_PORTS <<<"${GROUP_GATEWAY_PORTS_CSV}"
  IFS=',' read -r -a GATEWAY_PARSE_PORTS <<<"${GROUP_GATEWAY_PARSE_PORTS_CSV}"
  IFS=',' read -r -a JAEGER_OTLP_PORTS <<<"${GROUP_JAEGER_OTLP_LOGIN_PORTS_CSV}"
  IFS=',' read -r -a JAEGER_UI_PORTS <<<"${GROUP_JAEGER_UI_LOGIN_PORTS_CSV}"
  IFS=',' read -r -a LMCACHE_PORT_STARTS <<<"${GROUP_LMCACHE_INTERNAL_API_SERVER_PORT_START_CSV}"
  IFS=';' read -r -a VLLM_VISIBLE_DEVICES_BY_PROFILE <<<"${GROUP_VLLM_VISIBLE_DEVICES_SEMICOLON}"

  if [[ "${index}" -lt 0 || "${index}" -ge "${#PROFILE_IDS[@]}" ]]; then
    echo "SLURM_PROCID out of range for grouped profile mapping: index=${index}" >&2
    exit 93
  fi
  if [[ "${index}" -ge "${#VLLM_VISIBLE_DEVICES_BY_PROFILE[@]}" ]]; then
    echo "SLURM_PROCID out of range for grouped visible-devices mapping: index=${index}" >&2
    exit 94
  fi
  if [[ "${index}" -ge "${#GATEWAY_PORTS[@]}" || "${index}" -ge "${#GATEWAY_PARSE_PORTS[@]}" ]]; then
    echo "SLURM_PROCID out of range for grouped gateway-port mapping: index=${index}" >&2
    exit 96
  fi

  local PROFILE_ID="${PROFILE_IDS[${index}]}"
  local VLLM_SERVICE_PORT="${VLLM_PORTS[${index}]}"
  local GATEWAY_PORT="${GATEWAY_PORTS[${index}]}"
  local GATEWAY_PARSE_PORT="${GATEWAY_PARSE_PORTS[${index}]}"
  local JAEGER_OTLP_LOGIN_PORT="${JAEGER_OTLP_PORTS[${index}]}"
  local JAEGER_UI_LOGIN_PORT="${JAEGER_UI_PORTS[${index}]}"
  local LMCACHE_INTERNAL_API_SERVER_PORT_START="${LMCACHE_PORT_STARTS[${index}]}"
  local VLLM_VISIBLE_DEVICES="${VLLM_VISIBLE_DEVICES_BY_PROFILE[${index}]}"
  local OTEL_SERVICE_NAME_WORKER="${OTEL_SERVICE_NAME}-g${GROUP_NAME}-p${PROFILE_ID}"
  local VLLM_HIP_VISIBLE_DEVICES="0"
  if [[ "${VLLM_VISIBLE_DEVICES}" == *,* ]]; then
    IFS=',' read -r -a VLLM_VISIBLE_DEVICE_LIST <<<"${VLLM_VISIBLE_DEVICES}"
    local mapped_devices=()
    for i in "${!VLLM_VISIBLE_DEVICE_LIST[@]}"; do
      mapped_devices+=("${i}")
    done
    IFS=',' VLLM_HIP_VISIBLE_DEVICES="${mapped_devices[*]}"
  fi
  local VLLM_TENSOR_PARALLEL_SIZE="${GROUP_VLLM_TENSOR_PARALLEL_SIZE}"

  local VLLM_LOG="${JOB_LOG_DIR}/vllm.${SLURM_JOB_ID}.p${PROFILE_ID}.log"
  local GATEWAY_LOG="${JOB_LOG_DIR}/gateway.${SLURM_JOB_ID}.p${PROFILE_ID}.log"

  echo "group=${GROUP_NAME} profile=${PROFILE_ID} node=$(hostname) proc=${index} vllm_port=${VLLM_SERVICE_PORT}"

  local TUNNEL_PID=""
  local REVERSE_TUNNELS_ENABLED=1
  if [[ -n "${GROUP_LOCAL_MODE_SCRIPT:-}" ]]; then
    REVERSE_TUNNELS_ENABLED=0
  fi
  start_reverse_tunnels() {
    local retries=5
    local attempt=1
    local stagger_seconds=$(( (index % 4) + 1 ))
    sleep "${stagger_seconds}"

    while [[ "${attempt}" -le "${retries}" ]]; do
      ssh "${SSH_OPTIONS[@]}" -N                     -R "${VLLM_SERVICE_PORT}:127.0.0.1:${VLLM_SERVICE_PORT}"                     -R "${JAEGER_OTLP_LOGIN_PORT}:127.0.0.1:${JAEGER_OTLP_LOCAL_PORT}"                     -R "${JAEGER_UI_LOGIN_PORT}:127.0.0.1:${JAEGER_UI_LOCAL_PORT}"                     "${LOGIN_HOST}" &
      TUNNEL_PID="$!"

      sleep 1
      if kill -0 "${TUNNEL_PID}" >/dev/null 2>&1; then
        return 0
      fi

      wait "${TUNNEL_PID}" || true
      echo "Reverse tunnel startup failed (attempt ${attempt}/${retries}) for group=${GROUP_NAME} profile=${PROFILE_ID}; retrying." >&2
      attempt=$((attempt + 1))
      sleep 2
    done

    return 1
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

  cleanup() {
    set +e
    if [[ -n "${VLLM_PID:-}" ]]; then kill "${VLLM_PID}" >/dev/null 2>&1 || true; fi
    if [[ -n "${GATEWAY_PID:-}" ]]; then kill "${GATEWAY_PID}" >/dev/null 2>&1 || true; fi
    if [[ -n "${TUNNEL_PID:-}" ]]; then kill "${TUNNEL_PID}" >/dev/null 2>&1 || true; fi
  }
  trap cleanup EXIT INT TERM

  if [[ "${REVERSE_TUNNELS_ENABLED}" -eq 1 ]]; then
    if ! start_reverse_tunnels; then
      echo "One or more reverse tunnels failed to establish. Aborting startup." >&2
      exit 72
    fi
  else
    echo "group=${GROUP_NAME} profile=${PROFILE_ID} local-mode detected; skipping reverse tunnel setup."
  fi

  VLLM_CMD=(
    /opt/vllm-plugins/vllm_entrypoint.sh
    --model "${VLLM_MODEL_NAME}"
    --served-model-name "${VLLM_SERVED_MODEL_NAME}"
    --port "${VLLM_SERVICE_PORT}"
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --otlp-traces-endpoint "${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}"
    --collect-detailed-traces "${VLLM_COLLECT_DETAILED_TRACES}"
    --enable-prompt-tokens-details
    --logits-processors "${VLLM_LOGITS_PROCESSORS}"
  )

  echo "group=${GROUP_NAME} profile=${PROFILE_ID} proc=${index} OTEL_SERVICE_NAME=${OTEL_SERVICE_NAME_WORKER} HIP_VISIBLE_DEVICES=${VLLM_HIP_VISIBLE_DEVICES} ROCR_VISIBLE_DEVICES=${VLLM_VISIBLE_DEVICES}"
  {
    local cpu_count="unknown"
    if command -v nproc >/dev/null 2>&1; then
      cpu_count="$(nproc)"
    fi
    local cpu_ids="unknown"
    if [[ -r /proc/self/status ]]; then
      cpu_ids="$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)"
    fi
    echo "group=${GROUP_NAME} profile=${PROFILE_ID} proc=${index} cpu_count=${cpu_count} cpu_ids=${cpu_ids}"
    echo "group=${GROUP_NAME} profile=${PROFILE_ID} proc=${index} running rocm-smi before vLLM startup"
    if command -v rocm-smi >/dev/null 2>&1; then
      rocm-smi || true
    else
      echo "rocm-smi not found in PATH"
    fi
  } >"${VLLM_LOG}" 2>&1

  apptainer exec                 --rocm                 --cleanenv                 "${APPTAINER_HOME_ARGS[@]}"                 "${BIND_ARGS[@]}"                 --env PYTHONNOUSERSITE=1                 --env AITER_JIT_DIR="${AITER_JIT_DIR}"                 --env XDG_CACHE_HOME="${XDG_CACHE_HOME}"                 --env VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}"                 --env HF_HOME="${HF_HOME}"                 --env HF_HUB_CACHE="${HF_HUB_CACHE}"                 --env HF_TOKEN="${HF_TOKEN}"                 --env OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME_WORKER}"                 --env OTEL_EXPORTER_OTLP_TRACES_INSECURE="${OTEL_EXPORTER_OTLP_TRACES_INSECURE}"                 --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}"                 --env HIP_VISIBLE_DEVICES="${VLLM_HIP_VISIBLE_DEVICES}"                 --env ROCR_VISIBLE_DEVICES="${VLLM_VISIBLE_DEVICES}"                 --env VLLM_MODEL_NAME="${VLLM_MODEL_NAME}"                 --env VLLM_MODEL_EXTRA_ARGS_B64="${VLLM_MODEL_EXTRA_ARGS_B64}"                 --env VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${VLLM_FORCE_SEQ_TRUST_REMOTE_CODE}"                 --env LMCACHE_INTERNAL_API_SERVER_ENABLED=1                 --env PYTHONHASHSEED=0                 --env LMCACHE_INTERNAL_API_SERVER_PORT_START="${LMCACHE_INTERNAL_API_SERVER_PORT_START}"                 "${VLLM_SIF}"                 "${VLLM_CMD[@]}"                 >>"${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!

  local GATEWAY_PID=""
  if [[ -n "${GROUP_LOCAL_MODE_SCRIPT:-}" ]]; then
    local GATEWAY_CONFIG_PATH=""
    if [[ -n "${GROUP_GATEWAY_CONFIG:-}" ]]; then
      GATEWAY_CONFIG_PATH="${GROUP_GATEWAY_CONFIG}"
    elif [[ -f "${GATEWAY_CONFIG_DEFAULT}" ]]; then
      GATEWAY_CONFIG_PATH="${GATEWAY_CONFIG_DEFAULT}"
    else
      GATEWAY_CONFIG_PATH="${GATEWAY_CONFIG_FALLBACK}"
    fi

    local GATEWAY_PYTHON="python3"
    if [[ -x "${GATEWAY_VENV_DIR}/bin/python" ]]; then
      GATEWAY_PYTHON="${GATEWAY_VENV_DIR}/bin/python"
    fi

    local GATEWAY_CMD=(
      "${GATEWAY_PYTHON}"
      -m gateway
      start
      --config "${GATEWAY_CONFIG_PATH}"
      --host "${GATEWAY_HOST}"
      --venv-dir "${GATEWAY_VENV_DIR}"
      --port-profile-id "${PROFILE_ID}"
    )
    if [[ "${GATEWAY_SKIP_INSTALL}" == "1" ]]; then
      GATEWAY_CMD+=(--skip-install)
    fi

    local GATEWAY_JAEGER_API_BASE_URL="http://127.0.0.1:${JAEGER_UI_LOCAL_PORT}/api/traces"
    local GATEWAY_OTLP_TRACES_ENDPOINT="grpc://127.0.0.1:${JAEGER_OTLP_LOCAL_PORT}"

    echo "Starting gateway command (group=${GROUP_NAME} profile=${PROFILE_ID}): ${GATEWAY_CMD[*]}"
    echo "group=${GROUP_NAME} profile=${PROFILE_ID} gateway trace endpoints: jaeger_api=${GATEWAY_JAEGER_API_BASE_URL} otlp=${GATEWAY_OTLP_TRACES_ENDPOINT}"
    GATEWAY_JAEGER_API_BASE_URL_OVERRIDE="${GATEWAY_JAEGER_API_BASE_URL}"                 GATEWAY_OTLP_TRACES_ENDPOINT_OVERRIDE="${GATEWAY_OTLP_TRACES_ENDPOINT}"                   "${GATEWAY_CMD[@]}" >"${GATEWAY_LOG}" 2>&1 &
    GATEWAY_PID=$!
  else
    echo "group=${GROUP_NAME} profile=${PROFILE_ID} gateway startup skipped (no grouped workload mode requested)."
  fi

  WAIT_PIDS=("${VLLM_PID}")
  if [[ -n "${GATEWAY_PID}" ]]; then
    WAIT_PIDS+=("${GATEWAY_PID}")
  fi
  if [[ -n "${TUNNEL_PID}" ]]; then
    WAIT_PIDS+=("${TUNNEL_PID}")
  fi
  wait -n "${WAIT_PIDS[@]}"
  EXIT_CODE=$?
  echo "group=${GROUP_NAME} profile=${PROFILE_ID} process exited with code ${EXIT_CODE} at $(date)."
  exit "${EXIT_CODE}"
}

probe_http_url() {
  local url="$1"
  python3 -c "import sys, urllib.request; req = urllib.request.Request(sys.argv[1], method='GET'); resp = urllib.request.urlopen(req, timeout=3); sys.exit(0 if int(resp.status) == 200 else 1)" "$url" >/dev/null 2>&1
}

probe_tcp_port() {
  local host="$1"
  local port="$2"
  python3 -c "import socket, sys; sock = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2); sock.close()" "$host" "$port" >/dev/null 2>&1
}

wait_for_group_vllm_ready() {
  local timeout_seconds="${SBATCH_ORCHESTRATOR_READY_TIMEOUT_SECONDS:-900}"
  local poll_interval_seconds="${SBATCH_ORCHESTRATOR_READY_POLL_INTERVAL_SECONDS:-2}"
  local deadline=$((SECONDS + timeout_seconds))
  IFS=',' read -r -a VLLM_PORTS <<<"${GROUP_VLLM_PORTS_CSV}"
  IFS=',' read -r -a GATEWAY_PORTS <<<"${GROUP_GATEWAY_PORTS_CSV}"
  IFS=',' read -r -a GATEWAY_PARSE_PORTS <<<"${GROUP_GATEWAY_PARSE_PORTS_CSV}"

  while (( SECONDS < deadline )); do
    if [[ -n "${SRUN_PID:-}" ]] && ! kill -0 "${SRUN_PID}" >/dev/null 2>&1; then
      echo "Grouped worker step exited before orchestrator readiness checks completed." >&2
      return 1
    fi

    local all_ready=1
    for idx in "${!VLLM_PORTS[@]}"; do
      local vllm_port="${VLLM_PORTS[${idx}]}"
      local gateway_port="${GATEWAY_PORTS[${idx}]}"
      local gateway_parse_port="${GATEWAY_PARSE_PORTS[${idx}]}"
      if ! probe_http_url "http://127.0.0.1:${vllm_port}/v1/models"; then
        all_ready=0
        break
      fi
      if ! probe_tcp_port "127.0.0.1" "${gateway_port}"; then
        all_ready=0
        break
      fi
      if [[ "${gateway_parse_port}" -ne "${gateway_port}" ]] && ! probe_tcp_port "127.0.0.1" "${gateway_parse_port}"; then
        all_ready=0
        break
      fi
    done

    if [[ "${all_ready}" -eq 1 ]]; then
      return 0
    fi
    sleep "${poll_interval_seconds}"
  done
  return 1
}

start_shared_jaeger() {
  local APPTAINER_HOME_ARGS=()
  if [[ -n "${VLLM_APPTAINER_HOME}" ]]; then
    mkdir -p "${VLLM_APPTAINER_HOME}"
    APPTAINER_HOME_ARGS=(-H "${VLLM_APPTAINER_HOME}")
  fi

  apptainer run                 --cleanenv                 "${APPTAINER_HOME_ARGS[@]}"                 --env COLLECTOR_ZIPKIN_HOST_PORT=:9411                 "${JAEGER_SIF}"                 >"${JAEGER_LOG_SHARED}" 2>&1 &
  JAEGER_PID=$!

  sleep 1
  if ! kill -0 "${JAEGER_PID}" >/dev/null 2>&1; then
    wait "${JAEGER_PID}" || true
    echo "Shared Jaeger failed to start. See ${JAEGER_LOG_SHARED}" >&2
    exit 95
  fi
  echo "Started shared Jaeger for group=${GROUP_NAME} pid=${JAEGER_PID} log=${JAEGER_LOG_SHARED}"
}

cleanup_group_main() {
  set +e
  if [[ -n "${ORCHESTRATOR_PID:-}" ]] && kill -0 "${ORCHESTRATOR_PID}" >/dev/null 2>&1; then
    kill "${ORCHESTRATOR_PID}" >/dev/null 2>&1 || true
    wait "${ORCHESTRATOR_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${SRUN_PID:-}" ]] && kill -0 "${SRUN_PID}" >/dev/null 2>&1; then
    kill "${SRUN_PID}" >/dev/null 2>&1 || true
    wait "${SRUN_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${JAEGER_PID:-}" ]] && kill -0 "${JAEGER_PID}" >/dev/null 2>&1; then
    kill "${JAEGER_PID}" >/dev/null 2>&1 || true
    wait "${JAEGER_PID}" >/dev/null 2>&1 || true
  fi
}

export LOGIN_HOST GROUP_NAME GROUP_SIZE
export GROUP_PROFILE_IDS_CSV GROUP_VLLM_PORTS_CSV GROUP_GATEWAY_PORTS_CSV GROUP_GATEWAY_PARSE_PORTS_CSV
export GROUP_JAEGER_OTLP_LOGIN_PORTS_CSV GROUP_JAEGER_UI_LOGIN_PORTS_CSV
export GROUP_LMCACHE_INTERNAL_API_SERVER_PORT_START_CSV
export GROUP_VLLM_VISIBLE_DEVICES_SEMICOLON GROUP_VLLM_TENSOR_PARALLEL_SIZE
export JAEGER_OTLP_LOCAL_PORT JAEGER_UI_LOCAL_PORT JAEGER_SIF VLLM_SIF
export VLLM_MODEL_NAME VLLM_SERVED_MODEL_NAME
export VLLM_APPTAINER_HOME HF_HOME HF_HUB_CACHE HF_TOKEN
export AITER_JIT_DIR XDG_CACHE_HOME VLLM_CACHE_ROOT
export OTEL_SERVICE_NAME OTEL_EXPORTER_OTLP_TRACES_INSECURE OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
export VLLM_COLLECT_DETAILED_TRACES VLLM_LOGITS_PROCESSORS
export VLLM_MODEL_EXTRA_ARGS_B64 VLLM_FORCE_SEQ_TRUST_REMOTE_CODE
export GATEWAY_CONFIG_DEFAULT GATEWAY_CONFIG_FALLBACK GATEWAY_VENV_DIR GATEWAY_HOST GATEWAY_SKIP_INSTALL GROUP_GATEWAY_CONFIG
export JOB_LOG_DIR GROUP_LOCAL_MODE_SCRIPT GROUP_LOCAL_MODE_LOG JAEGER_LOG_SHARED
export -f run_group_worker
trap cleanup_group_main EXIT INT TERM
start_shared_jaeger
srun               --nodes=1               --ntasks="${GROUP_SIZE}"               --ntasks-per-node="${GROUP_SIZE}"               --kill-on-bad-exit=1               bash -lc 'run_group_worker' &
SRUN_PID=$!

if [[ -n "${GROUP_LOCAL_MODE_SCRIPT}" ]]; then
  echo "Detected grouped local-mode script; waiting for grouped vLLM readiness."
  if ! wait_for_group_vllm_ready; then
    echo "Timed out waiting for grouped vLLM readiness before running grouped local-mode script." >&2
    exit 98
  fi

  echo "Executing grouped local-mode script: ${GROUP_LOCAL_MODE_SCRIPT}"
  set +e
  bash -lc "${GROUP_LOCAL_MODE_SCRIPT}" >"${GROUP_LOCAL_MODE_LOG}" 2>&1 &
  ORCHESTRATOR_PID=$!
  set -e
else
  wait "${SRUN_PID}"
  exit $?
fi

set +e
wait -n "${SRUN_PID}" "${ORCHESTRATOR_PID}"
FIRST_EXIT_CODE=$?
set -e

if ! kill -0 "${ORCHESTRATOR_PID}" >/dev/null 2>&1; then
  ORCHESTRATOR_EXIT_CODE="${FIRST_EXIT_CODE}"
  set +e
  wait "${ORCHESTRATOR_PID}"
  WAIT_ORCHESTRATOR_EXIT_CODE=$?
  set -e
  if [[ "${WAIT_ORCHESTRATOR_EXIT_CODE}" -ne 127 ]]; then
    ORCHESTRATOR_EXIT_CODE="${WAIT_ORCHESTRATOR_EXIT_CODE}"
  fi
  if [[ "${ORCHESTRATOR_EXIT_CODE}" -ne 0 ]]; then
    echo "Grouped workload command failed with exit code ${ORCHESTRATOR_EXIT_CODE}." >&2
    exit "${ORCHESTRATOR_EXIT_CODE}"
  fi

  echo "Grouped workload command finished successfully; stopping grouped workers."
  if kill -0 "${SRUN_PID}" >/dev/null 2>&1; then
    kill "${SRUN_PID}" >/dev/null 2>&1 || true
  fi
  wait "${SRUN_PID}" >/dev/null 2>&1 || true
  exit 0
fi

echo "Grouped worker step exited before grouped workload command completion." >&2
if kill -0 "${ORCHESTRATOR_PID}" >/dev/null 2>&1; then
  kill "${ORCHESTRATOR_PID}" >/dev/null 2>&1 || true
fi
wait "${ORCHESTRATOR_PID}" >/dev/null 2>&1 || true
wait "${SRUN_PID}" >/dev/null 2>&1 || true
exit "${FIRST_EXIT_CODE}"
