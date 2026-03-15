#!/usr/bin/env bash
#SBATCH --job-name=vllm_job
#SBATCH --output=/work1/talati/yichaoy/vllm-otel/apptainer/logs/slurm.%j.out
#SBATCH --error=/work1/talati/yichaoy/vllm-otel/apptainer/logs/slurm.%j.err
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --partition=mi3008x

set -euo pipefail

echo "Job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"

LOGIN_HOST=login
VLLM_SERVICE_PORT=11451
JAEGER_OTLP_PORT=4317
JAEGER_UI_PORT=16686

JAEGER_SIF=/work1/talati/yichaoy/apptainer_imgs/jaeger-all-in-one-1.57.sif
VLLM_SIF=/work1/talati/yichaoy/apptainer_imgs/vllm-openai-otel-v0.16.0-otel-lp-rocm.sif

VLLM_MODEL_NAME=moonshotai/Kimi-K2.5
VLLM_SERVED_MODEL_NAME=Kimi-K2.5
VLLM_TENSOR_PARALLEL_SIZE=8
VLLM_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

VLLM_APPTAINER_HOME=''
HF_HOME=/work1/talati/yichaoy/huggingface
HF_HUB_CACHE=/work1/talati/yichaoy/huggingface/hub
HF_TOKEN="${HF_TOKEN:-}"

AITER_JIT_DIR=/tmp/vllm-aiter-jit-yichaoy
XDG_CACHE_HOME=/tmp/vllm-runtime-yichaoy/xdg-cache
VLLM_CACHE_ROOT=/tmp/vllm-runtime-yichaoy/xdg-cache/vllm

OTEL_SERVICE_NAME=vllm-server
OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://127.0.0.1:4317
VLLM_COLLECT_DETAILED_TRACES=all
VLLM_LOGITS_PROCESSORS=forceSeq.force_sequence_logits_processor:ForceSequenceAdapter
VLLM_FORCE_SEQ_TRUST_REMOTE_CODE=true

JOB_LOG_DIR=/work1/talati/yichaoy/vllm-otel/apptainer/logs
mkdir -p "${JOB_LOG_DIR}" "${AITER_JIT_DIR}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

JAEGER_LOG="${JOB_LOG_DIR}/jaeger.${SLURM_JOB_ID}.log"
VLLM_LOG="${JOB_LOG_DIR}/vllm.${SLURM_JOB_ID}.log"

SSH_OPTIONS=(-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
VLLM_EXTRA_ARGS=(--mm-encoder-tp-mode data --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2)

TUNNEL_PIDS=()
start_tunnel() {
  local remote_port="$1"
  local local_port="$2"
  ssh "${SSH_OPTIONS[@]}" -N -R "${remote_port}:127.0.0.1:${local_port}" "${LOGIN_HOST}" &
  TUNNEL_PIDS+=("$!")
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
  if [[ -n "${JAEGER_PID:-}" ]]; then kill "${JAEGER_PID}" >/dev/null 2>&1 || true; fi
  for pid in "${TUNNEL_PIDS[@]}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

# Reverse tunnels from compute node to login node.
start_tunnel "${VLLM_SERVICE_PORT}" "${VLLM_SERVICE_PORT}"
start_tunnel "${JAEGER_OTLP_PORT}" "${JAEGER_OTLP_PORT}"
start_tunnel "${JAEGER_UI_PORT}" "${JAEGER_UI_PORT}"

apptainer run               --cleanenv               "${APPTAINER_HOME_ARGS[@]}"               --env COLLECTOR_ZIPKIN_HOST_PORT=:9411               "${JAEGER_SIF}"               >"${JAEGER_LOG}" 2>&1 &
JAEGER_PID=$!

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
if [[ ${#VLLM_EXTRA_ARGS[@]} -gt 0 ]]; then
  VLLM_CMD+=("${VLLM_EXTRA_ARGS[@]}")
fi

apptainer exec               --rocm               --cleanenv               "${APPTAINER_HOME_ARGS[@]}"               "${BIND_ARGS[@]}"               --env PYTHONNOUSERSITE=1               --env AITER_JIT_DIR="${AITER_JIT_DIR}"               --env XDG_CACHE_HOME="${XDG_CACHE_HOME}"               --env VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}"               --env HF_HOME="${HF_HOME}"               --env HF_HUB_CACHE="${HF_HUB_CACHE}"               --env HF_TOKEN="${HF_TOKEN}"               --env OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME}"               --env OTEL_EXPORTER_OTLP_TRACES_INSECURE="${OTEL_EXPORTER_OTLP_TRACES_INSECURE}"               --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}"               --env HIP_VISIBLE_DEVICES="${VLLM_VISIBLE_DEVICES}"               --env ROCR_VISIBLE_DEVICES="${VLLM_VISIBLE_DEVICES}"               --env VLLM_MODEL_NAME="${VLLM_MODEL_NAME}"               --env VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${VLLM_FORCE_SEQ_TRUST_REMOTE_CODE}"               "${VLLM_SIF}"               "${VLLM_CMD[@]}"               >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

wait -n "${JAEGER_PID}" "${VLLM_PID}"
EXIT_CODE=$?
echo "One service exited with code ${EXIT_CODE} at $(date)."
exit "${EXIT_CODE}"
