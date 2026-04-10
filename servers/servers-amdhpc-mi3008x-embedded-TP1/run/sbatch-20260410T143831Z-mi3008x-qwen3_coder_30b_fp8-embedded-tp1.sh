#!/usr/bin/env bash
#SBATCH --job-name=vllm_embedded_tp1_mi3008x_tp1
#SBATCH --output=/work1/talati/yichaoy/vllm-otel/servers/servers-amdhpc-mi3008x-embedded-TP1/logs/slurm.%j.out
#SBATCH --error=/work1/talati/yichaoy/vllm-otel/servers/servers-amdhpc-mi3008x-embedded-TP1/logs/slurm.%j.err
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=mi3008x

set -euo pipefail

REPO_ROOT=/work1/talati/yichaoy/vllm-otel
cd "${REPO_ROOT}"

export MODEL_CONFIG_PATH=/work1/talati/yichaoy/vllm-otel/configs/model_config.toml
export EXPERIMENT_SCRIPT=/work1/talati/yichaoy/vllm-otel/experiments/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/sweep-qps-docker-power-clean-amd/generated/20260410T143801Z/run_replay.sh
export VLLM_MODEL_KEY=qwen3_coder_30b_fp8
export VLLM_MODEL_EXTRA_ARGS_B64=WyItLXRydXN0LXJlbW90ZS1jb2RlIl0=
export VLLM_EXTRA_ENV_B64=''
export AMD_SMI_POWER_DAEMON_BIN="${AMD_SMI_POWER_DAEMON_BIN:-/home1/yichaoy/vllm-otel/.venv/bin/amd-smi-power-daemon}"
export AMD_POWER_READER_BIN="${AMD_POWER_READER_BIN:-/home1/yichaoy/vllm-otel/.venv/bin/amd-power-reader}"
export AMD_SMI_POWER_SOCKET_PATH="${AMD_SMI_POWER_SOCKET_PATH:-/tmp/amdsmi-power-reader.${SLURM_JOB_ID}.sock}"

bash /work1/talati/yichaoy/vllm-otel/servers/servers-amdhpc-mi3008x-embedded-TP1/start-services.sh
