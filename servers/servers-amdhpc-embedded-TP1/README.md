# Embedded AMD HPC TP1

This directory contains a lightweight launcher for AMD HPC jobs that run the
full stack directly on the allocated node:

- shared Jaeger
- one TP=1 vLLM per selected port profile
- one gateway per selected port profile
- one experiment-script invocation per selected port profile

Supported partitions:

- `mi3001x`: one service stack on port profile `0`
- `mi3008x`: eight service stacks on port profiles `0..7`, one GPU each

For `mi3008x`, the rendered sbatch does not use `srun`. It hardcodes and
unrolls all eight vLLM/gateway launches and pins each worker with
`ROCR_VISIBLE_DEVICES=<gpu>` and `HIP_VISIBLE_DEVICES=0`.

The user provides:

- a model key from [`configs/model_config.toml`](/srv/scratch/yichaoy2/work/vllm-otel/configs/model_config.toml)
- an experiment script path

The experiment script is invoked once per profile and receives the port profile
as its only positional argument.

## Render

```bash
python3 servers/servers-amdhpc-embedded-TP1/launch.py render \
  -p mi3008x \
  -m qwen3_coder_30b \
  -e ./experiments/sweep-qps-same-agent/generated/<timestamp>/run_replay.sh
```

Useful options:

- `--lmcache <size>`
- `--extra-vllm-args "<args>"`
- `--no-async-scheduling`
- `--env KEY=VALUE` (repeatable)

## Submit

```bash
python3 servers/servers-amdhpc-embedded-TP1/launch.py submit \
  -p mi3001x \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/single-agent/run_all.sh
```

## Runtime Notes

- Default config: [`server_config.toml`](/srv/scratch/yichaoy2/work/vllm-otel/servers/servers-amdhpc-embedded-TP1/server_config.toml)
- Rendered scripts/logs are written under `servers/servers-amdhpc-embedded-TP1/run/`
  and `servers/servers-amdhpc-embedded-TP1/logs/`
- vLLM is always rendered with `tensor-parallel-size=1`
- Model fit is validated against `75%` of one GPU's VRAM, even on `mi3008x`
- Override the experiment runner at job runtime with `EXPERIMENT_RUNNER=python3`
  if the supplied script is a Python entrypoint rather than a shell script
