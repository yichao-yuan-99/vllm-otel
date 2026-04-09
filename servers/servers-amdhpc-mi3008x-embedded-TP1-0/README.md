# AMD HPC MI3008X Embedded TP1 Profile 0

This directory contains a dedicated `mi3008x` embedded TP=1 launcher that keeps
the `servers/servers-amdhpc-mi3008x-embedded-TP1` partition/job shape, but only
starts port profile `0` on GPU `0`. The runtime end effect is intentionally
similar to `servers/servers-amdhpc-mi3001x-embedded-TP1`.

The stack is fixed to port profile `0` on one `mi3008x` GPU and includes:

- one shared AMD SMI power daemon for `amd-power-reader`
- shared Jaeger
- one TP=1 vLLM on port profile `0`
- one ctx-aware gateway (`gateway_ctx`) on port profile `0`
- one experiment-script invocation after the services are ready

## Render

```bash
python3 servers/servers-amdhpc-mi3008x-embedded-TP1-0/launch.py render \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/single-agent/run_all.sh
```

Useful options:

- `--time-limit <slurm time>`
- `--lmcache <size>`
- `--extra-vllm-args "<args>"`
- `--no-async-scheduling`
- `--env KEY=VALUE` (repeatable)

## Submit

```bash
python3 servers/servers-amdhpc-mi3008x-embedded-TP1-0/launch.py submit \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/single-agent/run_all.sh
```

Example with a longer wall-clock limit:

```bash
python3 servers/servers-amdhpc-mi3008x-embedded-TP1-0/launch.py submit \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/single-agent/run_all.sh \
  --time-limit 1-00:00:00
```

The rendered sbatch script is written under
`servers/servers-amdhpc-mi3008x-embedded-TP1-0/run/` and Slurm logs land under
`servers/servers-amdhpc-mi3008x-embedded-TP1-0/logs/`.

The experiment invocation writes combined stdout/stderr to:

- `servers/servers-amdhpc-mi3008x-embedded-TP1-0/logs/experiment.<run_id>.p0.log`

## Direct Node Run

If you already have a `mi3008x` allocation and want to run the job logic
directly on the node, you can also launch it without `sbatch`:

```bash
EXPERIMENT_SCRIPT=./experiments/single-agent/run_all.sh \
VLLM_MODEL_KEY=qwen3_coder_30b_fp8 \
bash servers/servers-amdhpc-mi3008x-embedded-TP1-0/start-services.sh
```

## Experiment Environment

The experiment is invoked as:

```bash
"${EXPERIMENT_RUNNER:-bash}" "${EXPERIMENT_SCRIPT}" 0
```

Before launch, the script exports:

- `PORT_PROFILE_ID=0`
- `VLLM_BASE_URL=http://127.0.0.1:11451`
- `GATEWAY_BASE_URL=http://127.0.0.1:11457` for `gateway_ctx`
- `GATEWAY_PARSE_BASE_URL=http://127.0.0.1:18171` for `gateway_ctx`
- `JAEGER_BASE_URL=http://127.0.0.1:16686`
- `AMD_SMI_POWER_SOCKET_PATH=<job socket>`

The launcher runs the experiment in the background and redirects its combined
stdout/stderr to the experiment log file above.

## Default Ports

- vLLM: `11451`
- gateway-ctx: `11457`
- gateway-ctx parse: `18171`
- Jaeger UI/API: `16686`
- Jaeger OTLP gRPC: `4317`

## Useful Overrides

- `EXPERIMENT_RUNNER`
- `VLLM_MODEL_KEY`
- `MODEL_CONFIG_PATH`
- `VLLM_MODEL_NAME`
- `VLLM_SERVED_MODEL_NAME`
- `VLLM_MODEL_EXTRA_ARGS_B64`
- `VLLM_SIF`
- `JAEGER_SIF`
- `AMD_SMI_POWER_DAEMON_BIN`
- `AMD_POWER_READER_BIN`
- `AMD_SMI_POWER_SOCKET_PATH`
- `HF_HOME`
- `HF_HUB_CACHE`
- `HF_TOKEN`
- `GATEWAY_HOST`
- `GATEWAY_CONFIG`
- `GATEWAY_VENV_DIR`
- `RUN_ID`

`launch.py` also accepts repeatable `--env KEY=VALUE` values. Those are encoded
and forwarded into the vLLM `apptainer exec` environment, matching the older
embedded renderer behavior without needing a large generated bash template.

## Prerequisite

Install the AMD power and gateway-ctx tools on the node-visible Python environment:

```bash
pip install -e ./amd-power-reader
pip install -e ./gateway_ctx
```

## Notes

- This workflow is intentionally fixed to `mi3008x` and port profile `0`.
- Only one GPU is used even though the job runs on the `mi3008x` partition.
- The default Slurm time limit is `12:00:00`, and you can override it with
  `--time-limit`.
- The launcher still validates model fit against `75%` of one MI300 GPU's VRAM.
- Compared with `servers/servers-amdhpc-mi3008x-embedded-TP1`, this variant
  does not launch profiles `1..7`.
