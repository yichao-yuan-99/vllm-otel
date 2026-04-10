# AMD HPC MI2508X Embedded TP2

This directory contains a dedicated `mi2508x` embedded TP=2 launcher with the
same runtime stack as
`servers/servers-amdhpc-interactive-mi2508x-embedded-TP2`, but wrapped in a
static batch-oriented layout with `launch.py` render/submit entrypoints.

The stack is fixed to port profiles `0..3` on one `mi2508x` node and includes:

- one shared AMD SMI power daemon for `amd-power-reader`
- shared Jaeger
- one TP=2 vLLM per port profile `0..3`, pinned to GPU pairs `0,1`, `2,3`,
  `4,5`, and `6,7`
- one ctx-aware gateway (`gateway_ctx`) per port profile `0..3`
- one experiment-script invocation per port profile after the services are ready

`start-services.sh` explicitly unrolls all four `launch_vllm`,
`wait_for_vllm_ready`, `launch_gateway`, `wait_for_gateway_ready`, and
`launch_experiment` calls so the runtime shape stays easy to inspect.

## Render

```bash
python3 servers/servers-amdhpc-mi2508x-embedded-TP2/launch.py render \
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
python3 servers/servers-amdhpc-mi2508x-embedded-TP2/launch.py submit \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/single-agent/run_all.sh
```

Example with a longer wall-clock limit:

```bash
python3 servers/servers-amdhpc-mi2508x-embedded-TP2/launch.py submit \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/single-agent/run_all.sh \
  --time-limit 1-00:00:00
```

The rendered sbatch script is written under
`servers/servers-amdhpc-mi2508x-embedded-TP2/run/` and Slurm logs land under
`servers/servers-amdhpc-mi2508x-embedded-TP2/logs/`.

Each experiment invocation also writes combined stdout/stderr to:

- `servers/servers-amdhpc-mi2508x-embedded-TP2/logs/experiment.<run_id>.p<profile_id>.log`

## Direct Node Run

If you already have a `mi2508x` allocation and want to run the job logic
directly on the node, you can launch it without `sbatch`:

```bash
EXPERIMENT_SCRIPT=./experiments/single-agent/run_all.sh \
VLLM_MODEL_KEY=qwen3_coder_30b_fp8 \
bash servers/servers-amdhpc-mi2508x-embedded-TP2/start-services.sh
```

## Experiment Environment

Each experiment is invoked as:

```bash
"${EXPERIMENT_RUNNER:-bash}" "${EXPERIMENT_SCRIPT}" <profile_id>
```

Before launch, the script exports per-profile values:

- `PORT_PROFILE_ID=<profile_id>`
- `VLLM_BASE_URL=http://127.0.0.1:<profile vllm port>`
- `GATEWAY_BASE_URL=http://127.0.0.1:<profile gateway port>` for `gateway_ctx`
- `GATEWAY_PARSE_BASE_URL=http://127.0.0.1:<profile gateway parse port>` for `gateway_ctx`
- `JAEGER_BASE_URL=http://127.0.0.1:16686`
- `AMD_SMI_POWER_SOCKET_PATH=<job socket>`

The launcher runs each experiment in the background and redirects its combined
stdout/stderr to the per-profile experiment log file above.

## Default Ports

- Jaeger UI/API: `16686`
- Jaeger OTLP gRPC: `4317`
- Profile `0`: vLLM `11451`, gateway-ctx `11457`, gateway-ctx parse `18171`
- Profile `1`: vLLM `24123`, gateway-ctx `24157`, gateway-ctx parse `28171`
- Profile `2`: vLLM `31987`, gateway-ctx `31955`, gateway-ctx parse `38171`
- Profile `3`: vLLM `40823`, gateway-ctx `40857`, gateway-ctx parse `48171`

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
and forwarded into every vLLM `apptainer exec` environment, matching the other
embedded renderers without needing a large generated bash template.

## Prerequisite

Install the AMD power and gateway-ctx tools on the node-visible Python environment:

```bash
pip install -e ./amd-power-reader
pip install -e ./gateway_ctx
```

## Notes

- This workflow is intentionally fixed to `mi2508x` and port profiles `0..3`.
- The default Slurm time limit is `12:00:00`, and you can override it with
  `--time-limit`.
- The launcher validates model fit against `75%` of one TP2 GPU pair's VRAM on
  `mi2508x` (`96 GB`).
- Profile `n` is pinned with `ROCR_VISIBLE_DEVICES` set to its assigned GPU
  pair, while `HIP_VISIBLE_DEVICES` is remapped inside each container to `0,1`.
- Jaeger is shared across the node, matching the interactive TP2 launcher.
