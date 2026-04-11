# AMD HPC MI2104X Embedded TP1

This directory contains a dedicated `mi2104x` embedded TP=1 launcher modeled
after `servers/servers-amdhpc-interactive-mi2104x-embedded-TP1`, but wired for
dedicated batch experiments through `launch.py`.

The stack partitions one four-GPU `mi2104x` node into four fixed TP=1 service
profiles:

- one shared AMD SMI power daemon for `amd-power-reader`
- shared Jaeger
- one TP=1 vLLM per port profile `0..3`, pinned to GPU `0..3`
- one ctx-aware gateway (`gateway_ctx`) per port profile `0..3`
- one experiment-script invocation per port profile after the services are ready

## Render

```bash
python3 servers/servers-amdhpc-mi2104x-embedded-TP1/launch.py render \
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
python3 servers/servers-amdhpc-mi2104x-embedded-TP1/launch.py submit \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/single-agent/run_all.sh
```

Example with a longer wall-clock limit:

```bash
python3 servers/servers-amdhpc-mi2104x-embedded-TP1/launch.py submit \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/single-agent/run_all.sh \
  --time-limit 12:00:00
```

The rendered sbatch script is written under
`servers/servers-amdhpc-mi2104x-embedded-TP1/run/` and Slurm logs land under
`servers/servers-amdhpc-mi2104x-embedded-TP1/logs/`.

Each experiment invocation writes combined stdout/stderr to:

- `servers/servers-amdhpc-mi2104x-embedded-TP1/logs/experiment.<run_id>.p0.log`
- `servers/servers-amdhpc-mi2104x-embedded-TP1/logs/experiment.<run_id>.p1.log`
- `servers/servers-amdhpc-mi2104x-embedded-TP1/logs/experiment.<run_id>.p2.log`
- `servers/servers-amdhpc-mi2104x-embedded-TP1/logs/experiment.<run_id>.p3.log`

## Direct Node Run

If you already have a `mi2104x` allocation and want to run the job logic
directly on the node, you can launch the same stack without `sbatch`:

```bash
EXPERIMENT_SCRIPT=./experiments/single-agent/run_all.sh \
VLLM_MODEL_KEY=qwen3_coder_30b_fp8 \
bash servers/servers-amdhpc-mi2104x-embedded-TP1/start-services.sh
```

## Experiment Environment

The launcher invokes the experiment as:

```bash
"${EXPERIMENT_RUNNER:-bash}" "${EXPERIMENT_SCRIPT}" "${profile_id}"
```

Before each profile launch, it exports:

- `PORT_PROFILE_ID=<profile id>`
- `VLLM_BASE_URL=http://127.0.0.1:<profile vllm port>`
- `GATEWAY_BASE_URL=http://127.0.0.1:<profile gateway port>` for `gateway_ctx`
- `GATEWAY_PARSE_BASE_URL=http://127.0.0.1:<profile gateway parse port>` for `gateway_ctx`
- `JAEGER_BASE_URL=http://127.0.0.1:16686`
- `AMD_SMI_POWER_SOCKET_PATH=<job socket>`

The experiment runs in the background and redirects combined stdout/stderr to
the per-profile experiment log file above.

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
- `AMD_SMI_POWER_SOCKET_PATH`
- `HF_HOME`
- `HF_HUB_CACHE`
- `HF_TOKEN`
- `GATEWAY_HOST`
- `GATEWAY_CONFIG`
- `GATEWAY_VENV_DIR`
- `RUN_ID`

`launch.py` also accepts repeatable `--env KEY=VALUE` values. Those are encoded
and forwarded into the vLLM `apptainer exec` environment without needing a
generated bash template per run.

By default the launcher resolves `VLLM_MODEL_KEY` to `qwen3_coder_30b_fp8`,
which maps to `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` in
`configs/model_config.toml`.

## Prerequisite

Install the AMD power and gateway tools on the node-visible Python environment:

```bash
pip install -e ./amd-power-reader
pip install -e ./gateway_ctx
```

## Notes

- The default Slurm time limit is `24:00:00`, matching the current `mi2104x`
  partition config.
- Model fit validation is based on `75%` of one MI210 GPU's VRAM because each
  profile is pinned to a single GPU.
- This workflow keeps the explicit four-profile layout from the interactive
  `mi2104x` launcher while adding a batch-oriented `launch.py` entrypoint.
