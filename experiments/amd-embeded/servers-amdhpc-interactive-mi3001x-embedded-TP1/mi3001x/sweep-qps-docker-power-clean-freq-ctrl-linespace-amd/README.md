# Interactive AMD HPC Embedded TP1 MI3001X Sweep QPS Power Clean Freq Ctrl Linespace AMD

This workflow is the single-node `mi3001x` interactive AMD variant of
`experiments/sweep-qps-docker-power-clean-gateway_multi-freq-ctrl-linesapce`.

It keeps the same clean split-plan lookup, Poisson replay generation, and
gateway `ctx-aware` start/end flow as the embedded TP1 variant, but assumes the
service stack is already running on an interactive `mi3001x` allocation via
`servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/client.py start`.

Key differences from the batch embedded-TP1 experiment:

- no `submit_embedded_tp1.sh` is generated
- the generated `run_replay.sh` sources the shared helper in
  `servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/experiment-env.sh`
  to validate `port_profile_id=0`, resolve ports, and wait for the interactive
  vLLM + gateway stack
- the runtime port profile is fixed to `0`
- the runtime GPU index is fixed to `0`
- outputs are still written under `profile-0/` for consistency with the
  embedded TP1 directory layout
- power logging uses `amd-power-reader`
- frequency control uses `freq-controller-linespace-amd`
- GPU reset uses `amd-reset-gpu-core-freq`

Default thresholds:

- `ctx-aware usage_threshold_tokens = 1445793`
- `ctx-aware scheduling_threshold_tokens = 1369699`
- `freq-controller-linespace-amd threshold = 1141416`

## Requirements

Install the local console scripts used by the generated runner:

```bash
python3 -m pip install -e './amd-power-reader'
python3 -m pip install -e './freq-controller-linespace-amd'
```

On the same interactive `mi3001x` node, start the service stack in the
background:

```bash
python3 servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/client.py start
```

`client.py start` is blocking: it prints progress until the background service
stack is ready, then exits while leaving the services running. That
launcher-managed stack starts `amd-smi-power-daemon` itself and prints the
active `AMD_SMI_POWER_SOCKET_PATH` when the stack is ready.
The generated `run_replay.sh` sources the shared helper from
`servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/experiment-env.sh`
and uses it for port resolution and service readiness checks. New generated
bundles default `AMD_SMI_POWER_SOCKET_PATH` to
`/tmp/amdsmi-power-reader.sock`, and the environment variable remains an
override if you need a different socket.

## Generate One Sweep Bundle

```bash
python3 experiments/amd-embeded/servers-amdhpc-interactive-mi3001x-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.1 \
  --time-constraint-s 1000 \
  --target-model qwen3_coder_30b_fp8 \
  --additional-suffix qwen3_fp8 \
  --split exclude-unranked
```

Optional overrides:

- `--ctx-aware-usage-threshold-tokens`
- `--ctx-aware-scheduling-threshold-tokens`
- `--freq-controller-threshold`
- `--additional-suffix`
- `--output-suffix`

This `mi3001x` variant assumes `port_profile_id=0` and `gpu_index=0`.
The compatibility flags `--port-profile` and `--gpu-index` still exist, but
they must remain `0`.

## Run Generated Sweep

After `client.py start` reports readiness:

```bash
bash experiments/amd-embeded/servers-amdhpc-interactive-mi3001x-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh
```

Explicitly pass the fixed `mi3001x` port profile:

```bash
bash experiments/amd-embeded/servers-amdhpc-interactive-mi3001x-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh 0
```

Useful runtime env overrides:

- `PORT_PROFILE_ID` (`0` only)
- `GPU_INDEX` (`0` only)
- `GATEWAY_BASE_URL`
- `AMD_SMI_POWER_SOCKET_PATH`
- `CTX_AWARE_USAGE_THRESHOLD_TOKENS`
- `CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS`
- `FREQ_CONTROLLER_CONFIG`
- `FREQ_CONTROLLER_THRESHOLD`
- `AMD_POWER_READER_BIN`
- `FREQ_CONTROLLER_BIN`
- `RESET_GPU_CORE_FREQ_BIN`
- `INTERACTIVE_CLIENT_SCRIPT`
- `INTERACTIVE_ENV_HELPER`
- `INTERACTIVE_START_SERVICES_COMMAND`
- `INTERACTIVE_START_SERVICES_SCRIPT`
- `SERVICE_READY_TIMEOUT_SECONDS`
- `SERVICE_READY_POLL_INTERVAL_SECONDS`

`run_replay.sh` performs a service preflight against profile `0` before it
starts the replay and prints a hint to
`python3 servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/client.py start`
if the stack is not ready yet.

When the replay finishes, stop the background stack with:

```bash
python3 servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/client.py stop
```

## Generated Files

For each QPS slug `qpsX`:

- `generated/<timestamp>/qpsX/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<selected-plan>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

Default replay output layout:

- `results/replay/amd-embeded/servers-amdhpc-interactive-mi3001x-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/profile-0/`

Each profile output also includes:

- AMD power logs under `<replay_output_dir>/power/`
- freq controller logs under `<replay_output_dir>/freq-control-linespace-amd/`
- gateway ctx-aware sampler logs under
  `<replay_output_dir>/gateway-output/job/ctx_aware_*_profile-<port_profile_id>.jsonl`

## Tests

```bash
pytest experiments/amd-embeded/servers-amdhpc-interactive-mi3001x-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/test -q
```
