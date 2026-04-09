# Interactive AMD HPC Embedded TP1 MI3001X Sweep QPS Power Clean AMD

This workflow is the single-node `mi3001x` interactive AMD variant of
`experiments/sweep-qps-docker-power-clean`.

It keeps the same clean split-plan lookup and Poisson replay generation as the
embedded TP1 variant, but assumes you have already started the service stack on
an interactive `mi3001x` allocation with
`servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/client.py start`.

Key differences from the batch embedded-TP1 experiment:

- no `submit_embedded_tp1.sh` is generated
- the generated `run_replay.sh` sources the shared helper in
  `servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/experiment-env.sh`
  to validate `port_profile_id=0`, resolve ports, and wait for the interactive
  vLLM + gateway stack
- the runtime port profile is fixed to `0`
- the runtime GPU index is fixed to `0`
- outputs are written under `profile-0/` for consistency with the embedded TP1
  directory layout
- power logging uses `amd-power-reader`

## Requirements

Install the local console script used by the generated runner:

```bash
python3 -m pip install -e './amd-power-reader'
```

On the same interactive `mi3001x` node, start the service stack in the
background:

```bash
python3 servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/client.py start
```

`client.py start` is blocking: it prints progress until the background service
stack is ready, then exits while leaving the services running. That
launcher-managed stack starts `amd-smi-power-daemon` itself. The generated
`run_replay.sh` sources the shared helper from
`servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/experiment-env.sh`
and waits for the service stack before it starts replaying. New generated
bundles default `AMD_SMI_POWER_SOCKET_PATH` to
`/tmp/amdsmi-power-reader.sock`, and the environment variable remains an
override if you need a different socket.

## Generate One Sweep Bundle

```bash
python3 experiments/amd-embeded/servers-amdhpc-interactive-mi3001x-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-amd/generate_experiment.py \
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

- `--additional-suffix`

This `mi3001x` variant assumes `port_profile_id=0` and `gpu_index=0`.
The compatibility flags `--port-profile` and `--gpu-index` still exist, but
they must remain `0`.

## Run Generated Sweep

After `client.py start` reports readiness:

```bash
bash experiments/amd-embeded/servers-amdhpc-interactive-mi3001x-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-amd/generated/<timestamp>/run_replay.sh
```

Explicitly pass the fixed `mi3001x` port profile:

```bash
bash experiments/amd-embeded/servers-amdhpc-interactive-mi3001x-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-amd/generated/<timestamp>/run_replay.sh 0
```

Useful runtime env overrides:

- `PORT_PROFILE_ID` (`0` only)
- `GPU_INDEX` (`0` only)
- `AMD_SMI_POWER_SOCKET_PATH`
- `AMD_POWER_READER_BIN`
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

- `results/replay/amd-embeded/servers-amdhpc-interactive-mi3001x-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-amd/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/profile-0/`

Each profile output also includes:

- AMD power logs under `<replay_output_dir>/power/`

## Tests

```bash
pytest experiments/amd-embeded/servers-amdhpc-interactive-mi3001x-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-amd/test -q
```
