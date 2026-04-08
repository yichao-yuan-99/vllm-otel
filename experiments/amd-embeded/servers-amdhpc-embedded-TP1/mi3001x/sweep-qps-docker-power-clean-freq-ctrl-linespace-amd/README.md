# Embedded AMD HPC TP1 MI3001X Sweep QPS Power Clean Freq Ctrl Linespace AMD

This workflow is the single-node `mi3001x` embedded-TP1 AMD variant of
`experiments/sweep-qps-docker-power-clean-gateway_multi-freq-ctrl-linesapce`.

It keeps the same clean split-plan lookup, Poisson replay generation, and
gateway `ctx-aware` start/end flow, but adapts the runtime model to
`servers/servers-amdhpc-embedded-TP1` for the `mi3001x` partition:

- the generated `run_replay.sh` is invoked once on a single `mi3001x` node
- the runtime port profile is fixed to `0`
- the runtime GPU index is fixed to `0`
- outputs are still written under `profile-0/` for consistency with the embedded
  TP1 directory layout
- power logging uses `amd-power-reader`
- frequency control uses `freq-controller-linespace-amd`
- GPU reset uses `amd-reset-gpu-core-freq`

If launched through
`servers/servers-amdhpc-embedded-TP1/launch.py`, the job-level sbatch already
starts `amd-smi-power-daemon` and exports `AMD_SMI_POWER_SOCKET_PATH` for the
experiment script. For direct runs, start the daemon yourself first.

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

For direct runs outside embedded TP1, start the daemon first:

```bash
amd-smi-power-daemon &
export AMD_SMI_POWER_SOCKET_PATH=/tmp/amdsmi-power-reader.sock
```

## Generate One Sweep Bundle

```bash
python3 experiments/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.25,0.3,0.35 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --split rest
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

## Run Generated Sweep Directly

```bash
bash experiments/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh
```

Explicitly pass the fixed `mi3001x` port profile:

```bash
bash experiments/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh 0
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

## Submit Through Embedded TP1

Each generated bundle now includes a submit helper next to `run_replay.sh`:

```bash
bash experiments/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/submit_embedded_tp1.sh
```

Equivalent raw command:

```bash
python3 servers/servers-amdhpc-embedded-TP1/launch.py submit \
  -p mi3001x \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh
```

`mi3001x` runs a single embedded service stack on profile `0`, and this
experiment writes each replay under `profile-0/`.

## Generated Files

For each QPS slug `qpsX`:

- `generated/<timestamp>/qpsX/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<selected-plan>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/submit_embedded_tp1.sh`
- `generated/<timestamp>/manifest.json`

Default replay output layout:

- `results/replay/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/profile-0/`

Each profile output also includes:

- AMD power logs under `<replay_output_dir>/power/`
- freq controller logs under `<replay_output_dir>/freq-control-linespace-amd/`
- gateway ctx-aware sampler logs under
  `<replay_output_dir>/gateway-output/job/ctx_aware_*_profile-<port_profile_id>.jsonl`

## Tests

```bash
pytest experiments/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/test -q
```
