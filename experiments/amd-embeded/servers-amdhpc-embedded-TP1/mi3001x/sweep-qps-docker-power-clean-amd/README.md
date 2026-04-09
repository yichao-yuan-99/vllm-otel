# Embedded AMD HPC TP1 MI3001X Sweep QPS Power Clean AMD

This workflow is the single-node `mi3001x` embedded-TP1 AMD variant of
`experiments/sweep-qps-docker-power-clean`.

It keeps the same clean split-plan lookup and Poisson replay generation, but
adapts the runtime wrapper to `servers/servers-amdhpc-embedded-TP1` on
`mi3001x`:

- the generated `run_replay.sh` is invoked once on a single `mi3001x` node
- the runtime port profile is fixed to `0`
- the runtime GPU index is fixed to `0`
- outputs are written under `profile-0/` for consistency with the embedded TP1
  directory layout
- power logging uses `amd-power-reader`

If launched through
`servers/servers-amdhpc-embedded-TP1/launch.py`, the job-level sbatch already
starts `amd-smi-power-daemon` and exports `AMD_SMI_POWER_SOCKET_PATH` for the
experiment script. For direct runs, start the daemon yourself first.

## Requirements

Install the local console scripts used by the generated runner:

```bash
python3 -m pip install -e './amd-power-reader'
```

For direct runs outside embedded TP1, start the daemon first:

```bash
amd-smi-power-daemon &
export AMD_SMI_POWER_SOCKET_PATH=/tmp/amdsmi-power-reader.sock
```

## Generate One Sweep Bundle

```bash
python3 experiments/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-amd/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.1 \
  --time-constraint-s 1000 \
  --target-model qwen3_coder_30b_fp8 \
  --additional-suffix qwen3_fp8
  --split exclude-unranked
```

Optional overrides:

- `--additional-suffix`

This `mi3001x` variant assumes `port_profile_id=0` and `gpu_index=0`.
The compatibility flags `--port-profile` and `--gpu-index` still exist, but
they must remain `0`.

## Run Generated Sweep Directly

```bash
bash experiments/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-amd/generated/<timestamp>/run_replay.sh
```

Explicitly pass the fixed `mi3001x` port profile:

```bash
bash experiments/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-amd/generated/<timestamp>/run_replay.sh 0
```

Useful runtime env overrides:

- `PORT_PROFILE_ID` (`0` only)
- `GPU_INDEX` (`0` only)
- `AMD_SMI_POWER_SOCKET_PATH`
- `AMD_POWER_READER_BIN`

## Submit Through Embedded TP1

Each generated bundle now includes a submit helper next to `run_replay.sh`:

```bash
bash experiments/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-amd/generated/<timestamp>/submit_embedded_tp1.sh
```

Equivalent raw command:

```bash
python3 servers/servers-amdhpc-embedded-TP1/launch.py submit \
  -p mi3001x \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-amd/generated/<timestamp>/run_replay.sh
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

- `results/replay/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-amd/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/profile-0/`

Each profile output also includes:

- AMD power logs under `<replay_output_dir>/power/`

## Tests

```bash
pytest experiments/amd-embeded/servers-amdhpc-embedded-TP1/mi3001x/sweep-qps-docker-power-clean-amd/test -q
```
