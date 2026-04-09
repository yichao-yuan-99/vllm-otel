# Embedded AMD HPC TP1 MI3001X Sweep QPS Power Clean AMD

This workflow generates a replay bundle for
`servers/servers-amdhpc-mi3001x-embedded-TP1`.

Key behavior:

- the generated `run_replay.sh` is launched once on one `mi3001x` node
- all QPS points run sequentially in that single process
- the runtime port profile is fixed to `0`
- the runtime GPU index is fixed to `0`
- outputs are written under `profile-0/`
- power logging uses `amd-power-reader`

## Requirements

```bash
python3 -m pip install -e './amd-power-reader'
```

## Generate One Sweep Bundle

```bash
python3 experiments/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-amd/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.1,0.2,0.3 \
  --time-constraint-s 1000 \
  --target-model qwen3_coder_30b \
  --split exclude-unranked
```

Optional overrides:

- `--additional-suffix`

This `mi3001x` workflow assumes `port_profile_id=0` and `gpu_index=0`.
The compatibility flags `--port-profile` and `--gpu-index` still exist, but
they must remain `0`.

## Run Generated Sweep Directly

```bash
bash experiments/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-amd/generated/<timestamp>/run_replay.sh
```

Explicitly pass the fixed `mi3001x` port profile:

```bash
bash experiments/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-amd/generated/<timestamp>/run_replay.sh 0
```

Useful runtime env overrides:

- `PORT_PROFILE_ID` (`0` only)
- `GPU_INDEX` (`0` only)
- `AMD_SMI_POWER_SOCKET_PATH`
- `AMD_POWER_READER_BIN`

## Submit Through MI3001X Embedded TP1

```bash
bash experiments/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-amd/generated/<timestamp>/submit_embedded_tp1.sh
```

Equivalent raw command:

```bash
python3 servers/servers-amdhpc-mi3001x-embedded-TP1/launch.py submit \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-amd/generated/<timestamp>/run_replay.sh
```

Default replay output layout:

- `results/replay/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-amd/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/profile-0/`

Each profile output also includes:

- AMD power logs under `<replay_output_dir>/power/`

## Tests

```bash
pytest experiments/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-amd/test -q
```
