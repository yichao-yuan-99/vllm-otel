# Embedded AMD HPC TP2 MI2508X Sweep QPS Power Clean AMD

This workflow generates a replay bundle for
`servers/servers-amdhpc-mi2508x-embedded-TP2`.

Key behavior:

- the generated `run_replay.sh` is launched once per port profile `0..3`
- `--qps-list` supports at most `4` values
- QPS values are assigned from low profile id to high profile id
- if there are fewer than `4` QPS values, higher profile ids exit cleanly without replay work
- each assigned profile uses a fixed GPU pair: `0->0,1`, `1->2,3`, `2->4,5`, `3->6,7`
- AMD power logging uses `amd-power-reader`

## Requirements

```bash
python3 -m pip install -e './amd-power-reader'
```

## Generate One Sweep Bundle

```bash
python3 experiments/amd-embeded/servers-amdhpc-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-amd/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.1,0.2 \
  --time-constraint-s 1000 \
  --target-model qwen3_coder_30b_fp8 \
  --additional-suffix qwen3_fp8 \
  --split exclude-unranked
```

Optional overrides:

- `--port-profile` sets the default profile used when `run_replay.sh` is started without an explicit argument
- `--gpu-index` is compatibility-only and must match the default GPU pair for `--port-profile` if provided
- `--additional-suffix`

## Run Generated Sweep Directly

```bash
bash experiments/amd-embeded/servers-amdhpc-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-amd/generated/<timestamp>/run_replay.sh
```

Example: profile `0` runs the first QPS, profile `1` runs the second QPS, and
profiles above the assigned range just print a skip message and exit `0`.

Useful runtime env overrides:

- `PORT_PROFILE_ID`
- `GPU_INDEX`
- `AMD_SMI_POWER_SOCKET_PATH`
- `AMD_POWER_READER_BIN`

## Submit Through MI2508X Embedded TP2

```bash
bash experiments/amd-embeded/servers-amdhpc-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-amd/generated/<timestamp>/submit_embedded_tp2.sh
```

Equivalent raw command:

```bash
python3 servers/servers-amdhpc-mi2508x-embedded-TP2/launch.py submit \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/amd-embeded/servers-amdhpc-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-amd/generated/<timestamp>/run_replay.sh
```

Default replay output layout:

- `results/replay/amd-embeded/servers-amdhpc-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-amd/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/profile-<assigned-profile>/`

Each profile output also includes:

- AMD power logs under `<replay_output_dir>/power/`

## Tests

```bash
pytest experiments/amd-embeded/servers-amdhpc-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-amd/test -q
```
