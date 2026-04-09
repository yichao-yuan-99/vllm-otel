# Embedded AMD HPC TP1 MI3008X Sweep QPS Power Clean Ctx-Aware AMD

This workflow generates a replay bundle for
`servers/servers-amdhpc-mi3008x-embedded-TP1` with:

- `amd-power-reader`
- gateway `ctx-aware` start/end

Key behavior:

- the generated `run_replay.sh` is launched once per port profile `0..7`
- `--qps-list` supports at most `8` values
- QPS values are assigned from low profile id to high profile id
- if there are fewer than `8` QPS values, higher profile ids exit cleanly without replay work
- each assigned profile uses `gpu_index=port_profile_id`

Default thresholds:

- `ctx-aware usage_threshold_tokens = 1445793`
- `ctx-aware scheduling_threshold_tokens = 1369699`

## Requirements

```bash
python3 -m pip install -e './amd-power-reader'
```

## Generate One Sweep Bundle

```bash
python3 experiments/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/sweep-qps-docker-power-clean-ctx-aware-amd/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.1,0.2,0.3 \
  --time-constraint-s 1000 \
  --target-model qwen3_coder_30b_fp8 \
  --additional-suffix qwen3_fp8 \
  --split exclude-unranked
```

Optional overrides:

- `--port-profile`
- `--gpu-index`
- `--ctx-aware-usage-threshold-tokens`
- `--ctx-aware-scheduling-threshold-tokens`
- `--additional-suffix`

`--gpu-index` is compatibility-only and must match `--port-profile` if provided.
At runtime the script still defaults to `gpu_index=port_profile_id`.

## Run Generated Sweep Directly

```bash
bash experiments/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/sweep-qps-docker-power-clean-ctx-aware-amd/generated/<timestamp>/run_replay.sh
```

Example: profile `0` runs the first QPS, profile `1` runs the second QPS, and
profiles above the assigned range just print a skip message and exit `0`.

Useful runtime env overrides:

- `PORT_PROFILE_ID`
- `GPU_INDEX`
- `GATEWAY_BASE_URL`
- `AMD_SMI_POWER_SOCKET_PATH`
- `CTX_AWARE_USAGE_THRESHOLD_TOKENS`
- `CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS`
- `AMD_POWER_READER_BIN`
- `CURL_BIN`

## Submit Through MI3008X Embedded TP1

```bash
bash experiments/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/sweep-qps-docker-power-clean-ctx-aware-amd/generated/<timestamp>/submit_embedded_tp1.sh
```

Equivalent raw command:

```bash
python3 servers/servers-amdhpc-mi3008x-embedded-TP1/launch.py submit \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/sweep-qps-docker-power-clean-ctx-aware-amd/generated/<timestamp>/run_replay.sh
```

Default replay output layout:

- `results/replay/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/sweep-qps-docker-power-clean-ctx-aware-amd/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/profile-<assigned-profile>/`

Each profile output also includes:

- AMD power logs under `<replay_output_dir>/power/`
- gateway ctx-aware sampler logs under
  `<replay_output_dir>/gateway-output/job/ctx_aware_*_profile-<port_profile_id>.jsonl`

## Tests

```bash
pytest experiments/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/sweep-qps-docker-power-clean-ctx-aware-amd/test -q
```
