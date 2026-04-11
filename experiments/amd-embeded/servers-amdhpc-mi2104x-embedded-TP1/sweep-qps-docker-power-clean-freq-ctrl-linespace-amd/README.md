# Embedded AMD HPC TP1 MI2104X Sweep QPS Power Clean Freq Ctrl Linespace AMD

This workflow generates a replay bundle for
`servers/servers-amdhpc-mi2104x-embedded-TP1` with:

- `amd-power-reader`
- gateway `ctx-aware` start/end
- `freq-controller-linespace-amd`
- `amd-reset-gpu-core-freq`

Key behavior:

- the generated `run_replay.sh` is launched once per port profile `0..3`
- `--qps-list` supports at most `4` values
- QPS values are assigned from low profile id to high profile id
- if there are fewer than `4` QPS values, higher profile ids exit cleanly without replay work
- each assigned profile uses `gpu_index=port_profile_id`

Default thresholds:

- `ctx-aware usage_threshold_tokens = 1445793`
- `ctx-aware scheduling_threshold_tokens = 1369699`
- `freq-controller-linespace-amd threshold = 1141416`

## Requirements

```bash
python3 -m pip install -e './amd-power-reader'
python3 -m pip install -e './freq-controller-linespace-amd'
```

## Generate One Sweep Bundle

```bash
python3 experiments/amd-embeded/servers-amdhpc-mi2104x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.25,0.3,0.35 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --additional-suffix qwen3_fp8 \
  --split rest
```

Optional overrides:

- `--port-profile`
- `--gpu-index`
- `--ctx-aware-usage-threshold-tokens`
- `--ctx-aware-scheduling-threshold-tokens`
- `--freq-controller-threshold`
- `--additional-suffix`
- `--output-suffix`

`--gpu-index` is compatibility-only and must match `--port-profile` if provided.
At runtime the script still defaults to `gpu_index=port_profile_id`.

## Run Generated Sweep Directly

```bash
bash experiments/amd-embeded/servers-amdhpc-mi2104x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh
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
- `FREQ_CONTROLLER_CONFIG`
- `FREQ_CONTROLLER_THRESHOLD`
- `AMD_POWER_READER_BIN`
- `FREQ_CONTROLLER_BIN`
- `RESET_GPU_CORE_FREQ_BIN`

If you do not override those env vars, the generated script resolves the three
tool binaries from `<repo>/.venv/bin/`.

## Submit Through MI2104X Embedded TP1

```bash
bash experiments/amd-embeded/servers-amdhpc-mi2104x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/submit_embedded_tp1.sh
```

Equivalent raw command:

```bash
python3 servers/servers-amdhpc-mi2104x-embedded-TP1/launch.py submit \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/amd-embeded/servers-amdhpc-mi2104x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh
```

Default replay output layout:

- `results/replay/amd-embeded/servers-amdhpc-mi2104x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd<suffix>/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/profile-<assigned-profile>/`

Each profile output also includes:

- AMD power logs under `<replay_output_dir>/power/`
- freq controller logs under `<replay_output_dir>/freq-control-linespace-amd/`
- gateway ctx-aware sampler logs under
  `<replay_output_dir>/gateway-output/job/ctx_aware_*_profile-<port_profile_id>.jsonl`

## Tests

```bash
pytest experiments/amd-embeded/servers-amdhpc-mi2104x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/test -q
```
