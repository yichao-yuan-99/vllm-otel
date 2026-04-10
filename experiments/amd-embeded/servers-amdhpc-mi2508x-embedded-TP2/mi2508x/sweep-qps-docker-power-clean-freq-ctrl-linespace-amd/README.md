# Embedded AMD HPC TP2 MI2508X Sweep QPS Power Clean Freq Ctrl Linespace AMD

This workflow generates a replay bundle for
`servers/servers-amdhpc-mi2508x-embedded-TP2` with:

- `amd-power-reader`
- gateway `ctx-aware` start/end
- `freq-controller-linespace-amd`
- `amd-reset-gpu-core-freq`

Key behavior:

- the generated `run_replay.sh` is launched once per port profile `0..3`
- `--qps-list` supports at most `4` values
- QPS values are assigned from low profile id to high profile id
- if there are fewer than `4` QPS values, higher profile ids exit cleanly without replay work
- each assigned profile uses a fixed GPU pair: `0->0,1`, `1->2,3`, `2->4,5`, `3->6,7`

Default thresholds:

- `ctx-aware usage_threshold_tokens = 1445793`
- `ctx-aware scheduling_threshold_tokens = 1369699`
- `freq-controller-linespace-amd threshold = 1141416`

For MI2508X TP2 runs, prefer the model-specific thresholds recorded in
`servers/servers-amdhpc-interactive-mi2508x-embedded-TP2/README.token-limits.md`:

| Target model | Total tokens | `ctx-aware usage_threshold_tokens` | `ctx-aware scheduling_threshold_tokens` | `freq-controller-linespace-amd threshold` |
| --- | ---: | ---: | ---: | ---: |
| `qwen3_coder_30b_fp8` | `901840` | `856748` | `811656` | `676380` |
| `qwen3_coder_30b` | `569040` | `540588` | `512136` | `426780` |

## Requirements

```bash
python3 -m pip install -e './amd-power-reader'
python3 -m pip install -e './freq-controller-linespace-amd'
```

## Generate One Sweep Bundle

```bash
python3 experiments/amd-embeded/servers-amdhpc-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.1,0.2 \
  --time-constraint-s 1000 \
  --target-model qwen3_coder_30b_fp8 \
  --ctx-aware-usage-threshold-tokens 856748 \
  --ctx-aware-scheduling-threshold-tokens 811656 \
  --freq-controller-threshold 676380 \
  --additional-suffix qwen3_fp8 \
  --split exclude-unranked
```

Example for `qwen3_coder_30b`:

```bash
python3 experiments/amd-embeded/servers-amdhpc-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.1,0.2 \
  --time-constraint-s 1000 \
  --target-model qwen3_coder_30b \
  --ctx-aware-usage-threshold-tokens 540588 \
  --ctx-aware-scheduling-threshold-tokens 512136 \
  --freq-controller-threshold 426780 \
  --split exclude-unranked
```

Optional overrides:

- `--port-profile`
- `--gpu-index`
- `--ctx-aware-usage-threshold-tokens`
- `--ctx-aware-scheduling-threshold-tokens`
- `--freq-controller-threshold`
- `--additional-suffix`
- `--output-suffix`

## Run Generated Sweep Directly

```bash
bash experiments/amd-embeded/servers-amdhpc-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh
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

## Submit Through MI2508X Embedded TP2

```bash
bash experiments/amd-embeded/servers-amdhpc-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/submit_embedded_tp2.sh
```

Equivalent raw command:

```bash
python3 servers/servers-amdhpc-mi2508x-embedded-TP2/launch.py submit \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/amd-embeded/servers-amdhpc-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh
```

Default replay output layout:

- `results/replay/amd-embeded/servers-amdhpc-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd<suffix>/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/profile-<assigned-profile>/`

Each profile output also includes:

- AMD power logs under `<replay_output_dir>/power/`
- freq controller logs under `<replay_output_dir>/freq-control-linespace-amd/`
- gateway ctx-aware sampler logs under
  `<replay_output_dir>/gateway-output/job/ctx_aware_*_profile-<port_profile_id>.jsonl`

## Tests

```bash
pytest experiments/amd-embeded/servers-amdhpc-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/test -q
```
