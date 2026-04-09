# Embedded AMD HPC TP1 MI3001X Sweep QPS Power Clean Freq Ctrl Linespace AMD

This workflow generates a replay bundle for
`servers/servers-amdhpc-mi3001x-embedded-TP1` with:

- `amd-power-reader`
- gateway `ctx-aware` start/end
- `freq-controller-linespace-amd`

Key behavior:

- the generated `run_replay.sh` is launched once on one `mi3001x` node
- all QPS points run sequentially in that single process
- the runtime port profile is fixed to `0`
- the runtime GPU index is fixed to `0`
- outputs are written under `profile-0/`

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

Non-FP8 example that works with the clean plans already present under
`results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/`:

```bash
python3 experiments/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.1 \
  --time-constraint-s 3600 \
  --target-model qwen3_coder_30b \
  --split exclude-unranked
```

FP8 example with the same source run and shape as the non-FP8 example.
This requires suffixed clean plans such as
`replay-plan.clean.token.exclude-unranked.qwen3_fp8.json`:

```bash
python3 experiments/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.1 \
  --time-constraint-s 3600 \
  --target-model qwen3_coder_30b_fp8 \
  --split exclude-unranked \
  --additional-suffix qwen3_fp8
```

Optional overrides:

- `--ctx-aware-usage-threshold-tokens`
- `--ctx-aware-scheduling-threshold-tokens`
- `--freq-controller-threshold`
- `--additional-suffix`
- `--output-suffix`

Use `--additional-suffix qwen3_fp8` when the source run only has suffixed
clean plans, for example `replay-plan.clean.token.exclude-unranked.qwen3_fp8.json`.

This `mi3001x` workflow assumes `port_profile_id=0` and `gpu_index=0`.
The compatibility flags `--port-profile` and `--gpu-index` still exist, but
they must remain `0`.

## Run Generated Sweep Directly

```bash
bash experiments/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh
```

Explicitly pass the fixed `mi3001x` port profile:

```bash
bash experiments/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh 0
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

## Submit Through MI3001X Embedded TP1

```bash
bash experiments/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/submit_embedded_tp1.sh
```

Equivalent raw command:

```bash
python3 servers/servers-amdhpc-mi3001x-embedded-TP1/launch.py submit \
  -m <target-model> \
  -e ./experiments/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh
```

`<target-model>` must match the `--target-model` used when generating the bundle.

Default replay output layout:

- `results/replay/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/profile-0/`

Each profile output also includes:

- AMD power logs under `<replay_output_dir>/power/`
- freq controller logs under `<replay_output_dir>/freq-control-linespace-amd/`
- gateway ctx-aware sampler logs under
  `<replay_output_dir>/gateway-output/job/ctx_aware_*_profile-<port_profile_id>.jsonl`

## Tests

```bash
pytest experiments/amd-embeded/servers-amdhpc-mi3001x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/test -q
```
