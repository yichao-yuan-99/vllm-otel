# Embedded AMD HPC TP1 MI3008X Sweep QPS Power Clean Freq Ctrl Linespace AMD Sweep SLO

This workflow generates a replay bundle for
`servers/servers-amdhpc-mi3008x-embedded-TP1` with:

- `amd-power-reader`
- gateway `ctx-aware` start/end
- gateway `slo-aware` start/end
- `freq-controller-linespace-slo-amd`
- `amd-reset-gpu-core-freq`

Key behavior:

- the generated `run_replay.sh` is launched once per port profile `0..7`
- `--qps-list` supports at most `8` values
- QPS values are assigned from low profile id to high profile id
- `--target-slo-list` does not consume extra profile slots
- each assigned profile keeps its one QPS slot and replays that slot once per SLO
  value as sequential rounds
- if there are fewer than `8` QPS values, higher profile ids exit cleanly
  without replay work
- each assigned profile uses `gpu_index=port_profile_id`

For example, with `6` QPS values and `2` SLO values, only profiles `0..5` are
used, and each of those profiles runs two rounds for its assigned QPS.

Default thresholds and policies:

- `ctx-aware usage_threshold_tokens = 1445793`
- `ctx-aware scheduling_threshold_tokens = 1369699`
- `ctx-aware policy_mode = "throughput"`
- `slo-aware policy_mode = "push-back-half-slack"`
- `freq-controller-linespace-slo-amd threshold = 1141416`

## Requirements

Run the server with `gateway_ctx` enabled so both `/ctx-aware/*` and
`/slo-aware/*` endpoints are available on each profile gateway.

```bash
python3 -m pip install -e './amd-power-reader'
python3 -m pip install -e './freq-controller-linespace-slo-amd'
```

## Generate One Sweep Bundle

```bash
python3 experiments/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.25,0.3,0.35 \
  --target-slo-list 8,10 \
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

`--gpu-index` is compatibility-only and must match `--port-profile` if
provided. At runtime the script still defaults to `gpu_index=port_profile_id`.

## Run Generated Sweep Directly

```bash
bash experiments/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo/generated/<timestamp>/run_replay.sh
```

Useful runtime env overrides:

- `PORT_PROFILE_ID`
- `GPU_INDEX`
- `GATEWAY_BASE_URL`
- `AMD_SMI_POWER_SOCKET_PATH`
- `CTX_AWARE_USAGE_THRESHOLD_TOKENS`
- `CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS`
- `FREQ_CONTROLLER_THRESHOLD`
- `FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH`
- `FREQ_CONTROLLER_CONFIG`
- `AMD_POWER_READER_BIN`
- `FREQ_CONTROLLER_BIN`
- `RESET_GPU_CORE_FREQ_BIN`
- `CURL_BIN`
- `PYTHON_BIN`

If you do not override `GATEWAY_BASE_URL`, the generated script resolves the
gateway port from `configs/port_profiles.toml`.

If you do not override `FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH`, the generated
script defaults it to
`/tmp/vllm-gateway-ctx-profile-<port_profile_id>.sock`.

If you do not override the tool binary env vars, the generated script resolves
the AMD helper binaries from `<repo>/.venv/bin/` and pins the default
`PYTHON_BIN` and `CURL_BIN` to absolute paths discovered at generation time.

## Submit Through MI3008X Embedded TP1

```bash
bash experiments/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo/generated/<timestamp>/submit_embedded_tp1.sh
```

Equivalent raw command:

```bash
python3 servers/servers-amdhpc-mi3008x-embedded-TP1/launch.py submit \
  -m qwen3_coder_30b_fp8 \
  -e ./experiments/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo/generated/<timestamp>/run_replay.sh
```

Default replay output layout:

- `results/replay/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo<suffix>/<dataset>/<agent>/split/<split>/<qps>/<slo>/<timestamp>/profile-<assigned-profile>/`

Each profile output also includes:

- AMD power logs under `<replay_output_dir>/power/`
- freq controller logs under `<replay_output_dir>/freq-control-linespace/`
- gateway ctx-aware sampler logs under
  `<replay_output_dir>/gateway-output/job/ctx_aware_*_profile-<port_profile_id>.jsonl`
- gateway SLO-aware decision logs under
  `<replay_output_dir>/gateway-output/job/slo_aware_decisions_*.jsonl`

## Tests

```bash
pytest experiments/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo/test -q
```
