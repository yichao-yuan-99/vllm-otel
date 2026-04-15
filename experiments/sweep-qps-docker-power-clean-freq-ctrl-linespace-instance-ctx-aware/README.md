# Sweep QPS Docker Power Clean Freq Ctrl Linespace Instance Ctx Aware

This workflow is the `freq-controller-linespace-instance-ctx-aware` variant of
`experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance`.

It keeps the same clean split-plan lookup, Zeus power logging, and
`freq-controller-linespace-instance` control flow, but the generated runner
also enables gateway ctx-aware mode before each replay and disables it again
after that replay finishes.

Compared to `sweep-qps-docker-power-clean-freq-ctrl-linespace-instance`:

- each QPS run calls `POST /ctx-aware/start` before replay
- each QPS run calls `POST /ctx-aware/end` after replay
- the generated runner resolves `GATEWAY_BASE_URL` from
  `configs/port_profiles.toml` when not provided explicitly

Default ctx-aware thresholds:

- `usage_threshold_tokens = 501280`
- `scheduling_threshold_tokens = 474897`

Default controller threshold:

- `threshold = 395784`

The underlying replay launch pattern is still Poisson, so `--qps-list`
represents Poisson request rates and requires `--poisson-seed`.

Because `freq-controller-linespace-instance` waits for gateway
`job_active=true`, it can be started before replay and remain pending until the
replay job actually begins.

Default replay output layout:

- default threshold (`395784`):
  `results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/`
- non-default threshold:
  `results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware-<threshold>/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/`

Inside each replay output directory, the generated runner writes:

- power logs under `power/`
- `freq-controller-linespace-instance` query and decision logs under
  `freq-control-linespace-instance/` using the
  `freq-controller-ls-instance.*.jsonl` filename prefix

Plan lookup always uses clean split plans:

- `replay-plan.clean.<metric>.top(.<suffix>).json`
- `replay-plan.clean.<metric>.rest(.<suffix>).json`
- `replay-plan.clean.<metric>.exclude-unranked(.<suffix>).json`

## Requirements

Run the Docker environment with `gateway_ctx`, for example:

```bash
python3 servers/servers-docker/client.py start \
  -m qwen3_coder_30b_fp8 \
  -p 2 \
  -l h100_nvl_gpu3_single \
  --gateway-ctx \
  -b
```

Install the local console scripts used by the generated run script:

- `freq-controller-linespace-instance`
- `zeus-power-reader`
- `reset-gpu-core-freq`

```bash
python3 -m pip install -e './freq-controller-linespace-instance'
python3 -m pip install -e './zeus-power-reader'
```

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.02,0.04,0.08 \
  --time-constraint-s 10800 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 2 \
  --split exclude-unranked \
  --additional-suffix qwen3_fp8 \
  --gpu-id 2
```

Optional flags:

- `--split-two-group-metric context_usage`
- `--ctx-aware-usage-threshold-tokens 501280`
- `--ctx-aware-scheduling-threshold-tokens 474897`
- `--freq-controller-threshold 395784`

If `--freq-controller-threshold` is non-default and you do not pass
`--replay-output-root`, the generated replay output root is automatically
suffixed with that threshold value.

`--split` values are `top`, `rest`, or `exclude-unranked`. `full` is accepted
as a compatibility alias for `exclude-unranked`.

## Run Generated Sweep

```bash
bash experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware/generated/<timestamp>/run_replay.sh 3
```

Optional runtime overrides:

- `GATEWAY_BASE_URL=http://127.0.0.1:<gateway_port>`
- `CTX_AWARE_USAGE_THRESHOLD_TOKENS=...`
- `CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS=...`
- `FREQ_CONTROLLER_CONFIG=/path/to/freq-controller-linespace-instance.toml`
- `FREQ_CONTROLLER_THRESHOLD=...`
- `FREQ_CONTROLLER_BIN=freq-controller-linespace-instance`
- `ZEUS_POWER_READER_BIN=zeus-power-reader`
- `ZEUSD_SOCKET_PATH=/var/run/zeusd.sock`
- `CURL_BIN=curl`

If `GATEWAY_BASE_URL` is not set, the runner resolves `gateway_port` from
`configs/port_profiles.toml` using the selected port profile ID.

Use `FREQ_CONTROLLER_CONFIG` when you want to override linespace-instance
policy fields such as the frequency list or threshold.

## Generated Files

For each QPS slug `qpsX`:

- `generated/<timestamp>/qpsX/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<selected-plan>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

## Post-Process

After runs finish, the shared post-process pipeline can still extract and
visualize the linespace-instance controller logs through the
`freq-control-linespace-instance` flow.

```bash
python3 post-process/run_all.py \
  --root-dir results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware
```

For non-default controller thresholds, point `--root-dir` at the matching
suffixed replay root, for example:

```bash
python3 post-process/run_all.py \
  --root-dir results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware-200
```

## Tests

```bash
.venv/bin/pytest experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-ctx-aware/test -q
```
