# Sweep QPS Docker Power Clean Freq Ctrl Linespace Sweep SLO OG 80P

This workflow is the gateway-only SLO-aware variant of
`experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo-dual`.

It keeps the same clean split-plan lookup, QPS/SLO sweep shape, Zeus power
logging, and gateway control flow, but it uses plain
`freq-controller-linespace` like
`experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace`.

For every `(qps, target_slo)` replay job:

- gateway ctx-aware mode starts first with `policy_mode = "throughput"`
- gateway SLO-aware mode then starts with `target_tokens_per_s = <target_slo>`
  and `policy_mode = "push-back-80p-slack"`
- `freq-controller-linespace` starts independently with its usual arguments:
  `--log-dir`, `--threshold`, `--port-profile-id`, and `--gpu-index`
- the controller does not receive `--throughput-target`
- the controller does not receive `--gateway-ipc-socket-path`; it auto-detects
  the matching per-profile `gateway_ctx` IPC socket when available
- after the replay finishes, the runner stops the controller and calls
  `POST /slo-aware/end` and `POST /ctx-aware/end`

`POST /slo-aware/start` requires ctx-aware mode to already be enabled, so the
generated runner always enables ctx-aware first.

Because `freq-controller-linespace` waits for gateway `job_active=true`, it
can still be started before replay and remain pending until the replay job
actually begins.

Default thresholds and policies:

- `ctx-aware usage_threshold_tokens = 501280`
- `ctx-aware scheduling_threshold_tokens = 474897`
- `ctx-aware policy_mode = "throughput"`
- `slo-aware policy_mode = "push-back-80p-slack"`
- `freq-controller-linespace threshold = 395784`

## Requirements

Run the Docker environment with `gateway_ctx`, for example:

```bash
python3 servers/servers-docker/client.py start \
  -m qwen3_coder_30b_fp8 \
  -p 13 \
  -l h100_nvl_gpu3_single \
  --gateway-ctx \
  -b
```

Install the local console scripts used by the generated runner:

```bash
python3 -m pip install -e './freq-controller-linespace'
python3 -m pip install -e './zeus-power-reader'
```

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo-og-80p/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.02,0.04,0.08 \
  --target-slo-list 8,10,12 \
  --time-constraint-s 10800 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 2 \
  --split exclude-unranked \
  --additional-suffix qwen3_fp8 \
  --gpu-id 2
```

Optional overrides:

- `--split-two-group-metric context_usage`
- `--ctx-aware-usage-threshold-tokens`
- `--ctx-aware-scheduling-threshold-tokens`
- `--freq-controller-threshold`

`--split` values are `top`, `rest`, or `exclude-unranked`. `full` is accepted
as a compatibility alias for `exclude-unranked`.

## Run Generated Sweep

```bash
bash experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo-og-80p/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo-og-80p/generated/<timestamp>/run_replay.sh 3
```

Optional runtime env overrides:

- `GATEWAY_BASE_URL=http://127.0.0.1:<gateway_port>`
- `CTX_AWARE_USAGE_THRESHOLD_TOKENS=...`
- `CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS=...`
- `FREQ_CONTROLLER_CONFIG=/path/to/freq-controller-linespace.toml`
- `FREQ_CONTROLLER_THRESHOLD=...`
- `FREQ_CONTROLLER_BIN=freq-controller-linespace`
- `ZEUS_POWER_READER_BIN=zeus-power-reader`
- `RESET_GPU_CORE_FREQ_BIN=reset-gpu-core-freq`
- `CURL_BIN=curl`
- `PYTHON_BIN=python3`
- `ZEUSD_SOCKET_PATH=/var/run/zeusd.sock`

If `GATEWAY_BASE_URL` is not set, the runner resolves `gateway_port` from
`configs/port_profiles.toml` using the selected port profile ID.

## Generated Files

For each `(qps, target_slo)` pair:

- `generated/<timestamp>/<qps-slug>/<slo-slug>/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<selected-plan>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

Default replay output layout:

- `results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo-og-80p/<dataset>/<agent>/split/<split>/<qps>/<slo>/<timestamp>/`

Each replay output directory includes:

- Zeus power logs under `power/`
- `freq-controller-linespace` logs under `freq-control-linespace/`
- gateway ctx-aware sampler logs under `job/`, including additive SLO-aware
  fields such as `slo_aware_enabled`, `slo_target_tokens_per_s`, and
  `ralexation_agent_count`
- gateway SLO-aware event logs under
  `gateway-output/job/slo_aware_decisions_<job_started_at>.jsonl`

## Post-Process

The shared post-process pipeline handles both the controller logs and the
gateway SLO-aware decision logs:

```bash
python3 post-process/run_all.py \
  --root-dir results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo-og-80p
```

## Tests

```bash
.venv/bin/pytest experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo-og-80p/test -q
```
