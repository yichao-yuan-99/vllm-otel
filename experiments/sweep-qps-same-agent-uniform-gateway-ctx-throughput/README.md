# Sweep QPS Same Agent Uniform Gateway Ctx Throughput

This workflow is the `gateway_ctx` variant of
`experiments/sweep-qps-same-agent-uniform/`.

It takes one replay plan that was already compiled with
`replayer compile --single-trail`, then expanded with
`derived-single-plan/generate_derived_single_plan.py`, and generates a local
experiment bundle that sweeps QPS while replaying that trail with Zeus power
logging.

Like the non-ctx variant, this bundle uses a `uniform` replay launch pattern,
which in this repository means a fixed inter-launch interval of exactly
`1 / qps`.

The generated `run_replay.sh` also enables gateway ctx-aware mode before each
replay job and disables it again after that replay finishes.

Unlike `experiments/sweep-qps-same-agent-uniform-gateway-ctx/`, this variant
always starts ctx-aware mode with `policy_mode = "throughput"`.

Default ctx-aware thresholds:

- `usage_threshold_tokens = 501280`
- `scheduling_threshold_tokens = 474897`
- `policy_mode = "throughput"`

For each QPS point in `--qps-list`, the generated `run_replay.sh` does:

1. starts `zeus-power-reader`
2. enables gateway ctx-aware mode
3. runs replay once
4. disables gateway ctx-aware mode
5. stops power reader

Power logs are written to a `power/` subdirectory inside each replay output
directory.

Default replay output layout:

- `results/replay/sweep-qps-same-agent-uniform-gateway-ctx-throughput/<dataset-lineage>/trail/<safe-source-trail>/<qps>/<timestamp>/`

`<dataset-lineage>` is inferred from `source_job_dir` inside the replay plan
when possible. If the lineage cannot be inferred, the output starts at
`trail/<safe-source-trail>/...`.

Pass `--output-suffix <suffix>` to switch the top-level directory to
`trail-<safe-suffix>/...`, for example `--output-suffix lmcache` writes under
`trail-lmcache/<safe-source-trail>/...`.

## Requirements

Run a `gateway_ctx` instance that matches the selected port profile. One
possible setup is:

```bash
python3 servers/servers-docker/client.py start \
  -m qwen3_coder_30b_fp8 \
  -p 13 \
  -l h100_nvl_gpu3_single \
  --gateway-ctx \
  -b
```

Install local Zeus tools:

```bash
pip install -e ./zeus-power-reader
```

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-same-agent-uniform-gateway-ctx-throughput/generate_experiment.py \
  --source-plan results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/seed-7-size-500.replay-plan.trail-profile-1_run_20260306T064004Z_02c951cc220f_9e28187e7c8c998594c2be521322d0b4.qwen3_fp8_10th.json \
  --randomize-seed 11 \
  --qps-list 0.25,0.3,0.35 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 13 \
  --power-gpu-indices 3
```

Optional overrides:

- `--power-gpu-indices 0,1`
- `--output-suffix lmcache`
- `--ctx-aware-usage-threshold-tokens`
- `--ctx-aware-scheduling-threshold-tokens`

## Compile Then Derive The Plan First

One common flow is:

1. pick a `source_trail_name`, for example from
   `original-analysis/percentiles/context-usage-percentiles.json`
2. compile that trail into its own plan
3. derive a multi-worker plan from that single-trail plan
4. generate the QPS sweep from the derived plan

Example:

```bash
RUN_DIR="results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z"
PERCENTILES_JSON="$RUN_DIR/original-analysis/percentiles/context-usage-percentiles.json"
TRAIL_NAME="$(jq -r '.source_trail_names_by_percentile["50"]' "$PERCENTILES_JSON")"

python3 -m replayer compile \
  --job-dir "$RUN_DIR" \
  --model qwen3_coder_30b_fp8 \
  --single-trail "$TRAIL_NAME" \
  --plan-out "$RUN_DIR/replay-plan.trail-50th.json"

python3 derived-single-plan/generate_derived_single_plan.py \
  --plan "$RUN_DIR/replay-plan.trail-50th.json" \
  --seed 7 \
  --size 500 \
  --port-profile 13

python3 experiments/sweep-qps-same-agent-uniform-gateway-ctx-throughput/generate_experiment.py \
  --source-plan "$RUN_DIR/seed-7-size-500.replay-plan.trail-50th.json" \
  --randomize-seed 11 \
  --qps-list 0.25,0.3,0.35 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 13 \
  --power-gpu-indices 3
```

## Run The Generated Sweep

```bash
bash experiments/sweep-qps-same-agent-uniform-gateway-ctx-throughput/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-qps-same-agent-uniform-gateway-ctx-throughput/generated/<timestamp>/run_replay.sh 3
```

Optional runtime env overrides for the generated runner:

- `GATEWAY_BASE_URL=http://127.0.0.1:<gateway_port>`
- `CTX_AWARE_USAGE_THRESHOLD_TOKENS=...`
- `CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS=...`
- `CURL_BIN=curl`
- `PYTHON_BIN=python3`

If `GATEWAY_BASE_URL` is not set, the runner resolves `gateway_port` from
`configs/port_profiles.toml` using the selected port profile ID.

## Generated Files

For each QPS slug `qpsX`:

- `generated/<timestamp>/qpsX/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<source-plan-name>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`
