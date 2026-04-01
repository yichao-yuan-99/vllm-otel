# Sweep Freq Same Agent

This workflow takes one replay plan that was already compiled with
`replayer compile --single-trail`, then expanded with
`derived-single-plan/generate_derived_single_plan.py`, and generates a local
experiment bundle that sweeps GPU core frequency while replaying that exact
trail at one fixed launch target and time window.

Supported launch modes:

- fixed QPS via `--qps` plus `--poisson-seed`
- fixed max concurrency via `--max-concurrent`

Unlike the split-based experiment generators, this script does not look up
plans from `--source-run-dir`. You pass the exact derived plan with
`--source-plan`. Original non-derived single-trail plans are rejected.

For each frequency range in `--freq-list`, the generated `run_replay.sh` does:

1. `set-gpu-core-freq` for `--gpu-id`
2. starts `zeus-power-reader`
3. runs replay once
4. stops power reader
5. `reset-gpu-core-freq`

Power logs are written to a `power/` subdirectory inside each replay output
directory.

Default replay output layout:

- `results/replay/sweep-freq-same-agent/<dataset-lineage>/trail/<safe-source-trail>/<launch-target>/<timestamp>/<freq-slug>/`

Examples:

- QPS mode: `<launch-target> = qps0_2`
- max-concurrency mode: `<launch-target> = c8`

`<dataset-lineage>` is inferred from `source_job_dir` inside the replay plan
when possible. If the lineage cannot be inferred, the output starts at
`trail/<safe-source-trail>/...`.

Example `freq-slug`: `core-345-1305`.

## Generate One Sweep Bundle

QPS / Poisson mode:

```bash
python3 experiments/sweep-freq-same-agent/generate_experiment.py \
  --source-plan results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/seed-7-size-500.replay-plan.trail-profile-1_run_20260306T064004Z_02c951cc220f_9e28187e7c8c998594c2be521322d0b4.qwen3_fp8_10th.json \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps 0.2 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 0 \
  --freq-list \
  "345:345;345:510;345:660;345:810;345:915;345:1005;345:1185;345:1305;345:1410;345:1530;345:1680" \
  --gpu-id 0
```

Max-concurrency mode:

```bash
python3 experiments/sweep-freq-same-agent/generate_experiment.py \
  --source-plan results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/seed-7-size-500.replay-plan.trail-profile-1_run_20260306T064004Z_02c951cc220f_9e28187e7c8c998594c2be521322d0b4.qwen3_fp8_10th.json \
  --randomize-seed 11 \
  --max-concurrent 8 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 0 \
  --freq-list \
  "345:345;345:510;345:660;345:810;345:915;345:1005;345:1185;345:1305;345:1410;345:1530;345:1680" \
  --gpu-id 0
```

`--freq-list` is a semicolon-separated list of `min_mhz:max_mhz`.
Exactly one of `--qps` or `--max-concurrent` is required.
`--poisson-seed` is required with `--qps` and rejected with `--max-concurrent`.

## Compile Then Derive The Plan First

One common flow is:

1. pick a `source_trail_name`, for example from
   `original-analysis/percentiles/context-usage-percentiles.json`
2. compile that trail into its own plan
3. derive a multi-worker plan from that single-trail plan
4. generate the frequency sweep from that derived plan

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
  --port-profile 0

python3 experiments/sweep-freq-same-agent/generate_experiment.py \
  --source-plan "$RUN_DIR/seed-7-size-500.replay-plan.trail-50th.json" \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps 0.2 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 0 \
  --freq-list \
  "345:345;345:510;345:660;345:810;345:915;345:1005;345:1185;345:1305;345:1410;345:1530;345:1680" \
  --gpu-id 0
```

Or generate the same single-trail sweep in max-concurrency mode:

```bash
python3 experiments/sweep-freq-same-agent/generate_experiment.py \
  --source-plan "$RUN_DIR/seed-7-size-500.replay-plan.trail-50th.json" \
  --randomize-seed 11 \
  --max-concurrent 8 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 0 \
  --freq-list \
  "345:345;345:510;345:660;345:810;345:915;345:1005;345:1185;345:1305;345:1410;345:1530;345:1680" \
  --gpu-id 0
```

## Run The Generated Sweep

```bash
bash experiments/sweep-freq-same-agent/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-freq-same-agent/generated/<timestamp>/run_replay.sh 3
```

## Generated Files

For each frequency slug `core-<min>-<max>`:

- `generated/<timestamp>/freq/<freq-slug>/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<source-plan-name>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

## Prerequisites

Install local Zeus tools so the generated script can call:

- `set-gpu-core-freq`
- `reset-gpu-core-freq`
- `zeus-power-reader`

```bash
pip install -e ./zeus-power-reader
```
