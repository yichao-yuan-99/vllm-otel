# Sweep MaxCon Same Agent

This workflow takes one replay plan that was already compiled with
`replayer compile --single-trail`, then expanded with
`derived-single-plan/generate_derived_single_plan.py`, and generates a local
experiment bundle that sweeps `max_concurrent` while replaying that trail.

Unlike the split-based max-concurrency generators, this script does not look up
plans from `--source-run-dir`. You pass the exact derived plan with
`--source-plan`. Original non-derived single-trail plans are rejected.

Default replay output layout:

- `results/replay/sweep-maxcon-same-agent/<dataset-lineage>/trail/<safe-source-trail>/c<max-concurrent>/<timestamp>/`

`<dataset-lineage>` is inferred from `source_job_dir` inside the replay plan
when possible. If the lineage cannot be inferred, the output starts at
`trail/<safe-source-trail>/...`.

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-maxcon-same-agent/generate_experiment.py \
  --source-plan results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/seed-7-size-500.replay-plan.trail-profile-1_run_20260306T064004Z_02c951cc220f_9e28187e7c8c998594c2be521322d0b4.qwen3_fp8_10th.json \
  --randomize-seed 11 \
  --max-concurrent-list 100,150 \
  --time-constraint-s 1800 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 2
```

`--concurrency-list` is accepted as an alias for `--max-concurrent-list`.

## Compile Then Derive The Plan First

One common flow is:

1. pick a `source_trail_name`, for example from
   `original-analysis/percentiles/context-usage-percentiles.json`
2. compile that trail into its own plan
3. derive a multi-worker plan from that single-trail plan
4. generate the max-concurrency sweep from the derived plan

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

python3 experiments/sweep-maxcon-same-agent/generate_experiment.py \
  --source-plan "$RUN_DIR/seed-7-size-500.replay-plan.trail-50th.json" \
  --randomize-seed 11 \
  --max-concurrent-list 1,2,4,8,16 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 0
```

## Run The Generated Sweep

```bash
bash experiments/sweep-maxcon-same-agent/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-maxcon-same-agent/generated/<timestamp>/run_replay.sh 3
```

## Generated Files

For each concurrency slug `cN`:

- `generated/<timestamp>/cN/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<source-plan-name>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`
