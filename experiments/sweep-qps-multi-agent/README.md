# Sweep QPS Multi Agent

This workflow takes a list of replay plans that were already compiled with
`replayer compile --single-trail` and generates a local 2D experiment bundle
that sweeps:

- trails / plans
- QPS points

Each matrix point runs with Zeus power logging.

Unlike the split-based QPS sweep generators, this script does not look up
plans from `--source-run-dir`. You pass the exact single-trail plans with
repeated `--source-plan` arguments.

For each plan x QPS point, the generated `run_replay.sh` does:

1. starts `zeus-power-reader`
2. runs replay once
3. stops power reader

Power logs are written to a `power/` subdirectory inside each replay output
directory.

Default replay output layout:

- `results/replay/sweep-qps-multi-agent/<dataset-lineage>/trail/<batch-trail-slug>/<qps>/<timestamp>/`

Each input plan keeps its own inferred dataset lineage from `source_job_dir`
when possible. If a plan's lineage cannot be inferred, that plan falls back to
`trail/<batch-trail-slug>/...`.

## Generate One 2D Sweep Bundle

```bash
python3 experiments/sweep-qps-multi-agent/generate_experiment.py \
  --source-plan results/.../replay-plan.trail-profile-1_run_alpha.json \
  --source-plan results/.../replay-plan.trail-profile-2_run_beta.json \
  --source-plan results/.../replay-plan.trail-profile-4_run_gamma.json \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.25,0.3,0.35 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 0 \
  --power-gpu-indices 0
```

Optional `--power-gpu-indices` accepts comma-separated IDs, e.g. `0,1`.

## Compile The Single-Trail Plans First

One common flow is:

1. pick multiple `source_trail_name` values, for example from
   `original-analysis/percentiles/context-usage-percentiles.json`
2. compile each trail into its own plan
3. generate the 2D sweep from those plans

Example:

```bash
RUN_DIR="results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z"
PERCENTILES_JSON="$RUN_DIR/original-analysis/percentiles/context-usage-percentiles.json"
TRAIL_A="$(jq -r '.source_trail_names_by_percentile["10"]' "$PERCENTILES_JSON")"
TRAIL_B="$(jq -r '.source_trail_names_by_percentile["50"]' "$PERCENTILES_JSON")"
TRAIL_C="$(jq -r '.source_trail_names_by_percentile["90"]' "$PERCENTILES_JSON")"

python3 -m replayer compile --job-dir "$RUN_DIR" --model qwen3_coder_30b_fp8 --single-trail "$TRAIL_A" --plan-out "$RUN_DIR/replay-plan.trail-10th.json"
python3 -m replayer compile --job-dir "$RUN_DIR" --model qwen3_coder_30b_fp8 --single-trail "$TRAIL_B" --plan-out "$RUN_DIR/replay-plan.trail-50th.json"
python3 -m replayer compile --job-dir "$RUN_DIR" --model qwen3_coder_30b_fp8 --single-trail "$TRAIL_C" --plan-out "$RUN_DIR/replay-plan.trail-90th.json"

python3 experiments/sweep-qps-multi-agent/generate_experiment.py \
  --source-plan "$RUN_DIR/replay-plan.trail-10th.json" \
  --source-plan "$RUN_DIR/replay-plan.trail-50th.json" \
  --source-plan "$RUN_DIR/replay-plan.trail-90th.json" \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.25,0.3,0.35 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 0 \
  --power-gpu-indices 0
```

## Run The Generated Sweep

```bash
bash experiments/sweep-qps-multi-agent/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-qps-multi-agent/generated/<timestamp>/run_replay.sh 3
```

## Generated Files

For each trail slug `trail/<batch-trail-slug>` and QPS slug `qpsX`:

- `generated/<timestamp>/trail/<batch-trail-slug>/<qps-slug>/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/plan-XX.<source-plan-name>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

## Prerequisites

Install local Zeus tools:

```bash
pip install -e ./zeus-power-reader
```
