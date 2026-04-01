# Sweep QPS Same Agent Uniform MultiGPU

This workflow is the fixed-interval sibling of
`experiments/sweep-qps-same-agent/`, but it is meant for port profiles that
serve one model across multiple GPUs.

It takes one replay plan that was already compiled with
`replayer compile --single-trail`, then expanded with
`derived-single-plan/generate_derived_single_plan.py`, and generates a local
experiment bundle that sweeps QPS while replaying that trail with Zeus power
logging.

Unlike the Poisson variant, this bundle uses a `uniform` replay launch pattern,
which in this repository means a fixed inter-launch interval of exactly
`1 / qps`.

For each QPS point in `--qps-list`, the generated `run_replay.sh` does:

1. starts `zeus-power-reader`
2. runs replay once
3. stops power reader

Power logs are written to a `power/` subdirectory inside each replay output
directory.

Use `--power-gpu-indices` to pass the full GPU list that corresponds to the
selected `--port-profile`. For example, if port profile `2` targets a DP=2
server running on GPUs `2,3`, use `--port-profile 2 --power-gpu-indices 2,3`.

Default replay output layout:

- `results/replay/sweep-qps-same-agent-uniform-multigpu/<dataset-lineage>/trail/<safe-source-trail>/<qps>/<timestamp>/`

`<dataset-lineage>` is inferred from `source_job_dir` inside the replay plan
when possible. If the lineage cannot be inferred, the output starts at
`trail/<safe-source-trail>/...`.

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-same-agent-uniform-multigpu/generate_experiment.py \
  --source-plan results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/seed-7-size-500.replay-plan.trail-profile-1_run_20260306T064004Z_02c951cc220f_9e28187e7c8c998594c2be521322d0b4.qwen3_fp8_10th.json \
  --randomize-seed 11 \
  --qps-list 0.25,0.3,0.35 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 2 \
  --power-gpu-indices 2,3
```

Optional `--power-gpu-indices` accepts comma-separated IDs, e.g. `2,3`.

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
  --port-profile 2

python3 experiments/sweep-qps-same-agent-uniform-multigpu/generate_experiment.py \
  --source-plan "$RUN_DIR/seed-7-size-500.replay-plan.trail-50th.json" \
  --randomize-seed 11 \
  --qps-list 0.25,0.3,0.35 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 2 \
  --power-gpu-indices 2,3
```

## Run The Generated Sweep

```bash
bash experiments/sweep-qps-same-agent-uniform-multigpu/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-qps-same-agent-uniform-multigpu/generated/<timestamp>/run_replay.sh 3
```

If you override the port profile at runtime, make sure the generated bundle's
`--power-gpu-indices` still matches the GPUs used by that profile. Otherwise,
regenerate the bundle with the correct GPU list.

## Generated Files

For each QPS slug `qpsX`:

- `generated/<timestamp>/qpsX/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<source-plan-name>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

## Prerequisites

Install local Zeus tools:

```bash
pip install -e ./zeus-power-reader
```
