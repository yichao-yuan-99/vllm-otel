# Sweep QPS Sweep Freq Same Agent Uniform

This workflow extends `experiments/sweep-freq-same-agent-uniform/` by taking a
`--qps-list` instead of one `--qps`.

It takes one replay plan that was already compiled with
`replayer compile --single-trail`, then expanded with
`derived-single-plan/generate_derived_single_plan.py`, and generates a local
experiment bundle that sweeps QPS and, for each QPS point, sweeps GPU core
frequency while replaying that exact trail.

This bundle always uses the `uniform` replay launch pattern, which in this
repository means a fixed inter-launch interval of exactly `1 / qps`.

For each QPS point in `--qps-list`, and for each frequency range in
`--freq-list`, the generated `run_replay.sh` does:

1. `set-gpu-core-freq` for `--gpu-id`
2. starts `zeus-power-reader`
3. runs replay once
4. stops power reader
5. `reset-gpu-core-freq`

Power logs are written to a `power/` subdirectory inside each replay output
directory.

Default replay output layout:

- `results/replay/sweep-qps-sweep-freq-same-agent-uniform/<dataset-lineage>/trail/<safe-source-trail>/<qps>/<timestamp>/<freq-slug>/`

`<dataset-lineage>` is inferred from `source_job_dir` inside the replay plan
when possible. If the lineage cannot be inferred, the output starts at
`trail/<safe-source-trail>/...`.

Example `freq-slug`: `core-345-1305`.

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-sweep-freq-same-agent-uniform/generate_experiment.py \
  --source-plan /srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/seed-7-size-500.replay-plan.trail-profile-2_run_20260306T073714Z_34baff0dd44f_fa5a35fdf538c682d3381ed3bca44602.qwen3_fp8_75th.json \
  --randomize-seed 11 \
  --qps-list 0.1,0.3 \
  --time-constraint-s 1800 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 13 \
  --freq-list \
  "345:345;345:510;345:660;345:810;345:915;345:1005;345:1185;345:1305;345:1410;345:1530;345:1680" \
  --gpu-id 3
```

`--qps-list` is a comma-separated list of uniform launch rates.
`--freq-list` is a semicolon-separated list of `min_mhz:max_mhz`.

## Compile Then Derive The Plan First

One common flow is:

1. pick a `source_trail_name`, for example from
   `original-analysis/percentiles/context-usage-percentiles.json`
2. compile that trail into its own plan
3. derive a multi-worker plan from that single-trail plan
4. generate the QPS x frequency sweep from that derived plan

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

python3 experiments/sweep-qps-sweep-freq-same-agent-uniform/generate_experiment.py \
  --source-plan "$RUN_DIR/seed-7-size-500.replay-plan.trail-50th.json" \
  --randomize-seed 11 \
  --qps-list 0.25,0.3,0.35 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 0 \
  --freq-list \
  "345:345;345:510;345:660;345:810;345:915;345:1005;345:1185;345:1305;345:1410;345:1530;345:1680" \
  --gpu-id 0
```

## Run The Generated Sweep

```bash
bash experiments/sweep-qps-sweep-freq-same-agent-uniform/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-qps-sweep-freq-same-agent-uniform/generated/<timestamp>/run_replay.sh 3
```

## Generated Files

For each QPS slug `qpsX` and frequency slug `core-<min>-<max>`:

- `generated/<timestamp>/<qps-slug>/freq/<freq-slug>/replay.toml`

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
