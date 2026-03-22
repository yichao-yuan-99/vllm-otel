# Sweep QPS Docker Power Clean

This workflow is a sweep-QPS version of `single-qps/local` and adds Zeus power
logging like `single-qps-sweep-freq` (without GPU frequency control).

Compared to `single-qps/local`:

- input accepts `--qps-list` instead of a single `--qps`
- generated runner executes all QPS points sequentially
- each run starts/stops `zeus-power-reader`
- power logs are written to `<replay_output_dir>/power/`
- plan lookup always uses clean split plans:
  - `replay-plan.clean.<metric>.top(.<suffix>).json`
  - `replay-plan.clean.<metric>.rest(.<suffix>).json`
  - `replay-plan.clean.<metric>.exclude-unranked(.<suffix>).json`

Default replay output layout:

- `results/replay/sweep-qps-docker-power-clean/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/`
  where `<dataset>/<agent>` is inferred from `--source-run-dir` by dropping
  the first (`<model>`) and last (`<run-dir>`) path segments.

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-docker-power-clean/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.25,0.3,0.35 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 13 \
  --split rest \
  --additional-suffix qwen3_fp8 \
  --power-gpu-indices 3
```

Optional `--power-gpu-indices` accepts comma-separated IDs, e.g. `0,1`.

`--split` values are `top`, `rest`, or `exclude-unranked` (`full` is accepted
as a compatibility alias for `exclude-unranked`).

## Run Generated Sweep

```bash
bash experiments/sweep-qps-docker-power-clean/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-qps-docker-power-clean/generated/<timestamp>/run_replay.sh 3
```

## Generated Files

For each QPS slug `qpsX`:

- `generated/<timestamp>/qpsX/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

## Prerequisites

Install local Zeus tools:

```bash
pip install -e ./zeus-power-reader
```
