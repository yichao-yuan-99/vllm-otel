# Sweep QPS Docker Power Clean Fixed Freq

This workflow is a sweep-QPS version of `single-qps/local` and adds Zeus power
logging like `single-qps-sweep-freq`, but uses one fixed GPU core clock range
for the entire sweep.

Compared to `single-qps/local`:

- input accepts `--qps-list` instead of a single `--qps`
- generated runner executes all QPS points sequentially
- each run starts/stops `zeus-power-reader`
- before the first replay, the runner locks every selected GPU to
  `core-345-<freq>`
- after the sweep exits, the runner resets those GPU core clocks
- power logs are written to `<replay_output_dir>/power/`
- plan lookup always uses clean split plans:
  - `replay-plan.clean.<metric>.top(.<suffix>).json`
  - `replay-plan.clean.<metric>.rest(.<suffix>).json`
  - `replay-plan.clean.<metric>.exclude-unranked(.<suffix>).json`

Default replay output layout:

- `results/replay/sweep-qps-docker-power-clean-fixed-freq/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/core-345-<freq>/`
  where `<dataset>/<agent>` is inferred from `--source-run-dir` by dropping
  the first (`<model>`) and last (`<run-dir>`) path segments.

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-docker-power-clean-fixed-freq/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.25,0.3,0.35 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 13 \
  --split rest \
  --additional-suffix qwen3_fp8 \
  --power-gpu-indices 3 \
  --freq 1305
```

`--freq` is required and becomes the fixed maximum core clock in MHz. The
generated runner always uses a fixed minimum of `345 MHz`.

`--power-gpu-indices` accepts comma-separated IDs, e.g. `0,1`. Those same GPUs
are the ones whose clocks get locked and reset.

`--split` values are `top`, `rest`, or `exclude-unranked` (`full` is accepted
as a compatibility alias for `exclude-unranked`).

## Run Generated Sweep

```bash
bash experiments/sweep-qps-docker-power-clean-fixed-freq/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-qps-docker-power-clean-fixed-freq/generated/<timestamp>/run_replay.sh 3
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

The same editable install also provides `set-gpu-core-freq` and
`reset-gpu-core-freq`.
