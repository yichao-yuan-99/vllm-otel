# Single QPS Sweep Frequency (Local)

This workflow extends `experiments/single-qps/local` with GPU core clock sweeping.

For each frequency range in `--freq-list`, the generated `run_replay.sh` does:

1. `set-gpu-core-freq` for `--gpu-id`
2. starts `zeus-power-reader`
3. runs replay once
4. stops power reader
5. `reset-gpu-core-freq`

Power logs are written to a `power/` subdirectory inside each replay output directory.

Default replay output root is changed from:

- `results/replay/single-qps/split/<split>/<qps>/<timestamp>/`

to:

- `results/replay/single-qps-sweep-freq/split/<split>/<qps>/<timestamp>/<freq-slug>/`

Example `freq-slug`: `core-345-1305`.

## Generate One Frequency-Sweep Experiment

```bash
python3 experiments/single-qps-sweep-freq/local/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps 0.2 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 0 \
  --split rest \
  --additional-suffix qwen3_fp8 \
  --freq-list "345:1305;345:1005;345:810;345:510" \
  --gpu-id 0
```

`--freq-list` is a semicolon-separated list of `min_mhz:max_mhz`.

The example above means:

- run once at `345:1305`
- run once at `345:1005`
- run once at `345:810`
- run once at `345:510`

## Run The Generated Sweep

```bash
bash experiments/single-qps-sweep-freq/local/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/single-qps-sweep-freq/local/generated/<timestamp>/run_replay.sh 3
```

## Prerequisites

Install local Zeus tools so the generated script can call:

- `set-gpu-core-freq`
- `reset-gpu-core-freq`
- `zeus-power-reader`

```bash
pip install -e ./zeus-power-reader
```
