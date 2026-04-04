# Sweep QPS Docker Power Clean Freq Ctrl Linespace

This workflow is the `freq-controller-linespace` variant of
`experiments/sweep-qps-docker-power-clean`.

It looks up already-compiled clean split plans from a profiled source run,
generates one replay config per QPS point, and produces a single runner that
wraps each replay with both Zeus power logging and `freq-controller-linespace`.

Compared to `sweep-qps-docker-power-clean`:

- each QPS run also starts and stops `freq-controller-linespace`
- linespace controller logs are written to `<replay_output_dir>/freq-control-linespace/`
- the generated runner resets GPU core clocks after each QPS point
- `--gpu-id` is used for both `freq-controller-linespace` and
  `zeus-power-reader`

The underlying replay launch pattern is still Poisson, so `--qps-list`
represents Poisson request rates and requires `--poisson-seed`.

Because `freq-controller-linespace` waits for gateway `job_active=true`, it
can be started before replay and remain pending until the replay job actually
begins.

Default controller threshold:

- threshold: `395784`

Default replay output layout:

- `results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/`

Inside each replay output directory, the generated runner writes:

- power logs under `power/`
- `freq-controller-linespace` query and decision logs under `freq-control-linespace/`
  using the `freq-controller-ls.*.jsonl` filename prefix

Plan lookup always uses clean split plans:

- `replay-plan.clean.<metric>.top(.<suffix>).json`
- `replay-plan.clean.<metric>.rest(.<suffix>).json`
- `replay-plan.clean.<metric>.exclude-unranked(.<suffix>).json`

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace/generate_experiment.py \
  --source-run-dir  results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/  \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.02,0.04,0.08 \
  --time-constraint-s 10800 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 2 \
  --split exclude-unranked \
  --additional-suffix qwen3_fp8 \
  --gpu-id 2
```

Optional flags:

- `--split-two-group-metric context_usage`
- `--freq-controller-threshold 395784`

`--split` values are `top`, `rest`, or `exclude-unranked`. `full` is accepted
as a compatibility alias for `exclude-unranked`.

## Run Generated Sweep

```bash
bash experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace/generated/<timestamp>/run_replay.sh 3
```

Optional runtime overrides:

- `FREQ_CONTROLLER_CONFIG=/path/to/freq-controller-linespace.toml`
- `FREQ_CONTROLLER_THRESHOLD=...`
- `FREQ_CONTROLLER_BIN=freq-controller-linespace`
- `ZEUS_POWER_READER_BIN=zeus-power-reader`
- `ZEUSD_SOCKET_PATH=/var/run/zeusd.sock`

Use `FREQ_CONTROLLER_CONFIG` when you want to override linespace-policy fields
such as the frequency list or threshold.

## Generated Files

For each QPS slug `qpsX`:

- `generated/<timestamp>/qpsX/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<selected-plan>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

## Post-Process

After runs finish, the shared post-process pipeline can extract and visualize
the linespace controller logs through the `freq-control-linespace` flow.

```bash
python3 post-process/run_all.py \
  --root-dir results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace
```

Each run produces:

- `post-processed/freq-control-linespace/freq-control-summary.json`
- `post-processed/visualization/freq-control-linespace/freq-control-timeline.png`

## Prerequisites

Install the local console scripts used by the generated run script:

- `freq-controller-linespace`
- `zeus-power-reader`
- `reset-gpu-core-freq`

```bash
python3 -m pip install -e './freq-controller-linespace'
python3 -m pip install -e './zeus-power-reader'
```

## Tests

```bash
.venv/bin/pytest experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace/test -q
```
