# Sweep QPS Docker Power Clean Freq Ctrl Linespace Sweep SLO

This workflow is the `freq-controller-linespace-slo` variant of
`experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace`.

It looks up already-compiled clean split plans from a profiled source run,
generates one replay config per `(qps, target_slo)` pair, and produces a
single runner that wraps each replay with both Zeus power logging and
`freq-controller-linespace-slo`.

Compared to `sweep-qps-docker-power-clean-freq-ctrl-linespace`:

- the controller is `freq-controller-linespace-slo`
- generation requires `--target-slo-list`
- the generated runner sweeps QPS in the outer loop and target SLO in the
  inner loop
- each `(qps, target_slo)` run writes to its own replay output directory
- the controller receives both the context threshold and the throughput target

The underlying replay launch pattern is still Poisson, so `--qps-list`
represents Poisson request rates and requires `--poisson-seed`.

Because `freq-controller-linespace-slo` waits for gateway `job_active=true`, it
can be started before replay and remain pending until the replay job actually
begins.

Default controller context threshold:

- threshold: `395784`

Required throughput sweep input:

- `--target-slo-list`

Default replay output layout:

- `results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo/<dataset>/<agent>/split/<split>/<qps>/<slo>/<timestamp>/`

Inside each replay output directory, the generated runner writes:

- power logs under `power/`
- `freq-controller-linespace-slo` logs under `freq-control-linespace/`
  using the `freq-controller-ls.query.*.jsonl`,
  `freq-controller-ls.decision.*.jsonl`, and
  `freq-controller-ls.slo-decision.*.jsonl` filename prefixes

Plan lookup always uses clean split plans:

- `replay-plan.clean.<metric>.top(.<suffix>).json`
- `replay-plan.clean.<metric>.rest(.<suffix>).json`
- `replay-plan.clean.<metric>.exclude-unranked(.<suffix>).json`

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.02,0.04,0.08 \
  --target-slo-list 8,10,12 \
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
bash experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo/generated/<timestamp>/run_replay.sh 3
```

Optional runtime overrides:

- `FREQ_CONTROLLER_CONFIG=/path/to/freq-controller-linespace-slo.toml`
- `FREQ_CONTROLLER_THRESHOLD=...`
- `FREQ_CONTROLLER_BIN=freq-controller-linespace-slo`
- `ZEUS_POWER_READER_BIN=zeus-power-reader`
- `ZEUSD_SOCKET_PATH=/var/run/zeusd.sock`

The QPS list and SLO list are fixed when the bundle is generated.

## Generated Files

For each `(qps, target_slo)` pair:

- `generated/<timestamp>/<qps-slug>/<slo-slug>/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<selected-plan>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

## Post-Process

After runs finish, the shared post-process pipeline can extract and visualize
both the linespace control history and the SLO-triggered decisions.

```bash
python3 post-process/run_all.py \
  --root-dir results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo
```

Each run produces:

- `post-processed/freq-control-linespace/freq-control-summary.json`
- `post-processed/visualization/freq-control-linespace/freq-control-timeline.png`
- `post-processed/slo-decision/slo-decision-summary.json`
- `post-processed/visualization/slo-decision/slo-decision-timeline.png`

## Prerequisites

Install the local console scripts used by the generated run script:

- `freq-controller-linespace-slo`
- `zeus-power-reader`
- `reset-gpu-core-freq`

```bash
python3 -m pip install -e './freq-controller-linespace-slo'
python3 -m pip install -e './zeus-power-reader'
```

## Tests

```bash
.venv/bin/pytest experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo/test -q
```
