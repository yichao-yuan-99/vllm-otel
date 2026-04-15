# Sweep QPS Docker Power Clean Freq Ctrl Linespace Instance SLO

This workflow is the `freq-controller-linespace-instance-slo` variant of
`experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance`.

It looks up already-compiled clean split plans from a profiled source run,
generates one replay config per QPS point, and produces a single runner that
wraps each replay with both Zeus power logging and
`freq-controller-linespace-instance-slo`.

Compared to `sweep-qps-docker-power-clean-freq-ctrl-linespace-instance`:

- each QPS run also starts and stops `freq-controller-linespace-instance-slo`
- generation requires one bundle-wide throughput target via `--target-slo`
- the controller receives both the context threshold and the throughput target
- controller logs are written to
  `<replay_output_dir>/freq-control-linespace-instance-slo/`
- the generated runner resets GPU core clocks after each QPS point
- `--gpu-id` is used for both `freq-controller-linespace-instance-slo` and
  `zeus-power-reader`

The underlying replay launch pattern is still Poisson, so `--qps-list`
represents Poisson request rates and requires `--poisson-seed`.

Because `freq-controller-linespace-instance-slo` waits for gateway
`job_active=true`, it can be started before replay and remain pending until the
replay job actually begins.

Default controller context threshold:

- threshold: `395784`

Required bundle input:

- `--target-slo`

Default replay output layout:

- default threshold (`395784`):
  `results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/<dataset>/<agent>/split/<split>/<qps>/<slo>/<timestamp>/`
- non-default threshold:
  `results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo-<threshold>/<dataset>/<agent>/split/<split>/<qps>/<slo>/<timestamp>/`

`<slo>` is the formatted `--target-slo` value, for example `45` or `6.5`.

Inside each replay output directory, the generated runner writes:

- power logs under `power/`
- `freq-controller-linespace-instance-slo` query, decision, control-error, and
  SLO-decision logs under `freq-control-linespace-instance-slo/` using the
  `freq-controller-ls-instance-slo.*.jsonl` filename prefix

Plan lookup always uses clean split plans:

- `replay-plan.clean.<metric>.top(.<suffix>).json`
- `replay-plan.clean.<metric>.rest(.<suffix>).json`
- `replay-plan.clean.<metric>.exclude-unranked(.<suffix>).json`

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.02,0.04,0.08 \
  --target-slo 12 \
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

If `--freq-controller-threshold` is non-default and you do not pass
`--replay-output-root`, the generated replay output root is automatically
suffixed with that threshold value.

`--split` values are `top`, `rest`, or `exclude-unranked`. `full` is accepted
as a compatibility alias for `exclude-unranked`.

## Run Generated Sweep

```bash
bash experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/generated/<timestamp>/run_replay.sh 3
```

Optional runtime overrides:

- `FREQ_CONTROLLER_CONFIG=/path/to/freq-controller-linespace-instance-slo.toml`
- `FREQ_CONTROLLER_THRESHOLD=...`
- `FREQ_CONTROLLER_TARGET_SLO=...`
- `FREQ_CONTROLLER_BIN=freq-controller-linespace-instance-slo`
- `ZEUS_POWER_READER_BIN=zeus-power-reader`
- `ZEUSD_SOCKET_PATH=/var/run/zeusd.sock`

Use `FREQ_CONTROLLER_CONFIG` when you want to override instance-slo policy
fields such as the frequency list or threshold while still keeping the
bundle-generated throughput target default.

When `FREQ_CONTROLLER_TARGET_SLO` is set, the replay output directory also uses
that value for the `<slo>` path component.

## Generated Files

For each QPS slug `qpsX`:

- `generated/<timestamp>/qpsX/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<selected-plan>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

## Post-Process

This experiment writes both frequency-control logs and
`freq-controller-ls-instance-slo.slo-decision.*.jsonl` logs under
`freq-control-linespace-instance-slo/`.

The shared post-process helpers now auto-detect that layout, so the standard
extractors and visualization scripts can generate:

- `post-processed/freq-control-linespace-instance-slo/freq-control-summary.json`
- `post-processed/visualization/freq-control-linespace-instance-slo/freq-control-timeline.png`
- `post-processed/slo-decision/slo-decision-summary.json`
- `post-processed/visualization/slo-decision/slo-decision-timeline.png`

## Prerequisites

Install the local console scripts used by the generated run script:

- `freq-controller-linespace-instance-slo`
- `zeus-power-reader`
- `reset-gpu-core-freq`

```bash
python3 -m pip install -e './freq-controller-linespace-instance-slo'
python3 -m pip install -e './zeus-power-reader'
```

## Tests

```bash
.venv/bin/pytest experiments/sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo/test -q
```
