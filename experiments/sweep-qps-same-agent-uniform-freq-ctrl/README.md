# Sweep QPS Same Agent Uniform Freq Ctrl

This workflow is the sweep-QPS counterpart to
`experiments/same-agent-uniform-freq-ctrl`.

It takes one replay plan that was already compiled with
`replayer compile --single-trail`, then expanded with
`derived-single-plan/generate_derived_single_plan.py`, and generates a local
experiment bundle that sweeps QPS while launching `freq-controller` before each
replay run and keeping Zeus power logging enabled.

Unlike the Poisson sibling, this bundle uses a `uniform` replay launch pattern,
which in this repository means a fixed inter-launch interval of exactly
`1 / qps`.

For each QPS point in `--qps-list`, the generated `run_replay.sh` does:

1. starts `zeus-power-reader`
2. starts `freq-controller`
3. runs replay once
4. stops `freq-controller`
5. stops `zeus-power-reader`
6. resets GPU core clocks as a cleanup fallback

Because `freq-controller` waits for gateway `job_active=true`, it can be
started before replay and remain pending until the replay job actually begins.

Default freq-controller bounds:

- lower bound: `369364`
- upper bound: `395784`

These correspond to `0.7` and `0.75` of `527664`.

Default replay output layout:

- `results/replay/sweep-qps-same-agent-uniform-freq-ctrl/<dataset-lineage>/trail/<safe-source-trail>/<qps>/<timestamp>/`

Within each replay output directory, the generated script writes:

- power logs under `power/`
- freq-controller query and decision JSONL logs under `freq-control/`

`<dataset-lineage>` is inferred from `source_job_dir` inside the replay plan
when possible. If the lineage cannot be inferred, the output starts at
`trail/<safe-source-trail>/...`.

Pass `--output-suffix <suffix>` to switch the top-level directory to
`trail-<safe-suffix>/...`, for example `trail-lmcache/<safe-source-trail>/...`.

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-same-agent-uniform-freq-ctrl/generate_experiment.py \
  --source-plan /srv/scratch/yichaoy2/work/vllm-otel/results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/seed-7-size-500.replay-plan.trail-profile-2_run_20260306T073714Z_34baff0dd44f_fa5a35fdf538c682d3381ed3bca44602.qwen3_fp8_75th.json \
  --randomize-seed 11 \
  --qps-list 0.2,0.225,0.25 \
  --time-constraint-s 1800 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 3 \
  --gpu-id 2
```

Optional flags:

- `--freq-controller-lower-bound 369364`
- `--freq-controller-upper-bound 395784`
- `--output-suffix lmcache`

`--gpu-id` is used for both `freq-controller` and `zeus-power-reader`.

## Run The Generated Sweep

```bash
bash experiments/sweep-qps-same-agent-uniform-freq-ctrl/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-qps-same-agent-uniform-freq-ctrl/generated/<timestamp>/run_replay.sh 5
```

Optional runtime overrides:

- `FREQ_CONTROLLER_CONFIG=/path/to/controller.toml`
- `FREQ_CONTROLLER_LOWER_BOUND=...`
- `FREQ_CONTROLLER_UPPER_BOUND=...`
- `FREQ_CONTROLLER_BIN=freq-controller`
- `ZEUS_POWER_READER_BIN=zeus-power-reader`
- `ZEUSD_SOCKET_PATH=/var/run/zeusd.sock`

## Generated Files

For each QPS slug `qpsX`:

- `generated/<timestamp>/qpsX/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<source-plan-name>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

## Post-Process

After runs finish, the shared post-process pipeline can extract and visualize
the freq-controller logs.

```bash
python3 post-process/run_all.py \
  --root-dir results/replay/sweep-qps-same-agent-uniform-freq-ctrl
```

Each run produces:

- `post-processed/freq-control/freq-control-summary.json`
- `post-processed/visualization/freq-control/freq-control-timeline.png`

## Prerequisites

Install the local console scripts used by the generated run script:

- `freq-controller`
- `zeus-power-reader`
- `reset-gpu-core-freq`

```bash
python3 -m pip install -e './freq-controller'
python3 -m pip install -e './zeus-power-reader'
```

## Tests

```bash
.venv/bin/pytest experiments/sweep-qps-same-agent-uniform-freq-ctrl/test -q
```
