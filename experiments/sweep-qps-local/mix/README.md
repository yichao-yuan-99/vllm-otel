# Sweep QPS Local (Mix Rest + Top%)

This workflow builds local-mode sbatch bundles at a fixed QPS while sweeping
how much of the `top` split is mixed into the `rest` split.

For each requested top percentage `p`, it generates a mixed plan:

- all workers from `rest`
- plus the first `floor(len(top_workers) * p / 100)` workers from `top`

Each generated experiment directory includes:

- `plan/replay-plan.mix.<percent>.json`: generated mixed replay plan
- `replay.toml`: replay config (fixed QPS for all experiments)
- `run_local_replay.sh`: local-mode script executed by sbatch
- `gateway-config.toml`: per-experiment gateway config (port profile + output root)
- `sbatch.sh`: rendered sbatch script (not submitted automatically)

Mixed plans are cached locally and reused across runs when the source top/rest
plan files are unchanged.

Rendered `sbatch.sh` files are patched so Slurm `--output/--error` and runtime
logs (`JOB_LOG_DIR`) go to:

- `<replay output dir>/sbatch-logs/`

The entire generated batch directory is also copied under replay outputs:

- source: `experiments/sweep-qps-local/mix/generated/<utc-timestamp>/`
- destination: `results/replay/<utc-timestamp>/generated/`

## Generate Experiments

This generator expects split plans by default:

- `<source-run-dir>/replay-plan.token.top.json`
- `<source-run-dir>/replay-plan.token.rest.json`

If those do not exist, it also falls back to:

- `<source-run-dir>/replay-plan.context.top.json`
- `<source-run-dir>/replay-plan.context.rest.json`
- legacy `<source-run-dir>/replay-plan.top.json` and `<source-run-dir>/replay-plan.rest.json`

Additional suffix variants from `replayer compile --additional-suffix` are also
detected automatically when top/rest share the same suffix, for example:

- `<source-run-dir>/replay-plan.token.top.v2.json`
- `<source-run-dir>/replay-plan.token.rest.v2.json`

You can override with `--top-plan-path` / `--rest-plan-path`.

```bash
python3 experiments/sweep-qps-local/mix/generate_replay_configs.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --qps 0.1 \
  --top-percent-list 0,25,50,75,100 \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --time-constraint-s 1800 \
  --lmcache 100 \
  -p mi3001x \
  -m qwen3_coder_30b
```

Default output root:

- `experiments/sweep-qps-local/mix/generated/<utc-timestamp>/`

Default cache root:

- `experiments/sweep-qps-local/mix/generated/cache/`

Override cache root if needed:

```bash
python3 experiments/sweep-qps-local/mix/generate_replay_configs.py \
  ... \
  --cache-dir /path/to/mix-plan-cache
```

### Using Additional Suffix

If your compiled plans use an additional suffix (e.g., `replay-plan.v2.json` or `replay-plan.token.top.v2.json`), pass `--additional-suffix`:

```bash
python3 experiments/sweep-qps-local/mix/generate_replay_configs.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --qps 0.1 \
  --top-percent-list 0,25,50,75,100 \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --time-constraint-s 1800 \
  --additional-suffix v2 \
  -p mi3001x \
  -m qwen3_coder_30b
```

With `--additional-suffix v2`, the generator:
- Looks for base plan at `replay-plan.v2.json`
- Discovers split plans at `replay-plan.token.top.v2.json` and `replay-plan.token.rest.v2.json`
- Outputs mixed plans as `replay-plan.mix.p50.v2.json`

Within that batch directory:

- `<percent>/` directories (for example `p0/`, `p25/`, `p50/`)
- `submit_all.sh`
- `manifest.json`

## Submit

Submit one experiment:

```bash
sbatch experiments/sweep-qps-local/mix/generated/<utc-timestamp>/p50/sbatch.sh
```

Submit all generated experiments:

```bash
bash experiments/sweep-qps-local/mix/generated/<utc-timestamp>/submit_all.sh
```

## Notes

- This is single-profile local mode (default `--port-profile 0`).
- You can override profile with `-P/--port-profile`.
- You can pass `--lmcache <size>` to forward LMCache settings into rendered sbatch via `render-sbatch.py start --lmcache`.
- You can pass `--no-async-scheduling` to include `--no-async-scheduling` in rendered vLLM startup.
- It supports the same replay JSON overlay options as `sweep-qps-local/split` (`--launch-policy-override-json`, `--extra-replay-json`).
