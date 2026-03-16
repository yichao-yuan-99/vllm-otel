# Sweep QPS Local (Split Top/Rest Plans)

This workflow builds local-mode sbatch bundles for QPS sweeps using split replay
plans (`top` and `rest`) and
`servers/servers-amdhpc/render-sbatch.py start --local-mode ...`.

For each group (`top`, `rest`) and each target QPS, one experiment directory is
generated with:

- `replay.toml`: replay config
- `run_local_replay.sh`: local-mode script executed by sbatch
- `gateway-config.toml`: per-experiment gateway config (port profile + output root)
- `sbatch.sh`: rendered sbatch script (not submitted automatically)

Rendered `sbatch.sh` files are patched so Slurm `--output/--error` and runtime
logs (`JOB_LOG_DIR`) go to:

- `<replay output dir>/sbatch-logs/`

The entire generated batch directory is also copied under replay outputs:

- source: `experiments/sweep-qps-local/split/generated/<utc-timestamp>/`
- destination: `results/replay/<utc-timestamp>/generated/`

## Generate Experiments

This generator expects split plans by default:

- `<source-run-dir>/replay-plan.token.top.json`
- `<source-run-dir>/replay-plan.token.rest.json`

If those do not exist, it also falls back to:

- `<source-run-dir>/replay-plan.context.top.json`
- `<source-run-dir>/replay-plan.context.rest.json`
- legacy `<source-run-dir>/replay-plan.top.json` and `<source-run-dir>/replay-plan.rest.json`

You can override with `--top-plan-path` / `--rest-plan-path`.

```bash
python3 experiments/sweep-qps-local/split/generate_replay_configs.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.05,0.1,0.2,0.4 \
  --time-constraint-s 1800 \
  --lmcache 100 \
  -p mi3001x \
  -m qwen3_coder_30b
```

Default output root:

- `experiments/sweep-qps-local/split/generated/<utc-timestamp>/`

Within that batch directory:

- `top/qps*/`
- `rest/qps*/`
- `submit_all.sh` (submits both groups)
- `manifest.json`

## Submit

Submit one experiment:

```bash
sbatch experiments/sweep-qps-local/split/generated/<utc-timestamp>/top/qps0_1/sbatch.sh
```

Submit all generated experiments:

```bash
bash experiments/sweep-qps-local/split/generated/<utc-timestamp>/submit_all.sh
```

## Notes

- This is single-profile local mode (default `--port-profile 0`).
- You can override profile with `-P/--port-profile`.
- You can pass `--lmcache <size>` to forward LMCache settings into rendered sbatch via `render-sbatch.py start --lmcache`.
- You can pass `--no-async-scheduling` to include `--no-async-scheduling` in rendered vLLM startup.
- Baseline sweep-qps options are supported (including replay JSON overlays), plus render-sbatch model/partition options (`-m`, `-p`).
