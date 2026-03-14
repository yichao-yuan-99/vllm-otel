# Sweep QPS Local (Single Profile)

This workflow builds local-mode sbatch bundles for QPS sweeps using
`servers/servers-amdhpc/render-sbatch.py start --local-mode ...`.

Each generated experiment (one per QPS) is isolated in its own subdirectory and contains:

- `replay.toml`: replay config
- `run_local_replay.sh`: local-mode script executed by sbatch
- `sbatch.sh`: rendered sbatch script (not submitted automatically)

Each generated batch root also includes:

- `submit_all.sh`: submits every generated experiment in that batch with `sbatch`

## Generate Experiments

```bash
python3 experiments/sweep-qps-local/single/generate_replay_configs.py \
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

- `experiments/sweep-qps-local/single/generated/<utc-timestamp>/`

Within that batch directory, each QPS has its own subdirectory:

- `qps0_05/`
- `qps0_1/`
- `qps0_2/`
- ...

The script also writes:

- `manifest.json` (batch summary + `sbatch` submit commands)

## Submit

Each experiment is submitted independently:

```bash
sbatch experiments/sweep-qps-local/single/generated/<utc-timestamp>/qps0_1/sbatch.sh
```

Or submit the whole batch:

```bash
bash experiments/sweep-qps-local/single/generated/<utc-timestamp>/submit_all.sh
```

## Notes

- This is single-profile local mode (default `--port-profile 0`).
- You can override profile with `-P/--port-profile`.
- You can pass `--lmcache <size>` to forward LMCache settings into rendered sbatch via `render-sbatch.py start --lmcache`.
- Baseline sweep-qps options are supported (including replay JSON overlays),
  plus render-sbatch model/partition options (`-m`, `-p`).
