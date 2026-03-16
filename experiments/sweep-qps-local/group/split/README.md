# Sweep QPS Local Group Split (sbatch-orchestrator)

This workflow has the same split-plan goal as `experiments/sweep-qps-local/split`,
but executes jobs via `sbatch-orchestrator` on a grouped sbatch run.

For each split group (`top`, `rest`) and each target QPS, one experiment directory
is generated with:

- `replay.toml`
- `run_orchestrated_replay.sh`

At the batch root, it also generates:

- `job-list.txt` (absolute job-script paths for sbatch-orchestrator)
- `manifest.json`

Grouped sbatch provisioning is documented in
`sbatch-orchestrator/README.md`.

## Generate Replay Bundles

```bash
python3 experiments/sweep-qps-local/group/split/generate_replay_configs.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.05,0.1,0.2,0.4 \
  --time-constraint-s 1800
```

Default output root:

- `experiments/sweep-qps-local/group/split/generated/<utc-timestamp>/`

## Submit

```bash
bash sbatch-orchestrator/submit-start-group.sh \
  --job-list experiments/sweep-qps-local/group/split/generated/<utc-timestamp>/job-list.txt
```

Run artifacts are written under `sbatch-orchestrator/logs/<utc-timestamp>/`.

## Notes

- Split-plan discovery behavior matches `.../split/generate_replay_configs.py` (`token` then `context` then legacy).
- This generator does not render grouped sbatch scripts; see `sbatch-orchestrator/README.md`.
