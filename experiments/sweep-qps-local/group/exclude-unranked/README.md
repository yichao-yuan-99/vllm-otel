# Sweep QPS Local Group Exclude-Unranked (sbatch-orchestrator)

This workflow has the same exclude-unranked goal as
`experiments/sweep-qps-local/exclude-unranked`,
but executes jobs via `sbatch-orchestrator` on a grouped sbatch run.

For each target QPS, one experiment directory is generated with:

- `replay.toml`
- `run_orchestrated_replay.sh`

At the batch root, it also generates:

- `job-list.txt` (absolute job-script paths for sbatch-orchestrator)
- `manifest.json`

Grouped sbatch provisioning is documented in
`sbatch-orchestrator/README.md`.

## Generate Replay Bundles

```bash
python3 experiments/sweep-qps-local/group/exclude-unranked/generate_replay_configs.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.05,0.1,0.2,0.4 \
  --time-constraint-s 1800
```

Default output root:

- `experiments/sweep-qps-local/group/exclude-unranked/generated/<utc-timestamp>/`

Default replay plan:

- `<source-run-dir>/replay-plan.exclude-unranked.json`

## Submit

```bash
bash sbatch-orchestrator/submit-start-group.sh \
  --job-list experiments/sweep-qps-local/group/exclude-unranked/generated/<utc-timestamp>/job-list.txt
```

Run artifacts are written under `sbatch-orchestrator/logs/<utc-timestamp>/`.

## Notes

- This generator does not render grouped sbatch scripts; see `sbatch-orchestrator/README.md`.
- You can override the default plan with `--plan-path`.
- Generated replay configs use fallback `port_profile_id = 0`; runtime `PORT_PROFILE_ID` still comes from sbatch-orchestrator slot assignment.
