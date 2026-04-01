# Single MaxCon Clean

This experiment is the clean-plan counterpart to `single-maxcon`.

Compared to `single-maxcon`:

- plan lookup always uses clean split plans:
  - `replay-plan.clean.<metric>.top(.<suffix>).json`
  - `replay-plan.clean.<metric>.rest(.<suffix>).json`
  - `replay-plan.clean.<metric>.exclude-unranked(.<suffix>).json`
- `full` is accepted as a compatibility alias for `exclude-unranked`
- default replay output layout includes inferred dataset lineage:
  - `results/replay/single-maxcon-clean/<dataset>/<agent>/split/<split>/c<max-concurrent>/<timestamp>/`

Use the local generator here:

- `experiments/single-maxcon-clean/local/generate_experiment.py`

Runbook:

- `experiments/single-maxcon-clean/local/README.md`
