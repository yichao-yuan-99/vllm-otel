# SLO-Decision Visualization

This directory renders one timeline figure per run from
`post-processed/slo-decision/slo-decision-summary.json`.

The figure shows:

- moving-average min output throughput at SLO-triggered decisions
- the configured throughput target
- the frequency target chosen at each SLO-triggered decision

## Script

- `post-process/visualization/slo-decision/generate_all_figures.py`

## Single Run

```bash
python post-process/visualization/slo-decision/generate_all_figures.py \
  --run-dir <run-dir>
```

Default input:

```text
<run-dir>/post-processed/slo-decision/slo-decision-summary.json
```

Default output directory:

```text
<run-dir>/post-processed/visualization/slo-decision/
```

Outputs:

- `slo-decision-timeline.<png|pdf|svg>`
- `figures-manifest.json`

Optional arguments:

- `--slo-decision-input <path>`
- `--output-dir <dir>`
- `--format <png|pdf|svg>`
- `--dpi <positive-int>`

## Batch Mode

```bash
python post-process/visualization/slo-decision/generate_all_figures.py \
  --root-dir <root-dir>
```

Optional arguments:

- `--max-procs <positive-int>` (default: `MAX_PROCS` env var, else CPU count)
- `--dry-run`
- `--format <png|pdf|svg>`
- `--dpi <positive-int>`
