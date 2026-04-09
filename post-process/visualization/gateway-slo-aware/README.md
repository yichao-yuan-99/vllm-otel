# Gateway SLO-Aware Visualization

This directory renders one event-timeline figure per run from
`post-processed/gateway/slo-aware-log/slo-aware-events.json`.

The figure shows:

- gateway min and average stored throughput around SLO-aware decisions
- the configured throughput target
- per-event throughput for agents entering or leaving `ralexation`
- per-event slack and `ralexation` duration

## Script

- `post-process/visualization/gateway-slo-aware/generate_all_figures.py`

## Single Run

```bash
python post-process/visualization/gateway-slo-aware/generate_all_figures.py \
  --run-dir <run-dir>
```

Default input:

```text
<run-dir>/post-processed/gateway/slo-aware-log/slo-aware-events.json
```

Default output directory:

```text
<run-dir>/post-processed/visualization/gateway-slo-aware/
```

Outputs:

- `slo-aware-events-timeline.<png|pdf|svg>`
- `figures-manifest.json`

Optional arguments:

- `--slo-aware-input <path>`
- `--output-dir <dir>`
- `--format <png|pdf|svg>`
- `--dpi <positive-int>`

## Batch Mode

```bash
python post-process/visualization/gateway-slo-aware/generate_all_figures.py \
  --root-dir <root-dir>
```

Optional arguments:

- `--max-procs <positive-int>` (default: `MAX_PROCS` env var, else CPU count)
- `--dry-run`
- `--format <png|pdf|svg>`
- `--dpi <positive-int>`
