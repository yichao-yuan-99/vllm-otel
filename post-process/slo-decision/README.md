# SLO-Decision Post-Process

This directory extracts the SLO-triggered controller decisions emitted by the
SLO linespace controller from:

- `freq-control-linespace/freq-controller-ls.slo-decision.*.jsonl`
- `freq-control-linespace/freq-controller-ls-amd.slo-decision.*.jsonl`
- `freq-control-linespace-instance-slo/freq-controller-ls-instance-slo.slo-decision.*.jsonl`

For backward compatibility, it also accepts runs where that file was written
directly at the run root.

It is tolerant of missing SLO-decision logs:

- if the log does not exist, extraction still succeeds
- the output marks `slo_decision_log_found: false`
- `decision_points` is empty instead of failing the run

It produces:

1. normalized SLO decision points with `time_offset_s` relative to replay start
2. summary stats such as target throughput, window throughput range, and
   frequency range

## Script

- `post-process/slo-decision/extract_run.py`

## Single Run

```bash
python post-process/slo-decision/extract_run.py \
  --run-dir <run-dir>
```

Default output:

```text
<run-dir>/post-processed/slo-decision/slo-decision-summary.json
```

Optional arguments:

- `--output <path>`

## Batch Mode

```bash
python post-process/slo-decision/extract_run.py \
  --root-dir <root-dir>
```

Optional arguments:

- `--max-procs <positive-int>` (default: `MAX_PROCS` env var, else CPU count)
- `--dry-run`

## Output Fields (`slo-decision-summary.json`)

- `source_run_dir`
- `source_type`
- `source_slo_decision_log_dir_name`
- `source_slo_decision_log_paths`
- `experiment_started_at`
- `experiment_finished_at`
- `time_constraint_s`
- `analysis_window_start_utc`
- `analysis_window_end_utc`
- `service_failure_detected`
- `service_failure_cutoff_time_utc`
- `slo_decision_log_found`
- `slo_decision_point_count`
- `slo_decision_change_count`
- `first_slo_decision_time_offset_s`
- `target_output_throughput_tokens_per_s`
- `min_window_min_output_tokens_per_s`
- `max_window_min_output_tokens_per_s`
- `min_frequency_mhz`
- `max_frequency_mhz`
- `decision_points`

## Visualization

Use the visualization helper to render an SLO-decision figure per run:

```bash
python post-process/visualization/slo-decision/generate_all_figures.py \
  --run-dir <run-dir>
```

See:

- `post-process/visualization/slo-decision/README.md`
