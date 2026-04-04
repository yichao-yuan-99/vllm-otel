# Freq-Control Post-Process

This directory extracts run-level freq-controller query and decision history
from:

- `freq-control/freq-controller.query.*.jsonl`
- `freq-control/freq-controller.decision.*.jsonl`
- `freq-control/freq-controller.control-error.*.jsonl`
- `freq-control-seg/freq-controller.query.*.jsonl`
- `freq-control-seg/freq-controller.decision.*.jsonl`
- `freq-control-seg/freq-controller.control-error.*.jsonl`
- `freq-control-linespace/freq-controller-ls.query.*.jsonl`
- `freq-control-linespace/freq-controller-ls.decision.*.jsonl`
- `freq-control-linespace/freq-controller-ls.control-error.*.jsonl`

For backward compatibility, it also accepts older runs where those files were
written directly at the run root.

It is tolerant of missing freq-controller logs:

- if neither log exists, extraction still succeeds
- the output marks `freq_control_log_found: false`
- point arrays are empty instead of failing the run

It produces:

1. normalized query points with `time_offset_s` relative to replay start
2. normalized decision points with context-window and frequency targets
3. normalized control-error points from zeusd write failures on the control path
4. summary stats such as max context, frequency range, and change count

## Script

- `post-process/freq-control/extract_run.py`

## Single Run

```bash
python post-process/freq-control/extract_run.py \
  --run-dir <run-dir>
```

Default output:

```text
<run-dir>/post-processed/<freq-control|freq-control-seg|freq-control-linespace>/freq-control-summary.json
```

Optional arguments:

- `--output <path>`

## Batch Mode

```bash
python post-process/freq-control/extract_run.py \
  --root-dir <root-dir>
```

Optional arguments:

- `--max-procs <positive-int>` (default: `MAX_PROCS` env var, else CPU count)
- `--dry-run`

## Output Fields (`freq-control-summary.json`)

- `source_run_dir`
- `source_type`
- `source_query_log_paths`
- `source_decision_log_paths`
- `source_control_error_log_paths`
- `experiment_started_at`
- `experiment_finished_at`
- `time_constraint_s`
- `analysis_window_start_utc`
- `analysis_window_end_utc`
- `service_failure_detected`
- `service_failure_cutoff_time_utc`
- `freq_control_log_found`
- `query_log_found`
- `decision_log_found`
- `control_error_log_found`
- `query_point_count`
- `pending_query_point_count`
- `active_query_point_count`
- `query_error_count`
- `control_error_point_count`
- `decision_point_count`
- `decision_change_count`
- `first_job_active_time_offset_s`
- `first_control_error_time_offset_s`
- `lower_bound`
- `upper_bound`
- `linespace_policy_detected`
- `target_context_usage_threshold`
- `segment_count`
- `segment_width_context_usage`
- `source_freq_control_log_dir_name`
- `segmented_policy_detected`
- `low_freq_threshold`
- `low_freq_cap_mhz`
- `min_effective_min_frequency_mhz`
- `max_effective_min_frequency_mhz`
- `max_context_usage`
- `max_window_context_usage`
- `min_frequency_mhz`
- `max_frequency_mhz`
- `query_points`
- `decision_points`
- `control_error_points`

## Visualization

Use the visualization helper to render a freq-control figure per run:

```bash
python post-process/visualization/freq-control/generate_all_figures.py \
  --run-dir <run-dir>
```

See:

- `post-process/visualization/freq-control/README.md`
