# Gateway SLO-Aware Log Post-Process

This directory extracts gateway SLO-aware `ralexation` decision logs from:

- `gateway-output/job/slo_aware_decisions_*.jsonl`

It is tolerant of missing logs when used on a single run:

- if the log does not exist, extraction still succeeds
- the output marks `slo_aware_log_found: false`
- `events` is empty instead of failing the run

It produces:

1. normalized SLO-aware events with `time_offset_s` relative to replay start
2. summary stats such as event counts, unique-agent count, throughput ranges,
   slack ranges, and wake/resume reasons

## Script

- `post-process/gateway/slo-aware-log/extract_run.py`

## Single Run

```bash
python post-process/gateway/slo-aware-log/extract_run.py \
  --run-dir <run-dir>
```

Default output:

```text
<run-dir>/post-processed/gateway/slo-aware-log/slo-aware-events.json
```

Optional arguments:

- `--output <path>`
- `--slo-aware-log <path>`

## Batch Mode

```bash
python post-process/gateway/slo-aware-log/extract_run.py \
  --root-dir <root-dir>
```

Optional arguments:

- `--max-procs <positive-int>` (default: `MAX_PROCS` env var, else CPU count)
- `--dry-run`

## Output Fields (`slo-aware-events.json`)

- `source_run_dir`
- `source_type`
- `source_slo_aware_log_paths`
- `experiment_started_at`
- `experiment_finished_at`
- `time_constraint_s`
- `analysis_window_start_utc`
- `analysis_window_end_utc`
- `service_failure_detected`
- `service_failure_cutoff_time_utc`
- `slo_aware_log_found`
- `slo_aware_event_count`
- `unique_agent_count`
- `unique_api_token_hashes`
- `first_slo_aware_event_time_offset_s`
- `target_output_throughput_tokens_per_s`
- `event_type_counts`
- `wake_reason_counts`
- `resume_disposition_counts`
- `min_output_tokens_per_s_at_events`
- `max_output_tokens_per_s_at_events`
- `min_slo_slack_s`
- `max_slo_slack_s`
- `min_ralexation_duration_s`
- `max_ralexation_duration_s`
- `min_observed_min_output_tokens_per_s`
- `max_observed_min_output_tokens_per_s`
- `min_observed_avg_output_tokens_per_s`
- `max_observed_avg_output_tokens_per_s`
- `events`

## Visualization

Use the visualization helper to render a gateway SLO-aware event timeline per
run:

```bash
python post-process/visualization/gateway-slo-aware/generate_all_figures.py \
  --run-dir <run-dir>
```
