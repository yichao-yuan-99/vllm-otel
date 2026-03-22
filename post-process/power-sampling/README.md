# Power Sampling Post-Process

This directory samples GPU power at prefill-concurrency tick timestamps and
summarizes power statistics by prefill concurrency.

It depends on outputs from:

- `post-process/power/extract_run.py`
- `post-process/prefill-concurrency/extract_run.py`

Inputs:

- `<run-dir>/post-processed/power/power-summary.json`
- `<run-dir>/post-processed/prefill-concurrency/prefill-concurrency-timeseries.json`

The extraction logic:

1. load power points `(time_offset_s, power_w)` from `power-summary.json`
2. build a piecewise-linear interpolation model over power
3. clamp outside the power range to the nearest endpoint power
4. sample power at every prefill-concurrency tick time
5. aggregate `avg/min/max/std` power for each concurrency value (`0`, `1`, `2`, ...)
6. also aggregate a combined `non-zero` power summary

## Script

- `post-process/power-sampling/extract_run.py`

## Single Run

Command:

```bash
python post-process/power-sampling/extract_run.py \
  --run-dir <run-dir>
```

Default output:

```text
<run-dir>/post-processed/power-sampling/power-sampling-summary.json
```

Optional arguments:

- `--power-summary <path>`
- `--prefill-timeseries <path>`
- `--output <path>`

## Batch Mode

Command:

```bash
python post-process/power-sampling/extract_run.py \
  --root-dir <root-dir>
```

Discovery rule:

- any subdirectory that has both:
  - `post-processed/power/power-summary.json`
  - `post-processed/prefill-concurrency/prefill-concurrency-timeseries.json`

Optional arguments:

- `--max-procs <positive-int>`
- `--dry-run`

## Output Fields (`power-sampling-summary.json`)

- `source_run_dir`
- `source_power_summary_path`
- `source_prefill_concurrency_timeseries_path`
- `source_type`
- `service_failure_detected`
- `service_failure_cutoff_time_utc`
- `power_log_found`
- `request_count`
- `prefill_activity_count`
- `total_duration_s`
- `tick_ms`
- `tick_s`
- `prefill_tick_count`
- `power_point_count`
- `sampled_tick_count`
- `all_power_stats_w` (`sample_count`, `avg_power_w`, `min_power_w`, `max_power_w`, `std_power_w`)
- `non_zero_power_stats_w` (same fields)
- `concurrency_power_stats_w`:
  - one entry per concurrency value (`"0"`, `"1"`, ...)
  - each entry includes `concurrency` and the same power stat fields
- `sampling_method` (`interpolation`, `outside_power_range`)
