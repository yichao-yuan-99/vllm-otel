# Power Post-Process

This directory extracts run-level GPU power summaries from
`power/power-log.jsonl`.

The extractor is intentionally tolerant of missing power logs:

- if `power/power-log.jsonl` does not exist, extraction still succeeds
- the output marks `power_log_found: false`
- stats are reported as empty/zero values instead of failing the run

It produces three main outputs:

1. aggregate power stats (`avg`, `min`, `max`) in watts
2. total energy estimate (`total_energy_j`, `total_energy_kwh`)
3. a power-over-time series (`power_points`) for visualization

When the power log contains more than one GPU reading per sample, the summary
also includes per-GPU series under `per_gpu_power` so downstream visualization
can render one figure for each GPU in addition to the aggregate figure.

## Script

- `post-process/power/extract_run.py`

## Single Run

Command:

```bash
python post-process/power/extract_run.py \
  --run-dir <run-dir>
```

Supported run layouts:

- replay output: `replay/summary.json`
- con-driver output: `meta/results.json` and `meta/run_manifest.json`

Default output:

```text
<run-dir>/post-processed/power/power-summary.json
```

Optional arguments:

- `--output <path>`

## Batch Mode

Command:

```bash
python post-process/power/extract_run.py \
  --root-dir <root-dir>
```

Discovery rule:

- any subdirectory that has either:
  - `replay/summary.json`
  - `meta/results.json` and `meta/run_manifest.json`

Optional arguments:

- `--max-procs <positive-int>` (default: `MAX_PROCS` env var, else CPU count)
- `--dry-run`

## How The Window Is Chosen

Only power samples within the analysis window are kept:

- start: experiment `started_at`
- end: earliest available of:
  - experiment `finished_at`
  - `started_at + time_constraint_s` (if present)
  - service-failure cutoff time (if detected)

Behavior details:

- samples before run start are dropped
- samples after the chosen end time are dropped
- if service failure is detected, cutoff metadata is copied into output

## Power Aggregation Method

For each JSONL record:

- timestamp is taken from the earliest endpoint `timestamp_s` when present
- otherwise it falls back to the record-level `timestamp`
- total power is the sum of all numeric `gpu_power_w` values across endpoints
  and GPU IDs

After filtering and sorting by timestamp:

- `power_stats_w` is computed from all retained samples
- energy uses trapezoidal integration between adjacent points:
  - `energy_j += ((p_prev + p_curr) / 2) * delta_seconds`
- `per_gpu_power` mirrors the same stats/energy/points structure for each GPU
  series seen in the log

## Output Fields (`power-summary.json`)

- `source_run_dir`
- `source_type` (`replay` or `con-driver`)
- `source_power_log_path`
- `experiment_started_at`
- `experiment_finished_at`
- `time_constraint_s`
- `analysis_window_start_utc`
- `analysis_window_end_utc`
- `service_failure_detected`
- `service_failure_cutoff_time_utc`
- `power_log_found`
- `power_sample_count`
- `power_stats_w`:
  - `avg`
  - `min`
  - `max`
- `total_energy_j`
- `total_energy_kwh`
- `power_points`:
  - `time_offset_s`
  - `power_w`
- `per_gpu_power`:
  - `gpu_key`
  - `gpu_id`
  - `display_label`
  - `source_endpoint`
  - `power_sample_count`
  - `power_stats_w`
  - `total_energy_j`
  - `total_energy_kwh`
  - `power_points`

## Visualization

Use the visualization helper to render a power figure per run:

```bash
python post-process/visualization/power/generate_all_figures.py \
  --run-dir <run-dir>
```

See:

- `post-process/visualization/power/README.md`
