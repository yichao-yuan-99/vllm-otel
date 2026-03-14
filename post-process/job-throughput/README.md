# Job Throughput Post-Process

This directory extracts moving replay/job throughput in jobs per second for each
run.

For each sampled time point `t`, throughput is computed from finished jobs in the
window `[t - delta, t + delta]`, clipped to the run boundaries. Only jobs with a
valid `finished_at` timestamp are counted.

That means:

- unfinished or timed-out jobs with no `finished_at` are ignored
- jobs with a valid `finished_at` are counted, including replay
  `time_bound_finished` jobs
- the default throughput series includes any finished status, including
  `cancelled`
- the output also includes a second copy with `cancelled` jobs excluded
- if the run metadata includes `time_constraint_s`, the throughput timeline is
  hard-capped there and jobs finishing after that cutoff are ignored
- if a run lasts `x` seconds and sampling is `1 Hz`, the extractor writes `x`
  throughput points at times `0, 1, ..., x - 1`

## Script

- `post-process/job-throughput/extract_run.py`

## Single Run

Command:

```bash
python post-process/job-throughput/extract_run.py \
  --run-dir <run-dir>
```

Supported run layouts:

- replay output: `replay/summary.json`
- con-driver output: `meta/results.json` and `meta/run_manifest.json`

Default output:

```text
<run-dir>/post-processed/job-throughput/job-throughput-timeseries.json
```

Optional arguments:

- `--output <path>`
- `--timepoint-freq-hz <positive-float>` (default `1.0`)
- `--window-size-s <positive-float>` (default `600.0`)

Notes:

- `--window-size-s` is the half-width `delta` from the original
  `(t - delta, t + delta)` definition
- the effective full window width is `2 * window_size_s`

## Batch Mode

Command:

```bash
python post-process/job-throughput/extract_run.py \
  --root-dir <root-dir>
```

Discovery rule:

- any subdirectory that has either:
  - `replay/summary.json`
  - `meta/results.json` and `meta/run_manifest.json`

Optional arguments:

- `--max-procs <positive-int>`
- `--dry-run`
- `--timepoint-freq-hz <positive-float>`
- `--window-size-s <positive-float>`

## Output Fields

- `source_run_dir`
- `source_type` (`replay` or `con-driver`)
- `experiment_started_at`
- `experiment_finished_at`
- `time_constraint_s`
- `replay_count`
- `finished_replay_count`
- `finished_replay_count_excluding_cancelled`
- `cancelled_finished_replay_count`
- `total_duration_s`
- `timepoint_frequency_hz`
- `timepoint_interval_s`
- `window_size_s`
- `window_width_s`
- `sample_count`
- `throughput_points`:
  - `time_s`
  - `throughput_jobs_per_s`
- `throughput_points_excluding_cancelled`:
  - `time_s`
  - `throughput_jobs_per_s`

## Visualization

Use the visualization helper to render two throughput figures per run:

```bash
python post-process/visualization/job-throughput/generate_all_figures.py \
  --run-dir <run-dir>
```

Default output directory:

```text
<run-dir>/post-processed/visualization/job-throughput/
```

It writes:

- `job-throughput.<format>`
- `job-throughput-excluding-cancelled.<format>`
- `figures-manifest.json`

See:

- `post-process/visualization/job-throughput/README.md`
