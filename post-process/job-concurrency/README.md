# Job Concurrency Post-Process

This directory extracts run-level job concurrency from per-job start/end
timestamps.

Method:

- collect each job interval from run metadata (`started_at`, `finished_at`)
- convert intervals to offsets in seconds since run start
- for each second bucket `[t, t + 1)`, add `+1` for every job interval that
  overlaps that bucket

The result is a per-second concurrency timeline.

## Script

- `post-process/job-concurrency/extract_run.py`

## Single Run

Command:

```bash
python post-process/job-concurrency/extract_run.py \
  --run-dir <run-dir>
```

Supported run layouts:

- replay output: `replay/summary.json`
- con-driver output: `meta/results.json` and `meta/run_manifest.json`

Default output:

```text
<run-dir>/post-processed/job-concurrency/job-concurrency-timeseries.json
```

Optional arguments:

- `--output <path>`

## Batch Mode

Command:

```bash
python post-process/job-concurrency/extract_run.py \
  --root-dir <root-dir>
```

Discovery rule:

- any subdirectory that has either:
  - `replay/summary.json`
  - `meta/results.json` and `meta/run_manifest.json`

Optional arguments:

- `--max-procs <positive-int>`
- `--dry-run`

## Output Fields

- `source_run_dir`
- `source_type` (`replay` or `con-driver`)
- `experiment_started_at`
- `experiment_finished_at`
- `time_constraint_s`
- `service_failure_detected`
- `service_failure_cutoff_time_utc`
- `replay_count`
- `jobs_with_valid_range_count`
- `total_duration_s`
- `sample_count`
- `max_concurrency`
- `avg_concurrency`
- `concurrency_points`:
  - `second`
  - `concurrency`
- `job_intervals_preview` (first 20 valid intervals):
  - `job_id`
  - `status`
  - `start_offset_s`
  - `end_offset_s`
  - `duration_s`

## Visualization

Use the visualization helper to render a concurrency figure per run:

```bash
python post-process/visualization/job-concurrency/generate_all_figures.py \
  --run-dir <run-dir>
```

Default output directory:

```text
<run-dir>/post-processed/visualization/job-concurrency/
```

It writes:

- `job-concurrency.<format>`
- `figures-manifest.json`

See:

- `post-process/visualization/job-concurrency/README.md`
