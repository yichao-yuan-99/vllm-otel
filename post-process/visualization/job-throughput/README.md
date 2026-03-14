This directory contains figure-generation scripts for post-processed job
throughput timeseries.

It visualizes each run's moving throughput with two line charts:

- x-axis: time (`time_s`)
- y-axis: throughput (`throughput_jobs_per_s`)
- annotation: finished jobs, sample count, avg/min/max throughput, peak time,
  run duration, window size, and sampling frequency

The default variants are:

- all finished statuses
- excluding `cancelled`

The plotting style matches the existing post-process figures:

- serif font
- readable grid lines
- filled throughput curve
- peak marker and stats box

## Script

- `post-process/visualization/job-throughput/generate_all_figures.py`

## Single Run

Input file (default):

- `<run-dir>/post-processed/job-throughput/job-throughput-timeseries.json`

Command:

```bash
python post-process/visualization/job-throughput/generate_all_figures.py \
  --run-dir <run-dir>
```

Default output directory:

```text
<run-dir>/post-processed/visualization/job-throughput/
```

It writes:

- `job-throughput.<format>` (default `.png`)
- `job-throughput-excluding-cancelled.<format>` (default `.png`)
- `figures-manifest.json`

Optional parameters (single-run mode):

- `--timeseries-input <path>`
- `--output-dir <path>`
- `--format {png,pdf,svg}`
- `--dpi <positive-int>`

## Batch Processing

When you have many runs under one root directory:

```bash
python post-process/visualization/job-throughput/generate_all_figures.py \
  --root-dir results/replay
```

Discovery rule:

- any subdirectory that has:
  - `post-processed/job-throughput/job-throughput-timeseries.json`

Optional worker count:

```bash
python post-process/visualization/job-throughput/generate_all_figures.py \
  --root-dir results/replay \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/visualization/job-throughput/generate_all_figures.py \
  --root-dir results/replay \
  --dry-run
```

Batch-wide rendering options:

- `--format {png,pdf,svg}`
- `--dpi <positive-int>`

## Note

`matplotlib` is required to render figures. If it is missing, install it in your
environment, for example:

```bash
pip install matplotlib
```
