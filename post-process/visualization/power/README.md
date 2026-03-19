This directory contains figure-generation scripts for post-processed power
summary output.

It visualizes each run's GPU power change over time with one line chart:

- x-axis: time from run start (`time_offset_s`)
- y-axis: total GPU power (`power_w`)
- annotation: sample count, avg/min/max power, peak time, and total energy

The plotting style matches the existing post-process figures:

- serif font
- readable grid lines
- filled curve
- peak marker and stats box

## Script

- `post-process/visualization/power/generate_all_figures.py`

## Single Run

Input file (default):

- `<run-dir>/post-processed/power/power-summary.json`

Command:

```bash
python post-process/visualization/power/generate_all_figures.py \
  --run-dir <run-dir>
```

Default output directory:

```text
<run-dir>/post-processed/visualization/power/
```

It writes:

- `gpu-power-over-time.<format>` (default `.png`)
- `figures-manifest.json`

Optional parameters (single-run mode):

- `--power-input <path>`
- `--output-dir <path>`
- `--format {png,pdf,svg}`
- `--dpi <positive-int>`

## Batch Processing

When you have many runs under one root directory:

```bash
python post-process/visualization/power/generate_all_figures.py \
  --root-dir results/replay
```

Discovery rule:

- any subdirectory that has:
  - `post-processed/power/power-summary.json`

Optional worker count:

```bash
python post-process/visualization/power/generate_all_figures.py \
  --root-dir results/replay \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/visualization/power/generate_all_figures.py \
  --root-dir results/replay \
  --dry-run
```

Batch-wide rendering options:

- `--format {png,pdf,svg}`
- `--dpi <positive-int>`

## Note

`matplotlib` is required to render figures. If it is missing, install it in
your environment, for example:

```bash
pip install matplotlib
```
