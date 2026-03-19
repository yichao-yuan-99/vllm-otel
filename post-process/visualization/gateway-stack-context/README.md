This directory contains figure-generation scripts for post-processed gateway
stacked context usage.

It visualizes the stacked context-usage metric as line charts:

- raw chart from `context-usage-stacked-histogram.json`
- smoothed charts using centered windows `(t-w, t+w)` with
  `w = 10s, 30s, 60s, 120s`

Each figure uses:

- x-axis: second from run start (`second`)
- y-axis: accumulated context usage in the 1-second bucket (`accumulated_value`)
- annotation: sample count, avg/min/max, peak second, and sum of values

## Script

- `post-process/visualization/gateway-stack-context/generate_all_figures.py`

## Single Run

Input directory (default):

- `<run-dir>/post-processed/gateway/stack-context/`

Required file:

- `context-usage-stacked-histogram.json`

Command:

```bash
python post-process/visualization/gateway-stack-context/generate_all_figures.py \
  --run-dir <run-dir>
```

Default output directory:

```text
<run-dir>/post-processed/visualization/gateway-stack-context/
```

It writes:

- `context-usage-stacked-histogram.<format>` (default `.png`)
- `context-usage-stacked-histogram-smoothed-10s.<format>`
- `context-usage-stacked-histogram-smoothed-30s.<format>`
- `context-usage-stacked-histogram-smoothed-60s.<format>`
- `context-usage-stacked-histogram-smoothed-120s.<format>`
- `figures-manifest.json`

Optional parameters (single-run mode):

- `--stack-context-input-dir <path>`
- `--output-dir <path>`
- `--format {png,pdf,svg}`
- `--dpi <positive-int>`

## Batch Processing

When you have many runs under one root directory:

```bash
python post-process/visualization/gateway-stack-context/generate_all_figures.py \
  --root-dir results/replay
```

Discovery rule:

- any subdirectory that has:
  - `post-processed/gateway/stack-context/context-usage-stacked-histogram.json`

Optional worker count:

```bash
python post-process/visualization/gateway-stack-context/generate_all_figures.py \
  --root-dir results/replay \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/visualization/gateway-stack-context/generate_all_figures.py \
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
