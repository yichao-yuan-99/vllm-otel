This directory contains figure-generation scripts for post-processed gateway
stacked token throughput.

It visualizes each metric from gateway stack histograms as its own line chart:

- `prompt_tokens`
- `cached_tokens`
- `compute_prompt_tokens`
- `completion_tokens`
- `compute_prompt_plus_completion_tokens`

For each metric, it generates:

- 1 raw figure
- 4 smoothed figures using centered windows `(t-w, t+w)` with
  `w = 10s, 30s, 60s, 120s`

Each figure uses:

- x-axis: second from run start (`second`)
- y-axis: accumulated value in the 1-second bucket (`accumulated_value`)
- annotation: sample count, avg/min/max, peak second, and sum of values

## Script

- `post-process/visualization/gateway-stack/generate_all_figures.py`

## Single Run

Input directory (default):

- `<run-dir>/post-processed/gateway/stack/`

Required files:

- `prompt-tokens-stacked-histogram.json`
- `cached-tokens-stacked-histogram.json`
- `compute-prompt-tokens-stacked-histogram.json`
- `completion-tokens-stacked-histogram.json`
- `compute-prompt-plus-completion-tokens-stacked-histogram.json`

Command:

```bash
python post-process/visualization/gateway-stack/generate_all_figures.py \
  --run-dir <run-dir>
```

Default output directory:

```text
<run-dir>/post-processed/visualization/gateway-stack/
```

It writes:

- `prompt-tokens-stacked-histogram.<format>` (default `.png`)
- `prompt-tokens-stacked-histogram-smoothed-10s.<format>`
- `prompt-tokens-stacked-histogram-smoothed-30s.<format>`
- `prompt-tokens-stacked-histogram-smoothed-60s.<format>`
- `prompt-tokens-stacked-histogram-smoothed-120s.<format>`
- `cached-tokens-stacked-histogram.<format>` (default `.png`)
- `cached-tokens-stacked-histogram-smoothed-10s.<format>`
- `cached-tokens-stacked-histogram-smoothed-30s.<format>`
- `cached-tokens-stacked-histogram-smoothed-60s.<format>`
- `cached-tokens-stacked-histogram-smoothed-120s.<format>`
- `compute-prompt-tokens-stacked-histogram.<format>` (default `.png`)
- `compute-prompt-tokens-stacked-histogram-smoothed-10s.<format>`
- `compute-prompt-tokens-stacked-histogram-smoothed-30s.<format>`
- `compute-prompt-tokens-stacked-histogram-smoothed-60s.<format>`
- `compute-prompt-tokens-stacked-histogram-smoothed-120s.<format>`
- `completion-tokens-stacked-histogram.<format>` (default `.png`)
- `completion-tokens-stacked-histogram-smoothed-10s.<format>`
- `completion-tokens-stacked-histogram-smoothed-30s.<format>`
- `completion-tokens-stacked-histogram-smoothed-60s.<format>`
- `completion-tokens-stacked-histogram-smoothed-120s.<format>`
- `compute-prompt-plus-completion-tokens-stacked-histogram.<format>` (default `.png`)
- `compute-prompt-plus-completion-tokens-stacked-histogram-smoothed-10s.<format>`
- `compute-prompt-plus-completion-tokens-stacked-histogram-smoothed-30s.<format>`
- `compute-prompt-plus-completion-tokens-stacked-histogram-smoothed-60s.<format>`
- `compute-prompt-plus-completion-tokens-stacked-histogram-smoothed-120s.<format>`
- `figures-manifest.json`

Optional parameters (single-run mode):

- `--stack-input-dir <path>`
- `--output-dir <path>`
- `--format {png,pdf,svg}`
- `--dpi <positive-int>`

## Batch Processing

When you have many runs under one root directory:

```bash
python post-process/visualization/gateway-stack/generate_all_figures.py \
  --root-dir results/replay
```

Discovery rule:

- any subdirectory that has all 5 required histogram inputs under
  `post-processed/gateway/stack/`

Optional worker count:

```bash
python post-process/visualization/gateway-stack/generate_all_figures.py \
  --root-dir results/replay \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/visualization/gateway-stack/generate_all_figures.py \
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
