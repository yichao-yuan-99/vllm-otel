This directory contains figure-generation scripts for post-processed
freq-control summary output. It supports both baseline summaries under
`post-processed/freq-control/`, linespace-controller summaries under
`post-processed/freq-control-linespace/`, and segmented-controller summaries
under `post-processed/freq-control-seg/`.

It renders one three-panel timeline per run:

- top panel: query-time context usage and decision-window context usage
- middle panel: applied GPU frequency and frequency-change decisions
- bottom panel: temporary gateway read failures and control-path zeusd write failures

The figure also shows:

- pending query samples before replay start when present
- temporary read failures and control-write failures on their own aligned event plot
- lower and upper controller bounds
- segmented-policy markers for `lfth` and `low_freq_cap_mhz` when present
- run-start marker at `t = 0`
- a summary stats box with sample counts, control-error count, and frequency range

## Script

- `post-process/visualization/freq-control/generate_all_figures.py`

## Single Run

Input file (default):

- `<run-dir>/post-processed/freq-control-seg/freq-control-summary.json` if present
- otherwise `<run-dir>/post-processed/freq-control-linespace/freq-control-summary.json` if present
- otherwise `<run-dir>/post-processed/freq-control/freq-control-summary.json`

Command:

```bash
python post-process/visualization/freq-control/generate_all_figures.py \
  --run-dir <run-dir>
```

Default output directory:

```text
<run-dir>/post-processed/visualization/<freq-control|freq-control-seg|freq-control-linespace>/
```

It writes:

- `freq-control-timeline.<format>` (default `.png`)
- `figures-manifest.json`

Optional parameters (single-run mode):

- `--freq-control-input <path>`
- `--output-dir <path>`
- `--format {png,pdf,svg}`
- `--dpi <positive-int>`

## Batch Processing

```bash
python post-process/visualization/freq-control/generate_all_figures.py \
  --root-dir results/replay
```

Discovery rule:

- any subdirectory that has:
  - `post-processed/freq-control/freq-control-summary.json`
  - or `post-processed/freq-control-linespace/freq-control-summary.json`
  - or `post-processed/freq-control-seg/freq-control-summary.json`

Optional worker count:

```bash
python post-process/visualization/freq-control/generate_all_figures.py \
  --root-dir results/replay \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/visualization/freq-control/generate_all_figures.py \
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
