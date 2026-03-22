# Replay-Plan Fine-Grained N-Gram Frequency

This directory contains a fine-grained n-gram frequency analyzer for replay plans.

For each replay plan:
- each worker is treated as one trace,
- each request index is treated as one step,
- for every step `i`, stats are computed from a window defined by `--window-mode`:
  - `centered`: `[i-w, i+w]`
  - `trailing`: `[i-w, i]` (single-side)
  - `cumulative`: `[0, i]`
- n-gram frequency statistics are produced per step (avg, std, min, max, etc.).

## Output

The script writes output JSON files under a subdirectory of `output/` by default:
- default output dir: `analysis/gram-freq-fine/output/run-<UTC timestamp>/`
- one JSON file per valid replay plan
- output filename includes window size and mode:
  `<plan-stem>.w<window-size>.<window-mode>.gram-freq-fine.json`
  - exception: for `cumulative`, window size is ignored and filename is
    `<plan-stem>.cumulative.gram-freq-fine.json`

Each output JSON contains:
- plan metadata
- `window_size`
- `window_mode`
- analyzed `n_values`
- per-trace step series with windowed n-gram stats
- per n-gram stats include `avg`, `std`, `p25`, `p75`, `min`, `max`, `unique_count`, `total_occurrences`

## Usage

```bash
python3 analysis/gram-freq-fine/compute_replay_plan_gram_freq_fine.py \
  --root /path/to/root
```

Optional flags:
- `--window-size W`: window size `w` (default: `2`)
  - ignored when `--window-mode cumulative`
- `--window-mode centered|trailing|cumulative`: centered=`[i-w,i+w]`, trailing=`[i-w,i]`, cumulative=`[0,i]` (default: `centered`)
- `--n-values 1,2`: comma-separated n values (default: `1,2`)
- `--output-dir /path/to/output-dir`: override output directory
- `--jobs N`: parallel worker count (default: CPU count)
- `--no-progress`: disable interactive progress bar

## Progress Bar

By default, when running in an interactive terminal, the script shows a live progress bar while replay plans are being processed.

## Visualization

You can plot one fine-grained output JSON as a large PDF line chart:
- x-axis: normalized step (`step_index / step_count`)
- y-axis: selected stats
- each selected stat is rendered in a separate figure/PDF
- by default, generated stats are: `avg`, `p25`, `p75`, `std`

```bash
python3 analysis/gram-freq-fine/plot_gram_freq_fine_line_chart.py \
  --input /path/to/replay-plan.w2.centered.gram-freq-fine.json \
  --n 1
```

To apply this to all fine-result JSON files under a directory:

```bash
python3 analysis/gram-freq-fine/plot_gram_freq_fine_line_chart.py \
  --root /path/to/analysis/gram-freq-fine/output \
  --n 1
```

In `--root` mode, each PDF is written next to its input JSON file.
Default PDF filename includes window size/mode, n, and stat:
- `<input-stem>.w<window-size>.<window-mode>.n<n>.<stat>.line.pdf`
  - exception: for `cumulative`, window size is ignored and default is
    `<input-stem>.cumulative.n<n>.<stat>.line.pdf`

Optional flags:
- `--output /path/to/chart.pdf`: output path (`--input` mode only)
  - if one stat is selected, exactly one PDF is written to this path
  - if multiple stats are selected, one PDF per stat is written using this path as base (e.g. `.avg.pdf`, `.p25.pdf`, ...)
- `--glob '*.gram-freq-fine.json'`: recursive pattern used by `--root`
- `--stats avg,p25,p75,std`: comma-separated stats to draw as lines
- `--jobs N`: parallel worker process count for `--root` mode (default: CPU count)
- `--figure-width W`: PDF width in inches (default: `18.0`)
- `--figure-height H`: PDF height in inches (default: `10.5`)
