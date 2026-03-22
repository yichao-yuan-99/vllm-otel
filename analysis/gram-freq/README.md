# Replay-Plan N-Gram Frequency Stats

This directory contains a script that computes per-file 1-gram and 2-gram
frequency statistics from replay plan request responses
(`forced_token_ids` in each request).

## Definitions

- 1-gram: a single token id.
- 2-gram: two consecutive token ids.
- Frequency: occurrence count of each unique n-gram within one replay plan file.

For each replay plan file, we compute:
- average frequency
- minimum frequency
- maximum frequency
- standard deviation of frequency

## What the script does

Given an input root directory:
1. Recursively finds files with names starting with `replay-plan` and ending with `.json`.
2. Processes each replay plan in parallel.
3. Outputs one CSV row per replay plan file.

## Usage

```bash
python3 analysis/gram-freq/compute_replay_plan_gram_freq.py \
  --root /path/to/root
```

Optional flags:
- `--output /path/to/file.csv`: output path override (default: `analysis/gram-freq/output/replay-plan-gram-freq-summary.csv`).
- `--jobs N`: parallel worker count (default: CPU count).
- `--absolute-paths`: write absolute file paths in CSV.
- `--no-progress`: disable the interactive progress bar.

By default, when running in an interactive terminal, the script shows a live progress bar while processing replay plans.

## CSV Columns

- `plan_path`
- `worker_count`
- `request_count`
- `request_with_forced_token_ids`
- `missing_forced_token_ids_count`
- `total_token_count`
- `unigram_unique_count`
- `unigram_total_occurrences`
- `unigram_freq_avg`
- `unigram_freq_min`
- `unigram_freq_max`
- `unigram_freq_std`
- `bigram_unique_count`
- `bigram_total_occurrences`
- `bigram_freq_avg`
- `bigram_freq_min`
- `bigram_freq_max`
- `bigram_freq_std`
- `error`
