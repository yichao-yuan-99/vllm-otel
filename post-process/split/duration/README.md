# Split Duration Post-Process

This directory splits worker jobs by context-usage rank and summarizes each run.

Context usage proxy:

- per-job `max_request_length` (`prompt_tokens + decode_tokens`)

For each run:

- jobs are sorted by `max_request_length`
- jobs are assigned into percentile bins (`0-10%`, `10-20%`, ..., `90-100%`)
- jobs with no valid token-usage requests (missing prompt/completion token usage for all requests) are excluded from bins/tables
- each bin reports average of:
  - `duration_s`
  - `turn_count`
  - `prompt_tokens`
  - `decode_tokens`
  - `cached_prompt_tokens`

That gives a per-run 5x10 table by default.

## Scripts

- `post-process/split/duration/extract_run.py`
- `post-process/split/duration/aggregate_runs_csv.py`

## Single Run Extraction

Command:

```bash
python post-process/split/duration/extract_run.py \
  --run-dir <run-dir>
```

Default output:

```text
<run-dir>/post-processed/split/duration/duration-split-summary.json
```

Optional arguments:

- `--output <path>`
- `--split-count <positive-int>` (default `10`)

## Batch Extraction

Command:

```bash
python post-process/split/duration/extract_run.py \
  --root-dir <root-dir>
```

Discovery rule:

- any subdirectory that has a direct `gateway-output/` child

Optional arguments:

- `--max-procs <positive-int>`
- `--dry-run`
- `--split-count <positive-int>`

## Aggregate CSV Tables From Root (Includes Extraction)

This command extracts all discovered runs first (unless `--skip-extract`), then
writes one CSV table per metric.

Command:

```bash
python post-process/split/duration/aggregate_runs_csv.py \
  --root-dir <root-dir>
```

Default output directory:

```text
<root-dir>/split-duration-tables/
```

Generated files:

- `duration_s.csv`
- `turn_count.csv`
- `prompt_tokens.csv`
- `decode_tokens.csv`
- `cached_prompt_tokens.csv`
- `split-duration-tables-manifest.json`

Optional arguments:

- `--output-dir <dir>`
- `--max-procs <positive-int>`
- `--split-count <positive-int>`
- `--dry-run`
- `--skip-extract`

CSV shape:

- each row = one run (`run_path`)
- each column = one percentile bin (`0-10%`, `10-20%`, ...)
