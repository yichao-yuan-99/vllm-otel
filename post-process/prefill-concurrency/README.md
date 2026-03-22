This directory contains extraction scripts for prefill-phase concurrency.

It is similar to `job-concurrency`, but uses each request's prefill window
instead of each job's full runtime window.

The extraction logic:

1. load `post-processed/gateway/llm-requests/llm-requests.json`
2. build per-request prefill activity ranges (`prefill_start_offset_s`, `prefill_end_offset_s`)
3. use 10ms ticks (`tick_ms=10` by default) across the full run duration
4. accumulate active prefill requests per tick to produce a concurrency series
5. summarize min/max/avg prefill concurrency
6. summarize contiguous interval lengths for each concurrency value (`0`, `1`, `2`, ...)

Interval summary details:

- `0` means no prefill is active at that tick
- contiguous ticks with the same value are treated as one interval
- for each value `x`, stats include:
  - `interval_count`
  - `avg/min/max/std` interval length in ticks
  - `avg/min/max/std` interval length in seconds

These are written to:

- `prefill-concurrency-stats.json` under `concurrency_interval_length_stats`

## Script

- `post-process/prefill-concurrency/extract_run.py`

## Single Run

Default input:

- `<run-dir>/post-processed/gateway/llm-requests/llm-requests.json`

Command:

```bash
python post-process/prefill-concurrency/extract_run.py \
  --run-dir <run-dir>
```

Default output directory:

```text
<run-dir>/post-processed/prefill-concurrency/
```

Generated files:

- `prefill-activities.json`
- `prefill-concurrency-timeseries.json`
- `prefill-concurrency-stats.json`

Optional parameters (single-run mode):

- `--llm-requests <path>`
- `--output-dir <path>`
- `--tick-ms <positive-int>`

## Batch Processing

When you have many runs under one root directory:

```bash
python post-process/prefill-concurrency/extract_run.py \
  --root-dir results/replay
```

Discovery rule:

- any subdirectory with:
  - `post-processed/gateway/llm-requests/llm-requests.json`

Optional worker count:

```bash
python post-process/prefill-concurrency/extract_run.py \
  --root-dir results/replay \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/prefill-concurrency/extract_run.py \
  --root-dir results/replay \
  --dry-run
```
