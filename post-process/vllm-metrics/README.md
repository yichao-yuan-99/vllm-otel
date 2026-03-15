Contains notes for reading `vllm-log/` artifacts from `con-driver` and `replayer`.

Shared parsing helpers live under:

- `post-process/vllm-metrics/common/`

The raw metrics parser is:

- `post-process/vllm-metrics/common/parse_metrics.py`
- function: `parse_metric_content_to_json(metric_content: str)`
- implementation uses `prometheus_client.parser.text_string_to_metric_families`

User-facing wrapper script:

- `post-process/vllm-metrics/parse_record.py`
- reads a raw scrape-record JSON with top-level `content`
- writes parsed JSON to stdout or `--output`

Run-level extractor script:

- `post-process/vllm-metrics/extract_run.py`
- reads `<run-dir>/vllm-log/`
- supports single-profile and cluster-mode layouts
- extracts only `vllm*` `gauge` and `counter` series
- writes `<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.json`
- for counters, stores deltas `v_t - v_{t-1}` with first value fixed to `0.0`

Timeseries summary script:

- `post-process/vllm-metrics/summarize_timeseries.py`
- reads `<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.json`
- computes `min`, `max`, `avg` for each metric series
- writes `<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.stats.json`

## Usage

Typical workflow:

1. start from a run directory that already contains `vllm-log/`
2. optionally inspect or parse one raw scrape record
3. extract run-level gauge/counter timeseries for downstream plotting
4. compute run-level min/max/avg summary from extracted timeseries

### Parse One Raw Record

Input:

- one JSON record with a top-level `content` field

Command:

```bash
python post-process/vllm-metrics/parse_record.py \
  --input post-process/vllm-metrics/tests/example.json \
  --output post-process/vllm-metrics/tests/example.parsed.json
```

Behavior:

- reads the raw Prometheus text from `content`
- keeps only `vllm*` metric families
- writes structured JSON to `--output`, or prints to stdout if `--output` is omitted

### Extract One Run

Input:

- Single-profile mode:
  - `<run-dir>/vllm-log/blocks.index.json` (optional)
  - `<run-dir>/vllm-log/block-*.tar.gz`
- Cluster mode:
  - `<run-dir>/vllm-log/profile-*/blocks.index.json` (optional)
  - `<run-dir>/vllm-log/profile-*/block-*.tar.gz`

Command:

```bash
python post-process/vllm-metrics/extract_run.py \
  --run-dir tests/output/con-driver/job-20260301T014756Z
```

The extractor shows a live progress bar over `vllm-log` blocks while it reads
the run.

Default output:

```text
<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.json
```

Custom output:

```bash
python post-process/vllm-metrics/extract_run.py \
  --run-dir tests/output/con-driver/job-20260301T014756Z \
  --output tmp/gauge-counter-timeseries.json
```

Behavior:

- scans all recorded vLLM metric blocks for the run
- parses each raw scrape record
- keeps only `gauge` and `counter` families whose names start with `vllm`
- excludes metric names containing `created` (timestamp-style helper series)
- converts counters to per-scrape deltas
- writes one consolidated timeseries JSON file for visualization

### Extract All Runs Under One Root Directory

When you have many replay/con-driver outputs under one root path, use recursive
discovery mode:

```bash
python post-process/vllm-metrics/extract_run.py \
  --root-dir tests/output/con-driver
```

Discovery rule:

- any subdirectory that has a direct `vllm-log/` child is treated as a run directory
- `post-processed/vllm-log/` is ignored to avoid re-processing output directories

Optional worker count:

```bash
python post-process/vllm-metrics/extract_run.py \
  --root-dir tests/output/con-driver \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/vllm-metrics/extract_run.py \
  --root-dir tests/output/con-driver \
  --dry-run
```

### Summarize Extracted Timeseries

Input:

- `<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.json`

Command:

```bash
python post-process/vllm-metrics/summarize_timeseries.py \
  --run-dir tests/output/con-driver/job-20260301T014756Z
```

Default output:

```text
<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.stats.json
```

Custom input/output:

```bash
python post-process/vllm-metrics/summarize_timeseries.py \
  --run-dir tests/output/con-driver/job-20260301T014756Z \
  --input tmp/gauge-counter-timeseries.json \
  --output tmp/gauge-counter-timeseries.stats.json
```

Behavior:

- reads each metric series from `gauge-counter-timeseries.json`
- computes `min`, `max`, `avg` and `sample_count` for that series
- writes one summary JSON file next to the extracted timeseries by default

Batch summarize from a root directory:

```bash
python post-process/vllm-metrics/summarize_timeseries.py \
  --root-dir tests/output/con-driver
```

Discovery rule:

- any subdirectory that has `post-processed/vllm-log/gauge-counter-timeseries.json`

Optional worker count:

```bash
python post-process/vllm-metrics/summarize_timeseries.py \
  --root-dir tests/output/con-driver \
  --max-procs 8
```

Optional dry-run:

```bash
python post-process/vllm-metrics/summarize_timeseries.py \
  --root-dir tests/output/con-driver \
  --dry-run
```

### Downstream Step

After extraction, the normal next step is visualization:

```bash
python post-process/visualization/vllm-metrics/generate_all_figures.py \
  --run-dir tests/output/con-driver/job-20260301T014756Z
```

That consumes:

- `<run-dir>/post-processed/vllm-log/gauge-counter-timeseries.json`

## Layout

Each run that enables vLLM logging writes:

- `vllm-log/blocks.index.json`
- `vllm-log/block-000000.tar.gz`
- `vllm-log/block-000001.tar.gz`
- ...

In cluster mode, each profile has its own subdirectory:

- `vllm-log/profile-0/block-000000.tar.gz`
- `vllm-log/profile-1/block-000000.tar.gz`
- ...

When cluster mode is detected, post-process adds `port_profile_id` to metric
labels so timeseries from different profiles do not collide.

Each `.tar.gz` contains exactly one member:

- `block-000000.tar.gz` -> `block-000000.jsonl`

## JSONL Record Format

Each line in the JSONL file is one raw scrape of the vLLM `/metrics` endpoint.

The record shape is:

```json
{
  "timestamp": 1772330000,
  "captured_at": "2026-03-01T02:00:00.000000+00:00",
  "content": "# HELP ...\n# TYPE ...\nvllm:num_requests_running{...} 5.0\n..."
}
```

Fields:

- `timestamp`: unix timestamp in seconds when the scrape record was written
- `captured_at`: ISO-8601 UTC timestamp for the scrape
- `content`: exact plain-text body returned by the metrics endpoint

There is no parsed `families` structure anymore. The raw Prometheus text is stored
directly.

When you want a parsed JSON structure for analysis, call
`parse_metric_content_to_json(...)` from `common/parse_metrics.py` on the raw
`content` string. This helper keeps only `vllm*` metric families.

Example: parse `tests/example.json` into a structured JSON file:

```bash
python post-process/vllm-metrics/parse_record.py \
  --input post-process/vllm-metrics/tests/example.json \
  --output post-process/vllm-metrics/tests/example.parsed.json
```

Example: extract gauge/counter timeseries from a run directory:

```bash
python post-process/vllm-metrics/extract_run.py \
  --run-dir tests/output/con-driver/job-20260301T014756Z
```

Output shape:

```json
{
  "source_run_dir": "...",
  "source_vllm_log_dir": "...",
  "cluster_mode": true,
  "port_profile_ids": [0, 1, 2, 3, 4],
  "first_captured_at": "...",
  "metric_count": 123,
  "metrics": {
    "vllm:num_requests_running|engine=0|model_name=...": {
      "name": "vllm:num_requests_running",
      "sample_name": "vllm:num_requests_running",
      "family": "vllm:num_requests_running",
      "type": "gauge",
      "help": "...",
      "labels": {
        "engine": "0",
        "model_name": "..."
      },
      "captured_at": ["..."],
      "value": [0.0, 1.0, 2.0],
      "time_from_start_s": [0.0, 1.0, 2.0]
    }
  }
}
```

## Block Index

`blocks.index.json` stores run-level metadata:

- endpoint
- interval / timeout
- block size
- total records
- per-block filenames and timestamps

## Quick Inspection

List archive members:

```bash
tar -tzf tests/output/con-driver/job-*/vllm-log/block-000000.tar.gz
```

Print the first few raw scrape records:

```bash
tar -xOzf tests/output/con-driver/job-*/vllm-log/block-000000.tar.gz | head
```

## Example Metric Table

Generated from:

- `post-process/vllm-metrics/tests/example.json`

Sorted by metric `type`.

| metric | type | help |
| --- | --- | --- |
| `vllm:external_prefix_cache_hits` | `counter` | External prefix cache hits from KV connector cross-instance cache sharing, in terms of number of cached tokens. |
| `vllm:external_prefix_cache_queries` | `counter` | External prefix cache queries from KV connector cross-instance cache sharing, in terms of number of queried tokens. |
| `vllm:generation_tokens` | `counter` | Number of generation tokens processed. |
| `vllm:mm_cache_hits` | `counter` | Multi-modal cache hits, in terms of number of cached items. |
| `vllm:mm_cache_queries` | `counter` | Multi-modal cache queries, in terms of number of queried items. |
| `vllm:num_preemptions` | `counter` | Cumulative number of preemption from the engine. |
| `vllm:prefix_cache_hits` | `counter` | Prefix cache hits, in terms of number of cached tokens. |
| `vllm:prefix_cache_queries` | `counter` | Prefix cache queries, in terms of number of queried tokens. |
| `vllm:prompt_tokens` | `counter` | Number of prefill tokens processed. |
| `vllm:request_success` | `counter` | Count of successfully processed requests. |
| `vllm:cache_config_info` | `gauge` | Information of the LLMEngine CacheConfig |
| `vllm:e2e_request_latency_seconds_created` | `gauge` | Histogram of e2e request latency in seconds. |
| `vllm:engine_sleep_state` | `gauge` | Engine sleep state; awake = 0 means engine is sleeping; awake = 1 means engine is awake; weights_offloaded = 1 means sleep level 1; discard_all = 1 means sleep level 2. |
| `vllm:external_prefix_cache_hits_created` | `gauge` | External prefix cache hits from KV connector cross-instance cache sharing, in terms of number of cached tokens. |
| `vllm:external_prefix_cache_queries_created` | `gauge` | External prefix cache queries from KV connector cross-instance cache sharing, in terms of number of queried tokens. |
| `vllm:generation_tokens_created` | `gauge` | Number of generation tokens processed. |
| `vllm:inter_token_latency_seconds_created` | `gauge` | Histogram of inter-token latency in seconds. |
| `vllm:iteration_tokens_total_created` | `gauge` | Histogram of number of tokens per engine_step. |
| `vllm:kv_cache_usage_perc` | `gauge` | KV-cache usage. 1 means 100 percent usage. |
| `vllm:mm_cache_hits_created` | `gauge` | Multi-modal cache hits, in terms of number of cached items. |
| `vllm:mm_cache_queries_created` | `gauge` | Multi-modal cache queries, in terms of number of queried items. |
| `vllm:num_preemptions_created` | `gauge` | Cumulative number of preemption from the engine. |
| `vllm:num_requests_running` | `gauge` | Number of requests in model execution batches. |
| `vllm:num_requests_waiting` | `gauge` | Number of requests waiting to be processed. |
| `vllm:prefix_cache_hits_created` | `gauge` | Prefix cache hits, in terms of number of cached tokens. |
| `vllm:prefix_cache_queries_created` | `gauge` | Prefix cache queries, in terms of number of queried tokens. |
| `vllm:prompt_tokens_created` | `gauge` | Number of prefill tokens processed. |
| `vllm:request_decode_time_seconds_created` | `gauge` | Histogram of time spent in DECODE phase for request. |
| `vllm:request_generation_tokens_created` | `gauge` | Number of generation tokens processed. |
| `vllm:request_inference_time_seconds_created` | `gauge` | Histogram of time spent in RUNNING phase for request. |
| `vllm:request_max_num_generation_tokens_created` | `gauge` | Histogram of maximum number of requested generation tokens. |
| `vllm:request_params_max_tokens_created` | `gauge` | Histogram of the max_tokens request parameter. |
| `vllm:request_params_n_created` | `gauge` | Histogram of the n request parameter. |
| `vllm:request_prefill_kv_computed_tokens_created` | `gauge` | Histogram of new KV tokens computed during prefill (excluding cached tokens). |
| `vllm:request_prefill_time_seconds_created` | `gauge` | Histogram of time spent in PREFILL phase for request. |
| `vllm:request_prompt_tokens_created` | `gauge` | Number of prefill tokens processed. |
| `vllm:request_queue_time_seconds_created` | `gauge` | Histogram of time spent in WAITING phase for request. |
| `vllm:request_success_created` | `gauge` | Count of successfully processed requests. |
| `vllm:request_time_per_output_token_seconds_created` | `gauge` | Histogram of time_per_output_token_seconds per request. |
| `vllm:time_to_first_token_seconds_created` | `gauge` | Histogram of time to first token in seconds. |
| `vllm:e2e_request_latency_seconds` | `histogram` | Histogram of e2e request latency in seconds. |
| `vllm:inter_token_latency_seconds` | `histogram` | Histogram of inter-token latency in seconds. |
| `vllm:iteration_tokens_total` | `histogram` | Histogram of number of tokens per engine_step. |
| `vllm:request_decode_time_seconds` | `histogram` | Histogram of time spent in DECODE phase for request. |
| `vllm:request_generation_tokens` | `histogram` | Number of generation tokens processed. |
| `vllm:request_inference_time_seconds` | `histogram` | Histogram of time spent in RUNNING phase for request. |
| `vllm:request_max_num_generation_tokens` | `histogram` | Histogram of maximum number of requested generation tokens. |
| `vllm:request_params_max_tokens` | `histogram` | Histogram of the max_tokens request parameter. |
| `vllm:request_params_n` | `histogram` | Histogram of the n request parameter. |
| `vllm:request_prefill_kv_computed_tokens` | `histogram` | Histogram of new KV tokens computed during prefill (excluding cached tokens). |
| `vllm:request_prefill_time_seconds` | `histogram` | Histogram of time spent in PREFILL phase for request. |
| `vllm:request_prompt_tokens` | `histogram` | Number of prefill tokens processed. |
| `vllm:request_queue_time_seconds` | `histogram` | Histogram of time spent in WAITING phase for request. |
| `vllm:request_time_per_output_token_seconds` | `histogram` | Histogram of time_per_output_token_seconds per request. |
| `vllm:time_to_first_token_seconds` | `histogram` | Histogram of time to first token in seconds. |
