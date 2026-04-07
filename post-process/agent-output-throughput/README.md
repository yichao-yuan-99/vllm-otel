# Agent Output Throughput Post-Process

This directory extracts output-token throughput for each agent from
`gateway-output` request logs, plus run-level totals and summary stats across
agents.

The per-agent throughput formula is:

```text
sum(completion_tokens across that agent's requests)
/
sum(total LLM request duration across that agent's requests)
```

The denominator is full LLM request time, not decode-only time.

## Script

- `post-process/agent-output-throughput/extract_run.py`

## Single Run

Command:

```bash
python post-process/agent-output-throughput/extract_run.py \
  --run-dir <run-dir>
```

Supported gateway layouts:

- `gateway-output/run_*/...`
- `gateway-output/profile-*/run_*/...`

Default output:

```text
<run-dir>/post-processed/agent-output-throughput/agent-output-throughput.json
```

For multi-profile runs, the extractor also writes one separate file per profile:

```text
<run-dir>/post-processed/agent-output-throughput/profile-<id>/agent-output-throughput.json
```

Optional arguments:

- `--output <path>`

## Batch Mode

Command:

```bash
python post-process/agent-output-throughput/extract_run.py \
  --root-dir <root-dir>
```

Discovery rule:

- any subdirectory that has a direct `gateway-output/` child is treated as a run
  directory

Optional arguments:

- `--max-procs <positive-int>`
- `--dry-run`

## Output Fields

Top-level fields include:

- `source_run_dir`
- `source_gateway_output_dir`
- `service_failure_detected`
- `service_failure_cutoff_time_utc`
- `agent_count`
- `request_count`
- `requests_with_output_tokens`
- `requests_with_llm_request_duration`
- `requests_with_output_tokens_and_llm_request_duration`
- `output_tokens`
- `completion_tokens`
- `llm_request_duration_s`
- `output_throughput_tokens_per_s`
- `agent_output_throughput_tokens_per_s_summary`
  - `sample_count`
  - `avg`
  - `min`
  - `max`
  - `std`
- `agent_output_throughput_tokens_per_s_histogram`
  - `metric`
  - `bin_size`
  - `sample_count`
  - `bin_count`
  - `min`
  - `max`
  - `bins`
    - each bin includes `bin_start`, `bin_end`, and `count`
- `multi_profile`
- `port_profile_ids`
- `series_keys`
- `series_by_profile`
- `agents`

Each entry in `agents` includes:

- `gateway_run_id`
- `gateway_profile_id`
- `api_token_hash`
- `replay_worker_status` (from `replay/summary.json` when available)
- `replay_completed` (`true` only when `replay_worker_status == "completed"`)
- `request_count`
- `requests_with_output_tokens`
- `requests_with_llm_request_duration`
- `requests_with_output_tokens_and_llm_request_duration`
- `output_tokens`
- `completion_tokens`
- `llm_request_duration_s`
- `output_throughput_tokens_per_s`

## Notes

- `completion_tokens` are used as output tokens
- request duration uses `request_duration_ms`, with fallback to `duration_ms`,
  with final fallback to `request_end_time - request_start_time`
- if a service failure cutoff is detected, requests after the cutoff are ignored
- replay worker status is optional enrichment; when `replay/summary.json` is
  present, agents are matched by `sha256(api_token)`
- the histogram uses contiguous `1.0` token/s bins across the observed agent
  throughput range

## Visualization

You can render figures from the extracted JSON with:

```bash
python post-process/visualization/agent-output-throughput/generate_all_figures.py \
  --run-dir <run-dir>
```
