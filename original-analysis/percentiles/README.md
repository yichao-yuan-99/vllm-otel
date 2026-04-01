This directory selects one recorded trail at each `5, 10, ..., 95` percentile
of context usage for a run.

Context usage follows the same definition used by `original-analysis/split`:

- request length = `prompt_tokens + completion_tokens`
- trail context usage = max request length within that trail

The output is designed to feed directly into
`python -m replayer compile --single-trail ...`.

## Script

- `original-analysis/percentiles/extract_run.py`

## Per-Run Extraction

Input run layout:

- `<run-dir>/gateway-output/`
- per gateway job: `requests/model_inference.jsonl`

Command:

```bash
python original-analysis/percentiles/extract_run.py \
  --run-dir <run-dir>
```

Default output:

```text
<run-dir>/original-analysis/percentiles/context-usage-percentiles.json
```

Optional:

- `--output <json-path>` (only with `--run-dir`)

## Batch Mode

```bash
python original-analysis/percentiles/extract_run.py \
  --root-dir <root-dir>
```

Optional:

- `--max-procs <positive-int>`
- `--dry-run`

## Selection Rule

1. Discover all gateway trails from:
   - `gateway-output/run_*`
   - `gateway-output/profile-*/run_*`
2. Drop unranked trails with no valid request length.
3. Also drop any trail containing a request with `status_code=499`.
   This matches the stricter clean-compatible trail set used by the split flow.
4. Sort ranked trails by context usage ascending.
5. For each percentile `p` in `5, 10, ..., 95`, pick the actual trail at the
   nearest-rank index `ceil(N * p / 100)`.

This means percentiles are low-to-high context usage:

- `5` = one of the smallest-context trails
- `50` = median-ish trail
- `95` = one of the largest-context trails

If there are only a few ranked trails, multiple percentiles may map to the same
trail. That is expected.

## Output Fields

Top-level fields:

- `source_run_dir`
- `source_gateway_output_dir`
- `metric`
- `context_usage_definition`
- `selection_method`
- `percentiles`
- `trail_count_total`
- `trail_count_ranked`
- `trail_count_unranked`
- `trail_count_with_status_499`
- `unranked_mode`
- `unranked_criteria`
- `ranked_trails`
- `unranked_trails`
- `selected_trails`
- `source_trail_names_by_percentile`

Each `selected_trails` row includes:

- `percentile`
- `source_trail_name`
- `gateway_run_id`
- `gateway_profile_id`
- `context_length`
- `trail_duration_s`
- `request_count`
- `requests_with_length`
- `context_rank_ascending`
- `context_rank_descending`

`source_trail_name` matches the value expected by
`replayer compile --single-trail`.

Each `ranked_trails` / `unranked_trails` row also includes `trail_duration_s`
when it can be derived. The extractor uses:

1. `manifest.json.run_end_time - manifest.json.run_start_time`
2. fallback: last `agent_end` minus first `agent_start` in `events/lifecycle.jsonl`

If neither timing source is available, `trail_duration_s` is `null`.

## Example

Extract one run:

```bash
python original-analysis/percentiles/extract_run.py \
  --run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z
```

Compile a replay plan for the 50th-percentile trail:

```bash
RUN_DIR=results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z
PERCENTILES_JSON="$RUN_DIR/original-analysis/percentiles/context-usage-percentiles.json"
TRAIL_NAME="$(jq -r '.source_trail_names_by_percentile[\"50\"]' "$PERCENTILES_JSON")"

python -m replayer compile \
  --job-dir "$RUN_DIR" \
  --port-profile-id 1 \
  --single-trail "$TRAIL_NAME"
```
