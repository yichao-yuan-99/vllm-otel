This directory contains a selector for derived `post-processed/` outputs.

`select_post_processed.py` keeps the last `-x/--percent` of a run's timeline, rewrites supported JSON artifacts into a sibling derived directory, rebases offsets so the selected window starts at `0`, and regenerates visualization outputs from the selected data.

Example:

```bash
python3 post-process-select/select_post_processed.py \
  --run-dir /path/to/run \
  --percent 50
```

Recursive mode:

```bash
python3 post-process-select/select_post_processed.py \
  --root-dir /path/to/results-root \
  --percent 50
```

In `--root-dir` mode the script recursively scans for directories containing:

```text
post-processed/global/trial-timing-summary.json
```

and processes every matching run.

If the original run spans `1:00pm` to `2:00pm`, `--percent 50` keeps data from `1:30pm` to `2:00pm`. A record that originally started at `+35s` from `1:30pm` becomes `+5s` in the derived output.

By default the output is written to:

```text
<run-dir>/post-processed-<percent>
```

Examples:

```text
--percent 50   -> post-processed-50
--percent 12.5 -> post-processed-12_5
```

You can override that with `--output-dir`, and `--overwrite` allows reusing an existing destination directory.

`--output-dir` is only available for single-run mode (`--run-dir` or `--post-processed-dir`). In `--root-dir` mode each run writes to its own default sibling directory.

`--dry-run` lists the discovered run directories without writing outputs, and `--max-procs` controls parallel processing in `--root-dir` mode.

Supported outputs are rewritten in-place to preserve the original post-process schema where possible. Today that includes:

- `service-failure`
- `global`
- `global-progress`
- `job-throughput`
- `job-concurrency`
- `request-throughput`
- `gateway/llm-requests`
- `agent-output-throughput`
- `gateway/usage`
- `prefill-concurrency`
- `split/duration`
- `vllm-log`
- `vllm-metrics`
- `power`
- `power-sampling`
- `gateway/stack`
- `gateway/stack-context`
- `gateway/stack-kv`
- `gateway/ctx-aware-log`
- `gateway/slo-aware-log`
- `freq-control`
- `freq-control-seg`
- `freq-control-linespace`
- `freq-control-linespace-amd`
- `freq-control-linespace-multi`
- `slo-decision`
- `key-stats`

The selector also writes `selection-summary.json` at the output root with the selected time window plus the written and skipped files.

Visualization outputs are regenerated under `visualization/` in the derived directory rather than copied byte-for-byte from the source tree. Supported regenerated figures currently include:

- `job-throughput`
- `request-throughput`
- `agent-output-throughput`
- `job-concurrency`
- `prefill-concurrency`
- `gateway-stack`
- `gateway-stack-context`
- `gateway-stack-kv`
- `stacked-per-agent`
- `gateway-ctx-aware`
- `gateway-slo-aware`
- `vllm-metrics`
- `power`
- `freq-control`
- `slo-decision`
