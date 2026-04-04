# Freq Controller Linespace SLO

This directory contains a standalone SLO-aware linespace GPU frequency
controller package.

It keeps the existing linespace context-usage policy, gateway IPC polling,
zeusd integration, moving-average windows, and JSONL logging flow, and adds a
second control signal from gateway `GET /ipc/output-throughput`.

## Behavior

- The controller still samples gateway context usage at `context_query_hz` and
  keeps a moving average over the last `control_interval_s`.
- It also samples gateway output-throughput summary at the same rate and keeps
  a moving average of `min_output_tokens_per_s`.
- The output-throughput SLO target is configured as
  `target_output_throughput_tokens_per_s`.
- At each control decision:
  - if moving-average `min_output_tokens_per_s` is below the target, the
    controller forces a one-step frequency increase
  - otherwise it follows the original linespace context-usage policy
- If gateway has no throughput samples yet
  (`throughput_agent_count = 0`, summary values are `null`), the controller
  skips the SLO override until the throughput window has data.
- Temporary gateway read failures reuse the last successful snapshot for either
  signal when available.

## Linespace Policy

- if moving-average context usage `>= target_context_usage_threshold`, the
  controller selects the maximum configured frequency
- if moving-average context usage `< target_context_usage_threshold`, the
  interval `[0, target_context_usage_threshold)` is split into `N - 1` equal
  segments, where `N` is the number of configured frequency levels
- each segment maps directly to one non-maximum frequency level, from the
  minimum frequency at the lowest segment up to the second-highest frequency at
  the segment just below the threshold

## Config

Default shared settings live in
[script-shared.toml](/srv/scratch/yichaoy2/work/vllm-otel/freq-controller-linespace-slo/script-shared.toml):

- `frequency_mhz_levels`
- `control_interval_s = 5`
- `context_query_hz = 5`
- `target_context_usage_threshold = 395784`

Required SLO field:

- `target_output_throughput_tokens_per_s`

Required linespace field:

- `target_context_usage_threshold`

Optional aliases accepted in TOML:

- `target_context_tokens_threshold`
- `target_context_threshold`
- `threshold`
- `target_output_tokens_per_s`
- `target_output_throughput`
- `throughput_target`

## Usage

Install locally:

```bash
python3 -m pip install -e './freq-controller-linespace-slo'
```

Run directly from CLI arguments:

```bash
freq-controller-linespace-slo \
  --log-dir ./logs \
  --threshold 395784 \
  --throughput-target 12 \
  --port-profile-id 0 \
  --gpu-index 0
```

Example TOML:

```toml
schema_version = 1

[controller]
target_context_usage_threshold = 395784
target_output_throughput_tokens_per_s = 12
```

Optional config override:

```bash
freq-controller-linespace-slo \
  --config freq-controller-linespace-slo/config.toml \
  --log-dir ./logs \
  --port-profile-id 0 \
  --gpu-index 0
```

## Logs

This package keeps the linespace log filename prefix for compatibility:

```text
<log-dir>/freq-controller-ls.query.<timestamp>.jsonl
<log-dir>/freq-controller-ls.decision.<timestamp>.jsonl
<log-dir>/freq-controller-ls.control-error.<timestamp>.jsonl
<log-dir>/freq-controller-ls.slo-decision.<timestamp>.jsonl
```

The query log includes the existing context fields plus:

- `throughput_agent_count`
- `min_output_tokens_per_s`
- `max_output_tokens_per_s`
- `avg_output_tokens_per_s`
- `throughput_sample_count_window`

The decision log includes the existing linespace fields plus:

- `decision_policy`
- `slo_override_applied`
- `window_min_output_tokens_per_s`
- `throughput_sample_count`
- `target_output_throughput_tokens_per_s`
- `context_target_frequency_index`

The SLO-decision log contains only decisions where the throughput SLO path took
precedence. It reuses the same decision fields as the main decision log so it
can be post-processed independently.

The control-error log records zeusd write failures on the control path and lets
the controller continue running. It includes:

- `reason`
- `action`
- `error`
- `attempted_frequency_index`
- `attempted_frequency_mhz`
- `current_frequency_index`
- `current_frequency_mhz`
- `moving_average_context_usage`
- `sample_count`

## Tests

```bash
python3 -m pip install -e './freq-controller-linespace-slo[dev]'
pytest freq-controller-linespace-slo/tests -q
```
