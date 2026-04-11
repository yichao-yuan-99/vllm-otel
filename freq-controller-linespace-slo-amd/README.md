# Freq Controller Linespace SLO AMD

This directory contains an AMD linespace variant with output-throughput SLO
precedence.

It keeps the AMD-specific max-`sclk` control path and multi-GPU targeting from
`freq-controller-linespace-amd`, and adds the second gateway
`/ipc/output-throughput` control signal from `freq-controller-linespace-slo`.

## Behavior

- The controller samples gateway context usage at `context_query_hz` and keeps
  a moving average over the last `control_interval_s`.
- It also samples gateway output-throughput summary at the same rate and keeps
  a moving average of `min_output_tokens_per_s`.
- The output-throughput SLO target is configured as
  `target_output_throughput_tokens_per_s`.
- At each control decision:
  - if moving-average `min_output_tokens_per_s` is below the target, the
    controller forces a one-step frequency increase by default
  - with `--aggresive`, an SLO violation jumps directly to the highest
    configured frequency for that decision
  - otherwise it follows the AMD linespace context-usage policy
- The controller only updates AMD `sclk max`; it does not change `sclk min`.
- `--gpu-index` accepts one GPU id or a comma-separated GPU list, and the same
  max-cap is applied to each selected GPU.
- On shutdown, the controller resets `sclk max` back to the configured reset
  frequency, or to the highest configured frequency if no explicit reset value
  is provided.

If gateway has no throughput samples yet (`throughput_agent_count = 0`, summary
values are `null`), the controller skips the SLO override until the throughput
window has data. Temporary gateway read failures reuse the last successful
snapshot for either signal when available.

## Linespace Policy

- if moving-average context usage `>= target_context_usage_threshold`, the
  controller selects the maximum configured frequency
- if moving-average context usage `< target_context_usage_threshold`, the
  interval `[0, target_context_usage_threshold)` is split into `N - 1` equal
  segments, where `N` is the number of configured frequency levels
- each segment maps directly to one non-maximum frequency level, from the
  minimum configured max-cap at the lowest segment up to the second-highest cap
  just below the threshold

This package keeps the AMD log filename prefix `freq-controller-ls-amd` and
adds one extra SLO-only decision log.

## Config

Default shared settings live in
[script-shared.toml](/home1/yichaoy/vllm-otel/freq-controller-linespace-slo-amd/script-shared.toml):

- `frequency_mhz_levels`
- `control_interval_s = 5`
- `context_query_hz = 5`
- `target_context_usage_threshold = 1141416`

Required fields:

- `target_context_usage_threshold`
- `target_output_throughput_tokens_per_s`

Optional aliases accepted in TOML:

- `target_context_tokens_threshold`
- `target_context_threshold`
- `threshold`
- `target_output_tokens_per_s`
- `target_output_throughput`
- `throughput_target`

AMD target selection stays on the CLI:

- `--port-profile-id`
- `--gpu-index`
- `--aggresive`

`amd.gpu_index` and `amd.gpu_indices` are intentionally rejected in TOML so the
target GPU set is always explicit at launch time.

Optional AMD table fields:

- `amd.command_path`
  Defaults to `amd-set-gpu-core-freq`
- `amd.script_path`
  Optional pass-through override for the wrapper's underlying shell script
- `amd.reset_max_frequency_mhz`
  Defaults to the highest configured frequency level

## Usage

Install locally:

```bash
python3 -m pip install -e './freq-controller-linespace-slo-amd'
```

Run directly from CLI arguments:

```bash
freq-controller-linespace-slo-amd \
  --log-dir ./logs \
  --threshold 1141416 \
  --throughput-target 12 \
  --aggresive \
  --port-profile-id 0 \
  --gpu-index 0
```

To apply the same cap to multiple GPUs, pass a comma-separated list:

```bash
freq-controller-linespace-slo-amd \
  --log-dir ./logs \
  --threshold 1141416 \
  --throughput-target 12 \
  --port-profile-id 2 \
  --gpu-index 2,3
```

Optional explicit gateway IPC override:

```bash
freq-controller-linespace-slo-amd \
  --log-dir ./logs \
  --threshold 1141416 \
  --throughput-target 12 \
  --port-profile-id 0 \
  --gateway-ipc-socket-path /tmp/vllm-gateway-ctx-profile-0.sock \
  --gpu-index 0
```

Example TOML:

```toml
schema_version = 1

[controller]
frequency_mhz_levels = [500, 800, 1700]
target_context_usage_threshold = 1141416
target_output_throughput_tokens_per_s = 12

[amd]
reset_max_frequency_mhz = 1700
```

## Logs

Logs are written as:

```text
<log-dir>/freq-controller-ls-amd.query.<timestamp>.jsonl
<log-dir>/freq-controller-ls-amd.decision.<timestamp>.jsonl
<log-dir>/freq-controller-ls-amd.control-error.<timestamp>.jsonl
<log-dir>/freq-controller-ls-amd.slo-decision.<timestamp>.jsonl
```

The query log includes the AMD context fields plus:

- `throughput_agent_count`
- `min_output_tokens_per_s`
- `max_output_tokens_per_s`
- `avg_output_tokens_per_s`
- `throughput_sample_count_window`

The decision log includes the AMD linespace fields plus:

- `decision_policy`
- `slo_override_applied`
- `window_min_output_tokens_per_s`
- `throughput_sample_count`
- `target_output_throughput_tokens_per_s`
- `context_target_frequency_index`

The SLO-decision log contains only decisions where the throughput SLO path took
precedence.

## AMD Control Path

Each frequency update is applied through `amd-set-gpu-core-freq` with only the
max limit:

```text
amd-set-gpu-core-freq --gpu-index <gpu_index> --max-mhz <mhz>
```

With a multi-GPU `--gpu-index` value such as `2,3`, the controller runs that
command once per selected GPU.

## Tests

```bash
python3 -m pip install -e './freq-controller-linespace-slo-amd[dev]'
pytest freq-controller-linespace-slo-amd/tests -q
```
