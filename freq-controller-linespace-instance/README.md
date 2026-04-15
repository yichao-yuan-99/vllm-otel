This directory contains an instance variant of `freq-controller-linespace`.

It uses the same gateway polling, zeusd integration, moving-average window,
and JSONL logging flow as the linespace controller, but packages the
direct-jump policy as its own standalone controller.

## Behavior

- if moving average `>= target_context_usage_threshold`, the controller selects
  the maximum configured frequency
- if moving average `< target_context_usage_threshold`, the interval
  `[0, target_context_usage_threshold)` is split into `N - 1` equal segments,
  where `N` is the number of configured frequency levels
- each segment maps directly to one non-maximum frequency level, from the
  minimum frequency at the lowest segment up to the second-highest frequency at
  the segment just below the threshold
- on each control decision, the controller jumps directly to that segment's
  target frequency instead of moving only one level up or down
- `--gpu-index` accepts either one GPU id or a comma-separated GPU list, and
  the selected frequency is applied to every GPU in that target set

Example: if the threshold is `395784` and there are `11` frequency levels,
then the controller creates `10` equal-width context-usage segments below the
threshold. A moving average just below the threshold selects the
second-highest frequency, while `0` selects the minimum frequency.

The instance controller uses the log filename prefix
`freq-controller-ls-instance`.

## Config

Default shared settings live in
[script-shared.toml](/srv/scratch/yichaoy2/work/vllm-otel/freq-controller-linespace-instance/script-shared.toml):

- `frequency_mhz_levels`
- `control_interval_s = 5`
- `context_query_hz = 5`
- `target_context_usage_threshold = 395784`

Required linespace policy field:

- `target_context_usage_threshold`

Optional aliases accepted in TOML:

- `target_context_tokens_threshold`
- `target_context_threshold`
- `threshold`

Like the baseline controller, GPU targeting stays on the CLI:

- `--port-profile-id`
- `--gpu-index`
- `--only-freq-list-geq`

`zeusd.gpu_index` and `zeusd.gpu_indices` are intentionally rejected in TOML so
the target GPU set is always explicit at launch time.

If `--only-freq-list-geq` is provided, the controller filters
`frequency_mhz_levels` to values greater than or equal to that MHz threshold
before computing the initial frequency and linespace segments.

## Usage

Install locally:

```bash
python3 -m pip install -e './freq-controller-linespace-instance'
```

Run directly from CLI arguments:

```bash
freq-controller-linespace-instance \
  --log-dir ./logs \
  --threshold 395784 \
  --only-freq-list-geq 915 \
  --port-profile-id 0 \
  --gpu-index 0
```

To apply the same linespace decision to multiple GPUs, pass a comma-separated
list:

```bash
freq-controller-linespace-instance \
  --log-dir ./logs \
  --threshold 395784 \
  --port-profile-id 2 \
  --gpu-index 2,3
```

Example TOML:

```toml
schema_version = 1

[controller]
target_context_usage_threshold = 395784
```

Optional config override:

```bash
freq-controller-linespace-instance \
  --config freq-controller-linespace-instance/config.toml \
  --log-dir ./logs \
  --port-profile-id 0 \
  --gpu-index 0
```

If `gateway.ipc_socket_path` is not set, the controller auto-selects the
per-profile IPC socket under `/tmp/` and prefers an active `gateway_ctx`
listener when present:

- `/tmp/vllm-gateway-ctx-profile-<port_profile_id>.sock`
- `/tmp/vllm-gateway-profile-<port_profile_id>.sock`

Logs are written as:

```text
<log-dir>/freq-controller-ls-instance.query.<timestamp>.jsonl
<log-dir>/freq-controller-ls-instance.decision.<timestamp>.jsonl
<log-dir>/freq-controller-ls-instance.control-error.<timestamp>.jsonl
```

The decision log includes the baseline fields plus:

- `target_context_usage_threshold`
- `segment_count`
- `segment_width_context_usage`
- `target_frequency_index`
- `gpu_indices`

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
- `gpu_indices`

## Tests

```bash
python3 -m pip install -e './freq-controller-linespace-instance[dev]'
pytest freq-controller-linespace-instance/tests -q
```
