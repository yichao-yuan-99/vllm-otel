This directory contains an AMD linespace variant of `freq-controller-linespace`.

It keeps the same gateway polling, moving-average window, linespace policy, and
JSONL logging flow, but swaps the GPU control path to AMD and only changes the
maximum `sclk` limit. The minimum frequency is left untouched.

## Behavior

- if moving average `>= target_context_usage_threshold`, the controller selects
  the maximum configured frequency
- if moving average `< target_context_usage_threshold`, the interval
  `[0, target_context_usage_threshold)` is split into `N - 1` equal segments,
  where `N` is the number of configured frequency levels
- each segment maps directly to one non-maximum frequency level, from the
  minimum configured max-cap at the lowest segment up to the second-highest
  cap just below the threshold
- on each control decision, the controller snaps directly to that segment's
  target frequency by updating only `sclk max`
- `--gpu-index` accepts either one GPU id or a comma-separated GPU list, and
  the selected max-cap is applied to every GPU in that target set
- on shutdown, the controller resets `sclk max` back to the configured reset
  frequency, or to the highest configured frequency if no explicit reset value
  is provided

The AMD controller uses the log filename prefix `freq-controller-ls-amd`.

## Config

Default shared settings live in
[script-shared.toml](/work1/talati/yichaoy/vllm-otel/freq-controller-linespace-amd/script-shared.toml):

- `frequency_mhz_levels`
- `control_interval_s = 5`
- `context_query_hz = 5`
- `target_context_usage_threshold = 1141416`

Required controller fields:

- `frequency_mhz_levels`
- `target_context_usage_threshold`

Optional aliases accepted in TOML:

- `target_context_tokens_threshold`
- `target_context_threshold`
- `threshold`

AMD target selection stays on the CLI:

- `--port-profile-id`
- `--gpu-index`

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
python3 -m pip install -e './freq-controller-linespace-amd'
```

Run directly from CLI arguments:

```bash
freq-controller-linespace-amd \
  --log-dir ./logs \
  --threshold 1141416 \
  --port-profile-id 0 \
  --gpu-index 1
```

To apply the same cap to multiple GPUs, pass a comma-separated list:

```bash
freq-controller-linespace-amd \
  --log-dir ./logs \
  --threshold 1141416 \
  --port-profile-id 2 \
  --gpu-index 2,3
```

Example TOML:

```toml
schema_version = 1

[controller]
frequency_mhz_levels = [500, 800, 1700]
target_context_usage_threshold = 1141416

[amd]
reset_max_frequency_mhz = 1700
```

Optional config override:

```bash
freq-controller-linespace-amd \
  --config freq-controller-linespace-amd/config.toml \
  --log-dir ./logs \
  --port-profile-id 0 \
  --gpu-index 1
```

Logs are written as:

```text
<log-dir>/freq-controller-ls-amd.query.<timestamp>.jsonl
<log-dir>/freq-controller-ls-amd.decision.<timestamp>.jsonl
<log-dir>/freq-controller-ls-amd.control-error.<timestamp>.jsonl
```

The decision log includes the baseline linespace fields plus the chosen target
frequency index and `gpu_indices`. The control-error log records max-frequency
write failures and lets the controller continue running.

## AMD Control Path

Each frequency update is applied through `amd-set-gpu-core-freq` with only the
max limit:

```text
amd-set-gpu-core-freq --gpu-index <gpu_index> --max-mhz <mhz>
```

With a multi-GPU `--gpu-index` value such as `2,3`, the controller runs that
command once per selected GPU. This controller does not change `sclk min`.

## Tests

```bash
python3 -m pip install -e './freq-controller-linespace-amd[dev]'
pytest freq-controller-linespace-amd/tests -q
```
