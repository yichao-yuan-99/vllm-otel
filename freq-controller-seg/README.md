This directory contains a segmented variant of `freq-controller`.

It uses the same gateway polling, zeusd integration, moving-average window, and
JSONL logging flow as the baseline controller, but it adds a second threshold
that controls when the controller is allowed to use the true minimum
frequency.

## Behavior

Compared to the baseline lower/upper bound policy, this controller adds:

- `low_freq_threshold` (`lfth`): context-token threshold that gates access to
  the true minimum frequency
- `low_freq_cap_mhz`: the lowest frequency allowed whenever context usage is at
  or above `low_freq_threshold`

At each control decision:

- if moving average `< low_freq_threshold`, the controller may decrease all the
  way to the real minimum frequency
- if moving average `>= low_freq_threshold`, the effective minimum frequency is
  `low_freq_cap_mhz`
- if moving average is below the regular lower bound but still above the low
  threshold, it may decrease only down to `low_freq_cap_mhz`
- if the current frequency is already below `low_freq_cap_mhz` and moving
  average rises back to or above `low_freq_threshold`, it increases one step at
  a time until it reaches `low_freq_cap_mhz`
- if moving average `> upper bound`, it increases one level like the baseline
  controller
- otherwise it holds

The segmented controller keeps the log filename prefix as `freq-controller`.
Segmented experiment flows now write those logs under `freq-control-seg/`, and
the shared freq-control post-process/visualization scripts auto-detect that
layout.

## Config

Default shared settings live in
[script-shared.toml](/srv/scratch/yichaoy2/work/vllm-otel/freq-controller-seg/script-shared.toml):

- `frequency_mhz_levels`
- `control_interval_s = 5`
- `context_query_hz = 5`
- `low_freq_threshold = 6000`
- `low_freq_cap_mhz = 900`

Required segmented policy fields:

- `target_context_usage_lower_bound`
- `target_context_usage_upper_bound`

`low_freq_cap_mhz` must be one of the configured frequency levels, and
`low_freq_threshold` must be less than or equal to the regular lower bound.

Optional aliases accepted in TOML:

- `lfth` for `low_freq_threshold`
- `low_freq_cap` for `low_freq_cap_mhz`

## Usage

Install locally:

```bash
python3 -m pip install -e './freq-controller-seg'
```

Run directly from CLI arguments:

```bash
freq-controller-seg \
  --log-dir ./logs \
  --lower-bound 12000 \
  --upper-bound 22000 \
  --port-profile-id 0 \
  --gpu-index 0
```

Example TOML:

```toml
schema_version = 1

[controller]
target_context_usage_lower_bound = 12000
target_context_usage_upper_bound = 22000
low_freq_threshold = 6000
low_freq_cap_mhz = 900
```

Optional segmented-policy override from CLI:

```bash
freq-controller-seg \
  --log-dir ./logs \
  --lower-bound 12000 \
  --upper-bound 22000 \
  --low-freq-threshold 8000 \
  --low-freq-cap 1005 \
  --port-profile-id 0 \
  --gpu-index 0
```

Optional config override:

```bash
freq-controller-seg \
  --config freq-controller-seg/config.toml \
  --log-dir ./logs \
  --port-profile-id 0 \
  --gpu-index 0
```

Logs are written as:

```text
<log-dir>/freq-controller.query.<timestamp>.jsonl
<log-dir>/freq-controller.decision.<timestamp>.jsonl
```

The decision log includes the baseline fields plus:

- `low_freq_threshold`
- `low_freq_cap_mhz`
- `effective_min_frequency_mhz`

## Tests

```bash
python3 -m pip install -e './freq-controller-seg[dev]'
pytest freq-controller-seg/tests -q
```
