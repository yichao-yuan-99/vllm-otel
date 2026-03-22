# NVML Power Control Latency Bench

This directory contains a small benchmark script to measure how long it takes
for NVML locked-clock control to show up in observed GPU clock telemetry.

## Script

- `bench-powerctl-latency/measure_freq_lock_latency.py`

## What It Does

1. takes a user-provided GPU index (expected to be idle)
2. reads baseline graphics clock (MHz)
3. applies `nvmlDeviceSetGpuLockedClocks` with default target `1305/1305 MHz`
4. polls clock telemetry at `1000 Hz` by default
5. stops when a clock change is observed or after `5s` timeout
6. resets locked clocks by default (`nvmlDeviceResetGpuLockedClocks`)

## Usage

```bash
python bench-powerctl-latency/measure_freq_lock_latency.py \
  --gpu-index 0
```

Default behavior:

- `--target-min-mhz 1305`
- `--target-max-mhz 1305` (same as min if omitted)
- `--poll-hz 1000`
- `--timeout-s 5`

Common options:

- `--output-json <path>`: save JSON result to file
- `--min-delta-mhz <int>`: required delta from baseline to count as changed

## Example

```bash
python bench-powerctl-latency/measure_freq_lock_latency.py \
  --gpu-index 2 \
  --target-min-mhz 1305 \
  --poll-hz 1000 \
  --timeout-s 5 \
  --output-json /tmp/powerctl-latency-gpu2.json
```

## Output

The script prints a JSON summary including:

- `clock_changed` / `timed_out`
- `baseline_clock_mhz` / `observed_clock_mhz`
- `latency_from_set_start_ms`
- `latency_from_set_return_ms`
- `set_call_latency_ms`
- `sample_count`

Exit codes:

- `0`: clock change observed
- `2`: timed out without observing change
- `1`: error
