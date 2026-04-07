This directory contains a multi-GPU linespace variant of `freq-controller`.

It keeps the same linespace policy as
`[freq-controller-linespace](/srv/scratch/yichaoy2/work/vllm-otel/freq-controller-linespace/README.md)`,
but it applies each selected core clock to a whole GPU group instead of a
single device.

This is intended for a port profile that fronts one tensor-parallel vLLM
server. In that setup, one gateway IPC endpoint represents the whole server,
and `freq-controller-linespace-multi` fans the same clock decision out to the
full GPU list for that server.

## Behavior

- the moving-average context policy is unchanged from the single-GPU
  linespace controller
- one control decision is computed from gateway `total_context_tokens`
- the chosen clock is applied to every GPU listed in `--gpu-indices`
- shutdown resets core clocks for every listed GPU

The multi controller uses the log filename prefix `freq-controller-ls-multi`.

## Config

Default shared settings live in
[script-shared.toml](/srv/scratch/yichaoy2/work/vllm-otel/freq-controller-linespace-multi/script-shared.toml):

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

Like the single-GPU controller, gateway and GPU targeting stay on the CLI:

- `--port-profile-id`
- `--gpu-indices`

`zeusd.gpu_index` and `zeusd.gpu_indices` are intentionally rejected in TOML so
the selected GPU group is always explicit at launch time.

## Usage

Install locally:

```bash
python3 -m pip install -e './freq-controller-linespace-multi'
```

Run directly from CLI arguments:

```bash
freq-controller-linespace-multi \
  --log-dir ./logs \
  --threshold 395784 \
  --port-profile-id 2 \
  --gpu-indices 2 3
```

`--gpu-indices` accepts either space-separated or comma-separated values:

```bash
freq-controller-linespace-multi \
  --log-dir ./logs \
  --threshold 395784 \
  --port-profile-id 2 \
  --gpu-indices 2,3
```

Example TOML:

```toml
schema_version = 1

[controller]
target_context_usage_threshold = 395784
```

Optional config override:

```bash
freq-controller-linespace-multi \
  --config freq-controller-linespace-multi/config.toml \
  --log-dir ./logs \
  --port-profile-id 2 \
  --gpu-indices 2 3
```

Logs are written as:

```text
<log-dir>/freq-controller-ls-multi.query.<timestamp>.jsonl
<log-dir>/freq-controller-ls-multi.decision.<timestamp>.jsonl
<log-dir>/freq-controller-ls-multi.control-error.<timestamp>.jsonl
```

Each log entry includes the active `gpu_indices` so one controller run can be
traced back to the tensor-parallel group it managed.

## Notes

- Run one controller instance per tensor-parallel vLLM server.
- Make sure `--gpu-indices` matches the GPUs actually used by the selected
  `--port-profile-id`.
- On control-path failures, the controller logs the error and keeps running.
  Cleanup still performs a best-effort reset across the full GPU list.

## Tests

```bash
python3 -m pip install -e './freq-controller-linespace-multi[dev]'
pytest freq-controller-linespace-multi/tests -q
```
