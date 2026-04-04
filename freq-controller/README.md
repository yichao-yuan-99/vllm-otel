This directory contains a frequency controller that adjusts one GPU core clock
level from the gateway's live context usage. GPU frequency control is applied
through `zeusd`, following the same underlying Zeus flow used by
`zeus-power-reader`.

## Behavior

- The controller can run directly from CLI arguments without a TOML config file.
- Before applying frequency control, it polls gateway `GET /ipc/context` and
  waits in a pending state until `job_active` becomes `true`.
- It starts at the middle level of the configured frequency list.
- It queries gateway `GET /ipc/context` repeatedly and uses
  `total_context_tokens`.
- If an active gateway read fails temporarily, the controller logs the error
  and reuses the last successful snapshot so control can continue.
- It keeps a moving average over the last control interval.
- At each control decision timestamp:
  - if moving average `< lower bound`, decrease one frequency level
  - if moving average `> upper bound`, increase one frequency level
  - otherwise hold
- Frequency changes are clamped to the lowest and highest configured levels.
- On exit, it resets the GPU core clocks.
- It always writes two JSONL log files under the provided `--log-dir`.

## Config

Default shared settings live in [script-shared.toml](/srv/scratch/yichaoy2/work/vllm-otel/freq-controller/script-shared.toml):

- `frequency_mhz_levels`: defaults are defined in `script-shared.toml`
- `control_interval_s = 5`
- `context_query_hz = 5`

The frequency list is normalized into ascending order before control starts.

Optional config fields:

- `shared.config_path`: path to a shared TOML file
- shared controller fields may also be set directly in the main TOML file,
  either top-level or under `[controller]`
- `target_context_usage_lower_bound` and
  `target_context_usage_upper_bound` may still be set in TOML for backward
  compatibility, but the CLI flags take precedence

Optional gateway fields:

- `gateway.ipc_socket_path`: overrides the derived path
- `gateway.timeout_s`: default `5`

Optional zeusd fields:

- `zeusd.socket_path`: default `/var/run/zeusd.sock`

Runtime arguments:

- `--target-context-usage-lower-bound` or `--lower-bound`: required unless
  configured in TOML
- `--target-context-usage-upper-bound` or `--upper-bound`: required unless
  configured in TOML
- `--port-profile-id`: default `0`
- `--gpu-index`: default `0`

The controller rejects `gateway.port_profile_id` and `zeusd.gpu_index` in TOML.
Those values must be provided as CLI arguments instead.

If `gateway.ipc_socket_path` is not set, the controller uses the same default
gateway IPC naming convention:

- profile `0` -> `/tmp/vllm-gateway-profile-0.sock`
- profile `3` -> `/tmp/vllm-gateway-profile-3.sock`

Example override config:

```toml
schema_version = 1

[shared]
config_path = "./my-shared.toml"
```

Example shared config override:

```toml
schema_version = 1

frequency_mhz_levels = [1785, 1740, 1695, 1650]
control_interval_s = 5
context_query_hz = 5
```

## Usage

Install locally:

```bash
python3 -m pip install -e './freq-controller'
```

Run:

```bash
freq-controller \
  --log-dir ./logs \
  --lower-bound 12000 \
  --upper-bound 22000 \
  --port-profile-id 0 \
  --gpu-index 0
```

Optional config override:

```bash
freq-controller \
  --config freq-controller/config.toml \
  --log-dir ./logs \
  --lower-bound 12000 \
  --upper-bound 22000 \
  --port-profile-id 0 \
  --gpu-index 0
```

`zeusd.socket_path` defaults to `/var/run/zeusd.sock`, so you only need a
`[zeusd]` config section if you want to override that path. If `--config` is
omitted, the controller uses [script-shared.toml](/srv/scratch/yichaoy2/work/vllm-otel/freq-controller/script-shared.toml).

The controller logs are written as:

```text
<log-dir>/freq-controller.query.<timestamp>.jsonl
<log-dir>/freq-controller.decision.<timestamp>.jsonl
```

The query log is written at the gateway polling frequency. It includes the live
`context_usage` value plus `phase`, `job_active`, `agent_count`, and
`sample_count_window` while control is active. Temporary gateway read failures
are recorded in the `error` field; when that happens, the logged values are
reused from the last successful snapshot.

The decision log is written at the control interval. It includes
`window_context_usage`, `current_frequency_mhz`, `target_frequency_mhz`,
`action`, `changed`, `sample_count`, and the configured lower and upper bounds.

## Tests

```bash
python3 -m pip install -e './freq-controller[dev]'
pytest freq-controller/tests -q
```
