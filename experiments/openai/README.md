# OpenAI Benchmark via Gateway Lite + Con-Driver

This experiment flow runs Harbor benchmarks through `con-driver` while routing model
traffic through `gateway_lite`, with `gateway_lite` forwarding to an OpenAI
endpoint (default: `https://api.openai.com/v1`).

## What This Adds

- `start_gateway.sh`: starts `gateway_lite` on a chosen port profile and points it
  at an OpenAI-compatible upstream.
- `run_dataset.sh`: runs one benchmark dataset through `con-driver` with
  gateway lifecycle tracking (agent IDs only, no key splitting required).
- `run_all.sh`: runs all supported datasets sequentially.

## Supported Datasets

- `terminal-bench@2.0` (`n_task=89`)
- `livecodebench` (`n_task=100`)
- `dabstep` (`n_task=450`)
- `swebench-verified` (`n_task=500`)

## Prerequisites

1. Export your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

2. Use repo root as your working directory.

3. Ensure Harbor access works in your environment (same requirement as other
   `con-driver` runs in this repo).

## Step 1: Start Gateway Lite (OpenAI Upstream)

Run in terminal A:

```bash
bash experiments/openai/start_gateway.sh
```

Defaults:

- `--port-profile-id 0`
- `--upstream-base-url https://api.openai.com/v1`
- `--upstream-api-key` defaults to `OPENAI_API_KEY`
- gateway-lite injects the real upstream key; con-driver uses per-agent IDs only

Optional example:

```bash
bash experiments/openai/start_gateway.sh \
  --port-profile-id 4 \
  --upstream-base-url https://api.openai.com/v1
```

## Step 2: Run One Benchmark Dataset

Run in terminal B:

```bash
bash experiments/openai/run_dataset.sh --dataset dabstep
```

Default run settings:

- agent: `mini-swe-agent`
- model: `openai/gpt-4.1-mini`
- max concurrency: `1`
- gateway control URL resolved from `configs/port_profiles.toml` using `--port-profile-id`
- agent API base exported automatically (`OPENAI_API_BASE`, `HOSTED_VLLM_API_BASE`, `OPENAI_BASE_URL`)
  and loopback hosts are rewritten to `CON_DRIVER_CONTAINER_HOST` (default `192.168.5.1`)
- results root: `experiments/results/openai`

Example with overrides:

```bash
bash experiments/openai/run_dataset.sh \
  --dataset terminal-bench@2.0 \
  --port-profile-id 4 \
  --agent mini-swe-agent \
  --model openai/gpt-4.1-mini \
  --max-concurrent 4
```

Dry run:

```bash
bash experiments/openai/run_dataset.sh --dataset livecodebench --dry-run
```

## Run All Datasets

```bash
bash experiments/openai/run_all.sh
```

Example:

```bash
bash experiments/openai/run_all.sh \
  --port-profile-id 4 \
  --model openai/gpt-4.1-mini \
  --max-concurrent 4
```

## Output Layout

Runs are stored under:

- `experiments/results/openai/<dataset-slug>/<agent>/`

Each invocation creates a run directory with standard `con-driver` artifacts
(`meta/`, `logs/`, `trials/`, `gateway-output/`, etc.).
