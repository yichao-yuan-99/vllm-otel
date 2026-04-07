# Sweep QPS Docker Power Clean Gateway Ctx

This workflow is the `gateway_ctx` variant of
`experiments/sweep-qps-docker-power-clean`.

It keeps the same clean split-plan lookup and Zeus power logging behavior, but
the generated runner also enables gateway ctx-aware mode before each replay job
and disables it again after that replay finishes.

Default ctx-aware thresholds:

- `usage_threshold_tokens = 501280`
- `scheduling_threshold_tokens = 474897`

These are the example threshold values used in `gateway_ctx`.

## Requirements

Run the Docker environment with `gateway_ctx`, for example:

```bash
python3 servers/servers-docker/client.py start \
  -m qwen3_coder_30b_fp8 \
  -p 13 \
  -l h100_nvl_gpu3_single \
  --gateway-ctx \
  -b
```

Install local Zeus tools:

```bash
pip install -e ./zeus-power-reader
```

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-docker-power-clean-gateway-ctx/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/dabstep/mini-swe-agent/dabstep-20260306T194929Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.25,0.3,0.35 \
  --time-constraint-s 12600 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 13 \
  --split rest \
  --additional-suffix qwen3_fp8 \
  --power-gpu-indices 3
```

Optional overrides:

- `--ctx-aware-usage-threshold-tokens`
- `--ctx-aware-scheduling-threshold-tokens`

## Run Generated Sweep

```bash
bash experiments/sweep-qps-docker-power-clean-gateway-ctx/generated/<timestamp>/run_replay.sh
```

Override port profile at runtime:

```bash
bash experiments/sweep-qps-docker-power-clean-gateway-ctx/generated/<timestamp>/run_replay.sh 3
```

Optional runtime env overrides for the generated runner:

- `GATEWAY_BASE_URL=http://127.0.0.1:<gateway_port>`
- `CTX_AWARE_USAGE_THRESHOLD_TOKENS=...`
- `CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS=...`
- `CURL_BIN=curl`
- `PYTHON_BIN=python3`

If `GATEWAY_BASE_URL` is not set, the runner resolves `gateway_port` from
`configs/port_profiles.toml` using the selected port profile ID.

## Generated Files

For each QPS slug `qpsX`:

- `generated/<timestamp>/qpsX/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

Default replay output layout:

- `results/replay/sweep-qps-docker-power-clean-gateway-ctx/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/`

Each replay output still includes Zeus power logs under:

- `<replay_output_dir>/power/`
