# Sweep QPS Docker Power Clean Gateway Multi Lowest Usage

This workflow is the `gateway_multi` + `lowest_usage` variant of
`experiments/sweep-qps-docker-power-clean-gateway_multi`.

It keeps the same clean split-plan lookup, multi-profile replay, and Zeus power
logging flow, but the generated runner also sets the public `gateway_multi`
assignment policy to `lowest_usage` before the sweep starts.

The generated bundle accepts a `gateway_multi`-style port-profile selection:

- pass `--port-profile` as a comma-separated list like `13,14`
- or repeat it, for example `--port-profile 13 --port-profile 14`
- the first selected port profile becomes the control/public profile used by
  `replayer replay`
- that same control/public profile is used to resolve the `gateway_multi`
  `/policy` endpoint unless `GATEWAY_BASE_URL` is provided
- `--power-gpu-indices` should include the full backend GPU list, for example
  `2,3`

## Requirements

Run the Docker environment with `servers-docker-multi`, for example:

```bash
python3 servers/servers-docker-multi/client.py start \
  -m qwen3_coder_30b_fp8 \
  -p 13,14 \
  -l h100_nvl_gpu23 \
  -b
```

Install local Zeus tools:

```bash
pip install -e ./zeus-power-reader
```

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-docker-power-clean-gateway_multi-lowest_usage/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.15 \
  --time-constraint-s 1800 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 2,13 \
  --split exclude-unranked \
  --additional-suffix qwen3_fp8 \
  --power-gpu-indices 2,3
```

`--split` values are `top`, `rest`, or `exclude-unranked`. `full` is accepted
as a compatibility alias for `exclude-unranked`.

## Run Generated Sweep

```bash
bash experiments/sweep-qps-docker-power-clean-gateway_multi-lowest_usage/generated/<timestamp>/run_replay.sh
```

Override the gateway_multi port-profile selection at runtime:

```bash
bash experiments/sweep-qps-docker-power-clean-gateway_multi-lowest_usage/generated/<timestamp>/run_replay.sh 13,14
```

Optional runtime env overrides for the generated runner:

- `PORT_PROFILE_IDS=13,14`
- `GATEWAY_BASE_URL=http://127.0.0.1:<gateway_port>`
- `CURL_BIN=curl`
- `ZEUS_POWER_READER_BIN=zeus-power-reader`
- `PYTHON_BIN=python3`

If `GATEWAY_BASE_URL` is not set, the runner resolves the control gateway port
from `configs/port_profiles.toml` using the first selected port profile ID,
then sends:

```json
{"assignment_policy": "lowest_usage"}
```

to `POST /policy` before the replay sweep starts.

If you override the port-profile list at runtime, make sure the generated
bundle's `--power-gpu-indices` still matches the GPUs used by that
`gateway_multi` selection. Otherwise, regenerate the bundle with the correct GPU
list.

## Generated Files

For each QPS slug `qpsX`:

- `generated/<timestamp>/qpsX/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<selected-plan>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

Default replay output layout:

- `results/replay/sweep-qps-docker-power-clean-gateway_multi-lowest_usage/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/`

For multi-profile runs, replay now splits vLLM metrics by backend profile:

- `<replay_output_dir>/vllm-log/profile-<port-profile-id>/`

Each replay output still includes Zeus power logs under:

- `<replay_output_dir>/power/`

## Tests

```bash
.venv/bin/pytest experiments/sweep-qps-docker-power-clean-gateway_multi-lowest_usage/test -q
```
