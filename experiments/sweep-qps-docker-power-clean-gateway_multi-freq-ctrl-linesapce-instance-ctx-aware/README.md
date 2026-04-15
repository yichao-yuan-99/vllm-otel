# Sweep QPS Docker Power Clean Gateway Multi Freq Ctrl Linesapce Instance Ctx Aware

This workflow is the `gateway_multi` + independent
`freq-controller-linespace-instance` variant of
`experiments/sweep-qps-docker-power-clean-gateway_multi-freq-ctrl-linesapce`.

It keeps the same clean split-plan lookup, multi-profile replay, Zeus power
logging, and gateway ctx-aware flow, but the generated runner also:

- starts one `freq-controller-linespace-instance` process per backend
  profile/GPU pair before each replay job
- maps `--port-profile` and `--power-gpu-indices` positionally, so their counts
  must match
- resets GPU core clocks for every backend GPU after each QPS point
- writes controller logs under
  `<replay_output_dir>/freq-control-linespace-instance/profile-<port-profile-id>/`

Default thresholds:

- `ctx-aware usage_threshold_tokens = 501280`
- `ctx-aware scheduling_threshold_tokens = 474897`
- `freq-controller-linespace-instance threshold = 395784`

The generated bundle accepts a `gateway_multi`-style port-profile selection:

- pass `--port-profile` as a comma-separated list like `0,1,2,3`
- or repeat it, for example
  `--port-profile 0 --port-profile 1 --port-profile 2 --port-profile 3`
- the first selected port profile becomes the control/public profile used by
  `replayer replay`
- that same control/public profile is used to resolve the `gateway_multi`
  `/ctx-aware/*` endpoints unless `GATEWAY_BASE_URL` is provided
- `--power-gpu-indices` must list one backend GPU per selected port profile in
  the same order, for example
  `--port-profile 0,1,2,3 --power-gpu-indices 0,1,2,3`

## Requirements

Run the Docker environment with `servers-docker-multi`, for example:

```bash
python3 servers/servers-docker-multi/client.py start \
  -m qwen3_coder_30b_fp8 \
  -p 13,14 \
  -l h100_nvl_gpu23 \
  -b
```

Install the local console scripts used by the generated runner:

```bash
python3 -m pip install -e './freq-controller-linespace-instance'
python3 -m pip install -e './zeus-power-reader'
```

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-docker-power-clean-gateway_multi-freq-ctrl-linesapce-instance-ctx-aware/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.32 \
  --time-constraint-s 10800 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 0,1,2,3 \
  --split exclude-unranked \
  --additional-suffix qwen3_fp8 \
  --power-gpu-indices 0,1,2,3
```

Optional overrides:

- `--ctx-aware-usage-threshold-tokens`
- `--ctx-aware-scheduling-threshold-tokens`
- `--freq-controller-threshold`
- `--output-suffix`

`--split` values are `top`, `rest`, or `exclude-unranked`. `full` is accepted
as a compatibility alias for `exclude-unranked`.

## Run Generated Sweep

```bash
bash experiments/sweep-qps-docker-power-clean-gateway_multi-freq-ctrl-linesapce-instance-ctx-aware/generated/<timestamp>/run_replay.sh
```

Override the gateway_multi port-profile selection at runtime:

```bash
bash experiments/sweep-qps-docker-power-clean-gateway_multi-freq-ctrl-linesapce-instance-ctx-aware/generated/<timestamp>/run_replay.sh 0,1,2,3
```

Optional runtime env overrides for the generated runner:

- `PORT_PROFILE_IDS=0,1,2,3`
- `GATEWAY_BASE_URL=http://127.0.0.1:<gateway_port>`
- `CTX_AWARE_USAGE_THRESHOLD_TOKENS=...`
- `CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS=...`
- `FREQ_CONTROLLER_CONFIG=/path/to/freq-controller-linespace-instance.toml`
- `FREQ_CONTROLLER_THRESHOLD=...`
- `FREQ_CONTROLLER_BIN=freq-controller-linespace-instance`
- `RESET_GPU_CORE_FREQ_BIN=reset-gpu-core-freq`
- `CURL_BIN=curl`
- `ZEUS_POWER_READER_BIN=zeus-power-reader`
- `PYTHON_BIN=python3`

If `GATEWAY_BASE_URL` is not set, the runner resolves the control gateway port
from `configs/port_profiles.toml` using the first selected port profile ID.

If you override the port-profile list at runtime, keep the same number of
profiles as the generated bundle's `--power-gpu-indices` list. Independent
`freq-controller-linespace-instance` controllers are paired with backend GPUs
by list position.

Before each replay job, the runner sends the selected ctx-aware thresholds to
`POST /ctx-aware/start`, launches one `freq-controller-linespace-instance` per
backend, and after the replay finishes it stops the controllers and calls
`POST /ctx-aware/end`.

## Generated Files

For each QPS slug `qpsX`:

- `generated/<timestamp>/qpsX/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<selected-plan>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

Default replay output layout:

- `results/replay/sweep-qps-docker-power-clean-gateway_multi-freq-ctrl-linesapce-instance-ctx-aware/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/`

With `--output-suffix <suffix>`, the top-level replay directory name becomes:

- `results/replay/sweep-qps-docker-power-clean-gateway_multi-freq-ctrl-linesapce-instance-ctx-aware-<suffix>/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/`

For multi-profile runs, replay still splits vLLM metrics by backend profile:

- `<replay_output_dir>/vllm-log/profile-<port-profile-id>/`

Each replay output also includes:

- Zeus power logs under `<replay_output_dir>/power/`
- root freq-control directory `<replay_output_dir>/freq-control-linespace-instance/`
- one controller log directory per backend under
  `<replay_output_dir>/freq-control-linespace-instance/profile-<port-profile-id>/`
- gateway ctx-aware sampler logs under
  `<replay_output_dir>/gateway-output/job/ctx_aware_*_profile-<port-profile-id>.jsonl`

## Post-Process

This variant writes nested per-profile instance-controller logs. The shared
freq-control post-process currently recognizes the flat
`freq-control-linespace-instance/` layout and the multi-controller
`freq-control-linespace-multi/` layout, so these per-profile subdirectories are
best inspected directly for now.

## Tests

```bash
.venv/bin/pytest experiments/sweep-qps-docker-power-clean-gateway_multi-freq-ctrl-linesapce-instance-ctx-aware/test -q
```
