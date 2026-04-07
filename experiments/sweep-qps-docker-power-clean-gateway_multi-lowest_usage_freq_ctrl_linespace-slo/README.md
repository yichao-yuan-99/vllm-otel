# Sweep QPS Docker Power Clean Gateway Multi Lowest Usage Freq Ctrl Linespace SLO

This workflow is the `gateway_multi` + `lowest_usage` + independent
`freq-controller-linespace-slo` variant of
`experiments/sweep-qps-docker-power-clean-gateway_multi-lowest_usage_freq_ctrl_linespace`.

It keeps the same clean split-plan lookup, multi-profile replay, Zeus power
logging, and gateway assignment-policy setup, but the generated runner also:

- requires `--target-slo-list`
- sweeps QPS in the outer loop and target SLO in the inner loop
- starts one `freq-controller-linespace-slo` process per backend profile/GPU
  pair before each `(qps, target_slo)` replay job
- passes the same target SLO to each backend controller for that replay job
- resets GPU core clocks for every backend GPU after each `(qps, target_slo)`
  point
- writes controller logs under
  `<replay_output_dir>/freq-control-linespace/profile-<port-profile-id>/`

Default controller threshold:

- `freq-controller-linespace-slo threshold = 395784`

The generated bundle accepts a `gateway_multi`-style port-profile selection:

- pass `--port-profile` as a comma-separated list like `13,14`
- or repeat it, for example `--port-profile 13 --port-profile 14`
- the first selected port profile becomes the control/public profile used by
  `replayer replay`
- that same control/public profile is used to resolve the `gateway_multi`
  `/policy` endpoint unless `GATEWAY_BASE_URL` is provided
- `--power-gpu-indices` must list one backend GPU per selected port profile in
  the same order, for example `--port-profile 13,14 --power-gpu-indices 2,3`

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
python3 -m pip install -e './freq-controller-linespace-slo'
python3 -m pip install -e './zeus-power-reader'
```

## Generate One Sweep Bundle

```bash
python3 experiments/sweep-qps-docker-power-clean-gateway_multi-lowest_usage_freq_ctrl_linespace-slo/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.02,0.04,0.08 \
  --target-slo-list 8,10,12 \
  --time-constraint-s 10800 \
  --target-model qwen3_coder_30b_fp8 \
  --port-profile 2,13 \
  --split exclude-unranked \
  --additional-suffix qwen3_fp8 \
  --power-gpu-indices 2,3
```

Optional overrides:

- `--freq-controller-threshold`

`--split` values are `top`, `rest`, or `exclude-unranked`. `full` is accepted
as a compatibility alias for `exclude-unranked`.

## Run Generated Sweep

```bash
bash experiments/sweep-qps-docker-power-clean-gateway_multi-lowest_usage_freq_ctrl_linespace-slo/generated/<timestamp>/run_replay.sh
```

Override the gateway_multi port-profile selection at runtime:

```bash
bash experiments/sweep-qps-docker-power-clean-gateway_multi-lowest_usage_freq_ctrl_linespace-slo/generated/<timestamp>/run_replay.sh 13,14
```

Optional runtime env overrides for the generated runner:

- `PORT_PROFILE_IDS=13,14`
- `GATEWAY_BASE_URL=http://127.0.0.1:<gateway_port>`
- `FREQ_CONTROLLER_CONFIG=/path/to/freq-controller-linespace-slo.toml`
- `FREQ_CONTROLLER_THRESHOLD=...`
- `FREQ_CONTROLLER_BIN=freq-controller-linespace-slo`
- `RESET_GPU_CORE_FREQ_BIN=reset-gpu-core-freq`
- `CURL_BIN=curl`
- `ZEUS_POWER_READER_BIN=zeus-power-reader`
- `PYTHON_BIN=python3`

If `GATEWAY_BASE_URL` is not set, the runner resolves the control gateway port
from `configs/port_profiles.toml` using the first selected port profile ID.

Before the sweep starts, the runner sends:

```json
{"assignment_policy": "lowest_usage"}
```

to `POST /policy`.

If you override the port-profile list at runtime, keep the same number of
profiles as the generated bundle's `--power-gpu-indices` list. Independent
`freq-controller-linespace-slo` instances are paired with backend GPUs by list
position.

Because `freq-controller-linespace-slo` waits for gateway `job_active=true`, it
can be started before replay and remain pending until the replay job actually
begins.

## Generated Files

For each `(qps, target_slo)` pair:

- `generated/<timestamp>/<qps-slug>/<slo-slug>/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<selected-plan>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

Default replay output layout:

- `results/replay/sweep-qps-docker-power-clean-gateway_multi-lowest_usage_freq_ctrl_linespace-slo/<dataset>/<agent>/split/<split>/<qps>/<slo>/<timestamp>/`

For multi-profile runs, replay still splits vLLM metrics by backend profile:

- `<replay_output_dir>/vllm-log/profile-<port-profile-id>/`

Each replay output also includes:

- Zeus power logs under `<replay_output_dir>/power/`
- root freq-control directory `<replay_output_dir>/freq-control-linespace/`
- one controller log directory per backend under
  `<replay_output_dir>/freq-control-linespace/profile-<port-profile-id>/`

Each backend controller writes the usual linespace files plus SLO decisions:

- `freq-controller-ls.query.*.jsonl`
- `freq-controller-ls.decision.*.jsonl`
- `freq-controller-ls.slo-decision.*.jsonl`

## Post-Process

The shared freq-control post-process can aggregate nested
`freq-control-linespace/profile-<id>/` logs and produce per-profile summaries
and figures. The SLO-decision extractor currently expects a flat
`freq-control-linespace/` layout, so these per-profile
`freq-controller-ls.slo-decision.*.jsonl` logs are best inspected directly for
now.

## Tests

```bash
.venv/bin/pytest experiments/sweep-qps-docker-power-clean-gateway_multi-lowest_usage_freq_ctrl_linespace-slo/test -q
```
