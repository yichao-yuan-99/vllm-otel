# Interactive AMD HPC Embedded TP2 MI2508X Sweep QPS Power Clean Freq Ctrl Linespace AMD

This workflow is the interactive `mi2508x` TP2 AMD variant of
`experiments/sweep-qps-docker-power-clean-gateway_multi-freq-ctrl-linesapce`.

It keeps the same clean split-plan lookup, Poisson replay generation, and
gateway `ctx-aware` flow as the embedded AMD workflows, but assumes the service
stack is already running on an interactive `mi2508x` allocation via
`servers/servers-amdhpc-interactive-mi2508x-embedded-TP2/client.py start`.

Key differences from the batch embedded experiments:

- no `submit_embedded_tp1.sh` is generated
- the generated `run_replay.sh` sources the shared helper in
  `servers/servers-amdhpc-interactive-mi2508x-embedded-TP2/experiment-env.sh`
  to validate `port_profile_id`, validate the TP2 GPU pair, resolve ports, and
  wait for the interactive vLLM + gateway stack
- QPS points are assigned to port profiles `0..3` in ascending order
- each port profile uses a fixed GPU pair: `0->0,1`, `1->2,3`, `2->4,5`,
  `3->6,7`
- outputs are written under `profile-<port_profile_id>/`
- power logging uses `amd-power-reader`
- frequency control uses `freq-controller-linespace-amd`
- GPU reset uses `amd-reset-gpu-core-freq` once per GPU in the selected pair

Threshold selection:

The generator still has generic built-in defaults:

- `ctx-aware usage_threshold_tokens = 1445793`
- `ctx-aware scheduling_threshold_tokens = 1369699`
- `freq-controller-linespace-amd threshold = 1141416`

For interactive `mi2508x` TP2 runs, prefer the model-specific thresholds
recorded in
`servers/servers-amdhpc-interactive-mi2508x-embedded-TP2/README.token-limits.md`:

| Target model | Total tokens | `ctx-aware usage_threshold_tokens` | `ctx-aware scheduling_threshold_tokens` | `freq-controller-linespace-amd threshold` |
| --- | ---: | ---: | ---: | ---: |
| `qwen3_coder_30b_fp8` | `901840` | `856748` | `811656` | `676380` |
| `qwen3_coder_30b` | `569040` | `540588` | `512136` | `426780` |

## Requirements

Install the local console scripts used by the generated runner:

```bash
python3 -m pip install -e './amd-power-reader'
python3 -m pip install -e './freq-controller-linespace-amd'
```

On the same interactive `mi2508x` node, start the service stack in the
background:

```bash
python3 servers/servers-amdhpc-interactive-mi2508x-embedded-TP2/client.py start
```

`client.py start` is blocking: it prints progress until the background service
stack is ready, then exits while leaving the services running. That
launcher-managed stack starts `amd-smi-power-daemon` itself and prints the
active `AMD_SMI_POWER_SOCKET_PATH` when the stack is ready. The generated
`run_replay.sh` sources the shared helper from
`servers/servers-amdhpc-interactive-mi2508x-embedded-TP2/experiment-env.sh`
and uses it for port resolution and service readiness checks. New generated
bundles default `AMD_SMI_POWER_SOCKET_PATH` to
`/tmp/amdsmi-power-reader.sock`, and the environment variable remains an
override if you need a different socket.

## Generate One Sweep Bundle

```bash
python3 experiments/amd-embeded/servers-amdhpc-interactive-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.1,0.2 \
  --time-constraint-s 1000 \
  --target-model qwen3_coder_30b_fp8 \
  --ctx-aware-usage-threshold-tokens 856748 \
  --ctx-aware-scheduling-threshold-tokens 811656 \
  --freq-controller-threshold 676380 \
  --additional-suffix qwen3_fp8 \
  --split exclude-unranked
```

Example for `qwen3_coder_30b`:

```bash
python3 experiments/amd-embeded/servers-amdhpc-interactive-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generate_experiment.py \
  --source-run-dir results/qwen3-coder-30b/swebench-verified/mini-swe-agent/swebench-verified-20260306T062226Z/ \
  --poisson-seed 7 \
  --randomize-seed 11 \
  --qps-list 0.1,0.2 \
  --time-constraint-s 1000 \
  --target-model qwen3_coder_30b \
  --ctx-aware-usage-threshold-tokens 540588 \
  --ctx-aware-scheduling-threshold-tokens 512136 \
  --freq-controller-threshold 426780 \
  --split exclude-unranked
```

Optional overrides:

- `--port-profile`
- `--gpu-index`
- `--ctx-aware-usage-threshold-tokens`
- `--ctx-aware-scheduling-threshold-tokens`
- `--freq-controller-threshold`
- `--additional-suffix`
- `--output-suffix`

This `mi2508x` TP2 variant supports port profiles `0..3`. If `--gpu-index` is
provided, it must match the default port profile pair:

- profile `0` -> `0,1`
- profile `1` -> `2,3`
- profile `2` -> `4,5`
- profile `3` -> `6,7`

At runtime, the generated script chooses the QPS point assigned to the selected
port profile and defaults the GPU pair to that profile's assigned pair.

## Run Generated Sweep

After `client.py start` reports readiness:

```bash
bash experiments/amd-embeded/servers-amdhpc-interactive-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh
```

Explicitly select a port profile:

```bash
bash experiments/amd-embeded/servers-amdhpc-interactive-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/generated/<timestamp>/run_replay.sh 3
```

Useful runtime env overrides:

- `PORT_PROFILE_ID` (`0..3`)
- `GPU_INDEX` (`0,1`, `2,3`, `4,5`, or `6,7`)
- `GATEWAY_BASE_URL`
- `AMD_SMI_POWER_SOCKET_PATH`
- `CTX_AWARE_USAGE_THRESHOLD_TOKENS`
- `CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS`
- `FREQ_CONTROLLER_CONFIG`
- `FREQ_CONTROLLER_THRESHOLD`
- `AMD_POWER_READER_BIN`
- `FREQ_CONTROLLER_BIN`
- `RESET_GPU_CORE_FREQ_BIN`
- `INTERACTIVE_CLIENT_SCRIPT`
- `INTERACTIVE_ENV_HELPER`
- `INTERACTIVE_START_SERVICES_COMMAND`
- `INTERACTIVE_START_SERVICES_SCRIPT`
- `SERVICE_READY_TIMEOUT_SECONDS`
- `SERVICE_READY_POLL_INTERVAL_SECONDS`

`run_replay.sh` performs a service preflight against the selected port profile
before it starts the replay and prints a hint to
`python3 servers/servers-amdhpc-interactive-mi2508x-embedded-TP2/client.py start`
if the stack is not ready yet.

When the replay finishes, stop the background stack with:

```bash
python3 servers/servers-amdhpc-interactive-mi2508x-embedded-TP2/client.py stop
```

## Generated Files

For each assigned QPS slug `qpsX`:

- `generated/<timestamp>/qpsX/replay.toml`

Batch-level outputs:

- `generated/<timestamp>/plan/<selected-plan>.json`
- `generated/<timestamp>/run_replay.sh`
- `generated/<timestamp>/manifest.json`

Default replay output layout:

- `results/replay/amd-embeded/servers-amdhpc-interactive-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/<dataset>/<agent>/split/<split>/<qps>/<timestamp>/profile-<port_profile_id>/`

Each profile output also includes:

- AMD power logs under `<replay_output_dir>/power/`
- freq controller logs under `<replay_output_dir>/freq-control-linespace-amd/`
- gateway ctx-aware sampler logs under
  `<replay_output_dir>/gateway-output/job/ctx_aware_*_profile-<port_profile_id>.jsonl`

## Tests

```bash
pytest experiments/amd-embeded/servers-amdhpc-interactive-mi2508x-embedded-TP2/mi2508x/sweep-qps-docker-power-clean-freq-ctrl-linespace-amd/test -q
```
