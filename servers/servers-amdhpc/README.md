# Apptainer HPC Control Plane (Server + Client)

This directory now supports a login-node control plane for AMD HPC clusters:

- `servers/servers-amdhpc/server.py`: HTTP server on port `23971` by default.
- `servers/servers-amdhpc/client.py`: Typer CLI frontend that sends JSON POST commands.

The server keeps cluster partition config, loads shared model definitions, submits `sbatch` jobs, and manages one active vLLM service job per port profile.

## Files

- `servers/servers-amdhpc/pyproject.toml`: Python package/dependency setup.
- `servers/servers-amdhpc/server_config.toml`: cluster/server settings and partition config.
- `servers/servers-amdhpc/tests/test_live_multi_profile.py`: gated live integration test for concurrent profiles `5..9`.
- `configs/model_config.toml`: shared model definitions used by the AMD HPC control plane.
- `servers/servers-amdhpc/run/control_state.json`: active job state (auto-generated).
- `servers/servers-amdhpc/logs/`: slurm + jaeger + vllm logs (auto-generated).
- `servers/servers-amdhpc/run/old/` and `servers/servers-amdhpc/logs/old/`: auto-archived stale files from prior server runs.

## 1) Configure

```bash
python3 -m pip install -e ./servers/servers-amdhpc
```

Edit `servers/servers-amdhpc/server_config.toml` for cluster settings and partitions.
Edit `configs/model_config.toml` for model definitions.
Export runtime env vars in your shell before starting the server (for example: `HF_HOME`, `HF_HUB_CACHE`, optional `HF_TOKEN`, and optional image-path overrides like `APPTAINER_IMGS`, `JAEGER_SIF`, `VLLM_SIF`).
The server no longer reads `servers/servers-amdhpc/.env` or `server.env_file`.
`cluster.job_name_prefix` is combined with the selected port profile ID when Slurm jobs are submitted, so profile `3` with prefix `vllm_job_` becomes job name `vllm_job_3`.
For Slurm-managed startup, login<->compute service ports come from `configs/port_profiles.toml`, not fixed `cluster.service_port` settings.
Image refs now live in `[images]` inside `server_config.toml`. SIF filenames are inferred from the image basename by replacing `:` with `-` and adding `.sif`.

`server_config.toml` includes the partition specs requested in `PROMPT.md` (`mi2104x`, `mi2508x`, `mi3001x`, `mi3008x`, `mi3258x`). Model entries are loaded from the shared top-level `configs/model_config.toml` with:

- `vllm_model_name`
- `served_model_name`
- `weight_vram_gb`
- `extra_args`

Start is blocked when `weight_vram_gb > 0.75 * partition.total_vram_gb`.
`cluster.startup_timeout_after_running = true` means startup timeout is counted only after Slurm state is `RUNNING`.

## 2) Start server (login node)

```bash
python3 servers/servers-amdhpc/server.py --config servers/servers-amdhpc/server_config.toml
```

Default bind is `0.0.0.0:23971`.
Server startup now ensures the required SIF files exist and pulls them automatically if they are missing.

## 3) Use client

```bash
python3 servers/servers-amdhpc/client.py status -P 0
python3 servers/servers-amdhpc/client.py start -P 0 -p mi3008x -m kimi_k2_5
python3 servers/servers-amdhpc/client.py start -P 0 -p mi3008x -m kimi_k2_5 -b
python3 servers/servers-amdhpc/client.py up -P 0
python3 servers/servers-amdhpc/client.py wait-up -P 0 --timeout-seconds 900
python3 servers/servers-amdhpc/client.py logs -P 0 -n 200
python3 servers/servers-amdhpc/client.py stop -P 0
```

`-P` is the short alias for `--port-profile`.
Client default server URL is derived from the selected local tunnel record. By default the local control port is `23971 + <port_profile>` (for example: profile `0` -> `23971`, profile `1` -> `23972`).

## 4) Run live integration test

`servers/servers-amdhpc/tests/test_live_multi_profile.py` is a real cluster integration test. It assumes:

- the AMD HPC control server is already running
- the SSH target is `amd-hpc` by default
- profiles `5..9` are available
- partition `mi2104x` and model `qwen3_coder_30b` are valid for the current config

The test is skipped by default. To run it:

```bash
RUN_AMDHPC_LIVE_TESTS=1 \
python3 -m unittest discover -s servers/servers-amdhpc/tests -p 'test_*.py'
```

What it does:

- starts port profiles `5..9` concurrently with `mi2104x` and `qwen3_coder_30b`
- waits for all five starts to finish
- checks vLLM, Jaeger UI, and Jaeger OTLP health on all five profiles
- runs five concurrent query workers for two minutes
- gracefully stops all started profiles at the end

Optional overrides:

- `AMDHPC_SSH_TARGET`: override the default SSH target
- `AMDHPC_START_TIMEOUT_SECONDS`
- `AMDHPC_STOP_TIMEOUT_SECONDS`
- `AMDHPC_REQUEST_TIMEOUT_SECONDS`
- `AMDHPC_MODEL_READY_TIMEOUT_SECONDS`
- `AMDHPC_QUERY_DURATION_SECONDS`
- `AMDHPC_QUERY_PROGRESS_INTERVAL_SECONDS`
- `AMDHPC_COMMAND_HEARTBEAT_SECONDS`

## Commands implemented

- `start`: always starts the local SSH tunnel first and rejects the request if a tunnel daemon is already running for that port profile.
- `start`: defaults `--ssh-target` to `amd-hpc` (override with `--ssh-target` as needed).
- `start`: the selected port profile sets the local forwarded vLLM/Jaeger ports and the local forwarded control port.
- `start`: remote vLLM/Jaeger ports on the login node also come from the selected port profile; the remote control server stays on `23971` unless overridden with `--server-port`.
- `start`: the selected port profile is sent to the remote control server and is also used there to pick the login<->compute reverse-tunnel ports.
- `start`: Slurm `job_name` is computed as `cluster.job_name_prefix + <port_profile>`.
- `start`: submit `sbatch` job with reverse tunnels for:
  - vLLM API: `127.0.0.1:<profile.vllm_port>`
  - Jaeger OTLP gRPC: login `127.0.0.1:<profile.jaeger_otlp_port>` -> compute `127.0.0.1:4317`
  - Jaeger UI/API: login `127.0.0.1:<profile.jaeger_api_port>` -> compute `127.0.0.1:16686`
- `start`: every control-plane command requires a specific `--port-profile`/`-P`, so multiple profiles can run concurrently without sharing state.
- `start`: AMD HPC always adds `--trust-remote-code` to the vLLM serve args, and force-sequence tokenizer bootstrap is forced to use remote code as well.
- `start`: when `partition.gpus_per_node > 1`, AMD HPC also adds `--distributed_executor_backend ray` automatically.
- `start`: model `extra_args` from `configs/model_config.toml` are forwarded into the vLLM launch through `VLLM_MODEL_EXTRA_ARGS_B64` after that normalization.
- `start --block/-b`: block until vLLM + Jaeger endpoints are up.
- `start --block/-b`: streams per-step progress updates (`validate`, `submit`, `record`, `wait_services`).
- `start --block/-b`: startup timeout starts after Slurm reaches `RUNNING` (while `PENDING`, timeout is deferred).
- `start --block/-b`: if interrupted (`Ctrl-C`), client automatically enters blocking stop cleanup and then tears down the local tunnel.
- `start`: fails fast if the selected port profile's service ports are already occupied on the login node.
- `stop`: always waits for the remote job to disappear from Slurm and then stops the local tunnel automatically.
- `stop`: if no active job exists, it still stops the local tunnel for that port profile.
- `up`: check whether both tunneled vLLM + Jaeger endpoints are currently up for the selected port profile.
- `wait-up`: block until both tunneled vLLM + Jaeger endpoints are up for the selected port profile.
- `wait-up`: supports `--defer-timeout-until-running/--timeout-from-submit`.
- `logs`: tail slurm + jaeger + vllm logs for the selected port profile.
- `status`: show the selected port profile status plus the full active-profile map and configured partitions/models.

## HTTP API (server)

All commands accept POST JSON:

- `POST /start` with `{"port_profile": 0, "partition": "...", "model": "...", "block": false}`
- `POST /stop` with `{"port_profile": 0, "block": false}`
- `POST /stop/poll` with `{"port_profile": 0}`
- `POST /start/status` with `{"port_profile": 0}`
- `POST /stop/status` with `{"port_profile": 0}`
- `POST /up` with `{"port_profile": 0}`
- `POST /wait-up` with `{"port_profile": 0, "timeout_seconds": 900, "poll_interval_seconds": 2.0, "defer_timeout_until_running": true}`
- `POST /logs` with `{"port_profile": 0, "lines": 200}`
- `POST /status` with `{"port_profile": 0}`

Optional aliases are also available under `/command/<name>`.

## Shutdown behavior

When the control server exits gracefully (`Ctrl-C`/`SIGTERM`), it automatically attempts to stop any active allocated job (`server_exit` cleanup).
