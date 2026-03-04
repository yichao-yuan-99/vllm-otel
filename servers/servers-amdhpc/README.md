# Apptainer HPC Control Plane (Server + Client)

This directory now supports a login-node control plane for AMD HPC clusters:

- `servers/servers-amdhpc/server.py`: HTTP server on port `23971` by default.
- `servers/servers-amdhpc/client.py`: Typer CLI frontend that sends JSON POST commands.
- `servers/servers-amdhpc/client_d.py`: SSH tunnel daemon for client/server on different machines.

The server keeps cluster partition config, loads shared model definitions, submits `sbatch` jobs, and manages one active vLLM service job per port profile.

## Files

- `servers/servers-amdhpc/pyproject.toml`: Python package/dependency setup.
- `servers/servers-amdhpc/server_config.toml`: cluster/server settings and partition config.
- `configs/model_config.toml`: shared model definitions used by the AMD HPC control plane.
- `servers/servers-amdhpc/pull_images.py`: pull Jaeger + vLLM OCI images to SIF files.
- `servers/servers-amdhpc/run/control_state.json`: active job state (auto-generated).
- `servers/servers-amdhpc/logs/`: slurm + jaeger + vllm logs (auto-generated).
- `servers/servers-amdhpc/run/old/` and `servers/servers-amdhpc/logs/old/`: auto-archived stale files from prior server runs.

## 1) Configure

```bash
python3 -m pip install -e ./servers/servers-amdhpc
```

Edit `servers/servers-amdhpc/server_config.toml` for cluster settings and partitions.
Edit `configs/model_config.toml` for model definitions.
Export runtime env vars in your shell before starting the server (for example: `HF_HOME`, `HF_HUB_CACHE`, optional `HF_TOKEN`, and any image overrides like `APPTAINER_IMGS`, `JAEGER_IMAGE`, `VLLM_IMAGE`).
The server no longer reads `servers/servers-amdhpc/.env` or `server.env_file`.

`server_config.toml` includes the partition specs requested in `PROMPT.md` (`mi2104x`, `mi2508x`, `mi3001x`, `mi3008x`, `mi3258x`). Model entries are loaded from the shared top-level `configs/model_config.toml` with:

- `vllm_model_name`
- `served_model_name`
- `weight_vram_gb`
- `extra_args`

Start is blocked when `weight_vram_gb > 0.75 * partition.total_vram_gb`.
`cluster.startup_timeout_after_running = true` means startup timeout is counted only after Slurm state is `RUNNING`.

## 2) Pull images (required before server startup)

```bash
python3 servers/servers-amdhpc/pull_images.py --config servers/servers-amdhpc/server_config.toml
```

Use `--force` to re-pull and overwrite existing SIF files.

The server now fails fast at startup if required SIF files are missing.

## 3) Start server (login node)

```bash
python3 servers/servers-amdhpc/server.py --config servers/servers-amdhpc/server_config.toml
```

Default bind is `0.0.0.0:23971`.

## 4) Use client

```bash
python3 servers/servers-amdhpc/client.py clientd-start --ssh-target amd-hpc -P 0
python3 servers/servers-amdhpc/client.py clientd-status -P 0
python3 servers/servers-amdhpc/client.py status -P 0
python3 servers/servers-amdhpc/client.py start -P 0 -p mi3008x -m kimi_k2_5
python3 servers/servers-amdhpc/client.py start -P 0 -p mi3008x -m kimi_k2_5 -b
python3 servers/servers-amdhpc/client.py up -P 0
python3 servers/servers-amdhpc/client.py wait-up -P 0 --timeout-seconds 900
python3 servers/servers-amdhpc/client.py logs -P 0 -n 200
python3 servers/servers-amdhpc/client.py stop -P 0
python3 servers/servers-amdhpc/client.py stop -P 0 -b
python3 servers/servers-amdhpc/client.py stop-poll -P 0
python3 servers/servers-amdhpc/client.py clientd-stop -P 0
```

`-P` is the short alias for `--port-profile`.
Client default server URL is derived from the selected local `client-d` record. By default the local control port is `23971 + <port_profile>` (for example: profile `0` -> `23971`, profile `1` -> `23972`).

## Commands implemented

- `clientd-start`: start local SSH forwarding via `ssh <target>`, using a required port profile ID from `configs/port_profiles.toml`.
- `clientd-start`: the selected port profile sets the local forwarded vLLM/Jaeger ports and the local forwarded control port.
- `clientd-start`: remote vLLM/Jaeger ports on the login node also come from the selected port profile; the remote control server stays on `23971` unless overridden with `--server-port`.
- `clientd-stop`: stop the local SSH forwarding daemon for the specified port profile.
- `clientd-status`: show whether forwarding is running for the specified port profile.
- `start`: submit `sbatch` job with reverse tunnels for:
  - vLLM API: `127.0.0.1:<profile.vllm_port>`
  - Jaeger OTLP gRPC: `127.0.0.1:<profile.jaeger_otlp_port>`
  - Jaeger UI/API: `127.0.0.1:<profile.jaeger_api_port>`
- `start`: every control-plane command requires a specific `--port-profile`/`-P`, so multiple profiles can run concurrently without sharing state.
- `start`: if model `extra_args` includes `--trust-remote-code`, force-sequence tokenizer bootstrap also enables remote code.
- `start`: model `extra_args` from `configs/model_config.toml` are forwarded into the vLLM launch through `VLLM_MODEL_EXTRA_ARGS_B64`.
- `start --block/-b`: block until vLLM + Jaeger endpoints are up.
- `start --block/-b`: streams per-step progress updates (`validate`, `submit`, `record`, `wait_services`).
- `start --block/-b`: startup timeout starts after Slurm reaches `RUNNING` (while `PENDING`, timeout is deferred).
- `start --block/-b`: if interrupted (`Ctrl-C`), client automatically enters blocking stop cleanup and waits for cancellation.
- `start`: fails fast if the selected port profile's service ports are already occupied on the login node.
- `stop`: send `scancel` for the active job on the selected port profile (non-blocking by default).
- `stop --block/-b`: block until it disappears from `squeue` (queries scoped to `-u $USER`).
- `stop --block/-b`: streams per-step progress updates (`validate`, `cancel`, `wait_slurm`).
- `stop-poll`: check whether a prior non-block `stop` has fully finished for the selected port profile.
- `up`: check whether both tunneled vLLM + Jaeger endpoints are currently up for the selected port profile.
- `wait-up`: block until both tunneled vLLM + Jaeger endpoints are up for the selected port profile.
- `wait-up`: supports `--defer-timeout-until-running/--timeout-from-submit`.
- `logs`: tail slurm + jaeger + vllm logs for the selected port profile.
- `status`: show the selected port profile status plus the full active-profile map and configured partitions/models.

You can also manage the tunnel directly:

```bash
python3 servers/servers-amdhpc/client_d.py start --ssh-target <hpc-login-target> -P 0
python3 servers/servers-amdhpc/client_d.py status -P 0
python3 servers/servers-amdhpc/client_d.py stop -P 0
```

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

## Legacy script

`servers/servers-amdhpc/run_services.sh` remains available for non-Slurm local workflows.
Set `VLLM_EXTRA_ARGS_JSON='["--flag","value"]'` in its env file, or provide `VLLM_MODEL_EXTRA_ARGS_B64` directly, when you need extra vLLM launch args.

## Shutdown behavior

When the control server exits gracefully (`Ctrl-C`/`SIGTERM`), it automatically attempts to stop any active allocated job (`server_exit` cleanup).
