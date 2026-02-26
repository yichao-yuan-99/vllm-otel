# Apptainer HPC Control Plane (Server + Client)

This directory now supports a login-node control plane for AMD HPC clusters:

- `apptainer/server.py`: HTTP server on port `23971` by default.
- `apptainer/client.py`: Typer CLI frontend that sends JSON POST commands.
- `apptainer/client_d.py`: SSH tunnel daemon for client/server on different machines.

The server keeps cluster partition + model allowlist config, submits `sbatch` jobs, and manages one active vLLM service job at a time.

## Files

- `apptainer/pyproject.toml`: Python package/dependency setup.
- `apptainer/server_config.toml`: cluster partitions and allowed models.
- `apptainer/run/control_state.json`: active job state (auto-generated).
- `apptainer/logs/`: slurm + jaeger + vllm logs (auto-generated).
- `apptainer/run/old/` and `apptainer/logs/old/`: auto-archived stale files from prior server runs.

## 1) Configure

```bash
python3 -m pip install -e ./apptainer
```

Edit `apptainer/server_config.toml`.
Export runtime env vars in your shell before starting the server (for example: `HF_HOME`, `HF_HUB_CACHE`, optional `HF_TOKEN`, and any image overrides like `APPTAINER_IMGS`, `JAEGER_IMAGE`, `VLLM_IMAGE`).
The server no longer reads `apptainer/.env` or `server.env_file`.

`server_config.toml` includes the partition specs requested in `PROMPT.md` (`mi2508x`, `mi3001x`, `mi3008x`, `mi3258x`) and model entries with:

- `vllm_model_name`
- `served_model_name`
- `weight_vram_gb`
- `extra_args`

Start is blocked when `weight_vram_gb > 0.75 * partition.total_vram_gb`.

## 2) Start server (login node)

```bash
python3 apptainer/server.py --config apptainer/server_config.toml
```

Default bind is `0.0.0.0:23971`.

## 3) Use client

```bash
python3 apptainer/client.py clientd-start --ssh-target <hpc-login-target>
python3 apptainer/client.py clientd-status
python3 apptainer/client.py status
python3 apptainer/client.py pull
python3 apptainer/client.py start -p mi3008x -m kimi_k2_5
python3 apptainer/client.py start -p mi3008x -m kimi_k2_5 -b
python3 apptainer/client.py up
python3 apptainer/client.py wait-up --timeout-seconds 900
python3 apptainer/client.py logs -n 200
python3 apptainer/client.py test
python3 apptainer/client.py stop
python3 apptainer/client.py stop -b
python3 apptainer/client.py stop-poll
python3 apptainer/client.py clientd-stop
```

Client default server URL is `http://127.0.0.1:23971`.

## Commands implemented

- `pull`: pull Jaeger + vLLM OCI images to SIF under `APPTAINER_IMGS`.
- `clientd-start`: start local SSH forwarding for server/vLLM/Jaeger ports via `ssh <target>`.
- `clientd-stop`: stop local SSH forwarding daemon.
- `clientd-status`: show whether forwarding daemon is running.
- `start`: submit `sbatch` job with reverse tunnels for:
  - vLLM API: `127.0.0.1:11451`
  - Jaeger OTLP gRPC: `127.0.0.1:4317`
  - Jaeger UI/API: `127.0.0.1:16686`
- `start --block/-b`: block until vLLM + Jaeger endpoints are up.
- `start --block/-b`: streams per-step progress updates (`validate`, `submit`, `record`, `wait_services`).
- `start`: fails fast if `cluster.service_port` is already occupied on login node.
- `stop`: send `scancel` for active job (non-blocking by default).
- `stop --block/-b`: block until it disappears from `squeue` (queries scoped to `-u $USER`).
- `stop --block/-b`: streams per-step progress updates (`validate`, `cancel`, `wait_slurm`).
- `stop-poll`: check whether a prior non-block `stop` has fully finished.
- `up`: check whether both tunneled vLLM + Jaeger endpoints are currently up.
- `wait-up`: block until both tunneled vLLM + Jaeger endpoints are up.
- `logs`: tail slurm + jaeger + vllm logs.
- `test`: run OTEL + force-sequence smoke tests, with live phase updates in the client.
- `test`: includes a preflight check for `http://127.0.0.1:<service_port>/v1/models` and fails fast on wrong/non-vLLM endpoints.
- `status`: show active job and configured partitions/models.

You can also manage the tunnel directly:

```bash
python3 apptainer/client_d.py start --ssh-target <hpc-login-target>
python3 apptainer/client_d.py status
python3 apptainer/client_d.py stop
```

## HTTP API (server)

All commands accept POST JSON:

- `POST /pull`
- `POST /start` with `{"partition": "...", "model": "...", "block": false}`
- `POST /stop` with optional `{"block": false}`
- `POST /stop/poll`
- `POST /start/status`
- `POST /stop/status`
- `POST /up`
- `POST /wait-up` with optional `{"timeout_seconds": 900, "poll_interval_seconds": 2.0}`
- `POST /logs` with optional `{"lines": 200}`
- `POST /test`
- `POST /test/status`
- `POST /status`

Optional aliases are also available under `/command/<name>`.

## Legacy script

`apptainer/run_services.sh` remains available for non-Slurm local workflows.

## Shutdown behavior

When the control server exits gracefully (`Ctrl-C`/`SIGTERM`), it automatically attempts to stop any active allocated job (`server_exit` cleanup).
