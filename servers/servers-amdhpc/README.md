# Apptainer HPC Control Plane (Server + Client)

This directory now supports a login-node control plane for AMD HPC clusters:

- `servers/servers-amdhpc/server.py`: HTTP server on port `23971` by default.
- `servers/servers-amdhpc/client.py`: Typer CLI frontend that sends JSON POST commands.

The server keeps cluster partition config, loads shared model definitions, submits `sbatch` jobs, and manages active vLLM service jobs per port profile (single-profile jobs and grouped multi-profile jobs).

## Files

- `servers/servers-amdhpc/pyproject.toml`: Python package/dependency setup.
- `servers/servers-amdhpc/server_config.toml`: cluster/server settings and partition config.
- `servers/servers-amdhpc/README.remote-host.md`: using `-r/--remote-host` to override SSH target.
- `servers/servers-amdhpc/README.start-many.md`: vectorized single-profile starts (`start-many`).
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
You can optionally set `partition.<name>.sif_img` to override vLLM SIF per partition; when `sif_img` exists on disk, that partition uses it instead of the global `[images]` vLLM SIF.
If `sif_img` is a bare filename (for example `my-image.sif`), it is resolved under `APPTAINER_IMGS`.

`server_config.toml` includes the partition specs requested in `PROMPT.md` (`mi2104x`, `mi2508x`, `mi3001x`, `mi3008x`, `mi3258x`). Model entries are loaded from the shared top-level `configs/model_config.toml` with:

- `vllm_model_name`
- `served_model_name`
- `weight_vram_gb`
- `extra_args`

Start is blocked when `weight_vram_gb > 0.75 * partition.total_vram_gb`.
Grouped start (`start-group`) is blocked when `weight_vram_gb > 0.75 * per_profile_vram`, where `per_profile_vram` is computed from the even GPU split on one node.
`cluster.startup_timeout_after_running = true` means startup timeout is counted only after Slurm state is `RUNNING`.

## 2) Start server (login node)

```bash
python3 servers/servers-amdhpc/server.py --config servers/servers-amdhpc/server_config.toml
```

Default bind is `0.0.0.0:23971`.
Server startup now ensures the required SIF files exist and pulls them automatically if they are missing.

## 3) Use client

```bash
python3 servers/servers-amdhpc/client.py status -r amd-hpc -P 0
python3 servers/servers-amdhpc/client.py start -r amd-hpc -P 0 -p mi3008x -m kimi_k2_5
python3 servers/servers-amdhpc/client.py start -r amd-hpc -P 0 -p mi3008x -m kimi_k2_5 -b
python3 servers/servers-amdhpc/client.py start -r amd-hpc -P 0 -p mi3008x -m kimi_k2_5 --lmcache 100
python3 servers/servers-amdhpc/client.py start -r amd-hpc -P 0 -p mi3008x -m kimi_k2_5 --extra-vllm-args "--enable-expert-parallel"
python3 servers/servers-amdhpc/client.py start-many -r amd-hpc -L 0,1,2,3 -p mi3008x -m qwen3_coder_30b
python3 servers/servers-amdhpc/client.py start-group -r amd-hpc -g bench_a -L 0,1,2,3 -p mi3008x -m kimi_k2_5
python3 servers/servers-amdhpc/client.py start-group -r amd-hpc -g bench_a -L 0,1,2,3 -p mi3008x -m kimi_k2_5 --lmcache 100
python3 servers/servers-amdhpc/client.py start-group -r amd-hpc -g bench_a -L 0,1,2,3 -p mi3008x -m kimi_k2_5 --extra-vllm-args "--enable-expert-parallel"
python3 servers/servers-amdhpc/client.py start-group -r amd-hpc -g bench_b -L 0,1,2,3,4,5,6,7 -p mi3008x -m qwen3_coder_30b
python3 servers/servers-amdhpc/client.py group-status -r amd-hpc -g bench_a
python3 servers/servers-amdhpc/client.py up -r amd-hpc -P 0
python3 servers/servers-amdhpc/client.py wait-up -r amd-hpc -P 0 --timeout-seconds 900
python3 servers/servers-amdhpc/client.py logs -r amd-hpc -P 0 -n 200
python3 servers/servers-amdhpc/client.py alive-profiles -r amd-hpc
python3 servers/servers-amdhpc/client.py stop-alive-profiles -r amd-hpc
python3 servers/servers-amdhpc/client.py stop -r amd-hpc -P 0
python3 servers/servers-amdhpc/client.py stop-group -r amd-hpc -g bench_a
```

`-P` is the short alias for `--port-profile`.
`-r` / `--remote-host` is an alias for `--ssh-target` and is accepted by all client commands; see `servers/servers-amdhpc/README.remote-host.md`.
Client default server URL is derived from the selected local tunnel record. By default the local control port is `23971 + <port_profile>` (for example: profile `0` -> `23971`, profile `1` -> `23972`).
Client `start` now also launches a per-profile local gateway daemon in the background after services are up, and `stop` tears it down first.
For vectorized independent starts across many profiles (one start job per profile), see `servers/servers-amdhpc/README.start-many.md`.

## Render sbatch only (no submit)

Use `render-sbatch.py` when you want the same `start` / `start-group` interface but only write the sbatch script once, without starting client-d/gateway or submitting to Slurm.
Detailed usage (including `--local-mode`) is in `servers/servers-amdhpc/README.render-sbatch.md`.

```bash
# Single-profile render
python3 servers/servers-amdhpc/render-sbatch.py start -P 0 -p mi3008x -m qwen3_coder_30b

# Grouped render (4 profiles on one node -> tp=2)
python3 servers/servers-amdhpc/render-sbatch.py start-group -g mi300_tp2 -L 0,1,2,3 -p mi3008x -m qwen3_coder_30b

# Add extra vLLM args
python3 servers/servers-amdhpc/render-sbatch.py start -P 0 -p mi3008x -m qwen3_coder_30b --extra-vllm-args "--enable-expert-parallel"
```

Optional:
- add `--lmcache <size>` to inject LMCache settings into the rendered vLLM command/env
- add `--extra-vllm-args "<args>"` to append additional vLLM start flags
- add `--check-port-availability` to fail render if selected login-node ports are currently in use

## Grouped Single-Node GPU Split

`start-group` runs all grouped workers on one node and divides that node's GPUs evenly across profiles.

How allocation works:

- `group_size = len(profile_list)`
- `gpus_per_profile = partition.gpus_per_node / group_size` (must divide evenly)
- each profile gets a fixed contiguous GPU slice on that node
- `tensor_parallel_size` for each profile is set to `gpus_per_profile`

Model-fit rule for grouped launch:

- per-profile VRAM = `partition.gpu_memory_gb * gpus_per_profile`
- grouped start is rejected when `weight_vram_gb > 0.75 * per_profile_vram`

Examples on `mi3008x` (`8 x 192GB` GPUs):

- `--profile-list 0,1,2,3` -> `group_size=4`, `gpus_per_profile=2`, `tp=2`, per-profile VRAM=`384GB`
- `--profile-list 0,1,2,3,4,5,6,7` -> `group_size=8`, `gpus_per_profile=1`, `tp=1`, per-profile VRAM=`192GB`

Example commands:

```bash
# 4 profiles on one MI3008x node -> 2 GPUs/profile (tp=2)
python3 servers/servers-amdhpc/client.py start-group -r amd-hpc -g mi300_tp2 -L 0,1,2,3 -p mi3008x -m qwen3_coder_30b

# 8 profiles on one MI3008x node -> 1 GPU/profile (tp=1)
python3 servers/servers-amdhpc/client.py start-group -r amd-hpc -g mi300_tp1 -L 0,1,2,3,4,5,6,7 -p mi3008x -m qwen3_coder_30b
```

Validation and failure cases:

- rejected when `group_size > partition.gpus_per_node`
- rejected when `partition.gpus_per_node % group_size != 0`
- rejection details include partition GPU info, group size, and selected profile list

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
- all client commands accept `--ssh-target`/`--remote-host`/`-r` for a consistent CLI surface.
- `start`: the selected port profile sets the local forwarded vLLM/Jaeger ports and the local forwarded control port.
- `start`: remote vLLM/Jaeger ports on the login node also come from the selected port profile; the remote control server stays on `23971` unless overridden with `--server-port`.
- `start`: the selected port profile is sent to the remote control server and is also used there to pick the login<->compute reverse-tunnel ports.
- `start`: Slurm `job_name` is computed as `cluster.job_name_prefix + <port_profile>`.
- `start`: submit `sbatch` job with reverse tunnels for:
  - vLLM API: `127.0.0.1:<profile.vllm_port>`
  - Jaeger OTLP gRPC: login `127.0.0.1:<profile.jaeger_otlp_port>` -> compute `127.0.0.1:4317`
  - Jaeger UI/API: login `127.0.0.1:<profile.jaeger_api_port>` -> compute `127.0.0.1:16686`
- `start`: single-profile control-plane commands use `--port-profile`/`-P`; grouped commands use `--group-name` + `--profile-list`.
- `start`: after services are up, it auto-starts a local gateway daemon in the background for the selected port profile.
- `start`: if `--block/-b` is not set, client still waits for `/wait-up` readiness before launching gateway.
- `start`: supports repeatable `--env KEY=VALUE` to inject extra environment variables into the vLLM apptainer process.
- `start`/`start-group`: optional `--lmcache <size>` injects `LMCACHE_MAX_LOCAL_CPU_SIZE=<size>` and appends `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'` to vLLM startup args.
- `start`/`start-group`: always inject `LMCACHE_INTERNAL_API_SERVER_ENABLED=1` and `PYTHONHASHSEED=0` into vLLM apptainer env.
- `start`: always inject `LMCACHE_INTERNAL_API_SERVER_PORT_START=<profile.lmcache_port>` from the selected `port_profile`.
- `start-group`: always inject per-worker `LMCACHE_INTERNAL_API_SERVER_PORT_START` from each selected profile's `lmcache_port`.
- `start-group`: launches client-d for each profile in `--profile-list`, then calls grouped control-plane start, then launches one local gateway daemon per profile.
- `start-group`: grouped start is blocking; it waits until all grouped profiles report vLLM + Jaeger ready.
- `start-group`: supports repeatable `--env KEY=VALUE` and applies those variables to every grouped vLLM worker.
- `start-group`: submits one single-node Slurm job with one worker task per profile; node/task index maps to profile-specific ports from `--profile-list`.
- `start-group`: GPUs on that node are divided evenly across profiles; each profile gets `tp = gpus_per_profile`.
- `start-group`: group size must evenly divide `partition.gpus_per_node` (for example `4` profiles on `mi3008x` => `2` GPUs/profile, `8` profiles => `1` GPU/profile).
- `start-group`: streams per-profile progress lines while waiting on grouped startup (`/start/status` for each profile).
- `group-status`: reports grouped run status (group name, profiles, job IDs, per-profile status).
- `group-status`: prints lightweight query progress lines before returning JSON payload.
- `start`: gateway startup defaults to `gateway/config.toml` and falls back to `gateway/config.example.toml` when `config.toml` is missing.
- `start`: AMD HPC always adds `--trust-remote-code` to the vLLM serve args, and force-sequence tokenizer bootstrap is forced to use remote code as well.
- `start`: when `partition.gpus_per_node > 1`, AMD HPC also adds `--distributed_executor_backend ray` automatically.
- `start`: model `extra_args` from `configs/model_config.toml` are forwarded into the vLLM launch through `VLLM_MODEL_EXTRA_ARGS_B64` after that normalization.
- `start --block/-b`: block until vLLM + Jaeger endpoints are up.
- `start --block/-b`: streams per-step progress updates (`validate`, `submit`, `record`, `wait_services`).
- `start --block/-b`: startup timeout starts after Slurm reaches `RUNNING` (while `PENDING`, timeout is deferred).
- `start --block/-b`: if interrupted (`Ctrl-C`), client automatically enters blocking stop cleanup and then tears down the local tunnel.
- `start`: fails fast if the selected port profile's service ports are already occupied on the login node.
- `stop`: always stops the local gateway daemon first, then waits for the remote job to disappear from Slurm, then stops the local tunnel.
- `stop`: if no active job exists, it still stops the local gateway and local tunnel for that port profile.
- `stop-group`: stops a grouped run by `--group-name`, blocks until the grouped job is gone, then tears down gateway/client-d for every profile in the group.
- `stop-group`: streams per-profile progress lines while waiting on grouped shutdown (`/stop/status` for each profile).
- `up`: check whether both tunneled vLLM + Jaeger endpoints are currently up for the selected port profile.
- `wait-up`: block until both tunneled vLLM + Jaeger endpoints are up for the selected port profile.
- `wait-up`: supports `--defer-timeout-until-running/--timeout-from-submit`.
- `logs`: tail slurm + jaeger + vllm logs for the selected port profile.
- `logs` for grouped runs uses profile-specific logs (`jaeger.<job_id>.p<profile>.log`, `vllm.<job_id>.p<profile>.log`).
- `alive-profiles`: scan all configured port profiles and report which profiles are alive.
- `alive-profiles`: grouped info is deduplicated into top-level `data.groups`; profile entries keep only `group_name`.
- `alive-profiles`: per-profile output is compact (status summary fields) and no longer embeds full raw `/status` payloads.
- `alive-profiles --verbose`: include full raw per-profile payloads under `profiles[*].raw`.
- `stop-alive-profiles`: group-aware stop; grouped profiles are stopped through one `group/stop` call per group.
- `status`: show the selected port profile status plus the full active-profile map and configured partitions/models.
- Single-profile commands (`-P`) are rejected for profiles that belong to an active group (`start`, `stop`, `stop-poll`, `up`, `wait-up`, `logs`).

## HTTP API (server)

All commands accept POST JSON:

- `POST /start` with `{"port_profile": 0, "partition": "...", "model": "...", "block": false, "lmcache": 100}`
- `POST /stop` with `{"port_profile": 0, "block": false}`
- `POST /group/start` with `{"group_name": "bench_a", "profile_list": [0,1,2,3], "partition": "...", "model": "...", "block": true, "lmcache": 100}`
- `POST /group/stop` with `{"group_name": "bench_a", "block": true}`
- `POST /group/status` with `{"group_name": "bench_a"}`
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
