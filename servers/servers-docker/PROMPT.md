# Docker Daemon Frontend (Typer) - Concise Command List

the goal here is to make the servers/servers-docker/ a package that setup docker vllm + jaeger in a local environment

the top level aim of vllm + jaeger setup is to expose valid service through three ports in the localhost

It needs two set of information
- the model information. the user provide the model name, the details of the model for vllm serving is in the top level configs/model_config.toml. only the configured model can be served
- the port information. the user provide an index which specify the port combination configured in the top level configs/port_profiles.toml

The package is a deamon front end like architecture, each machine can only have one deamon. each deamon only allows one environment to be active.

The frontend provide a CLI based on typer. it starts the deamon if not exists. it also has a command to close the deamon
The package should not expose or document a manual runtime operation path; launch/operate/stop must go through this CLI only.

The package maintain different lanuch configs in toml
The config includes GPU type, which GPUs to use, per gpu memory, total gpu memory, and tensor parallelism for lanuch
E.g. by default we are using NVIDIA H100 NVL, GPU 2/3, 80GB each gpu, total 160GB, tp size is 2.


Goal: one local daemon per machine, one active environment per daemon.
Each environment is selected by:
- `model` key from `configs/model_config.toml`
- `port-profile` ID from `configs/port_profiles.toml`
- `launch-profile` key from `servers/servers-docker/launch_profiles.toml` (GPU/tp settings)

## Command List (Apptainer-style, Docker context)

```bash
# daemon lifecycle
python3 servers/servers-docker/client.py daemon-start
python3 servers/servers-docker/client.py daemon-status
python3 servers/servers-docker/client.py daemon-stop

# inspect configured options
python3 servers/servers-docker/client.py profiles models
python3 servers/servers-docker/client.py profiles ports
python3 servers/servers-docker/client.py profiles launches

# start one environment (auto-start daemon if needed)
# profile 0 = default ports; model must exist in configs/model_config.toml
python3 servers/servers-docker/client.py start -m qwen3_coder_30b -p 0 -l h100_nvl_gpu23 -b

# runtime checks
python3 servers/servers-docker/client.py status
python3 servers/servers-docker/client.py up
python3 servers/servers-docker/client.py wait-up --timeout-seconds 900
python3 servers/servers-docker/client.py logs -n 200
python3 servers/test/client.py --port-profile 0

# stop environment
python3 servers/servers-docker/client.py stop
python3 servers/servers-docker/client.py stop -b
python3 servers/servers-docker/client.py stop-poll
```

## Task Semantics

- `daemon-start`: ensure one daemon process exists on the host; if already running, return existing daemon info (idempotent no-op).
- `daemon-status`: report daemon liveness plus current active environment metadata (or `none`).
- `daemon-stop`: terminate daemon; if an environment is active, stop it first; if daemon is absent, return success/no-op.

- `profiles models`: list allowed model keys from `configs/model_config.toml`.
- `profiles ports`: list allowed port-profile IDs and resolved ports from `configs/port_profiles.toml`.
- `profiles launches`: list allowed launch-profile keys from `servers/servers-docker/launch_profiles.toml` (GPU selection, per-GPU memory, total memory, TP size).

- `start -m <model> -p <profile-id> -l <launch-profile>`: validate all keys, enforce single-active-environment constraint, reject if `weight_vram_gb > 0.75 * total_gpu_memory_gb`, materialize internal compose env file under daemon runtime cache, then start vLLM + Jaeger for that selection.
- `start ... -b`: same as `start`, but block until all target localhost endpoints are healthy (`vllm`, `jaeger API/UI`, `jaeger OTLP`) or timeout/failure.

- `status`: return full active environment state (selected model/profile/launch, effective ports, service URLs, and lifecycle state).
- `up`: perform immediate health checks against expected endpoints and return pass/fail with per-service detail.
- `wait-up --timeout-seconds N`: poll `up` until success or timeout `N`, then return final result.
- `logs -n N`: tail last `N` lines from vLLM/Jaeger compose logs.
- `servers/test/client.py --port-profile <id>`: run smoke checks for functional serving/tracing against ports from that profile.

- `stop`: trigger asynchronous environment teardown and return immediately.
- `stop -b`: perform blocking teardown and return only after all services are down.
- `stop-poll`: query status/result of a previously issued non-blocking `stop`.

## Minimum `start` semantics

- Reject unknown model keys (must exist in `configs/model_config.toml`).
- Reject unknown port profile IDs (must exist in `configs/port_profiles.toml`).
- Reject unknown launch profile keys (must exist in `servers/servers-docker/launch_profiles.toml`).
- Reject `start` if selected model `weight_vram_gb` exceeds `75%` of selected launch profile `total_gpu_memory_gb`.
- Reject `start` when another environment is active.
- Materialize/refresh internal compose env file from selected model + port profile.
- Bring up `vllm` + `jaeger` via Docker Compose using pushed images only (no local Dockerfile build).
- Expose localhost ports from selected profile:
  - `vllm_port`
  - `jaeger_api_port`
  - `jaeger_otlp_port`
