# Interactive AMD HPC Embedded TP1

This directory contains:

- a bash launcher for an already allocated interactive `mi3001x` node
- a small background control CLI that can start, stop, and inspect that launcher
- a shared `experiment-env.sh` helper sourced by the interactive experiment bundles

The stack matches the rendered embedded-TP1 sbatch job minus the experiment
phase:

- one shared AMD SMI power daemon for `amd-power-reader`
- shared Jaeger
- one TP=1 vLLM on port profile `0`
- one ctx-aware gateway (`gateway_ctx`) on port profile `0`

The raw launcher blocks in the foreground and keeps the services alive until you
stop it with `Ctrl-C`. The background CLI wraps that launcher with a pidfile and
launcher log so you can stop it later from another shell.

## Run

From an interactive allocation on a `mi3001x` node:

```bash
bash servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/start-services.sh
```

The foreground launcher writes logs under
`servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/logs/`.
It also starts one `amd-smi-power-daemon` for the interactive stack and waits
for its Unix socket to become ready before launching Jaeger, vLLM, and
`gateway_ctx`.

## Background Control

Start the stack in the background:

```bash
python3 servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/client.py start
```

`start` is blocking: it prints launcher progress while waiting and returns only
after the launcher logs `Services are ready` or fails during startup.

Stop it later:

```bash
python3 servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/client.py stop
```

`stop` is also blocking: it prints shutdown progress while waiting for the
background launcher to exit after it tears down Jaeger, vLLM, and `gateway_ctx`.

Check whether it is still running:

```bash
python3 servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/client.py status
```

The background wrapper keeps its pid record at
`servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/logs/background-service.pid.json`
and writes launcher stdout/stderr to
`servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/logs/launcher.<run_id>.log`.
The underlying launcher still writes the AMD daemon log to
`servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/logs/amd-smi-power-daemon.<run_id>.log`.

## Default Ports

- vLLM: `11451`
- gateway: `11457`
- gateway parse: `18171`
- Jaeger UI/API: `16686`
- Jaeger OTLP gRPC: `4317`

## Useful Overrides

- `VLLM_MODEL_KEY`
- `MODEL_CONFIG_PATH`
- `VLLM_MODEL_NAME`
- `VLLM_SERVED_MODEL_NAME`
- `VLLM_MODEL_EXTRA_ARGS_B64`
- `VLLM_SIF`
- `JAEGER_SIF`
- `AMD_SMI_POWER_DAEMON_BIN`
- `AMD_SMI_POWER_SOCKET_PATH`
- `HF_HOME`
- `HF_HUB_CACHE`
- `HF_TOKEN`
- `GATEWAY_HOST`
- `GATEWAY_CONFIG`
- `GATEWAY_VENV_DIR`
- `RUN_ID`

Example:

```bash
VLLM_MODEL_KEY=qwen3_coder_30b_fp8 \
bash servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/start-services.sh
```

The same environment overrides also apply when using `client.py start`, for
example:

```bash
VLLM_MODEL_KEY=qwen3_coder_30b_fp8 \
python3 servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/client.py start
```

`VLLM_MODEL_KEY` resolves `VLLM_MODEL_NAME`, `VLLM_SERVED_MODEL_NAME`, and the
default `VLLM_MODEL_EXTRA_ARGS_B64` from
`configs/model_config.toml`. If you need a raw Hugging Face path that is not in
`configs/model_config.toml`, you can still set `VLLM_MODEL_NAME` directly and
pair it with `VLLM_SERVED_MODEL_NAME`.

When the stack is ready it prints the active `AMD_SMI_POWER_SOCKET_PATH`. By
default that is `/tmp/amdsmi-power-reader.sock`, and you can still override it
in the environment before launch if you need a different socket. You can pass
that value to `amd-power-reader --socket-path ...` from another shell.

## Prerequisite

Install the AMD power tools on the node-visible Python environment:

```bash
pip install -e ./amd-power-reader
pip install -e ./gateway_ctx
```

## Notes

- Defaults match the current `mi3001x` embedded-TP1 sbatch render where
  practical, including the same AMD power daemon, SIF paths, ports, and
  OTEL/`gateway_ctx` wiring.
- Service ports stay fixed to embedded port profile `0`.
- Unlike the sbatch job, this script still does not launch any experiment
  script.
