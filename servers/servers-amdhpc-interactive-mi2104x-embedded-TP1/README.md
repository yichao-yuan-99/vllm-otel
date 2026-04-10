# Interactive AMD HPC Embedded TP1 MI2104X

This directory contains:

- a bash launcher for an already allocated interactive `mi2104x` node
- a small background control CLI that can start, stop, and inspect that launcher
- a shared `experiment-env.sh` helper sourced by interactive experiment bundles

The stack follows the same interactive embedded pattern as the `mi2508x` and
`mi3008x` launchers, but partitions a four-GPU `mi2104x` node into four TP=1
service profiles:

- one shared AMD SMI power daemon for `amd-power-reader`
- shared Jaeger
- one TP=1 vLLM per port profile `0..3`, pinned to GPU `0..3`
- one ctx-aware gateway (`gateway_ctx`) per port profile `0..3`

Like the interactive `mi2508x` embedded launcher, `start-services.sh`
explicitly unrolls the four vLLM and `gateway_ctx` launches instead of deriving
them from a looped GPU list at runtime.

The raw launcher blocks in the foreground and keeps the services alive until you
stop it with `Ctrl-C`. The background CLI wraps that launcher with a pidfile and
launcher log so you can stop it later from another shell.

## Run

From an interactive allocation on a `mi2104x` node:

```bash
bash servers/servers-amdhpc-interactive-mi2104x-embedded-TP1/start-services.sh
```

The foreground launcher writes logs under
`servers/servers-amdhpc-interactive-mi2104x-embedded-TP1/logs/`.
It starts one `amd-smi-power-daemon`, waits for its Unix socket to become
ready, launches shared Jaeger, then starts all four TP=1 vLLM and
`gateway_ctx` pairs for profiles `0..3`.

## Background Control

Start the stack in the background:

```bash
python3 servers/servers-amdhpc-interactive-mi2104x-embedded-TP1/client.py start
```

`start` is blocking: it prints launcher progress while waiting and returns only
after the launcher logs `Services are ready` or fails during startup.

Stop it later:

```bash
python3 servers/servers-amdhpc-interactive-mi2104x-embedded-TP1/client.py stop
```

`stop` is also blocking: it prints shutdown progress while waiting for the
background launcher to exit after it tears down Jaeger, all vLLM processes, and
all `gateway_ctx` processes.

Check whether it is still running:

```bash
python3 servers/servers-amdhpc-interactive-mi2104x-embedded-TP1/client.py status
```

The background wrapper keeps its pid record at
`servers/servers-amdhpc-interactive-mi2104x-embedded-TP1/logs/background-service.pid.json`
and writes launcher stdout/stderr to
`servers/servers-amdhpc-interactive-mi2104x-embedded-TP1/logs/launcher.<run_id>.log`.
The underlying launcher also writes:

- `amd-smi-power-daemon.<run_id>.log`
- `jaeger.<run_id>.shared.log`
- `vllm.<run_id>.p<profile>.log`
- `gateway.<run_id>.p<profile>.log`

## Default Ports

- Jaeger UI/API: `16686`
- Jaeger OTLP gRPC: `4317`
- Profile `0`: vLLM `11451`, gateway `11457`, gateway parse `18171`
- Profile `1`: vLLM `24123`, gateway `24157`, gateway parse `28171`
- Profile `2`: vLLM `31987`, gateway `31955`, gateway parse `38171`
- Profile `3`: vLLM `40823`, gateway `40857`, gateway parse `48171`

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
bash servers/servers-amdhpc-interactive-mi2104x-embedded-TP1/start-services.sh
```

The same environment overrides also apply when using `client.py start`, for
example:

```bash
python3 servers/servers-amdhpc-interactive-mi2104x-embedded-TP1/client.py start
```

By default the launcher resolves `VLLM_MODEL_KEY` to `qwen3_14b`, which maps to
`Qwen/Qwen3-14B-FP8` in `configs/model_config.toml`. You can still override
that with `VLLM_MODEL_KEY`, `VLLM_MODEL_NAME`, or
`VLLM_MODEL_EXTRA_ARGS_B64` when needed.

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

- Defaults match the current interactive AMD HPC embedded launcher settings
  where practical, including the same AMD power daemon, shared Jaeger, ports,
  and OTEL/`gateway_ctx` wiring.
- The default vLLM image points to
  `vllm-vllm-openai-rocm:v0.17.1-otel-lp-rocm-lmcache-gfx942.sif`; override
  `VLLM_SIF` if you need a different build.
- Service ports stay fixed to embedded port profiles `0..3`.
- Profile `n` is pinned with `ROCR_VISIBLE_DEVICES=n` while
  `HIP_VISIBLE_DEVICES=0` stays fixed inside each container.
- Unlike the sbatch job, this script still does not launch any experiment
  script.
