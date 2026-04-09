# AMD SMI Power Daemon

This directory contains a small Unix-domain-socket daemon and client for reading AMD GPU power with the ROCm `amdsmi` Python API.

The daemon initializes AMD SMI once at startup with `amdsmi_init()`, caches processor handles from `amdsmi_get_processor_handles()`, and serves on-demand power requests by calling `amdsmi_get_power_info()`. On shutdown it calls `amdsmi_shut_down()`.

## Files

- `amd_smi_power_daemon.py`: starts the daemon process
- `amd_smi_power_client.py`: requests a GPU power sample from the daemon
- `amdsmi_power_service.py`: shared AMD SMI and IPC logic
- `get_amd_gpu_clockfreq.py`: reads the current GPU core or memory clock
- `reset_amd_gpu_sclk.py`: resets sclk by locking it to 1700 MHz
- `set_amd_gpu_clockfreq.py`: bridges to the existing GPU clock shell script
- `test_get_amd_gpu_clockfreq.py`: mocked tests for clock reads
- `test_reset_amd_gpu_sclk.py`: mocked tests for sclk reset
- `test_set_amd_gpu_clockfreq.py`: mocked tests for the shell-script bridge
- `test_amdsmi_power_service.py`: mocked IPC round-trip tests

## Requirements

- ROCm AMD SMI Python bindings installed so `import amdsmi` works
- Linux/Unix environment with Unix domain sockets

## Start The Daemon

```bash
python amd-power-reader/deamon/amd_smi_power_daemon.py
```

By default, the daemon listens on:

```text
/tmp/amdsmi-power-reader.sock
```

You can override that path:

```bash
python amd-power-reader/deamon/amd_smi_power_daemon.py \
  --socket-path /tmp/custom-amdsmi.sock
```

## Query Power From The Client

Read GPU 0 power:

```bash
python amd-power-reader/deamon/amd_smi_power_client.py --gpu-index 0 --pretty
```

Print only the watt value:

```bash
python amd-power-reader/deamon/amd_smi_power_client.py --gpu-index 0 --power-only
```

List the GPUs cached by the daemon:

```bash
python amd-power-reader/deamon/amd_smi_power_client.py --list-gpus --pretty
```

## Frequency Bridge Command

This directory also includes a thin bridge command for the existing system script:

```text
/usr/local/bin/set_gpu_clockfreq.sh
```

The Python wrapper does not implement frequency logic itself. It only translates repo-friendly CLI flags into the shell script arguments.
By default it executes the script as `sudo /usr/local/bin/set_gpu_clockfreq.sh` because the system script requires elevated privileges here.

### Usage

Set max core clock on GPU 0:

```bash
python amd-power-reader/deamon/set_amd_gpu_clockfreq.py \
  --clock sclk \
  --limit max \
  --value 1500 \
  --gpu-index 0
```

Set min memory clock on all GPUs:

```bash
python amd-power-reader/deamon/set_amd_gpu_clockfreq.py \
  --clock mclk \
  --limit min \
  --value 900
```

Use `--no-sudo` only if you intentionally want to bypass that default for a different script or environment.

### Argument Mapping

- `--clock sclk` -> `-clock sclk`
- `--clock mclk` -> `-clock mclk`
- `--limit min` -> `-limit min`
- `--limit max` -> `-limit max`
- `--value N` -> `-value N`
- `--gpu-index N` -> `-gpu N`
- omit `--gpu-index` -> script applies to all GPUs

## Reset SCLK Command

This directory also includes a convenience reset command for sclk. In this project, "reset sclk" means locking both the min and max sclk limits to `1700` MHz.

### Usage

Reset GPU 0 sclk to 1700 MHz:

```bash
python amd-power-reader/deamon/reset_amd_gpu_sclk.py --gpu-index 0
```

Reset sclk to 1700 MHz on all GPUs:

```bash
python amd-power-reader/deamon/reset_amd_gpu_sclk.py
```

Use `--no-sudo` only if you intentionally want to bypass that default for a different script or environment.

### What It Runs

For GPU `N`, the wrapper runs these two commands in order:

```text
sudo /usr/local/bin/set_gpu_clockfreq.sh -clock sclk -limit max -value 1700 -gpu N
sudo /usr/local/bin/set_gpu_clockfreq.sh -clock sclk -limit min -value 1700 -gpu N
```

## Frequency Read Command

This directory also includes a read-side command that uses the AMD SMI Python API directly to report the current clock frequency.

### Usage

Read current core clock for GPU 0:

```bash
python amd-power-reader/deamon/get_amd_gpu_clockfreq.py \
  --clock sclk \
  --gpu-index 0 \
  --pretty
```

Read current memory clock for all GPUs:

```bash
python amd-power-reader/deamon/get_amd_gpu_clockfreq.py \
  --clock mclk \
  --pretty
```

Print only the current MHz for one GPU:

```bash
python amd-power-reader/deamon/get_amd_gpu_clockfreq.py \
  --clock sclk \
  --gpu-index 0 \
  --mhz-only
```

### Output

For a single GPU, the command returns JSON with:

- `current_mhz`: current clock in MHz
- `min_mhz`: current lower bound in MHz
- `max_mhz`: current upper bound in MHz
- `clk_locked`: whether the clock is currently locked
- `current_level`: current DPM level index from `amdsmi_get_clk_freq()`
- `supported_mhz`: supported frequencies in MHz

## Public IPC Interface

The daemon exposes a newline-delimited JSON protocol over a Unix domain socket.

- Default socket path: `/tmp/amdsmi-power-reader.sock`
- Transport: `AF_UNIX` stream socket
- Encoding: UTF-8 JSON
- Framing: one JSON object per line

The bundled client is just a thin wrapper around these IPC commands.

### Commands

`ping`

Request:

```json
{"command": "ping"}
```

Response:

```json
{"ok": true, "timestamp": "2026-04-07T18:30:00.123456Z", "gpu_count": 4}
```

`list_gpus`

Request:

```json
{"command": "list_gpus"}
```

Response fields:

- `ok`: success flag
- `timestamp`: daemon response timestamp in UTC
- `gpus`: cached GPU metadata from AMD SMI enumeration

`get_power`

Request:

```json
{"command": "get_power", "gpu_index": 0}
```

Response fields:

- `ok`: success flag
- `timestamp`: daemon response timestamp in UTC
- `gpu`: metadata for the selected GPU
- `power_w`: selected socket power in watts, when available
- `power_info`: raw `amdsmi_get_power_info()` payload

Error responses use the same JSON-line framing:

```json
{
  "ok": false,
  "timestamp": "2026-04-07T18:30:00.123456Z",
  "error": "gpu_index 99 out of range; found 4 GPU(s)",
  "error_type": "IndexError"
}
```

### Raw IPC Example

This example sends `get_power` directly to the daemon without using `amd_smi_power_client.py`:

```bash
python - <<'PY'
import json
import socket

sock_path = "/tmp/amdsmi-power-reader.sock"
request = {"command": "get_power", "gpu_index": 0}

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect(sock_path)
    client.sendall(json.dumps(request).encode("utf-8") + b"\n")
    response = client.recv(65536).decode("utf-8").strip()

print(response)
PY
```

### Client To IPC Mapping

- `amd_smi_power_client.py --list-gpus` -> `{"command": "list_gpus"}`
- `amd_smi_power_client.py --gpu-index N` -> `{"command": "get_power", "gpu_index": N}`

## Example Response

```json
{
  "ok": true,
  "timestamp": "2026-04-07T18:30:00.123456Z",
  "gpu": {
    "index": 0,
    "bdf": "0000:83:00.0",
    "uuid": "deff740f-0000-1000-8082-87300da37c72",
    "hip_id": 3,
    "hip_uuid": "GPU-de8287300da37c72",
    "drm_card": 3,
    "drm_render": 131,
    "hsa_id": 5
  },
  "power_w": 40.0,
  "power_info": {
    "socket_power": 40,
    "current_socket_power": "N/A",
    "average_socket_power": 40,
    "gfx_voltage": "N/A",
    "soc_voltage": "N/A",
    "mem_voltage": "N/A",
    "power_limit": 300000000
  }
}
```

## Run Tests

```bash
pytest \
  amd-power-reader/deamon/test_get_amd_gpu_clockfreq.py \
  amd-power-reader/deamon/test_reset_amd_gpu_sclk.py \
  amd-power-reader/deamon/test_set_amd_gpu_clockfreq.py \
  amd-power-reader/deamon/test_amdsmi_power_service.py
```

## AMD Docs

- https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-api.html
