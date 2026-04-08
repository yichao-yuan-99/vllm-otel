# AMD Power Reader

A Python utility for streaming AMD GPU power readings to a JSONL file through the local AMD SMI daemon, plus AMD GPU core clock helpers.

## Overview

This package provides:

1. `amd-power-reader` - Poll the local AMD SMI daemon and write GPU power to `power-log.jsonl`
2. `amd-set-gpu-core-freq` - Set AMD GPU sclk min, max, or both
3. `amd-reset-gpu-core-freq` - Reset AMD GPU sclk to the project default of `1700` MHz
4. `amd-smi-power-daemon` - Start the local AMD SMI Unix-socket daemon
5. `amd-smi-power-client` - Query the daemon directly

## Prerequisites

### AMD SMI Python Bindings

The daemon uses the ROCm `amdsmi` Python bindings. They must already be available on the system.

### GPU Clock Script

Core clock commands bridge to:

```text
/usr/local/bin/set_gpu_clockfreq.sh
```

The frequency helpers invoke that script through `sudo` by default because `/usr/local/bin/set_gpu_clockfreq.sh` requires elevated privileges here. Pass `--no-sudo` only if you intentionally want to run a custom script path directly.

## Installation

```bash
pip install -e ./amd-power-reader
```

## Start The Daemon

Start the AMD SMI daemon before using `amd-power-reader`:

```bash
amd-smi-power-daemon
```

Optional custom socket:

```bash
amd-smi-power-daemon --socket-path /tmp/custom-amdsmi.sock
```

## Power Reader

Stream AMD GPU power readings from the daemon to a JSONL file.

### Usage

```bash
# Monitor all GPUs reported by the daemon
amd-power-reader --output-dir ./logs

# Monitor specific GPUs
amd-power-reader --output-dir ./logs --gpu-indices 0 1

# Custom socket path and poll interval
amd-power-reader --output-dir ./logs --socket-path /tmp/custom-amdsmi.sock --interval-ms 250
```

### Output Format

The output file `power-log.jsonl` contains one JSON object per line:

```jsonl
{"timestamp":"2026-04-07T18:30:00.123456Z","payload":{"/tmp/amdsmi-power-reader.sock":{"timestamp_s":1775586600.123,"gpu_power_w":{"0":40.0,"1":45.0},"gpu_payload":{"0":{"gpu":{"index":0},"power_info":{"socket_power":40}},"1":{"gpu":{"index":1},"power_info":{"socket_power":45}}}}}}
```

### Power Reader Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output-dir` | `-o` | (required) | Directory to write `power-log.jsonl` |
| `--gpu-indices` | `-g` | all daemon GPUs | GPU indices to monitor |
| `--socket-path` | `-s` | `/tmp/amdsmi-power-reader.sock` | Path to the daemon socket |
| `--interval-ms` | `-i` | `100` | Polling interval in milliseconds |
| `--timeout` | `-t` | `5.0` | IPC timeout in seconds |

## GPU Frequency Control

### Set GPU Core Frequency

Set AMD GPU sclk min, max, or both:

```bash
amd-set-gpu-core-freq --gpu-index 0 --min-mhz 1200 --max-mhz 1500
amd-set-gpu-core-freq -g 0 -n 1200 -x 1500

# Max only
amd-set-gpu-core-freq --gpu-index 0 --max-mhz 1500

# Min only
amd-set-gpu-core-freq --gpu-index 0 --min-mhz 1200
```

When both limits are provided, it bridges to `/usr/local/bin/set_gpu_clockfreq.sh` twice:

```text
sudo /usr/local/bin/set_gpu_clockfreq.sh -clock sclk -limit max -value 1500 -gpu 0
sudo /usr/local/bin/set_gpu_clockfreq.sh -clock sclk -limit min -value 1200 -gpu 0
```

When only one limit is provided, only that single shell command is executed.

### Reset GPU Core Frequency

Reset AMD GPU `sclk max` to the project default of `1700` MHz:

```bash
amd-reset-gpu-core-freq --gpu-index 0
amd-reset-gpu-core-freq -g 0
```

Internally this runs:

```text
sudo /usr/local/bin/set_gpu_clockfreq.sh -clock sclk -limit max -value 1700 -gpu 0
```

## Example Workflow

```bash
# 1. Start the daemon
amd-smi-power-daemon &
DAEMON_PID=$!

# 2. Lock GPU frequency for a benchmark run
amd-set-gpu-core-freq -g 0 -n 1200 -x 1500

# 3. Start power monitoring
amd-power-reader --output-dir ./experiment-logs --gpu-indices 0 &
POWER_READER_PID=$!

# 4. Run your workload
./your-gpu-workload

# 5. Stop power monitoring and the daemon
kill $POWER_READER_PID
kill $DAEMON_PID

# 6. Reset GPU core clock
amd-reset-gpu-core-freq -g 0

# 7. Inspect the power log
cat ./experiment-logs/power-log.jsonl
```

## Programmatic Usage

```python
from pathlib import Path
from amd_power_reader import run_power_reader
from set_gpu_core_freq import main as set_core_freq
from reset_gpu_core_freq import main as reset_core_freq

set_core_freq(["-g", "0", "-n", "1200", "-x", "1500"])

run_power_reader(
    output_dir=Path("./logs"),
    gpu_indices=[0, 1],
)

reset_core_freq(["-g", "0"])
```

## More Detailed Daemon Docs

For the raw IPC contract and lower-level helper commands, see [deamon/README.md](deamon/README.md).
