# Zeus Power Reader

A Python utility for streaming GPU/CPU power readings from [zeusd](https://github.com/ml-energy/zeus) daemon to a JSONL file with microsecond-resolution ISO timestamps, plus GPU frequency control utilities.

## Overview

This package provides:

1. **zeus-power-reader** - Stream power readings to JSONL file
2. **set-gpu-core-freq** - Lock GPU core clock frequency
3. **set-gpu-mem-freq** - Lock GPU memory clock frequency
4. **reset-gpu-core-freq** - Reset GPU core clock to defaults
5. **reset-gpu-mem-freq** - Reset GPU memory clock to defaults

## Prerequisites

### Start zeusd Daemon

The zeusd daemon must be running:

```bash
sudo zeusd serve --socket-path /var/run/zeusd.sock --socket-permissions 666
```

> **Note:** All utilities automatically set the `ZEUSD_SOCK_PATH` environment variable before importing the zeus package, so no additional environment configuration is needed.

## Installation

```bash
pip install -e ./zeus-power-reader
```

---

## Power Reader

Stream power readings from zeusd to a JSONL file.

### Usage

```bash
# Basic usage - monitor GPUs 0,1,2,3
zeus-power-reader --output-dir ./logs

# Custom GPU indices
zeus-power-reader --output-dir ./logs --gpu-indices 0 1

# Custom socket path
zeus-power-reader --output-dir ./logs --socket-path /custom/path/zeusd.sock
```

### Output Format

The output file `power-log.jsonl` contains one JSON object per line:

```jsonl
{"timestamp": "2026-03-17T04:50:16.123Z", "payload": {"/var/run/zeusd.sock": {"timestamp_s": 1710641416.123, "gpu_power_w": {"0": 125.5, "1": 98.3, "2": 110.2, "3": 87.1}, "cpu_power_w": {}}}}
{"timestamp": "2026-03-17T04:50:16.234Z", "payload": {"/var/run/zeusd.sock": {"timestamp_s": 1710641416.234, "gpu_power_w": {"0": 126.1, "1": 97.8, "2": 111.0, "3": 86.5}, "cpu_power_w": {}}}}
```

### Power Reader Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output-dir` | `-o` | (required) | Directory to write `power-log.jsonl` |
| `--gpu-indices` | `-g` | `0 1 2 3` | GPU indices to monitor |
| `--socket-path` | `-s` | `/var/run/zeusd.sock` | Path to zeusd socket |

---

## GPU Frequency Control

### Set GPU Core Frequency

Lock GPU core clock to a specific frequency range:

```bash
# Lock GPU 0 core clock to 1200-1500 MHz
set-gpu-core-freq --gpu-index 0 --min-mhz 1200 --max-mhz 1500

# Short form
set-gpu-core-freq -g 0 -n 1200 -x 1500
```

### Set GPU Memory Frequency

Lock GPU memory clock to a specific frequency range:

```bash
# Lock GPU 0 memory clock to 5000-6000 MHz
set-gpu-mem-freq --gpu-index 0 --min-mhz 5000 --max-mhz 6000

# Short form
set-gpu-mem-freq -g 0 -n 5000 -x 6000
```

### Reset GPU Core Frequency

Reset GPU core clock to default (unlocked):

```bash
reset-gpu-core-freq --gpu-index 0

# Short form
reset-gpu-core-freq -g 0
```

### Reset GPU Memory Frequency

Reset GPU memory clock to default (unlocked):

```bash
reset-gpu-mem-freq --gpu-index 0

# Short form
reset-gpu-mem-freq -g 0
```

### Frequency Control Options

| Option | Short | Description |
|--------|-------|-------------|
| `--gpu-index` | `-g` | GPU index to configure (required) |
| `--min-mhz` | `-n` | Minimum clock in MHz (required for set) |
| `--max-mhz` | `-x` | Maximum clock in MHz (required for set) |

---

## Example Workflow

```bash
# 1. Start zeusd daemon
sudo zeusd serve --socket-path /var/run/zeusd.sock --socket-permissions 666

# 2. Lock GPU frequencies for consistent benchmarking
set-gpu-core-freq -g 0 -n 1200 -x 1500
set-gpu-mem-freq -g 0 -n 5000 -x 6000

# 3. Start power monitoring in background
zeus-power-reader --output-dir ./experiment-logs &
POWER_READER_PID=$!

# 4. Run your experiment
./your-gpu-workload

# 5. Stop power monitoring
kill $POWER_READER_PID

# 6. Reset frequencies to defaults
reset-gpu-core-freq -g 0
reset-gpu-mem-freq -g 0

# 7. Analyze power logs
cat ./experiment-logs/power-log.jsonl
```

---

## Programmatic Usage

You can also use the modules programmatically:

```python
from pathlib import Path
from zeus_power_reader import run_power_reader
from set_gpu_core_freq import main as set_core_freq
from reset_gpu_core_freq import main as reset_core_freq

# Set GPU frequency
set_core_freq(["-g", "0", "-n", "1200", "-x", "1500"])

# Stream power readings
run_power_reader(
    output_dir=Path("./logs"),
    gpu_indices=[0, 1, 2, 3],
)

# Reset GPU frequency
reset_core_freq(["-g", "0"])
```

---

## Troubleshooting

### "Cannot reach the Zeusd daemon"

- Verify zeusd is running: `sudo systemctl status zeusd` or `ps aux | grep zeusd`
- Check socket permissions: `ls -la /var/run/zeusd.sock`
- Verify the socket path matches

### "Permission denied" when accessing socket

```bash
# Restart zeusd with 666 permissions
sudo zeusd serve --socket-path /var/run/zeusd.sock --socket-permissions 666
```

### ImportError for zeus module

```bash
pip install zeus
```

### GPU frequency setting fails

- Ensure you have root/sudo privileges (required for GPU frequency control)
- Check GPU indices are valid: `nvidia-smi`
- Verify frequency ranges are supported by your GPU

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     zeus-power-reader package                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    Unix Domain Socket    ┌─────────────┐   │
│  │  zeus-power-   │  ══════════════════════► │   zeusd     │   │
│  │  reader         │   /var/run/zeusd.sock    │   daemon    │   │
│  └─────────────────┘                          └─────────────┘   │
│          │                                            │         │
│          ▼                                            ▼         │
│  ┌─────────────────┐                          ┌─────────────┐   │
│  │ power-log.jsonl │                          │   GPUs      │   │
│  │ (output file)   │                          │   CPUs      │   │
│  └─────────────────┘                          └─────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              GPU Frequency Control                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │   │
│  │  │ set-gpu- │  │ set-gpu- │  │ reset-   │  │ reset-   │ │   │
│  │  │ core-freq│  │ mem-freq │  │ gpu-core │  │ gpu-mem  │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## License

Same as the parent project.
