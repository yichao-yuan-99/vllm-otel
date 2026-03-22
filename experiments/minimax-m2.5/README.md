# MiniMax-M2.5 Experiments

This directory contains con-driver configs and helper scripts for running
MiniMax-M2.5 across:

- `swebench-verified`
- `terminal-bench@2.0`
- `livecodebench`
- `dabstep`

## Files

- `configs/config.swebench-verified.mswe.toml`: swebench-verified with mini-swe-agent.
- `configs/config.swebench-verfied.terminus2.toml`: swebench-verified with terminus-2.
- `configs/config.terminal-bench-2.0.mswe.toml`: terminal-bench@2.0 with mini-swe-agent.
- `configs/config.terminal-bench-2.0.terminus2.toml`: terminal-bench@2.0 with terminus-2.
- `configs/config.livecodebench.mswe.toml`: livecodebench with mini-swe-agent.
- `configs/config.livecodebench.terminus2.toml`: livecodebench with terminus-2.
- `configs/config.dabstep.mswe.toml`: dabstep with mini-swe-agent.
- `configs/config.dabstep.terminus2.toml`: dabstep with terminus-2.
- `start.py`: Python start script for one benchmark + one agent run.
- `batch_benchmarks.sh`: very naive batch runner with hardcoded benchmarks.

Note: the terminus swebench config filename intentionally keeps `verfied`.
`start.py` accepts both `terminal-bench@2.0` and `terminalbench@2.0`.

## Preconditions

1. Profiles `0,1,2,3,4` are up and serving the same MiniMax-M2.5 model.
2. Gateway is up for the same profile set.
3. Run commands from repo root.

## Run One Experiment (Single Benchmark + Single Agent)

Use per-profile concurrency:

```bash
python experiments/minimax-m2.5/start.py \
  --benchmark swebench-verified \
  --agent mini-swe-agent \
  --per-profile-conc 5
```

Or provide an explicit list:

```bash
python experiments/minimax-m2.5/start.py \
  --benchmark livecodebench \
  --agent terminus-2 \
  --port-profile-id-list 0,1,2,3,4 \
  --max-concurrent-list 5,5,5,5,5
```

Dry run:

```bash
python experiments/minimax-m2.5/start.py \
  --benchmark dabstep \
  --agent mini-swe-agent \
  --per-profile-conc 5 \
  --dry-run
```

## Run Batch (Hardcoded Benchmarks)

```bash
# Hardcoded benchmarks: swebench-verified, terminal-bench@2.0, livecodebench, dabstep
# For each benchmark: mini-swe-agent then terminus-2
bash experiments/minimax-m2.5/batch_benchmarks.sh \
  --per-profile-conc 5
```

## Output Layout

All configs use:

`experiments/results/minimax-m2.5/<benchmark>/<agent>/`

Default roots:

- `experiments/results/minimax-m2.5/swebench-verified/mini-swe-agent/`
- `experiments/results/minimax-m2.5/swebench-verified/terminus-2/`
- `experiments/results/minimax-m2.5/terminal-bench-2.0/mini-swe-agent/`
- `experiments/results/minimax-m2.5/terminal-bench-2.0/terminus-2/`
- `experiments/results/minimax-m2.5/livecodebench/mini-swe-agent/`
- `experiments/results/minimax-m2.5/livecodebench/terminus-2/`
- `experiments/results/minimax-m2.5/dabstep/mini-swe-agent/`
- `experiments/results/minimax-m2.5/dabstep/terminus-2/`

Each invocation creates a timestamped run directory with `meta/`, `logs/`,
`trials/`, `gateway-output/`, and optional `vllm-log/` artifacts.
