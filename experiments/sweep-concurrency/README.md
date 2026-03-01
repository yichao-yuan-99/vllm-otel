# Sweep Concurrency

This experiment now uses a single source workload and replay-based scaling.

Strategy:

1. Run `con-driver` once at `max_concurrent = 15` to generate the source job.
2. Compile `replay-plan.json` from that source job.
3. Replay the exact same worker/request structure with launch-policy overlays
   that change only `max_concurrent` to:
   - `30`
   - `60`
   - `90`
   - `120`

This avoids workload drift across concurrency points. The recorded LLM request
sequence comes from one source job, and replay changes only the launch policy.

## Files

Only these files are needed now:

- `configs/config.15.toml`
  - the source `con-driver` run at concurrency `15`
- `run_source.sh`
  - part A: runs the source `con-driver` job at concurrency `15`
- `run_replay.sh`
  - part B: compiles replay and launches the scaled replay runs

## Preconditions

Before running the sweep:

1. Pick the target `port_profile_id` and start the matching `vLLM` server and gateway.
2. Make sure Harbor is installed and the repo `.venv` is usable.
3. Expect outputs under `experiments/results/sweep-concurrency/`.

## Source Workload

The source config is:

```text
experiments/sweep-concurrency/configs/config.15.toml
```

It runs:

- `pool = "swebench-verified"`
- `pattern = "eager"`
- `max_concurrent = 15`
- `n_task = 300`
- `sample_without_replacement = true`
- `agent = "terminus-2"`

The source run writes under:

```text
experiments/results/sweep-concurrency/15/
```

## Part A: Record The Source Workload

Run the source `con-driver` job:

```bash
bash experiments/sweep-concurrency/run_source.sh --port-profile-id 3
```

This produces the source job under:

```text
experiments/results/sweep-concurrency/15/job-<timestamp>/
```

## Part B: Replay The Same Workload At Higher Concurrency

Replay from the latest `15`-concurrency source job:

```bash
bash experiments/sweep-concurrency/run_replay.sh --port-profile-id 3
```

Or replay from a specific source job:

```bash
bash experiments/sweep-concurrency/run_replay.sh \
  --port-profile-id 3 \
  --source-job-dir experiments/results/sweep-concurrency/15/job-20260301T000000Z
```

This does:

1. compile `replay-plan.json` inside the source job directory
2. replay the exact same workload at `30`, `60`, `90`, and `120`

## Replay Overlay

For each replay target, the script applies this style of launch-policy overlay:

```json
{
  "max_concurrent": 60,
  "seed": null,
  "pattern": {
    "name": "eager"
  },
  "pattern_args": {}
}
```

Only the launch policy changes. The replay workers and their recorded request
payloads stay identical to the source job.

## Output Layout

The source run goes under:

```text
experiments/results/sweep-concurrency/15/job-<timestamp>/
```

Replay runs go under:

```text
experiments/results/sweep-concurrency/30/<source-job>.replayed-c30/
experiments/results/sweep-concurrency/60/<source-job>.replayed-c60/
experiments/results/sweep-concurrency/90/<source-job>.replayed-c90/
experiments/results/sweep-concurrency/120/<source-job>.replayed-c120/
```

Useful files:

- source run:
  - `meta/run_manifest.json`
  - `meta/events.jsonl`
  - `meta/results.json`
  - `replay-plan.json`
- replay run:
  - `replay/summary.json`
  - `replay/workers/*.json`

## Notes

- This experiment intentionally does not run an independent `con-driver` sample
  at `30/60/90/120`.
- `run_source.sh` and `run_replay.sh` both require `--port-profile-id`.

## Why This Strategy Fits The Goal

- The goal is only to investigate the serving system, not the full adaptive
  agent workflow.
- For that goal, the narrow scope is useful: the source workload is fixed, and
  only replay scheduling changes.
- That makes the comparison cleaner because prompt mix, worker structure, and
  request payloads stay constant across `15/30/60/90/120`.
- Differences are therefore much easier to attribute to serving behavior rather
  than to sampling drift or agent-level feedback loops.
- This is intentionally a serving-system scaling study, not a fresh end-to-end
  agent benchmark at each concurrency point.

## Determinism Note

- Even with `temperature = 0`, LLM outputs may still drift across runs.
- Common causes include batching differences, floating-point nondeterminism,
  GPU kernel behavior, cache state, and backend implementation details.
- That is another reason to keep one recorded source workload and scale it via
  replay: it removes workload drift even though model-serving nondeterminism can
  still exist.
