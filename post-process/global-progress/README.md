# Global Progress Post-Process

This directory extracts replay-progress milestones for each run.

It computes the time (seconds since run start) when the first:

- 50 replays finished
- 100 replays finished
- ...
- until all replays for that run

## Script

- `post-process/global-progress/extract_run.py`
- `post-process/global-progress/aggregate_runs_csv.py`

## Single Run

Command:

```bash
python post-process/global-progress/extract_run.py \
  --run-dir <run-dir>
```

Default output:

```text
<run-dir>/post-processed/global-progress/replay-progress-summary.json
```

Optional arguments:

- `--output <path>`
- `--milestone-step <positive-int>` (default `50`)

## Batch Mode

Command:

```bash
python post-process/global-progress/extract_run.py \
  --root-dir <root-dir>
```

Discovery rule:

- any subdirectory that has either:
  - `replay/summary.json`
  - `meta/results.json` and `meta/run_manifest.json`

Optional arguments:

- `--max-procs <positive-int>`
- `--dry-run`
- `--milestone-step <positive-int>`

## Aggregate CSV From Root (Includes Extraction)

This command extracts `global-progress` summaries for all discovered runs first,
then writes one CSV.

Command:

```bash
python post-process/global-progress/aggregate_runs_csv.py \
  --root-dir <root-dir>
```

Default output:

```text
<root-dir>/replay-progress-summary.csv
```

Optional arguments:

- `--output <csv-path>`
- `--max-procs <positive-int>`
- `--milestone-step <positive-int>`
- `--dry-run` (list discovered runs and exit)
- `--skip-extract` (aggregate existing summaries only)

CSV columns:

- `run_path`
- `source_type`
- `replay_count`
- `finished_replay_count`
- `milestone_step`
- `finish_time_s_at_<N>` for each discovered milestone count `N`

## Output Fields

- `source_run_dir`
- `source_type` (`replay` or `con-driver`)
- `experiment_started_at`
- `replay_count`
- `finished_replay_count`
- `milestone_step`
- `milestones`:
  - `replay_count`
  - `finish_time_s` (seconds since run start; `null` if insufficient finished runs)
