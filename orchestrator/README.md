## Orchestrator

`orchestrator` runs a batch of config-driven jobs across a list of port-profile
endpoints.

Supported job runners:

- `con-driver`
- `replay`

Behavior:

- accepts a directory containing job config files
- for replay jobs, checks gateway health (`/healthz`) for every requested port profile before launching jobs
- each port profile runs at most one job at a time
- when a profile becomes idle, the next pending job is launched on that profile
- writes per-job logs plus a final `summary.json`

## Usage

```bash
python -m orchestrator \
  --job-type replay \
  --jobs-dir experiments/sweep-concurrency/configs \
  --port-profile-id-list 0,1,2,3,4,5
```

For `con-driver` jobs:

```bash
python -m orchestrator \
  --job-type con-driver \
  --jobs-dir experiments/single-agent/configs \
  --port-profile-id-list 0,1,2
```

Key options:

- `--config-glob`: file matching pattern inside `--jobs-dir` (default: `*.toml`)
- `--output-dir`: where logs + `summary.json` are written
- `--poll-interval-s`: how often child processes are polled
- `--health-timeout-s`: timeout for gateway health checks
- `--fail-fast`: stop launching new jobs after first failure

Each job config is launched with the selected profile via `--port-profile-id`.
