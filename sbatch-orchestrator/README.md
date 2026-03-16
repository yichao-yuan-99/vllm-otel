# sbatch-orchestrator

`sbatch-orchestrator` is an optional runtime mode for grouped sbatch scripts rendered by:

- `servers/servers-amdhpc/render-sbatch.py start-group ...`

Recommended setup: render grouped sbatch with `--local-mode ./sbatch-orchestrator/entrypoint.sh`.

When enabled, the grouped sbatch job still starts one vLLM worker per selected port profile, then:

1. waits for all grouped vLLM ports to become ready,
2. runs `sbatch-orchestrator/entrypoint.sh`,
3. assigns listed shell-script jobs to free profile/port slots,
4. passes the assigned port as positional arg 1 to each script,
5. exits after the job list is finished (and stops grouped workers).

If orchestration env vars are not provided, grouped sbatch behavior is unchanged.

## Files

- `sbatch-orchestrator/orchestrator.py`: Python scheduler.
- `sbatch-orchestrator/entrypoint.sh`: stable sbatch-facing launcher.
- `sbatch-orchestrator/render-start-group.sh`: wrapper around `render-sbatch.py start-group` that copies rendered sbatch to a fixed path.
- `sbatch-orchestrator/render-and-submit-start-group.sh`: convenience wrapper that runs render + submit in one command.
- `sbatch-orchestrator/submit-start-group.sh`: submission wrapper that snapshots sbatch/job-list and redirects logs into a timestamped run directory.

## Job List Format

Set `SBATCH_ORCHESTRATOR_JOB_LIST` to a file with one job per line.

- Empty lines and lines starting with `#` are ignored.
- Each non-empty line must start with an absolute script path.
- Optional extra args may follow.
- Relative script paths are rejected.

Example:

```text
# /abs/path/to/script [optional extra args...]
/home/user/jobs/replay_qps_010.sh
/home/user/jobs/replay_qps_020.sh --seed 7
/home/user/jobs/replay_qps_040.sh
```

Each script is launched as:

```bash
bash /abs/path/to/script.sh <assigned_port> [extra args...]
```

The child process also gets:

- `PORT_PROFILE_ID`
- `VLLM_PORT`
- `VLLM_SERVICE_PORT`
- `SBATCH_ORCHESTRATOR_SLOT_INDEX`
- `SBATCH_ORCHESTRATOR_JOB_INDEX`

## End-to-End Usage

1. Build a job list file with absolute script paths (example `jobs.txt`).
2. Render + submit in one command.

Without LMCache:

```bash
bash sbatch-orchestrator/render-and-submit-start-group.sh \
  --job-list path/to/jobs.txt \
  -g bench_orch \
  -L 0,1,2,3,4,5,6,7 \
  -p mi3008x \
  -m qwen3_coder_30b
```

Half mode (use only even grouped profile IDs and their aligned ports: `0,2,4,6`):

```bash
bash sbatch-orchestrator/render-and-submit-start-group.sh \
  --job-list path/to/jobs.txt \
  --half \
  -g bench_orch \
  -L 0,1,2,3,4,5,6,7 \
  -p mi3008x \
  -m qwen3_coder_30b
```

Or, with LMCache enabled:

```bash
bash sbatch-orchestrator/render-and-submit-start-group.sh \
  --job-list path/to/jobs.txt \
  -g bench_orch \
  -L 0,1,2,3,4,5,6,7 \
  -p mi3008x \
  -m qwen3_coder_30b \
  --lmcache 100
```

This creates `sbatch-orchestrator/logs/<utc-timestamp>/` with:

- copied `start-group.sbatch.sh` (patched `#SBATCH --output`, `#SBATCH --error`, and `JOB_LOG_DIR`)
- copied `job-list.txt`
- `job-list.path.txt`
- `orchestrator-summary.json`
- `sbatch.submit.out`
- `slurm_job_id.txt` (when parseable)
- `submission-info.json`

Manual submission is still supported:

```bash
SBATCH_ORCHESTRATOR_JOB_LIST=path/to/jobs.txt \
sbatch ./sbatch-orchestrator/start-group.sbatch.sh
```

`render-start-group.sh` forwards all args to `render-sbatch.py start-group`.
If you do not pass `--local-mode`, it automatically injects:

- `--local-mode ./sbatch-orchestrator/entrypoint.sh`

It also rewrites the render JSON output so `data.sbatch_script` points to the fixed file:

- `./sbatch-orchestrator/start-group.sbatch.sh`

## Optional Env Vars

- `SBATCH_ORCHESTRATOR_PROFILE_IDS_CSV`: defaults to grouped `GROUP_PROFILE_IDS_CSV`.
- `SBATCH_ORCHESTRATOR_PORTS_CSV`: defaults to grouped `GROUP_VLLM_PORTS_CSV`.
- `SBATCH_ORCHESTRATOR_POLL_INTERVAL_S`: scheduler poll interval (default `1.0`).
- `SBATCH_ORCHESTRATOR_FAIL_FAST=1`: stop launching new jobs after first failure.
- `SBATCH_ORCHESTRATOR_SUMMARY_PATH`: optional override for summary JSON path.
  Default path when unset: `sbatch-orchestrator/logs/orchestrator-summary.<SLURM_JOB_ID>.json`
  (the `submit-start-group.sh` wrapper sets this explicitly to its run directory).
- `SBATCH_ORCHESTRATOR_READY_TIMEOUT_SECONDS`: sbatch-side readiness timeout before launching orchestrator (default `900`).
- `SBATCH_ORCHESTRATOR_READY_POLL_INTERVAL_SECONDS`: readiness poll interval (default `2`).

## Notes

- This mode is for absolute-path script execution inside grouped sbatch jobs.
- Port/profile assignment comes from grouped render output (`--profile-list` order).
- Exit code is non-zero when any job fails (or orchestration is interrupted).
- `render-start-group.sh` auto-injects `--local-mode ./sbatch-orchestrator/entrypoint.sh` when omitted.
- `submit-start-group.sh` is recommended when you want reproducible per-run artifacts under `sbatch-orchestrator/logs/<utc-timestamp>/`.
- `--half` is supported by `render-and-submit-start-group.sh` and `submit-start-group.sh`.
  It patches the submitted sbatch copy to keep only even grouped profile IDs and aligned profile/port/gateway/Jaeger/LMCache mappings, and updates `GROUP_SIZE`/`#SBATCH --ntasks` accordingly.
