# Concurrent Trial Driver

`con-driver` is a standalone package for launching many `harbor trials start` runs concurrently.

It downloads one or more Harbor datasets, builds a combined task pool, samples tasks uniformly, and launches trials in parallel using eager or Poisson arrivals.
It is gateway-aware by default: each launched agent gets a unique API token, every launch is wrapped with `/agent/start` and `/agent/end`, and each run calls `/job/start` + `/job/end`.
Gateway artifacts are stored inside each con-driver run directory under `gateway_job_output_root` (a run-local subdirectory, default `gateway-output`).
When `port_profile_id` is set, con-driver also probes the live served model from `vLLM /v1/models`, derives the Harbor model and endpoint wiring automatically, and points agents at `gateway_parse_port` by default.

## Install / Run

Preferred (shared `.venv` wrapper):

```bash
bash con-driver/run_con_driver.sh --help
```

`con-driver/run_con_driver.sh` always runs from `./.venv` and forwards all args to `con-driver`.

Direct install/run also works:

```bash
pip install -e ./con-driver
con-driver --help
python -m con_driver --help
```

Compatibility wrapper (repo-local):

```bash
python con-driver/driver.py --help
```

## Required Inputs

- `--driver-backend`: backend mode. Only `harbor` is supported.
- `--config`: optional TOML config file.
- `--pool`: comma-separated Harbor dataset specs, e.g. `swebench-verified,terminal-bench@2.0`.
- `--pattern`: `eager` or `poisson` (`possion` alias supported).
- `--pattern-args`: optional extra args for the pattern.
  - Poisson supports `--rate=<arrivals_per_second>` or `--mean-interval-s=<seconds>`.
- `--max-concurrent`: maximum concurrent trial processes.
- `--n-task`: total launches.
- `--results-dir`: root output directory.
- `--harbor-bin`: command prefix for Harbor (default: `harbor`).
- `--port-profile-id`: port profile numeric ID from [`../configs/port_profiles.toml`](../configs/port_profiles.toml).
- `--agent-name`: Harbor agent name. TOML keys `agent` and `agent_name` are also accepted.
- `--sample-without-replacement`: optional; disables repeated task sampling.
- `--vllm-log/--no-vllm-log`: optional; run a separate vLLM metrics monitor process.
- `--gateway/--no-gateway`: enable/disable gateway mode (default: enabled).
- `--gateway-url`: gateway base URL.
  With `port_profile_id`, the default is `http://127.0.0.1:<gateway_parse_port>`.
- `--gateway-job-output-root`: run-local output subdirectory sent to gateway `/job/start` (default: `gateway-output`).
- `--gateway-timeout-s`: timeout for gateway lifecycle API calls (default: `3600.0`).
  Set this above gateway `GATEWAY_JOB_END_TRACE_WAIT_SECONDS`.

Any extra args/options are forwarded to:

- `harbor trials start ...`

## Examples

Eager launch pattern:

```bash
con-driver \
  --pool="terminal-bench@2.0" \
  --pattern=eager \
  --max-concurrent=2 \
  --n-task=10 \
  --sample-without-replacement \
  --port-profile-id=1 \
  --agent-name=terminus-2 \
  --results-dir=con-driver/output/eager-run \
  --dry-run
```

Poisson launch pattern (mean 5 seconds between arrivals):

```bash
con-driver \
  --pool="swebench-verified,terminal-bench@2.0" \
  --pattern=poisson \
  --pattern-args="--mean-interval-s=5" \
  --max-concurrent=4 \
  --n-task=10 \
  --port-profile-id=1 \
  --agent-name=terminus-2 \
  --results-dir=con-driver/output/poisson-run \
  --dry-run
```

Use `--dry-run` to validate parsing and command construction without launching trials.

## TOML Config

Pass `--config <path.toml>`. The file can use top-level keys or a `[driver]` table.

```toml
[driver]
driver_backend = "harbor"
pool = "hello-world@1.0"
pattern = "eager"
max_concurrent = 1
n_task = 1
results_dir = "con-driver-test"
port_profile_id = 1
agent = "terminus-2"
dry_run = false
sample_without_replacement = true

# Optional
pattern_args = "--mean-interval-s=5"
harbor_bin = "harbor"
seed = 7
# With port_profile_id, vllm_log defaults to true and endpoint defaults to
# http://127.0.0.1:<vllm_port>/metrics.
# vllm_log = true
# vllm_log_endpoint = "http://127.0.0.1:24123/metrics"
vllm_log_interval_s = 1.0
vllm_log_timeout_s = 5.0
gateway = true
# With port_profile_id, gateway_url defaults to
# http://127.0.0.1:<gateway_parse_port>.
# gateway_url = "http://127.0.0.1:28171"
gateway_job_output_root = "gateway-output"
gateway_timeout_s = 3600.0

# Optional extra Harbor args only.
# con-driver auto-populates --agent, --model, api_base/base_url, model_info,
# and agent-specific defaults from con-driver/configs/agent_defaults.toml.
forwarded_args = [
  "--agent-kwarg", "some_other_setting=value",
]
```

Run with config:

```bash
bash con-driver/run_con_driver.sh --config con-driver/configs/config.example.toml
```

CLI flags override config values.

## Output Layout

Harbor dataset downloads are cached outside run outputs under:

- `con-driver/.cache/harbor-datasets/`
  - shared across runs
  - ignored by git

Non-dataset mode writes run artifacts into a nested subdirectory:

- `--results-dir/job-<timestamp>/{trials,logs,meta}/`: run outputs
- `--results-dir/job-<timestamp>/CON_DRIVER_OUTPUT`: con-driver marker file
- `meta/config.toml`: resolved run config snapshot
- `vllm-log/` (optional): monitor logs and compressed metrics blocks
- Gateway run output location sent to `/job/start`: `--results-dir/job-<timestamp>/<gateway_job_output_root>`

Dataset mode triggers when all of the following are true:

- `pattern=eager`
- one dataset in `--pool`
- `--sample-without-replacement`
- `n_task == dataset_size`

In dataset mode:

- `--results-dir/<dataset>-<timestamp>/{trials,logs,meta}/` stores run outputs
- `--results-dir/<dataset>-<timestamp>/CON_DRIVER_OUTPUT` marks the run directory

## Notes

- Unknown CLI args are forwarded to Harbor.
- Harbor dataset downloads are cached in `con-driver/.cache/harbor-datasets/`, not inside `results_dir`.
- `driver_backend` and `backend` config keys are both accepted (`backend` is a compatibility alias).
- `forwarded_args` and `harbor_args` config keys are both accepted (`harbor_args` is a compatibility alias).
- The driver rejects `-p/--path`, `--trial-name`, and `--trials-dir` in forwarded args because it manages those fields.
- With `port_profile_id`, con-driver probes the single live served model from `http://127.0.0.1:<vllm_port>/v1/models` and uses `hosted_vllm/<served_model_name>` for Harbor automatically.
- With `port_profile_id`, con-driver sets `api_base`, `base_url`, and universal endpoint environment variables for the Harbor launch automatically.
- With `port_profile_id`, do not forward `--model` or agent kwargs for `api_base`, `base_url`, or `model_info`; those are managed by con-driver.
- In gateway mode, con-driver appends a unique per-agent `api_key` token automatically; do not hardcode `api_key` in forwarded args.
- In gateway mode, wrapper also sets `OPENAI_API_KEY`, `LITELLM_API_KEY`, and `HOSTED_VLLM_API_KEY` to the same per-agent token so Terminus/LiteLLM requests match `/agent/start`.
- Sampling is uniform over the merged task list.
- With `--sample-without-replacement`, each sampled task path appears at most once; this requires `n_task <= pool_size`.
- For Poisson arrivals, first launch happens immediately, then inter-arrival delays are exponential.
