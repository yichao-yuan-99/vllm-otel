# METHOD: Concurrent Harbor Trial Driver

## Goal

Provide a small, deterministic scheduler that launches many `harbor trials start` commands with configurable concurrency and arrival timing.

## Structure

- `driver.py`: lightweight repo-local script entrypoint.
- `src/con_driver/cli.py`: CLI parsing, config merge, and runtime assembly.
- `src/con_driver/scheduler.py`: task sampling, scheduling loop, subprocess lifecycle, logs/manifests.
- `src/con_driver/gateway_wrapper.py`: per-launch wrapper that performs gateway `/agent/start` and `/agent/end`.
- `src/con_driver/backends/harbor/backend.py`: Harbor-specific dataset prep and launch command construction.
- `src/con_driver/patterns.py`: arrival policies (`eager`, `poisson`).
- `src/con_driver/vllm_metrics_monitor.py`: optional Prometheus polling sidecar for vLLM metrics.

## Runtime Flow

1. Parse CLI args and optional TOML config.
2. If `port_profile_id` is set, resolve ports from `configs/port_profiles.toml`, probe the live served model from `vLLM /v1/models`, and synthesize Harbor model/endpoint args.
3. Resolve Harbor command prefix and forwarded Harbor args.
4. Prepare the task pool by downloading datasets with `harbor datasets download`.
5. Create a run output directory and (if gateway mode is enabled) call `POST /job/start` with a per-run output location on the active gateway endpoint(s).
6. Build a launch plan from sampled tasks.
7. For each launch in gateway mode:
   - generate a unique API token,
   - append it as `--agent-kwarg api_key=<token>`,
   - wrap the Harbor command with `con_driver.gateway_wrapper` to call `POST /agent/start` and `POST /agent/end`.
8. Launch up to `max_concurrent` subprocesses using selected arrival pattern.
9. Persist events and run manifest under `results_dir`.
10. Compute aggregate success/failure and optional reward average from trial `result.json` files.
11. If gateway mode is enabled, call `POST /job/end` with final run status on the same gateway endpoint(s).

## Constraints

- Backend is Harbor-only.
- The scheduler owns task path/trial ID/trials dir.
- Forwarded args cannot include `-p/--path`, `--trial-name`, or `--trials-dir`.
- Gateway mode is enabled by default and, when `port_profile_id` is set, routes agent requests through `gateway_parse_port`.
