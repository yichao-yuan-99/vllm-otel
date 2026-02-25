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
2. Resolve Harbor command prefix and forwarded Harbor args.
3. Prepare the task pool by downloading datasets with `harbor datasets download`.
4. Create a run output directory and (if gateway mode is enabled) call `POST /job/start` with a per-run output location.
5. Build a launch plan from sampled tasks.
6. For each launch in gateway mode:
   - generate a unique API token,
   - append it as `--agent-kwarg api_key=<token>`,
   - wrap the Harbor command with `con_driver.gateway_wrapper` to call `POST /agent/start` and `POST /agent/end`.
7. Launch up to `max_concurrent` subprocesses using selected arrival pattern.
8. Persist events and run manifest under `results_dir`.
9. Compute aggregate success/failure and optional reward average from trial `result.json` files.
10. If gateway mode is enabled, call `POST /job/end` with final run status.

## Constraints

- Backend is Harbor-only.
- The scheduler owns task path/trial ID/trials dir.
- Forwarded args cannot include `-p/--path`, `--trial-name`, or `--trials-dir`.
- Gateway mode is enabled by default and assumes agent requests are routed through gateway `.../v1`.
