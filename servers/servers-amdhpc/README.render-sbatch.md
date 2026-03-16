# render-sbatch.py

`servers/servers-amdhpc/render-sbatch.py` renders Slurm sbatch scripts without submitting jobs.

It mirrors the `client.py start` / `start-group` options for partition/model/profile selection, plus rendering-only options.

## Basic render

```bash
# Single-profile sbatch render
python3 servers/servers-amdhpc/render-sbatch.py start \
  -P 0 \
  -p mi3008x \
  -m qwen3_coder_30b

# Grouped sbatch render
python3 servers/servers-amdhpc/render-sbatch.py start-group \
  -g bench_a \
  -L 0,1,2,3 \
  -p mi3008x \
  -m qwen3_coder_30b
```

Optional flags:

- `--lmcache <size>`: enables LMCache options in rendered vLLM startup.
- `--no-async-scheduling`: appends `--no-async-scheduling` to rendered vLLM startup.
- `--env KEY=VALUE` (repeatable): inject extra environment variables into rendered vLLM process.
- `--check-port-availability`: validates selected login-node ports are currently free (off by default).
- Render mode resolves the effective vLLM SIF path but does not require the file to exist at render time.

## Local mode

Use `--local-mode <script>` on `start` to render a single-profile sbatch that runs the full workflow directly on the allocated node (no login-node reverse tunnels).

```bash
python3 servers/servers-amdhpc/render-sbatch.py start \
  -P 0 \
  -p mi3008x \
  -m qwen3_coder_30b \
  --local-mode ./scripts/run_local_replay.sh
```

`<script>` must be an existing file path. It is resolved to an absolute path at render time.

### Local-mode sbatch lifecycle

The rendered local-mode sbatch does this in order:

1. Launches Jaeger and vLLM with Apptainer.
2. Waits until both are ready (`vLLM /v1/models` and Jaeger UI endpoint).
3. Launches gateway for the selected `--port-profile`.
4. Waits for gateway listeners to be reachable.
5. Executes your `--local-mode` script.
6. Waits for that script to finish.
7. Gracefully tears down gateway, vLLM, and Jaeger.

The sbatch exits with the same exit code as your local-mode script.

### Environment exposed to your local-mode script

The rendered sbatch exports:

- `VLLM_BASE_URL` (for example `http://127.0.0.1:<vllm_port>`)
- `GATEWAY_BASE_URL` (for example `http://127.0.0.1:<gateway_port>`)
- `GATEWAY_PARSE_BASE_URL` (for example `http://127.0.0.1:<gateway_parse_port>`)
- `JAEGER_BASE_URL` (for example `http://127.0.0.1:<jaeger_ui_port>`)
- `PORT_PROFILE_ID`

### Gateway overrides for local mode

You can override runtime behavior at job runtime via environment variables:

- `GATEWAY_CONFIG` (explicit gateway config path)
- `GATEWAY_VENV_DIR` (default: `<repo>/.venv`)
- `GATEWAY_HOST` (default: `127.0.0.1`)
- `GATEWAY_SKIP_INSTALL` (default: `1`)

If `GATEWAY_CONFIG` is unset, the script uses `gateway/config.toml` when present, otherwise `gateway/config.example.toml`.

## Grouped local mode (entrypoint injection)

Grouped renders (`start-group`) also support `--local-mode <script>`.

This injects a grouped workload script that runs after grouped vLLM workers are ready.
Use it to inject the sbatch orchestrator entrypoint directly into the rendered grouped sbatch:

```bash
python3 servers/servers-amdhpc/render-sbatch.py start-group \
  -g bench_orch \
  -L 0,1,2,3,4,5,6,7 \
  -p mi3008x \
  -m qwen3_coder_30b \
  --local-mode ./sbatch-orchestrator/entrypoint.sh
```

At runtime, set the orchestrator job list before `sbatch`:

```bash
SBATCH_ORCHESTRATOR_JOB_LIST=/abs/path/to/jobs.txt \
sbatch /abs/path/to/rendered-group-sbatch.sh
```

See `sbatch-orchestrator/README.md` for job-list format and optional env vars.

## Current limitation

- Grouped local mode runs after grouped vLLM workers are up; it does not start gateway processes automatically.
