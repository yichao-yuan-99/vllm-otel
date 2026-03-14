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
- `--env KEY=VALUE` (repeatable): inject extra environment variables into rendered vLLM process.
- `--check-port-availability`: validates selected login-node ports are currently free (off by default).

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

## Current limitation

- `--local-mode` is currently supported on `start` only.
- `start-group --local-mode ...` is intentionally rejected.
