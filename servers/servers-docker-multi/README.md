# vLLM + Jaeger + Gateway Multi (Docker Package)

This package is the multi-backend variant of `servers/servers-docker`.

It starts:

- one independent `jaeger` container per selected port profile
- one independent `vllm` container per selected port profile
- one host-side `gateway_multi` process that fronts all of those backends

The implementation intentionally reuses the same model config, port profile
config, Docker images, and compose conventions from `servers/servers-docker`.
Each backend still uses the existing single-stack Docker compose file; the
multi client simply materializes one compose project per backend profile.

## Key Behavior

`servers-docker-multi` treats a multi-GPU launch profile as a list of GPUs to
split across the selected backend profiles.

Example:

```bash
python3 servers/servers-docker-multi/client.py start \
  -m qwen3_coder_30b \
  -p 0,1 \
  -l h100_nvl_gpu23 \
  -b
```

This means:

- backend 1 uses port profile `0` on GPU `2`
- backend 2 uses port profile `1` on GPU `3`
- each backend runs an independent `vllm` + `jaeger`
- one host `gateway_multi` is started over port profiles `0` and `1`

For now, the number of selected port profiles must exactly match the number of
visible GPUs in the chosen launch profile.

## Commands

List profiles:

```bash
python3 servers/servers-docker-multi/client.py profiles models
python3 servers/servers-docker-multi/client.py profiles ports
python3 servers/servers-docker-multi/client.py profiles launches
```

Start:

```bash
python3 servers/servers-docker-multi/client.py start -m qwen3_coder_30b -p 0,1 -l h100_nvl_gpu23 -b
python3 servers/servers-docker-multi/client.py start -m qwen3_coder_30b -p 0 -p 1 -l h100_nvl_gpu23 -b
python3 servers/servers-docker-multi/client.py start -m qwen3_coder_30b -p 0,1 -l h100_nvl_gpu23 --lmcache 100 -b
python3 servers/servers-docker-multi/client.py start -m qwen3_coder_30b -p 2 -l h100_nvl_gpu2_single -b
```

Status and health:

```bash
python3 servers/servers-docker-multi/client.py status
python3 servers/servers-docker-multi/client.py up
python3 servers/servers-docker-multi/client.py wait-up --timeout-seconds 900
```

Logs:

```bash
python3 servers/servers-docker-multi/client.py logs -n 200
```

Stop:

```bash
python3 servers/servers-docker-multi/client.py stop -b
python3 servers/servers-docker-multi/client.py stop -b -m qwen3_coder_30b -p 0,1 -l h100_nvl_gpu23
python3 servers/servers-docker-multi/client.py daemon-stop
```

## Endpoints

The first selected port profile is the public control/profile for `gateway_multi`.

If you started with `-p 0,1`, then:

- raw gateway health: `http://localhost:11457/healthz`
- parsed gateway health: `http://localhost:18171/healthz`

The backend vLLM services still exist separately:

- profile `0` vLLM: `http://localhost:11451/v1/models`
- profile `1` vLLM: `http://localhost:24123/v1/models`

Each backend profile also keeps its own IPC socket through `gateway_multi`, for
example:

- `/tmp/vllm-gateway-profile-0.sock`
- `/tmp/vllm-gateway-profile-1.sock`

## Notes

- `gateway_multi` assignment policy is currently `round_robin`.
- This package reuses:
  - `configs/model_config.toml`
  - `configs/port_profiles.toml`
  - `servers/servers-docker/launch_profiles.toml`
  - `servers/servers-docker/service_images.toml`
  - `servers/servers-docker/docker-compose.yml`
- startup logs are written under `servers/servers-docker-multi/logs/`

## Tests

```bash
.venv/bin/python -m pytest servers/servers-docker-multi/tests -q
```
