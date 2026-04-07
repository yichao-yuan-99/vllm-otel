# vLLM + Jaeger + Gateway (Docker Package)

This package runs:

- `jaeger` (trace backend + UI) in Docker
- `vllm` (OpenAI-compatible server with OTEL packages) in Docker
- `gateway` on host as a managed local daemon by default
- `gateway_ctx` on host instead when `start --gateway-ctx` is used

Environment launch and operations are managed only through `servers/servers-docker/client.py`.
Runtime compose uses pushed Docker Hub images only (no local Dockerfile build step).
The compose `--env-file` is generated and managed internally at runtime; there is no user-editable docker `.env` workflow.

Image build/push instructions:

- `servers/docker/README.md`

## 1) Export Hugging Face env vars in your current shell

```bash
export HF_HOME=/path/to/hf-home
export HF_HUB_CACHE=/path/to/hf-hub-cache
export HF_TOKEN=your_hf_token
```

`HF_TOKEN` can be unset for public models.

Gateway config is resolved in this order:

- `gateway/config.toml`
- `gateway/config.example.toml`

With `--gateway-ctx`, the launcher resolves the ctx-aware package config instead:

- `gateway_ctx/config.toml`
- `gateway_ctx/config.example.toml`

## 2) Daemon CLI Package

If your system `python3` does not have `typer`, use `./.venv/bin/python` instead.

List available profiles:

```bash
python3 servers/servers-docker/client.py profiles models
python3 servers/servers-docker/client.py profiles ports
python3 servers/servers-docker/client.py profiles launches
```

Start one environment:

```bash
python3 servers/servers-docker/client.py start -m qwen3_coder_30b -p 0 -l h100_nvl_gpu23 -b
python3 servers/servers-docker/client.py start -m qwen3_coder_30b -p 0 -l h100_nvl_gpu23 --gateway-ctx -b
python3 servers/servers-docker/client.py start -m qwen3_coder_30b -p 0 -l h100_nvl_gpu23 --lmcache 100 -b
python3 servers/servers-docker/client.py start -m qwen3_coder_30b_fp8 -p 2 -l h100_nvl_gpu2_single --gpu-memory-utilization 0.75 -b
```

Runtime names are suffixed with `<model>-<launch-profile>` (from `-m` and `-l`),
so different model/launch selections can run in parallel as separate Docker Compose projects when ports do not collide.
Gateway is started automatically for the selected `-p` port profile and stopped by `stop`/`daemon-stop`.
Use `--gateway-ctx` on `start` to launch `gateway_ctx` instead of the plain gateway package for that environment.

`start` materializes compose variables internally from:

- `configs/model_config.toml`
- `configs/port_profiles.toml`
- `servers/servers-docker/launch_profiles.toml`
- `servers/servers-docker/service_images.toml`

GPU selection from the launch profile is enforced through Docker Compose GPU
`device_ids`, not only by `NVIDIA_VISIBLE_DEVICES`.
Optional `--lmcache <size>` mirrors the AMD HPC client behavior: it sets
`LMCACHE_MAX_LOCAL_CPU_SIZE=<size>` and appends
`--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'`
to the vLLM startup args.
Optional `--gpu-memory-utilization <value>` (`0 < value <= 1`) appends
`--gpu-memory-utilization <value>` to the vLLM startup args and overrides any
model-config value for that same flag.

Change the Docker tags in `servers/servers-docker/service_images.toml` if you need a different Jaeger or vLLM image.

Operate environment:

```bash
python3 servers/servers-docker/client.py status
python3 servers/servers-docker/client.py up
python3 servers/servers-docker/client.py wait-up --timeout-seconds 900
python3 servers/servers-docker/client.py logs -n 200
```

Direct Docker bypass for the vLLM container:

```bash
docker logs --tail 200 vllm-openai-otel-lp-<model>-<launch-profile>
# example:
# docker logs --tail 200 vllm-openai-otel-lp-qwen3-coder-30b-h100-nvl-gpu23
```

Use the package command above by default. The direct `docker logs` form is only
for quick inspection when you explicitly want to bypass `client.py`.
`client.py logs` now includes both Docker Compose logs and a tail of the gateway daemon log for the active port profile.

Blocking startup also saves a repo-local startup log under `servers/servers-docker/logs/`.
That startup log includes the stepwise client progress plus `docker compose up/down` output.
A sibling compose log file in the same directory captures the full `docker compose logs --no-color --timestamps jaeger vllm` output collected during startup.

Stop environment:

```bash
python3 servers/servers-docker/client.py stop -b
python3 servers/servers-docker/client.py stop -b -m qwen3_coder_30b -p 0 -l h100_nvl_gpu23
python3 servers/servers-docker/client.py daemon-stop
```

Use `stop -b -m/-p/-l` to stop one specific environment selection. All three
selection flags must be provided together for targeted stop.

Run smoke tests from the separate test package:

```bash
python3 servers/test/client.py --port-profile 0
```

## 3) Verify Endpoints

```bash
curl http://localhost:11451/v1/models
curl http://localhost:11457/healthz
curl http://localhost:18171/healthz
```

Use ports from your selected port profile if changed.
When `--lmcache` is enabled, LMCache metrics are exposed on the selected
profile's `lmcache_port` from `configs/port_profiles.toml`.

If you change the launch profile GPU selection, restart the environment so the
generated Docker Compose GPU override is re-materialized and applied.

Jaeger UI: `http://localhost:16686`

Force-sequence request example (directly to vLLM):

```bash
curl -s "http://localhost:11451/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{
    "model":"Qwen3-Coder-30B-A3B-Instruct",
    "messages":[{"role":"user","content":"ignored"}],
    "max_tokens":16,
    "temperature":0,
    "vllm_xargs":{
      "forced_token_ids":[42,53,99],
      "force_eos_after_sequence":true
    }
  }'
```

`VLLM_COLLECT_DETAILED_TRACES` defaults to `all` so vLLM emits spans into Jaeger.
Prompt token details are enabled in the vLLM container (`--enable-prompt-tokens-details`).
Custom logits processor loading is enabled via `VLLM_LOGITS_PROCESSORS` (default: `forceSeq.force_sequence_logits_processor:ForceSequenceAdapter`).
Model `extra_args` from `configs/model_config.toml` are forwarded into the Docker vLLM launch command automatically.
Docker startup always injects `LMCACHE_INTERNAL_API_SERVER_ENABLED=1`,
`LMCACHE_INTERNAL_API_SERVER_PORT_START=<profile.lmcache_port>`, and
`PYTHONHASHSEED=0` into the vLLM container env.
The force-sequence processor only activates for requests that include `vllm_xargs.forced_token_ids`.
At vLLM startup, `servers/docker/vllm_entrypoint.sh` resolves `eos_token_id` from `VLLM_MODEL_NAME` and exports `VLLM_FORCE_SEQUENCE_EOS_TOKEN_ID` automatically for the processor.
If a model `extra_args` entry includes `--trust-remote-code`, Docker startup now sets `VLLM_FORCE_SEQ_TRUST_REMOTE_CODE=true` automatically so the tokenizer bootstrap uses remote code as well.
