# One-Time Docker Compose

This directory contains a fixed Docker Compose environment equivalent to:

```bash
python3 servers/servers-docker/client.py start \
  -m qwen3_coder_30b \
  -p 4 \
  -l h100_nvl_gpu23 \
  -b
```

It launches:

- `jaegertracing/all-in-one:1.57`
- `yichaoyuan/vllm-openai-otel-lp:v0.14.0-otel-lp`

with these fixed selections:

- model: `qwen3_coder_30b`
- served model name: `Qwen3-Coder-30B-A3B-Instruct`
- port profile: `4`
- launch profile: `h100_nvl_gpu23`
- visible GPUs: `2,3`
- tensor parallel size: `2`

## Files

- [docker-compose.qwen3_coder_30b-p4-h100_nvl_gpu23.yml](/scratch/yichaoy2/work/vllm-otel/servers/one-time/docker-compose.qwen3_coder_30b-p4-h100_nvl_gpu23.yml)

## Before Start

Export these variables in the current shell:

```bash
export HF_HOME=/path/to/hf-home
export HF_HUB_CACHE=/path/to/hf-hub-cache
export HF_TOKEN=your_hf_token
```

`HF_TOKEN` can be unset for public access if the model does not require it.

Do not run this one-time compose at the same time as `servers/servers-docker/client.py`
or another local Jaeger instance. It binds the same fixed Jaeger host ports
that the managed Docker package uses, plus profile-4 vLLM on `52341`.

GPU selection is enforced with Docker GPU `device_ids`, not only by
`NVIDIA_VISIBLE_DEVICES`.

## Start

```bash
docker compose \
  -f servers/one-time/docker-compose.qwen3_coder_30b-p4-h100_nvl_gpu23.yml \
  up -d --pull always
```

Wait until vLLM is ready:

```bash
until curl -fsS http://127.0.0.1:52341/v1/models >/dev/null; do
  sleep 2
done
```

## Verify

vLLM:

```bash
curl -s http://127.0.0.1:52341/v1/models
```

Jaeger UI:

```text
http://127.0.0.1:56198
```

## Logs

Compose logs:

```bash
docker compose \
  -f servers/one-time/docker-compose.qwen3_coder_30b-p4-h100_nvl_gpu23.yml \
  logs -n 200 vllm jaeger
```

Direct vLLM container logs:

```bash
docker logs --tail 200 vllm-openai-otel-lp-one-time-p4
```

## Stop

```bash
docker compose \
  -f servers/one-time/docker-compose.qwen3_coder_30b-p4-h100_nvl_gpu23.yml \
  down --remove-orphans
```

## Ports

This compose uses port profile `4`:

- vLLM: `52341`
- Jaeger UI: `56198`
- Jaeger OTLP gRPC: `53477`
