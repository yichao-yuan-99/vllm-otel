# ROCm Image Build + Push

This guide builds and pushes the ROCm image from:

- `docker/Dockerfile.rocm`

Target image tag:

- `yichaoyuan/vllm-openai-otel:v0.16.0-otel-lp-rocm`

## 1) Login

```bash
docker login
```

## 2) Build

Run from repo root:

```bash
docker build \
  -f docker/Dockerfile.rocm \
  -t yichaoyuan/vllm-openai-otel:v0.16.0-otel-lp-rocm \
  .
```

## 3) Push

```bash
docker push yichaoyuan/vllm-openai-otel:v0.16.0-otel-lp-rocm
```

## 4) Optional Verify

```bash
docker image inspect yichaoyuan/vllm-openai-otel:v0.16.0-otel-lp-rocm --format '{{.Id}}'
docker pull yichaoyuan/vllm-openai-otel:v0.16.0-otel-lp-rocm
```

## 5) Use in Compose

Set in `docker/.env`:

```bash
VLLM_IMAGE_NAME=yichaoyuan/vllm-openai-otel:v0.16.0-otel-lp-rocm
```

Then start without forcing rebuild:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env up -d
```
