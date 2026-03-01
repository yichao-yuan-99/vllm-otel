# Docker Image Builder

`servers/docker` now provides a repo-local script that generates the Dockerfile from a selected vLLM base image, then builds and pushes the final image.
Run it from this repo checkout, because the build uses `servers/docker/forceSeq` and `servers/docker/vllm_entrypoint.sh` from the working tree.

Prerequisite: `docker login` must already have been run in the current shell environment before using this CLI.

## Commands

Render the generated Dockerfile to stdout:

```bash
python3 servers/docker/build_image.py render --base-tag v0.14.0
```

Render the generated Dockerfile to a file:

```bash
python3 servers/docker/build_image.py render --base-tag v0.14.0 --output /tmp/vllm-otel.Dockerfile
```

Build and push the standard CUDA image:

```bash
python3 servers/docker/build_image.py build-push --base-tag v0.14.0
```

Build and push the ROCm image:

```bash
python3 servers/docker/build_image.py build-push --base-tag v0.14.0 --rocm
```

Keep the generated Dockerfile used for build:

```bash
python3 servers/docker/build_image.py build-push \
  --base-tag v0.14.0 \
  --dockerfile-output servers/docker/Dockerfile.generated
```

Override the pushed image reference explicitly:

```bash
python3 servers/docker/build_image.py build-push \
  --base-tag v0.14.0 \
  --image yichaoyuan/vllm-openai-otel-lp:v0.14.0-otel-lp-2
```

## Base Image Selection

- Standard CUDA: `vllm/vllm-openai:<base-tag>`
- ROCm with `--rocm`: `vllm/vllm-openai-rocm:<base-tag>`

Examples:

- `--base-tag v0.14.0` -> `vllm/vllm-openai:v0.14.0`
- `--base-tag nightly` -> `vllm/vllm-openai:nightly`
- `--base-tag v0.14.0 --rocm` -> `vllm/vllm-openai-rocm:v0.14.0`

## Default Output Image Naming

If you do not pass `--image`, the CLI derives the pushed image name automatically.

- Standard CUDA repo: `yichaoyuan/vllm-openai-otel-lp`
- ROCm repo: `yichaoyuan/vllm-openai-otel`
- Standard tag: `<base-tag>-otel-lp`
- ROCm tag: `<base-tag>-otel-lp-rocm`

Examples:

- `--base-tag v0.14.0` -> `yichaoyuan/vllm-openai-otel-lp:v0.14.0-otel-lp`
- `--base-tag nightly` -> `yichaoyuan/vllm-openai-otel-lp:nightly-otel-lp`
- `--base-tag v0.16.0 --rocm` -> `yichaoyuan/vllm-openai-otel:v0.16.0-otel-lp-rocm`

You can override the defaults with either `--target-repo` and `--target-tag`, or with `--image`.

## Generated Dockerfile Contents

The generated Dockerfile always:

- starts from the selected vLLM base image
- installs the OpenTelemetry Python packages used by this repo
- copies `servers/docker/forceSeq`
- copies `servers/docker/vllm_entrypoint.sh`
- sets `PYTHONPATH=/opt/vllm-plugins`

## Runtime Packages

The Docker runtime package still selects the serving image from `servers/servers-docker/service_images.toml`.
The AMDHPC package still selects its image from its own config and environment.
This builder script only handles image generation, local build, and push.
