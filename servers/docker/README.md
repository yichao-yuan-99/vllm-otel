# Docker Image Builder

`servers/docker` now provides a repo-local script that generates the Dockerfile from a selected vLLM base image, then builds and pushes the final image.
Run it from this repo checkout, because the build uses `servers/docker/forceSeq` and `servers/docker/vllm_entrypoint.sh` from the working tree.

Prerequisite: `docker login` must already have been run in the current shell environment before using this CLI.

## Commands

Render the generated Dockerfile to stdout:

```bash
python3 servers/docker/build_image.py render --base-image vllm/vllm-openai:v0.14.0
```

Render the generated Dockerfile to a file:

```bash
python3 servers/docker/build_image.py render --base-image vllm/vllm-openai:v0.14.0 --output /tmp/vllm-otel.Dockerfile
```

Build and push using a full CUDA base image reference:

```bash
python3 servers/docker/build_image.py build-push --base-image vllm/vllm-openai:v0.14.0
```

Build and push using a full ROCm base image reference:

```bash
python3 servers/docker/build_image.py build-push --base-image rocm/vllm-dev:upstream_preview_releases_v0.17.0_20260303
```

Build and push from the official ROCm vLLM image tag:

```bash
python3 servers/docker/build_image.py build-push --base-image vllm/vllm-openai-rocm:v0.17.1
```

Build and push with ROCm LMCache bootstrap enabled (example `gfx942`):

```bash
python3 servers/docker/build_image.py build-push \
  --base-image vllm/vllm-openai-rocm:v0.17.1 \
  --gfx gfx942
```

Keep the generated Dockerfile used for build:

```bash
python3 servers/docker/build_image.py build-push \
  --base-image vllm/vllm-openai:v0.14.0 \
  --dockerfile-output servers/docker/Dockerfile.generated
```

Override the pushed image reference explicitly:

```bash
python3 servers/docker/build_image.py build-push \
  --base-image rocm/vllm-dev:upstream_preview_releases_v0.17.0_20260303 \
  --image yichaoyuan/vllm-openai-otel-lp:v0.14.0-otel-lp-2
```

## Base Image Selection

Pass the full image reference directly with `--base-image`.

Examples:

- `--base-image vllm/vllm-openai:v0.14.0`
- `--base-image vllm/vllm-openai:nightly`
- `--base-image vllm/vllm-openai-rocm:v0.17.1`
- `--base-image rocm/vllm-dev:upstream_preview_releases_v0.17.0_20260303`

## Default Output Image Naming

If you do not pass `--image`, the CLI derives the pushed image name automatically.

- Default repo: `<target-namespace>/<base-image-repo-slug>` (repo path with `/` replaced by `-`)
- Standard tag: `<base-image-tag>-otel-lp`
- ROCm tag: `<base-image-tag>-otel-lp-rocm`

ROCm defaults are auto-detected when `--base-image` contains `rocm`.
You can force ROCm default naming with `--rocm`.
Default namespace is `yichaoyuan` (override with `--target-namespace`, or fully override with `--target-repo`).

Examples:

- `--base-image vllm/vllm-openai:v0.14.0` -> `yichaoyuan/vllm-vllm-openai:v0.14.0-otel-lp`
- `--base-image vllm/vllm-openai:nightly` -> `yichaoyuan/vllm-vllm-openai:nightly-otel-lp`
- `--base-image vllm/vllm-openai-rocm:v0.17.1` -> `yichaoyuan/vllm-vllm-openai-rocm:v0.17.1-otel-lp-rocm`
- `--base-image rocm/vllm-dev:upstream_preview_releases_v0.17.0_20260303` -> `yichaoyuan/rocm-vllm-dev:upstream_preview_releases_v0.17.0_20260303-otel-lp-rocm`

You can override the defaults with either `--target-repo` and `--target-tag`, or with `--image`.

## ROCm LMCache Bootstrap (`--gfx`)

When `--gfx <arch>` is provided, the image sets `VLLM_LMCACHE_GFX=<arch>`.
At container startup, `servers/docker/vllm_entrypoint.sh` performs an on-the-fly LMCache install before launching vLLM:

1. clone `https://github.com/LMCache/LMCache.git`
2. checkout tag `v0.4.1`
3. install with:

```bash
PYTORCH_ROCM_ARCH="<arch>" \
TORCH_DONT_CHECK_COMPILER_ABI=1 \
CXX=hipcc \
BUILD_WITH_HIP=1 \
python3 -m pip install --no-build-isolation -e .
```

Notes:

- this is intentionally runtime bootstrap in the entrypoint, so build hosts do not need ROCm
- startup needs network access to clone LMCache unless you override source env vars
- optional runtime overrides:
  - `VLLM_LMCACHE_REPO_URL` (default `https://github.com/LMCache/LMCache.git`)
  - `VLLM_LMCACHE_REPO_TAG` (default `v0.4.1`)
  - `VLLM_LMCACHE_SRC_DIR` (default `/tmp/lmcache-src`)
  - `VLLM_LMCACHE_STAMP_DIR` (default `/tmp`)

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
