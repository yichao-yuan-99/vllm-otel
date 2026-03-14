# SIF Build With LMCache

`servers/sif/build_lmcache_sif.py` builds a SIF with LMCache preinstalled at image-build time.

This replaces runtime LMCache bootstrap in `servers/docker/vllm_entrypoint.sh`.

## What It Does

1. Uses either:
   - `--base-sif <path>` (existing SIF), or
   - `--base-image <image-ref>` (pulls with `apptainer pull`)
2. Clones `LMCache` and checks out a tag (`v0.4.1` by default).
3. Renders `servers/sif/lmcache-rocm.def.in`.
4. Runs `apptainer build` to produce a new SIF with LMCache baked in.
5. Removes runtime `install_lmcache_rocm_if_requested` call from `/opt/vllm-plugins/vllm_entrypoint.sh` if present in the base image.

## Example (From Base Docker Image)

```bash
python3 servers/sif/build_lmcache_sif.py \
  --base-image docker://yichaoyuan/vllm-vllm-openai-rocm:v0.17.1-otel-lp-rocm \
  --output-sif "$APPTAINER_IMGS/vllm-vllm-openai-rocm:v0.17.1-otel-lp-rocm-lmcache-gfx942.sif" \
  --gfx gfx942
```

## Example (From Existing SIF)

```bash
python3 servers/sif/build_lmcache_sif.py \
  --base-sif "$APPTAINER_IMGS/vllm-vllm-openai-rocm:v0.17.1-otel-lp-rocm.sif" \
  --output-sif "$APPTAINER_IMGS/vllm-vllm-openai-rocm:v0.17.1-otel-lp-rocm-lmcache-gfx942.sif" \
  --gfx gfx942
```

## Notes

- Default LMCache repo/tag:
  - `https://github.com/LMCache/LMCache.git`
  - `v0.4.1`
- Override via:
  - `--lmcache-repo-url`
  - `--lmcache-repo-tag`
- Uses `--fakeroot` by default for `apptainer build`; pass `--no-fakeroot` to disable.
- Use `--force` to overwrite an existing output SIF.
