# Post-Process Multi

`post-process-multi` is the `gateway_multi` variant of `post-process`.

It mirrors the existing pipeline and keeps writing the same
`<run-dir>/post-processed/...` outputs so downstream tools stay compatible.
Only the stages that need `gateway_multi`-specific handling live here:

- `gateway/ctx-aware-log/extract_run.py`
- `visualization/gateway-ctx-aware/generate_all_figures.py`

All other stages fall back to the existing implementations under
`post-process/`, which already handle the current multi-GPU power summaries and
multi-profile `vllm-log/profile-*` layout.

## Run The Full Pipeline

```bash
python3 post-process-multi/run_all.py \
  --run-dir results/replay/sweep-qps-docker-power-clean-gateway_multi/<...>/<timestamp>
```

Example:

```bash
python3 post-process-multi/run_all.py \
  --run-dir results/replay/sweep-qps-docker-power-clean-gateway_multi/swebench-verified/mini-swe-agent/split/exclude-unranked/qps0_25/20260406T001613Z
```

## Key Difference

For `gateway_multi`, `gateway-output/job/` may contain multiple
`ctx_aware_*_profile-<id>.jsonl` files. The multi pipeline preserves every
profile-specific ctx-aware log, emits an aggregate top-level timeseries for
compatibility, and generates one figure per profile in addition to the
aggregate figure.
