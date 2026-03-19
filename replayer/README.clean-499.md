# Replay Compile Clean Mode (`--clean`)

This document describes `replayer compile --clean` behavior.

## Purpose

`--clean` removes source requests with `status_code=499` from compiled replay
plans and collapses request timing so those requests are treated as if they
never existed.

## Scope

- `--clean` is supported only with `--split-two-group-plans`.
- It is intended for split outputs (`top`, `rest`, and `exclude-unranked`)
  where unranked trails have already been handled by split inputs.

## Cleaning Rule

For each source request record in
`gateway-output/.../requests/model_inference.jsonl`:

- if `status_code == 499`: remove it from replay
- otherwise: keep it

Removed requests are not replayed at all.

## Timing Collapse Semantics

When a `499` request is removed, its request-time window is deleted:

- the next request starts at the removed request's original start timestamp
- if multiple consecutive requests are `499`, this collapse repeats until a
  non-`499` request is reached

Equivalent effect: consecutive `499` requests are treated as never happened.

Example:

- request `12` ends at `t=15`
- request `13` starts at `t=16` (`499`)
- request `14` starts at `t=20` (`499`)
- request `15` starts at `t=26`

With `--clean`, request `15` is scheduled to start at `t=16` (request `13`'s
start). Requests `13` and `14` are absent from the replay plan.

## Output File Naming

With split compile, `.clean` is inserted immediately after `replay-plan` and
before metric/group identifiers:

- if `--split-two-group-metric` is omitted, compile emits both `token` and
  `context` clean plan sets
- if `--split-two-group-metric` is set, compile emits only that metric

- no additional suffix (token example):
  - `replay-plan.clean.token.top.json`
  - `replay-plan.clean.token.rest.json`
  - `replay-plan.clean.token.exclude-unranked.json`
  - `replay-plan.clean.token.removal-stats.json`
  - `replay-plan.clean.token.removal-details.json`
- with `--additional-suffix qwen3_fp8`:
  - `replay-plan.clean.token.top.qwen3_fp8.json`
  - `replay-plan.clean.token.rest.qwen3_fp8.json`
  - `replay-plan.clean.token.exclude-unranked.qwen3_fp8.json`
  - `replay-plan.clean.token.removal-stats.qwen3_fp8.json`
  - `replay-plan.clean.token.removal-details.qwen3_fp8.json`

## Removal Report Files

When `--clean` is enabled, compile writes two extra JSON files:

- `...removal-stats.json`:
  - total removed request count
  - removed count by trail
  - removed count by worker
- `...removal-details.json`:
  - one row per removed request (trail, worker, request id/index, timestamps)

## Plan Metadata

Compiled plans include:

- `compile_options.clean: true`
- `compile_options.clean_removed_499_request_count`
