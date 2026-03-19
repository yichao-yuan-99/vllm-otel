# Replay Split Two-Group Plans

This document describes `replayer compile --split-two-group-plans` behavior,
including metric input files, output plan filenames, and compatibility notes.

## Purpose

`--split-two-group-plans` compiles three replay plans from one profiled job:

- a `top` group plan
- a `rest` group plan
- an `exclude-unranked` plan (the union of `top` + `rest`)

Grouping is read from precomputed analysis outputs under:

- `<job-dir>/original-analysis/split/`

## Metric Input Files

Choose grouping metric with:

- omit `--split-two-group-metric` to generate both `token_usage` and `context_usage`
- `--split-two-group-metric token_usage` to generate only token split plans
- `--split-two-group-metric context_usage` to generate only context split plans

Metric to source file mapping:

- `token_usage` -> `top-p-token-usage-two-groups.json`
- `context_usage` -> `top-p-context-usage-two-groups.json`

## Output Plan Filenames

Let `--plan-out` be `<dir>/replay-plan.json` (or default path).

Compiled split outputs are metric-qualified:

- `token_usage`:
  - `<dir>/replay-plan.token.top.json`
  - `<dir>/replay-plan.token.rest.json`
  - `<dir>/replay-plan.token.exclude-unranked.json`
- `context_usage`:
  - `<dir>/replay-plan.context.top.json`
  - `<dir>/replay-plan.context.rest.json`
  - `<dir>/replay-plan.context.exclude-unranked.json`

This avoids filename collisions when compiling both metrics for the same run.

With `--clean`, `.clean` is inserted immediately after `replay-plan`:

- `token_usage`:
  - `<dir>/replay-plan.clean.token.top.json`
  - `<dir>/replay-plan.clean.token.rest.json`
  - `<dir>/replay-plan.clean.token.exclude-unranked.json`
  - `<dir>/replay-plan.clean.token.removal-stats.json`
  - `<dir>/replay-plan.clean.token.removal-details.json`
- `context_usage`:
  - `<dir>/replay-plan.clean.context.top.json`
  - `<dir>/replay-plan.clean.context.rest.json`
  - `<dir>/replay-plan.clean.context.exclude-unranked.json`
  - `<dir>/replay-plan.clean.context.removal-stats.json`
  - `<dir>/replay-plan.clean.context.removal-details.json`

## Examples

Compile split plans for both token and context metrics (default):

```bash
python -m replayer compile \
  --job-dir <job-dir> \
  --port-profile-id 1 \
  --split-two-group-plans
```

Compile token-based split plans only:

```bash
python -m replayer compile \
  --job-dir <job-dir> \
  --port-profile-id 1 \
  --split-two-group-plans \
  --split-two-group-metric token_usage
```

Compile all discovered jobs under a root with both metrics:

```bash
python -m replayer compile \
  --job-root <jobs-root-dir> \
  --port-profile-id 1 \
  --split-two-group-plans
```

Compile all discovered jobs under a root with token split plans only:

```bash
python -m replayer compile \
  --job-root <jobs-root-dir> \
  --port-profile-id 1 \
  --split-two-group-plans \
  --split-two-group-metric token_usage
```

`--job-root` first discovers all valid job directories, then runs per-job
compile as parallel child processes and shows live batch progress.

Compile context-based split plans:

```bash
python -m replayer compile \
  --job-dir <job-dir> \
  --port-profile-id 1 \
  --split-two-group-plans \
  --split-two-group-metric context_usage
```

Compile token-based split plans with `499` cleaning:

```bash
python -m replayer compile \
  --job-dir <job-dir> \
  --port-profile-id 1 \
  --split-two-group-plans \
  --split-two-group-metric token_usage \
  --clean
```

## Reuse Behavior

When split mode is enabled, compile checks metric-qualified plan files for each
requested metric and may reuse existing plans when:

- all `top`, `rest`, and `exclude-unranked` files exist
- all are at current `compile_version`
- each file has matching `split_two_group.metric` and `split_two_group.group`

## Plan Metadata

Split plans include a `split_two_group` object with:

- `enabled`
- `metric`
- `source_path`
- `source_selected_p`
- `group` (`top`, `rest`, or `exclude-unranked`)

## Compatibility Note

Older workflows may still reference legacy names:

- `replay-plan.top.json`
- `replay-plan.rest.json`
- `replay-plan.exclude-unranked.json`

`replayer compile` now writes metric-qualified names listed above.

## Unranked Trail Handling

When split payloads are generated from `original-analysis/split/`, grouped
trail names come from ranked trails only.

During `replayer compile --split-two-group-plans`:

- if a compiled trail is missing from `group_top`/`group_rest` but is listed in
  `<job-dir>/original-analysis/split/top-p-usage-ratio-summary.json` under
  `unranked_trails`, compile no longer fails
- those unmatched-unranked trails are ignored for top/rest split plans and
  therefore excluded from the split `exclude-unranked` union plan as well
- those unmatched-unranked trails are reported as note fields in compile summary JSON
- compile still fails for unmatched trails that are not in `unranked_trails`

`--exclude-unranked-trails` is a non-split compile option and cannot be
combined with `--split-two-group-plans`.

`--clean` is a split-only option. For detailed `499` clean semantics and timing
collapse examples, see `replayer/README.clean-499.md`.
