This document describes strict split mode for `original-analysis/split`.

## Purpose

Strict mode excludes trails that contain any request with `status_code=499` from
ranked top/rest splitting.

Normal mode is still generated as before. Strict mode is generated in parallel,
using `.strict-499` output filenames.

## Strict Unranked Rule

In strict mode, a trail is unranked when either of the following is true:

- it has no valid request length (`prompt_tokens + completion_tokens`)
- it contains at least one request with `status_code=499`

## Per-Run Outputs

Normal outputs:

- `top-p-usage-ratio-summary.json`
- `top-p-token-usage-two-groups.json`
- `top-p-context-usage-two-groups.json`

Strict outputs:

- `top-p-usage-ratio-summary.strict-499.json`
- `top-p-token-usage-two-groups.strict-499.json`
- `top-p-context-usage-two-groups.strict-499.json`

## Root-Level CSV Outputs

When running with `--root-dir`, both normal and strict two-group tables are
written:

- `split-top-p-token-usage-two-group-table.csv`
- `split-top-p-context-usage-two-group-table.csv`
- `split-top-p-token-usage-two-group-table.strict-499.csv`
- `split-top-p-context-usage-two-group-table.strict-499.csv`

## Summary Metadata

Strict summary JSON includes:

- `unranked_mode: "strict_499"`
- `unranked_criteria`
- `trail_count_with_status_499`

Normal summary JSON includes:

- `unranked_mode: "normal"`
- `trail_count_with_status_499` (for reference)

