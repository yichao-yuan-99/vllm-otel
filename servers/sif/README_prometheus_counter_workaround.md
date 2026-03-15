# Prometheus Counter Negative Increment Workaround

## Problem

In some vLLM runs, the API server can fail inside metrics logging when
`prometheus_client.metrics.Counter.inc` receives a negative amount.

Observed stack trace (from replay logs):

- `vllm/v1/metrics/loggers.py` calls `Counter.inc(...)`
- `prometheus_client/metrics.py` raises:
  - `ValueError: Counters can only be incremented by non-negative amounts.`

This hard exception can break async output handling in vLLM.

## Root Cause

The installed Prometheus client implementation contains:

```python
if amount < 0:
    raise ValueError('Counters can only be incremented by non-negative amounts.')
```

When `amount` is negative, it throws instead of continuing.

## Fix Applied in SIF Build

`servers/sif/lmcache-rocm.def.in` now applies a build-time patch during `%post`
that rewrites this guard in `prometheus_client/metrics.py`:

- log to `stderr` with:
  - `Counters can only be incremented by non-negative amounts. value <amount>`
- set `amount = 0`
- do **not** raise an exception

Behavior becomes effectively:

```python
if amount < 0:
    print(
        "Counters can only be incremented by non-negative amounts. value "
        + str(amount),
        file=sys.stderr,
    )
    amount = 0
```

## Why This Is Done in the SIF Build

The workaround must exist inside the runtime container filesystem
(`/usr/local/lib/python3.12/dist-packages/prometheus_client/metrics.py`).
Applying it in `%post` guarantees the built SIF always contains the fix.

## Verification

After building a SIF, verify the patch is present:

```bash
apptainer exec <your-image>.sif \
  python3 - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("prometheus_client.metrics")
path = Path(spec.origin)
text = path.read_text(encoding="utf-8")
print(path)
print(
    "patched:",
    "Counters can only be incremented by non-negative amounts. value" in text
    and "amount = 0" in text
    and "raise ValueError('Counters can only be incremented by non-negative amounts.')" not in text,
)
PY
```

Expected output includes `patched: True`.
