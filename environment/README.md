# Python Environment Export

This directory captures the current shared `./.venv` in a portable form.

- Export source repo: `main` at commit `9a073a3347942fa737ea0e84a98c6ef1289375e3`
- Exported from Python `3.12.3`
- Export date: `2026-04-14`

## Files

- `requirements.lock.txt`: pinned third-party packages from the current `.venv`
- `requirements.local.txt`: repo-local editable installs that should be applied from a checkout of this repository

The split is intentional:

- a raw `pip freeze` from this `.venv` included machine-specific editable URLs
- `zeus` was frozen as a local `file:///...` path, which is not portable to another machine
- this export normalizes that dependency to `zeus==0.15.0`, which is available from PyPI

## Recreate On Another Machine

From a checkout of this repo at commit `9a073a3347942fa737ea0e84a98c6ef1289375e3`:

```bash
uv venv .venv --python 3.12.3
uv pip sync environment/requirements.lock.txt --python .venv/bin/python
uv pip install --python .venv/bin/python -r environment/requirements.local.txt
```

If you want the same active shell flow afterward:

```bash
source .venv/bin/activate
```

## Regenerate These Files

Run this from the repo root:

```bash
uv pip freeze --python .venv/bin/python --exclude-editable --exclude zeus > environment/requirements.lock.txt
```

Then add this line back to `environment/requirements.lock.txt`:

```txt
zeus==0.15.0
```

Keep `environment/requirements.local.txt` aligned with the editable packages currently installed in `.venv`.

## Notes

- This is a Python environment export, not a full system image. It does not capture OS packages, drivers, CUDA runtime outside Python wheels, or uncommitted source changes.
- The lock file contains GPU-related packages such as `nvidia-*` and `amdsmi`, so it is best reproduced on a compatible Linux machine.
- If the target machine intentionally differs in hardware or CUDA stack, you may need a separate lock file for that platform.
