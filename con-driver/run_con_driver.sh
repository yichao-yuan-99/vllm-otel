#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_CON_DRIVER="${VENV_DIR}/bin/con-driver"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

if [[ ! -x "${VENV_CON_DRIVER}" ]]; then
  if ! "${VENV_PYTHON}" -m pip -q install -e "${REPO_ROOT}/con-driver" >/dev/null 2>&1; then
    if ! "${VENV_PYTHON}" -m pip -q install --no-build-isolation -e "${REPO_ROOT}/con-driver" >/dev/null 2>&1; then
      echo "warning: could not install con-driver into .venv; using source fallback" >&2
    fi
  fi
fi

if [[ -x "${VENV_CON_DRIVER}" ]]; then
  exec "${VENV_CON_DRIVER}" "$@"
fi

# Fallback: run source module from the shared .venv python.
PYTHONPATH_PARTS=("${REPO_ROOT}/con-driver/src")

if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_PARTS+=("${PYTHONPATH}")
fi

PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")" \
  exec "${VENV_PYTHON}" -m con_driver "$@"
