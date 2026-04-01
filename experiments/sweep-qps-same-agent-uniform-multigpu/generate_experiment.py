#!/usr/bin/env python3
"""Generate a fixed-interval uniform same-agent QPS sweep bundle for multigpu port profiles."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


_THIS_FILE = Path(__file__).resolve()
_BASE_GENERATOR_PATH = (
    _THIS_FILE.parents[1] / "sweep-qps-same-agent" / "generate_experiment.py"
).resolve()


def _load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "generate_sweep_qps_same_agent_base_multigpu",
        _BASE_GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base generator from {_BASE_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_sweep_qps_same_agent_base_multigpu"] = module
    spec.loader.exec_module(module)
    return module


_BASE_MODULE = _load_base_module()
_BASE_MODULE.configure_experiment_variant(
    experiment_dir_name="sweep-qps-same-agent-uniform-multigpu",
    launch_pattern_name="uniform",
    launch_pattern_label="Uniform",
    launch_seed_option=None,
    launch_seed_dest=None,
    launch_seed_manifest_key=None,
    generator_script_path=_THIS_FILE,
)

REPO_ROOT = _BASE_MODULE.REPO_ROOT
MODEL_CONFIG_PATH = _BASE_MODULE.MODEL_CONFIG_PATH
DEFAULT_OUTPUT_CONFIG_DIR = _BASE_MODULE.DEFAULT_OUTPUT_CONFIG_DIR
DEFAULT_REPLAY_OUTPUT_ROOT = _BASE_MODULE.DEFAULT_REPLAY_OUTPUT_ROOT


def _sync_base_module() -> None:
    _BASE_MODULE.REPO_ROOT = REPO_ROOT
    _BASE_MODULE.MODEL_CONFIG_PATH = MODEL_CONFIG_PATH
    _BASE_MODULE.DEFAULT_OUTPUT_CONFIG_DIR = DEFAULT_OUTPUT_CONFIG_DIR
    _BASE_MODULE.DEFAULT_REPLAY_OUTPUT_ROOT = DEFAULT_REPLAY_OUTPUT_ROOT
    _BASE_MODULE.GENERATOR_SCRIPT_PATH = _THIS_FILE


def path_for_config(path: Path) -> str:
    _sync_base_module()
    return _BASE_MODULE.path_for_config(path)


def main(argv: list[str] | None = None) -> int:
    _sync_base_module()
    return _BASE_MODULE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
