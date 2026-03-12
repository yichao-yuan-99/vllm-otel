from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest


def load_generator_module() -> object:
    module_path = (
        Path(__file__).resolve().parents[1] / "generate_replay_configs.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location("generate_replay_configs", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_replay_configs"] = module
    spec.loader.exec_module(module)
    return module


def test_main_generates_configs_and_manifest(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    results_root = repo_root / "results"
    source_run_dir = results_root / "qwen3-coder-30b" / "dabstep" / "mini-swe-agent" / "run-1"
    source_run_dir.mkdir(parents=True)
    (source_run_dir / "replay-plan.json").write_text("{}", encoding="utf-8")

    output_config_dir = tmp_path / "generated"
    module.REPO_ROOT = repo_root
    module.RESULTS_ROOT = results_root

    exit_code = module.main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--concurrency-list",
            "15,30",
            "--num-tasks",
            "120",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    config_15 = tomllib.loads((output_config_dir / "replay.c15.toml").read_text(encoding="utf-8"))
    config_30 = tomllib.loads((output_config_dir / "replay.c30.toml").read_text(encoding="utf-8"))
    assert config_15["replay"]["num_tasks"] == 120
    assert config_15["replay"]["launch_policy_override"]["max_concurrent"] == 15
    assert (
        config_15["replay"]["output_dir"]
        == "results/replay/qwen3-coder-30b/dabstep/mini-swe-agent/sweep-concurrency/15"
    )
    assert config_30["replay"]["launch_policy_override"]["max_concurrent"] == 30

    manifest = json.loads((output_config_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["num_tasks"] == 120
    assert manifest["concurrency_list"] == [15, 30]
    assert len(manifest["generated_configs"]) == 2


def test_main_forwards_optional_replay_settings(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    results_root = repo_root / "results"
    source_run_dir = results_root / "x" / "y" / "z" / "run-2"
    source_run_dir.mkdir(parents=True)
    (source_run_dir / "replay-plan.json").write_text("{}", encoding="utf-8")

    output_config_dir = tmp_path / "generated"
    module.REPO_ROOT = repo_root
    module.RESULTS_ROOT = results_root

    exit_code = module.main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--concurrency-list",
            "4",
            "--num-tasks",
            "50",
            "--output-config-dir",
            str(output_config_dir),
            "--port-profile-id",
            "3",
            "--vllm-log-interval-s",
            "2.0",
            "--vllm-log-timeout-s",
            "7.0",
            "--launch-policy-override-json",
            '{"pattern":{"name":"poisson"}}',
            "--extra-replay-json",
            '{"launch_policy_override":{"seed":11},"custom_key":"x"}',
        ]
    )
    assert exit_code == 0

    config = tomllib.loads((output_config_dir / "replay.c4.toml").read_text(encoding="utf-8"))
    replay = config["replay"]
    assert replay["port_profile_id"] == 3
    assert replay["vllm_log_interval_s"] == 2.0
    assert replay["vllm_log_timeout_s"] == 7.0
    assert replay["custom_key"] == "x"
    assert replay["launch_policy_override"]["seed"] == 11
    assert replay["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert replay["launch_policy_override"]["max_concurrent"] == 4


def test_main_rejects_source_outside_results(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    results_root = repo_root / "results"
    source_run_dir = tmp_path / "outside-results" / "run"
    source_run_dir.mkdir(parents=True)
    (source_run_dir / "replay-plan.json").write_text("{}", encoding="utf-8")

    output_config_dir = tmp_path / "generated"
    module.REPO_ROOT = repo_root
    module.RESULTS_ROOT = results_root

    exit_code = module.main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--concurrency-list",
            "2",
            "--num-tasks",
            "20",
            "--output-config-dir",
            str(output_config_dir),
            "--plan-path",
            str(source_run_dir / "replay-plan.json"),
        ]
    )
    assert exit_code == 1
