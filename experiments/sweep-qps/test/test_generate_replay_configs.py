from __future__ import annotations

import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path


def load_generator_module() -> object:
    module_path = (
        Path(__file__).resolve().parents[1] / "generate_replay_configs.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "generate_replay_qps_configs",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_replay_qps_configs"] = module
    spec.loader.exec_module(module)
    return module


def resolve_single_output_batch_dir(output_config_root: Path) -> Path:
    candidates = sorted(path for path in output_config_root.iterdir() if path.is_dir())
    assert len(candidates) == 1
    batch_dir = candidates[0]
    assert re.fullmatch(r"\d{8}T\d{6}Z", batch_dir.name) is not None
    return batch_dir


def test_main_generates_configs_and_manifest(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    results_root = repo_root / "results"
    source_run_dir = (
        results_root / "qwen3-coder-30b" / "dabstep" / "mini-swe-agent" / "run-1"
    )
    source_run_dir.mkdir(parents=True)
    (source_run_dir / "replay-plan.json").write_text("{}", encoding="utf-8")

    output_config_dir = tmp_path / "generated"
    module.REPO_ROOT = repo_root
    module.RESULTS_ROOT = results_root

    exit_code = module.main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--qps-list",
            "1.2,1.5",
            "--poisson-seed",
            "7",
            "--randomize-seed",
            "11",
            "--time-constraint-s",
            "600",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    output_config_batch_dir = resolve_single_output_batch_dir(output_config_dir)
    config_12 = tomllib.loads(
        (output_config_batch_dir / "replay.qps1_2.toml").read_text(encoding="utf-8")
    )
    config_15 = tomllib.loads(
        (output_config_batch_dir / "replay.qps1_5.toml").read_text(encoding="utf-8")
    )

    replay_12 = config_12["replay"]
    replay_15 = config_15["replay"]
    assert replay_12["randomize_seed"] == 11
    assert replay_12["time_constraint_s"] == 600.0
    assert "num_tasks" not in replay_12
    assert replay_12["launch_policy_override"]["seed"] == 7
    assert replay_12["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert replay_12["launch_policy_override"]["pattern_args"]["rate"] == 1.2
    assert replay_15["launch_policy_override"]["pattern_args"]["rate"] == 1.5

    manifest = json.loads((output_config_batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["output_config_root_dir"] == str(output_config_dir.resolve())
    assert manifest["output_config_dir"] == str(output_config_batch_dir.resolve())
    assert re.fullmatch(r"\d{8}T\d{6}Z", manifest["batch_timestamp"]) is not None
    replay_prefix = f"results/replay/{manifest['batch_timestamp']}/"
    assert replay_12["output_dir"].startswith(replay_prefix)
    assert replay_15["output_dir"].startswith(replay_prefix)
    batch_dir_12 = replay_12["output_dir"].removesuffix("/qps1_2")
    batch_dir_15 = replay_15["output_dir"].removesuffix("/qps1_5")
    assert batch_dir_12 == batch_dir_15
    assert batch_dir_12 == (
        f"results/replay/{manifest['batch_timestamp']}/"
        "qwen3-coder-30b/dabstep/mini-swe-agent/sweep-qps"
    )
    assert manifest["replay_output_batch_dir"] == batch_dir_12
    assert manifest["poisson_seed"] == 7
    assert manifest["randomize_seed"] == 11
    assert manifest["time_constraint_s"] == 600.0
    assert manifest["qps_list"] == [1.2, 1.5]
    assert len(manifest["generated_configs"]) == 2


def test_main_forwards_optional_settings_and_merges_overrides(tmp_path: Path) -> None:
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
            "--qps-list",
            "2.0",
            "--poisson-seed",
            "17",
            "--randomize-seed",
            "23",
            "--time-constraint-s",
            "180",
            "--output-config-dir",
            str(output_config_dir),
            "--port-profile-id",
            "3",
            "--vllm-log-interval-s",
            "2.0",
            "--vllm-log-timeout-s",
            "7.0",
            "--launch-policy-override-json",
            '{"max_concurrent":4,"pattern_args":{"mean_interval_s":5}}',
            "--extra-replay-json",
            '{"launch_policy_override":{"max_concurrent":2},"custom_key":"x"}',
        ]
    )
    assert exit_code == 0

    output_config_batch_dir = resolve_single_output_batch_dir(output_config_dir)
    config = tomllib.loads(
        (output_config_batch_dir / "replay.qps2.toml").read_text(encoding="utf-8")
    )
    replay = config["replay"]
    assert replay["port_profile_id"] == 3
    assert replay["vllm_log_interval_s"] == 2.0
    assert replay["vllm_log_timeout_s"] == 7.0
    assert replay["custom_key"] == "x"
    assert replay["launch_policy_override"]["max_concurrent"] == 4
    assert replay["launch_policy_override"]["seed"] == 17
    assert replay["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert replay["launch_policy_override"]["pattern_args"]["mean_interval_s"] == 5
    assert replay["launch_policy_override"]["pattern_args"]["rate"] == 2.0


def test_main_rejects_extra_replay_num_tasks(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    results_root = repo_root / "results"
    source_run_dir = results_root / "x" / "y" / "z" / "run-3"
    source_run_dir.mkdir(parents=True)
    (source_run_dir / "replay-plan.json").write_text("{}", encoding="utf-8")

    output_config_dir = tmp_path / "generated"
    module.REPO_ROOT = repo_root
    module.RESULTS_ROOT = results_root

    exit_code = module.main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--qps-list",
            "1.0",
            "--poisson-seed",
            "1",
            "--randomize-seed",
            "2",
            "--time-constraint-s",
            "180",
            "--output-config-dir",
            str(output_config_dir),
            "--extra-replay-json",
            '{"num_tasks":100}',
        ]
    )
    assert exit_code == 1


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
            "--qps-list",
            "1.0",
            "--poisson-seed",
            "1",
            "--randomize-seed",
            "2",
            "--time-constraint-s",
            "180",
            "--output-config-dir",
            str(output_config_dir),
            "--plan-path",
            str(source_run_dir / "replay-plan.json"),
        ]
    )
    assert exit_code == 1
