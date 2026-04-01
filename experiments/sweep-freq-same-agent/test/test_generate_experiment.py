from __future__ import annotations

import importlib.util
import json
import re
import stat
import sys
import tomllib
from pathlib import Path


def load_generator_module() -> object:
    module_path = (Path(__file__).resolve().parents[1] / "generate_experiment.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "generate_sweep_freq_same_agent_experiment",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_sweep_freq_same_agent_experiment"] = module
    spec.loader.exec_module(module)
    return module


def resolve_single_output_batch_dir(output_config_root: Path) -> Path:
    candidates = sorted(path for path in output_config_root.iterdir() if path.is_dir())
    assert len(candidates) == 1
    batch_dir = candidates[0]
    assert re.fullmatch(r"\d{8}T\d{6}Z", batch_dir.name) is not None
    return batch_dir


def write_minimal_model_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "",
                "[models.test_model]",
                'served_model_name = "Test Served Model"',
                'vllm_model_name = "org/test-model"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_single_trail_plan(
    plan_path: Path,
    *,
    single_trail: str | None = "profile-2/run_beta",
    source_job_dir: str = "/remote/work/results/model-a/dataset-a/agent-a/run-1",
    is_derived: bool = False,
) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    compile_options = {
        "exclude_unranked_trails": False,
        "clean": False,
    }
    if single_trail is not None:
        compile_options["single_trail"] = single_trail

    plan_payload = {
        "compile_options": compile_options,
        "source_job_dir": source_job_dir,
        "replay_target": {
            "model": "Test Served Model",
        },
        "launch_policy": {
            "strategy": "config_ordered",
            "max_concurrent": 1,
            "pattern": {"name": "eager"},
            "pattern_args": {},
        },
        "workers": [],
    }
    if is_derived:
        plan_payload["is_derived"] = True
        plan_payload["derived_single_plan"] = {
            "seed": 7,
            "size": 500,
            "source_plan_path": "/remote/work/results/model-a/dataset-a/agent-a/run-1/replay-plan.trail-profile-2_run_beta.json",
        }
    plan_path.write_text(
        json.dumps(plan_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def test_main_generates_sweep_freq_same_agent_qps_bundle(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    module.REPO_ROOT = repo_root
    module.MODEL_CONFIG_PATH = repo_root / "configs" / "model_config.toml"
    write_minimal_model_config(module.MODEL_CONFIG_PATH)

    source_plan_path = tmp_path / "input-plan" / "replay-plan.trail-profile-2_run_beta.json"
    write_single_trail_plan(source_plan_path, is_derived=True)

    output_config_dir = tmp_path / "generated"
    exit_code = module.main(
        [
            "--source-plan",
            str(source_plan_path),
            "--poisson-seed",
            "7",
            "--randomize-seed",
            "11",
            "--qps",
            "0.2",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--freq-list",
            "345:810;345:1305",
            "--gpu-id",
            "0",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    batch_dir = resolve_single_output_batch_dir(output_config_dir)
    batch_timestamp = batch_dir.name
    expected_plan_copy = batch_dir / "plan" / source_plan_path.name

    replay_config = tomllib.loads(
        (batch_dir / "freq" / "core-345-810" / "replay.toml").read_text(encoding="utf-8")
    )
    replay_payload = replay_config["replay"]
    assert replay_payload["plan"] == module.path_for_config(expected_plan_copy)
    assert replay_payload["randomize_seed"] == 11
    assert replay_payload["time_constraint_s"] == 600.0
    assert replay_payload["port_profile_id"] == 3
    assert replay_payload["launch_policy_override"]["seed"] == 7
    assert replay_payload["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert replay_payload["launch_policy_override"]["pattern_args"]["rate"] == 0.2
    assert replay_payload["output_dir"].endswith(
        "results/replay/sweep-freq-same-agent/"
        f"dataset-a/agent-a/trail/profile-2_run_beta/qps0_2/{batch_timestamp}/core-345-810"
    )

    replay_config_high = tomllib.loads(
        (batch_dir / "freq" / "core-345-1305" / "replay.toml").read_text(encoding="utf-8")
    )
    assert replay_config_high["replay"]["output_dir"].endswith(
        "results/replay/sweep-freq-same-agent/"
        f"dataset-a/agent-a/trail/profile-2_run_beta/qps0_2/{batch_timestamp}/core-345-1305"
    )

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["source_plan"] == str(source_plan_path)
    assert manifest["source_plan_copy"] == str(expected_plan_copy)
    assert manifest["source_plan_is_derived"] is True
    assert manifest["source_job_dir"] == "/remote/work/results/model-a/dataset-a/agent-a/run-1"
    assert manifest["source_dataset_lineage"] == "dataset-a/agent-a"
    assert manifest["source_trail_name"] == "profile-2/run_beta"
    assert manifest["source_trail_slug"] == "profile-2_run_beta"
    assert manifest["source_plan_model"] == "Test Served Model"
    assert manifest["target_model"] == "test_model"
    assert manifest["launch_pattern"] == "poisson"
    assert manifest["launch_mode"] == "qps"
    assert manifest["launch_label"] == "qps"
    assert manifest["launch_value"] == "0.2"
    assert manifest["launch_target_slug"] == "qps0_2"
    assert manifest["poisson_seed"] == 7
    assert manifest["qps"] == 0.2
    assert manifest["qps_slug"] == "qps0_2"
    assert manifest["max_concurrent"] is None
    assert manifest["max_concurrent_slug"] is None
    assert manifest["gpu_id"] == 0
    assert manifest["freq_list"] == "345:810;345:1305"
    assert manifest["replay_output_base_dir"].endswith(
        "results/replay/sweep-freq-same-agent/"
        f"dataset-a/agent-a/trail/profile-2_run_beta/qps0_2/{batch_timestamp}"
    )
    assert len(manifest["frequency_points"]) == 2
    assert manifest["frequency_points"][0]["slug"] == "core-345-810"
    assert manifest["frequency_points"][1]["slug"] == "core-345-1305"

    run_script_path = batch_dir / "run_replay.sh"
    run_script_text = run_script_path.read_text(encoding="utf-8")
    assert "[sweep-freq-same-agent]" in run_script_text
    assert "SOURCE_TRAIL_NAME=profile-2/run_beta" in run_script_text
    assert "LAUNCH_MODE_NAME=qps" in run_script_text
    assert "LAUNCH_LABEL_NAME=qps" in run_script_text
    assert "LAUNCH_VALUE=0.2" in run_script_text
    assert "run_one_frequency 345 810 core-345-810 freq/core-345-810/replay.toml" in run_script_text
    assert "run_one_frequency 345 1305 core-345-1305 freq/core-345-1305/replay.toml" in run_script_text
    assert run_script_path.stat().st_mode & stat.S_IXUSR


def test_main_generates_sweep_freq_same_agent_maxcon_bundle(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    module.REPO_ROOT = repo_root
    module.MODEL_CONFIG_PATH = repo_root / "configs" / "model_config.toml"
    write_minimal_model_config(module.MODEL_CONFIG_PATH)

    source_plan_path = tmp_path / "input-plan" / "replay-plan.trail-profile-2_run_beta.json"
    write_single_trail_plan(source_plan_path, is_derived=True)

    output_config_dir = tmp_path / "generated"
    exit_code = module.main(
        [
            "--source-plan",
            str(source_plan_path),
            "--randomize-seed",
            "11",
            "--max-concurrent",
            "8",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--freq-list",
            "345:810;345:1305",
            "--gpu-id",
            "0",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    batch_dir = resolve_single_output_batch_dir(output_config_dir)
    batch_timestamp = batch_dir.name
    expected_plan_copy = batch_dir / "plan" / source_plan_path.name

    replay_config = tomllib.loads(
        (batch_dir / "freq" / "core-345-810" / "replay.toml").read_text(encoding="utf-8")
    )
    replay_payload = replay_config["replay"]
    assert replay_payload["plan"] == module.path_for_config(expected_plan_copy)
    assert replay_payload["randomize_seed"] == 11
    assert replay_payload["time_constraint_s"] == 600.0
    assert replay_payload["port_profile_id"] == 3
    assert replay_payload["launch_policy_override"]["max_concurrent"] == 8
    assert "seed" not in replay_payload["launch_policy_override"]
    assert "pattern" not in replay_payload["launch_policy_override"]
    assert replay_payload["output_dir"].endswith(
        "results/replay/sweep-freq-same-agent/"
        f"dataset-a/agent-a/trail/profile-2_run_beta/c8/{batch_timestamp}/core-345-810"
    )

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["launch_pattern"] == "poisson"
    assert manifest["launch_mode"] == "max_concurrent"
    assert manifest["launch_label"] == "max_concurrent"
    assert manifest["launch_value"] == "8"
    assert manifest["launch_target_slug"] == "c8"
    assert manifest["poisson_seed"] is None
    assert manifest["qps"] is None
    assert manifest["qps_slug"] is None
    assert manifest["max_concurrent"] == 8
    assert manifest["max_concurrent_slug"] == "c8"
    assert manifest["replay_output_base_dir"].endswith(
        "results/replay/sweep-freq-same-agent/"
        f"dataset-a/agent-a/trail/profile-2_run_beta/c8/{batch_timestamp}"
    )

    run_script_path = batch_dir / "run_replay.sh"
    run_script_text = run_script_path.read_text(encoding="utf-8")
    assert "LAUNCH_MODE_NAME=max_concurrent" in run_script_text
    assert "LAUNCH_LABEL_NAME=max_concurrent" in run_script_text
    assert "LAUNCH_VALUE=8" in run_script_text
    assert run_script_path.stat().st_mode & stat.S_IXUSR


def test_main_rejects_original_single_trail_plan(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    module.REPO_ROOT = repo_root
    module.MODEL_CONFIG_PATH = repo_root / "configs" / "model_config.toml"
    write_minimal_model_config(module.MODEL_CONFIG_PATH)

    source_plan_path = tmp_path / "input-plan" / "replay-plan.trail-profile-2_run_beta.json"
    write_single_trail_plan(source_plan_path, is_derived=False)

    output_config_dir = tmp_path / "generated"
    exit_code = module.main(
        [
            "--source-plan",
            str(source_plan_path),
            "--poisson-seed",
            "7",
            "--randomize-seed",
            "11",
            "--qps",
            "0.2",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--freq-list",
            "345:810",
            "--gpu-id",
            "0",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 1


def test_main_rejects_plan_missing_single_trail_metadata(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    module.REPO_ROOT = repo_root
    module.MODEL_CONFIG_PATH = repo_root / "configs" / "model_config.toml"
    write_minimal_model_config(module.MODEL_CONFIG_PATH)

    source_plan_path = tmp_path / "input-plan" / "replay-plan.json"
    write_single_trail_plan(source_plan_path, single_trail=None, is_derived=True)

    output_config_dir = tmp_path / "generated"
    exit_code = module.main(
        [
            "--source-plan",
            str(source_plan_path),
            "--poisson-seed",
            "7",
            "--randomize-seed",
            "11",
            "--qps",
            "0.2",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--freq-list",
            "345:810",
            "--gpu-id",
            "0",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 1
