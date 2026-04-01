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
        "generate_sweep_qps_multi_agent_experiment",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_sweep_qps_multi_agent_experiment"] = module
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
    single_trail: str | None,
    source_job_dir: str = "/remote/work/results/model-a/dataset-a/agent-a/run-1",
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
    plan_path.write_text(
        json.dumps(plan_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def test_main_generates_sweep_qps_multi_agent_bundle(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    module.REPO_ROOT = repo_root
    module.MODEL_CONFIG_PATH = repo_root / "configs" / "model_config.toml"
    write_minimal_model_config(module.MODEL_CONFIG_PATH)

    source_plan_path_a = tmp_path / "input-plan" / "replay-plan.trail-profile-1_run_alpha.json"
    source_plan_path_b = tmp_path / "input-plan" / "replay-plan.trail-profile-2_run_beta.json"
    write_single_trail_plan(source_plan_path_a, single_trail="profile-1/run_alpha")
    write_single_trail_plan(source_plan_path_b, single_trail="profile-2/run_beta")

    output_config_dir = tmp_path / "generated"
    exit_code = module.main(
        [
            "--source-plan",
            str(source_plan_path_a),
            "--source-plan",
            str(source_plan_path_b),
            "--poisson-seed",
            "7",
            "--randomize-seed",
            "11",
            "--qps-list",
            "0.25,0.3",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--power-gpu-indices",
            "0,1",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    batch_dir = resolve_single_output_batch_dir(output_config_dir)
    batch_timestamp = batch_dir.name
    expected_plan_copy_a = batch_dir / "plan" / f"plan-00.{source_plan_path_a.name}"
    expected_plan_copy_b = batch_dir / "plan" / f"plan-01.{source_plan_path_b.name}"

    replay_config_a_qps025 = tomllib.loads(
        (batch_dir / "trail" / "profile-1_run_alpha" / "qps0_25" / "replay.toml").read_text(
            encoding="utf-8"
        )
    )
    replay_a_qps025 = replay_config_a_qps025["replay"]
    assert replay_a_qps025["plan"] == module.path_for_config(expected_plan_copy_a)
    assert replay_a_qps025["launch_policy_override"]["seed"] == 7
    assert replay_a_qps025["launch_policy_override"]["pattern_args"]["rate"] == 0.25
    assert replay_a_qps025["output_dir"].endswith(
        "results/replay/sweep-qps-multi-agent/"
        f"dataset-a/agent-a/trail/profile-1_run_alpha/qps0_25/{batch_timestamp}"
    )

    replay_config_b_qps03 = tomllib.loads(
        (batch_dir / "trail" / "profile-2_run_beta" / "qps0_3" / "replay.toml").read_text(
            encoding="utf-8"
        )
    )
    replay_b_qps03 = replay_config_b_qps03["replay"]
    assert replay_b_qps03["plan"] == module.path_for_config(expected_plan_copy_b)
    assert replay_b_qps03["launch_policy_override"]["pattern_args"]["rate"] == 0.3
    assert replay_b_qps03["output_dir"].endswith(
        "results/replay/sweep-qps-multi-agent/"
        f"dataset-a/agent-a/trail/profile-2_run_beta/qps0_3/{batch_timestamp}"
    )

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["source_plan_count"] == 2
    assert manifest["poisson_seed"] == 7
    assert manifest["qps_list"] == [0.25, 0.3]
    assert manifest["power_gpu_indices"] == [0, 1]
    assert len(manifest["trail_points"]) == 2
    assert manifest["trail_points"][0]["source_trail_name"] == "profile-1/run_alpha"
    assert manifest["trail_points"][0]["batch_trail_slug"] == "profile-1_run_alpha"
    assert manifest["trail_points"][1]["source_trail_name"] == "profile-2/run_beta"
    assert manifest["trail_points"][1]["batch_trail_slug"] == "profile-2_run_beta"
    assert len(manifest["matrix_points"]) == 4
    assert manifest["matrix_points"][0]["qps"] == 0.25
    assert manifest["matrix_points"][0]["batch_trail_slug"] == "profile-1_run_alpha"
    assert manifest["matrix_points"][3]["qps"] == 0.3
    assert manifest["matrix_points"][3]["batch_trail_slug"] == "profile-2_run_beta"

    run_script_path = batch_dir / "run_replay.sh"
    run_script_text = run_script_path.read_text(encoding="utf-8")
    assert "[sweep-qps-multi-agent]" in run_script_text
    assert "POWER_GPU_INDICES=(0 1)" in run_script_text
    assert (
        "run_one_job profile-1_run_alpha profile-1/run_alpha 0.25 qps0_25 "
        "trail/profile-1_run_alpha/qps0_25/replay.toml"
    ) in run_script_text
    assert (
        "run_one_job profile-2_run_beta profile-2/run_beta 0.3 qps0_3 "
        "trail/profile-2_run_beta/qps0_3/replay.toml"
    ) in run_script_text
    assert run_script_path.stat().st_mode & stat.S_IXUSR


def test_main_rejects_non_single_trail_plan(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    module.REPO_ROOT = repo_root
    module.MODEL_CONFIG_PATH = repo_root / "configs" / "model_config.toml"
    write_minimal_model_config(module.MODEL_CONFIG_PATH)

    source_plan_path_a = tmp_path / "input-plan" / "replay-plan.trail-profile-1_run_alpha.json"
    source_plan_path_b = tmp_path / "input-plan" / "replay-plan.json"
    write_single_trail_plan(source_plan_path_a, single_trail="profile-1/run_alpha")
    write_single_trail_plan(source_plan_path_b, single_trail=None)

    output_config_dir = tmp_path / "generated"
    exit_code = module.main(
        [
            "--source-plan",
            str(source_plan_path_a),
            "--source-plan",
            str(source_plan_path_b),
            "--poisson-seed",
            "7",
            "--randomize-seed",
            "11",
            "--qps-list",
            "0.25,0.3",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 1
