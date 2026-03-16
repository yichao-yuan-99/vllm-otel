from __future__ import annotations

import importlib.util
import json
import re
import shlex
import sys
import tomllib
from pathlib import Path


def load_generator_module() -> object:
    module_path = (
        Path(__file__).resolve().parents[1] / "generate_replay_configs.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "generate_replay_qps_local_group_split_configs",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_replay_qps_local_group_split_configs"] = module
    spec.loader.exec_module(module)
    return module


def resolve_output_batch_dir(output_config_root: Path) -> Path:
    candidates = sorted(path for path in output_config_root.iterdir() if path.is_dir())
    assert len(candidates) == 1
    batch_dir = candidates[0]
    assert re.fullmatch(r"\d{8}T\d{6}Z", batch_dir.name) is not None
    return batch_dir


def test_main_generates_grouped_orchestrator_bundle(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    results_root = repo_root / "results"
    source_run_dir = results_root / "model-a" / "dataset-b" / "agent-c" / "run-1"
    source_run_dir.mkdir(parents=True)
    split_top_plan_path = (source_run_dir / "replay-plan.token.top.json").resolve()
    split_rest_plan_path = (source_run_dir / "replay-plan.token.rest.json").resolve()
    split_top_plan_path.write_text("{}", encoding="utf-8")
    split_rest_plan_path.write_text("{}", encoding="utf-8")

    submit_wrapper_dir = repo_root / "sbatch-orchestrator"
    submit_wrapper_dir.mkdir(parents=True)
    submit_wrapper_path = repo_root / "sbatch-orchestrator" / "submit-start-group.sh"
    submit_wrapper_path.write_text("#!/usr/bin/env bash\necho submit\n", encoding="utf-8")
    submit_wrapper_path.chmod(0o750)

    output_config_dir = tmp_path / "generated"

    module.REPO_ROOT = repo_root
    module.RESULTS_ROOT = results_root

    exit_code = module.main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--qps-list",
            "0.1,0.2",
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

    batch_dir = resolve_output_batch_dir(output_config_dir)
    top_qps_01_dir = batch_dir / "top" / "qps0_1"
    top_qps_02_dir = batch_dir / "top" / "qps0_2"
    rest_qps_01_dir = batch_dir / "rest" / "qps0_1"
    rest_qps_02_dir = batch_dir / "rest" / "qps0_2"

    for exp_dir in (top_qps_01_dir, top_qps_02_dir, rest_qps_01_dir, rest_qps_02_dir):
        assert (exp_dir / "replay.toml").exists()
        assert (exp_dir / "run_orchestrated_replay.sh").exists()

    replay_top_01 = tomllib.loads(
        (top_qps_01_dir / "replay.toml").read_text(encoding="utf-8")
    )["replay"]
    assert replay_top_01["plan"] == module.path_for_config(split_top_plan_path)
    assert replay_top_01["randomize_seed"] == 11
    assert replay_top_01["time_constraint_s"] == 600.0
    assert replay_top_01["port_profile_id"] == 0
    assert replay_top_01["launch_policy_override"]["seed"] == 7
    assert replay_top_01["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert replay_top_01["launch_policy_override"]["pattern_args"]["rate"] == 0.1

    replay_rest_01 = tomllib.loads(
        (rest_qps_01_dir / "replay.toml").read_text(encoding="utf-8")
    )["replay"]
    assert replay_rest_01["plan"] == module.path_for_config(split_rest_plan_path)
    assert replay_rest_01["launch_policy_override"]["pattern_args"]["rate"] == 0.1

    job_script_content = (top_qps_01_dir / "run_orchestrated_replay.sh").read_text(
        encoding="utf-8"
    )
    assert "--config \"${SCRIPT_DIR}/replay.toml\"" in job_script_content
    assert "--port-profile-id \"${PORT_PROFILE_ID_VALUE}\"" in job_script_content

    job_list_path = (batch_dir / "job-list.txt").resolve()
    assert job_list_path.exists()
    job_lines = [
        line.strip()
        for line in job_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(job_lines) == 4
    for raw_line in job_lines:
        tokens = shlex.split(raw_line)
        assert len(tokens) == 1
        script_path = Path(tokens[0]).resolve()
        assert script_path.is_absolute()
        assert script_path.exists()
        assert script_path.name == "run_orchestrated_replay.sh"

    assert not (batch_dir / "sbatch.sh").exists()

    assert not (batch_dir / "submit_all.sh").exists()

    expected_replay_timestamp_dir = (
        repo_root / "results" / "replay" / batch_dir.name
    ).resolve()
    expected_generated_copy_dir = (expected_replay_timestamp_dir / "generated").resolve()
    assert (expected_generated_copy_dir / "top" / "qps0_1" / "replay.toml").exists()
    assert (expected_generated_copy_dir / "top" / "qps0_2" / "replay.toml").exists()
    assert (expected_generated_copy_dir / "rest" / "qps0_1" / "replay.toml").exists()
    assert (expected_generated_copy_dir / "rest" / "qps0_2" / "replay.toml").exists()
    assert (expected_generated_copy_dir / "job-list.txt").exists()
    assert not (expected_generated_copy_dir / "sbatch.sh").exists()
    assert not (expected_generated_copy_dir / "submit_all.sh").exists()
    assert (expected_generated_copy_dir / "manifest.json").exists()

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["default_port_profile"] == 0
    assert manifest["generated_batch_copy_dir"] == module.path_for_config(
        expected_generated_copy_dir
    )
    assert manifest["generation_command_raw"] == shlex.join(
        [
            sys.executable,
            str((Path(module.__file__).resolve())),
            "--source-run-dir",
            str(source_run_dir),
            "--qps-list",
            "0.1,0.2",
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
    assert manifest["plan_paths"]["top"] == str(split_top_plan_path)
    assert manifest["plan_paths"]["rest"] == str(split_rest_plan_path)
    assert manifest["split_groups"] == ["top", "rest"]
    assert manifest["job_list"] == str(job_list_path)
    assert (
        manifest["orchestrator_summary_default"]
        == "sbatch-orchestrator/logs/<utc-timestamp>/orchestrator-summary.json"
    )
    assert manifest["submit_command"] == (
        f"bash {module.path_for_config(submit_wrapper_path.resolve())} "
        f"--job-list {module.path_for_config(job_list_path)}"
    )
    assert len(manifest["generated_experiments"]) == 4
    assert len(manifest["generated_experiments_by_group"]["top"]) == 2
    assert len(manifest["generated_experiments_by_group"]["rest"]) == 2
    for experiment in manifest["generated_experiments"]:
        assert experiment["group"] in {"top", "rest"}
        assert experiment["plan_path"] in {str(split_top_plan_path), str(split_rest_plan_path)}
        assert experiment["job_script"].endswith("run_orchestrated_replay.sh")
