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
        "generate_replay_qps_local_split_configs",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_replay_qps_local_split_configs"] = module
    spec.loader.exec_module(module)
    return module


def resolve_output_batch_dir(output_config_root: Path) -> Path:
    candidates = sorted(path for path in output_config_root.iterdir() if path.is_dir())
    assert len(candidates) == 1
    batch_dir = candidates[0]
    assert re.fullmatch(r"\d{8}T\d{6}Z", batch_dir.name) is not None
    return batch_dir


def test_main_generates_local_mode_bundle(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    results_root = repo_root / "results"
    source_run_dir = results_root / "model-a" / "dataset-b" / "agent-c" / "run-1"
    source_run_dir.mkdir(parents=True)
    split_top_plan_path = (source_run_dir / "replay-plan.token.top.json").resolve()
    split_rest_plan_path = (source_run_dir / "replay-plan.token.rest.json").resolve()
    split_top_plan_path.write_text("{}", encoding="utf-8")
    split_rest_plan_path.write_text("{}", encoding="utf-8")

    server_config_path = repo_root / "servers" / "servers-amdhpc" / "server_config.toml"
    server_config_path.parent.mkdir(parents=True)
    server_config_path.write_text("[server]\nport=1\n", encoding="utf-8")

    output_config_dir = tmp_path / "generated"
    rendered_dir = tmp_path / "rendered"
    rendered_dir.mkdir(parents=True)

    module.REPO_ROOT = repo_root
    module.RESULTS_ROOT = results_root

    def fake_build_control_plane(server_config: Path) -> object:
        assert server_config.resolve() == server_config_path.resolve()
        return object()

    def fake_render_local_mode_sbatch(
        *,
        control_plane: object,
        port_profile_id: int,
        partition: str,
        model: str,
        local_mode_script_path: Path,
        check_port_availability: bool,
        lmcache_max_local_cpu_size: str | None,
        no_async_scheduling: bool,
    ) -> Path:
        del control_plane, check_port_availability
        assert port_profile_id == 0
        assert partition == "mi3001x"
        assert model == "qwen3_coder_30b"
        assert lmcache_max_local_cpu_size == "100"
        assert no_async_scheduling is False
        assert local_mode_script_path.exists()
        group_name = local_mode_script_path.parent.parent.name
        qps_name = local_mode_script_path.parent.name
        target = rendered_dir / f"{group_name}-{qps_name}.sh"
        target.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "#SBATCH --output=/tmp/original/slurm.%j.out\n"
            "#SBATCH --error=/tmp/original/slurm.%j.err\n"
            "JOB_LOG_DIR=/tmp/original\n"
            "GATEWAY_CONFIG_DEFAULT=/tmp/original-gateway-config.toml\n"
            "echo fake\n",
            encoding="utf-8",
        )
        return target

    module.build_control_plane = fake_build_control_plane
    module.render_local_mode_sbatch = fake_render_local_mode_sbatch

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
            "--partition",
            "mi3001x",
            "--model",
            "qwen3_coder_30b",
            "--lmcache",
            "100",
            "--server-config",
            str(server_config_path),
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
        assert (exp_dir / "run_local_replay.sh").exists()
        assert (exp_dir / "gateway-config.toml").exists()
        assert (exp_dir / "sbatch.sh").exists()

    sbatch_top_01 = (top_qps_01_dir / "sbatch.sh").read_text(encoding="utf-8")
    assert (
        f"GATEWAY_CONFIG_DEFAULT={(top_qps_01_dir / 'gateway-config.toml').resolve()}"
        in sbatch_top_01
    )
    expected_replay_dir_top_01 = (
        repo_root
        / "results"
        / "replay"
        / batch_dir.name
        / "model-a"
        / "dataset-b"
        / "agent-c"
        / "sweep-qps-local"
        / "split"
        / "top"
        / "qps0_1"
    ).resolve()
    expected_log_dir_top_01 = (expected_replay_dir_top_01 / "sbatch-logs").resolve()
    assert f"#SBATCH --output={expected_log_dir_top_01}/slurm.%j.out" in sbatch_top_01
    assert f"#SBATCH --error={expected_log_dir_top_01}/slurm.%j.err" in sbatch_top_01
    assert f'JOB_LOG_DIR=\"{expected_log_dir_top_01}\"' in sbatch_top_01

    gateway_top_01 = tomllib.loads(
        (top_qps_01_dir / "gateway-config.toml").read_text(encoding="utf-8")
    )
    assert gateway_top_01["schema_version"] == 1
    assert gateway_top_01["run"]["port_profile_id"] == 0
    assert gateway_top_01["run"]["output_root"] == str(
        (top_qps_01_dir / "gateway-artifacts").resolve()
    )

    submit_all_path = (batch_dir / "submit_all.sh").resolve()
    assert submit_all_path.exists()
    submit_all_content = submit_all_path.read_text(encoding="utf-8")
    assert 'sbatch "${SCRIPT_DIR}/top/qps0_1/sbatch.sh"' in submit_all_content
    assert 'sbatch "${SCRIPT_DIR}/top/qps0_2/sbatch.sh"' in submit_all_content
    assert 'sbatch "${SCRIPT_DIR}/rest/qps0_1/sbatch.sh"' in submit_all_content
    assert 'sbatch "${SCRIPT_DIR}/rest/qps0_2/sbatch.sh"' in submit_all_content

    top_replay = tomllib.loads((top_qps_01_dir / "replay.toml").read_text(encoding="utf-8"))[
        "replay"
    ]
    assert top_replay["plan"] == module.path_for_config(split_top_plan_path)
    assert top_replay["randomize_seed"] == 11
    assert top_replay["time_constraint_s"] == 600.0
    assert top_replay["port_profile_id"] == 0
    assert top_replay["launch_policy_override"]["seed"] == 7
    assert top_replay["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert top_replay["launch_policy_override"]["pattern_args"]["rate"] == 0.1
    assert top_replay["output_dir"].startswith("results/replay/")

    rest_replay = tomllib.loads((rest_qps_01_dir / "replay.toml").read_text(encoding="utf-8"))[
        "replay"
    ]
    assert rest_replay["plan"] == module.path_for_config(split_rest_plan_path)
    assert rest_replay["launch_policy_override"]["pattern_args"]["rate"] == 0.1

    local_script = (top_qps_01_dir / "run_local_replay.sh").read_text(encoding="utf-8")
    assert "--config \"${SCRIPT_DIR}/replay.toml\"" in local_script
    assert "--port-profile-id \"${PORT_PROFILE_ID_VALUE}\"" in local_script

    expected_replay_timestamp_dir = (
        repo_root / "results" / "replay" / batch_dir.name
    ).resolve()
    expected_generated_copy_dir = (expected_replay_timestamp_dir / "generated").resolve()
    assert (expected_generated_copy_dir / "top" / "qps0_1" / "replay.toml").exists()
    assert (expected_generated_copy_dir / "top" / "qps0_2" / "replay.toml").exists()
    assert (expected_generated_copy_dir / "rest" / "qps0_1" / "replay.toml").exists()
    assert (expected_generated_copy_dir / "rest" / "qps0_2" / "replay.toml").exists()
    assert (expected_generated_copy_dir / "submit_all.sh").exists()
    assert (expected_generated_copy_dir / "manifest.json").exists()

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["partition"] == "mi3001x"
    assert manifest["model"] == "qwen3_coder_30b"
    assert manifest["lmcache"] == 100
    assert manifest["no_async_scheduling"] is False
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
            "--partition",
            "mi3001x",
            "--model",
            "qwen3_coder_30b",
            "--lmcache",
            "100",
            "--server-config",
            str(server_config_path),
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert manifest["plan_paths"]["top"] == str(split_top_plan_path)
    assert manifest["plan_paths"]["rest"] == str(split_rest_plan_path)
    assert manifest["split_groups"] == ["top", "rest"]
    assert manifest["port_profile"] == 0
    assert manifest["submit_all_script"] == str(submit_all_path)
    assert manifest["submit_all_command"].startswith("bash ")
    assert len(manifest["generated_experiments"]) == 4
    assert len(manifest["generated_experiments_by_group"]["top"]) == 2
    assert len(manifest["generated_experiments_by_group"]["rest"]) == 2
    for experiment in manifest["generated_experiments"]:
        assert experiment["group"] in {"top", "rest"}
        assert experiment["plan_path"] in {str(split_top_plan_path), str(split_rest_plan_path)}
        assert experiment["gateway_config"].endswith("gateway-config.toml")
        assert experiment["sbatch_log_dir"].endswith("/sbatch-logs")
        assert "generated_config_copy_dir" not in experiment
        assert "generated_config_copies" not in experiment
