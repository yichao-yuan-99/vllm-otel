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
        "generate_replay_qps_local_single_configs",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_replay_qps_local_single_configs"] = module
    spec.loader.exec_module(module)
    return module


def resolve_single_output_batch_dir(output_config_root: Path) -> Path:
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
    (source_run_dir / "replay-plan.json").write_text("{}", encoding="utf-8")

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
    ) -> Path:
        del control_plane, check_port_availability
        assert port_profile_id == 0
        assert partition == "mi3001x"
        assert model == "qwen3_coder_30b"
        assert local_mode_script_path.exists()
        target = rendered_dir / f"{local_mode_script_path.parent.name}.sh"
        target.write_text("#!/usr/bin/env bash\necho fake\n", encoding="utf-8")
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
            "--server-config",
            str(server_config_path),
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    batch_dir = resolve_single_output_batch_dir(output_config_dir)
    qps_01_dir = batch_dir / "qps0_1"
    qps_02_dir = batch_dir / "qps0_2"
    for exp_dir in (qps_01_dir, qps_02_dir):
        assert (exp_dir / "replay.toml").exists()
        assert (exp_dir / "run_local_replay.sh").exists()
        assert (exp_dir / "sbatch.sh").exists()

    replay_01 = tomllib.loads((qps_01_dir / "replay.toml").read_text(encoding="utf-8"))["replay"]
    assert replay_01["randomize_seed"] == 11
    assert replay_01["time_constraint_s"] == 600.0
    assert replay_01["port_profile_id"] == 0
    assert replay_01["launch_policy_override"]["seed"] == 7
    assert replay_01["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert replay_01["launch_policy_override"]["pattern_args"]["rate"] == 0.1
    assert replay_01["output_dir"].startswith("results/replay/")

    local_script = (qps_01_dir / "run_local_replay.sh").read_text(encoding="utf-8")
    assert "--config \"${SCRIPT_DIR}/replay.toml\"" in local_script
    assert "--port-profile-id \"${PORT_PROFILE_ID_VALUE}\"" in local_script

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["partition"] == "mi3001x"
    assert manifest["model"] == "qwen3_coder_30b"
    assert manifest["port_profile"] == 0
    assert len(manifest["generated_experiments"]) == 2
