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
        "generate_sweep_qps_docker_power_clean_gateway_multi_lowest_usage_experiment",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[
        "generate_sweep_qps_docker_power_clean_gateway_multi_lowest_usage_experiment"
    ] = module
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


def write_clean_plan_set(source_run_dir: Path, *, suffix: str | None = None) -> tuple[Path, Path, Path]:
    plan_payload = {
        "replay_target": {
            "model": "Test Served Model",
        },
        "launch_policy": {
            "strategy": "config_ordered",
            "max_concurrent": 25,
            "pattern": {"name": "eager"},
            "pattern_args": {},
        },
        "workers": [],
    }
    suffix_text = "" if suffix is None else f".{suffix}"
    top_plan_path = source_run_dir / f"replay-plan.clean.token.top{suffix_text}.json"
    rest_plan_path = source_run_dir / f"replay-plan.clean.token.rest{suffix_text}.json"
    exclude_plan_path = (
        source_run_dir / f"replay-plan.clean.token.exclude-unranked{suffix_text}.json"
    )
    for plan_path in (top_plan_path, rest_plan_path, exclude_plan_path):
        plan_path.write_text(
            json.dumps(plan_payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return top_plan_path, rest_plan_path, exclude_plan_path


def prepare_module_and_source_run(
    tmp_path: Path,
    *,
    suffix: str | None = None,
) -> tuple[object, Path]:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    results_root = repo_root / "results"
    source_run_dir = results_root / "model-a" / "dataset-a" / "agent-a" / "run-1"
    source_run_dir.mkdir(parents=True)

    module.REPO_ROOT = repo_root
    module.RESULTS_ROOT = results_root
    module.MODEL_CONFIG_PATH = repo_root / "configs" / "model_config.toml"
    write_minimal_model_config(module.MODEL_CONFIG_PATH)
    write_clean_plan_set(source_run_dir, suffix=suffix)
    return module, source_run_dir


def test_main_generates_sweep_qps_docker_power_clean_gateway_multi_lowest_usage_bundle(
    tmp_path: Path,
) -> None:
    module, source_run_dir = prepare_module_and_source_run(tmp_path)
    output_config_dir = tmp_path / "generated"

    exit_code = module.main(
        [
            "--source-run-dir",
            str(source_run_dir),
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
            "3,4",
            "--split",
            "rest",
            "--power-gpu-indices",
            "2,3",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    batch_dir = resolve_single_output_batch_dir(output_config_dir)
    batch_timestamp = batch_dir.name
    expected_plan_copy = batch_dir / "plan" / "replay-plan.clean.token.rest.json"

    replay_config_qps025 = tomllib.loads(
        (batch_dir / "qps0_25" / "replay.toml").read_text(encoding="utf-8")
    )
    replay_qps025 = replay_config_qps025["replay"]
    assert replay_qps025["randomize_seed"] == 11
    assert replay_qps025["time_constraint_s"] == 600.0
    assert replay_qps025["port_profile_id"] == 3
    assert replay_qps025["port_profile_id_list"] == [3, 4]
    assert replay_qps025["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert replay_qps025["launch_policy_override"]["pattern_args"]["rate"] == 0.25
    assert replay_qps025["launch_policy_override"]["seed"] == 7
    assert replay_qps025["plan"] == module.BASE.path_for_config(expected_plan_copy)
    assert replay_qps025["output_dir"].endswith(
        "results/replay/sweep-qps-docker-power-clean-gateway_multi-lowest_usage/"
        f"dataset-a/agent-a/split/rest/qps0_25/{batch_timestamp}"
    )

    replay_config_qps03 = tomllib.loads(
        (batch_dir / "qps0_3" / "replay.toml").read_text(encoding="utf-8")
    )
    replay_qps03 = replay_config_qps03["replay"]
    assert replay_qps03["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert replay_qps03["launch_policy_override"]["pattern_args"]["rate"] == 0.3
    assert replay_qps03["launch_policy_override"]["seed"] == 7
    assert replay_qps03["plan"] == module.BASE.path_for_config(expected_plan_copy)
    assert replay_qps03["output_dir"].endswith(
        "results/replay/sweep-qps-docker-power-clean-gateway_multi-lowest_usage/"
        f"dataset-a/agent-a/split/rest/qps0_3/{batch_timestamp}"
    )

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["source_run_dir"] == str(source_run_dir)
    assert manifest["source_dataset_lineage"] == "dataset-a/agent-a"
    assert manifest["target_model"] == "test_model"
    assert manifest["split"] == "rest"
    assert manifest["split_two_group_metric"] == "token_usage"
    assert manifest["launch_pattern"] == "poisson"
    assert manifest["poisson_seed"] == 7
    assert manifest["randomize_seed"] == 11
    assert manifest["qps_list"] == [0.25, 0.3]
    assert manifest["port_profile"] == 3
    assert manifest["control_port_profile"] == 3
    assert manifest["port_profile_ids"] == [3, 4]
    assert manifest["assignment_policy"] == "lowest_usage"
    assert manifest["power_gpu_indices"] == [2, 3]
    assert manifest["selected_source_plan"].endswith("replay-plan.clean.token.rest.json")
    assert manifest["selected_plan_copy"] == str(expected_plan_copy)
    assert len(manifest["qps_points"]) == 2
    assert manifest["qps_points"][0]["qps"] == 0.25
    assert manifest["qps_points"][0]["qps_slug"] == "qps0_25"
    assert manifest["qps_points"][0]["power_output_dir"].endswith(
        "results/replay/sweep-qps-docker-power-clean-gateway_multi-lowest_usage/"
        f"dataset-a/agent-a/split/rest/qps0_25/{batch_timestamp}/power"
    )
    assert manifest["qps_points"][1]["qps"] == 0.3
    assert manifest["qps_points"][1]["qps_slug"] == "qps0_3"
    assert manifest["run_command_with_port_profiles"].endswith("run_replay.sh 3,4")

    run_script_path = batch_dir / "run_replay.sh"
    run_script_text = run_script_path.read_text(encoding="utf-8")
    assert "[sweep-qps-docker-power-clean-gateway_multi-lowest_usage]" in run_script_text
    assert "DEFAULT_PORT_PROFILE_IDS_CSV=3,4" in run_script_text
    assert 'PORT_PROFILE_IDS_VALUE="${1:-${PORT_PROFILE_IDS:-${DEFAULT_PORT_PROFILE_IDS_CSV}}}"' in run_script_text
    assert 'CONTROL_PORT_PROFILE_ID="${PORT_PROFILE_IDS_ARRAY[0]}"' in run_script_text
    assert 'PORT_PROFILE_IDS_CSV_NORMALIZED="$(IFS=,; printf' in run_script_text
    assert 'CURL_BIN="${CURL_BIN:-curl}"' in run_script_text
    assert "TARGET_ASSIGNMENT_POLICY=lowest_usage" in run_script_text
    assert "POWER_GPU_INDICES=(2 3)" in run_script_text
    assert "set_gateway_assignment_policy" in run_script_text
    assert 'GATEWAY_BASE_URL_VALUE="${GATEWAY_BASE_URL:-}"' in run_script_text
    assert '    --data "{\\"assignment_policy\\": \\"${TARGET_ASSIGNMENT_POLICY}\\"}" \\' in run_script_text
    assert '    "${GATEWAY_BASE_URL_RESOLVED}/policy"' in run_script_text
    assert "port_profiles=${PORT_PROFILE_IDS_CSV_NORMALIZED}" in run_script_text
    assert '    --port-profile-id "${CONTROL_PORT_PROFILE_ID}" \\' in run_script_text
    assert '    --port-profile-id-list "${PORT_PROFILE_IDS_CSV_NORMALIZED}"' in run_script_text
    assert "run_one_qps 0.25 qps0_25 qps0_25/replay.toml" in run_script_text
    assert "run_one_qps 0.3 qps0_3 qps0_3/replay.toml" in run_script_text
    assert run_script_path.stat().st_mode & stat.S_IXUSR


def test_main_supports_repeated_port_profile_flags_and_additional_suffix(
    tmp_path: Path,
) -> None:
    module, source_run_dir = prepare_module_and_source_run(tmp_path, suffix="fp8")
    output_config_dir = tmp_path / "generated"

    exit_code = module.main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--poisson-seed",
            "7",
            "--randomize-seed",
            "11",
            "--qps-list",
            "0.5",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "5",
            "--port-profile",
            "6",
            "--split",
            "full",
            "--additional-suffix",
            "fp8",
            "--power-gpu-indices",
            "0,1",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    batch_dir = resolve_single_output_batch_dir(output_config_dir)
    batch_timestamp = batch_dir.name
    expected_plan_copy = batch_dir / "plan" / "replay-plan.clean.token.exclude-unranked.fp8.json"

    replay_config = tomllib.loads(
        (batch_dir / "qps0_5" / "replay.toml").read_text(encoding="utf-8")
    )
    assert replay_config["replay"]["output_dir"].endswith(
        "results/replay/sweep-qps-docker-power-clean-gateway_multi-lowest_usage/"
        f"dataset-a/agent-a/split/exclude-unranked/qps0_5/{batch_timestamp}"
    )
    assert replay_config["replay"]["plan"] == module.BASE.path_for_config(expected_plan_copy)
    assert replay_config["replay"]["port_profile_id"] == 5
    assert replay_config["replay"]["port_profile_id_list"] == [5, 6]

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "exclude-unranked"
    assert manifest["additional_suffix"] == "fp8"
    assert manifest["port_profile"] == 5
    assert manifest["control_port_profile"] == 5
    assert manifest["port_profile_ids"] == [5, 6]
    assert manifest["assignment_policy"] == "lowest_usage"
    assert manifest["selected_source_plan"].endswith(
        "replay-plan.clean.token.exclude-unranked.fp8.json"
    )
    assert manifest["selected_plan_copy"] == str(expected_plan_copy)
    assert manifest["power_gpu_indices"] == [0, 1]

    run_script_text = (batch_dir / "run_replay.sh").read_text(encoding="utf-8")
    assert "DEFAULT_PORT_PROFILE_IDS_CSV=5,6" in run_script_text
    assert "TARGET_ASSIGNMENT_POLICY=lowest_usage" in run_script_text
    assert "POWER_GPU_INDICES=(0 1)" in run_script_text
    assert '    --port-profile-id-list "${PORT_PROFILE_IDS_CSV_NORMALIZED}"' in run_script_text


def test_main_rejects_duplicate_port_profile_ids(tmp_path: Path) -> None:
    module, source_run_dir = prepare_module_and_source_run(tmp_path)
    output_config_dir = tmp_path / "generated"

    exit_code = module.main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--poisson-seed",
            "7",
            "--randomize-seed",
            "11",
            "--qps-list",
            "0.25",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3,3",
            "--split",
            "rest",
            "--power-gpu-indices",
            "2,3",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )

    assert exit_code == 1
