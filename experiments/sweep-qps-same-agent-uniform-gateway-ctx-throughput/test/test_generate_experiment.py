from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import stat
import sys
import tomllib


def load_generator_module() -> object:
    module_path = (Path(__file__).resolve().parents[1] / "generate_experiment.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "generate_sweep_qps_same_agent_uniform_gateway_ctx_throughput_experiment",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[
        "generate_sweep_qps_same_agent_uniform_gateway_ctx_throughput_experiment"
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


def prepare_module(tmp_path: Path) -> object:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    module.REPO_ROOT = repo_root
    module.MODEL_CONFIG_PATH = repo_root / "configs" / "model_config.toml"
    write_minimal_model_config(module.MODEL_CONFIG_PATH)
    return module


def test_main_generates_sweep_qps_same_agent_uniform_gateway_ctx_throughput_bundle(
    tmp_path: Path,
) -> None:
    module = prepare_module(tmp_path)
    source_plan_path = tmp_path / "input-plan" / "replay-plan.trail-profile-2_run_beta.json"
    write_single_trail_plan(source_plan_path, is_derived=True)

    output_config_dir = tmp_path / "generated"
    exit_code = module.main(
        [
            "--source-plan",
            str(source_plan_path),
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
    expected_plan_copy = batch_dir / "plan" / source_plan_path.name

    replay_config_qps025 = tomllib.loads(
        (batch_dir / "qps0_25" / "replay.toml").read_text(encoding="utf-8")
    )
    replay_qps025 = replay_config_qps025["replay"]
    assert replay_qps025["randomize_seed"] == 11
    assert replay_qps025["time_constraint_s"] == 600.0
    assert replay_qps025["port_profile_id"] == 3
    assert replay_qps025["launch_policy_override"]["pattern"]["name"] == "uniform"
    assert replay_qps025["launch_policy_override"]["pattern_args"]["rate"] == 0.25
    assert "seed" not in replay_qps025["launch_policy_override"]
    assert replay_qps025["plan"] == module.path_for_config(expected_plan_copy)
    assert replay_qps025["output_dir"].endswith(
        "results/replay/sweep-qps-same-agent-uniform-gateway-ctx-throughput/"
        f"dataset-a/agent-a/trail/profile-2_run_beta/qps0_25/{batch_timestamp}"
    )

    replay_config_qps03 = tomllib.loads(
        (batch_dir / "qps0_3" / "replay.toml").read_text(encoding="utf-8")
    )
    replay_qps03 = replay_config_qps03["replay"]
    assert replay_qps03["launch_policy_override"]["pattern"]["name"] == "uniform"
    assert replay_qps03["launch_policy_override"]["pattern_args"]["rate"] == 0.3
    assert replay_qps03["plan"] == module.path_for_config(expected_plan_copy)
    assert replay_qps03["output_dir"].endswith(
        "results/replay/sweep-qps-same-agent-uniform-gateway-ctx-throughput/"
        f"dataset-a/agent-a/trail/profile-2_run_beta/qps0_3/{batch_timestamp}"
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
    assert manifest["launch_pattern"] == "uniform"
    assert "uniform_seed" not in manifest
    assert manifest["qps_list"] == [0.25, 0.3]
    assert manifest["power_gpu_indices"] == [0, 1]
    assert manifest["ctx_aware_policy_mode"] == module.CTX_AWARE_POLICY_MODE
    assert (
        manifest["ctx_aware_usage_threshold_tokens"]
        == module.DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS
    )
    assert (
        manifest["ctx_aware_scheduling_threshold_tokens"]
        == module.DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS
    )
    assert manifest["replay_output_root_base_dir"].endswith(
        "results/replay/sweep-qps-same-agent-uniform-gateway-ctx-throughput/"
        "dataset-a/agent-a/trail/profile-2_run_beta"
    )
    assert len(manifest["qps_points"]) == 2
    assert manifest["qps_points"][0]["qps"] == 0.25
    assert manifest["qps_points"][0]["qps_slug"] == "qps0_25"
    assert manifest["qps_points"][0]["power_output_dir"].endswith(
        "results/replay/sweep-qps-same-agent-uniform-gateway-ctx-throughput/"
        f"dataset-a/agent-a/trail/profile-2_run_beta/qps0_25/{batch_timestamp}/power"
    )
    assert manifest["qps_points"][1]["qps"] == 0.3
    assert manifest["qps_points"][1]["qps_slug"] == "qps0_3"

    run_script_path = batch_dir / "run_replay.sh"
    run_script_text = run_script_path.read_text(encoding="utf-8")
    assert "[sweep-qps-same-agent-uniform-gateway-ctx-throughput]" in run_script_text
    assert "SOURCE_TRAIL_NAME=profile-2/run_beta" in run_script_text
    assert f"mode={module.CTX_AWARE_POLICY_MODE}" in run_script_text
    assert (
        "DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS="
        f"{module.DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS}"
    ) in run_script_text
    assert (
        "DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS="
        f"{module.DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS}"
    ) in run_script_text
    assert 'GATEWAY_BASE_URL_VALUE="${GATEWAY_BASE_URL:-}"' in run_script_text
    assert "resolve_gateway_base_url() {" in run_script_text
    assert '${GATEWAY_BASE_URL_RESOLVED}/ctx-aware/start' in run_script_text
    assert '${GATEWAY_BASE_URL_RESOLVED}/ctx-aware/end' in run_script_text
    assert f'\\"policy_mode\\": \\"{module.CTX_AWARE_POLICY_MODE}\\"' in run_script_text
    assert "start_ctx_aware_mode" in run_script_text
    assert "end_ctx_aware_mode" in run_script_text
    assert "POWER_GPU_INDICES=(0 1)" in run_script_text
    assert "run_one_qps 0.25 qps0_25 qps0_25/replay.toml" in run_script_text
    assert "run_one_qps 0.3 qps0_3 qps0_3/replay.toml" in run_script_text
    assert run_script_path.stat().st_mode & stat.S_IXUSR


def test_main_applies_output_suffix_and_custom_ctx_aware_thresholds(
    tmp_path: Path,
) -> None:
    module = prepare_module(tmp_path)
    source_plan_path = tmp_path / "input-plan" / "replay-plan.trail-profile-2_run_beta.json"
    write_single_trail_plan(source_plan_path, is_derived=True)

    output_config_dir = tmp_path / "generated"
    exit_code = module.main(
        [
            "--source-plan",
            str(source_plan_path),
            "--randomize-seed",
            "11",
            "--qps-list",
            "0.25",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--power-gpu-indices",
            "2",
            "--output-suffix",
            "lmcache",
            "--ctx-aware-usage-threshold-tokens",
            "600000",
            "--ctx-aware-scheduling-threshold-tokens",
            "500000",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    batch_dir = resolve_single_output_batch_dir(output_config_dir)
    batch_timestamp = batch_dir.name
    replay_config = tomllib.loads(
        (batch_dir / "qps0_25" / "replay.toml").read_text(encoding="utf-8")
    )
    assert replay_config["replay"]["output_dir"].endswith(
        "results/replay/sweep-qps-same-agent-uniform-gateway-ctx-throughput/"
        f"dataset-a/agent-a/trail-lmcache/profile-2_run_beta/qps0_25/{batch_timestamp}"
    )

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_suffix"] == "lmcache"
    assert manifest["replay_output_trail_dir_name"] == "trail-lmcache"
    assert manifest["ctx_aware_policy_mode"] == module.CTX_AWARE_POLICY_MODE
    assert manifest["replay_output_root_base_dir"].endswith(
        "results/replay/sweep-qps-same-agent-uniform-gateway-ctx-throughput/"
        "dataset-a/agent-a/trail-lmcache/profile-2_run_beta"
    )
    assert manifest["power_gpu_indices"] == [2]
    assert manifest["ctx_aware_usage_threshold_tokens"] == 600000
    assert manifest["ctx_aware_scheduling_threshold_tokens"] == 500000

    run_script_text = (batch_dir / "run_replay.sh").read_text(encoding="utf-8")
    assert "DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS=600000" in run_script_text
    assert "DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS=500000" in run_script_text
    assert "POWER_GPU_INDICES=(2)" in run_script_text


def test_main_rejects_original_single_trail_plan(tmp_path: Path) -> None:
    module = prepare_module(tmp_path)
    source_plan_path = tmp_path / "input-plan" / "replay-plan.trail-profile-2_run_beta.json"
    write_single_trail_plan(source_plan_path, is_derived=False)

    output_config_dir = tmp_path / "generated"
    exit_code = module.main(
        [
            "--source-plan",
            str(source_plan_path),
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


def test_main_rejects_invalid_ctx_aware_threshold_order(tmp_path: Path) -> None:
    module = prepare_module(tmp_path)
    source_plan_path = tmp_path / "input-plan" / "replay-plan.trail-profile-2_run_beta.json"
    write_single_trail_plan(source_plan_path, is_derived=True)

    output_config_dir = tmp_path / "generated"
    exit_code = module.main(
        [
            "--source-plan",
            str(source_plan_path),
            "--randomize-seed",
            "11",
            "--qps-list",
            "0.25",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--ctx-aware-usage-threshold-tokens",
            "5000",
            "--ctx-aware-scheduling-threshold-tokens",
            "5000",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 1


def test_main_rejects_ctx_aware_scheduling_threshold_below_new_agent_floor(
    tmp_path: Path,
) -> None:
    module = prepare_module(tmp_path)
    source_plan_path = tmp_path / "input-plan" / "replay-plan.trail-profile-2_run_beta.json"
    write_single_trail_plan(source_plan_path, is_derived=True)

    output_config_dir = tmp_path / "generated"
    exit_code = module.main(
        [
            "--source-plan",
            str(source_plan_path),
            "--randomize-seed",
            "11",
            "--qps-list",
            "0.25",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--ctx-aware-usage-threshold-tokens",
            "5000",
            "--ctx-aware-scheduling-threshold-tokens",
            "2999",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 1
