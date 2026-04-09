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
        "generate_sweep_qps_docker_power_clean_freq_ctrl_linespace_sweep_slo_og_experiment",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[
        "generate_sweep_qps_docker_power_clean_freq_ctrl_linespace_sweep_slo_og_experiment"
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


def write_clean_plan_set(
    source_run_dir: Path,
    *,
    suffix: str | None = None,
) -> tuple[Path, Path, Path]:
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


def test_main_generates_gateway_only_slo_sweep_qps_slo_bundle(tmp_path: Path) -> None:
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
            "--target-slo-list",
            "8,12",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--split",
            "rest",
            "--gpu-id",
            "2",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    batch_dir = resolve_single_output_batch_dir(output_config_dir)
    batch_timestamp = batch_dir.name
    expected_plan_copy = batch_dir / "plan" / "replay-plan.clean.token.rest.json"

    replay_config_qps025_slo8 = tomllib.loads(
        (batch_dir / "qps0_25" / "slo8" / "replay.toml").read_text(encoding="utf-8")
    )
    replay_qps025_slo8 = replay_config_qps025_slo8["replay"]
    assert replay_qps025_slo8["randomize_seed"] == 11
    assert replay_qps025_slo8["time_constraint_s"] == 600.0
    assert replay_qps025_slo8["port_profile_id"] == 3
    assert replay_qps025_slo8["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert replay_qps025_slo8["launch_policy_override"]["pattern_args"]["rate"] == 0.25
    assert replay_qps025_slo8["launch_policy_override"]["seed"] == 7
    assert replay_qps025_slo8["plan"] == module.path_for_config(expected_plan_copy)
    assert replay_qps025_slo8["output_dir"].endswith(
        "results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo-og/"
        f"dataset-a/agent-a/split/rest/qps0_25/slo8/{batch_timestamp}"
    )

    replay_config_qps03_slo12 = tomllib.loads(
        (batch_dir / "qps0_3" / "slo12" / "replay.toml").read_text(encoding="utf-8")
    )
    replay_qps03_slo12 = replay_config_qps03_slo12["replay"]
    assert replay_qps03_slo12["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert replay_qps03_slo12["launch_policy_override"]["pattern_args"]["rate"] == 0.3
    assert replay_qps03_slo12["launch_policy_override"]["seed"] == 7
    assert replay_qps03_slo12["plan"] == module.path_for_config(expected_plan_copy)
    assert replay_qps03_slo12["output_dir"].endswith(
        "results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo-og/"
        f"dataset-a/agent-a/split/rest/qps0_3/slo12/{batch_timestamp}"
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
    assert manifest["target_slo_list"] == [8.0, 12.0]
    assert manifest["gpu_id"] == 2
    assert manifest["power_gpu_indices"] == [2]
    assert manifest["ctx_aware_enabled"] is True
    assert manifest["ctx_aware_policy_mode"] == module.CTX_AWARE_POLICY_MODE
    assert (
        manifest["ctx_aware_usage_threshold_tokens"]
        == module.DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS
    )
    assert (
        manifest["ctx_aware_scheduling_threshold_tokens"]
        == module.DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS
    )
    assert manifest["slo_aware_enabled"] is True
    assert manifest["slo_aware_policy_mode"] == module.SLO_AWARE_POLICY_MODE
    assert manifest["slo_targets_drive_gateway_only"] is True
    assert manifest["slo_targets_drive_gateway_and_freq_controller"] is False
    assert manifest["freq_controller_uses_slo_target"] is False
    assert manifest["freq_controller_threshold"] == 395784.0
    assert manifest["freq_controller_bin_default"] == "freq-controller-linespace"
    assert manifest["selected_source_plan"].endswith("replay-plan.clean.token.rest.json")
    assert manifest["selected_plan_copy"] == str(expected_plan_copy)
    assert manifest["total_qps_slo_runs"] == 4
    assert len(manifest["qps_points"]) == 2
    assert manifest["qps_points"][0]["qps"] == 0.25
    assert manifest["qps_points"][0]["qps_slug"] == "qps0_25"
    assert [point["target_slo"] for point in manifest["qps_points"][0]["slo_points"]] == [
        8.0,
        12.0,
    ]
    assert (
        manifest["qps_points"][0]["slo_points"][0]["slo_aware_target_tokens_per_s"] == 8.0
    )
    assert manifest["qps_points"][0]["slo_points"][0]["power_output_dir"].endswith(
        "results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo-og/"
        f"dataset-a/agent-a/split/rest/qps0_25/slo8/{batch_timestamp}/power"
    )
    assert manifest["qps_points"][0]["slo_points"][0]["freq_controller_log_dir"].endswith(
        "results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo-og/"
        f"dataset-a/agent-a/split/rest/qps0_25/slo8/{batch_timestamp}/freq-control-linespace"
    )
    assert manifest["qps_points"][1]["qps"] == 0.3
    assert manifest["qps_points"][1]["qps_slug"] == "qps0_3"

    run_script_path = batch_dir / "run_replay.sh"
    run_script_text = run_script_path.read_text(encoding="utf-8")
    assert "[sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo-og]" in run_script_text
    assert "GPU_ID=2" in run_script_text
    assert 'GATEWAY_BASE_URL_VALUE="${GATEWAY_BASE_URL:-}"' in run_script_text
    assert "FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH" not in run_script_text
    assert "gateway-ipc-socket-path" not in run_script_text
    assert "resolve_gateway_base_url() {" in run_script_text
    assert (
        "FREQ_CONTROLLER_BIN=\"${FREQ_CONTROLLER_BIN:-freq-controller-linespace}\""
        in run_script_text
    )
    assert (
        "DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS="
        f"{module.DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS}"
    ) in run_script_text
    assert (
        "DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS="
        f"{module.DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS}"
    ) in run_script_text
    assert "for qps_index in \"${!QPS_VALUES[@]}\"; do" in run_script_text
    assert "for slo_index in \"${!slo_values[@]}\"; do" in run_script_text
    assert "run_one_qps_slo" in run_script_text
    assert "start_ctx_aware_mode" in run_script_text
    assert "start_slo_aware_mode" in run_script_text
    assert "end_slo_aware_mode" in run_script_text
    assert "end_ctx_aware_mode" in run_script_text
    assert '${GATEWAY_BASE_URL_RESOLVED}/ctx-aware/start' in run_script_text
    assert '${GATEWAY_BASE_URL_RESOLVED}/slo-aware/start' in run_script_text
    assert '${GATEWAY_BASE_URL_RESOLVED}/slo-aware/end' in run_script_text
    assert '${GATEWAY_BASE_URL_RESOLVED}/ctx-aware/end' in run_script_text
    assert f'\\"policy_mode\\": \\"{module.CTX_AWARE_POLICY_MODE}\\"' in run_script_text
    assert f'\\"policy_mode\\": \\"{module.SLO_AWARE_POLICY_MODE}\\"' in run_script_text
    assert '"target_tokens_per_s\\": ${target_slo_value}' in run_script_text
    assert "\"--threshold\" \"${FREQ_CONTROLLER_THRESHOLD_VALUE}\"" in run_script_text
    assert "\"--throughput-target\" \"${target_slo_value}\"" not in run_script_text
    assert "qps_slug=\"${QPS_SLUGS[$qps_index]}\"" in run_script_text
    assert "slo_values=(8 12)" in run_script_text
    assert "slo_slugs=(slo8 slo12)" in run_script_text
    assert "qps0_25/slo8/replay.toml" in run_script_text
    assert "qps0_25/slo12/replay.toml" in run_script_text
    assert "qps0_3/slo8/replay.toml" in run_script_text
    assert "qps0_3/slo12/replay.toml" in run_script_text
    assert run_script_path.stat().st_mode & stat.S_IXUSR


def test_main_supports_additional_suffix_and_custom_thresholds(tmp_path: Path) -> None:
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
            "--target-slo-list",
            "6.5",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--split",
            "full",
            "--additional-suffix",
            "fp8",
            "--gpu-id",
            "2",
            "--ctx-aware-usage-threshold-tokens",
            "600000",
            "--ctx-aware-scheduling-threshold-tokens",
            "500000",
            "--freq-controller-threshold",
            "200",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    batch_dir = resolve_single_output_batch_dir(output_config_dir)
    batch_timestamp = batch_dir.name
    expected_plan_copy = batch_dir / "plan" / "replay-plan.clean.token.exclude-unranked.fp8.json"

    replay_config = tomllib.loads(
        (batch_dir / "qps0_5" / "slo6_5" / "replay.toml").read_text(encoding="utf-8")
    )
    assert replay_config["replay"]["output_dir"].endswith(
        "results/replay/sweep-qps-docker-power-clean-freq-ctrl-linespace-sweep-slo-og/"
        f"dataset-a/agent-a/split/exclude-unranked/qps0_5/slo6_5/{batch_timestamp}"
    )
    assert replay_config["replay"]["plan"] == module.path_for_config(expected_plan_copy)

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "exclude-unranked"
    assert manifest["additional_suffix"] == "fp8"
    assert manifest["selected_source_plan"].endswith(
        "replay-plan.clean.token.exclude-unranked.fp8.json"
    )
    assert manifest["selected_plan_copy"] == str(expected_plan_copy)
    assert manifest["ctx_aware_usage_threshold_tokens"] == 600000
    assert manifest["ctx_aware_scheduling_threshold_tokens"] == 500000
    assert manifest["freq_controller_threshold"] == 200.0

    run_script_text = (batch_dir / "run_replay.sh").read_text(encoding="utf-8")
    assert "DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS=600000" in run_script_text
    assert "DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS=500000" in run_script_text
    assert "DEFAULT_FREQ_CONTROLLER_THRESHOLD=200.0" in run_script_text


def test_main_rejects_invalid_ctx_aware_threshold_order(tmp_path: Path) -> None:
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
            "--target-slo-list",
            "8",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--split",
            "rest",
            "--gpu-id",
            "2",
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
            "--target-slo-list",
            "8",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--split",
            "rest",
            "--gpu-id",
            "2",
            "--ctx-aware-scheduling-threshold-tokens",
            "2999",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )

    assert exit_code == 1
