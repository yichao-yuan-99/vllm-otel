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
        "generate_amd_mi3008x_embedded_tp1_sweep_qps_freq_ctrl_linespace_amd_sweep_slo_experiment",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[
        "generate_amd_mi3008x_embedded_tp1_sweep_qps_freq_ctrl_linespace_amd_sweep_slo_experiment"
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


def test_main_generates_mi3008x_amd_sweep_slo_bundle(tmp_path: Path) -> None:
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
            "--split",
            "rest",
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
    assert replay_qps025_slo8["port_profile_id"] == 0
    assert replay_qps025_slo8["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert replay_qps025_slo8["launch_policy_override"]["pattern_args"]["rate"] == 0.25
    assert replay_qps025_slo8["launch_policy_override"]["seed"] == 7
    assert replay_qps025_slo8["plan"] == module.BASE.path_for_config(expected_plan_copy)
    assert replay_qps025_slo8["output_dir"].endswith(
        "results/replay/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/"
        "sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo-aggressive/"
        f"dataset-a/agent-a/split/rest/qps0_25/slo8/{batch_timestamp}"
    )

    replay_config_qps03_slo12 = tomllib.loads(
        (batch_dir / "qps0_3" / "slo12" / "replay.toml").read_text(encoding="utf-8")
    )
    replay_qps03_slo12 = replay_config_qps03_slo12["replay"]
    assert replay_qps03_slo12["port_profile_id"] == 1
    assert replay_qps03_slo12["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert replay_qps03_slo12["launch_policy_override"]["pattern_args"]["rate"] == 0.3
    assert replay_qps03_slo12["launch_policy_override"]["seed"] == 7
    assert replay_qps03_slo12["plan"] == module.BASE.path_for_config(expected_plan_copy)
    assert replay_qps03_slo12["output_dir"].endswith(
        "results/replay/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/"
        "sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo-aggressive/"
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
    assert manifest["max_qps_points"] == 8
    assert manifest["assigned_qps_points"] == 2
    assert manifest["total_slo_rounds"] == 2
    assert manifest["total_qps_slo_runs"] == 4
    assert manifest["port_profile"] == 0
    assert manifest["default_port_profile"] == 0
    assert manifest["gpu_index"] is None
    assert manifest["gpu_index_runtime_default"] == "match-port-profile"
    assert manifest["profile_output_suffix"] == "profile-<port_profile_id>"
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
    assert manifest["slo_targets_drive_gateway_and_freq_controller"] is True
    assert manifest["freq_controller_aggressive_slo_control"] is True
    assert manifest["freq_controller_threshold"] == module.DEFAULT_FREQ_CONTROLLER_THRESHOLD
    assert manifest["freq_controller_bin_default"] == module.DEFAULT_FREQ_CONTROLLER_BIN_NAME
    assert (
        manifest["freq_controller_gateway_ipc_socket_default"]
        == "/tmp/vllm-gateway-ctx-profile-<port_profile_id>.sock"
    )
    assert manifest["selected_source_plan"].endswith("replay-plan.clean.token.rest.json")
    assert manifest["selected_plan_copy"] == str(expected_plan_copy)
    assert len(manifest["qps_points"]) == 2
    assert manifest["qps_points"][0]["qps"] == 0.25
    assert manifest["qps_points"][0]["qps_slug"] == "qps0_25"
    assert manifest["qps_points"][0]["assigned_port_profile"] == 0
    assert manifest["qps_points"][0]["assigned_gpu_index"] == 0
    assert [point["target_slo"] for point in manifest["qps_points"][0]["slo_points"]] == [
        8.0,
        12.0,
    ]
    assert manifest["qps_points"][0]["slo_points"][0]["replay_output_dir"].endswith(
        "results/replay/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/"
        "sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo-aggressive/"
        f"dataset-a/agent-a/split/rest/qps0_25/slo8/{batch_timestamp}/profile-0"
    )
    assert manifest["qps_points"][0]["slo_points"][0]["power_output_dir"].endswith(
        "results/replay/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/"
        "sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo-aggressive/"
        f"dataset-a/agent-a/split/rest/qps0_25/slo8/{batch_timestamp}/profile-0/power"
    )
    assert manifest["qps_points"][0]["slo_points"][0]["freq_controller_log_dir"].endswith(
        "results/replay/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/"
        "sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo-aggressive/"
        f"dataset-a/agent-a/split/rest/qps0_25/slo8/{batch_timestamp}/profile-0/freq-control-linespace"
    )
    assert manifest["qps_points"][1]["assigned_port_profile"] == 1
    assert manifest["qps_points"][1]["assigned_gpu_index"] == 1
    assert manifest["run_command_with_port_profile"].endswith("run_replay.sh 0")
    assert manifest["submit_command_default"].endswith("submit_embedded_tp1.sh")
    assert "servers/servers-amdhpc-mi3008x-embedded-TP1/launch.py submit" in manifest[
        "embedded_tp1_submit_command"
    ]

    run_script_path = batch_dir / "run_replay.sh"
    run_script_text = run_script_path.read_text(encoding="utf-8")
    assert (
        "[amd-embeded-servers-amdhpc-mi3008x-embedded-TP1-sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo-aggressive]"
        in run_script_text
    )
    assert "DEFAULT_PORT_PROFILE_ID=0" in run_script_text
    assert 'PORT_PROFILE_ID_VALUE="${1:-${PORT_PROFILE_ID:-${DEFAULT_PORT_PROFILE_ID}}}"' in run_script_text
    assert 'GPU_INDEX_VALUE="${GPU_INDEX:-}"' in run_script_text
    assert "ASSIGNED_QPS_POINTS=2" in run_script_text
    assert "TOTAL_SLO_ROUNDS=2" in run_script_text
    assert (
        'PYTHON_BIN="${PYTHON_BIN:-'
        f"{module._resolve_default_python_bin()}"
        '}"'
    ) in run_script_text
    assert (
        'CURL_BIN="${CURL_BIN:-'
        f"{module._resolve_default_curl_bin()}"
        '}"'
    ) in run_script_text
    assert (
        'AMD_POWER_READER_BIN="${AMD_POWER_READER_BIN:-'
        f"{(module.REPO_ROOT / '.venv' / 'bin' / 'amd-power-reader').resolve()}"
        '}"'
    ) in run_script_text
    assert (
        'FREQ_CONTROLLER_BIN="${FREQ_CONTROLLER_BIN:-'
        f"{(module.REPO_ROOT / '.venv' / 'bin' / module.DEFAULT_FREQ_CONTROLLER_BIN_NAME).resolve()}"
        '}"'
    ) in run_script_text
    assert (
        'RESET_GPU_CORE_FREQ_BIN="${RESET_GPU_CORE_FREQ_BIN:-'
        f"{(module.REPO_ROOT / '.venv' / 'bin' / 'amd-reset-gpu-core-freq').resolve()}"
        '}"'
    ) in run_script_text
    assert (
        'DEFAULT_FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH="/tmp/vllm-gateway-ctx-profile-${PORT_PROFILE_ID_VALUE}.sock"'
        in run_script_text
    )
    assert (
        'FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH_VALUE="${FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH:-${DEFAULT_FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH}}"'
        in run_script_text
    )
    assert "select_qps_job() {" in run_script_text
    assert "no qps assigned for port_profile=${PORT_PROFILE_ID_VALUE}" in run_script_text
    assert "start_ctx_aware_mode" in run_script_text
    assert "start_slo_aware_mode" in run_script_text
    assert "end_slo_aware_mode" in run_script_text
    assert "end_ctx_aware_mode" in run_script_text
    assert "stop_freq_controller" in run_script_text
    assert "reset_gpu_core_if_needed" in run_script_text
    assert '${GATEWAY_BASE_URL_RESOLVED}/ctx-aware/start' in run_script_text
    assert '${GATEWAY_BASE_URL_RESOLVED}/slo-aware/start' in run_script_text
    assert '${GATEWAY_BASE_URL_RESOLVED}/slo-aware/end' in run_script_text
    assert '${GATEWAY_BASE_URL_RESOLVED}/ctx-aware/end' in run_script_text
    assert f'\\"policy_mode\\": \\"{module.CTX_AWARE_POLICY_MODE}\\"' in run_script_text
    assert f'\\"policy_mode\\": \\"{module.SLO_AWARE_POLICY_MODE}\\"' in run_script_text
    assert '"target_tokens_per_s\\": ${target_slo_value}' in run_script_text
    assert 'normalize_gpu_index "${GPU_INDEX_VALUE}" "${SELECTED_ASSIGNED_GPU_INDEX}"' in run_script_text
    assert '"--threshold" "${FREQ_CONTROLLER_THRESHOLD_VALUE}"' in run_script_text
    assert '"--throughput-target" "${target_slo_value}"' in run_script_text
    assert '"--aggresive"' in run_script_text
    assert (
        '"--gateway-ipc-socket-path" "${FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH_VALUE}"'
        in run_script_text
    )
    assert '"--port-profile-id" "${PORT_PROFILE_ID_VALUE}"' in run_script_text
    assert '"--gpu-index" "${GPU_INDEX_VALUE}"' in run_script_text
    assert '"${RESET_GPU_CORE_FREQ_BIN}" --gpu-index "${GPU_INDEX_VALUE}"' in run_script_text
    assert "for slo_index in \"${!SELECTED_SLO_VALUES[@]}\"; do" in run_script_text
    assert "SELECTED_SLO_VALUES=(8 12)" in run_script_text
    assert "SELECTED_SLO_SLUGS=(slo8 slo12)" in run_script_text
    assert "SELECTED_REPLAY_CONFIG_REFS=(qps0_25/slo8/replay.toml qps0_25/slo12/replay.toml)" in run_script_text
    assert "SELECTED_REPLAY_CONFIG_REFS=(qps0_3/slo8/replay.toml qps0_3/slo12/replay.toml)" in run_script_text
    assert run_script_path.stat().st_mode & stat.S_IXUSR

    submit_script_path = batch_dir / "submit_embedded_tp1.sh"
    submit_script_text = submit_script_path.read_text(encoding="utf-8")
    assert (
        'PYTHON_BIN="${PYTHON_BIN:-'
        f"{module._resolve_default_python_bin()}"
        '}"'
    ) in submit_script_text
    assert (
        'EMBEDDED_TP1_LAUNCH_SCRIPT="${EMBEDDED_TP1_LAUNCH_SCRIPT:-'
        'servers/servers-amdhpc-mi3008x-embedded-TP1/launch.py}"'
    ) in submit_script_text
    assert 'RUN_SCRIPT_PATH="${SCRIPT_DIR}/run_replay.sh"' in submit_script_text
    assert 'exec "${PYTHON_BIN}" "${EMBEDDED_TP1_LAUNCH_SCRIPT}" submit \\' in submit_script_text
    assert '  -m "${TARGET_MODEL}" \\' in submit_script_text
    assert '  -e "${RUN_SCRIPT_PATH}"' in submit_script_text
    assert (
        "# Equivalent raw command: "
        f"{module._resolve_default_python_bin()} "
    ) in submit_script_text
    assert "run_replay.sh" in submit_script_text
    assert submit_script_path.stat().st_mode & stat.S_IXUSR


def test_main_supports_output_suffix_and_nonzero_default_port_profile(tmp_path: Path) -> None:
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
            "2",
            "--gpu-index",
            "2",
            "--split",
            "full",
            "--additional-suffix",
            "fp8",
            "--output-suffix",
            "lmcache",
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
        "results/replay/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/"
        "sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo-aggressive-lmcache/"
        f"dataset-a/agent-a/split/exclude-unranked/qps0_5/slo6_5/{batch_timestamp}"
    )
    assert replay_config["replay"]["plan"] == module.BASE.path_for_config(expected_plan_copy)
    assert replay_config["replay"]["port_profile_id"] == 0

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "exclude-unranked"
    assert manifest["additional_suffix"] == "fp8"
    assert manifest["output_suffix"] == "lmcache"
    assert manifest["port_profile"] == 2
    assert manifest["default_port_profile"] == 2
    assert manifest["gpu_index"] == 2
    assert manifest["ctx_aware_usage_threshold_tokens"] == 600000
    assert manifest["ctx_aware_scheduling_threshold_tokens"] == 500000
    assert manifest["freq_controller_aggressive_slo_control"] is True
    assert manifest["freq_controller_threshold"] == 200.0
    assert manifest["total_slo_rounds"] == 1
    assert manifest["submit_command_default"].endswith("submit_embedded_tp1.sh")
    assert manifest["selected_source_plan"].endswith(
        "replay-plan.clean.token.exclude-unranked.fp8.json"
    )
    assert manifest["selected_plan_copy"] == str(expected_plan_copy)
    assert manifest["replay_output_root_dir"].endswith(
        "results/replay/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/"
        "sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo-aggressive-lmcache"
    )

    run_script_text = (batch_dir / "run_replay.sh").read_text(encoding="utf-8")
    assert "DEFAULT_PORT_PROFILE_ID=2" in run_script_text
    assert "DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS=600000" in run_script_text
    assert "DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS=500000" in run_script_text
    assert "DEFAULT_FREQ_CONTROLLER_THRESHOLD=200.0" in run_script_text
    assert '"--aggresive"' in run_script_text


def test_main_reuses_qps_slots_across_multiple_slo_rounds(tmp_path: Path) -> None:
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
            "0.1,0.2,0.3,0.4,0.5,0.6",
            "--target-slo-list",
            "8,12",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--split",
            "rest",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    batch_dir = resolve_single_output_batch_dir(output_config_dir)
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["max_qps_points"] == 8
    assert manifest["assigned_qps_points"] == 6
    assert manifest["total_slo_rounds"] == 2
    assert manifest["total_qps_slo_runs"] == 12
    assert len(manifest["qps_points"]) == 6
    assert [point["assigned_port_profile"] for point in manifest["qps_points"]] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
    assert all(len(point["slo_points"]) == 2 for point in manifest["qps_points"])


def test_main_rejects_more_than_eight_qps_points(tmp_path: Path) -> None:
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
            "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
            "--target-slo-list",
            "8",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--split",
            "rest",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )

    assert exit_code == 1


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
            "--split",
            "rest",
            "--ctx-aware-usage-threshold-tokens",
            "5000",
            "--ctx-aware-scheduling-threshold-tokens",
            "5000",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )

    assert exit_code == 1


def test_main_rejects_mismatched_gpu_index(tmp_path: Path) -> None:
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
            "2",
            "--gpu-index",
            "3",
            "--split",
            "rest",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )

    assert exit_code == 1


def test_main_rejects_empty_output_suffix(tmp_path: Path) -> None:
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
            "--split",
            "rest",
            "--output-suffix",
            " ",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )

    assert exit_code == 1
