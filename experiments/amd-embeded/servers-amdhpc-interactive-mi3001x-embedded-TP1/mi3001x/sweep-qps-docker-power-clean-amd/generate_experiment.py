#!/usr/bin/env python3
"""Generate one single-node mi3001x interactive AMD sweep-QPS power bundle."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import shlex
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
RESULTS_ROOT = REPO_ROOT / "results"
EXPERIMENT_DIR_NAME = "sweep-qps-docker-power-clean-amd"
MI3001X_SUBDIR = "mi3001x"
DEFAULT_MI3001X_PORT_PROFILE_ID = 0
DEFAULT_MI3001X_GPU_INDEX = 0
DEFAULT_OUTPUT_CONFIG_DIR = (
    REPO_ROOT
    / "experiments"
    / "amd-embeded"
    / "servers-amdhpc-interactive-mi3001x-embedded-TP1"
    / MI3001X_SUBDIR
    / EXPERIMENT_DIR_NAME
    / "generated"
)
DEFAULT_REPLAY_OUTPUT_ROOT = (
    Path("results")
    / "replay"
    / "amd-embeded"
    / "servers-amdhpc-interactive-mi3001x-embedded-TP1"
    / MI3001X_SUBDIR
    / EXPERIMENT_DIR_NAME
)
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model_config.toml"
EXPERIMENT_PATH = (
    "experiments/amd-embeded/servers-amdhpc-interactive-mi3001x-embedded-TP1/"
    f"{MI3001X_SUBDIR}/"
    f"{EXPERIMENT_DIR_NAME}"
)
EXPERIMENT_LOG_TAG = (
    "amd-embeded-servers-amdhpc-interactive-mi3001x-embedded-TP1-mi3001x-"
    "sweep-qps-docker-power-clean-amd"
)
PROFILE_OUTPUT_PLACEHOLDER = "profile-<port_profile_id>"
INTERACTIVE_SETUP_README = "servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/README.md"
INTERACTIVE_CLIENT_SCRIPT = (
    "servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/client.py"
)
INTERACTIVE_ENV_HELPER = (
    "servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/experiment-env.sh"
)
INTERACTIVE_START_SERVICES_SCRIPT = (
    "servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/start-services.sh"
)
INTERACTIVE_START_SERVICES_COMMAND = (
    f"python3 {INTERACTIVE_CLIENT_SCRIPT} start"
)
INTERACTIVE_STOP_SERVICES_COMMAND = (
    f"python3 {INTERACTIVE_CLIENT_SCRIPT} stop"
)
INTERACTIVE_FOREGROUND_START_COMMAND = (
    f"bash {INTERACTIVE_START_SERVICES_SCRIPT}"
)
AMD_SMI_POWER_SOCKET_EXAMPLE = "/tmp/amdsmi-power-reader.sock"
_SOURCE_GENERATOR_MODULE_NAME = (
    "generate_amd_interactive_mi3001x_sweep_qps_power_clean_amd_source"
)


def _load_source_generator_module() -> object:
    module_path = (
        REPO_ROOT
        / "experiments"
        / "amd-embeded"
        / "servers-amdhpc-embedded-TP1"
        / MI3001X_SUBDIR
        / EXPERIMENT_DIR_NAME
        / "generate_experiment.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        _SOURCE_GENERATOR_MODULE_NAME,
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load source generator module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SOURCE_GENERATOR_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source_generator_module()
BASE = SOURCE.BASE


def _sync_base_module_globals() -> None:
    BASE.REPO_ROOT = REPO_ROOT
    BASE.RESULTS_ROOT = RESULTS_ROOT
    BASE.MODEL_CONFIG_PATH = MODEL_CONFIG_PATH


def build_generation_command_raw(argv: list[str] | None) -> str:
    if argv is None:
        original = getattr(sys, "orig_argv", None)
        if isinstance(original, list) and original:
            return shlex.join([str(token) for token in original])
        return shlex.join([str(sys.executable), *[str(token) for token in sys.argv]])
    return shlex.join(
        [str(sys.executable), str(Path(__file__).resolve()), *[str(token) for token in argv]]
    )


def path_for_config(path: Path) -> str:
    _sync_base_module_globals()
    return BASE.path_for_config(path)


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def write_run_script(
    path: Path,
    *,
    default_port_profile: int,
    target_model: str,
    split: str,
    default_gpu_index: int,
    qps_jobs: list[Any],
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        f"REPO_ROOT={_shell_quote(str(REPO_ROOT.resolve()))}",
        f"DEFAULT_PORT_PROFILE_ID={default_port_profile}",
        f"DEFAULT_GPU_INDEX={default_gpu_index}",
        "PORT_PROFILE_ID_VALUE=\"${1:-${PORT_PROFILE_ID:-${DEFAULT_PORT_PROFILE_ID}}}\"",
        "GPU_INDEX_VALUE=\"${GPU_INDEX:-${DEFAULT_GPU_INDEX}}\"",
        "PYTHON_BIN=\"${PYTHON_BIN:-python3}\"",
        "AMD_POWER_READER_BIN=\"${AMD_POWER_READER_BIN:-amd-power-reader}\"",
        "AMD_SMI_POWER_SOCKET_PATH_VALUE=\"${AMD_SMI_POWER_SOCKET_PATH:-/tmp/amdsmi-power-reader.sock}\"",
        (
            "INTERACTIVE_ENV_HELPER="
            "\"${INTERACTIVE_ENV_HELPER:-"
            "servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/experiment-env.sh}\""
        ),
        "INTERACTIVE_ENV_HELPER_PATH=\"\"",
        "",
        "POWER_READER_PID=\"\"",
        "",
        "stop_power_reader() {",
        "  if [[ -n \"${POWER_READER_PID}\" ]]; then",
        "    kill \"${POWER_READER_PID}\" >/dev/null 2>&1 || true",
        "    wait \"${POWER_READER_PID}\" 2>/dev/null || true",
        "    POWER_READER_PID=\"\"",
        "  fi",
        "}",
        "",
        "cleanup() {",
        "  local exit_code=\"$1\"",
        "  stop_power_reader",
        "  return \"${exit_code}\"",
        "}",
        "",
        "trap '__exit_code=$?; trap - EXIT INT TERM; cleanup \"${__exit_code}\"; exit \"${__exit_code}\"' EXIT",
        f"trap 'echo \"[{EXPERIMENT_LOG_TAG}] interrupted\" >&2; exit 130' INT TERM",
        "",
        "resolve_interactive_env_helper() {",
        "  if [[ -n \"${INTERACTIVE_ENV_HELPER_PATH}\" ]]; then",
        "    return 0",
        "  fi",
        "  if [[ \"${INTERACTIVE_ENV_HELPER}\" = /* ]]; then",
        "    INTERACTIVE_ENV_HELPER_PATH=\"${INTERACTIVE_ENV_HELPER}\"",
        "  else",
        "    INTERACTIVE_ENV_HELPER_PATH=\"${REPO_ROOT}/${INTERACTIVE_ENV_HELPER}\"",
        "  fi",
        "  if [[ ! -f \"${INTERACTIVE_ENV_HELPER_PATH}\" ]]; then",
        (
            f"    echo \"[{EXPERIMENT_LOG_TAG}] error: missing interactive env "
            "helper: ${INTERACTIVE_ENV_HELPER_PATH}\" >&2"
        ),
        "    exit 1",
        "  fi",
        "}",
        "",
        "resolve_interactive_env_helper",
        "source \"${INTERACTIVE_ENV_HELPER_PATH}\"",
        (
            "PORT_PROFILE_ID_VALUE=\"$("
            "interactive_embedded_tp1_normalize_port_profile_id "
            "\"${PORT_PROFILE_ID_VALUE}\" \"${DEFAULT_PORT_PROFILE_ID}\" "
            f"\"{EXPERIMENT_LOG_TAG}\""
            ")\""
        ),
        (
            "GPU_INDEX_VALUE=\"$("
            "interactive_embedded_tp1_normalize_gpu_index "
            "\"${GPU_INDEX_VALUE}\" \"${DEFAULT_GPU_INDEX}\" "
            f"\"{EXPERIMENT_LOG_TAG}\""
            ")\""
        ),
        (
            "interactive_embedded_tp1_wait_for_services "
            "\"${PORT_PROFILE_ID_VALUE}\" "
            f"\"{EXPERIMENT_LOG_TAG}\""
        ),
        "export AMD_SMI_POWER_SOCKET_PATH=\"${AMD_SMI_POWER_SOCKET_PATH_VALUE}\"",
        "",
        "run_one_qps() {",
        "  local qps_value=\"$1\"",
        "  local qps_slug=\"$2\"",
        "  local replay_config_ref=\"$3\"",
        "  local replay_output_ref=\"$4\"",
        "  local replay_config_path=\"\"",
        "  local replay_output_base_dir=\"\"",
        "  local replay_output_dir=\"\"",
        "",
        "  if [[ \"${replay_config_ref}\" = /* ]]; then",
        "    replay_config_path=\"${replay_config_ref}\"",
        "  else",
        "    replay_config_path=\"${SCRIPT_DIR}/${replay_config_ref}\"",
        "  fi",
        "",
        "  if [[ \"${replay_output_ref}\" = /* ]]; then",
        "    replay_output_base_dir=\"${replay_output_ref}\"",
        "  else",
        "    replay_output_base_dir=\"${REPO_ROOT}/${replay_output_ref}\"",
        "  fi",
        "  replay_output_dir=\"${replay_output_base_dir}/profile-${PORT_PROFILE_ID_VALUE}\"",
        "  local power_output_dir=\"${replay_output_dir}/power\"",
        "",
        (
            f"  echo \"[{EXPERIMENT_LOG_TAG}] qps=${{qps_value}} slug=${{qps_slug}} "
            "output=${replay_output_dir} "
            "port_profile=${PORT_PROFILE_ID_VALUE} "
            "gpu_index=${GPU_INDEX_VALUE}\""
        ),
        "  mkdir -p \"${power_output_dir}\"",
        "  \"${AMD_POWER_READER_BIN}\" --output-dir \"${power_output_dir}\" --gpu-indices \"${GPU_INDEX_VALUE}\" --socket-path \"${AMD_SMI_POWER_SOCKET_PATH_VALUE}\" &",
        "  POWER_READER_PID=\"$!\"",
        "",
        "  \"${PYTHON_BIN}\" -m replayer replay \\",
        "    --config \"${replay_config_path}\" \\",
        "    --output-dir \"${replay_output_dir}\" \\",
        "    --port-profile-id \"${PORT_PROFILE_ID_VALUE}\"",
        "",
        "  stop_power_reader",
        "}",
        "",
        (
            f"echo \"[{EXPERIMENT_LOG_TAG}] target_model={target_model} split={split} "
            f"qps_points={len(qps_jobs)} gpu_index=${{GPU_INDEX_VALUE}} "
            "port_profile=${PORT_PROFILE_ID_VALUE}\""
        ),
        "",
    ]

    for job in qps_jobs:
        lines.append(
            "run_one_qps "
            f"{_shell_quote(job.qps_display)} "
            f"{_shell_quote(job.qps_slug)} "
            f"{_shell_quote(job.replay_config_relpath)} "
            f"{_shell_quote(path_for_config(job.replay_output_dir))}"
        )
    lines.extend(
        [
            "",
            f"echo \"[{EXPERIMENT_LOG_TAG}] completed {len(qps_jobs)} qps points\"",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o750)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_experiment.py",
        description=(
            "Generate a single-node mi3001x interactive sweep-QPS replay bundle "
            "with amd-power-reader power logging. The generated run_replay.sh "
            "sources the shared interactive experiment helper from "
            f"{INTERACTIVE_ENV_HELPER} and expects the service stack to already "
            f"be running via {INTERACTIVE_START_SERVICES_COMMAND}."
        ),
    )
    parser.add_argument(
        "--source-run-dir",
        required=True,
        help="Profiled source run directory under results/.",
    )
    parser.add_argument(
        "--poisson-seed",
        required=True,
        type=BASE.parse_non_negative_int,
        help="Poisson launch seed.",
    )
    parser.add_argument(
        "--randomize-seed",
        required=True,
        type=BASE.parse_non_negative_int,
        help="Replay randomization seed.",
    )
    parser.add_argument(
        "--qps-list",
        required=True,
        help="Comma-separated Poisson rates, e.g. 0.1,0.2,0.3",
    )
    parser.add_argument(
        "--time-constraint-s",
        required=True,
        type=float,
        help="Replay time limit in seconds.",
    )
    parser.add_argument(
        "--target-model",
        required=True,
        help="Target model key from configs/model_config.toml.",
    )
    parser.add_argument(
        "--port-profile",
        "-P",
        default=DEFAULT_MI3001X_PORT_PROFILE_ID,
        type=int,
        help=(
            "mi3001x interactive mode uses port profile 0. "
            "This option is accepted only for compatibility and must remain 0."
        ),
    )
    parser.add_argument(
        "--gpu-index",
        default=str(DEFAULT_MI3001X_GPU_INDEX),
        help=(
            "mi3001x interactive mode uses GPU index 0. "
            "This option is accepted only for compatibility and must remain 0."
        ),
    )
    parser.add_argument(
        "--split",
        required=True,
        help=(
            "Clean split mode: exclude-unranked | top | rest. "
            "Compatibility alias: full -> exclude-unranked."
        ),
    )
    parser.add_argument(
        "--split-two-group-metric",
        choices=sorted(BASE.SPLIT_METRIC_ALIASES),
        default="token_usage",
        help="Split grouping metric used for top/rest plan lookup (default: token_usage).",
    )
    parser.add_argument(
        "--output-config-dir",
        default=str(DEFAULT_OUTPUT_CONFIG_DIR),
        help=f"Generated bundle root (default: {DEFAULT_OUTPUT_CONFIG_DIR}).",
    )
    parser.add_argument(
        "--additional-suffix",
        default=None,
        help=(
            "Optional replay plan suffix for lookup. Example: with suffix fp8 and split "
            "rest, looks for replay-plan.clean.token.rest.fp8.json."
        ),
    )
    parser.add_argument(
        "--replay-output-root",
        default=str(DEFAULT_REPLAY_OUTPUT_ROOT),
        help=(
            "Replay output root. Default appends "
            "<dataset-lineage>/split/<split>/<qps>/<timestamp>/profile-<port_profile_id> "
            "under results/replay/amd-embeded/"
            "servers-amdhpc-interactive-mi3001x-embedded-TP1/mi3001x/"
            "sweep-qps-docker-power-clean-amd/. dataset-lineage is inferred from "
            "--source-run-dir by dropping the first (<model>) and last (<run-dir>) "
            "path segments."
        ),
    )
    return parser


def _skip_submit_script(
    path: Path,
    *,
    run_script_path: Path,
    target_model: str,
) -> None:
    del path, run_script_path, target_model


def _interactive_start_command(*, run_script_path: Path, target_model: str) -> str:
    del run_script_path, target_model
    return shlex.join(["python3", INTERACTIVE_CLIENT_SCRIPT, "start"])


def _configure_source_module() -> None:
    SOURCE.REPO_ROOT = REPO_ROOT
    SOURCE.RESULTS_ROOT = RESULTS_ROOT
    SOURCE.DEFAULT_OUTPUT_CONFIG_DIR = DEFAULT_OUTPUT_CONFIG_DIR
    SOURCE.DEFAULT_REPLAY_OUTPUT_ROOT = DEFAULT_REPLAY_OUTPUT_ROOT
    SOURCE.MODEL_CONFIG_PATH = MODEL_CONFIG_PATH
    SOURCE.EXPERIMENT_PATH = EXPERIMENT_PATH
    SOURCE.EXPERIMENT_LOG_TAG = EXPERIMENT_LOG_TAG
    SOURCE.PROFILE_OUTPUT_PLACEHOLDER = PROFILE_OUTPUT_PLACEHOLDER
    SOURCE._sync_base_module_globals = _sync_base_module_globals
    SOURCE.build_generation_command_raw = build_generation_command_raw
    SOURCE.build_parser = build_parser
    SOURCE.write_run_script = write_run_script
    SOURCE.write_submit_script = _skip_submit_script
    SOURCE.build_embedded_tp1_submit_command = _interactive_start_command


def _postprocess_manifest(manifest_path: Path) -> dict[str, Any]:
    summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    submit_script = summary.pop("submit_script", None)
    summary.pop("submit_command_default", None)
    summary.pop("embedded_tp1_submit_command", None)
    if isinstance(submit_script, str) and submit_script:
        submit_path = Path(submit_script)
        if submit_path.exists():
            submit_path.unlink()

    summary["service_mode"] = "interactive"
    summary["assumes_external_service_stack"] = True
    summary["service_preflight"] = True
    summary["interactive_setup_readme"] = INTERACTIVE_SETUP_README
    summary["interactive_client_script"] = INTERACTIVE_CLIENT_SCRIPT
    summary["interactive_env_helper_script"] = INTERACTIVE_ENV_HELPER
    summary["interactive_launcher_script"] = INTERACTIVE_START_SERVICES_SCRIPT
    summary["interactive_start_services_command"] = INTERACTIVE_START_SERVICES_COMMAND
    summary["interactive_stop_services_command"] = INTERACTIVE_STOP_SERVICES_COMMAND
    summary["interactive_foreground_start_services_command"] = (
        INTERACTIVE_FOREGROUND_START_COMMAND
    )
    summary["interactive_power_daemon_enabled"] = True
    summary["interactive_power_daemon_managed_by_launcher"] = True
    summary["interactive_power_socket_env_example"] = (
        f"export AMD_SMI_POWER_SOCKET_PATH={AMD_SMI_POWER_SOCKET_EXAMPLE}"
    )
    summary["interactive_execution_note"] = (
        f"Start the background service stack with "
        f"`{INTERACTIVE_START_SERVICES_COMMAND}` on the interactive node and wait "
        "for readiness. The generated `run_replay.sh` then sources the shared "
        f"helper at `{INTERACTIVE_ENV_HELPER}`, waits for the services if "
        "needed, defaults `AMD_SMI_POWER_SOCKET_PATH` to "
        "`/tmp/amdsmi-power-reader.sock` unless you override it in the "
        "environment, and runs the replay. When you are done, stop the "
        f"background stack with `{INTERACTIVE_STOP_SERVICES_COMMAND}`."
    )

    manifest_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _emit_failure_output(stdout_text: str, stderr_text: str) -> None:
    if stderr_text:
        print(stderr_text, end="", file=sys.stderr)
    elif stdout_text:
        print(stdout_text, end="", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    _configure_source_module()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
        ):
            exit_code = SOURCE.main(argv)
    except SystemExit as exc:
        _emit_failure_output(stdout_buffer.getvalue(), stderr_buffer.getvalue())
        code = exc.code if isinstance(exc.code, int) else 1
        return int(code)

    if exit_code != 0:
        _emit_failure_output(stdout_buffer.getvalue(), stderr_buffer.getvalue())
        return int(exit_code)

    try:
        source_summary = json.loads(stdout_buffer.getvalue())
    except json.JSONDecodeError:
        _emit_failure_output(stdout_buffer.getvalue(), stderr_buffer.getvalue())
        return 1

    manifest_path = Path(source_summary["manifest_path"]).resolve()
    summary = _postprocess_manifest(manifest_path)
    summary["manifest_path"] = str(manifest_path)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
