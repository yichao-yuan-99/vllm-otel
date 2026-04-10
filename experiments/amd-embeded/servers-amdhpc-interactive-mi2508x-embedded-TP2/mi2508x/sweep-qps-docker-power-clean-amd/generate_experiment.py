#!/usr/bin/env python3
"""Generate one interactive mi2508x TP2 AMD sweep-QPS power bundle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
RESULTS_ROOT = REPO_ROOT / "results"
EXPERIMENT_DIR_NAME = "sweep-qps-docker-power-clean-amd"
MI2508X_SUBDIR = "mi2508x"
MI2508X_MIN_PORT_PROFILE_ID = 0
MI2508X_MAX_PORT_PROFILE_ID = 3
MAX_QPS_POINTS = MI2508X_MAX_PORT_PROFILE_ID - MI2508X_MIN_PORT_PROFILE_ID + 1
DEFAULT_PORT_PROFILE_ID = MI2508X_MIN_PORT_PROFILE_ID
GPU_INDEX_RUNTIME_DEFAULT = "match-port-profile-gpu-pair"
GPU_PAIR_BY_PORT_PROFILE = {
    0: "0,1",
    1: "2,3",
    2: "4,5",
    3: "6,7",
}
DEFAULT_OUTPUT_CONFIG_DIR = (
    REPO_ROOT
    / "experiments"
    / "amd-embeded"
    / "servers-amdhpc-interactive-mi2508x-embedded-TP2"
    / MI2508X_SUBDIR
    / EXPERIMENT_DIR_NAME
    / "generated"
)
DEFAULT_REPLAY_OUTPUT_ROOT = (
    Path("results")
    / "replay"
    / "amd-embeded"
    / "servers-amdhpc-interactive-mi2508x-embedded-TP2"
    / MI2508X_SUBDIR
    / EXPERIMENT_DIR_NAME
)
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model_config.toml"
EXPERIMENT_PATH = (
    "experiments/amd-embeded/servers-amdhpc-interactive-mi2508x-embedded-TP2/"
    f"{MI2508X_SUBDIR}/"
    f"{EXPERIMENT_DIR_NAME}"
)
EXPERIMENT_LOG_TAG = (
    "amd-embeded-servers-amdhpc-interactive-mi2508x-embedded-TP2-mi2508x-"
    "sweep-qps-docker-power-clean-amd"
)
PROFILE_OUTPUT_PLACEHOLDER = "profile-<port_profile_id>"
INTERACTIVE_SETUP_README = "servers/servers-amdhpc-interactive-mi2508x-embedded-TP2/README.md"
INTERACTIVE_CLIENT_SCRIPT = (
    "servers/servers-amdhpc-interactive-mi2508x-embedded-TP2/client.py"
)
INTERACTIVE_ENV_HELPER = (
    "servers/servers-amdhpc-interactive-mi2508x-embedded-TP2/experiment-env.sh"
)
INTERACTIVE_START_SERVICES_SCRIPT = (
    "servers/servers-amdhpc-interactive-mi2508x-embedded-TP2/start-services.sh"
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
_BASE_GENERATOR_MODULE_NAME = (
    "generate_sweep_qps_docker_power_clean_experiment_for_interactive_mi2508x_tp2_amd"
)


@dataclass(frozen=True)
class QpsReplayJob:
    qps: float
    qps_display: str
    qps_slug: str
    assigned_port_profile: int
    assigned_gpu_index: str
    replay_config_relpath: str
    replay_config_path: Path
    replay_output_dir: Path
    power_output_dir: Path


def _load_base_generator_module() -> object:
    module_path = (
        REPO_ROOT
        / "experiments"
        / "sweep-qps-docker-power-clean"
        / "generate_experiment.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        _BASE_GENERATOR_MODULE_NAME,
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base generator module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_BASE_GENERATOR_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_generator_module()


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


def validate_mi2508x_port_profile(value: int, *, field_name: str) -> int:
    parsed = int(value)
    if parsed < MI2508X_MIN_PORT_PROFILE_ID or parsed > MI2508X_MAX_PORT_PROFILE_ID:
        raise ValueError(
            f"{field_name} must be between "
            f"{MI2508X_MIN_PORT_PROFILE_ID} and {MI2508X_MAX_PORT_PROFILE_ID}"
        )
    return parsed


def expected_gpu_pair_for_port_profile(port_profile: int) -> str:
    try:
        return GPU_PAIR_BY_PORT_PROFILE[int(port_profile)]
    except KeyError as exc:
        raise ValueError(f"unsupported mi2508x port profile id: {port_profile}") from exc


def parse_optional_gpu_index(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    gpu_indices = BASE.parse_gpu_index_list(text)
    normalized = ",".join(str(index) for index in gpu_indices)
    if normalized not in GPU_PAIR_BY_PORT_PROFILE.values():
        raise ValueError(
            "--gpu-index must be one of 0,1 2,3 4,5 6,7 for mi2508x interactive TP2"
        )
    return normalized


def ensure_supported_qps_point_count(qps_values: list[float]) -> None:
    if len(qps_values) > MAX_QPS_POINTS:
        raise ValueError(
            f"--qps-list supports at most {MAX_QPS_POINTS} QPS values for mi2508x interactive TP2"
        )


def build_assigned_profile_output_dir(path: Path, *, assigned_port_profile: int) -> str:
    return f"{path_for_config(path)}/profile-{assigned_port_profile}"


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def write_replay_config(path: Path, *, replay_payload: dict[str, Any]) -> None:
    _sync_base_module_globals()
    lines: list[str] = [
        f"# Auto-generated by {EXPERIMENT_PATH}/generate_experiment.py",
        "",
    ]
    BASE._append_toml_table(lines, "replay", replay_payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_script(
    path: Path,
    *,
    default_port_profile: int,
    target_model: str,
    split: str,
    qps_jobs: list[QpsReplayJob],
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        f"REPO_ROOT={_shell_quote(str(REPO_ROOT.resolve()))}",
        f"DEFAULT_PORT_PROFILE_ID={default_port_profile}",
        "PORT_PROFILE_ID_VALUE=\"${1:-${PORT_PROFILE_ID:-${DEFAULT_PORT_PROFILE_ID}}}\"",
        "GPU_INDEX_VALUE=\"${GPU_INDEX:-}\"",
        "PYTHON_BIN=\"${PYTHON_BIN:-python3}\"",
        "AMD_POWER_READER_BIN=\"${AMD_POWER_READER_BIN:-amd-power-reader}\"",
        "AMD_SMI_POWER_SOCKET_PATH_VALUE=\"${AMD_SMI_POWER_SOCKET_PATH:-/tmp/amdsmi-power-reader.sock}\"",
        (
            "INTERACTIVE_ENV_HELPER="
            "\"${INTERACTIVE_ENV_HELPER:-"
            "servers/servers-amdhpc-interactive-mi2508x-embedded-TP2/experiment-env.sh}\""
        ),
        "INTERACTIVE_ENV_HELPER_PATH=\"\"",
        f"ASSIGNED_QPS_POINTS={len(qps_jobs)}",
        "SELECTED_QPS_VALUE=\"\"",
        "SELECTED_QPS_SLUG=\"\"",
        "SELECTED_REPLAY_CONFIG_REF=\"\"",
        "SELECTED_REPLAY_OUTPUT_REF=\"\"",
        "SELECTED_ASSIGNED_GPU_INDEX=\"\"",
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
        "run_one_qps() {",
        "  local qps_value=\"$1\"",
        "  local qps_slug=\"$2\"",
        "  local replay_config_ref=\"$3\"",
        "  local replay_output_ref=\"$4\"",
        "  local replay_config_path=\"\"",
        "  local replay_output_base_dir=\"\"",
        "  local replay_output_dir=\"\"",
        "  local -a GPU_INDEX_ARGS=()",
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
        "  IFS=',' read -r -a GPU_INDEX_ARGS <<<\"${GPU_INDEX_VALUE}\"",
        "  \"${AMD_POWER_READER_BIN}\" --output-dir \"${power_output_dir}\" --gpu-indices \"${GPU_INDEX_ARGS[@]}\" --socket-path \"${AMD_SMI_POWER_SOCKET_PATH_VALUE}\" &",
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
        "select_qps_job() {",
        "  case \"${PORT_PROFILE_ID_VALUE}\" in",
    ]

    for job in qps_jobs:
        lines.extend(
            [
                f"    {job.assigned_port_profile})",
                f"      SELECTED_QPS_VALUE={_shell_quote(job.qps_display)}",
                f"      SELECTED_QPS_SLUG={_shell_quote(job.qps_slug)}",
                f"      SELECTED_REPLAY_CONFIG_REF={_shell_quote(job.replay_config_relpath)}",
                f"      SELECTED_REPLAY_OUTPUT_REF={_shell_quote(path_for_config(job.replay_output_dir))}",
                f"      SELECTED_ASSIGNED_GPU_INDEX={_shell_quote(job.assigned_gpu_index)}",
                "      ;;",
            ]
        )

    lines.extend(
        [
            "    *)",
            (
                f"      echo \"[{EXPERIMENT_LOG_TAG}] no qps assigned for "
                "port_profile=${PORT_PROFILE_ID_VALUE} "
                "assigned_qps_points=${ASSIGNED_QPS_POINTS}; skipping\""
            ),
            "      exit 0",
            "      ;;",
            "  esac",
            "}",
            "",
            "resolve_interactive_env_helper",
            "source \"${INTERACTIVE_ENV_HELPER_PATH}\"",
            (
                "PORT_PROFILE_ID_VALUE=\"$("
                "interactive_embedded_tp2_normalize_port_profile_id "
                "\"${PORT_PROFILE_ID_VALUE}\" \"\" "
                f"\"{EXPERIMENT_LOG_TAG}\""
                ")\""
            ),
            "select_qps_job",
            "if [[ -z \"${GPU_INDEX_VALUE}\" ]]; then",
            "  GPU_INDEX_VALUE=\"${SELECTED_ASSIGNED_GPU_INDEX}\"",
            "fi",
            (
                "GPU_INDEX_VALUE=\"$("
                "interactive_embedded_tp2_normalize_gpu_index "
                "\"${GPU_INDEX_VALUE}\" \"${SELECTED_ASSIGNED_GPU_INDEX}\" "
                f"\"{EXPERIMENT_LOG_TAG}\""
                ")\""
            ),
            (
                "interactive_embedded_tp2_wait_for_services "
                "\"${PORT_PROFILE_ID_VALUE}\" "
                f"\"{EXPERIMENT_LOG_TAG}\""
            ),
            "export AMD_SMI_POWER_SOCKET_PATH=\"${AMD_SMI_POWER_SOCKET_PATH_VALUE}\"",
            "",
            (
                f"echo \"[{EXPERIMENT_LOG_TAG}] target_model={target_model} split={split} "
                f"assigned_qps_points={len(qps_jobs)} selected_qps=${{SELECTED_QPS_VALUE}} "
                "gpu_index=${GPU_INDEX_VALUE} port_profile=${PORT_PROFILE_ID_VALUE}\""
            ),
            "",
            "run_one_qps "
            "\"${SELECTED_QPS_VALUE}\" "
            "\"${SELECTED_QPS_SLUG}\" "
            "\"${SELECTED_REPLAY_CONFIG_REF}\" "
            "\"${SELECTED_REPLAY_OUTPUT_REF}\"",
            "",
            (
                f"echo \"[{EXPERIMENT_LOG_TAG}] completed qps=${{SELECTED_QPS_VALUE}} "
                "port_profile=${PORT_PROFILE_ID_VALUE}\""
            ),
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
            "Generate a dedicated mi2508x interactive TP2 sweep-QPS replay bundle "
            "with amd-power-reader power logging. QPS points are assigned to "
            "port profiles 0..3 in ascending order."
        ),
    )
    parser.add_argument("--source-run-dir", required=True, help="Profiled source run directory under results/.")
    parser.add_argument("--poisson-seed", required=True, type=BASE.parse_non_negative_int, help="Poisson launch seed.")
    parser.add_argument("--randomize-seed", required=True, type=BASE.parse_non_negative_int, help="Replay randomization seed.")
    parser.add_argument(
        "--qps-list",
        required=True,
        help=(
            "Comma-separated Poisson rates, e.g. 0.1,0.2,0.3. "
            f"Supports at most {MAX_QPS_POINTS} values."
        ),
    )
    parser.add_argument("--time-constraint-s", required=True, type=float, help="Replay time limit in seconds.")
    parser.add_argument("--target-model", required=True, help="Target model key from configs/model_config.toml.")
    parser.add_argument(
        "--port-profile",
        "-P",
        default=DEFAULT_PORT_PROFILE_ID,
        type=int,
        help=(
            "Default port profile used when the generated run script is started "
            "without an explicit profile id. Runtime still accepts profile ids 0..3."
        ),
    )
    parser.add_argument(
        "--gpu-index",
        default=None,
        help=(
            "Optional compatibility check for manual default runs. If provided, "
            "it must match the GPU pair for --port-profile. Runtime otherwise "
            "defaults to the assigned pair for the selected port profile."
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
            "Optional replay plan suffix for lookup. Example: with suffix fp8 and split rest, "
            "looks for replay-plan.clean.token.rest.fp8.json."
        ),
    )
    parser.add_argument(
        "--replay-output-root",
        default=str(DEFAULT_REPLAY_OUTPUT_ROOT),
        help=(
            "Replay output root. Default appends "
            "<dataset-lineage>/split/<split>/<qps>/<timestamp>/profile-<assigned-port-profile> "
            "under results/replay/amd-embeded/servers-amdhpc-interactive-mi2508x-embedded-TP2/"
            "mi2508x/sweep-qps-docker-power-clean-amd/. "
            "dataset-lineage is inferred from --source-run-dir by dropping "
            "the first (<model>) and last (<run-dir>) path segments."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    generation_command_raw = build_generation_command_raw(argv)

    try:
        _sync_base_module_globals()
        source_run_dir = Path(args.source_run_dir).expanduser().resolve()
        if not source_run_dir.exists() or not source_run_dir.is_dir():
            raise ValueError(f"invalid --source-run-dir: {source_run_dir}")
        try:
            source_run_dir.relative_to(RESULTS_ROOT.resolve())
        except ValueError:
            raise ValueError("--source-run-dir must live under top-level results/")

        split = BASE.canonical_split(args.split)
        qps_values = BASE.parse_qps_list(str(args.qps_list))
        ensure_supported_qps_point_count(qps_values)
        time_constraint_s = BASE.parse_positive_float(
            str(args.time_constraint_s),
            field_name="--time-constraint-s",
        )
        default_port_profile = validate_mi2508x_port_profile(
            int(args.port_profile),
            field_name="--port-profile",
        )
        gpu_index = parse_optional_gpu_index(args.gpu_index)
        if gpu_index is not None and gpu_index != expected_gpu_pair_for_port_profile(
            default_port_profile
        ):
            raise ValueError(
                "--gpu-index must match the GPU pair for --port-profile when provided "
                "for mi2508x interactive TP2"
            )

        target_model = str(args.target_model).strip()
        if not target_model:
            raise ValueError("--target-model cannot be empty")
        target_model_specs = BASE._load_target_model_specs()
        target_model_spec = target_model_specs.get(target_model)
        if target_model_spec is None:
            available = ", ".join(sorted(target_model_specs.keys()))
            raise ValueError(
                f"unknown --target-model {target_model!r}; available keys: {available}"
            )
        additional_suffix = (
            str(args.additional_suffix).strip()
            if args.additional_suffix is not None and str(args.additional_suffix).strip()
            else None
        )

        output_config_root = Path(args.output_config_dir).expanduser().resolve()
        replay_output_root = Path(args.replay_output_root).expanduser()
        if not replay_output_root.is_absolute():
            replay_output_root = (REPO_ROOT / replay_output_root).resolve()
        else:
            replay_output_root = replay_output_root.resolve()
        source_dataset_lineage = BASE.derive_dataset_lineage_from_source_run_dir(source_run_dir)

        batch_timestamp = BASE.build_utc_timestamp_slug()
        batch_dir = (output_config_root / batch_timestamp).resolve()

        lookup_result = BASE._resolve_lookup_plan_paths(
            source_run_dir=source_run_dir,
            split=split,
            split_two_group_metric=str(args.split_two_group_metric),
            additional_suffix=additional_suffix,
        )
        related_plan_models: dict[str, str] = {}
        for plan_key, plan_path_text in lookup_result.related_plan_paths.items():
            plan_model = BASE._validate_plan_model_matches_target(
                plan_path=Path(plan_path_text),
                target_model_spec=target_model_spec,
            )
            related_plan_models[plan_key] = plan_model

        plan_copy_dir = (batch_dir / "plan").resolve()
        plan_copy_dir.mkdir(parents=True, exist_ok=True)
        selected_plan_copy_path = (plan_copy_dir / lookup_result.selected_plan_path.name).resolve()
        shutil.copy2(lookup_result.selected_plan_path, selected_plan_copy_path)

        qps_jobs: list[QpsReplayJob] = []
        for assigned_port_profile, qps in enumerate(qps_values):
            assigned_gpu_index = expected_gpu_pair_for_port_profile(assigned_port_profile)
            qps_slug = BASE.format_qps_slug(qps)
            replay_output_dir = (
                replay_output_root
                / source_dataset_lineage
                / "split"
                / split
                / qps_slug
                / batch_timestamp
            ).resolve()
            replay_payload = {
                "plan": BASE.path_for_config(selected_plan_copy_path),
                "output_dir": BASE.path_for_config(replay_output_dir),
                "randomize_seed": int(args.randomize_seed),
                "time_constraint_s": time_constraint_s,
                "port_profile_id": assigned_port_profile,
                "launch_policy_override": {
                    "seed": int(args.poisson_seed),
                    "pattern": {"name": "poisson"},
                    "pattern_args": {"rate": qps},
                },
            }
            replay_config_path = (batch_dir / qps_slug / "replay.toml").resolve()
            write_replay_config(replay_config_path, replay_payload=replay_payload)
            replay_config_relpath = str(replay_config_path.relative_to(batch_dir))

            qps_jobs.append(
                QpsReplayJob(
                    qps=qps,
                    qps_display=BASE.qps_display_text(qps),
                    qps_slug=qps_slug,
                    assigned_port_profile=assigned_port_profile,
                    assigned_gpu_index=assigned_gpu_index,
                    replay_config_relpath=replay_config_relpath,
                    replay_config_path=replay_config_path,
                    replay_output_dir=replay_output_dir,
                    power_output_dir=(replay_output_dir / "power").resolve(),
                )
            )

        run_script_path = (batch_dir / "run_replay.sh").resolve()
        write_run_script(
            run_script_path,
            default_port_profile=default_port_profile,
            target_model=target_model,
            split=split,
            qps_jobs=qps_jobs,
        )

        qps_summary = [
            {
                "qps": job.qps,
                "qps_slug": job.qps_slug,
                "assigned_port_profile": job.assigned_port_profile,
                "assigned_gpu_index": job.assigned_gpu_index,
                "replay_config": str(job.replay_config_path),
                "replay_output_dir_base": BASE.path_for_config(job.replay_output_dir),
                "replay_output_dir": build_assigned_profile_output_dir(
                    job.replay_output_dir,
                    assigned_port_profile=job.assigned_port_profile,
                ),
                "power_output_dir": (
                    f"{build_assigned_profile_output_dir(job.replay_output_dir, assigned_port_profile=job.assigned_port_profile)}/power"
                ),
            }
            for job in qps_jobs
        ]
        summary = {
            "status": "ok",
            "batch_timestamp": batch_timestamp,
            "source_run_dir": str(source_run_dir),
            "source_dataset_lineage": str(source_dataset_lineage),
            "target_model": target_model,
            "split": split,
            "split_two_group_metric": args.split_two_group_metric,
            "launch_pattern": "poisson",
            "poisson_seed": int(args.poisson_seed),
            "randomize_seed": int(args.randomize_seed),
            "qps_list": qps_values,
            "max_qps_points": MAX_QPS_POINTS,
            "assigned_qps_points": len(qps_jobs),
            "time_constraint_s": time_constraint_s,
            "port_profile": default_port_profile,
            "default_port_profile": default_port_profile,
            "gpu_index": gpu_index,
            "gpu_index_runtime_default": GPU_INDEX_RUNTIME_DEFAULT,
            "profile_output_suffix": PROFILE_OUTPUT_PLACEHOLDER,
            "additional_suffix": additional_suffix,
            "output_config_root_dir": str(output_config_root),
            "output_batch_dir": str(batch_dir),
            "plan_lookup_only": True,
            "selected_source_plan": str(lookup_result.selected_plan_path),
            "related_source_plans": lookup_result.related_plan_paths,
            "selected_plan_model": related_plan_models.get("selected"),
            "related_plan_models": related_plan_models,
            "selected_plan_copy": str(selected_plan_copy_path),
            "qps_points": qps_summary,
            "run_script": str(run_script_path),
            "run_command_default": f"bash {BASE.path_for_config(run_script_path)}",
            "run_command_with_port_profile": (
                f"bash {BASE.path_for_config(run_script_path)} {default_port_profile}"
            ),
            "generation_command_raw": generation_command_raw,
            "service_mode": "interactive",
            "assumes_external_service_stack": True,
            "service_preflight": True,
            "interactive_setup_readme": INTERACTIVE_SETUP_README,
            "interactive_client_script": INTERACTIVE_CLIENT_SCRIPT,
            "interactive_env_helper_script": INTERACTIVE_ENV_HELPER,
            "interactive_launcher_script": INTERACTIVE_START_SERVICES_SCRIPT,
            "interactive_start_services_command": INTERACTIVE_START_SERVICES_COMMAND,
            "interactive_stop_services_command": INTERACTIVE_STOP_SERVICES_COMMAND,
            "interactive_foreground_start_services_command": (
                INTERACTIVE_FOREGROUND_START_COMMAND
            ),
            "interactive_power_daemon_enabled": True,
            "interactive_power_daemon_managed_by_launcher": True,
            "interactive_power_socket_env_example": (
                f"export AMD_SMI_POWER_SOCKET_PATH={AMD_SMI_POWER_SOCKET_EXAMPLE}"
            ),
            "interactive_execution_note": (
                f"Start the background service stack with "
                f"`{INTERACTIVE_START_SERVICES_COMMAND}` on the interactive node and wait "
                "for readiness. The generated `run_replay.sh` then sources the shared "
                f"helper at `{INTERACTIVE_ENV_HELPER}`, waits for the services if "
                "needed, defaults `AMD_SMI_POWER_SOCKET_PATH` to "
                "`/tmp/amdsmi-power-reader.sock` unless you override it in the "
                "environment, and runs the replay. When you are done, stop the "
                f"background stack with `{INTERACTIVE_STOP_SERVICES_COMMAND}`."
            ),
        }
        manifest_path = (batch_dir / "manifest.json").resolve()
        manifest_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        summary["manifest_path"] = str(manifest_path)
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
