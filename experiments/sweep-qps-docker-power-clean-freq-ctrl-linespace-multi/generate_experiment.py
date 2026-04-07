#!/usr/bin/env python3
"""Generate one local sweep-QPS + power logging + multi-linespace bundle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
EXPERIMENT_DIR_NAME = "sweep-qps-docker-power-clean-freq-ctrl-linespace-multi"
DEFAULT_OUTPUT_CONFIG_DIR = REPO_ROOT / "experiments" / EXPERIMENT_DIR_NAME / "generated"
DEFAULT_REPLAY_OUTPUT_ROOT = Path("results") / "replay" / EXPERIMENT_DIR_NAME
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model_config.toml"
EXPERIMENT_LOG_TAG = EXPERIMENT_DIR_NAME
DEFAULT_FREQ_CONTROLLER_THRESHOLD = 395784.0
DEFAULT_FREQ_CONTROL_LOG_DIR_NAME = "freq-control-linespace-multi"

SPLIT_ALIASES = {
    "full": "exclude-unranked",
    "exclude-unranked": "exclude-unranked",
    "exclude_unranked": "exclude-unranked",
    "exclued-unranked": "exclude-unranked",
    "exclued_unranked": "exclude-unranked",
    "top": "top",
    "rest": "rest",
}
SPLIT_CHOICES = ("full", "exclude-unranked", "top", "rest")
SPLIT_METRIC_ALIASES = {
    "token_usage": "token",
    "context_usage": "context",
}
PLAN_MODEL_LINE_RE = re.compile(r'"model"\s*:\s*"((?:\\.|[^"\\])*)"')


@dataclass
class LookupPlanResult:
    selected_plan_path: Path
    related_plan_paths: dict[str, str]


@dataclass(frozen=True)
class TargetModelSpec:
    key: str
    served_model_name: str
    vllm_model_name: str


@dataclass(frozen=True)
class QpsReplayJob:
    qps: float
    qps_display: str
    qps_slug: str
    replay_config_relpath: str
    replay_config_path: Path
    replay_output_dir: Path
    power_output_dir: Path
    freq_controller_log_dir: Path


def build_utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_generation_command_raw(argv: list[str] | None) -> str:
    if argv is None:
        original = getattr(sys, "orig_argv", None)
        if isinstance(original, list) and original:
            return shlex.join([str(token) for token in original])
        return shlex.join([str(sys.executable), *[str(token) for token in sys.argv]])
    return shlex.join(
        [str(sys.executable), str(Path(__file__).resolve()), *[str(token) for token in argv]]
    )


def parse_non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def parse_positive_float(value: str, *, field_name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return parsed


def parse_qps_list(raw: str) -> list[float]:
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--qps-list must be a non-empty comma-separated list")

    values: list[float] = []
    seen: set[float] = set()
    for token in tokens:
        try:
            parsed = float(token)
        except ValueError as exc:
            raise ValueError(f"--qps-list contains non-numeric value: {token!r}") from exc
        if parsed <= 0:
            raise ValueError("--qps-list values must be > 0")
        if parsed in seen:
            raise ValueError(f"--qps-list contains duplicate value: {token}")
        seen.add(parsed)
        values.append(parsed)
    return values


def parse_gpu_index_list(raw: str) -> list[int]:
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--gpu-indices must be a non-empty comma-separated list")

    values: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        try:
            parsed = int(token)
        except ValueError as exc:
            raise ValueError(f"--gpu-indices contains non-integer value: {token!r}") from exc
        if parsed < 0:
            raise ValueError("--gpu-indices values must be >= 0")
        if parsed in seen:
            raise ValueError(f"--gpu-indices contains duplicate value: {token}")
        seen.add(parsed)
        values.append(parsed)
    return values


def canonical_split(raw: str) -> str:
    normalized = raw.strip().lower().replace("_", "-")
    canonical = SPLIT_ALIASES.get(normalized)
    if canonical is None:
        allowed = ", ".join(SPLIT_CHOICES)
        raise ValueError(f"unsupported --split value {raw!r}; choose one of: {allowed}")
    return canonical


def format_qps_slug(qps: float) -> str:
    text = format(qps, ".12g")
    text = text.replace("-", "m").replace(".", "_").replace("+", "")
    return f"qps{text}"


def qps_display_text(qps: float) -> str:
    return format(qps, ".12g")


def with_plan_name_suffix(plan_path: Path, suffix: str) -> Path:
    if not suffix:
        raise ValueError("plan suffix cannot be empty")
    file_name = plan_path.name
    dot_index = file_name.rfind(".")
    if dot_index <= 0:
        suffixed_name = f"{file_name}.{suffix}"
    else:
        suffixed_name = f"{file_name[:dot_index]}.{suffix}{file_name[dot_index:]}"
    return (plan_path.parent / suffixed_name).resolve()


def path_for_config(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def derive_dataset_lineage_from_source_run_dir(source_run_dir: Path) -> Path:
    try:
        source_relative = source_run_dir.resolve().relative_to(RESULTS_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("--source-run-dir must live under top-level results/") from exc

    if len(source_relative.parts) < 3:
        raise ValueError(
            "--source-run-dir under results/ must include at least "
            "<model>/<dataset>/<run-dir>"
        )
    dataset_lineage_parts = source_relative.parts[1:-1]
    return Path(*dataset_lineage_parts)


def _load_target_model_specs() -> dict[str, TargetModelSpec]:
    if not MODEL_CONFIG_PATH.exists():
        raise ValueError(f"missing model config: {MODEL_CONFIG_PATH}")
    payload = tomllib.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    raw_models = payload.get("models")
    if not isinstance(raw_models, dict):
        raise ValueError("configs/model_config.toml must include [models]")
    out: dict[str, TargetModelSpec] = {}
    for key, value in raw_models.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, dict)
            or not isinstance(value.get("served_model_name"), str)
            or not isinstance(value.get("vllm_model_name"), str)
        ):
            continue
        served_model_name = value["served_model_name"].strip()
        vllm_model_name = value["vllm_model_name"].strip()
        if not key.strip() or not served_model_name or not vllm_model_name:
            continue
        out[key] = TargetModelSpec(
            key=key,
            served_model_name=served_model_name,
            vllm_model_name=vllm_model_name,
        )
    if not out:
        raise ValueError("configs/model_config.toml contains no valid model keys")
    return out


def _with_optional_plan_suffix(plan_path: Path, additional_suffix: str | None) -> Path:
    if additional_suffix is None:
        return plan_path.resolve()
    return with_plan_name_suffix(plan_path.resolve(), additional_suffix)


def _resolve_lookup_plan_paths(
    *,
    source_run_dir: Path,
    split: str,
    split_two_group_metric: str,
    additional_suffix: str | None,
) -> LookupPlanResult:
    metric_alias = SPLIT_METRIC_ALIASES.get(split_two_group_metric)
    if metric_alias is None:
        raise ValueError(
            f"unsupported --split-two-group-metric {split_two_group_metric!r}; "
            f"supported: {sorted(SPLIT_METRIC_ALIASES)}"
        )

    base_plan_path = (source_run_dir / "replay-plan.json").resolve()
    clean_base_plan_path = with_plan_name_suffix(base_plan_path, "clean")
    top_path = with_plan_name_suffix(clean_base_plan_path, f"{metric_alias}.top")
    rest_path = with_plan_name_suffix(clean_base_plan_path, f"{metric_alias}.rest")
    exclude_unranked_path = with_plan_name_suffix(
        clean_base_plan_path,
        f"{metric_alias}.exclude-unranked",
    )
    top_path = _with_optional_plan_suffix(top_path, additional_suffix)
    rest_path = _with_optional_plan_suffix(rest_path, additional_suffix)
    exclude_unranked_path = _with_optional_plan_suffix(
        exclude_unranked_path,
        additional_suffix,
    )

    if split == "top":
        selected = top_path
    elif split == "rest":
        selected = rest_path
    else:
        selected = exclude_unranked_path

    related_paths = {
        "selected": str(selected),
        "top": str(top_path),
        "rest": str(rest_path),
        "exclude_unranked": str(exclude_unranked_path),
    }

    missing_paths = sorted(
        {
            str(Path(path_text))
            for path_text in related_paths.values()
            if not Path(path_text).is_file()
        }
    )
    if missing_paths:
        missing_text = ", ".join(missing_paths)
        suffix_hint = (
            f" --additional-suffix {additional_suffix}"
            if additional_suffix
            else " (if plans were compiled with suffix, pass --additional-suffix)"
        )
        raise ValueError(
            "missing required precompiled replay plans: "
            f"{missing_text}. This script only looks up existing plans.{suffix_hint}"
        )
    return LookupPlanResult(
        selected_plan_path=selected,
        related_plan_paths=related_paths,
    )


def _extract_replay_target_model_from_plan(plan_path: Path) -> str:
    in_replay_target = False
    replay_target_brace_depth = 0
    with plan_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not in_replay_target:
                replay_target_index = line.find('"replay_target"')
                if replay_target_index < 0:
                    continue
                in_replay_target = True
                replay_target_open_brace = line.find("{", replay_target_index)
                if replay_target_open_brace >= 0:
                    replay_target_section = line[replay_target_open_brace:]
                    replay_target_brace_depth = (
                        replay_target_section.count("{")
                        - replay_target_section.count("}")
                    )
                else:
                    replay_target_brace_depth = 0
            else:
                replay_target_brace_depth += line.count("{") - line.count("}")

            model_match = PLAN_MODEL_LINE_RE.search(line)
            if model_match is not None:
                raw_model_value = model_match.group(1)
                return json.loads(f"\"{raw_model_value}\"")

            if in_replay_target and replay_target_brace_depth <= 0:
                break

    raise ValueError(f"unable to read replay_target.model from plan: {plan_path}")


def _validate_plan_model_matches_target(
    *,
    plan_path: Path,
    target_model_spec: TargetModelSpec,
) -> str:
    plan_model = _extract_replay_target_model_from_plan(plan_path)
    allowed_models = {
        target_model_spec.key,
        target_model_spec.served_model_name,
        target_model_spec.vllm_model_name,
    }
    if plan_model not in allowed_models:
        allowed_text = ", ".join(sorted(allowed_models))
        raise ValueError(
            "selected plan model does not match --target-model: "
            f"plan={plan_path}, replay_target.model={plan_model!r}, "
            f"--target-model={target_model_spec.key!r}, expected one of: {allowed_text}"
        )
    return plan_model


def _format_toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_scalar(item) for item in value) + "]"
    raise ValueError(f"unsupported TOML scalar type: {type(value)!r}")


def _append_toml_table(lines: list[str], table_name: str, payload: dict[str, Any]) -> None:
    lines.append(f"[{table_name}]")
    nested_items: list[tuple[str, dict[str, Any]]] = []
    for key in sorted(payload.keys()):
        value = payload[key]
        if value is None:
            continue
        if isinstance(value, dict):
            nested_items.append((key, value))
            continue
        lines.append(f"{key} = {_format_toml_scalar(value)}")
    for key, nested in nested_items:
        lines.append("")
        _append_toml_table(lines, f"{table_name}.{key}", nested)


def write_replay_config(path: Path, *, replay_payload: dict[str, Any]) -> None:
    lines: list[str] = [
        (
            "# Auto-generated by "
            f"experiments/{EXPERIMENT_DIR_NAME}/generate_experiment.py"
        ),
        "",
    ]
    _append_toml_table(lines, "replay", replay_payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def write_run_script(
    path: Path,
    *,
    default_port_profile: int,
    target_model: str,
    split: str,
    power_gpu_indices: list[int],
    qps_jobs: list[QpsReplayJob],
    freq_controller_threshold: float,
) -> None:
    power_gpu_indices_payload = " ".join(str(gpu_index) for gpu_index in power_gpu_indices)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        f"REPO_ROOT={_shell_quote(str(REPO_ROOT.resolve()))}",
        f"DEFAULT_PORT_PROFILE_ID={default_port_profile}",
        "PORT_PROFILE_ID_VALUE=\"${1:-${PORT_PROFILE_ID:-${DEFAULT_PORT_PROFILE_ID}}}\"",
        "PYTHON_BIN=\"${PYTHON_BIN:-python3}\"",
        "ZEUS_POWER_READER_BIN=\"${ZEUS_POWER_READER_BIN:-zeus-power-reader}\"",
        (
            "FREQ_CONTROLLER_BIN="
            "\"${FREQ_CONTROLLER_BIN:-freq-controller-linespace-multi}\""
        ),
        "RESET_GPU_CORE_FREQ_BIN=\"${RESET_GPU_CORE_FREQ_BIN:-reset-gpu-core-freq}\"",
        "ZEUSD_SOCKET_PATH_VALUE=\"${ZEUSD_SOCKET_PATH:-}\"",
        "FREQ_CONTROLLER_CONFIG_VALUE=\"${FREQ_CONTROLLER_CONFIG:-}\"",
        f"DEFAULT_FREQ_CONTROLLER_THRESHOLD={freq_controller_threshold}",
        (
            "FREQ_CONTROLLER_THRESHOLD_VALUE="
            "\"${FREQ_CONTROLLER_THRESHOLD:-${DEFAULT_FREQ_CONTROLLER_THRESHOLD}}\""
        ),
        f"POWER_GPU_INDICES=({power_gpu_indices_payload})",
        "",
        "POWER_READER_PID=\"\"",
        "FREQ_CONTROLLER_PID=\"\"",
        "FREQ_CONTROLLER_STARTED=0",
        "",
        "stop_power_reader() {",
        "  if [[ -n \"${POWER_READER_PID}\" ]]; then",
        "    kill \"${POWER_READER_PID}\" >/dev/null 2>&1 || true",
        "    wait \"${POWER_READER_PID}\" 2>/dev/null || true",
        "    POWER_READER_PID=\"\"",
        "  fi",
        "}",
        "",
        "stop_freq_controller() {",
        "  if [[ -n \"${FREQ_CONTROLLER_PID}\" ]]; then",
        "    kill \"${FREQ_CONTROLLER_PID}\" >/dev/null 2>&1 || true",
        "    wait \"${FREQ_CONTROLLER_PID}\" 2>/dev/null || true",
        "    FREQ_CONTROLLER_PID=\"\"",
        "  fi",
        "}",
        "",
        "reset_gpu_core_if_needed() {",
        "  if [[ \"${FREQ_CONTROLLER_STARTED}\" -eq 1 ]]; then",
        "    local gpu_index=\"\"",
        "    for gpu_index in \"${POWER_GPU_INDICES[@]}\"; do",
        "      if ! \"${RESET_GPU_CORE_FREQ_BIN}\" --gpu-index \"${gpu_index}\"; then",
        (
            f"        echo \"[{EXPERIMENT_LOG_TAG}] warning: failed to reset GPU "
            "${gpu_index} core clocks\" >&2"
        ),
        "      fi",
        "    done",
        "    FREQ_CONTROLLER_STARTED=0",
        "  fi",
        "}",
        "",
        "cleanup() {",
        "  local exit_code=\"$1\"",
        "  stop_freq_controller",
        "  stop_power_reader",
        "  reset_gpu_core_if_needed",
        "  return \"${exit_code}\"",
        "}",
        "",
        (
            "trap '__exit_code=$?; trap - EXIT INT TERM; cleanup "
            "\"${__exit_code}\"; exit \"${__exit_code}\"' EXIT"
        ),
        f"trap 'echo \"[{EXPERIMENT_LOG_TAG}] interrupted\" >&2; exit 130' INT TERM",
        "",
        "run_one_qps() {",
        "  local qps_value=\"$1\"",
        "  local qps_slug=\"$2\"",
        "  local replay_config_ref=\"$3\"",
        "  local replay_output_ref=\"$4\"",
        "  local replay_config_path=\"\"",
        "  local replay_output_dir=\"\"",
        "",
        "  if [[ \"${replay_config_ref}\" = /* ]]; then",
        "    replay_config_path=\"${replay_config_ref}\"",
        "  else",
        "    replay_config_path=\"${SCRIPT_DIR}/${replay_config_ref}\"",
        "  fi",
        "",
        "  if [[ \"${replay_output_ref}\" = /* ]]; then",
        "    replay_output_dir=\"${replay_output_ref}\"",
        "  else",
        "    replay_output_dir=\"${REPO_ROOT}/${replay_output_ref}\"",
        "  fi",
        "  local power_output_dir=\"${replay_output_dir}/power\"",
        (
            "  local freq_controller_log_dir="
            f"\"${{replay_output_dir}}/{DEFAULT_FREQ_CONTROL_LOG_DIR_NAME}\""
        ),
        "",
        (
            f"  echo \"[{EXPERIMENT_LOG_TAG}] split={split} qps=${{qps_value}} "
            "slug=${qps_slug} "
            "gpu_indices=${POWER_GPU_INDICES[*]} "
            "threshold=${FREQ_CONTROLLER_THRESHOLD_VALUE} "
            "output=${replay_output_ref} "
            "port_profile=${PORT_PROFILE_ID_VALUE}\""
        ),
        "  mkdir -p \"${power_output_dir}\" \"${freq_controller_log_dir}\"",
        "  if [[ -n \"${ZEUSD_SOCKET_PATH_VALUE}\" ]]; then",
        (
            "    \"${ZEUS_POWER_READER_BIN}\" --output-dir "
            "\"${power_output_dir}\" --gpu-indices "
            "\"${POWER_GPU_INDICES[@]}\" --socket-path "
            "\"${ZEUSD_SOCKET_PATH_VALUE}\" &"
        ),
        "  else",
        (
            "    \"${ZEUS_POWER_READER_BIN}\" --output-dir "
            "\"${power_output_dir}\" --gpu-indices "
            "\"${POWER_GPU_INDICES[@]}\" &"
        ),
        "  fi",
        "  POWER_READER_PID=\"$!\"",
        "",
        "  FREQ_CONTROLLER_CMD=(",
        "    \"${FREQ_CONTROLLER_BIN}\"",
        "    \"--log-dir\" \"${freq_controller_log_dir}\"",
        "    \"--threshold\" \"${FREQ_CONTROLLER_THRESHOLD_VALUE}\"",
        "    \"--port-profile-id\" \"${PORT_PROFILE_ID_VALUE}\"",
        "    \"--gpu-indices\" \"${POWER_GPU_INDICES[@]}\"",
        "  )",
        "  if [[ -n \"${FREQ_CONTROLLER_CONFIG_VALUE}\" ]]; then",
        "    FREQ_CONTROLLER_CMD+=(\"--config\" \"${FREQ_CONTROLLER_CONFIG_VALUE}\")",
        "  fi",
        "  \"${FREQ_CONTROLLER_CMD[@]}\" &",
        "  FREQ_CONTROLLER_PID=\"$!\"",
        "  FREQ_CONTROLLER_STARTED=1",
        "  sleep 1",
        "  if ! kill -0 \"${FREQ_CONTROLLER_PID}\" >/dev/null 2>&1; then",
        (
            f"    echo \"[{EXPERIMENT_LOG_TAG}] "
            "freq-controller-linespace-multi failed to stay alive after startup\" >&2"
        ),
        "    wait \"${FREQ_CONTROLLER_PID}\" || true",
        "    exit 1",
        "  fi",
        "",
        "  \"${PYTHON_BIN}\" -m replayer replay \\",
        "    --config \"${replay_config_path}\" \\",
        "    --port-profile-id \"${PORT_PROFILE_ID_VALUE}\"",
        "",
        "  stop_freq_controller",
        "  stop_power_reader",
        "  reset_gpu_core_if_needed",
        "}",
        "",
        (
            f"echo \"[{EXPERIMENT_LOG_TAG}] target_model={target_model} split={split} "
            f"qps_points={len(qps_jobs)} "
            "gpu_indices=${POWER_GPU_INDICES[*]} "
            "threshold=${FREQ_CONTROLLER_THRESHOLD_VALUE} "
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
            "Generate a sweep-QPS replay bundle with multi-GPU power logging and "
            "freq-controller-linespace-multi by looking up an existing compiled "
            "clean split plan and emitting one sweep entrypoint script."
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
        type=parse_non_negative_int,
        help="Poisson launch seed.",
    )
    parser.add_argument(
        "--randomize-seed",
        required=True,
        type=parse_non_negative_int,
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
        required=True,
        type=int,
        help="Port profile ID for replay.",
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
        choices=sorted(SPLIT_METRIC_ALIASES),
        default="token_usage",
        help="Split grouping metric used for top/rest plan lookup (default: token_usage).",
    )
    parser.add_argument(
        "--gpu-indices",
        "--power-gpu-indices",
        dest="gpu_indices",
        required=True,
        help=(
            "Comma-separated GPU indices used by both "
            "freq-controller-linespace-multi and zeus-power-reader, "
            "for example: 2,3."
        ),
    )
    parser.add_argument(
        "--freq-controller-threshold",
        type=float,
        default=DEFAULT_FREQ_CONTROLLER_THRESHOLD,
        help=(
            "Default freq-controller-linespace-multi context threshold. "
            f"Default: {int(DEFAULT_FREQ_CONTROLLER_THRESHOLD)}."
        ),
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
            "<dataset-lineage>/split/<split>/<qps>/<timestamp> under "
            f"results/replay/{EXPERIMENT_DIR_NAME}/. "
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
        source_run_dir = Path(args.source_run_dir).expanduser().resolve()
        if not source_run_dir.exists() or not source_run_dir.is_dir():
            raise ValueError(f"invalid --source-run-dir: {source_run_dir}")
        try:
            source_run_dir.relative_to(RESULTS_ROOT.resolve())
        except ValueError:
            raise ValueError("--source-run-dir must live under top-level results/")

        split = canonical_split(args.split)
        qps_values = parse_qps_list(str(args.qps_list))
        power_gpu_indices = parse_gpu_index_list(str(args.gpu_indices))
        time_constraint_s = parse_positive_float(
            str(args.time_constraint_s),
            field_name="--time-constraint-s",
        )
        if args.port_profile < 0:
            raise ValueError("--port-profile must be >= 0")
        threshold_value = parse_positive_float(
            str(args.freq_controller_threshold),
            field_name="--freq-controller-threshold",
        )

        target_model = str(args.target_model).strip()
        if not target_model:
            raise ValueError("--target-model cannot be empty")
        target_model_specs = _load_target_model_specs()
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
        source_dataset_lineage = derive_dataset_lineage_from_source_run_dir(source_run_dir)

        batch_timestamp = build_utc_timestamp_slug()
        batch_dir = (output_config_root / batch_timestamp).resolve()

        lookup_result = _resolve_lookup_plan_paths(
            source_run_dir=source_run_dir,
            split=split,
            split_two_group_metric=str(args.split_two_group_metric),
            additional_suffix=additional_suffix,
        )
        related_plan_models: dict[str, str] = {}
        for plan_key, plan_path_text in lookup_result.related_plan_paths.items():
            plan_model = _validate_plan_model_matches_target(
                plan_path=Path(plan_path_text),
                target_model_spec=target_model_spec,
            )
            related_plan_models[plan_key] = plan_model

        plan_copy_dir = (batch_dir / "plan").resolve()
        plan_copy_dir.mkdir(parents=True, exist_ok=True)
        selected_plan_copy_path = (plan_copy_dir / lookup_result.selected_plan_path.name).resolve()
        shutil.copy2(lookup_result.selected_plan_path, selected_plan_copy_path)

        qps_jobs: list[QpsReplayJob] = []
        for qps in qps_values:
            qps_slug = format_qps_slug(qps)
            replay_output_dir = (
                replay_output_root
                / source_dataset_lineage
                / "split"
                / split
                / qps_slug
                / batch_timestamp
            ).resolve()
            replay_payload = {
                "plan": path_for_config(selected_plan_copy_path),
                "output_dir": path_for_config(replay_output_dir),
                "randomize_seed": int(args.randomize_seed),
                "time_constraint_s": time_constraint_s,
                "port_profile_id": int(args.port_profile),
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
                    qps_display=qps_display_text(qps),
                    qps_slug=qps_slug,
                    replay_config_relpath=replay_config_relpath,
                    replay_config_path=replay_config_path,
                    replay_output_dir=replay_output_dir,
                    power_output_dir=(replay_output_dir / "power").resolve(),
                    freq_controller_log_dir=(
                        replay_output_dir / DEFAULT_FREQ_CONTROL_LOG_DIR_NAME
                    ).resolve(),
                )
            )

        run_script_path = (batch_dir / "run_replay.sh").resolve()
        write_run_script(
            run_script_path,
            default_port_profile=int(args.port_profile),
            target_model=target_model,
            split=split,
            power_gpu_indices=power_gpu_indices,
            qps_jobs=qps_jobs,
            freq_controller_threshold=threshold_value,
        )

        qps_summary = [
            {
                "qps": job.qps,
                "qps_slug": job.qps_slug,
                "replay_config": str(job.replay_config_path),
                "replay_output_dir": path_for_config(job.replay_output_dir),
                "power_output_dir": path_for_config(job.power_output_dir),
                "freq_controller_log_dir": path_for_config(job.freq_controller_log_dir),
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
            "time_constraint_s": time_constraint_s,
            "port_profile": int(args.port_profile),
            "gpu_indices": power_gpu_indices,
            "power_gpu_indices": power_gpu_indices,
            "freq_controller_threshold": threshold_value,
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
            "run_command_default": f"bash {path_for_config(run_script_path)}",
            "run_command_with_port_profile": (
                f"bash {path_for_config(run_script_path)} {int(args.port_profile)}"
            ),
            "generation_command_raw": generation_command_raw,
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
