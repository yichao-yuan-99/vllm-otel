#!/usr/bin/env python3
"""Generate one derived-plan sweep-QPS + power logging replay experiment bundle.

This helper does all of the following:
1) Validates the selected replay plan exists and is a derived single-trail plan.
2) Validates the plan's replay target matches the requested target model.
3) Materializes per-QPS replay TOMLs + a runnable sweep script with power logging.
"""

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
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model_config.toml"
EXPERIMENT_DIR_NAME = "sweep-qps-same-agent"
EXPERIMENT_LOG_TAG = EXPERIMENT_DIR_NAME
GENERATOR_SCRIPT_PATH = Path(__file__).resolve()
LAUNCH_PATTERN_NAME = "poisson"
LAUNCH_PATTERN_LABEL = "Poisson"
LAUNCH_SEED_OPTION: str | None = "--poisson-seed"
LAUNCH_SEED_DEST: str | None = "poisson_seed"
LAUNCH_SEED_MANIFEST_KEY: str | None = "poisson_seed"


def _default_output_config_dir() -> Path:
    return REPO_ROOT / "experiments" / EXPERIMENT_DIR_NAME / "generated"


def _default_replay_output_root() -> Path:
    return Path("results") / "replay" / EXPERIMENT_DIR_NAME


DEFAULT_OUTPUT_CONFIG_DIR = _default_output_config_dir()
DEFAULT_REPLAY_OUTPUT_ROOT = _default_replay_output_root()

PLAN_MODEL_LINE_RE = re.compile(r'"model"\s*:\s*"((?:\\.|[^"\\])*)"')
PLAN_IS_DERIVED_LINE_RE = re.compile(r'"is_derived"\s*:\s*(true|false)')
PLAN_SINGLE_TRAIL_LINE_RE = re.compile(r'"single_trail"\s*:\s*"((?:\\.|[^"\\])*)"')
PLAN_SOURCE_JOB_DIR_LINE_RE = re.compile(r'"source_job_dir"\s*:\s*"((?:\\.|[^"\\])*)"')


def configure_experiment_variant(
    *,
    experiment_dir_name: str,
    launch_pattern_name: str,
    launch_pattern_label: str,
    launch_seed_option: str | None,
    launch_seed_dest: str | None,
    launch_seed_manifest_key: str | None,
    generator_script_path: Path,
) -> None:
    global EXPERIMENT_DIR_NAME
    global EXPERIMENT_LOG_TAG
    global GENERATOR_SCRIPT_PATH
    global LAUNCH_PATTERN_NAME
    global LAUNCH_PATTERN_LABEL
    global LAUNCH_SEED_OPTION
    global LAUNCH_SEED_DEST
    global LAUNCH_SEED_MANIFEST_KEY
    global DEFAULT_OUTPUT_CONFIG_DIR
    global DEFAULT_REPLAY_OUTPUT_ROOT

    EXPERIMENT_DIR_NAME = experiment_dir_name
    EXPERIMENT_LOG_TAG = experiment_dir_name
    GENERATOR_SCRIPT_PATH = generator_script_path.resolve()
    LAUNCH_PATTERN_NAME = launch_pattern_name
    LAUNCH_PATTERN_LABEL = launch_pattern_label
    LAUNCH_SEED_OPTION = launch_seed_option
    LAUNCH_SEED_DEST = launch_seed_dest
    LAUNCH_SEED_MANIFEST_KEY = launch_seed_manifest_key
    DEFAULT_OUTPUT_CONFIG_DIR = _default_output_config_dir()
    DEFAULT_REPLAY_OUTPUT_ROOT = _default_replay_output_root()


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


def build_utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_generation_command_raw(argv: list[str] | None) -> str:
    if argv is None:
        original = getattr(sys, "orig_argv", None)
        if isinstance(original, list) and original:
            return shlex.join([str(token) for token in original])
        return shlex.join([str(sys.executable), *[str(token) for token in sys.argv]])
    return shlex.join(
        [str(sys.executable), str(GENERATOR_SCRIPT_PATH), *[str(token) for token in argv]]
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
        raise ValueError("--power-gpu-indices must be a non-empty comma-separated list")

    values: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        try:
            parsed = int(token)
        except ValueError as exc:
            raise ValueError(
                f"--power-gpu-indices contains non-integer value: {token!r}"
            ) from exc
        if parsed < 0:
            raise ValueError("--power-gpu-indices values must be >= 0")
        if parsed in seen:
            raise ValueError(
                f"--power-gpu-indices contains duplicate value: {token}"
            )
        seen.add(parsed)
        values.append(parsed)
    return values


def format_qps_slug(qps: float) -> str:
    text = format(qps, ".12g")
    text = text.replace("-", "m").replace(".", "_").replace("+", "")
    return f"qps{text}"


def qps_display_text(qps: float) -> str:
    return format(qps, ".12g")


def safe_name(value: str) -> str:
    chars: list[str] = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars) or "value"


def path_for_config(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def normalize_output_suffix(raw: str | None) -> str | None:
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        raise ValueError("--output-suffix cannot be empty")
    return safe_name(stripped)


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


def _decode_json_string_match(match: re.Match[str]) -> str:
    return json.loads(f"\"{match.group(1)}\"")


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
                        replay_target_section.count("{") - replay_target_section.count("}")
                    )
                else:
                    replay_target_brace_depth = 0
            else:
                replay_target_brace_depth += line.count("{") - line.count("}")

            model_match = PLAN_MODEL_LINE_RE.search(line)
            if model_match is not None:
                return _decode_json_string_match(model_match)

            if in_replay_target and replay_target_brace_depth <= 0:
                break

    raise ValueError(f"unable to read replay_target.model from plan: {plan_path}")


def _extract_compile_single_trail_from_plan(plan_path: Path) -> str | None:
    in_compile_options = False
    compile_options_brace_depth = 0
    with plan_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not in_compile_options:
                compile_options_index = line.find('"compile_options"')
                if compile_options_index < 0:
                    continue
                in_compile_options = True
                compile_options_open_brace = line.find("{", compile_options_index)
                if compile_options_open_brace >= 0:
                    compile_options_section = line[compile_options_open_brace:]
                    compile_options_brace_depth = (
                        compile_options_section.count("{")
                        - compile_options_section.count("}")
                    )
                else:
                    compile_options_brace_depth = 0
            else:
                compile_options_brace_depth += line.count("{") - line.count("}")

            single_trail_match = PLAN_SINGLE_TRAIL_LINE_RE.search(line)
            if single_trail_match is not None:
                return _decode_json_string_match(single_trail_match)

            if in_compile_options and compile_options_brace_depth <= 0:
                break
    return None


def _extract_is_derived_from_plan(plan_path: Path) -> bool | None:
    with plan_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            is_derived_match = PLAN_IS_DERIVED_LINE_RE.search(line)
            if is_derived_match is None:
                continue
            return is_derived_match.group(1) == "true"
    return None


def _extract_source_job_dir_from_plan(plan_path: Path) -> str | None:
    with plan_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"workers"' in line:
                break
            source_job_dir_match = PLAN_SOURCE_JOB_DIR_LINE_RE.search(line)
            if source_job_dir_match is not None:
                return _decode_json_string_match(source_job_dir_match)
    return None


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


def _derive_dataset_lineage_from_results_run_dir(
    run_dir: Path,
    *,
    value_name: str,
) -> Path:
    parts = run_dir.expanduser().parts
    results_index = next((index for index, part in enumerate(parts) if part == "results"), None)
    if results_index is None:
        raise ValueError(f"{value_name} must include a results/<model>/<dataset>/<run-dir> path")
    relative_parts = parts[results_index + 1 :]
    if len(relative_parts) < 3:
        raise ValueError(
            f"{value_name} must include at least results/<model>/<dataset>/<run-dir>"
        )
    return Path(*relative_parts[1:-1])


def _maybe_derive_source_dataset_lineage(
    *,
    source_plan_path: Path,
    source_job_dir_text: str | None,
) -> Path | None:
    if source_job_dir_text:
        try:
            return _derive_dataset_lineage_from_results_run_dir(
                Path(source_job_dir_text),
                value_name="source_job_dir",
            )
        except ValueError:
            pass
    try:
        return _derive_dataset_lineage_from_results_run_dir(
            source_plan_path.parent,
            value_name="--source-plan parent",
        )
    except ValueError:
        return None


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
        f"# Auto-generated by experiments/{EXPERIMENT_DIR_NAME}/generate_experiment.py",
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
    source_trail_name: str,
    power_gpu_indices: list[int],
    qps_jobs: list[QpsReplayJob],
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
        f"SOURCE_TRAIL_NAME={_shell_quote(source_trail_name)}",
        "PYTHON_BIN=\"${PYTHON_BIN:-python3}\"",
        "ZEUS_POWER_READER_BIN=\"${ZEUS_POWER_READER_BIN:-zeus-power-reader}\"",
        "ZEUSD_SOCKET_PATH_VALUE=\"${ZEUSD_SOCKET_PATH:-}\"",
        f"POWER_GPU_INDICES=({power_gpu_indices_payload})",
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
        "",
        f"  echo \"[{EXPERIMENT_LOG_TAG}] trail=${{SOURCE_TRAIL_NAME}} qps=${{qps_value}} slug=${{qps_slug}} output=${{replay_output_ref}} port_profile=${{PORT_PROFILE_ID_VALUE}}\"",
        "  mkdir -p \"${power_output_dir}\"",
        "  if [[ -n \"${ZEUSD_SOCKET_PATH_VALUE}\" ]]; then",
        "    \"${ZEUS_POWER_READER_BIN}\" --output-dir \"${power_output_dir}\" --gpu-indices \"${POWER_GPU_INDICES[@]}\" --socket-path \"${ZEUSD_SOCKET_PATH_VALUE}\" &",
        "  else",
        "    \"${ZEUS_POWER_READER_BIN}\" --output-dir \"${power_output_dir}\" --gpu-indices \"${POWER_GPU_INDICES[@]}\" &",
        "  fi",
        "  POWER_READER_PID=\"$!\"",
        "",
        "  \"${PYTHON_BIN}\" -m replayer replay \\",
        "    --config \"${replay_config_path}\" \\",
        "    --port-profile-id \"${PORT_PROFILE_ID_VALUE}\"",
        "",
        "  stop_power_reader",
        "}",
        "",
        (
            f"echo \"[{EXPERIMENT_LOG_TAG}] target_model={target_model} "
            "trail=${SOURCE_TRAIL_NAME} "
            f"qps_points={len(qps_jobs)} "
            f"power_gpu_indices={power_gpu_indices_payload} "
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
            "Generate a sweep-QPS replay bundle with power logging from one precompiled "
            "derived single-trail replay plan and emit one sweep entrypoint script."
        ),
    )
    parser.add_argument(
        "--source-plan",
        required=True,
        help=(
            "Path to a derived replay plan produced from a single-trail source plan. "
            "Original non-derived single-trail plans are rejected."
        ),
    )
    if LAUNCH_SEED_OPTION is not None and LAUNCH_SEED_DEST is not None:
        parser.add_argument(
            LAUNCH_SEED_OPTION,
            required=True,
            dest=LAUNCH_SEED_DEST,
            type=parse_non_negative_int,
            help=f"{LAUNCH_PATTERN_LABEL} launch seed.",
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
        help="Comma-separated QPS rates, e.g. 0.1,0.2,0.3",
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
        "--power-gpu-indices",
        default="0",
        help=(
            "Comma-separated GPU indices passed to zeus-power-reader, "
            "for example: 0 or 0,1 (default: 0)."
        ),
    )
    parser.add_argument(
        "--output-config-dir",
        default=str(DEFAULT_OUTPUT_CONFIG_DIR),
        help=f"Generated bundle root (default: {DEFAULT_OUTPUT_CONFIG_DIR}).",
    )
    parser.add_argument(
        "--replay-output-root",
        default=str(DEFAULT_REPLAY_OUTPUT_ROOT),
        help=(
            "Replay output root. Default appends "
            "<dataset-lineage>/trail[-suffix]/<safe-source-trail>/<qps>/<timestamp> "
            f"under results/replay/{EXPERIMENT_DIR_NAME}/. If dataset-lineage "
            "cannot be derived from the plan, the path starts at "
            "trail[-suffix]/<safe-source-trail>/..."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        help=(
            "Optional suffix for the top-level trail output directory. "
            'For example, --output-suffix lmcache writes under trail-lmcache/ '
            "instead of trail/."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    generation_command_raw = build_generation_command_raw(argv)

    try:
        source_plan_path = Path(args.source_plan).expanduser().resolve()
        if not source_plan_path.exists() or not source_plan_path.is_file():
            raise ValueError(f"invalid --source-plan: {source_plan_path}")
        launch_seed = (
            int(getattr(args, LAUNCH_SEED_DEST))
            if LAUNCH_SEED_DEST is not None
            else None
        )

        qps_values = parse_qps_list(str(args.qps_list))
        time_constraint_s = parse_positive_float(
            str(args.time_constraint_s),
            field_name="--time-constraint-s",
        )
        if args.port_profile < 0:
            raise ValueError("--port-profile must be >= 0")
        power_gpu_indices = parse_gpu_index_list(str(args.power_gpu_indices))

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

        output_config_root = Path(args.output_config_dir).expanduser().resolve()
        replay_output_root = Path(args.replay_output_root).expanduser()
        if not replay_output_root.is_absolute():
            replay_output_root = (REPO_ROOT / replay_output_root).resolve()
        else:
            replay_output_root = replay_output_root.resolve()
        output_suffix = normalize_output_suffix(getattr(args, "output_suffix", None))
        trail_dir_name = (
            "trail" if output_suffix is None else f"trail-{output_suffix}"
        )

        is_derived = _extract_is_derived_from_plan(source_plan_path)
        if is_derived is not True:
            raise ValueError(
                "--source-plan must be a derived replay plan "
                "(expected top-level is_derived=true); original single-trail plans are rejected"
            )
        source_trail_name = _extract_compile_single_trail_from_plan(source_plan_path)
        if source_trail_name is None:
            raise ValueError(
                "--source-plan is missing compile_options.single_trail from the original source plan"
            )
        source_job_dir_text = _extract_source_job_dir_from_plan(source_plan_path)
        source_dataset_lineage = _maybe_derive_source_dataset_lineage(
            source_plan_path=source_plan_path,
            source_job_dir_text=source_job_dir_text,
        )
        source_trail_slug = safe_name(source_trail_name)
        source_plan_model = _validate_plan_model_matches_target(
            plan_path=source_plan_path,
            target_model_spec=target_model_spec,
        )

        batch_timestamp = build_utc_timestamp_slug()
        batch_dir = (output_config_root / batch_timestamp).resolve()

        plan_copy_dir = (batch_dir / "plan").resolve()
        plan_copy_dir.mkdir(parents=True, exist_ok=True)
        source_plan_copy_path = (plan_copy_dir / source_plan_path.name).resolve()
        shutil.copy2(source_plan_path, source_plan_copy_path)

        replay_output_root_base = replay_output_root
        if source_dataset_lineage is not None:
            replay_output_root_base = replay_output_root_base / source_dataset_lineage
        replay_output_root_base = (
            replay_output_root_base / trail_dir_name / source_trail_slug
        ).resolve()

        qps_jobs: list[QpsReplayJob] = []
        for qps in qps_values:
            qps_slug = format_qps_slug(qps)
            replay_output_dir = (
                replay_output_root_base / qps_slug / batch_timestamp
            ).resolve()
            replay_payload = {
                "plan": path_for_config(source_plan_copy_path),
                "output_dir": path_for_config(replay_output_dir),
                "randomize_seed": int(args.randomize_seed),
                "time_constraint_s": time_constraint_s,
                "port_profile_id": int(args.port_profile),
                "launch_policy_override": {
                    "pattern": {"name": LAUNCH_PATTERN_NAME},
                    "pattern_args": {"rate": qps},
                },
            }
            if launch_seed is not None:
                replay_payload["launch_policy_override"]["seed"] = launch_seed
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
                )
            )

        run_script_path = (batch_dir / "run_replay.sh").resolve()
        write_run_script(
            run_script_path,
            default_port_profile=int(args.port_profile),
            target_model=target_model,
            source_trail_name=source_trail_name,
            power_gpu_indices=power_gpu_indices,
            qps_jobs=qps_jobs,
        )

        qps_summary = [
            {
                "qps": job.qps,
                "qps_slug": job.qps_slug,
                "replay_config": str(job.replay_config_path),
                "replay_output_dir": path_for_config(job.replay_output_dir),
                "power_output_dir": path_for_config(job.power_output_dir),
            }
            for job in qps_jobs
        ]
        summary = {
            "status": "ok",
            "batch_timestamp": batch_timestamp,
            "source_plan": str(source_plan_path),
            "source_plan_copy": str(source_plan_copy_path),
            "source_plan_is_derived": True,
            "source_job_dir": source_job_dir_text,
            "source_dataset_lineage": (
                str(source_dataset_lineage) if source_dataset_lineage is not None else None
            ),
            "source_trail_name": source_trail_name,
            "source_trail_slug": source_trail_slug,
            "source_plan_model": source_plan_model,
            "target_model": target_model,
            "launch_pattern": LAUNCH_PATTERN_NAME,
            "randomize_seed": int(args.randomize_seed),
            "qps_list": qps_values,
            "time_constraint_s": time_constraint_s,
            "port_profile": int(args.port_profile),
            "power_gpu_indices": power_gpu_indices,
            "output_config_root_dir": str(output_config_root),
            "output_batch_dir": str(batch_dir),
            "output_suffix": output_suffix,
            "replay_output_trail_dir_name": trail_dir_name,
            "replay_output_root_base_dir": path_for_config(replay_output_root_base),
            "qps_points": qps_summary,
            "run_script": str(run_script_path),
            "run_command_default": f"bash {path_for_config(run_script_path)}",
            "run_command_with_port_profile": (
                f"bash {path_for_config(run_script_path)} {int(args.port_profile)}"
            ),
            "generation_command_raw": generation_command_raw,
        }
        if LAUNCH_SEED_MANIFEST_KEY is not None and launch_seed is not None:
            summary[LAUNCH_SEED_MANIFEST_KEY] = launch_seed
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
