#!/usr/bin/env python3
"""Generate one local single-QPS frequency-sweep replay experiment bundle.

This helper does all of the following:
1) Looks up existing replay plans for the requested split mode.
2) Validates selected plans match the requested target model.
3) Materializes per-frequency replay TOMLs + runnable sweep script.
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


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_CONFIG_DIR = (
    REPO_ROOT / "experiments" / "single-qps-sweep-freq" / "local" / "generated"
)
DEFAULT_REPLAY_OUTPUT_ROOT = Path("results") / "replay" / "single-qps-sweep-freq" / "split"
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model_config.toml"

SPLIT_ALIASES = {
    "full": "full",
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
class FrequencyRange:
    min_mhz: int
    max_mhz: int
    slug: str


@dataclass(frozen=True)
class FrequencyReplayJob:
    min_mhz: int
    max_mhz: int
    slug: str
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
    return shlex.join([str(sys.executable), str(Path(__file__).resolve()), *[str(token) for token in argv]])


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


def format_frequency_slug(min_mhz: int, max_mhz: int) -> str:
    return f"core-{min_mhz}-{max_mhz}"


def parse_frequency_ranges(raw: str) -> list[FrequencyRange]:
    tokens = [token.strip() for token in raw.split(";")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--freq-list must be a non-empty ';' separated list like 345:1305;345:1005")

    parsed_ranges: list[FrequencyRange] = []
    seen_ranges: set[tuple[int, int]] = set()
    for token in tokens:
        parts = [part.strip() for part in token.split(":")]
        if len(parts) != 2 or any(not part for part in parts):
            raise ValueError(
                f"invalid --freq-list token {token!r}; expected '<min_mhz>:<max_mhz>'"
            )
        try:
            min_mhz = int(parts[0])
            max_mhz = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"invalid --freq-list token {token!r}; min/max values must be integers"
            ) from exc
        if min_mhz <= 0 or max_mhz <= 0:
            raise ValueError("--freq-list values must be > 0")
        if min_mhz > max_mhz:
            raise ValueError(
                f"invalid --freq-list token {token!r}; min_mhz cannot exceed max_mhz"
            )
        pair = (min_mhz, max_mhz)
        if pair in seen_ranges:
            raise ValueError(f"--freq-list contains duplicate range: {min_mhz}:{max_mhz}")
        seen_ranges.add(pair)
        parsed_ranges.append(
            FrequencyRange(
                min_mhz=min_mhz,
                max_mhz=max_mhz,
                slug=format_frequency_slug(min_mhz, max_mhz),
            )
        )
    return parsed_ranges


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
    if split == "full":
        selected = _with_optional_plan_suffix(source_run_dir / "replay-plan.json", additional_suffix)
        related_paths = {"selected": str(selected)}
    elif split == "exclude-unranked":
        selected = _with_optional_plan_suffix(
            source_run_dir / "replay-plan.exclude-unranked.json",
            additional_suffix,
        )
        related_paths = {"selected": str(selected)}
    else:
        metric_alias = SPLIT_METRIC_ALIASES.get(split_two_group_metric)
        if metric_alias is None:
            raise ValueError(
                f"unsupported --split-two-group-metric {split_two_group_metric!r}; "
                f"supported: {sorted(SPLIT_METRIC_ALIASES)}"
            )

        base_plan_path = (source_run_dir / "replay-plan.json").resolve()
        top_path = with_plan_name_suffix(base_plan_path, f"{metric_alias}.top")
        rest_path = with_plan_name_suffix(base_plan_path, f"{metric_alias}.rest")
        top_path = _with_optional_plan_suffix(top_path, additional_suffix)
        rest_path = _with_optional_plan_suffix(rest_path, additional_suffix)
        selected = top_path if split == "top" else rest_path
        related_paths = {
            "selected": str(selected),
            "top": str(top_path),
            "rest": str(rest_path),
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
                        replay_target_section.count("{") - replay_target_section.count("}")
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
        "# Auto-generated by experiments/single-qps-sweep-freq/local/generate_experiment.py",
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
    qps: float,
    gpu_id: int,
    frequency_jobs: list[FrequencyReplayJob],
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        f"REPO_ROOT={_shell_quote(str(REPO_ROOT.resolve()))}",
        f"DEFAULT_PORT_PROFILE_ID={default_port_profile}",
        "PORT_PROFILE_ID_VALUE=\"${1:-${PORT_PROFILE_ID:-${DEFAULT_PORT_PROFILE_ID}}}\"",
        f"GPU_ID={gpu_id}",
        "PYTHON_BIN=\"${PYTHON_BIN:-python3}\"",
        "SET_GPU_CORE_FREQ_BIN=\"${SET_GPU_CORE_FREQ_BIN:-set-gpu-core-freq}\"",
        "RESET_GPU_CORE_FREQ_BIN=\"${RESET_GPU_CORE_FREQ_BIN:-reset-gpu-core-freq}\"",
        "ZEUS_POWER_READER_BIN=\"${ZEUS_POWER_READER_BIN:-zeus-power-reader}\"",
        "ZEUSD_SOCKET_PATH_VALUE=\"${ZEUSD_SOCKET_PATH:-}\"",
        "",
        "POWER_READER_PID=\"\"",
        "GPU_FREQ_LOCK_ACTIVE=0",
        "",
        "stop_power_reader() {",
        "  if [[ -n \"${POWER_READER_PID}\" ]]; then",
        "    kill \"${POWER_READER_PID}\" >/dev/null 2>&1 || true",
        "    wait \"${POWER_READER_PID}\" 2>/dev/null || true",
        "    POWER_READER_PID=\"\"",
        "  fi",
        "}",
        "",
        "reset_gpu_core_if_needed() {",
        "  if [[ \"${GPU_FREQ_LOCK_ACTIVE}\" -eq 1 ]]; then",
        "    if ! \"${RESET_GPU_CORE_FREQ_BIN}\" --gpu-index \"${GPU_ID}\"; then",
        "      echo \"[single-qps-sweep-freq] warning: failed to reset GPU ${GPU_ID} core clocks\" >&2",
        "    fi",
        "    GPU_FREQ_LOCK_ACTIVE=0",
        "  fi",
        "}",
        "",
        "cleanup() {",
        "  local exit_code=\"$1\"",
        "  stop_power_reader",
        "  reset_gpu_core_if_needed",
        "  return \"${exit_code}\"",
        "}",
        "",
        "trap '__exit_code=$?; trap - EXIT INT TERM; cleanup \"${__exit_code}\"; exit \"${__exit_code}\"' EXIT",
        "trap 'echo \"[single-qps-sweep-freq] interrupted\" >&2; exit 130' INT TERM",
        "",
        "run_one_frequency() {",
        "  local min_mhz=\"$1\"",
        "  local max_mhz=\"$2\"",
        "  local freq_slug=\"$3\"",
        "  local replay_config_ref=\"$4\"",
        "  local replay_output_ref=\"$5\"",
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
        "  echo \"[single-qps-sweep-freq] freq=${min_mhz}:${max_mhz} slug=${freq_slug} output=${replay_output_ref} port_profile=${PORT_PROFILE_ID_VALUE}\"",
        "  \"${SET_GPU_CORE_FREQ_BIN}\" --gpu-index \"${GPU_ID}\" --min-mhz \"${min_mhz}\" --max-mhz \"${max_mhz}\"",
        "  GPU_FREQ_LOCK_ACTIVE=1",
        "",
        "  mkdir -p \"${power_output_dir}\"",
        "  if [[ -n \"${ZEUSD_SOCKET_PATH_VALUE}\" ]]; then",
        "    \"${ZEUS_POWER_READER_BIN}\" --output-dir \"${power_output_dir}\" --gpu-indices \"${GPU_ID}\" --socket-path \"${ZEUSD_SOCKET_PATH_VALUE}\" &",
        "  else",
        "    \"${ZEUS_POWER_READER_BIN}\" --output-dir \"${power_output_dir}\" --gpu-indices \"${GPU_ID}\" &",
        "  fi",
        "  POWER_READER_PID=\"$!\"",
        "",
        "  \"${PYTHON_BIN}\" -m replayer replay \\",
        "    --config \"${replay_config_path}\" \\",
        "    --port-profile-id \"${PORT_PROFILE_ID_VALUE}\"",
        "",
        "  stop_power_reader",
        "  reset_gpu_core_if_needed",
        "}",
        "",
        (
            f"echo \"[single-qps-sweep-freq] target_model={target_model} split={split} "
            f"qps={qps} gpu_id={gpu_id} freq_points={len(frequency_jobs)} "
            "port_profile=${PORT_PROFILE_ID_VALUE}\""
        ),
        "",
    ]

    for job in frequency_jobs:
        lines.append(
            "run_one_frequency "
            f"{job.min_mhz} "
            f"{job.max_mhz} "
            f"{_shell_quote(job.slug)} "
            f"{_shell_quote(job.replay_config_relpath)} "
            f"{_shell_quote(path_for_config(job.replay_output_dir))}"
        )
    lines.extend(
        [
            "",
            f"echo \"[single-qps-sweep-freq] completed {len(frequency_jobs)} frequency points\"",
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
            "Generate a local single-QPS frequency-sweep replay bundle by looking up an "
            "existing compiled plan and emitting one sweep entrypoint script."
        ),
    )
    parser.add_argument("--source-run-dir", required=True, help="Profiled source run directory under results/.")
    parser.add_argument("--poisson-seed", required=True, type=parse_non_negative_int, help="Poisson launch seed.")
    parser.add_argument("--randomize-seed", required=True, type=parse_non_negative_int, help="Replay randomization seed.")
    parser.add_argument("--qps", required=True, type=float, help="Poisson launch rate (requests per second).")
    parser.add_argument("--time-constraint-s", required=True, type=float, help="Replay time limit in seconds.")
    parser.add_argument("--target-model", required=True, help="Target model key from configs/model_config.toml.")
    parser.add_argument("--port-profile", "-P", required=True, type=int, help="Port profile ID for replay.")
    parser.add_argument(
        "--split",
        required=True,
        help="Plan split mode: full | exclude-unranked | top | rest (accepts typo alias exclued-unranked).",
    )
    parser.add_argument(
        "--split-two-group-metric",
        choices=sorted(SPLIT_METRIC_ALIASES),
        default="token_usage",
        help="Split grouping metric used for top/rest plan lookup (default: token_usage).",
    )
    parser.add_argument(
        "--freq-list",
        required=True,
        help="Semicolon-separated core clock ranges in '<min>:<max>' format, e.g. 345:1305;345:1005.",
    )
    parser.add_argument(
        "--gpu-id",
        required=True,
        type=parse_non_negative_int,
        help="GPU index targeted by set-gpu-core-freq/reset-gpu-core-freq.",
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
            "looks for replay-plan.token.rest.fp8.json."
        ),
    )
    parser.add_argument(
        "--replay-output-root",
        default=str(DEFAULT_REPLAY_OUTPUT_ROOT),
        help=(
            "Replay output root. Default appends "
            "<split>/<qps>/<timestamp>/<freq-slug> under results/replay/single-qps-sweep-freq/split/."
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
        qps = parse_positive_float(str(args.qps), field_name="--qps")
        time_constraint_s = parse_positive_float(
            str(args.time_constraint_s),
            field_name="--time-constraint-s",
        )
        if args.port_profile < 0:
            raise ValueError("--port-profile must be >= 0")

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
        frequency_ranges = parse_frequency_ranges(str(args.freq_list))
        gpu_id = int(args.gpu_id)

        output_config_root = Path(args.output_config_dir).expanduser().resolve()
        replay_output_root = Path(args.replay_output_root).expanduser()
        if not replay_output_root.is_absolute():
            replay_output_root = (REPO_ROOT / replay_output_root).resolve()
        else:
            replay_output_root = replay_output_root.resolve()

        batch_timestamp = build_utc_timestamp_slug()
        qps_slug = format_qps_slug(qps)
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

        replay_output_base_dir = (
            replay_output_root / split / qps_slug / batch_timestamp
        ).resolve()

        frequency_jobs: list[FrequencyReplayJob] = []
        for frequency_range in frequency_ranges:
            replay_output_dir = (replay_output_base_dir / frequency_range.slug).resolve()
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
            replay_config_path = (
                batch_dir / "freq" / frequency_range.slug / "replay.toml"
            ).resolve()
            write_replay_config(replay_config_path, replay_payload=replay_payload)
            replay_config_relpath = str(replay_config_path.relative_to(batch_dir))

            frequency_jobs.append(
                FrequencyReplayJob(
                    min_mhz=frequency_range.min_mhz,
                    max_mhz=frequency_range.max_mhz,
                    slug=frequency_range.slug,
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
            split=split,
            qps=qps,
            gpu_id=gpu_id,
            frequency_jobs=frequency_jobs,
        )

        frequency_summary = [
            {
                "min_mhz": job.min_mhz,
                "max_mhz": job.max_mhz,
                "slug": job.slug,
                "replay_config": str(job.replay_config_path),
                "replay_output_dir": path_for_config(job.replay_output_dir),
                "power_output_dir": path_for_config(job.power_output_dir),
            }
            for job in frequency_jobs
        ]
        summary = {
            "status": "ok",
            "batch_timestamp": batch_timestamp,
            "source_run_dir": str(source_run_dir),
            "target_model": target_model,
            "split": split,
            "split_two_group_metric": args.split_two_group_metric,
            "poisson_seed": int(args.poisson_seed),
            "randomize_seed": int(args.randomize_seed),
            "qps": qps,
            "qps_slug": qps_slug,
            "time_constraint_s": time_constraint_s,
            "port_profile": int(args.port_profile),
            "gpu_id": gpu_id,
            "freq_list": str(args.freq_list),
            "additional_suffix": additional_suffix,
            "output_config_root_dir": str(output_config_root),
            "output_batch_dir": str(batch_dir),
            "plan_lookup_only": True,
            "selected_source_plan": str(lookup_result.selected_plan_path),
            "related_source_plans": lookup_result.related_plan_paths,
            "selected_plan_model": related_plan_models.get("selected"),
            "related_plan_models": related_plan_models,
            "selected_plan_copy": str(selected_plan_copy_path),
            "replay_output_base_dir": path_for_config(replay_output_base_dir),
            "frequency_points": frequency_summary,
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
