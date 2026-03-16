#!/usr/bin/env python3
"""Generate grouped split-plan sweep-qps bundles using sbatch-orchestrator.

For each split group (top/rest) and each target QPS, this script creates:

- replay config TOML
- per-job script executed by sbatch-orchestrator

At the batch level, it also creates:

- job-list.txt (absolute script paths)
- manifest.json
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_CONFIG_DIR = (
    REPO_ROOT / "experiments" / "sweep-qps-local" / "group" / "split" / "generated"
)
DEFAULT_SUBMIT_WRAPPER_REL_PATH = Path("sbatch-orchestrator") / "submit-start-group.sh"
VALID_TOML_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SPLIT_GROUPS = ("top", "rest")
SPLIT_PLAN_METRIC_ALIASES = ("token", "context")


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
    for token in tokens:
        try:
            parsed = float(token)
        except ValueError as exc:
            raise ValueError(f"--qps-list contains non-numeric value: {token!r}") from exc
        if parsed <= 0:
            raise ValueError("--qps-list values must be > 0")
        values.append(parsed)

    if len(set(values)) != len(values):
        raise ValueError("--qps-list cannot contain duplicate values")
    return values


def parse_optional_object_json(raw: str | None, *, field_name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    stripped = raw.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {field_name} JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must decode to a JSON object")
    return parsed


def merge_dict_overlay(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict_overlay(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def path_for_config(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def derive_replay_output_base(
    source_run_dir: Path,
    *,
    replay_root_dir: Path | None = None,
    batch_timestamp: str,
) -> Path:
    source_resolved = source_run_dir.expanduser().resolve()
    try:
        source_relative = source_resolved.relative_to(RESULTS_ROOT)
    except ValueError as exc:
        raise ValueError(
            "source_run_dir must be under top-level results/, for example: "
            "results/qwen3-coder-30b/livecodebench/mini-swe-agent/<run-dir>"
        ) from exc

    if len(source_relative.parts) < 2:
        raise ValueError(
            "source_run_dir under results/ must include at least one parent plus run dir"
        )
    lineage_without_run_dir = source_relative.parts[:-1]
    if replay_root_dir is None:
        replay_root = Path("results") / "replay"
    else:
        replay_root = replay_root_dir.expanduser().resolve()
    return (
        replay_root
        / batch_timestamp
        / Path(*lineage_without_run_dir)
        / "sweep-qps-local"
        / "group"
        / "split"
    )


def derive_replay_timestamp_output_dir(
    *,
    replay_root_dir: Path | None,
    batch_timestamp: str,
) -> Path:
    if replay_root_dir is None:
        replay_root = (REPO_ROOT / "results" / "replay").resolve()
    else:
        replay_root = replay_root_dir.expanduser().resolve()
    return (replay_root / batch_timestamp).resolve()


def format_toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, list):
        return "[" + ", ".join(format_toml_scalar(item) for item in value) + "]"
    raise ValueError(f"Unsupported TOML scalar type: {type(value)!r}")


def validate_toml_key(key: str) -> None:
    if not VALID_TOML_KEY_RE.match(key):
        raise ValueError(
            f"Unsupported TOML key {key!r}; only letters, digits, '_' and '-' are allowed"
        )


def append_toml_table(lines: list[str], table_name: str, payload: dict[str, Any]) -> None:
    lines.append(f"[{table_name}]")
    nested_items: list[tuple[str, dict[str, Any]]] = []
    for key in sorted(payload.keys()):
        validate_toml_key(key)
        value = payload[key]
        if value is None:
            continue
        if isinstance(value, dict):
            nested_items.append((key, value))
            continue
        lines.append(f"{key} = {format_toml_scalar(value)}")

    for key, nested in nested_items:
        lines.append("")
        append_toml_table(lines, f"{table_name}.{key}", nested)


def format_qps_for_slug(qps: float) -> str:
    text = format(qps, ".12g")
    text = text.replace("-", "m")
    text = text.replace(".", "_")
    text = text.replace("+", "")
    return f"qps{text}"


def with_plan_name_suffix(plan_path: Path, suffix: str) -> Path:
    if not suffix:
        raise ValueError("Plan suffix cannot be empty")
    file_name = plan_path.name
    dot_index = file_name.rfind(".")
    if dot_index <= 0:
        suffixed_name = f"{file_name}.{suffix}"
    else:
        suffixed_name = f"{file_name[:dot_index]}.{suffix}{file_name[dot_index:]}"
    return (plan_path.parent / suffixed_name).resolve()


def derive_metric_split_plan_paths(
    base_plan_path: Path,
    *,
    metric_alias: str,
) -> dict[str, Path]:
    return {
        "top": with_plan_name_suffix(base_plan_path, f"{metric_alias}.top"),
        "rest": with_plan_name_suffix(base_plan_path, f"{metric_alias}.rest"),
    }


def derive_default_split_plan_paths(base_plan_path: Path) -> dict[str, Path]:
    for metric_alias in SPLIT_PLAN_METRIC_ALIASES:
        candidate_paths = derive_metric_split_plan_paths(
            base_plan_path,
            metric_alias=metric_alias,
        )
        if all(path.exists() and path.is_file() for path in candidate_paths.values()):
            return candidate_paths

    legacy_paths = {
        "top": with_plan_name_suffix(base_plan_path, "top"),
        "rest": with_plan_name_suffix(base_plan_path, "rest"),
    }
    if all(path.exists() and path.is_file() for path in legacy_paths.values()):
        return legacy_paths

    return derive_metric_split_plan_paths(base_plan_path, metric_alias="token")


def build_utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_generation_command_raw(argv: list[str] | None) -> str:
    if argv is None:
        original = getattr(sys, "orig_argv", None)
        if isinstance(original, list) and original:
            return shlex.join([str(token) for token in original])
        return shlex.join([str(sys.executable), *[str(token) for token in sys.argv]])
    return shlex.join(
        [
            str(sys.executable),
            str(Path(__file__).resolve()),
            *[str(token) for token in argv],
        ]
    )


def build_replay_config_payload(
    *,
    plan_path: Path,
    output_dir: Path,
    qps: float,
    poisson_seed: int,
    randomize_seed: int,
    time_constraint_s: float,
    port_profile_id: int,
    vllm_log_interval_s: float | None,
    vllm_log_timeout_s: float | None,
    launch_policy_override_json: dict[str, Any],
    extra_replay_json: dict[str, Any],
) -> dict[str, Any]:
    replay_payload = copy.deepcopy(extra_replay_json)
    if "num_tasks" in replay_payload:
        raise ValueError(
            "extra replay payload cannot define 'num_tasks' for sweep-qps; "
            "time-constrained replay is unbounded by task count"
        )

    replay_payload["plan"] = path_for_config(plan_path)
    replay_payload["output_dir"] = path_for_config(output_dir)
    replay_payload["randomize_seed"] = randomize_seed
    replay_payload["time_constraint_s"] = time_constraint_s
    replay_payload["port_profile_id"] = port_profile_id

    if vllm_log_interval_s is not None:
        replay_payload["vllm_log_interval_s"] = vllm_log_interval_s
    if vllm_log_timeout_s is not None:
        replay_payload["vllm_log_timeout_s"] = vllm_log_timeout_s

    replay_launch_policy_override = replay_payload.get("launch_policy_override")
    if replay_launch_policy_override is None:
        replay_launch_policy_override = {}
    if not isinstance(replay_launch_policy_override, dict):
        raise ValueError("extra replay payload field 'launch_policy_override' must be an object")

    merged_launch_policy_override = merge_dict_overlay(
        replay_launch_policy_override,
        launch_policy_override_json,
    )

    pattern_args = merged_launch_policy_override.get("pattern_args")
    if pattern_args is None:
        pattern_args_payload: dict[str, Any] = {}
    elif isinstance(pattern_args, dict):
        pattern_args_payload = copy.deepcopy(pattern_args)
    else:
        raise ValueError("replay.launch_policy_override.pattern_args must be an object")

    pattern_args_payload["rate"] = qps
    merged_launch_policy_override["pattern_args"] = pattern_args_payload
    merged_launch_policy_override["pattern"] = {"name": "poisson"}
    merged_launch_policy_override["seed"] = poisson_seed

    replay_payload["launch_policy_override"] = merged_launch_policy_override
    return replay_payload


def write_replay_config(path: Path, *, replay_payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Auto-generated by experiments/sweep-qps-local/group/split/generate_replay_configs.py",
        "",
    ]
    append_toml_table(lines, "replay", replay_payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_orchestrator_job_script(
    *,
    path: Path,
    default_port_profile_id: int,
) -> None:
    content = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "if [[ \"$#\" -lt 1 ]]; then\n"
        "  echo \"usage: $0 <assigned_vllm_port>\" >&2\n"
        "  exit 2\n"
        "fi\n\n"
        "ASSIGNED_VLLM_PORT=\"$1\"\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        f"DEFAULT_PORT_PROFILE_ID={default_port_profile_id}\n"
        "PORT_PROFILE_ID_VALUE=\"${PORT_PROFILE_ID:-${DEFAULT_PORT_PROFILE_ID}}\"\n"
        "PYTHON_BIN=\"${PYTHON_BIN:-python3}\"\n\n"
        "echo \"[group-split-job] profile=${PORT_PROFILE_ID_VALUE} assigned_vllm_port=${ASSIGNED_VLLM_PORT}\"\n\n"
        "\"${PYTHON_BIN}\" -m replayer replay \\\n"
        "  --config \"${SCRIPT_DIR}/replay.toml\" \\\n"
        "  --port-profile-id \"${PORT_PROFILE_ID_VALUE}\"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o750)


def write_job_list(path: Path, *, job_script_paths: list[Path]) -> None:
    lines = [
        "# Auto-generated by experiments/sweep-qps-local/group/split/generate_replay_configs.py",
        "# One job script path per line (absolute path).",
    ]
    for job_script_path in job_script_paths:
        lines.append(shlex.quote(str(job_script_path.resolve())))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_generated_batch_dir_to_replay_timestamp(
    *,
    source_generated_batch_dir: Path,
    replay_timestamp_output_dir: Path,
) -> Path:
    generated_copy_dir = (replay_timestamp_output_dir / "generated").resolve()

    try:
        generated_copy_dir.relative_to(source_generated_batch_dir.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(
            "generated copy destination must not be inside source generated batch dir: "
            f"{generated_copy_dir}"
        )

    generated_copy_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_generated_batch_dir.resolve(),
        generated_copy_dir,
        dirs_exist_ok=True,
    )
    return generated_copy_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_replay_configs.py",
        description=(
            "Generate grouped split-plan sweep-qps bundles for sbatch-orchestrator "
            "(replay configs + job scripts + batch-level manifest)."
        ),
    )
    parser.add_argument(
        "--source-run-dir",
        required=True,
        help=(
            "Source run result directory that contains split replay plans "
            "(default discovery prefers replay-plan.token.top.json / "
            "replay-plan.token.rest.json, then context, then legacy .top/.rest)."
        ),
    )
    parser.add_argument(
        "--qps-list",
        required=True,
        help="Comma-separated Poisson rates (requests per second), e.g. '0.05,0.1,0.2'",
    )
    parser.add_argument(
        "--poisson-seed",
        required=True,
        type=parse_non_negative_int,
        help="Seed for Poisson inter-arrival sampling (launch-policy seed)",
    )
    parser.add_argument(
        "--randomize-seed",
        required=True,
        type=parse_non_negative_int,
        help="Seed for worker-order randomization (replay.randomize_seed)",
    )
    parser.add_argument(
        "--time-constraint-s",
        required=True,
        type=float,
        help="Replay wall-time limit in seconds (replay.time_constraint_s)",
    )
    parser.add_argument(
        "--output-config-dir",
        default=str(DEFAULT_OUTPUT_CONFIG_DIR),
        help=(
            "Root directory for generated experiment bundles "
            f"(default: {DEFAULT_OUTPUT_CONFIG_DIR})"
        ),
    )
    parser.add_argument(
        "--plan-path",
        default=None,
        help=(
            "Optional base replay plan path used to derive split plan paths. "
            "If omitted, uses <source-run-dir>/replay-plan.json and derives "
            "metric-qualified split plans (for example replay-plan.token.top.json)."
        ),
    )
    parser.add_argument(
        "--top-plan-path",
        default=None,
        help=(
            "Optional explicit top-group plan path override "
            "(default: discovered from --plan-path metric-qualified names)."
        ),
    )
    parser.add_argument(
        "--rest-plan-path",
        default=None,
        help=(
            "Optional explicit rest-group plan path override "
            "(default: discovered from --plan-path metric-qualified names)."
        ),
    )
    parser.add_argument(
        "--replay-root-dir",
        default=None,
        help=(
            "Optional root directory for replay outputs. "
            "<utc-timestamp>/source-lineage/sweep-qps-local/group/split/<group>/<qps> "
            "is appended under this root."
        ),
    )
    parser.add_argument(
        "--vllm-log-interval-s",
        type=float,
        default=None,
        help="Optional vLLM log interval forwarded to generated replay configs",
    )
    parser.add_argument(
        "--vllm-log-timeout-s",
        type=float,
        default=None,
        help="Optional vLLM log timeout forwarded to generated replay configs",
    )
    parser.add_argument(
        "--launch-policy-override-json",
        default=None,
        help=(
            "Optional JSON object merged into replay.launch_policy_override for all "
            "configs before enforcing Poisson fields"
        ),
    )
    parser.add_argument(
        "--extra-replay-json",
        default=None,
        help=(
            "Optional JSON object merged into [replay] for all generated configs. "
            "Useful to forward additional replayer options."
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

        output_config_root_dir = Path(args.output_config_dir).expanduser().resolve()
        submit_wrapper_path = (REPO_ROOT / DEFAULT_SUBMIT_WRAPPER_REL_PATH).resolve()
        if not submit_wrapper_path.exists() or not submit_wrapper_path.is_file():
            raise ValueError(f"missing submit wrapper: {submit_wrapper_path}")

        qps_values = parse_qps_list(args.qps_list)
        default_port_profile_id = 0

        time_constraint_s = parse_positive_float(
            str(args.time_constraint_s),
            field_name="--time-constraint-s",
        )

        if args.plan_path:
            base_plan_path = Path(args.plan_path).expanduser().resolve()
        else:
            base_plan_path = (source_run_dir / "replay-plan.json").resolve()

        default_split_plan_paths = derive_default_split_plan_paths(base_plan_path)
        top_plan_path = (
            Path(args.top_plan_path).expanduser().resolve()
            if args.top_plan_path
            else default_split_plan_paths["top"]
        )
        rest_plan_path = (
            Path(args.rest_plan_path).expanduser().resolve()
            if args.rest_plan_path
            else default_split_plan_paths["rest"]
        )
        split_plan_paths: dict[str, Path] = {
            "top": top_plan_path,
            "rest": rest_plan_path,
        }
        for split_group_name, split_plan_path in split_plan_paths.items():
            if not split_plan_path.exists() or not split_plan_path.is_file():
                raise ValueError(
                    f"{split_group_name} replay plan does not exist: {split_plan_path}"
                )

        replay_root_dir = (
            Path(args.replay_root_dir).expanduser().resolve()
            if args.replay_root_dir is not None
            else None
        )

        if args.vllm_log_interval_s is not None and args.vllm_log_interval_s <= 0:
            raise ValueError("--vllm-log-interval-s must be > 0")
        if args.vllm_log_timeout_s is not None and args.vllm_log_timeout_s <= 0:
            raise ValueError("--vllm-log-timeout-s must be > 0")

        launch_policy_override_json = parse_optional_object_json(
            args.launch_policy_override_json,
            field_name="--launch-policy-override-json",
        )
        extra_replay_json = parse_optional_object_json(
            args.extra_replay_json,
            field_name="--extra-replay-json",
        )

        batch_timestamp = build_utc_timestamp_slug()
        replay_output_base = derive_replay_output_base(
            source_run_dir,
            replay_root_dir=replay_root_dir,
            batch_timestamp=batch_timestamp,
        )
        replay_timestamp_output_dir = derive_replay_timestamp_output_dir(
            replay_root_dir=replay_root_dir,
            batch_timestamp=batch_timestamp,
        )

        batch_dir = (output_config_root_dir / batch_timestamp).resolve()
        replay_batch_output_base_resolved = (REPO_ROOT / replay_output_base).resolve()
        batch_dir.mkdir(parents=True, exist_ok=True)

        generated_experiments: list[dict[str, Any]] = []
        generated_experiments_by_group: dict[str, list[dict[str, Any]]] = {
            "top": [],
            "rest": [],
        }
        job_script_paths: list[Path] = []

        for split_group_name in SPLIT_GROUPS:
            split_plan_path = split_plan_paths[split_group_name]
            group_dir = (batch_dir / split_group_name).resolve()
            group_dir.mkdir(parents=True, exist_ok=True)
            replay_group_output_base = (
                replay_batch_output_base_resolved / split_group_name
            ).resolve()

            for qps in qps_values:
                qps_slug = format_qps_for_slug(qps)
                experiment_dir = (group_dir / qps_slug).resolve()
                experiment_dir.mkdir(parents=True, exist_ok=True)

                replay_output_dir = (replay_group_output_base / qps_slug).resolve()
                replay_output_dir.mkdir(parents=True, exist_ok=True)
                replay_payload = build_replay_config_payload(
                    plan_path=split_plan_path,
                    output_dir=replay_output_dir,
                    qps=qps,
                    poisson_seed=args.poisson_seed,
                    randomize_seed=args.randomize_seed,
                    time_constraint_s=time_constraint_s,
                    port_profile_id=default_port_profile_id,
                    vllm_log_interval_s=args.vllm_log_interval_s,
                    vllm_log_timeout_s=args.vllm_log_timeout_s,
                    launch_policy_override_json=launch_policy_override_json,
                    extra_replay_json=extra_replay_json,
                )
                replay_config_path = (experiment_dir / "replay.toml").resolve()
                write_replay_config(replay_config_path, replay_payload=replay_payload)

                job_script_path = (experiment_dir / "run_orchestrated_replay.sh").resolve()
                write_orchestrator_job_script(
                    path=job_script_path,
                    default_port_profile_id=default_port_profile_id,
                )
                job_script_paths.append(job_script_path)

                experiment_record = {
                    "group": split_group_name,
                    "plan_path": str(split_plan_path),
                    "qps": qps,
                    "qps_slug": qps_slug,
                    "experiment_dir": str(experiment_dir),
                    "replay_config": str(replay_config_path),
                    "job_script": str(job_script_path),
                    "job_list_entry": shlex.quote(str(job_script_path)),
                    "replay_output_dir": path_for_config(replay_output_dir),
                }
                generated_experiments.append(experiment_record)
                generated_experiments_by_group[split_group_name].append(experiment_record)

        job_list_path = (batch_dir / "job-list.txt").resolve()
        write_job_list(job_list_path, job_script_paths=job_script_paths)

        submit_command = (
            f"bash {path_for_config(submit_wrapper_path)} "
            f"--job-list {path_for_config(job_list_path)}"
        )
        summary = {
            "status": "ok",
            "batch_timestamp": batch_timestamp,
            "source_run_dir": str(source_run_dir),
            "plan_paths": {
                "top": str(top_plan_path),
                "rest": str(rest_plan_path),
            },
            "default_port_profile": default_port_profile_id,
            "output_config_root_dir": str(output_config_root_dir),
            "output_batch_dir": str(batch_dir),
            "replay_root_dir": str(replay_root_dir) if replay_root_dir is not None else None,
            "replay_output_batch_dir": path_for_config(replay_batch_output_base_resolved),
            "generated_batch_copy_dir": path_for_config(
                replay_timestamp_output_dir / "generated"
            ),
            "poisson_seed": args.poisson_seed,
            "randomize_seed": args.randomize_seed,
            "time_constraint_s": time_constraint_s,
            "qps_list": qps_values,
            "split_groups": list(SPLIT_GROUPS),
            "generation_command_raw": generation_command_raw,
            "job_list": str(job_list_path),
            "orchestrator_summary_default": (
                "sbatch-orchestrator/logs/<utc-timestamp>/orchestrator-summary.json"
            ),
            "submit_command": submit_command,
            "generated_experiments": generated_experiments,
            "generated_experiments_by_group": generated_experiments_by_group,
        }
        manifest_path = (batch_dir / "manifest.json").resolve()
        manifest_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        copy_generated_batch_dir_to_replay_timestamp(
            source_generated_batch_dir=batch_dir,
            replay_timestamp_output_dir=replay_timestamp_output_dir,
        )
        summary["manifest_path"] = str(manifest_path)
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
