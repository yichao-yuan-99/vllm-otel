#!/usr/bin/env python3
"""Generate one derived-plan max-concurrency sweep replay experiment bundle.

This helper does all of the following:
1) Validates the selected replay plan exists and is a derived single-trail plan.
2) Validates the plan's replay target matches the requested target model.
3) Materializes per-concurrency replay TOMLs + a runnable sweep script.
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
DEFAULT_OUTPUT_CONFIG_DIR = (
    REPO_ROOT / "experiments" / "sweep-maxcon-same-agent" / "generated"
)
DEFAULT_REPLAY_OUTPUT_ROOT = Path("results") / "replay" / "sweep-maxcon-same-agent"
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model_config.toml"

PLAN_MODEL_LINE_RE = re.compile(r'"model"\s*:\s*"((?:\\.|[^"\\])*)"')
PLAN_IS_DERIVED_LINE_RE = re.compile(r'"is_derived"\s*:\s*(true|false)')
PLAN_SINGLE_TRAIL_LINE_RE = re.compile(r'"single_trail"\s*:\s*"((?:\\.|[^"\\])*)"')
PLAN_SOURCE_JOB_DIR_LINE_RE = re.compile(r'"source_job_dir"\s*:\s*"((?:\\.|[^"\\])*)"')


@dataclass(frozen=True)
class TargetModelSpec:
    key: str
    served_model_name: str
    vllm_model_name: str


@dataclass(frozen=True)
class MaxConcurrencyReplayJob:
    max_concurrent: int
    max_concurrent_slug: str
    replay_config_relpath: str
    replay_config_path: Path
    replay_output_dir: Path


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


def parse_positive_int(value: str, *, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return parsed


def parse_positive_float(value: str, *, field_name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return parsed


def parse_max_concurrent_list(raw: str) -> list[int]:
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError(
            "--max-concurrent-list/--concurrency-list must be a non-empty "
            "comma-separated list"
        )

    values: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        try:
            parsed = int(token)
        except ValueError as exc:
            raise ValueError(
                "--max-concurrent-list/--concurrency-list contains non-integer "
                f"value: {token!r}"
            ) from exc
        if parsed <= 0:
            raise ValueError(
                "--max-concurrent-list/--concurrency-list values must be > 0"
            )
        if parsed in seen:
            raise ValueError(
                "--max-concurrent-list/--concurrency-list cannot contain duplicate "
                f"value: {token}"
            )
        seen.add(parsed)
        values.append(parsed)
    return values


def format_max_concurrent_slug(max_concurrent: int) -> str:
    return f"c{max_concurrent}"


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
        "# Auto-generated by experiments/sweep-maxcon-same-agent/generate_experiment.py",
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
    concurrency_jobs: list[MaxConcurrencyReplayJob],
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        f"DEFAULT_PORT_PROFILE_ID={default_port_profile}",
        "PORT_PROFILE_ID_VALUE=\"${1:-${PORT_PROFILE_ID:-${DEFAULT_PORT_PROFILE_ID}}}\"",
        f"SOURCE_TRAIL_NAME={_shell_quote(source_trail_name)}",
        "PYTHON_BIN=\"${PYTHON_BIN:-python3}\"",
        "",
        "run_one_maxcon() {",
        "  local max_concurrent_value=\"$1\"",
        "  local max_concurrent_slug=\"$2\"",
        "  local replay_config_ref=\"$3\"",
        "  local replay_output_ref=\"$4\"",
        "  local replay_config_path=\"\"",
        "",
        "  if [[ \"${replay_config_ref}\" = /* ]]; then",
        "    replay_config_path=\"${replay_config_ref}\"",
        "  else",
        "    replay_config_path=\"${SCRIPT_DIR}/${replay_config_ref}\"",
        "  fi",
        "",
        "  echo \"[sweep-maxcon-same-agent] trail=${SOURCE_TRAIL_NAME} max_concurrent=${max_concurrent_value} slug=${max_concurrent_slug} output=${replay_output_ref} port_profile=${PORT_PROFILE_ID_VALUE}\"",
        "  \"${PYTHON_BIN}\" -m replayer replay \\",
        "    --config \"${replay_config_path}\" \\",
        "    --port-profile-id \"${PORT_PROFILE_ID_VALUE}\"",
        "}",
        "",
        (
            f"echo \"[sweep-maxcon-same-agent] target_model={target_model} "
            "trail=${SOURCE_TRAIL_NAME} "
            f"concurrency_points={len(concurrency_jobs)} "
            "port_profile=${PORT_PROFILE_ID_VALUE}\""
        ),
        "",
    ]

    for job in concurrency_jobs:
        lines.append(
            "run_one_maxcon "
            f"{_shell_quote(str(job.max_concurrent))} "
            f"{_shell_quote(job.max_concurrent_slug)} "
            f"{_shell_quote(job.replay_config_relpath)} "
            f"{_shell_quote(path_for_config(job.replay_output_dir))}"
        )

    lines.extend(
        [
            "",
            f"echo \"[sweep-maxcon-same-agent] completed {len(concurrency_jobs)} concurrency points\"",
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
            "Generate a max-concurrency sweep replay bundle from one precompiled "
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
    parser.add_argument(
        "--max-concurrent-list",
        "--concurrency-list",
        dest="max_concurrent_list",
        required=True,
        help="Comma-separated replay max_concurrent values, e.g. 1,2,4,8,16",
    )
    parser.add_argument(
        "--randomize-seed",
        required=True,
        type=parse_non_negative_int,
        help="Replay randomization seed.",
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
        "--output-config-dir",
        default=str(DEFAULT_OUTPUT_CONFIG_DIR),
        help=f"Generated bundle root (default: {DEFAULT_OUTPUT_CONFIG_DIR}).",
    )
    parser.add_argument(
        "--replay-output-root",
        default=str(DEFAULT_REPLAY_OUTPUT_ROOT),
        help=(
            "Replay output root. Default appends "
            "<dataset-lineage>/trail/<safe-source-trail>/c<max-concurrent>/<timestamp> "
            "under results/replay/sweep-maxcon-same-agent/. If dataset-lineage "
            "cannot be derived from the plan, the path starts at "
            "trail/<safe-source-trail>/..."
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

        max_concurrency_values = parse_max_concurrent_list(str(args.max_concurrent_list))
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

        output_config_root = Path(args.output_config_dir).expanduser().resolve()
        replay_output_root = Path(args.replay_output_root).expanduser()
        if not replay_output_root.is_absolute():
            replay_output_root = (REPO_ROOT / replay_output_root).resolve()
        else:
            replay_output_root = replay_output_root.resolve()

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
            replay_output_root_base / "trail" / source_trail_slug
        ).resolve()

        concurrency_jobs: list[MaxConcurrencyReplayJob] = []
        for max_concurrent in max_concurrency_values:
            max_concurrent_slug = format_max_concurrent_slug(max_concurrent)
            replay_output_dir = (
                replay_output_root_base / max_concurrent_slug / batch_timestamp
            ).resolve()
            replay_payload = {
                "plan": path_for_config(source_plan_copy_path),
                "output_dir": path_for_config(replay_output_dir),
                "randomize_seed": int(args.randomize_seed),
                "time_constraint_s": time_constraint_s,
                "port_profile_id": int(args.port_profile),
                "launch_policy_override": {
                    "max_concurrent": max_concurrent,
                },
            }
            replay_config_path = (batch_dir / max_concurrent_slug / "replay.toml").resolve()
            write_replay_config(replay_config_path, replay_payload=replay_payload)
            replay_config_relpath = str(replay_config_path.relative_to(batch_dir))

            concurrency_jobs.append(
                MaxConcurrencyReplayJob(
                    max_concurrent=max_concurrent,
                    max_concurrent_slug=max_concurrent_slug,
                    replay_config_relpath=replay_config_relpath,
                    replay_config_path=replay_config_path,
                    replay_output_dir=replay_output_dir,
                )
            )

        run_script_path = (batch_dir / "run_replay.sh").resolve()
        write_run_script(
            run_script_path,
            default_port_profile=int(args.port_profile),
            target_model=target_model,
            source_trail_name=source_trail_name,
            concurrency_jobs=concurrency_jobs,
        )

        concurrency_summary = [
            {
                "max_concurrent": job.max_concurrent,
                "max_concurrent_slug": job.max_concurrent_slug,
                "replay_config": str(job.replay_config_path),
                "replay_output_dir": path_for_config(job.replay_output_dir),
            }
            for job in concurrency_jobs
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
            "randomize_seed": int(args.randomize_seed),
            "max_concurrency_list": max_concurrency_values,
            "time_constraint_s": time_constraint_s,
            "port_profile": int(args.port_profile),
            "output_config_root_dir": str(output_config_root),
            "output_batch_dir": str(batch_dir),
            "replay_output_root_base_dir": path_for_config(replay_output_root_base),
            "max_concurrency_points": concurrency_summary,
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
