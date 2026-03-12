#!/usr/bin/env python3
"""Generate replay config files for sweep-concurrency experiments.

This script creates a directory of replayer TOML config files. Each generated
config is intended to be consumed by:

    python -m orchestrator --job-type replay --jobs-dir <generated_dir> ...
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
VALID_TOML_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_concurrency_list(raw: str) -> list[int]:
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--concurrency-list must be a non-empty comma-separated list")

    values: list[int] = []
    for token in tokens:
        try:
            parsed = int(token)
        except ValueError as exc:
            raise ValueError(
                f"--concurrency-list contains non-integer value: {token!r}"
            ) from exc
        if parsed <= 0:
            raise ValueError("--concurrency-list values must be > 0")
        values.append(parsed)

    if len(set(values)) != len(values):
        raise ValueError("--concurrency-list cannot contain duplicate values")
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
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def derive_replay_output_base(
    source_run_dir: Path,
    *,
    replay_root_dir: Path | None = None,
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
    return replay_root / Path(*lineage_without_run_dir) / "sweep-concurrency"


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


def build_replay_config_payload(
    *,
    plan_path: Path,
    output_dir: Path,
    num_tasks: int,
    concurrency: int,
    port_profile_id: int | None,
    vllm_log_interval_s: float | None,
    vllm_log_timeout_s: float | None,
    launch_policy_override_json: dict[str, Any],
    extra_replay_json: dict[str, Any],
) -> dict[str, Any]:
    replay_payload = copy.deepcopy(extra_replay_json)

    replay_payload["plan"] = path_for_config(plan_path)
    replay_payload["output_dir"] = path_for_config(output_dir)
    replay_payload["num_tasks"] = num_tasks

    if port_profile_id is not None:
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
    merged_launch_policy_override["max_concurrent"] = concurrency
    replay_payload["launch_policy_override"] = merged_launch_policy_override
    return replay_payload


def write_replay_config(path: Path, *, replay_payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Auto-generated by experiments/sweep-concurrency/generate_replay_configs.py",
        "",
    ]
    append_toml_table(lines, "replay", replay_payload)
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_replay_configs.py",
        description="Generate replay config files for sweep-concurrency runs.",
    )
    parser.add_argument(
        "--source-run-dir",
        required=True,
        help="Source run result directory that contains replay-plan.json",
    )
    parser.add_argument(
        "--concurrency-list",
        required=True,
        help="Comma-separated target max_concurrent values, e.g. '1,2,3,5,10'",
    )
    parser.add_argument(
        "--num-tasks",
        required=True,
        type=parse_positive_int,
        help="Number of tasks to replay in each generated config",
    )
    parser.add_argument(
        "--output-config-dir",
        required=True,
        help="Directory where generated replay TOML configs will be written",
    )
    parser.add_argument(
        "--plan-path",
        default=None,
        help="Optional explicit replay plan path (default: <source-run-dir>/replay-plan.json)",
    )
    parser.add_argument(
        "--replay-root-dir",
        default=None,
        help=(
            "Optional root directory for generated replay outputs. "
            "Source lineage and sweep-concurrency/<n> are appended under this root."
        ),
    )
    parser.add_argument(
        "--port-profile-id",
        type=int,
        default=None,
        help="Optional replay port profile id forwarded to all generated configs",
    )
    parser.add_argument(
        "--vllm-log-interval-s",
        type=float,
        default=None,
        help="Optional vLLM log interval forwarded to all generated configs",
    )
    parser.add_argument(
        "--vllm-log-timeout-s",
        type=float,
        default=None,
        help="Optional vLLM log timeout forwarded to all generated configs",
    )
    parser.add_argument(
        "--launch-policy-override-json",
        default=None,
        help=(
            "Optional JSON object merged into replay.launch_policy_override for all "
            "configs before setting max_concurrent"
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

    try:
        source_run_dir = Path(args.source_run_dir).expanduser().resolve()
        if not source_run_dir.exists() or not source_run_dir.is_dir():
            raise ValueError(f"invalid --source-run-dir: {source_run_dir}")

        output_config_dir = Path(args.output_config_dir).expanduser().resolve()
        concurrency_values = parse_concurrency_list(args.concurrency_list)

        if args.plan_path:
            plan_path = Path(args.plan_path).expanduser().resolve()
        else:
            plan_path = (source_run_dir / "replay-plan.json").resolve()
        if not plan_path.exists() or not plan_path.is_file():
            raise ValueError(f"replay plan does not exist: {plan_path}")
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

        replay_output_base = derive_replay_output_base(
            source_run_dir,
            replay_root_dir=replay_root_dir,
        )

        generated_configs: list[dict[str, Any]] = []
        for concurrency in concurrency_values:
            replay_output_dir = (REPO_ROOT / replay_output_base / str(concurrency)).resolve()
            replay_payload = build_replay_config_payload(
                plan_path=plan_path,
                output_dir=replay_output_dir,
                num_tasks=args.num_tasks,
                concurrency=concurrency,
                port_profile_id=args.port_profile_id,
                vllm_log_interval_s=args.vllm_log_interval_s,
                vllm_log_timeout_s=args.vllm_log_timeout_s,
                launch_policy_override_json=launch_policy_override_json,
                extra_replay_json=extra_replay_json,
            )

            config_path = output_config_dir / f"replay.c{concurrency}.toml"
            write_replay_config(config_path, replay_payload=replay_payload)
            generated_configs.append(
                {
                    "concurrency": concurrency,
                    "config_path": str(config_path),
                    "replay_output_dir": path_for_config(replay_output_dir),
                }
            )

        summary = {
            "status": "ok",
            "source_run_dir": str(source_run_dir),
            "plan_path": str(plan_path),
            "replay_root_dir": str(replay_root_dir) if replay_root_dir is not None else None,
            "output_config_dir": str(output_config_dir),
            "num_tasks": args.num_tasks,
            "concurrency_list": concurrency_values,
            "generated_configs": generated_configs,
        }
        output_config_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_config_dir / "manifest.json"
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
