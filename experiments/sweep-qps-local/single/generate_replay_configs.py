#!/usr/bin/env python3
"""Generate local-mode single-profile sweep-qps experiment bundles.

For each target QPS, this script creates one experiment subdirectory containing:

- replay config TOML
- local-mode replay script used by render-sbatch
- rendered sbatch script (no submission)
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_CONFIG_DIR = REPO_ROOT / "experiments" / "sweep-qps-local" / "single" / "generated"
DEFAULT_SERVER_CONFIG_PATH = (
    REPO_ROOT / "servers" / "servers-amdhpc" / "server_config.toml"
)
VALID_TOML_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


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
        / "single"
    )


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


def build_utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
        "# Auto-generated by experiments/sweep-qps-local/single/generate_replay_configs.py",
        "",
    ]
    append_toml_table(lines, "replay", replay_payload)
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_local_mode_script(
    *,
    path: Path,
    port_profile_id: int,
) -> None:
    content = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        f"DEFAULT_PORT_PROFILE_ID={port_profile_id}\n"
        "PORT_PROFILE_ID_VALUE=\"${PORT_PROFILE_ID:-${DEFAULT_PORT_PROFILE_ID}}\"\n"
        "PYTHON_BIN=\"${PYTHON_BIN:-python3}\"\n\n"
        "\"${PYTHON_BIN}\" -m replayer replay \\\n"
        "  --config \"${SCRIPT_DIR}/replay.toml\" \\\n"
        "  --port-profile-id \"${PORT_PROFILE_ID_VALUE}\"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o750)


def write_gateway_config(
    path: Path,
    *,
    port_profile_id: int,
    output_root: Path,
) -> None:
    lines: list[str] = [
        "# Auto-generated by experiments/sweep-qps-local/single/generate_replay_configs.py",
        "",
        "schema_version = 1",
        "",
        "[run]",
        f"port_profile_id = {port_profile_id}",
        f"output_root = {json.dumps(str(output_root.resolve()), ensure_ascii=True)}",
        "",
        "[telemetry]",
        'service_name = "vllm-gateway"',
        "otlp_traces_insecure = true",
        "",
        "[gateway]",
        'artifact_compression = "none"',
        "job_end_trace_wait_seconds = 10",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def rewrite_sbatch_gateway_default(
    sbatch_path: Path,
    *,
    gateway_config_path: Path,
) -> None:
    content = sbatch_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^GATEWAY_CONFIG_DEFAULT=.*$", flags=re.MULTILINE)
    replacement = f"GATEWAY_CONFIG_DEFAULT={gateway_config_path.resolve()}"
    updated, count = pattern.subn(replacement, content, count=1)
    if count != 1:
        raise RuntimeError(
            f"failed to patch GATEWAY_CONFIG_DEFAULT in rendered sbatch: {sbatch_path}"
        )
    sbatch_path.write_text(updated, encoding="utf-8")


def rewrite_sbatch_log_paths(
    sbatch_path: Path,
    *,
    log_dir: Path,
) -> None:
    content = sbatch_path.read_text(encoding="utf-8")
    replacements = [
        (
            re.compile(r"^#SBATCH --output=.*$", flags=re.MULTILINE),
            f"#SBATCH --output={log_dir.resolve()}/slurm.%j.out",
            "SBATCH --output",
        ),
        (
            re.compile(r"^#SBATCH --error=.*$", flags=re.MULTILINE),
            f"#SBATCH --error={log_dir.resolve()}/slurm.%j.err",
            "SBATCH --error",
        ),
        (
            re.compile(r"^JOB_LOG_DIR=.*$", flags=re.MULTILINE),
            f'JOB_LOG_DIR={json.dumps(str(log_dir.resolve()), ensure_ascii=True)}',
            "JOB_LOG_DIR",
        ),
    ]

    updated = content
    for pattern, replacement, label in replacements:
        updated, count = pattern.subn(replacement, updated, count=1)
        if count != 1:
            raise RuntimeError(
                f"failed to patch {label} in rendered sbatch: {sbatch_path}"
            )

    sbatch_path.write_text(updated, encoding="utf-8")


def write_submit_all_script(path: Path, *, qps_slugs: list[str]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        "",
    ]
    for qps_slug in qps_slugs:
        lines.append(f"sbatch \"${{SCRIPT_DIR}}/{qps_slug}/sbatch.sh\"")
    lines.append("")
    content = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o750)


def build_control_plane(server_config_path: Path) -> Any:
    module_dir = REPO_ROOT / "servers" / "servers-amdhpc"
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)
    try:
        import control_plane as control_plane_module  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to import control_plane from {module_dir}") from exc
    return control_plane_module.ControlPlane(server_config_path, archive_previous_artifacts=False)


def render_local_mode_sbatch(
    *,
    control_plane: Any,
    port_profile_id: int,
    partition: str,
    model: str,
    local_mode_script_path: Path,
    check_port_availability: bool,
    lmcache_max_local_cpu_size: str | None,
) -> Path:
    result = control_plane.render_start_sbatch(
        port_profile_id=port_profile_id,
        partition=partition,
        model=model,
        extra_env={},
        lmcache_max_local_cpu_size=lmcache_max_local_cpu_size,
        local_mode_script=str(local_mode_script_path.resolve()),
        check_port_availability=check_port_availability,
    )
    data = result.data if isinstance(result.data, dict) else {}
    sbatch_script_raw = data.get("sbatch_script")
    if not isinstance(sbatch_script_raw, str) or not sbatch_script_raw.strip():
        raise RuntimeError("render_start_sbatch returned no sbatch_script path")
    sbatch_script_path = Path(sbatch_script_raw).expanduser().resolve()
    if not sbatch_script_path.exists():
        raise RuntimeError(f"rendered sbatch script does not exist: {sbatch_script_path}")
    return sbatch_script_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_replay_configs.py",
        description=(
            "Generate local-mode single-profile sweep-qps experiment bundles "
            "(replay config + local script + rendered sbatch)."
        ),
    )
    parser.add_argument(
        "--source-run-dir",
        required=True,
        help="Source run result directory that contains replay-plan.json",
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
        "--partition",
        "-p",
        required=True,
        help="render-sbatch partition key, e.g. mi3001x",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="render-sbatch model key, e.g. qwen3_coder_30b",
    )
    parser.add_argument(
        "--lmcache",
        type=int,
        default=None,
        help=(
            "Optional LMCache max local CPU size. If provided, this value is forwarded "
            "to render-sbatch/start --lmcache for generated sbatch scripts."
        ),
    )
    parser.add_argument(
        "--port-profile",
        "-P",
        type=int,
        default=0,
        help="Single local profile id used by render-sbatch and replay (default: 0)",
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
        help="Optional explicit replay plan path (default: <source-run-dir>/replay-plan.json)",
    )
    parser.add_argument(
        "--replay-root-dir",
        default=None,
        help=(
            "Optional root directory for replay outputs. "
            "<utc-timestamp>/source-lineage/sweep-qps-local/single/<qps> is appended under this root."
        ),
    )
    parser.add_argument(
        "--server-config",
        default=str(DEFAULT_SERVER_CONFIG_PATH),
        help=f"Path to servers-amdhpc server_config.toml (default: {DEFAULT_SERVER_CONFIG_PATH})",
    )
    parser.add_argument(
        "--check-port-availability",
        action="store_true",
        help="Validate selected profile ports are currently free before rendering sbatch.",
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

    try:
        source_run_dir = Path(args.source_run_dir).expanduser().resolve()
        if not source_run_dir.exists() or not source_run_dir.is_dir():
            raise ValueError(f"invalid --source-run-dir: {source_run_dir}")

        output_config_root_dir = Path(args.output_config_dir).expanduser().resolve()
        server_config_path = Path(args.server_config).expanduser().resolve()
        if not server_config_path.exists() or not server_config_path.is_file():
            raise ValueError(f"invalid --server-config: {server_config_path}")

        qps_values = parse_qps_list(args.qps_list)
        time_constraint_s = parse_positive_float(
            str(args.time_constraint_s),
            field_name="--time-constraint-s",
        )

        if args.port_profile < 0:
            raise ValueError("--port-profile must be >= 0")

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
        if args.lmcache is not None and args.lmcache <= 0:
            raise ValueError("--lmcache must be a positive integer")

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

        batch_dir = (output_config_root_dir / batch_timestamp).resolve()
        replay_batch_output_base_resolved = (REPO_ROOT / replay_output_base).resolve()
        batch_dir.mkdir(parents=True, exist_ok=True)

        control_plane = build_control_plane(server_config_path)

        generated_experiments: list[dict[str, Any]] = []
        generated_qps_slugs: list[str] = []
        for qps in qps_values:
            qps_slug = format_qps_for_slug(qps)
            generated_qps_slugs.append(qps_slug)
            experiment_dir = (batch_dir / qps_slug).resolve()
            experiment_dir.mkdir(parents=True, exist_ok=True)

            replay_output_dir = (replay_batch_output_base_resolved / qps_slug).resolve()
            sbatch_log_dir = (replay_output_dir / "sbatch-logs").resolve()
            sbatch_log_dir.mkdir(parents=True, exist_ok=True)
            replay_payload = build_replay_config_payload(
                plan_path=plan_path,
                output_dir=replay_output_dir,
                qps=qps,
                poisson_seed=args.poisson_seed,
                randomize_seed=args.randomize_seed,
                time_constraint_s=time_constraint_s,
                port_profile_id=args.port_profile,
                vllm_log_interval_s=args.vllm_log_interval_s,
                vllm_log_timeout_s=args.vllm_log_timeout_s,
                launch_policy_override_json=launch_policy_override_json,
                extra_replay_json=extra_replay_json,
            )
            replay_config_path = (experiment_dir / "replay.toml").resolve()
            write_replay_config(replay_config_path, replay_payload=replay_payload)

            local_mode_script_path = (experiment_dir / "run_local_replay.sh").resolve()
            write_local_mode_script(
                path=local_mode_script_path,
                port_profile_id=args.port_profile,
            )
            gateway_config_path = (experiment_dir / "gateway-config.toml").resolve()
            write_gateway_config(
                gateway_config_path,
                port_profile_id=args.port_profile,
                output_root=(experiment_dir / "gateway-artifacts").resolve(),
            )

            rendered_sbatch_path = render_local_mode_sbatch(
                control_plane=control_plane,
                port_profile_id=args.port_profile,
                partition=args.partition,
                model=args.model,
                local_mode_script_path=local_mode_script_path,
                check_port_availability=bool(args.check_port_availability),
                lmcache_max_local_cpu_size=(
                    str(args.lmcache) if args.lmcache is not None else None
                ),
            )
            bundled_sbatch_path = (experiment_dir / "sbatch.sh").resolve()
            shutil.copy2(rendered_sbatch_path, bundled_sbatch_path)
            rewrite_sbatch_log_paths(
                bundled_sbatch_path,
                log_dir=sbatch_log_dir,
            )
            rewrite_sbatch_gateway_default(
                bundled_sbatch_path,
                gateway_config_path=gateway_config_path,
            )
            bundled_sbatch_path.chmod(0o750)

            generated_experiments.append(
                {
                    "qps": qps,
                    "qps_slug": qps_slug,
                    "experiment_dir": str(experiment_dir),
                    "replay_config": str(replay_config_path),
                    "local_mode_script": str(local_mode_script_path),
                    "gateway_config": str(gateway_config_path),
                    "sbatch": str(bundled_sbatch_path),
                    "sbatch_log_dir": str(sbatch_log_dir),
                    "rendered_sbatch_source": str(rendered_sbatch_path),
                    "submit_command": f"sbatch {path_for_config(bundled_sbatch_path)}",
                    "replay_output_dir": path_for_config(replay_output_dir),
                }
            )

        submit_all_script_path = (batch_dir / "submit_all.sh").resolve()
        write_submit_all_script(
            submit_all_script_path,
            qps_slugs=generated_qps_slugs,
        )
        summary = {
            "status": "ok",
            "batch_timestamp": batch_timestamp,
            "source_run_dir": str(source_run_dir),
            "plan_path": str(plan_path),
            "server_config": str(server_config_path),
            "partition": args.partition,
            "model": args.model,
            "lmcache": args.lmcache,
            "port_profile": args.port_profile,
            "check_port_availability": bool(args.check_port_availability),
            "output_config_root_dir": str(output_config_root_dir),
            "output_batch_dir": str(batch_dir),
            "replay_root_dir": str(replay_root_dir) if replay_root_dir is not None else None,
            "replay_output_batch_dir": path_for_config(replay_batch_output_base_resolved),
            "poisson_seed": args.poisson_seed,
            "randomize_seed": args.randomize_seed,
            "time_constraint_s": time_constraint_s,
            "qps_list": qps_values,
            "submit_all_script": str(submit_all_script_path),
            "submit_all_command": f"bash {path_for_config(submit_all_script_path)}",
            "generated_experiments": generated_experiments,
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
