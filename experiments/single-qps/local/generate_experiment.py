#!/usr/bin/env python3
"""Generate one local single-QPS replay experiment bundle.

This helper does three things:
1) Compile (or reuse cached) replay plan artifacts for the requested split mode.
2) Materialize one replay config TOML for a single Poisson QPS target.
3) Emit one runnable script entrypoint that accepts a port profile override.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_CONFIG_DIR = REPO_ROOT / "experiments" / "single-qps" / "local" / "generated"
DEFAULT_REPLAY_OUTPUT_ROOT = Path("results") / "replay" / "single-qps" / "split"
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


@dataclass
class CompileCacheResult:
    cache_key: str
    cache_dir: Path
    cache_hit: bool
    compile_command: list[str]
    selected_plan_path: Path
    related_plan_paths: dict[str, str]


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
    normalized = raw.strip().lower()
    normalized = normalized.replace("_", "-")
    canonical = SPLIT_ALIASES.get(normalized)
    if canonical is None:
        allowed = ", ".join(SPLIT_CHOICES)
        raise ValueError(f"unsupported --split value {raw!r}; choose one of: {allowed}")
    return canonical


def format_qps_slug(qps: float) -> str:
    text = format(qps, ".12g")
    text = text.replace("-", "m").replace(".", "_").replace("+", "")
    return f"qps{text}"


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


def _load_target_model_keys() -> set[str]:
    if not MODEL_CONFIG_PATH.exists():
        raise ValueError(f"missing model config: {MODEL_CONFIG_PATH}")
    payload = tomllib.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    raw_models = payload.get("models")
    if not isinstance(raw_models, dict):
        raise ValueError("configs/model_config.toml must include [models]")
    out: set[str] = set()
    for key, value in raw_models.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        out.add(key)
    if not out:
        raise ValueError("configs/model_config.toml contains no valid model keys")
    return out


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


def _build_compile_request_payload(
    *,
    source_run_dir: Path,
    target_model: str,
    port_profile: int,
    split: str,
    split_two_group_metric: str,
) -> dict[str, Any]:
    if split in {"top", "rest"}:
        compile_variant = f"split-two-group:{split_two_group_metric}"
    elif split == "exclude-unranked":
        compile_variant = "exclude-unranked"
    else:
        compile_variant = "full"
    return {
        "source_run_dir": str(source_run_dir.resolve()),
        "target_model": target_model,
        "port_profile": int(port_profile),
        "split": split,
        "compile_variant": compile_variant,
    }


def _build_compile_command(
    *,
    python_bin: str,
    source_run_dir: Path,
    port_profile: int,
    split: str,
    split_two_group_metric: str,
    plan_out_path: Path,
    request_timeout_s: float | None,
) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "replayer",
        "compile",
        "--job-dir",
        str(source_run_dir.resolve()),
        "--port-profile-id",
        str(port_profile),
        "--plan-out",
        str(plan_out_path.resolve()),
    ]
    if request_timeout_s is not None:
        cmd.extend(["--request-timeout-s", str(request_timeout_s)])
    if split == "exclude-unranked":
        cmd.append("--exclude-unranked-trails")
    elif split in {"top", "rest"}:
        cmd.append("--split-two-group-plans")
        cmd.extend(["--split-two-group-metric", split_two_group_metric])
    return cmd


def _resolve_compile_paths(
    *,
    cache_dir: Path,
    split: str,
    split_two_group_metric: str,
) -> tuple[Path, Path, dict[str, str]]:
    if split == "full":
        plan_out = (cache_dir / "replay-plan.full.json").resolve()
        selected = plan_out
        related = {"selected": str(selected)}
        return plan_out, selected, related
    if split == "exclude-unranked":
        plan_out = (cache_dir / "replay-plan.exclude-unranked.json").resolve()
        selected = plan_out
        related = {"selected": str(selected)}
        return plan_out, selected, related

    metric_alias = SPLIT_METRIC_ALIASES.get(split_two_group_metric)
    if metric_alias is None:
        raise ValueError(
            f"unsupported --split-two-group-metric {split_two_group_metric!r}; "
            f"supported: {sorted(SPLIT_METRIC_ALIASES)}"
        )

    plan_out = (cache_dir / "replay-plan.json").resolve()
    top_path = with_plan_name_suffix(plan_out, f"{metric_alias}.top")
    rest_path = with_plan_name_suffix(plan_out, f"{metric_alias}.rest")
    selected = top_path if split == "top" else rest_path
    related = {
        "selected": str(selected),
        "top": str(top_path),
        "rest": str(rest_path),
    }
    return plan_out, selected, related


def _write_compile_logs(cache_dir: Path, *, stdout: str, stderr: str) -> tuple[Path, Path]:
    stdout_path = (cache_dir / "compile.stdout.log").resolve()
    stderr_path = (cache_dir / "compile.stderr.log").resolve()
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return stdout_path, stderr_path


def compile_plan_with_cache(
    *,
    source_run_dir: Path,
    target_model: str,
    port_profile: int,
    split: str,
    split_two_group_metric: str,
    request_timeout_s: float | None,
    cache_root: Path,
    python_bin: str,
) -> CompileCacheResult:
    compile_request = _build_compile_request_payload(
        source_run_dir=source_run_dir,
        target_model=target_model,
        port_profile=port_profile,
        split=split,
        split_two_group_metric=split_two_group_metric,
    )
    key_text = json.dumps(compile_request, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
    cache_dir = (cache_root / cache_key).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    compile_meta_path = (cache_dir / "compile.meta.json").resolve()
    plan_out_path, selected_plan_path, related_plan_paths = _resolve_compile_paths(
        cache_dir=cache_dir,
        split=split,
        split_two_group_metric=split_two_group_metric,
    )
    compile_command = _build_compile_command(
        python_bin=python_bin,
        source_run_dir=source_run_dir,
        port_profile=port_profile,
        split=split,
        split_two_group_metric=split_two_group_metric,
        plan_out_path=plan_out_path,
        request_timeout_s=request_timeout_s,
    )

    existing_meta_raw = None
    if compile_meta_path.exists():
        try:
            existing_meta_raw = json.loads(compile_meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_meta_raw = None

    expected_paths = [Path(path_text) for path_text in related_plan_paths.values()]
    paths_exist = all(path.is_file() for path in expected_paths)
    meta_matches = isinstance(existing_meta_raw, dict) and existing_meta_raw.get("compile_request") == compile_request
    if paths_exist and meta_matches:
        return CompileCacheResult(
            cache_key=cache_key,
            cache_dir=cache_dir,
            cache_hit=True,
            compile_command=compile_command,
            selected_plan_path=selected_plan_path,
            related_plan_paths=related_plan_paths,
        )

    result = subprocess.run(
        compile_command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path, stderr_path = _write_compile_logs(
        cache_dir,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "replayer compile failed"
            f" (rc={result.returncode}).\n"
            f"command: {shlex.join(compile_command)}\n"
            f"stdout log: {stdout_path}\n"
            f"stderr log: {stderr_path}"
        )

    missing_paths = [path for path in expected_paths if not path.is_file()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise RuntimeError(
            "replayer compile succeeded but expected plan files were not found: "
            f"{missing_text}"
        )

    compile_meta = {
        "compiled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "compile_request": compile_request,
        "compile_command": compile_command,
        "cache_key": cache_key,
        "plan_out_path": str(plan_out_path),
        "related_plan_paths": related_plan_paths,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    compile_meta_path.write_text(
        json.dumps(compile_meta, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    return CompileCacheResult(
        cache_key=cache_key,
        cache_dir=cache_dir,
        cache_hit=False,
        compile_command=compile_command,
        selected_plan_path=selected_plan_path,
        related_plan_paths=related_plan_paths,
    )


def write_replay_config(path: Path, *, replay_payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Auto-generated by experiments/single-qps/local/generate_experiment.py",
        "",
    ]
    _append_toml_table(lines, "replay", replay_payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_script(
    path: Path,
    *,
    default_port_profile: int,
    target_model: str,
    split: str,
    qps: float,
) -> None:
    content = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        f"DEFAULT_PORT_PROFILE_ID={default_port_profile}\n"
        "PORT_PROFILE_ID_VALUE=\"${1:-${PORT_PROFILE_ID:-${DEFAULT_PORT_PROFILE_ID}}}\"\n"
        "PYTHON_BIN=\"${PYTHON_BIN:-python3}\"\n\n"
        f"echo \"[single-qps] target_model={target_model} split={split} qps={qps} port_profile=${{PORT_PROFILE_ID_VALUE}}\"\n"
        "\"${PYTHON_BIN}\" -m replayer replay \\\n"
        "  --config \"${SCRIPT_DIR}/replay.toml\" \\\n"
        "  --port-profile-id \"${PORT_PROFILE_ID_VALUE}\"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o750)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_experiment.py",
        description=(
            "Generate a local single-QPS replay bundle with cached compile artifacts "
            "and a one-command replay entrypoint script."
        ),
    )
    parser.add_argument("--source-run-dir", required=True, help="Profiled source run directory under results/.")
    parser.add_argument("--poisson-seed", required=True, type=parse_non_negative_int, help="Poisson launch seed.")
    parser.add_argument("--randomize-seed", required=True, type=parse_non_negative_int, help="Replay randomization seed.")
    parser.add_argument("--qps", required=True, type=float, help="Poisson launch rate (requests per second).")
    parser.add_argument("--time-constraint-s", required=True, type=float, help="Replay time limit in seconds.")
    parser.add_argument("--target-model", required=True, help="Target model key from configs/model_config.toml.")
    parser.add_argument("--port-profile", "-P", required=True, type=int, help="Port profile ID for compile/replay.")
    parser.add_argument(
        "--split",
        required=True,
        help="Plan split mode: full | exclude-unranked | top | rest (accepts typo alias exclued-unranked).",
    )
    parser.add_argument(
        "--split-two-group-metric",
        choices=sorted(SPLIT_METRIC_ALIASES),
        default="token_usage",
        help="Split grouping metric used for top/rest plan compile (default: token_usage).",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=None,
        help="Optional compile tokenizer timeout passed to replayer compile.",
    )
    parser.add_argument(
        "--output-config-dir",
        default=str(DEFAULT_OUTPUT_CONFIG_DIR),
        help=f"Generated bundle root (default: {DEFAULT_OUTPUT_CONFIG_DIR}).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Compile cache root (default: <output-config-dir>/cache).",
    )
    parser.add_argument(
        "--replay-output-root",
        default=str(DEFAULT_REPLAY_OUTPUT_ROOT),
        help=(
            "Replay output root. Default appends "
            "<split>/<qps>/<timestamp> under results/replay/single-qps/split/."
        ),
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used for `-m replayer compile` (default: current python).",
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
        if args.request_timeout_s is not None and args.request_timeout_s <= 0:
            raise ValueError("--request-timeout-s must be > 0")

        target_model = str(args.target_model).strip()
        if not target_model:
            raise ValueError("--target-model cannot be empty")
        valid_model_keys = _load_target_model_keys()
        if target_model not in valid_model_keys:
            available = ", ".join(sorted(valid_model_keys))
            raise ValueError(
                f"unknown --target-model {target_model!r}; available keys: {available}"
            )

        output_config_root = Path(args.output_config_dir).expanduser().resolve()
        cache_root = (
            Path(args.cache_dir).expanduser().resolve()
            if args.cache_dir is not None
            else (output_config_root / "cache").resolve()
        )
        replay_output_root = Path(args.replay_output_root).expanduser()
        if not replay_output_root.is_absolute():
            replay_output_root = (REPO_ROOT / replay_output_root).resolve()
        else:
            replay_output_root = replay_output_root.resolve()

        batch_timestamp = build_utc_timestamp_slug()
        qps_slug = format_qps_slug(qps)
        batch_dir = (output_config_root / batch_timestamp).resolve()

        compile_result = compile_plan_with_cache(
            source_run_dir=source_run_dir,
            target_model=target_model,
            port_profile=args.port_profile,
            split=split,
            split_two_group_metric=str(args.split_two_group_metric),
            request_timeout_s=args.request_timeout_s,
            cache_root=cache_root,
            python_bin=str(args.python_bin),
        )

        plan_copy_dir = (batch_dir / "plan").resolve()
        plan_copy_dir.mkdir(parents=True, exist_ok=True)
        selected_plan_copy_path = (plan_copy_dir / compile_result.selected_plan_path.name).resolve()
        shutil.copy2(compile_result.selected_plan_path, selected_plan_copy_path)

        replay_output_dir = (
            replay_output_root / split / qps_slug / batch_timestamp
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
        replay_config_path = (batch_dir / "replay.toml").resolve()
        write_replay_config(replay_config_path, replay_payload=replay_payload)

        run_script_path = (batch_dir / "run_replay.sh").resolve()
        write_run_script(
            run_script_path,
            default_port_profile=int(args.port_profile),
            target_model=target_model,
            split=split,
            qps=qps,
        )

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
            "request_timeout_s": args.request_timeout_s,
            "output_config_root_dir": str(output_config_root),
            "output_batch_dir": str(batch_dir),
            "cache_root_dir": str(cache_root),
            "cache_key": compile_result.cache_key,
            "cache_dir": str(compile_result.cache_dir),
            "cache_hit": compile_result.cache_hit,
            "compile_command": compile_result.compile_command,
            "selected_cached_plan": str(compile_result.selected_plan_path),
            "related_cached_plans": compile_result.related_plan_paths,
            "selected_plan_copy": str(selected_plan_copy_path),
            "replay_output_dir": path_for_config(replay_output_dir),
            "replay_config": str(replay_config_path),
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
