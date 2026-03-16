#!/usr/bin/env python3
"""Generate local-mode mixed-plan sweep bundles at fixed QPS.

For each requested top-group percentage and fixed target QPS, this script
creates one experiment subdirectory containing:

- replay config TOML
- generated mixed replay plan JSON (rest + x% top workers; reused via local cache)
- local-mode replay script used by render-sbatch
- rendered sbatch script (no submission)
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_CONFIG_DIR = REPO_ROOT / "experiments" / "sweep-qps-local" / "mix" / "generated"
DEFAULT_SERVER_CONFIG_PATH = (
    REPO_ROOT / "servers" / "servers-amdhpc" / "server_config.toml"
)
SPLIT_PLAN_METRIC_ALIASES = ("token", "context")


def _load_split_generator_module() -> Any:
    module_path = (
        Path(__file__).resolve().parents[1] / "split" / "generate_replay_configs.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "generate_replay_qps_local_split_configs_shared",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_replay_qps_local_split_configs_shared"] = module
    spec.loader.exec_module(module)
    return module


_SPLIT = _load_split_generator_module()

# Exposed for tests/mocks, then synced into _SPLIT at runtime.
build_control_plane = _SPLIT.build_control_plane
render_local_mode_sbatch = _SPLIT.render_local_mode_sbatch


def _sync_split_module_globals() -> None:
    _SPLIT.REPO_ROOT = REPO_ROOT
    _SPLIT.RESULTS_ROOT = RESULTS_ROOT
    _SPLIT.build_control_plane = build_control_plane
    _SPLIT.render_local_mode_sbatch = render_local_mode_sbatch


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


def parse_top_percent_list(raw: str) -> list[float]:
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--top-percent-list must be a non-empty comma-separated list")

    values: list[float] = []
    for token in tokens:
        try:
            parsed = float(token)
        except ValueError as exc:
            raise ValueError(
                f"--top-percent-list contains non-numeric value: {token!r}"
            ) from exc
        if parsed < 0 or parsed > 100:
            raise ValueError("--top-percent-list values must be within [0, 100]")
        values.append(parsed)

    if len(set(values)) != len(values):
        raise ValueError("--top-percent-list cannot contain duplicate values")
    return values


def parse_optional_object_json(raw: str | None, *, field_name: str) -> dict[str, Any]:
    return _SPLIT.parse_optional_object_json(raw, field_name=field_name)


def path_for_config(path: Path) -> str:
    _sync_split_module_globals()
    return _SPLIT.path_for_config(path)


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_mix_cache_request(
    *,
    top_plan_path: Path,
    rest_plan_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "mix-plan-cache.v1",
        "selection_rule": "take first floor(len(top_workers) * top_percent / 100)",
        "top_plan": file_identity(top_plan_path),
        "rest_plan": file_identity(rest_plan_path),
    }


def resolve_mix_cache_dir(
    *,
    cache_root: Path,
    cache_request: dict[str, Any],
) -> tuple[str, Path]:
    key_text = json.dumps(cache_request, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
    cache_dir = (cache_root / cache_key).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_meta = {
        "created_at": now_iso8601_utc(),
        "cache_key": cache_key,
        "cache_request": cache_request,
    }
    cache_meta_path = (cache_dir / "mix-cache.meta.json").resolve()
    if not cache_meta_path.exists():
        cache_meta_path.write_text(
            json.dumps(cache_meta, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return cache_key, cache_dir


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
        / "mix"
    )


def derive_replay_timestamp_output_dir(
    *,
    replay_root_dir: Path | None,
    batch_timestamp: str,
) -> Path:
    _sync_split_module_globals()
    return _SPLIT.derive_replay_timestamp_output_dir(
        replay_root_dir=replay_root_dir,
        batch_timestamp=batch_timestamp,
    )


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


def _extract_suffix_from_candidate(
    *,
    file_name: str,
    prefix: str,
    extension: str,
) -> str | None:
    if extension:
        if not file_name.endswith(extension):
            return None
        base_name = file_name[: -len(extension)]
    else:
        base_name = file_name

    if not base_name.startswith(prefix):
        return None

    suffix_part = base_name[len(prefix) :]
    if not suffix_part:
        return ""
    if not suffix_part.startswith("."):
        return None
    return suffix_part[1:]


def discover_split_plan_pair(
    base_plan_path: Path,
    *,
    metric_alias: str | None,
    preferred_suffix: str | None = None,
) -> dict[str, Path] | None:
    base_stem = base_plan_path.stem
    extension = base_plan_path.suffix
    if metric_alias is None:
        top_prefix = f"{base_stem}.top"
        rest_prefix = f"{base_stem}.rest"
    else:
        top_prefix = f"{base_stem}.{metric_alias}.top"
        rest_prefix = f"{base_stem}.{metric_alias}.rest"

    top_by_suffix: dict[str, Path] = {}
    rest_by_suffix: dict[str, Path] = {}
    for candidate_path in base_plan_path.parent.iterdir():
        if not candidate_path.is_file():
            continue
        candidate_name = candidate_path.name
        top_suffix = _extract_suffix_from_candidate(
            file_name=candidate_name,
            prefix=top_prefix,
            extension=extension,
        )
        if top_suffix is not None:
            top_by_suffix[top_suffix] = candidate_path.resolve()

        rest_suffix = _extract_suffix_from_candidate(
            file_name=candidate_name,
            prefix=rest_prefix,
            extension=extension,
        )
        if rest_suffix is not None:
            rest_by_suffix[rest_suffix] = candidate_path.resolve()

    common_suffixes = set(top_by_suffix).intersection(rest_by_suffix)
    if not common_suffixes:
        return None

    if preferred_suffix is not None:
        if preferred_suffix not in common_suffixes:
            return None
        selected_suffix = preferred_suffix
    elif "" in common_suffixes:
        selected_suffix = ""
    else:
        selected_suffix = max(
            common_suffixes,
            key=lambda suffix: (
                max(
                    top_by_suffix[suffix].stat().st_mtime_ns,
                    rest_by_suffix[suffix].stat().st_mtime_ns,
                ),
                suffix,
            ),
        )
    return {
        "top": top_by_suffix[selected_suffix],
        "rest": rest_by_suffix[selected_suffix],
    }


def maybe_strip_last_stem_component(base_plan_path: Path) -> tuple[Path, str] | None:
    stem = base_plan_path.stem
    dot_index = stem.rfind(".")
    if dot_index <= 0:
        return None
    stripped_stem = stem[:dot_index]
    inferred_suffix = stem[dot_index + 1 :]
    if not stripped_stem or not inferred_suffix:
        return None
    stripped_path = (base_plan_path.parent / f"{stripped_stem}{base_plan_path.suffix}").resolve()
    return stripped_path, inferred_suffix


def discover_default_split_plan_paths(
    *,
    base_plan_path: Path,
    preferred_suffix: str | None = None,
) -> dict[str, Path] | None:
    for metric_alias in SPLIT_PLAN_METRIC_ALIASES:
        discovered_paths = discover_split_plan_pair(
            base_plan_path,
            metric_alias=metric_alias,
            preferred_suffix=preferred_suffix,
        )
        if discovered_paths is not None:
            return discovered_paths

    return discover_split_plan_pair(
        base_plan_path,
        metric_alias=None,
        preferred_suffix=preferred_suffix,
    )


def derive_default_split_plan_paths(
    base_plan_path: Path,
    *,
    preferred_suffix: str | None = None,
) -> dict[str, Path]:
    discovered_paths = discover_default_split_plan_paths(
        base_plan_path=base_plan_path,
        preferred_suffix=preferred_suffix,
    )
    if discovered_paths is not None:
        return discovered_paths

    stripped_candidate = maybe_strip_last_stem_component(base_plan_path)
    stripped_base_plan_path: Path | None = None
    inferred_suffix: str | None = None
    if stripped_candidate is not None:
        stripped_base_plan_path, inferred_suffix = stripped_candidate
        preferred_for_stripped = preferred_suffix or inferred_suffix
        discovered_paths = discover_default_split_plan_paths(
            base_plan_path=stripped_base_plan_path,
            preferred_suffix=preferred_for_stripped,
        )
        if discovered_paths is not None:
            return discovered_paths

        discovered_paths = discover_default_split_plan_paths(
            base_plan_path=stripped_base_plan_path,
        )
        if discovered_paths is not None:
            return discovered_paths

    fallback_base_plan_path = stripped_base_plan_path or base_plan_path
    fallback_paths = derive_metric_split_plan_paths(
        fallback_base_plan_path,
        metric_alias="token",
    )
    suffix_to_apply = preferred_suffix or inferred_suffix
    if not suffix_to_apply:
        return fallback_paths
    return {
        split_group_name: with_plan_name_suffix(path, suffix_to_apply)
        for split_group_name, path in fallback_paths.items()
    }


def build_utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso8601_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_generation_command_raw(argv: list[str] | None) -> str:
    """Return a shell-safe raw command string for this generator invocation."""
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


def format_percent_for_slug(percent: float) -> str:
    text = format(percent, ".12g")
    text = text.replace("-", "m")
    text = text.replace(".", "_")
    text = text.replace("+", "")
    return f"p{text}"


def safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return default


def worker_sort_key(worker: dict[str, Any]) -> tuple[float, int, str]:
    run_offset = safe_float(worker.get("run_offset_s"), default=float("inf"))
    launch_priority = safe_int(worker.get("launch_priority"), default=sys.maxsize)
    worker_id = str(worker.get("worker_id") or worker.get("trial_id") or "")
    return (run_offset, launch_priority, worker_id)


def clone_workers_with_reindexed_launch_priority(
    workers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cloned = copy.deepcopy(workers)
    for launch_priority, worker in enumerate(cloned):
        worker["launch_priority"] = launch_priority
    return cloned


def read_plan_payload(plan_path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {field_name} JSON: {plan_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must decode to a JSON object: {plan_path}")
    workers = payload.get("workers")
    if not isinstance(workers, list):
        raise ValueError(f"{field_name} missing workers list: {plan_path}")
    for index, worker in enumerate(workers):
        if not isinstance(worker, dict):
            raise ValueError(
                f"{field_name} workers[{index}] is not an object: {plan_path}"
            )
    return payload


def compute_selected_top_workers_count(*, total_top_workers: int, top_percent: float) -> int:
    if total_top_workers <= 0:
        return 0
    if top_percent <= 0:
        return 0
    if top_percent >= 100:
        return total_top_workers
    count = int(math.floor((total_top_workers * top_percent) / 100.0 + 1e-12))
    return max(0, min(total_top_workers, count))


def build_mixed_plan_payload(
    *,
    top_plan_payload: dict[str, Any],
    rest_plan_payload: dict[str, Any],
    top_plan_path: Path,
    rest_plan_path: Path,
    top_percent: float,
) -> dict[str, Any]:
    top_workers_payload = top_plan_payload.get("workers")
    rest_workers_payload = rest_plan_payload.get("workers")
    if not isinstance(top_workers_payload, list) or not isinstance(rest_workers_payload, list):
        raise ValueError("both top/rest plan payloads must include a workers list")

    total_top_workers = len(top_workers_payload)
    selected_top_workers_count = compute_selected_top_workers_count(
        total_top_workers=total_top_workers,
        top_percent=top_percent,
    )

    selected_top_workers = copy.deepcopy(top_workers_payload[:selected_top_workers_count])
    all_rest_workers = copy.deepcopy(rest_workers_payload)

    mixed_workers = selected_top_workers + all_rest_workers
    mixed_workers.sort(key=worker_sort_key)
    mixed_workers = clone_workers_with_reindexed_launch_priority(mixed_workers)

    mixed_payload = copy.deepcopy(rest_plan_payload)
    mixed_payload["workers"] = mixed_workers
    mixed_payload["mix_top_rest"] = {
        "enabled": True,
        "top_percent": top_percent,
        "top_workers_total": total_top_workers,
        "top_workers_selected": selected_top_workers_count,
        "rest_workers_total": len(rest_workers_payload),
        "top_plan_path": str(top_plan_path.resolve()),
        "rest_plan_path": str(rest_plan_path.resolve()),
        "generated_at": now_iso8601_utc(),
        "selection_rule": "take first floor(len(top_workers) * top_percent / 100)",
    }
    split_two_group_payload = mixed_payload.get("split_two_group")
    if isinstance(split_two_group_payload, dict):
        mixed_payload["split_two_group"] = copy.deepcopy(split_two_group_payload)
        mixed_payload["split_two_group"]["group"] = "mixed"
        mixed_payload["split_two_group"]["top_percent"] = top_percent

    return mixed_payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def build_mix_plan_stats(
    *,
    top_percent: float,
    top_percent_slug: str,
    mixed_plan_payload: dict[str, Any],
) -> dict[str, Any]:
    mix_payload = mixed_plan_payload.get("mix_top_rest")
    if not isinstance(mix_payload, dict):
        raise ValueError("mixed plan payload missing mix_top_rest object")
    return {
        "schema_version": "mix-plan-cache-meta.v1",
        "top_percent": top_percent,
        "top_percent_slug": top_percent_slug,
        "top_workers_total": int(mix_payload.get("top_workers_total", 0)),
        "top_workers_selected": int(mix_payload.get("top_workers_selected", 0)),
        "rest_workers_total": int(mix_payload.get("rest_workers_total", 0)),
        "mixed_workers_total": len(mixed_plan_payload.get("workers", [])),
        "generated_at": now_iso8601_utc(),
    }


def load_mix_plan_stats(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != "mix-plan-cache-meta.v1":
        return None
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_replay_configs.py",
        description=(
            "Generate local-mode sweep bundles at fixed QPS while sweeping top-group "
            "mix ratio (rest + x% top)."
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
        "--qps",
        required=True,
        type=float,
        help="Fixed Poisson rate (requests per second) used for all generated experiments.",
    )
    parser.add_argument(
        "--top-percent-list",
        required=True,
        help=(
            "Comma-separated percentages of top workers to include (0-100), "
            "for example '0,25,50,75,100'."
        ),
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
        "--no-async-scheduling",
        action="store_true",
        help=(
            "Forward --no-async-scheduling to render-sbatch/start so generated sbatch "
            "scripts include it in vLLM startup."
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
        "--cache-dir",
        default=None,
        help="Mixed-plan cache root (default: <output-config-dir>/cache).",
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
            "<utc-timestamp>/source-lineage/sweep-qps-local/mix/<mix-percent> "
            "is appended under this root."
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
    parser.add_argument(
        "--additional-suffix",
        default=None,
        help=(
            "Optional suffix for input/output plan names. When set, the generator "
            "looks for split plans with this suffix (e.g., replay-plan.token.top.<suffix>.json) "
            "and also appends the suffix to generated mixed plan names "
            "(e.g., replay-plan.mix.p50.<suffix>.json)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    generation_command_raw = build_generation_command_raw(argv)

    try:
        _sync_split_module_globals()

        source_run_dir = Path(args.source_run_dir).expanduser().resolve()
        if not source_run_dir.exists() or not source_run_dir.is_dir():
            raise ValueError(f"invalid --source-run-dir: {source_run_dir}")

        output_config_root_dir = Path(args.output_config_dir).expanduser().resolve()
        server_config_path = Path(args.server_config).expanduser().resolve()
        if not server_config_path.exists() or not server_config_path.is_file():
            raise ValueError(f"invalid --server-config: {server_config_path}")

        qps = parse_positive_float(str(args.qps), field_name="--qps")
        top_percent_values = parse_top_percent_list(args.top_percent_list)
        time_constraint_s = parse_positive_float(
            str(args.time_constraint_s),
            field_name="--time-constraint-s",
        )

        if args.port_profile < 0:
            raise ValueError("--port-profile must be >= 0")

        if args.plan_path:
            base_plan_path = Path(args.plan_path).expanduser().resolve()
        else:
            suffix_part = f".{args.additional_suffix}" if args.additional_suffix else ""
            base_plan_path = (source_run_dir / f"replay-plan{suffix_part}.json").resolve()

        default_split_plan_paths = derive_default_split_plan_paths(
            base_plan_path,
            preferred_suffix=args.additional_suffix,
        )
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

        for split_group_name, split_plan_path in {
            "top": top_plan_path,
            "rest": rest_plan_path,
        }.items():
            if not split_plan_path.exists() or not split_plan_path.is_file():
                raise ValueError(
                    f"{split_group_name} replay plan does not exist: {split_plan_path}"
                )

        cache_root = (
            Path(args.cache_dir).expanduser().resolve()
            if args.cache_dir is not None
            else (output_config_root_dir / "cache").resolve()
        )
        cache_request = build_mix_cache_request(
            top_plan_path=top_plan_path,
            rest_plan_path=rest_plan_path,
        )
        cache_key, cache_dir = resolve_mix_cache_dir(
            cache_root=cache_root,
            cache_request=cache_request,
        )

        top_plan_payload: dict[str, Any] | None = None
        rest_plan_payload: dict[str, Any] | None = None

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
        replay_timestamp_output_dir = derive_replay_timestamp_output_dir(
            replay_root_dir=replay_root_dir,
            batch_timestamp=batch_timestamp,
        )

        batch_dir = (output_config_root_dir / batch_timestamp).resolve()
        replay_batch_output_base_resolved = (REPO_ROOT / replay_output_base).resolve()
        batch_dir.mkdir(parents=True, exist_ok=True)

        control_plane = build_control_plane(server_config_path)

        generated_experiments: list[dict[str, Any]] = []
        generated_submit_paths: list[str] = []
        cache_hit_count = 0
        cache_miss_count = 0

        for top_percent in top_percent_values:
            percent_slug = format_percent_for_slug(top_percent)
            experiment_dir = (batch_dir / percent_slug).resolve()
            experiment_dir.mkdir(parents=True, exist_ok=True)

            suffix_part = f".{args.additional_suffix}" if args.additional_suffix else ""
            cached_mixed_plan_path = (
                cache_dir / f"replay-plan.mix.{percent_slug}{suffix_part}.json"
            ).resolve()
            cached_stats_path = (
                cache_dir / f"replay-plan.mix.{percent_slug}{suffix_part}.meta.json"
            ).resolve()
            cache_hit = cached_mixed_plan_path.is_file()
            plan_stats = load_mix_plan_stats(cached_stats_path)

            if cache_hit:
                cache_hit_count += 1
                if plan_stats is None:
                    cached_mixed_plan_payload = read_plan_payload(
                        cached_mixed_plan_path,
                        field_name=f"cached mixed replay plan ({percent_slug})",
                    )
                    plan_stats = build_mix_plan_stats(
                        top_percent=top_percent,
                        top_percent_slug=percent_slug,
                        mixed_plan_payload=cached_mixed_plan_payload,
                    )
                    write_json(cached_stats_path, plan_stats)
            else:
                cache_miss_count += 1
                if top_plan_payload is None or rest_plan_payload is None:
                    top_plan_payload = read_plan_payload(
                        top_plan_path,
                        field_name="top replay plan",
                    )
                    rest_plan_payload = read_plan_payload(
                        rest_plan_path,
                        field_name="rest replay plan",
                    )
                mixed_plan_payload = build_mixed_plan_payload(
                    top_plan_payload=top_plan_payload,
                    rest_plan_payload=rest_plan_payload,
                    top_plan_path=top_plan_path,
                    rest_plan_path=rest_plan_path,
                    top_percent=top_percent,
                )
                write_json(cached_mixed_plan_path, mixed_plan_payload)
                plan_stats = build_mix_plan_stats(
                    top_percent=top_percent,
                    top_percent_slug=percent_slug,
                    mixed_plan_payload=mixed_plan_payload,
                )
                write_json(cached_stats_path, plan_stats)

            mixed_plan_path = (experiment_dir / "plan" / cached_mixed_plan_path.name).resolve()
            mixed_plan_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_mixed_plan_path, mixed_plan_path)

            replay_output_dir = (replay_batch_output_base_resolved / percent_slug).resolve()
            sbatch_log_dir = (replay_output_dir / "sbatch-logs").resolve()
            sbatch_log_dir.mkdir(parents=True, exist_ok=True)

            replay_payload = _SPLIT.build_replay_config_payload(
                plan_path=mixed_plan_path,
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
            _SPLIT.write_replay_config(replay_config_path, replay_payload=replay_payload)

            local_mode_script_path = (experiment_dir / "run_local_replay.sh").resolve()
            _SPLIT.write_local_mode_script(
                path=local_mode_script_path,
                port_profile_id=args.port_profile,
            )
            gateway_config_path = (experiment_dir / "gateway-config.toml").resolve()
            _SPLIT.write_gateway_config(
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
                no_async_scheduling=bool(args.no_async_scheduling),
            )
            bundled_sbatch_path = (experiment_dir / "sbatch.sh").resolve()
            shutil.copy2(rendered_sbatch_path, bundled_sbatch_path)
            _SPLIT.rewrite_sbatch_log_paths(
                bundled_sbatch_path,
                log_dir=sbatch_log_dir,
            )
            _SPLIT.rewrite_sbatch_gateway_default(
                bundled_sbatch_path,
                gateway_config_path=gateway_config_path,
            )
            bundled_sbatch_path.chmod(0o750)

            if plan_stats is None:
                raise RuntimeError(
                    f"failed to resolve cached mix plan stats for {percent_slug}: {cached_stats_path}"
                )
            submit_relpath = f"{percent_slug}/sbatch.sh"
            top_workers_selected = int(plan_stats.get("top_workers_selected", 0))
            top_workers_total = int(plan_stats.get("top_workers_total", 0))
            rest_workers_total = int(plan_stats.get("rest_workers_total", 0))
            mixed_workers_total = int(plan_stats.get("mixed_workers_total", 0))
            experiment_record = {
                "top_percent": top_percent,
                "top_percent_slug": percent_slug,
                "mixed_plan_path": str(mixed_plan_path),
                "cached_mixed_plan_path": str(cached_mixed_plan_path),
                "cache_hit": cache_hit,
                "top_workers_total": top_workers_total,
                "top_workers_selected": top_workers_selected,
                "rest_workers_total": rest_workers_total,
                "mixed_workers_total": mixed_workers_total,
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
            generated_experiments.append(experiment_record)
            generated_submit_paths.append(submit_relpath)

        submit_all_script_path = (batch_dir / "submit_all.sh").resolve()
        _SPLIT.write_submit_all_script(
            submit_all_script_path,
            sbatch_relpaths=generated_submit_paths,
        )

        summary = {
            "status": "ok",
            "batch_timestamp": batch_timestamp,
            "source_run_dir": str(source_run_dir),
            "plan_paths": {
                "top": str(top_plan_path),
                "rest": str(rest_plan_path),
            },
            "server_config": str(server_config_path),
            "partition": args.partition,
            "model": args.model,
            "lmcache": args.lmcache,
            "no_async_scheduling": bool(args.no_async_scheduling),
            "port_profile": args.port_profile,
            "check_port_availability": bool(args.check_port_availability),
            "output_config_root_dir": str(output_config_root_dir),
            "output_batch_dir": str(batch_dir),
            "cache_root_dir": str(cache_root),
            "cache_key": cache_key,
            "cache_dir": str(cache_dir),
            "cache_hit_count": cache_hit_count,
            "cache_miss_count": cache_miss_count,
            "replay_root_dir": str(replay_root_dir) if replay_root_dir is not None else None,
            "replay_output_batch_dir": path_for_config(replay_batch_output_base_resolved),
            "generated_batch_copy_dir": path_for_config(
                replay_timestamp_output_dir / "generated"
            ),
            "qps": qps,
            "top_percent_list": top_percent_values,
            "top_percent_selection_rule": (
                "take first floor(len(top_workers) * top_percent / 100) workers from top plan"
            ),
            "poisson_seed": args.poisson_seed,
            "randomize_seed": args.randomize_seed,
            "time_constraint_s": time_constraint_s,
            "generation_command_raw": generation_command_raw,
            "submit_all_script": str(submit_all_script_path),
            "submit_all_command": f"bash {path_for_config(submit_all_script_path)}",
            "generated_experiments": generated_experiments,
        }
        manifest_path = (batch_dir / "manifest.json").resolve()
        manifest_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        _SPLIT.copy_generated_batch_dir_to_replay_timestamp(
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
