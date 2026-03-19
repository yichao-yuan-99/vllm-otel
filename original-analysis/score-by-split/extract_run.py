from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


DEFAULT_OUTPUT_NAME = "score-by-split-summary.json"
NORMAL_TOKEN_TWO_GROUP_NAME = "top-p-token-usage-two-groups.json"
NORMAL_CONTEXT_TWO_GROUP_NAME = "top-p-context-usage-two-groups.json"
STRICT_499_SUFFIX = "strict-499"
STRICT_TOKEN_TWO_GROUP_NAME = f"top-p-token-usage-two-groups.{STRICT_499_SUFFIX}.json"
STRICT_CONTEXT_TWO_GROUP_NAME = (
    f"top-p-context-usage-two-groups.{STRICT_499_SUFFIX}.json"
)

ROOT_TOKEN_TABLE_NAME = "score-by-split-token-usage-two-group-table.csv"
ROOT_CONTEXT_TABLE_NAME = "score-by-split-context-usage-two-group-table.csv"
ROOT_STRICT_TOKEN_TABLE_NAME = (
    f"score-by-split-token-usage-two-group-table.{STRICT_499_SUFFIX}.csv"
)
ROOT_STRICT_CONTEXT_TABLE_NAME = (
    f"score-by-split-context-usage-two-group-table.{STRICT_499_SUFFIX}.csv"
)

MODE_NORMAL = "normal"
MODE_STRICT_499 = "strict_499"

_RUN_NAME_HASH_PATTERN = re.compile(r"^run_[^_]+_([0-9a-fA-F]{8,64})_[0-9a-fA-F]+$")


def _default_max_procs() -> int:
    max_procs_env = os.getenv("MAX_PROCS")
    if max_procs_env:
        try:
            parsed = int(max_procs_env)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        return 1
    return cpu_count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate verifier rewards by split groups generated under "
            "original-analysis/split."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="One run directory containing split outputs + trials results.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with original-analysis/split/*two-groups*.json will be considered."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: <run-dir>/original-analysis/score-by-split/"
            f"{DEFAULT_OUTPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered run directories and exit (only for --root-dir).",
    )
    parser.add_argument(
        "--max-procs",
        type=int,
        default=_default_max_procs(),
        help=(
            "Number of worker processes for --root-dir mode. "
            "Default: MAX_PROCS env var, else CPU count."
        ),
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl_dict_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        parsed = _string_or_none(item)
        if parsed is not None:
            output.append(parsed)
    return output


def _split_dir(run_dir: Path) -> Path:
    return run_dir / "original-analysis" / "split"


def _score_by_split_dir(run_dir: Path) -> Path:
    return run_dir / "original-analysis" / "score-by-split"


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (_score_by_split_dir(run_dir) / DEFAULT_OUTPUT_NAME).resolve()


def _extract_api_token_from_command(command: Any) -> str | None:
    if not isinstance(command, list):
        return None
    command_parts = [part for part in command if isinstance(part, str)]
    for index, part in enumerate(command_parts[:-1]):
        if part == "--api-token":
            candidate = _string_or_none(command_parts[index + 1])
            if candidate is not None:
                return candidate
    return None


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_reward_from_trial_result(path: Path) -> float | None:
    if not path.is_file():
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    verifier_result = payload.get("verifier_result")
    if not isinstance(verifier_result, dict):
        return None
    rewards_payload = verifier_result.get("rewards")
    if not isinstance(rewards_payload, dict):
        return None
    return _float_or_none(rewards_payload.get("reward"))


def _build_trial_info_by_hash(
    run_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    results_path = run_dir / "meta" / "results.json"
    if not results_path.is_file():
        raise ValueError(f"Missing required file: {results_path}")

    results_raw = _load_json(results_path)
    if not isinstance(results_raw, list):
        raise ValueError(f"results must be JSON array: {results_path}")

    trial_by_hash: dict[str, dict[str, Any]] = {}
    trial_rows: list[dict[str, Any]] = []

    for raw_entry in results_raw:
        if not isinstance(raw_entry, dict):
            continue

        trial_id = _string_or_none(raw_entry.get("trial_id"))
        if trial_id is None:
            continue

        api_token = _extract_api_token_from_command(raw_entry.get("command"))
        api_token_hash = _sha256_hex(api_token) if api_token is not None else None

        reward_path = run_dir / "trials" / trial_id / "result.json"
        reward_value = _extract_reward_from_trial_result(reward_path)

        row = {
            "trial_id": trial_id,
            "api_token_hash": api_token_hash,
            "reward": reward_value,
            "result_path": str(reward_path.resolve()),
            "result_exists": reward_path.is_file(),
        }
        trial_rows.append(row)

        if api_token_hash is not None and api_token_hash not in trial_by_hash:
            trial_by_hash[api_token_hash] = row

    return trial_by_hash, trial_rows


def _api_token_hash_from_trail_manifest(trail_dir: Path) -> str | None:
    manifest_path = trail_dir / "manifest.json"
    if manifest_path.is_file():
        payload = _load_json(manifest_path)
        if isinstance(payload, dict):
            parsed = _string_or_none(payload.get("api_token_hash"))
            if parsed is not None:
                return parsed.lower()

    lifecycle_path = trail_dir / "events" / "lifecycle.jsonl"
    if lifecycle_path.is_file():
        for record in _iter_jsonl_dict_records(lifecycle_path):
            parsed = _string_or_none(record.get("api_token_hash"))
            if parsed is not None:
                return parsed.lower()

    return None


def _api_token_hash_prefix_from_run_name(run_name: str) -> str | None:
    matched = _RUN_NAME_HASH_PATTERN.match(run_name)
    if matched is None:
        return None
    return matched.group(1).lower()


def _trail_dir_from_name(run_dir: Path, trail_name: str) -> Path:
    gateway_output_dir = run_dir / "gateway-output"
    trail_parts = [part for part in trail_name.split("/") if part]
    if not trail_parts:
        return gateway_output_dir
    return gateway_output_dir.joinpath(*trail_parts)


def _resolve_trial_for_trail(
    run_dir: Path,
    trail_name: str,
    *,
    trial_by_hash: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    trail_dir = _trail_dir_from_name(run_dir, trail_name)
    if not trail_dir.is_dir():
        return None

    token_hash = _api_token_hash_from_trail_manifest(trail_dir)
    if token_hash is not None:
        direct_match = trial_by_hash.get(token_hash)
        if direct_match is not None:
            return direct_match

    hash_prefix = _api_token_hash_prefix_from_run_name(trail_dir.name)
    if hash_prefix is None:
        return None

    candidates = [
        row for token_hash_key, row in trial_by_hash.items() if token_hash_key.startswith(hash_prefix)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _summarize_group(
    run_dir: Path,
    trail_names: list[str],
    *,
    trial_by_hash: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    trail_count = len(trail_names)
    resolved_trial_count = 0
    trial_with_score_count = 0
    score = 0.0

    for trail_name in trail_names:
        trial_row = _resolve_trial_for_trail(
            run_dir,
            trail_name,
            trial_by_hash=trial_by_hash,
        )
        if trial_row is None:
            continue

        resolved_trial_count += 1
        reward = _float_or_none(trial_row.get("reward"))
        if reward is None:
            continue

        trial_with_score_count += 1
        score += reward

    score_rate = (score / trail_count) if trail_count > 0 else None
    score_rate_scored_only = (
        (score / trial_with_score_count) if trial_with_score_count > 0 else None
    )

    return {
        "trail_count": trail_count,
        "resolved_trial_count": resolved_trial_count,
        "missing_trial_count": trail_count - resolved_trial_count,
        "trial_with_score_count": trial_with_score_count,
        "missing_score_count": resolved_trial_count - trial_with_score_count,
        "score": score,
        "score_rate": score_rate,
        "score_rate_scored_only": score_rate_scored_only,
    }


def _split_input_specs() -> list[tuple[str, str, str]]:
    return [
        ("token_usage", MODE_NORMAL, NORMAL_TOKEN_TWO_GROUP_NAME),
        ("context_usage", MODE_NORMAL, NORMAL_CONTEXT_TWO_GROUP_NAME),
        ("token_usage", MODE_STRICT_499, STRICT_TOKEN_TWO_GROUP_NAME),
        ("context_usage", MODE_STRICT_499, STRICT_CONTEXT_TWO_GROUP_NAME),
    ]


def _extract_split_score_for_file(
    run_dir: Path,
    split_file_path: Path,
    *,
    metric: str,
    mode: str,
    trial_by_hash: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = _load_json(split_file_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Split payload must be a JSON object: {split_file_path}")

    group_top_payload = payload.get("group_top")
    group_rest_payload = payload.get("group_rest")
    if not isinstance(group_top_payload, dict) or not isinstance(group_rest_payload, dict):
        raise ValueError(
            "Split payload must contain group_top/group_rest objects: "
            f"{split_file_path}"
        )

    top_trail_names = _string_list(group_top_payload.get("trail_names"))
    rest_trail_names = _string_list(group_rest_payload.get("trail_names"))

    group_top = _summarize_group(
        run_dir,
        top_trail_names,
        trial_by_hash=trial_by_hash,
    )
    group_rest = _summarize_group(
        run_dir,
        rest_trail_names,
        trial_by_hash=trial_by_hash,
    )

    top_rate = _float_or_none(group_top.get("score_rate"))
    rest_rate = _float_or_none(group_rest.get("score_rate"))

    score_gap = _float_or_none(group_top.get("score"))
    if score_gap is not None:
        score_gap -= _float_or_none(group_rest.get("score")) or 0.0

    score_rate_gap = None
    if top_rate is not None and rest_rate is not None:
        score_rate_gap = top_rate - rest_rate

    return {
        "metric": metric,
        "mode": mode,
        "source_split_file": str(split_file_path.resolve()),
        "selection": payload.get("selection"),
        "trail_count_ranked": payload.get("trail_count_ranked"),
        "group_top": group_top,
        "group_rest": group_rest,
        "score_gap": score_gap,
        "score_rate_gap": score_rate_gap,
    }


def extract_score_by_split_from_run_dir(run_dir: Path) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    split_dir = _split_dir(resolved_run_dir)
    if not split_dir.is_dir():
        raise ValueError(f"Missing required directory: {split_dir}")

    trial_by_hash, trial_rows = _build_trial_info_by_hash(resolved_run_dir)

    scored_rewards = [
        reward
        for reward in (_float_or_none(row.get("reward")) for row in trial_rows)
        if reward is not None
    ]
    run_score = sum(scored_rewards)

    split_scores: dict[str, dict[str, dict[str, Any] | None]] = {
        "token_usage": {MODE_NORMAL: None, MODE_STRICT_499: None},
        "context_usage": {MODE_NORMAL: None, MODE_STRICT_499: None},
    }

    for metric, mode, file_name in _split_input_specs():
        split_file_path = split_dir / file_name
        if not split_file_path.is_file():
            continue
        split_scores[metric][mode] = _extract_split_score_for_file(
            resolved_run_dir,
            split_file_path,
            metric=metric,
            mode=mode,
            trial_by_hash=trial_by_hash,
        )

    discovered_split_files = [
        str((split_dir / file_name).resolve())
        for _, _, file_name in _split_input_specs()
        if (split_dir / file_name).is_file()
    ]
    if not discovered_split_files:
        raise ValueError(
            "No split two-group files found under original-analysis/split. "
            f"Checked: {[file_name for _, _, file_name in _split_input_specs()]}"
        )

    return {
        "source_run_dir": str(resolved_run_dir),
        "source_split_dir": str(split_dir.resolve()),
        "source_results_path": str((resolved_run_dir / "meta" / "results.json").resolve()),
        "source_trials_dir": str((resolved_run_dir / "trials").resolve()),
        "discovered_split_files": discovered_split_files,
        "run_trial_count": len(trial_rows),
        "run_trial_with_score_count": len(scored_rewards),
        "run_missing_score_count": len(trial_rows) - len(scored_rewards),
        "run_score": run_score,
        "run_score_rate": (run_score / len(trial_rows)) if trial_rows else None,
        "run_score_rate_scored_only": (
            (run_score / len(scored_rewards)) if scored_rewards else None
        ),
        "split_scores": split_scores,
    }


def extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_output_path = (
        output_path or _default_output_path_for_run(resolved_run_dir)
    ).expanduser().resolve()

    payload = extract_score_by_split_from_run_dir(resolved_run_dir)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved_output_path


def discover_run_dirs_with_split(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    split_file_names = {file_name for _, _, file_name in _split_input_specs()}

    for file_name in split_file_names:
        for split_path in root_dir.rglob(file_name):
            if not split_path.is_file():
                continue
            if split_path.parent.name != "split":
                continue
            if split_path.parent.parent.name != "original-analysis":
                continue
            run_dirs.add(split_path.parent.parent.parent.resolve())
    return sorted(run_dirs)


def _extract_run_dir_worker(run_dir_text: str) -> tuple[str, str | None, str | None]:
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        output_path = extract_run_dir(run_dir)
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(output_path), None)


def _run_root_dir_sequential(run_dirs: list[Path]) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            output_path = extract_run_dir(run_dir)
            print(f"[done] {run_dir} -> {output_path}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(run_dirs: list[Path], *, max_procs: int) -> int:
    failure_count = 0
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_path_text, error_text in executor.map(
            _extract_run_dir_worker,
            [str(run_dir) for run_dir in run_dirs],
        ):
            if error_text is None:
                print(f"[done] {run_dir_text} -> {output_path_text}")
            else:
                failure_count += 1
                print(f"[error] {run_dir_text}: {error_text}", file=sys.stderr)
    return failure_count


def _build_score_table_row(
    run_dir: Path,
    *,
    metric: str,
    mode: str,
    summary_output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, str] | None:
    summary_path = (_score_by_split_dir(run_dir) / summary_output_name).resolve()
    if not summary_path.is_file():
        print(f"[warn] Missing summary file for table row: {summary_path}", file=sys.stderr)
        return None

    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        print(f"[warn] Invalid summary payload (not object): {summary_path}", file=sys.stderr)
        return None

    split_scores = payload.get("split_scores")
    if not isinstance(split_scores, dict):
        print(
            f"[warn] Missing split_scores in summary payload: {summary_path}",
            file=sys.stderr,
        )
        return None

    metric_payload = split_scores.get(metric)
    if not isinstance(metric_payload, dict):
        print(
            f"[warn] Missing split_scores.{metric} in summary payload: {summary_path}",
            file=sys.stderr,
        )
        return None

    mode_payload = metric_payload.get(mode)
    if not isinstance(mode_payload, dict):
        print(
            f"[warn] Missing split_scores.{metric}.{mode} in summary payload: {summary_path}",
            file=sys.stderr,
        )
        return None

    group_top = mode_payload.get("group_top")
    group_rest = mode_payload.get("group_rest")
    if not isinstance(group_top, dict) or not isinstance(group_rest, dict):
        print(
            f"[warn] Missing group_top/group_rest for {metric}/{mode}: {summary_path}",
            file=sys.stderr,
        )
        return None

    selection_payload = mode_payload.get("selection")
    selected_p = ""
    if isinstance(selection_payload, dict):
        selected_p = str(selection_payload.get("selected_p", ""))

    return {
        "run_path": str(run_dir.resolve()),
        "selected_p": selected_p,
        "group_top_trail_count": str(group_top.get("trail_count", "")),
        "group_top_score": str(group_top.get("score", "")),
        "group_top_score_rate": str(group_top.get("score_rate", "")),
        "group_rest_trail_count": str(group_rest.get("trail_count", "")),
        "group_rest_score": str(group_rest.get("score", "")),
        "group_rest_score_rate": str(group_rest.get("score_rate", "")),
        "score_gap": str(mode_payload.get("score_gap", "")),
        "score_rate_gap": str(mode_payload.get("score_rate_gap", "")),
    }


def _write_root_score_table(
    *,
    root_dir: Path,
    run_dirs: list[Path],
    metric: str,
    mode: str,
    output_name: str,
) -> Path:
    rows: list[dict[str, str]] = []
    for run_dir in run_dirs:
        row = _build_score_table_row(run_dir, metric=metric, mode=mode)
        if row is not None:
            rows.append(row)

    output_path = (root_dir / output_name).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_path",
                "selected_p",
                "group_top_trail_count",
                "group_top_score",
                "group_top_score_rate",
                "group_rest_trail_count",
                "group_rest_score",
                "group_rest_score_rate",
                "score_gap",
                "score_rate_gap",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def _write_root_score_tables(root_dir: Path, run_dirs: list[Path]) -> tuple[Path, Path, Path, Path]:
    token_table_path = _write_root_score_table(
        root_dir=root_dir,
        run_dirs=run_dirs,
        metric="token_usage",
        mode=MODE_NORMAL,
        output_name=ROOT_TOKEN_TABLE_NAME,
    )
    context_table_path = _write_root_score_table(
        root_dir=root_dir,
        run_dirs=run_dirs,
        metric="context_usage",
        mode=MODE_NORMAL,
        output_name=ROOT_CONTEXT_TABLE_NAME,
    )
    strict_token_table_path = _write_root_score_table(
        root_dir=root_dir,
        run_dirs=run_dirs,
        metric="token_usage",
        mode=MODE_STRICT_499,
        output_name=ROOT_STRICT_TOKEN_TABLE_NAME,
    )
    strict_context_table_path = _write_root_score_table(
        root_dir=root_dir,
        run_dirs=run_dirs,
        metric="context_usage",
        mode=MODE_STRICT_499,
        output_name=ROOT_STRICT_CONTEXT_TABLE_NAME,
    )
    return (
        token_table_path,
        context_table_path,
        strict_token_table_path,
        strict_context_table_path,
    )


def _main_run_dir(args: argparse.Namespace) -> int:
    if args.dry_run:
        raise ValueError("--dry-run can only be used with --root-dir")
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    resolved_output_path = extract_run_dir(run_dir, output_path=output_path)
    print(str(resolved_output_path))
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output:
        raise ValueError("--output can only be used with --run-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_split(root_dir)
    print(f"Discovered {len(run_dirs)} run directories under {root_dir}")
    if not run_dirs:
        return 0
    if args.dry_run:
        for run_dir in run_dirs:
            print(str(run_dir))
        return 0

    worker_count = min(args.max_procs, len(run_dirs))
    print(f"Running extraction with {worker_count} worker process(es)")

    if worker_count <= 1:
        failure_count = _run_root_dir_sequential(run_dirs)
    else:
        try:
            failure_count = _run_root_dir_parallel(run_dirs, max_procs=worker_count)
        except (PermissionError, OSError) as exc:
            print(
                f"[warn] Unable to start process pool ({exc}); falling back to sequential.",
                file=sys.stderr,
            )
            failure_count = _run_root_dir_sequential(run_dirs)

    if failure_count:
        print(
            f"Completed with {failure_count} failure(s) out of {len(run_dirs)} run directories.",
            file=sys.stderr,
        )

    (
        token_table_path,
        context_table_path,
        strict_token_table_path,
        strict_context_table_path,
    ) = _write_root_score_tables(root_dir, run_dirs)

    print(f"Wrote token score table: {token_table_path}")
    print(f"Wrote context score table: {context_table_path}")
    print(f"Wrote strict token score table: {strict_token_table_path}")
    print(f"Wrote strict context score table: {strict_context_table_path}")

    if failure_count:
        return 1

    print(f"Completed extraction for {len(run_dirs)} run directories.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run_dir:
        return _main_run_dir(args)
    return _main_root_dir(args)


if __name__ == "__main__":
    raise SystemExit(main())
