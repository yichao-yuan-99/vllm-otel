from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any
from typing import Iterable


DEFAULT_OUTPUT_NAME = "top-p-usage-ratio-summary.json"
TOKEN_TWO_GROUP_OUTPUT_NAME = "top-p-token-usage-two-groups.json"
CONTEXT_TWO_GROUP_OUTPUT_NAME = "top-p-context-usage-two-groups.json"
ROOT_TOKEN_TABLE_NAME = "split-top-p-token-usage-two-group-table.csv"
ROOT_CONTEXT_TABLE_NAME = "split-top-p-context-usage-two-group-table.csv"
STRICT_499_SUFFIX = "strict-499"
STRICT_OUTPUT_NAME = f"top-p-usage-ratio-summary.{STRICT_499_SUFFIX}.json"
STRICT_TOKEN_TWO_GROUP_OUTPUT_NAME = (
    f"top-p-token-usage-two-groups.{STRICT_499_SUFFIX}.json"
)
STRICT_CONTEXT_TWO_GROUP_OUTPUT_NAME = (
    f"top-p-context-usage-two-groups.{STRICT_499_SUFFIX}.json"
)
ROOT_STRICT_TOKEN_TABLE_NAME = (
    f"split-top-p-token-usage-two-group-table.{STRICT_499_SUFFIX}.csv"
)
ROOT_STRICT_CONTEXT_TABLE_NAME = (
    f"split-top-p-context-usage-two-group-table.{STRICT_499_SUFFIX}.csv"
)
UNRANKED_MODE_NORMAL = "normal"
UNRANKED_MODE_STRICT_499 = "strict_499"
DEFAULT_P_MIN = 1
DEFAULT_P_MAX = 99

_INTEGER_PATTERN = re.compile(r"[-+]?\d+")


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
            "Analyze top-p trail share of token usage and context usage "
            "for p=1..99."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        default=None,
        help="Run result root directory containing gateway-output/.",
    )
    target_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with a direct gateway-output/ child will be processed."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Default: <run-dir>/original-analysis/split/"
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


def _iter_jsonl_dict_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                yield payload


def _profile_id_from_name(name: str) -> int | None:
    prefix = "profile-"
    if not name.startswith(prefix):
        return None
    raw = name[len(prefix) :]
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def discover_gateway_run_dirs(gateway_output_dir: Path) -> list[tuple[Path, int | None]]:
    run_dirs: list[tuple[Path, int | None]] = []

    for run_dir in sorted(gateway_output_dir.glob("run_*")):
        if run_dir.is_dir():
            run_dirs.append((run_dir, None))

    for child in sorted(gateway_output_dir.iterdir()):
        if not child.is_dir():
            continue
        profile_id = _profile_id_from_name(child.name)
        if profile_id is None:
            continue
        for run_dir in sorted(child.glob("run_*")):
            if run_dir.is_dir():
                run_dirs.append((run_dir, profile_id))
    return run_dirs


def discover_run_dirs_with_gateway_output(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for gateway_output_dir in root_dir.rglob("gateway-output"):
        if not gateway_output_dir.is_dir():
            continue
        run_dirs.add(gateway_output_dir.parent.resolve())
    return sorted(run_dirs)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if _INTEGER_PATTERN.fullmatch(stripped):
            try:
                return int(stripped)
            except ValueError:
                return None
    return None


def _extract_usage_tokens(record: dict[str, Any]) -> tuple[int | None, int | None, int]:
    response = record.get("response")
    if not isinstance(response, dict):
        return None, None, 0
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None, None, 0

    prompt_tokens = _int_or_none(usage.get("prompt_tokens"))
    decode_tokens = _int_or_none(usage.get("completion_tokens"))

    cached_tokens = 0
    prompt_tokens_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_tokens_details, dict):
        parsed_cached = _int_or_none(prompt_tokens_details.get("cached_tokens"))
        if parsed_cached is not None:
            cached_tokens = parsed_cached
    return prompt_tokens, decode_tokens, cached_tokens


def _extract_status_code(record: dict[str, Any]) -> int | None:
    status_code = _int_or_none(record.get("status_code"))
    if status_code is not None:
        return status_code
    response_summary = record.get("response_summary")
    if isinstance(response_summary, dict):
        return _int_or_none(response_summary.get("status_code"))
    return None


def _extract_trail_stats(
    gateway_run_dir: Path,
    *,
    profile_id: int | None,
) -> dict[str, Any]:
    requests_path = gateway_run_dir / "requests" / "model_inference.jsonl"
    if not requests_path.is_file():
        raise ValueError(f"Missing required file: {requests_path}")

    request_count = 0
    trail_context_length: int | None = None
    trail_token_usage_total = 0
    requests_with_length = 0
    requests_with_token_usage = 0
    requests_with_status_499 = 0
    has_status_499 = False

    for record in _iter_jsonl_dict_records(requests_path):
        request_count += 1
        status_code = _extract_status_code(record)
        if status_code == 499:
            has_status_499 = True
            requests_with_status_499 += 1
        prompt_tokens, decode_tokens, cached_tokens = _extract_usage_tokens(record)
        if prompt_tokens is not None and decode_tokens is not None:
            request_length = prompt_tokens + decode_tokens
            if trail_context_length is None:
                trail_context_length = request_length
            else:
                trail_context_length = max(trail_context_length, request_length)
            requests_with_length += 1

            request_token_usage = prompt_tokens + decode_tokens - cached_tokens
            trail_token_usage_total += request_token_usage
            requests_with_token_usage += 1

    return {
        "gateway_run_id": gateway_run_dir.name,
        "gateway_profile_id": profile_id,
        "request_count": request_count,
        "context_length": trail_context_length,
        "token_usage": trail_token_usage_total,
        "requests_with_length": requests_with_length,
        "requests_with_token_usage": requests_with_token_usage,
        "has_status_499": has_status_499,
        "requests_with_status_499": requests_with_status_499,
    }


def _top_count_for_percentile(trail_count: int, p: int) -> int:
    if trail_count <= 0:
        return 0
    top_count = math.ceil((trail_count * p) / 100.0)
    top_count = max(1, top_count)
    top_count = min(trail_count, top_count)
    return top_count


def _build_percentile_rows(
    ranked_trails: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[float | None],
    list[float | None],
    list[float | None],
    list[float | None],
]:
    total_token_usage = sum(int(trail["token_usage"]) for trail in ranked_trails)
    total_context_length = sum(int(trail["context_length"]) for trail in ranked_trails)

    percentile_rows: list[dict[str, Any]] = []
    token_ratio_series: list[float | None] = []
    context_ratio_series: list[float | None] = []
    token_share_series: list[float | None] = []
    context_share_series: list[float | None] = []

    for p in range(DEFAULT_P_MIN, DEFAULT_P_MAX + 1):
        top_count = _top_count_for_percentile(len(ranked_trails), p)
        top_trails = ranked_trails[:top_count]
        rest_trails = ranked_trails[top_count:]

        top_token_usage = sum(int(trail["token_usage"]) for trail in top_trails)
        rest_token_usage = sum(int(trail["token_usage"]) for trail in rest_trails)
        top_context_length = sum(int(trail["context_length"]) for trail in top_trails)
        rest_context_length = sum(int(trail["context_length"]) for trail in rest_trails)

        token_share = None
        if total_token_usage > 0:
            token_share = top_token_usage / total_token_usage
        context_share = None
        if total_context_length > 0:
            context_share = top_context_length / total_context_length

        token_ratio = None
        if rest_token_usage > 0:
            token_ratio = top_token_usage / rest_token_usage
        context_ratio = None
        if rest_context_length > 0:
            context_ratio = top_context_length / rest_context_length

        percentile_rows.append(
            {
                "p": p,
                "top_trail_count": len(top_trails),
                "rest_trail_count": len(rest_trails),
                "top_token_usage_total": top_token_usage,
                "rest_token_usage_total": rest_token_usage,
                "top_context_length_total": top_context_length,
                "rest_context_length_total": rest_context_length,
                "top_token_usage_ratio": token_ratio,
                "top_context_usage_ratio": context_ratio,
                "top_token_usage_share": token_share,
                "top_context_usage_share": context_share,
            }
        )
        token_ratio_series.append(token_ratio)
        context_ratio_series.append(context_ratio)
        token_share_series.append(token_share)
        context_share_series.append(context_share)

    return (
        percentile_rows,
        token_ratio_series,
        context_ratio_series,
        token_share_series,
        context_share_series,
    )


def _trail_name(trail: dict[str, Any]) -> str:
    gateway_run_id = str(trail.get("gateway_run_id") or "")
    profile_id = trail.get("gateway_profile_id")
    if isinstance(profile_id, int):
        return f"profile-{profile_id}/{gateway_run_id}"
    return gateway_run_id


def _ratio_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _build_two_group_payload(
    *,
    ranked_trails: list[dict[str, Any]],
    percentile_rows: list[dict[str, Any]],
    ratio_field: str,
    metric_name: str,
) -> dict[str, Any]:
    selected_row: dict[str, Any] | None = None
    selection_reason = "fallback_first_percentile"

    for row in percentile_rows:
        ratio_value = _ratio_or_none(row.get(ratio_field))
        if ratio_value is not None and ratio_value > 1.0:
            selected_row = row
            selection_reason = "first_ratio_gt_1"
            break

    if selected_row is None:
        for row in percentile_rows:
            ratio_value = _ratio_or_none(row.get(ratio_field))
            if ratio_value is not None and ratio_value >= 1.0:
                selected_row = row
                selection_reason = "first_ratio_ge_1"
                break

    if selected_row is None and percentile_rows:
        selected_row = percentile_rows[0]

    trail_count_ranked = len(ranked_trails)
    if selected_row is None:
        selected_p = DEFAULT_P_MIN
        selected_top_count = 0
        selected_rest_count = 0
        selected_ratio = None
    else:
        selected_p = int(selected_row.get("p", DEFAULT_P_MIN))
        selected_top_count = int(selected_row.get("top_trail_count", 0))
        selected_rest_count = int(selected_row.get("rest_trail_count", 0))
        selected_ratio = _ratio_or_none(selected_row.get(ratio_field))

    selected_top_count = max(0, min(trail_count_ranked, selected_top_count))
    selected_rest_count = max(0, min(trail_count_ranked, selected_rest_count))

    top_group = ranked_trails[:selected_top_count]
    rest_group = ranked_trails[selected_top_count:]
    if trail_count_ranked > 0 and len(rest_group) != selected_rest_count:
        selected_rest_count = len(rest_group)

    def _percentage(count: int) -> float:
        if trail_count_ranked <= 0:
            return 0.0
        return round((count * 100.0) / float(trail_count_ranked), 6)

    return {
        "metric": metric_name,
        "ratio_field": ratio_field,
        "selection": {
            "criterion": "first_ratio_gt_1_then_ge_1_then_fallback_p1",
            "selected_p": selected_p,
            "selected_ratio": selected_ratio,
            "selected_top_trial_count": len(top_group),
            "selected_rest_trial_count": len(rest_group),
            "selection_reason": selection_reason,
        },
        "trail_count_ranked": trail_count_ranked,
        "group_top": {
            "trial_count": len(top_group),
            "trial_percentage": _percentage(len(top_group)),
            "trail_names": [_trail_name(trail) for trail in top_group],
        },
        "group_rest": {
            "trial_count": len(rest_group),
            "trial_percentage": _percentage(len(rest_group)),
            "trail_names": [_trail_name(trail) for trail in rest_group],
        },
    }


def _load_trails_from_run_dir(run_dir: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    resolved_run_dir = run_dir.expanduser().resolve()
    gateway_output_dir = resolved_run_dir / "gateway-output"
    if not gateway_output_dir.is_dir():
        raise ValueError(f"Missing required directory: {gateway_output_dir}")

    discovered_run_dirs = discover_gateway_run_dirs(gateway_output_dir)
    if not discovered_run_dirs:
        raise ValueError(
            "No run_* artifacts found under gateway-output. "
            "Expected either gateway-output/run_* or gateway-output/profile-*/run_*."
        )

    trails: list[dict[str, Any]] = []
    for gateway_run_dir, profile_id in discovered_run_dirs:
        trails.append(_extract_trail_stats(gateway_run_dir, profile_id=profile_id))
    return resolved_run_dir, gateway_output_dir.resolve(), trails


def _is_trail_unranked(
    trail: dict[str, Any],
    *,
    unranked_mode: str,
) -> bool:
    if not isinstance(trail.get("context_length"), int):
        return True
    if unranked_mode == UNRANKED_MODE_NORMAL:
        return False
    if unranked_mode == UNRANKED_MODE_STRICT_499:
        return bool(trail.get("has_status_499"))
    raise ValueError(f"Unsupported unranked mode: {unranked_mode}")


def _split_ranked_and_unranked_trails(
    trails: list[dict[str, Any]],
    *,
    unranked_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked_trails: list[dict[str, Any]] = []
    unranked_trails: list[dict[str, Any]] = []
    for trail in trails:
        trail_copy = dict(trail)
        if _is_trail_unranked(trail_copy, unranked_mode=unranked_mode):
            unranked_trails.append(trail_copy)
        else:
            ranked_trails.append(trail_copy)
    return ranked_trails, unranked_trails


def _unranked_criteria_description(unranked_mode: str) -> str:
    if unranked_mode == UNRANKED_MODE_NORMAL:
        return "trails without any valid request length (prompt_tokens + completion_tokens)"
    if unranked_mode == UNRANKED_MODE_STRICT_499:
        return (
            "trails without valid request length OR trails containing any request with "
            "status_code=499"
        )
    raise ValueError(f"Unsupported unranked mode: {unranked_mode}")


def _build_split_payload_from_trails(
    *,
    resolved_run_dir: Path,
    gateway_output_dir: Path,
    trails: list[dict[str, Any]],
    unranked_mode: str,
) -> dict[str, Any]:
    ranked_trails, unranked_trails = _split_ranked_and_unranked_trails(
        trails,
        unranked_mode=unranked_mode,
    )

    ranked_trails.sort(
        key=lambda trail: (
            -int(trail["context_length"]),
            str(trail.get("gateway_run_id") or ""),
        )
    )
    for index, trail in enumerate(ranked_trails, start=1):
        trail["context_rank"] = index

    (
        percentile_rows,
        token_ratio_series,
        context_ratio_series,
        token_share_series,
        context_share_series,
    ) = _build_percentile_rows(ranked_trails)

    token_two_group = _build_two_group_payload(
        ranked_trails=ranked_trails,
        percentile_rows=percentile_rows,
        ratio_field="top_token_usage_ratio",
        metric_name="token_usage",
    )
    context_two_group = _build_two_group_payload(
        ranked_trails=ranked_trails,
        percentile_rows=percentile_rows,
        ratio_field="top_context_usage_ratio",
        metric_name="context_usage",
    )

    trail_count_with_status_499 = sum(1 for trail in trails if bool(trail.get("has_status_499")))

    return {
        "source_run_dir": str(resolved_run_dir),
        "source_gateway_output_dir": str(gateway_output_dir),
        "trail_count_total": len(trails),
        "trail_count_ranked": len(ranked_trails),
        "trail_count_unranked": len(unranked_trails),
        "trail_count_with_status_499": trail_count_with_status_499,
        "unranked_mode": unranked_mode,
        "unranked_criteria": _unranked_criteria_description(unranked_mode),
        "ratio_definition": "top/rest",
        "share_definition": "top/total",
        "percentiles": list(range(DEFAULT_P_MIN, DEFAULT_P_MAX + 1)),
        "table_2x99": {
            "top_p_token_usage_ratio": token_ratio_series,
            "top_p_context_usage_ratio": context_ratio_series,
            "top_p_token_usage_share": token_share_series,
            "top_p_context_usage_share": context_share_series,
        },
        "by_percentile": percentile_rows,
        "ranked_trails": ranked_trails,
        "unranked_trails": unranked_trails,
        "two_group_summaries": {
            "token_usage": token_two_group,
            "context_usage": context_two_group,
        },
    }


def extract_split_from_run_dir(run_dir: Path) -> dict[str, Any]:
    resolved_run_dir, gateway_output_dir, trails = _load_trails_from_run_dir(run_dir)
    return _build_split_payload_from_trails(
        resolved_run_dir=resolved_run_dir,
        gateway_output_dir=gateway_output_dir,
        trails=trails,
        unranked_mode=UNRANKED_MODE_NORMAL,
    )


def extract_split_strict_from_run_dir(run_dir: Path) -> dict[str, Any]:
    resolved_run_dir, gateway_output_dir, trails = _load_trails_from_run_dir(run_dir)
    return _build_split_payload_from_trails(
        resolved_run_dir=resolved_run_dir,
        gateway_output_dir=gateway_output_dir,
        trails=trails,
        unranked_mode=UNRANKED_MODE_STRICT_499,
    )


def _default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "original-analysis" / "split" / DEFAULT_OUTPUT_NAME).resolve()


def _strict_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "original-analysis" / "split" / STRICT_OUTPUT_NAME).resolve()


def _two_group_output_paths(output_dir: Path, *, strict_499: bool) -> tuple[Path, Path]:
    if strict_499:
        token_name = STRICT_TOKEN_TWO_GROUP_OUTPUT_NAME
        context_name = STRICT_CONTEXT_TWO_GROUP_OUTPUT_NAME
    else:
        token_name = TOKEN_TWO_GROUP_OUTPUT_NAME
        context_name = CONTEXT_TWO_GROUP_OUTPUT_NAME
    return (output_dir / token_name).resolve(), (output_dir / context_name).resolve()


def extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
    resolved_output_path = (output_path or _default_output_path_for_run(run_dir)).expanduser().resolve()
    output_dir = resolved_output_path.parent
    if output_path is None:
        strict_output_path = _strict_output_path_for_run(run_dir)
    else:
        strict_output_path = (output_dir / STRICT_OUTPUT_NAME).resolve()
    token_output_path, context_output_path = _two_group_output_paths(
        output_dir,
        strict_499=False,
    )
    strict_token_output_path, strict_context_output_path = _two_group_output_paths(
        output_dir,
        strict_499=True,
    )

    resolved_run_dir, gateway_output_dir, trails = _load_trails_from_run_dir(run_dir)
    result = _build_split_payload_from_trails(
        resolved_run_dir=resolved_run_dir,
        gateway_output_dir=gateway_output_dir,
        trails=trails,
        unranked_mode=UNRANKED_MODE_NORMAL,
    )
    strict_result = _build_split_payload_from_trails(
        resolved_run_dir=resolved_run_dir,
        gateway_output_dir=gateway_output_dir,
        trails=trails,
        unranked_mode=UNRANKED_MODE_STRICT_499,
    )

    result["additional_outputs"] = {
        "token_usage_two_groups": str(token_output_path),
        "context_usage_two_groups": str(context_output_path),
        "strict_499_summary": str(strict_output_path),
        "strict_499_token_usage_two_groups": str(strict_token_output_path),
        "strict_499_context_usage_two_groups": str(strict_context_output_path),
    }
    strict_result["additional_outputs"] = {
        "token_usage_two_groups": str(strict_token_output_path),
        "context_usage_two_groups": str(strict_context_output_path),
        "normal_summary": str(resolved_output_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    strict_output_path.write_text(
        json.dumps(strict_result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    token_payload = result.get("two_group_summaries", {}).get("token_usage")
    context_payload = result.get("two_group_summaries", {}).get("context_usage")
    strict_token_payload = strict_result.get("two_group_summaries", {}).get("token_usage")
    strict_context_payload = strict_result.get("two_group_summaries", {}).get(
        "context_usage"
    )
    token_output_path.write_text(
        json.dumps(token_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    context_output_path.write_text(
        json.dumps(context_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    strict_token_output_path.write_text(
        json.dumps(strict_token_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    strict_context_output_path.write_text(
        json.dumps(strict_context_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved_output_path


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


def _build_two_group_table_row(
    run_dir: Path,
    *,
    metric_key: str,
    summary_output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, str] | None:
    summary_path = (run_dir / "original-analysis" / "split" / summary_output_name).resolve()
    if not summary_path.is_file():
        print(f"[warn] Missing summary file for table row: {summary_path}", file=sys.stderr)
        return None

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        print(f"[warn] Invalid summary payload (not object): {summary_path}", file=sys.stderr)
        return None
    two_group_summaries = payload.get("two_group_summaries")
    if not isinstance(two_group_summaries, dict):
        print(
            f"[warn] Missing two_group_summaries in summary payload: {summary_path}",
            file=sys.stderr,
        )
        return None
    metric_payload = two_group_summaries.get(metric_key)
    if not isinstance(metric_payload, dict):
        print(
            f"[warn] Missing two_group_summaries.{metric_key} in summary payload: {summary_path}",
            file=sys.stderr,
        )
        return None

    group_top = metric_payload.get("group_top")
    group_rest = metric_payload.get("group_rest")
    if not isinstance(group_top, dict) or not isinstance(group_rest, dict):
        print(
            f"[warn] Missing group_top/group_rest for {metric_key}: {summary_path}",
            file=sys.stderr,
        )
        return None

    return {
        "run_path": str(run_dir.resolve()),
        "group_top_trail_count": str(group_top.get("trial_count", "")),
        "group_top_trail_percentage": str(group_top.get("trial_percentage", "")),
        "group_rest_trail_count": str(group_rest.get("trial_count", "")),
        "group_rest_trail_percentage": str(group_rest.get("trial_percentage", "")),
    }


def _write_root_two_group_table(
    *,
    root_dir: Path,
    run_dirs: list[Path],
    metric_key: str,
    output_name: str,
    summary_output_name: str = DEFAULT_OUTPUT_NAME,
) -> Path:
    rows: list[dict[str, str]] = []
    for run_dir in run_dirs:
        row = _build_two_group_table_row(
            run_dir,
            metric_key=metric_key,
            summary_output_name=summary_output_name,
        )
        if row is not None:
            rows.append(row)

    output_path = (root_dir / output_name).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_path",
                "group_top_trail_count",
                "group_top_trail_percentage",
                "group_rest_trail_count",
                "group_rest_trail_percentage",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def _write_root_two_group_tables(root_dir: Path, run_dirs: list[Path]) -> tuple[Path, Path]:
    token_table_path = _write_root_two_group_table(
        root_dir=root_dir,
        run_dirs=run_dirs,
        metric_key="token_usage",
        output_name=ROOT_TOKEN_TABLE_NAME,
    )
    context_table_path = _write_root_two_group_table(
        root_dir=root_dir,
        run_dirs=run_dirs,
        metric_key="context_usage",
        output_name=ROOT_CONTEXT_TABLE_NAME,
    )
    _write_root_two_group_table(
        root_dir=root_dir,
        run_dirs=run_dirs,
        metric_key="token_usage",
        output_name=ROOT_STRICT_TOKEN_TABLE_NAME,
        summary_output_name=STRICT_OUTPUT_NAME,
    )
    _write_root_two_group_table(
        root_dir=root_dir,
        run_dirs=run_dirs,
        metric_key="context_usage",
        output_name=ROOT_STRICT_CONTEXT_TABLE_NAME,
        summary_output_name=STRICT_OUTPUT_NAME,
    )
    return token_table_path, context_table_path


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

    run_dirs = discover_run_dirs_with_gateway_output(root_dir)
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
    token_table_path, context_table_path = _write_root_two_group_tables(root_dir, run_dirs)
    strict_token_table_path = (root_dir / ROOT_STRICT_TOKEN_TABLE_NAME).resolve()
    strict_context_table_path = (root_dir / ROOT_STRICT_CONTEXT_TABLE_NAME).resolve()
    print(f"Wrote token split table: {token_table_path}")
    print(f"Wrote context split table: {context_table_path}")
    print(f"Wrote strict token split table: {strict_token_table_path}")
    print(f"Wrote strict context split table: {strict_context_table_path}")
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
