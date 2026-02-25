"""Validation script for replay outputs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object in {path}, got {type(parsed)!r}")
        rows.append(parsed)
    return rows


def require_file(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Missing required file: {path}")
    if not path.is_file():
        raise ValueError(f"Expected file path: {path}")


def extract_response_text(response_payload: Any) -> str:
    if isinstance(response_payload, str):
        return response_payload
    if isinstance(response_payload, dict):
        choices = response_payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                text_value = first.get("text")
                if isinstance(text_value, str):
                    return text_value
        output_text = response_payload.get("output_text")
        if isinstance(output_text, str):
            return output_text
    raise ValueError("Unable to extract response text")


def extract_prompt_content(request_payload: Any) -> tuple[str, Any]:
    if not isinstance(request_payload, dict):
        raise ValueError("Request payload is not a JSON object")

    if "messages" in request_payload:
        messages = request_payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Request field 'messages' is not a list")
        return "messages", messages

    if "prompt" in request_payload:
        return "prompt", request_payload.get("prompt")

    if "input" in request_payload:
        return "input", request_payload.get("input")

    raise ValueError("Unable to extract prompt content from request payload")


def maybe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def extract_usage(response_payload: Any) -> dict[str, int | None]:
    usage_prompt: int | None = None
    usage_completion: int | None = None
    usage_cached: int | None = None

    if isinstance(response_payload, dict):
        usage = response_payload.get("usage")
        if isinstance(usage, dict):
            usage_prompt = maybe_int(usage.get("prompt_tokens"))
            usage_completion = maybe_int(usage.get("completion_tokens"))
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                usage_cached = maybe_int(details.get("cached_tokens"))

    return {
        "prompt_tokens": usage_prompt,
        "completion_tokens": usage_completion,
        "cached_tokens": usage_cached,
    }


def relative_error(source: float, replay: float) -> float:
    source_abs = abs(source)
    if source_abs == 0:
        return 0.0 if replay == 0 else float("inf")
    return abs(replay - source) / source_abs


def summarize_errors(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "avg": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "avg": statistics.fmean(values),
        "max": max(values),
    }


@dataclass
class RunData:
    run_dir: Path
    api_token_hash: str
    run_start: datetime
    run_end: datetime
    requests: list[dict[str, Any]]

    @property
    def run_duration_s(self) -> float:
        return max(0.0, (self.run_end - self.run_start).total_seconds())


def load_gateway_runs(root: Path) -> dict[str, RunData]:
    gateway_output_dir = root / "gateway-output"
    if not gateway_output_dir.exists() or not gateway_output_dir.is_dir():
        raise ValueError(f"Missing gateway-output directory: {gateway_output_dir}")

    result: dict[str, RunData] = {}
    for run_dir in sorted(gateway_output_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        requests_path = run_dir / "requests" / "model_inference.jsonl"
        require_file(manifest_path)
        require_file(requests_path)
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid manifest payload in {manifest_path}")

        api_token_hash = manifest.get("api_token_hash")
        run_start_raw = manifest.get("run_start_time")
        run_end_raw = manifest.get("run_end_time")
        if not isinstance(api_token_hash, str) or not api_token_hash:
            raise ValueError(f"Invalid api_token_hash in {manifest_path}")
        if not isinstance(run_start_raw, str) or not run_start_raw:
            raise ValueError(f"Invalid run_start_time in {manifest_path}")
        if not isinstance(run_end_raw, str) or not run_end_raw:
            raise ValueError(f"Invalid run_end_time in {manifest_path}")

        run_start = parse_iso8601(run_start_raw)
        run_end = parse_iso8601(run_end_raw)
        if run_end < run_start:
            raise ValueError(f"run_end_time < run_start_time in {manifest_path}")

        requests = read_jsonl(requests_path)
        requests.sort(
            key=lambda row: row.get("request_id", "")
            if not isinstance(row.get("request_start_time"), str)
            else row.get("request_start_time")
        )

        result[api_token_hash] = RunData(
            run_dir=run_dir,
            api_token_hash=api_token_hash,
            run_start=run_start,
            run_end=run_end,
            requests=requests,
        )
    if not result:
        raise ValueError(f"No run_* artifacts found under: {gateway_output_dir}")
    return result


def request_duration_ms(record: dict[str, Any]) -> float:
    raw = record.get("request_duration_ms")
    if raw is None:
        raw = record.get("duration_ms")
    if isinstance(raw, (int, float)):
        return float(raw)
    raise ValueError("Missing request_duration_ms/duration_ms")


def request_error_context(
    *,
    token_hash: str,
    idx: int,
    source_req: dict[str, Any],
    replay_req: dict[str, Any],
) -> str:
    source_request_id = source_req.get("request_id")
    replay_request_id = replay_req.get("request_id")
    return (
        f"{token_hash[:12]} req#{idx} "
        f"request_id={replay_request_id!r} "
        f"original_request_id={source_request_id!r}"
    )


def validate(
    *,
    source_job_dir: Path,
    replay_run_dir: Path,
    report_out: Path | None,
) -> tuple[int, dict[str, Any]]:
    source_runs = load_gateway_runs(source_job_dir)
    replay_runs = load_gateway_runs(replay_run_dir)

    source_hashes = set(source_runs.keys())
    replay_hashes = set(replay_runs.keys())

    exact_failures: list[str] = []
    if source_hashes != replay_hashes:
        missing = sorted(source_hashes - replay_hashes)
        extra = sorted(replay_hashes - source_hashes)
        if missing:
            exact_failures.append(f"Missing replay runs for api_token_hash: {missing}")
        if extra:
            exact_failures.append(f"Unexpected replay runs for api_token_hash: {extra}")

    matched_hashes = sorted(source_hashes & replay_hashes)

    agent_duration_rel_errors: list[float] = []
    request_duration_rel_errors: list[float] = []
    compared_requests = 0

    for token_hash in matched_hashes:
        source_run = source_runs[token_hash]
        replay_run = replay_runs[token_hash]

        source_req_count = len(source_run.requests)
        replay_req_count = len(replay_run.requests)
        if source_req_count != replay_req_count:
            exact_failures.append(
                f"Request count mismatch for {token_hash[:12]}: "
                f"source={source_req_count}, replay={replay_req_count}"
            )
            continue

        agent_duration_rel_errors.append(
            relative_error(source_run.run_duration_s, replay_run.run_duration_s)
        )

        for idx, (source_req, replay_req) in enumerate(
            zip(source_run.requests, replay_run.requests)
        ):
            context = request_error_context(
                token_hash=token_hash,
                idx=idx,
                source_req=source_req,
                replay_req=replay_req,
            )
            try:
                source_prompt_kind, source_prompt_content = extract_prompt_content(
                    source_req.get("request")
                )
                replay_prompt_kind, replay_prompt_content = extract_prompt_content(
                    replay_req.get("request")
                )
            except Exception as exc:  # noqa: BLE001
                exact_failures.append(
                    f"Prompt content extraction failed for {context}: {exc}"
                )
                continue
            if source_prompt_kind != replay_prompt_kind:
                exact_failures.append(
                    f"Prompt content kind mismatch for {context}: "
                    f"source={source_prompt_kind}, replay={replay_prompt_kind}"
                )
            elif source_prompt_content != replay_prompt_content:
                exact_failures.append(
                    f"Prompt content mismatch for {context}"
                )

            try:
                source_text = extract_response_text(source_req.get("response"))
                replay_text = extract_response_text(replay_req.get("response"))
            except Exception as exc:  # noqa: BLE001
                exact_failures.append(
                    f"Response content extraction failed for {context}: {exc}"
                )
                continue
            if source_text != replay_text:
                exact_failures.append(
                    f"Response content mismatch for {context}"
                )

            source_usage = extract_usage(source_req.get("response"))
            replay_usage = extract_usage(replay_req.get("response"))
            for usage_key in ["prompt_tokens", "completion_tokens", "cached_tokens"]:
                source_value = source_usage.get(usage_key)
                replay_value = replay_usage.get(usage_key)
                if source_value is not None and replay_value != source_value:
                    exact_failures.append(
                        f"Usage mismatch for {context} field={usage_key}: "
                        f"source={source_value}, replay={replay_value}"
                    )

            try:
                source_dur_ms = request_duration_ms(source_req)
                replay_dur_ms = request_duration_ms(replay_req)
                request_duration_rel_errors.append(
                    relative_error(source_dur_ms, replay_dur_ms)
                )
            except Exception as exc:  # noqa: BLE001
                exact_failures.append(
                    f"Request duration parse failed for {context}: {exc}"
                )
            compared_requests += 1

    # Job duration similarity compares envelope across all runs.
    source_start = min(run.run_start for run in source_runs.values())
    source_end = max(run.run_end for run in source_runs.values())
    replay_start = min(run.run_start for run in replay_runs.values())
    replay_end = max(run.run_end for run in replay_runs.values())
    source_job_duration_s = max(0.0, (source_end - source_start).total_seconds())
    replay_job_duration_s = max(0.0, (replay_end - replay_start).total_seconds())
    job_duration_rel_error = relative_error(source_job_duration_s, replay_job_duration_s)

    report: dict[str, Any] = {
        "status": "ok" if not exact_failures else "failed",
        "source_job_dir": str(source_job_dir),
        "replay_run_dir": str(replay_run_dir),
        "exact": {
            "passed": len(exact_failures) == 0,
            "failure_count": len(exact_failures),
            "failures": exact_failures,
            "matched_worker_count": len(matched_hashes),
            "compared_request_count": compared_requests,
        },
        "similarity": {
            "job_duration": {
                "source_s": source_job_duration_s,
                "replay_s": replay_job_duration_s,
                "relative_error": job_duration_rel_error,
            },
            "agent_duration_relative_error": summarize_errors(agent_duration_rel_errors),
            "request_duration_relative_error": summarize_errors(request_duration_rel_errors),
        },
    }

    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(
            json.dumps(report, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    exit_code = 0 if not exact_failures else 2
    return exit_code, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m replayer.validate",
        description="Validate replay output against a source profiled job.",
    )
    parser.add_argument(
        "--source-job-dir",
        required=True,
        help="Path to source con-driver profiled job directory.",
    )
    parser.add_argument(
        "--replay-run-dir",
        required=True,
        help="Path to replay run output directory.",
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help="Optional path to write JSON validation report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        source_job_dir = Path(args.source_job_dir).expanduser().resolve()
        replay_run_dir = Path(args.replay_run_dir).expanduser().resolve()
        report_out = (
            Path(args.report_out).expanduser().resolve()
            if args.report_out is not None
            else None
        )
        exit_code, report = validate(
            source_job_dir=source_job_dir,
            replay_run_dir=replay_run_dir,
            report_out=report_out,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=True, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
