from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_OUTPUT_NAME = "service-failure.json"
DETECTOR_VERSION = 3
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_RUN_STAMP_RE = re.compile(r"(\d{4})(\d{2})(\d{2})T\d{6}Z")
_ISO_TS_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))"
)
_FULL_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{1,6})?)")
_SHORT_TS_RE = re.compile(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


_LOCAL_TIMEZONE = datetime.now().astimezone().tzinfo or timezone.utc


@dataclass(frozen=True)
class _FailureRule:
    code: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class _FailureEvent:
    code: str
    log_path: Path
    line_number: int
    line_text: str
    timestamp_utc: datetime | None


_FAILURE_RULES: tuple[_FailureRule, ...] = (
    _FailureRule(
        code="async_llm_output_handler_failed",
        pattern=re.compile(r"AsyncLLM output_handler failed", re.IGNORECASE),
    ),
    _FailureRule(
        code="python_traceback",
        pattern=re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE),
    ),
)


_WARNING_RULES: tuple[_FailureRule, ...] = (
    _FailureRule(
        code="counter_negative_increment",
        pattern=re.compile(
            r"Counters can only be incremented by non-negative amounts",
            re.IGNORECASE,
        ),
    ),
)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_LOCAL_TIMEZONE).astimezone(timezone.utc)
    return dt.astimezone(timezone.utc)


def _isoformat_utc(dt: datetime) -> str:
    return _to_utc(dt).isoformat().replace("+00:00", "Z")


def _cleanup_log_line(line: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", line.rstrip("\n"))


def _infer_run_year(run_dir: Path) -> int:
    for part in run_dir.parts:
        match = _RUN_STAMP_RE.search(part)
        if match is None:
            continue
        return int(match.group(1))
    return datetime.now(tz=timezone.utc).year


def parse_iso8601_to_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _to_utc(parsed)


def _parse_timestamp_from_line(
    line: str,
    *,
    run_year: int,
) -> datetime | None:
    iso_match = _ISO_TS_RE.search(line)
    if iso_match is not None:
        return parse_iso8601_to_utc(iso_match.group(1))

    full_match = _FULL_TS_RE.search(line)
    if full_match is not None:
        text = full_match.group(1)
        dt_formats = ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S")
        for fmt in dt_formats:
            try:
                parsed = datetime.strptime(text, fmt)
            except ValueError:
                continue
            return _to_utc(parsed)

    short_match = _SHORT_TS_RE.search(line)
    if short_match is not None:
        text = short_match.group(1)
        try:
            parsed = datetime.strptime(f"{run_year}-{text}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
        return _to_utc(parsed)
    return None


def default_output_path_for_run(run_dir: Path) -> Path:
    return (run_dir / "post-processed" / "service-failure" / DEFAULT_OUTPUT_NAME).resolve()


def cutoff_datetime_utc_from_payload(payload: dict[str, Any] | None) -> datetime | None:
    if not isinstance(payload, dict):
        return None
    return parse_iso8601_to_utc(payload.get("cutoff_time_utc"))


def _select_primary_event(events: list[_FailureEvent]) -> _FailureEvent:
    with_timestamp = [event for event in events if event.timestamp_utc is not None]
    if with_timestamp:
        return min(
            with_timestamp,
            key=lambda event: (
                event.timestamp_utc,
                str(event.log_path),
                event.line_number,
            ),
        )
    return events[0]


def _discover_log_files(sbatch_logs_dir: Path) -> list[Path]:
    candidates = [
        path
        for path in sbatch_logs_dir.iterdir()
        if path.is_file() and path.suffix in {".log", ".out", ".err"}
    ]
    return sorted(candidates)


def _event_sort_key(event: _FailureEvent) -> tuple[datetime, str, int, str]:
    return (
        event.timestamp_utc
        if event.timestamp_utc is not None
        else datetime.max.replace(tzinfo=timezone.utc),
        str(event.log_path),
        event.line_number,
        event.code,
    )


def _preview_events(events: list[_FailureEvent]) -> list[dict[str, Any]]:
    preview = []
    for event in sorted(events, key=_event_sort_key)[:20]:
        preview.append(
            {
                "rule": event.code,
                "log_path": str(event.log_path.resolve()),
                "line_number": event.line_number,
                "timestamp_utc": (
                    _isoformat_utc(event.timestamp_utc)
                    if event.timestamp_utc is not None
                    else None
                ),
                "line": event.line_text,
            }
        )
    return preview


def detect_service_failure(run_dir: Path) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    sbatch_logs_dir = resolved_run_dir / "sbatch-logs"
    now_utc = datetime.now(tz=timezone.utc)

    if not sbatch_logs_dir.is_dir():
        return {
            "detector_version": DETECTOR_VERSION,
            "source_run_dir": str(resolved_run_dir),
            "source_sbatch_logs_dir": str(sbatch_logs_dir),
            "sbatch_logs_exists": False,
            "service_failure_detected": False,
            "cutoff_time_utc": None,
            "cutoff_epoch_s": None,
            "matched_rule": None,
            "matched_log_path": None,
            "matched_line_number": None,
            "matched_line": None,
            "event_count": 0,
            "events_preview": [],
            "warning_count": 0,
            "warnings_preview": [],
            "detected_at_utc": _isoformat_utc(now_utc),
        }

    run_year = _infer_run_year(resolved_run_dir)
    failure_events: list[_FailureEvent] = []
    warning_events: list[_FailureEvent] = []
    for log_path in _discover_log_files(sbatch_logs_dir):
        last_timestamp: datetime | None = None
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                cleaned = _cleanup_log_line(raw_line)
                parsed_timestamp = _parse_timestamp_from_line(
                    cleaned,
                    run_year=run_year,
                )
                if parsed_timestamp is not None:
                    last_timestamp = parsed_timestamp
                for rule in _FAILURE_RULES:
                    if rule.pattern.search(cleaned) is None:
                        continue
                    failure_events.append(
                        _FailureEvent(
                            code=rule.code,
                            log_path=log_path,
                            line_number=line_number,
                            line_text=cleaned,
                            timestamp_utc=parsed_timestamp or last_timestamp,
                        )
                    )
                for rule in _WARNING_RULES:
                    if rule.pattern.search(cleaned) is None:
                        continue
                    warning_events.append(
                        _FailureEvent(
                            code=rule.code,
                            log_path=log_path,
                            line_number=line_number,
                            line_text=cleaned,
                            timestamp_utc=parsed_timestamp or last_timestamp,
                        )
                    )

    warning_preview = _preview_events(warning_events)

    if not failure_events:
        return {
            "detector_version": DETECTOR_VERSION,
            "source_run_dir": str(resolved_run_dir),
            "source_sbatch_logs_dir": str(sbatch_logs_dir.resolve()),
            "sbatch_logs_exists": True,
            "service_failure_detected": False,
            "cutoff_time_utc": None,
            "cutoff_epoch_s": None,
            "matched_rule": None,
            "matched_log_path": None,
            "matched_line_number": None,
            "matched_line": None,
            "event_count": 0,
            "events_preview": [],
            "warning_count": len(warning_events),
            "warnings_preview": warning_preview,
            "detected_at_utc": _isoformat_utc(now_utc),
        }

    primary_event = _select_primary_event(failure_events)
    cutoff_time_utc = (
        _isoformat_utc(primary_event.timestamp_utc)
        if primary_event.timestamp_utc is not None
        else None
    )
    cutoff_epoch_s = (
        round(primary_event.timestamp_utc.timestamp(), 6)
        if primary_event.timestamp_utc is not None
        else None
    )

    return {
        "detector_version": DETECTOR_VERSION,
        "source_run_dir": str(resolved_run_dir),
        "source_sbatch_logs_dir": str(sbatch_logs_dir.resolve()),
        "sbatch_logs_exists": True,
        "service_failure_detected": True,
        "cutoff_time_utc": cutoff_time_utc,
        "cutoff_epoch_s": cutoff_epoch_s,
        "matched_rule": primary_event.code,
        "matched_log_path": str(primary_event.log_path.resolve()),
        "matched_line_number": primary_event.line_number,
        "matched_line": primary_event.line_text,
        "event_count": len(failure_events),
        "events_preview": _preview_events(failure_events),
        "warning_count": len(warning_events),
        "warnings_preview": warning_preview,
        "detected_at_utc": _isoformat_utc(now_utc),
    }


def ensure_service_failure_payload(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_output_path = (output_path or default_output_path_for_run(resolved_run_dir)).expanduser().resolve()

    if not force_refresh and resolved_output_path.is_file():
        try:
            payload = json.loads(resolved_output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if (
            isinstance(payload, dict)
            and payload.get("detector_version") == DETECTOR_VERSION
        ):
            return payload

    payload = detect_service_failure(resolved_run_dir)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
