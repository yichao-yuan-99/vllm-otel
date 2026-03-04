from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


_SUPPORTED_METRIC_TYPES = {"gauge", "counter"}


@dataclass(frozen=True)
class ParsedMetricRecord:
    captured_at: str
    families: dict[str, Any]


def _parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _build_series_key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    parts = [name]
    for key in sorted(labels):
        parts.append(f"{key}={labels[key]}")
    return "|".join(parts)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_block_entries(vllm_log_dir: Path) -> list[dict[str, Any]]:
    index_path = vllm_log_dir / "blocks.index.json"
    if index_path.is_file():
        payload = _load_json(index_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid block index payload: {index_path}")
        blocks = payload.get("blocks")
        if not isinstance(blocks, list):
            raise ValueError(f"Invalid block index blocks list: {index_path}")
        return [item for item in blocks if isinstance(item, dict)]
    return [
        {"file": path.name, "member": None}
        for path in sorted(vllm_log_dir.glob("block-*.tar.gz"))
    ]


def _read_jsonl_member(tar_path: Path, member_name: str | None = None) -> list[dict[str, Any]]:
    with tarfile.open(tar_path, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            return []
        selected_member = None
        if member_name is not None:
            for member in members:
                if member.name == member_name:
                    selected_member = member
                    break
        if selected_member is None:
            selected_member = members[0]
        extracted = archive.extractfile(selected_member)
        if extracted is None:
            return []
        lines = extracted.read().decode("utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object in {tar_path}, got {type(parsed)!r}")
        records.append(parsed)
    return records


def _parse_metric_record(record: dict[str, Any]) -> ParsedMetricRecord:
    captured_at = record.get("captured_at")
    if not isinstance(captured_at, str) or not captured_at.strip():
        raise ValueError("Metric record is missing captured_at")

    families_payload = record.get("families")
    if isinstance(families_payload, dict):
        return ParsedMetricRecord(captured_at=captured_at, families=families_payload)

    content = record.get("content")
    if isinstance(content, str):
        from .parse_metrics import parse_metric_content_to_json

        parsed = parse_metric_content_to_json(content)
        families = parsed.get("families")
        if not isinstance(families, dict):
            raise ValueError("Parsed metric payload is missing families")
        return ParsedMetricRecord(captured_at=captured_at, families=families)

    raise ValueError("Metric record must contain either families or content")


def count_metric_blocks_in_run_dir(run_dir: Path) -> int:
    vllm_log_dir = run_dir / "vllm-log"
    if not vllm_log_dir.is_dir():
        raise ValueError(f"Missing vllm-log directory: {vllm_log_dir}")
    return len(_load_block_entries(vllm_log_dir))


def load_metric_records_from_run_dir(
    run_dir: Path,
    *,
    on_block_loaded: Callable[[int, int], None] | None = None,
) -> list[ParsedMetricRecord]:
    vllm_log_dir = run_dir / "vllm-log"
    if not vllm_log_dir.is_dir():
        raise ValueError(f"Missing vllm-log directory: {vllm_log_dir}")
    block_entries = _load_block_entries(vllm_log_dir)
    records: list[ParsedMetricRecord] = []
    total_blocks = len(block_entries)
    for index, block in enumerate(block_entries, start=1):
        file_name = block.get("file")
        if not isinstance(file_name, str) or not file_name:
            continue
        tar_path = vllm_log_dir / file_name
        if not tar_path.is_file():
            raise ValueError(f"Missing vllm log block: {tar_path}")
        member_name = block.get("member")
        member_name_value = member_name if isinstance(member_name, str) else None
        for record in _read_jsonl_member(tar_path, member_name=member_name_value):
            records.append(_parse_metric_record(record))
        if on_block_loaded is not None:
            on_block_loaded(index, total_blocks)
    return records


def build_gauge_counter_timeseries(records: list[ParsedMetricRecord]) -> dict[str, Any]:
    if not records:
        return {"first_captured_at": None, "metrics": {}}

    first_dt = _parse_iso8601(records[0].captured_at)
    metrics: dict[str, dict[str, Any]] = {}
    previous_counter_value: dict[str, float] = {}

    for record in records:
        current_dt = _parse_iso8601(record.captured_at)
        relative_time_s = round((current_dt - first_dt).total_seconds(), 6)

        for family_name, family_payload in record.families.items():
            if not isinstance(family_payload, dict):
                continue
            family_type = family_payload.get("type")
            if family_type not in _SUPPORTED_METRIC_TYPES:
                continue
            samples = family_payload.get("samples")
            if not isinstance(samples, list):
                continue
            help_text = family_payload.get("help")
            help_value = help_text if isinstance(help_text, str) else ""

            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                sample_name = sample.get("name")
                labels = sample.get("labels")
                value = sample.get("value")
                if not isinstance(sample_name, str) or not sample_name.startswith("vllm"):
                    continue
                if not isinstance(labels, dict):
                    continue
                if not isinstance(value, (int, float)):
                    continue

                normalized_labels = {str(key): str(val) for key, val in labels.items()}
                series_key = _build_series_key(family_name, normalized_labels)
                metric_entry = metrics.setdefault(
                    series_key,
                    {
                        "name": family_name,
                        "sample_name": sample_name,
                        "family": family_name,
                        "type": family_type,
                        "help": help_value,
                        "labels": normalized_labels,
                        "captured_at": [],
                        "value": [],
                        "time_from_start_s": [],
                    },
                )
                numeric_value = float(value)
                if family_type == "counter":
                    previous_value = previous_counter_value.get(series_key)
                    output_value = 0.0 if previous_value is None else numeric_value - previous_value
                    previous_counter_value[series_key] = numeric_value
                else:
                    output_value = numeric_value

                metric_entry["captured_at"].append(record.captured_at)
                metric_entry["value"].append(output_value)
                metric_entry["time_from_start_s"].append(relative_time_s)

    return {
        "first_captured_at": records[0].captured_at,
        "metrics": metrics,
    }


def extract_gauge_counter_timeseries_from_run_dir(
    run_dir: Path,
    *,
    on_block_loaded: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    records = load_metric_records_from_run_dir(run_dir, on_block_loaded=on_block_loaded)
    result = build_gauge_counter_timeseries(records)
    return {
        "source_run_dir": str(run_dir.resolve()),
        "source_vllm_log_dir": str((run_dir / "vllm-log").resolve()),
        "first_captured_at": result["first_captured_at"],
        "metric_count": len(result["metrics"]),
        "metrics": result["metrics"],
    }
