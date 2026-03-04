from __future__ import annotations

import time
from typing import Any

from prometheus_client.parser import text_string_to_metric_families


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def parse_metric_content_to_json(metric_content: str) -> dict[str, Any]:
    families: dict[str, Any] = {}

    for family in text_string_to_metric_families(metric_content):
        if not family.name.startswith("vllm"):
            continue

        family_payload = {
            "type": family.type,
            "help": family.documentation,
            "samples": [],
        }
        for sample in family.samples:
            family_payload["samples"].append(
                {
                    "name": str(sample.name),
                    "labels": dict(sample.labels),
                    "value": float(sample.value),
                    "timestamp": _json_safe(sample.timestamp),
                    "exemplar": _json_safe(sample.exemplar),
                    "native_histogram": _json_safe(sample.native_histogram),
                }
            )
        families[family.name] = family_payload

    return {
        "timestamp": int(time.time()),
        "families": families,
    }
