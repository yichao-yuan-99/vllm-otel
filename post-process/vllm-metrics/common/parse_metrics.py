from __future__ import annotations

import re
import time
from typing import Any

try:
    from prometheus_client.parser import text_string_to_metric_families
except ModuleNotFoundError:  # pragma: no cover
    text_string_to_metric_families = None


_HELP_RE = re.compile(r"^#\s*HELP\s+(\S+)\s+(.*)$")
_TYPE_RE = re.compile(r"^#\s*TYPE\s+(\S+)\s+(\S+)\s*$")
_SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]?Inf)"
    r"(?:\s+(-?\d+(?:\.\d+)?))?$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _unescape_label_value(raw: str) -> str:
    return (
        raw.replace(r"\\", "\\")
        .replace(r"\"", '"')
        .replace(r"\n", "\n")
        .replace(r"\t", "\t")
    )


def _parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    labels: dict[str, str] = {}
    for key, value in _LABEL_RE.findall(raw):
        labels[key] = _unescape_label_value(value)
    return labels


def _normalize_family_name(metric_name: str, metric_type: str | None) -> str:
    if metric_type == "counter" and metric_name.endswith("_total"):
        return metric_name[: -len("_total")]
    return metric_name


def _resolve_family_name(
    sample_name: str,
    *,
    metric_to_family: dict[str, str],
) -> str:
    direct = metric_to_family.get(sample_name)
    if direct is not None:
        return direct
    for suffix in ("_bucket", "_sum", "_count"):
        if sample_name.endswith(suffix):
            base = sample_name[: -len(suffix)]
            resolved = metric_to_family.get(base)
            if resolved is not None:
                return resolved
    return sample_name


def _help_for_family(
    family_name: str,
    *,
    sample_name: str,
    family_to_type: dict[str, str],
    help_by_metric: dict[str, str],
) -> str:
    if family_name in help_by_metric:
        return help_by_metric[family_name]
    if sample_name in help_by_metric:
        return help_by_metric[sample_name]
    family_type = family_to_type.get(family_name)
    if family_type == "counter":
        counter_metric_name = f"{family_name}_total"
        if counter_metric_name in help_by_metric:
            return help_by_metric[counter_metric_name]
    return ""


def _fallback_parse_metric_content_to_json(metric_content: str) -> dict[str, Any]:
    help_by_metric: dict[str, str] = {}
    metric_to_family: dict[str, str] = {}
    family_to_type: dict[str, str] = {}
    families: dict[str, Any] = {}

    for raw_line in metric_content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        help_match = _HELP_RE.match(line)
        if help_match is not None:
            metric_name, help_text = help_match.groups()
            help_by_metric[metric_name] = help_text
            continue

        type_match = _TYPE_RE.match(line)
        if type_match is not None:
            metric_name, metric_type = type_match.groups()
            family_name = _normalize_family_name(metric_name, metric_type)
            metric_to_family[metric_name] = family_name
            family_to_type[family_name] = metric_type
            continue

        if line.startswith("#"):
            continue

        sample_match = _SAMPLE_RE.match(line)
        if sample_match is None:
            continue
        sample_name, raw_labels, raw_value, raw_timestamp = sample_match.groups()
        family_name = _resolve_family_name(sample_name, metric_to_family=metric_to_family)
        if not family_name.startswith("vllm"):
            continue

        family_type = family_to_type.get(family_name, "untyped")
        family_payload = families.setdefault(
            family_name,
            {
                "type": family_type,
                "help": _help_for_family(
                    family_name,
                    sample_name=sample_name,
                    family_to_type=family_to_type,
                    help_by_metric=help_by_metric,
                ),
                "samples": [],
            },
        )
        family_payload["samples"].append(
            {
                "name": sample_name,
                "labels": _parse_labels(raw_labels),
                "value": float(raw_value),
                "timestamp": (
                    int(raw_timestamp)
                    if raw_timestamp is not None and raw_timestamp.isdigit()
                    else _json_safe(float(raw_timestamp)) if raw_timestamp is not None else None
                ),
                "exemplar": None,
                "native_histogram": None,
            }
        )

    return {
        "timestamp": int(time.time()),
        "families": families,
    }


def parse_metric_content_to_json(metric_content: str) -> dict[str, Any]:
    if text_string_to_metric_families is None:
        return _fallback_parse_metric_content_to_json(metric_content)

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
                    "native_histogram": _json_safe(getattr(sample, "native_histogram", None)),
                }
            )
        families[family.name] = family_payload

    return {
        "timestamp": int(time.time()),
        "families": families,
    }
