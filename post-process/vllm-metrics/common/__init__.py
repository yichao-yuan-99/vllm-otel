from __future__ import annotations

from typing import Any


def parse_metric_content_to_json(metric_content: str) -> dict[str, Any]:
    from .parse_metrics import parse_metric_content_to_json as _impl

    return _impl(metric_content)


def count_metric_blocks_in_run_dir(run_dir):
    from .extract_timeseries import count_metric_blocks_in_run_dir as _impl

    return _impl(run_dir)


def load_metric_records_from_run_dir(run_dir, **kwargs):
    from .extract_timeseries import load_metric_records_from_run_dir as _impl

    return _impl(run_dir, **kwargs)


def build_gauge_counter_timeseries(records):
    from .extract_timeseries import build_gauge_counter_timeseries as _impl

    return _impl(records)


def extract_gauge_counter_timeseries_from_run_dir(run_dir, **kwargs):
    from .extract_timeseries import (
        extract_gauge_counter_timeseries_from_run_dir as _impl,
    )

    return _impl(run_dir, **kwargs)


__all__ = [
    "build_gauge_counter_timeseries",
    "count_metric_blocks_in_run_dir",
    "extract_gauge_counter_timeseries_from_run_dir",
    "load_metric_records_from_run_dir",
    "parse_metric_content_to_json",
]
