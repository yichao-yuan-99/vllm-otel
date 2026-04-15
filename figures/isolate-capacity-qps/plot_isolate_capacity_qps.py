#!/usr/bin/env python3
"""Render the isolate-capacity-qps 1x3 bar-chart figure from materialized JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"
CASE_COLORS = ("#1d4ed8", "#d97706", "#0f766e", "#be123c", "#7c3aed")


def _import_matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import ticker
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to render this figure. "
            "Install it in your environment, for example: pip install matplotlib"
        ) from exc
    return plt, ticker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the isolate-capacity-qps figure from a materialized JSON dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_isolate_capacity_qps.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. "
            "Default: figures/isolate-capacity-qps/output/<input-stem>.pdf"
        ),
    )
    parser.add_argument(
        "--title",
        default="Capacity Comparison Across QPS",
        help="Figure title.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=13.8,
        help="Figure width in inches (default: 13.8).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=4.8,
        help="Figure height in inches (default: 4.8).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI for raster outputs (default: 220).",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


def _default_output_path(input_path: Path) -> Path:
    return (DEFAULT_OUTPUT_DIR / f"{input_path.stem}.pdf").resolve()


def _format_metric_value(metric_unit: str, value: float) -> str:
    if metric_unit == "jobs/s":
        return f"{value:.3f}"
    if metric_unit == "s":
        return f"{value:.2f}"
    return f"{value:,.0f}"


def _y_axis_formatter(metric_unit: str, ticker: Any) -> Any:
    if metric_unit == "jobs/s":
        return ticker.FuncFormatter(lambda value, _: f"{value:.2f}")
    if metric_unit == "s":
        return ticker.FuncFormatter(lambda value, _: f"{value:.1f}")
    return ticker.FuncFormatter(lambda value, _: f"{value:,.0f}")


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(input_path)
    )

    payload = _load_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {input_path}")

    raw_cases = payload.get("cases")
    raw_metrics = payload.get("metrics")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"Materialized payload has no cases: {input_path}")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ValueError(f"Materialized payload has no metrics: {input_path}")

    cases: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            continue
        case_label = raw_case.get("case_label")
        metrics = raw_case.get("metrics")
        if not isinstance(case_label, str) or not case_label:
            continue
        if not isinstance(metrics, dict):
            continue
        cases.append(raw_case)
    if not cases:
        raise ValueError(f"Materialized payload has no usable cases: {input_path}")

    metrics: list[dict[str, Any]] = []
    for raw_metric in raw_metrics:
        if not isinstance(raw_metric, dict):
            continue
        metric_key = raw_metric.get("metric_key")
        panel_title = raw_metric.get("panel_title")
        metric_unit = raw_metric.get("metric_unit")
        y_axis_label = raw_metric.get("y_axis_label")
        if not all(
            isinstance(value, str) and value
            for value in (metric_key, panel_title, metric_unit, y_axis_label)
        ):
            continue
        metrics.append(raw_metric)
    if not metrics:
        raise ValueError(f"Materialized payload has no usable metrics: {input_path}")

    plt, ticker = _import_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10.5,
            "axes.titlesize": 12.6,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(args.figure_width, args.figure_height),
        squeeze=False,
    )
    axes_list = list(axes[0])

    subtitle_parts = [f"{len(cases)} source runs"]
    raw_qps_slugs = payload.get("source_qps_slugs")
    if isinstance(raw_qps_slugs, list):
        qps_slugs = [slug for slug in raw_qps_slugs if isinstance(slug, str) and slug]
        if qps_slugs:
            subtitle_parts.append(f"qps: {', '.join(qps_slugs)}")
    subtitle_parts.append("bar height = arithmetic mean of the source figure data")

    figure.suptitle(args.title, x=0.065, y=0.985, ha="left", fontweight="semibold")
    figure.text(
        0.065,
        0.94,
        " | ".join(subtitle_parts),
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#334155",
    )

    for axis, metric in zip(axes_list, metrics):
        metric_key = str(metric["metric_key"])
        metric_unit = str(metric["metric_unit"])
        values: list[float] = []
        labels: list[str] = []
        colors: list[str] = []

        for case_index, case in enumerate(cases):
            labels.append(str(case["case_label"]))
            colors.append(CASE_COLORS[case_index % len(CASE_COLORS)])
            case_metrics = case.get("metrics")
            if not isinstance(case_metrics, dict):
                raise ValueError(f"Case is missing metrics: {case}")
            metric_summary = case_metrics.get(metric_key)
            if not isinstance(metric_summary, dict):
                raise ValueError(f"Case is missing metric {metric_key!r}: {case}")
            value = _float_or_none(metric_summary.get("value"))
            if value is None:
                raise ValueError(
                    f"Metric {metric_key!r} is missing numeric field 'value' in {case}"
                )
            values.append(value)

        x_positions = list(range(len(values)))
        bars = axis.bar(
            x_positions,
            values,
            color=colors,
            width=0.64,
            linewidth=0.0,
        )
        axis.set_title(str(metric["panel_title"]), loc="left", fontweight="semibold")
        axis.set_ylabel(str(metric["y_axis_label"]))
        axis.set_xticks(x_positions, labels, rotation=18, ha="right")
        axis.yaxis.set_major_formatter(_y_axis_formatter(metric_unit, ticker))
        axis.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.55)
        axis.set_axisbelow(True)

        max_value = max(values) if values else 0.0
        axis.set_ylim(0.0, max_value * 1.18 if max_value > 0.0 else 1.0)
        label_offset = max_value * 0.025 if max_value > 0.0 else 0.05
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + (bar.get_width() / 2.0),
                bar.get_height() + label_offset,
                _format_metric_value(metric_unit, value),
                ha="center",
                va="bottom",
                fontsize=9.1,
                color="#0f172a",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
