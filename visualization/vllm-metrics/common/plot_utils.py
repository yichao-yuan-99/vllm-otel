from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PAGE_WIDTH = 792.0
PAGE_HEIGHT = 612.0
PLOT_LEFT = 72.0
PLOT_BOTTOM = 90.0
PLOT_WIDTH = 640.0
PLOT_HEIGHT = 420.0
SERIES_COLORS = [
    (0.11, 0.37, 0.69),
    (0.78, 0.27, 0.19),
    (0.15, 0.55, 0.31),
    (0.55, 0.38, 0.72),
]


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_text_command(
    x: float,
    y: float,
    text: str,
    *,
    size: float = 12.0,
) -> str:
    return f"BT /F1 {size:.2f} Tf {x:.2f} {y:.2f} Td ({_pdf_escape(text)}) Tj ET"


def _build_rotated_text_command(
    x: float,
    y: float,
    text: str,
    *,
    size: float = 12.0,
) -> str:
    return (
        f"BT /F1 {size:.2f} Tf 0 1 -1 0 {x:.2f} {y:.2f} Tm "
        f"({_pdf_escape(text)}) Tj ET"
    )


def _format_tick(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def load_timeseries_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid timeseries payload: {path}")
    return payload


def build_series_label(metric: dict[str, Any]) -> str:
    labels = metric.get("labels")
    if not isinstance(labels, dict) or not labels:
        return str(metric.get("name") or "series")
    parts = [f"{key}={labels[key]}" for key in sorted(labels)]
    return ", ".join(parts)


def select_metric_series(payload: dict[str, Any], metric_name: str) -> list[dict[str, Any]]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Timeseries payload is missing metrics")

    series: list[dict[str, Any]] = []
    for metric in metrics.values():
        if not isinstance(metric, dict):
            continue
        if metric.get("name") != metric_name:
            continue
        xs = metric.get("time_from_start_s")
        ys = metric.get("value")
        if not isinstance(xs, list) or not isinstance(ys, list):
            continue
        if len(xs) != len(ys):
            raise ValueError(f"{metric_name} series has mismatched x/y lengths")
        if not xs:
            continue
        series.append(
            {
                "label": build_series_label(metric),
                "x": [float(value) for value in xs],
                "y": [float(value) for value in ys],
                "labels": dict(metric.get("labels") or {}),
            }
        )
    if not series:
        raise ValueError(f"No {metric_name} series found in payload")
    return series


def build_grouped_average_series(
    series: list[dict[str, Any]],
    *,
    group_size: int,
) -> list[dict[str, Any]]:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    grouped: list[dict[str, Any]] = []
    for item in series:
        xs = item["x"]
        ys = item["y"]
        grouped_xs: list[float] = []
        grouped_ys: list[float] = []
        for start in range(0, len(xs), group_size):
            x_group = xs[start : start + group_size]
            y_group = ys[start : start + group_size]
            if not x_group:
                continue
            grouped_xs.append(sum(x_group) / len(x_group))
            grouped_ys.append(sum(y_group) / len(y_group))
        grouped.append(
            {
                "label": item["label"],
                "x": grouped_xs,
                "y": grouped_ys,
                "labels": dict(item.get("labels") or {}),
            }
        )
    return grouped


def write_curated_series_json(
    *,
    output_path: Path,
    title: str,
    y_axis_label: str,
    series: list[dict[str, Any]],
    x_axis_label: str = "time_from_start_s",
) -> Path:
    payload = {
        "title": title,
        "x_axis_label": x_axis_label,
        "y_axis_label": y_axis_label,
        "series_count": len(series),
        "series": series,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def render_line_chart_pdf(
    *,
    series: list[dict[str, Any]],
    output_path: Path,
    title: str,
    y_axis_label: str,
    x_axis_label: str = "time_from_start_s",
) -> None:
    max_x = max(max(item["x"]) for item in series if item["x"])
    max_y = max(max(item["y"]) for item in series if item["y"])
    min_y = min(min(item["y"]) for item in series if item["y"])
    x_max = max(max_x, 1.0)
    y_min = min(0.0, min_y)
    y_max = max(max_y, 1.0)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def map_x(value: float) -> float:
        return PLOT_LEFT + (value / x_max) * PLOT_WIDTH

    def map_y(value: float) -> float:
        return PLOT_BOTTOM + ((value - y_min) / (y_max - y_min)) * PLOT_HEIGHT

    commands: list[str] = []
    commands.append("1 w 0 0 0 RG")
    commands.append(
        f"{PLOT_LEFT:.2f} {PLOT_BOTTOM:.2f} {PLOT_WIDTH:.2f} {PLOT_HEIGHT:.2f} re S"
    )

    x_ticks = 6
    for index in range(x_ticks + 1):
        value = (x_max / x_ticks) * index
        x = map_x(value)
        commands.append(
            f"0.85 0.85 0.85 RG {x:.2f} {PLOT_BOTTOM:.2f} m "
            f"{x:.2f} {PLOT_BOTTOM + PLOT_HEIGHT:.2f} l S"
        )
        commands.append(
            f"0 0 0 RG {x:.2f} {PLOT_BOTTOM:.2f} m {x:.2f} {PLOT_BOTTOM - 4:.2f} l S"
        )
        commands.append(
            _build_text_command(x - 10.0, PLOT_BOTTOM - 20.0, _format_tick(value), size=10.0)
        )

    y_ticks = 5
    for index in range(y_ticks + 1):
        value = y_min + ((y_max - y_min) / y_ticks) * index
        y = map_y(value)
        commands.append(
            f"0.90 0.90 0.90 RG {PLOT_LEFT:.2f} {y:.2f} m "
            f"{PLOT_LEFT + PLOT_WIDTH:.2f} {y:.2f} l S"
        )
        commands.append(
            f"0 0 0 RG {PLOT_LEFT:.2f} {y:.2f} m {PLOT_LEFT - 4:.2f} {y:.2f} l S"
        )
        commands.append(
            _build_text_command(PLOT_LEFT - 45.0, y - 3.0, _format_tick(value), size=10.0)
        )

    for index, item in enumerate(series):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        points = list(zip(item["x"], item["y"]))
        if not points:
            continue
        commands.append(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG 1.5 w")
        start_x, start_y = points[0]
        commands.append(f"{map_x(start_x):.2f} {map_y(start_y):.2f} m")
        for raw_x, raw_y in points[1:]:
            commands.append(f"{map_x(raw_x):.2f} {map_y(raw_y):.2f} l")
        commands.append("S")

    commands.append(_build_text_command(PLOT_LEFT, PAGE_HEIGHT - 42.0, title, size=16.0))
    commands.append(
        _build_text_command(
            PLOT_LEFT + (PLOT_WIDTH / 2.0) - 45.0,
            PLOT_BOTTOM - 45.0,
            x_axis_label,
            size=12.0,
        )
    )
    commands.append(
        _build_rotated_text_command(
            PLOT_LEFT - 58.0,
            PLOT_BOTTOM + (PLOT_HEIGHT / 2.0) - 40.0,
            y_axis_label,
            size=12.0,
        )
    )

    legend_x = PLOT_LEFT + PLOT_WIDTH - 190.0
    legend_y = PAGE_HEIGHT - 60.0
    for index, item in enumerate(series):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        y = legend_y - (index * 16.0)
        commands.append(
            f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG 2 w "
            f"{legend_x:.2f} {y:.2f} m {legend_x + 16.0:.2f} {y:.2f} l S"
        )
        commands.append(
            _build_text_command(legend_x + 22.0, y - 4.0, item["label"], size=9.0)
        )

    content = "\n".join(commands).encode("utf-8")

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH:.2f} {PAGE_HEIGHT:.2f}] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("utf-8")
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream"
    )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf)
