#!/usr/bin/env python3
"""Plot selected KV-cache-usage time series on one figure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"
SERIES_COLORS = ("#1d4ed8", "#d97706", "#0f766e", "#be123c", "#7c3aed")


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
        description="Plot the kv-usage-time figure from a materialized JSON dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON from materialize_kv_usage_time.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: figures/kv-usage-time/output/<input-stem>.pdf",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title. Empty by default.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=10.8,
        help="Figure width in inches (default: 10.8).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=4.133333333333334,
        help="Figure height in inches (default: 4.133333333333334).",
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


def _series_display_label(raw_item: dict[str, Any], index: int) -> str:
    frequency_mhz = _float_or_none(raw_item.get("frequency_mhz"))
    if frequency_mhz is not None:
        if frequency_mhz.is_integer():
            frequency_label = str(int(frequency_mhz))
        else:
            frequency_label = f"{frequency_mhz:g}"
        return f"max freq. {frequency_label}"

    label = raw_item.get("series_label")
    if not isinstance(label, str) or not label:
        run_slug = raw_item.get("run_slug")
        if isinstance(run_slug, str) and run_slug:
            return run_slug
        return f"series-{index+1}"
    return label


def _series_matches_frequency(raw_item: dict[str, Any], frequency_mhz: int) -> bool:
    parsed_frequency = _float_or_none(raw_item.get("frequency_mhz"))
    if parsed_frequency is not None:
        return math.isclose(parsed_frequency, float(frequency_mhz))

    run_slug = raw_item.get("run_slug")
    if isinstance(run_slug, str) and run_slug.endswith(f"-{frequency_mhz}"):
        return True

    return False


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
    raw_series = payload.get("series")
    if not isinstance(raw_series, list) or not raw_series:
        raise ValueError(f"Materialized payload has no series: {input_path}")

    plt, ticker = _import_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 15,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    figure, axis = plt.subplots(figsize=(args.figure_width, args.figure_height))

    plotted_series_count = 0
    freq_660_color: str | None = None
    for index, raw_item in enumerate(raw_series):
        if not isinstance(raw_item, dict):
            continue
        raw_points = raw_item.get("points")
        if not isinstance(raw_points, list):
            continue

        x_values_minutes: list[float] = []
        y_values_percent: list[float] = []
        for point in raw_points:
            if not isinstance(point, dict):
                continue
            time_s = _float_or_none(point.get("time_from_start_s"))
            value = _float_or_none(point.get("value"))
            if time_s is None or value is None:
                continue
            x_values_minutes.append(time_s / 60.0)
            y_values_percent.append(value * 100.0)
        if not x_values_minutes:
            continue

        label = _series_display_label(raw_item, index)

        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        axis.plot(
            x_values_minutes,
            y_values_percent,
            color=color,
            linewidth=2.2,
            label=label,
        )
        if _series_matches_frequency(raw_item, 660):
            freq_660_color = color
        plotted_series_count += 1

    if plotted_series_count == 0:
        raise ValueError(f"No plottable series were found in {input_path}")

    if isinstance(args.title, str) and args.title.strip():
        axis.set_title(args.title, loc="left", fontweight="semibold")

    axis.set_xlabel("Time From Start (minutes)")
    axis.set_ylabel("Context Cache Usage (%)")
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
    axis.minorticks_on()
    axis.margins(x=0.0)
    axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    left_spine = axis.spines["left"]
    for side in ("top", "right"):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_linewidth(left_spine.get_linewidth())
        axis.spines[side].set_edgecolor(left_spine.get_edgecolor())
    if freq_660_color is not None:
        axis.text(
            120.0,
            90.0,
            "thrashing regime",
            color=freq_660_color,
            fontsize=15,
            ha="left",
            va="center",
        )
    axis.legend(loc="upper left", frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
