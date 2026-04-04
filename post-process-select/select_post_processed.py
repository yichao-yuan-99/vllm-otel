#!/usr/bin/env python3
"""Select the last X percent of one run's post-processed outputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUTPUT_PREFIX = "post-processed-"
DEFAULT_SELECTION_SUMMARY_NAME = "selection-summary.json"
DEFAULT_REQUIRED_DISCOVERY_RELATIVE_PATH = "global/trial-timing-summary.json"
DEFAULT_PREFILL_TICK_MS = 10
DEFAULT_MILESTONE_STEP = 50
DEFAULT_TIMEPOINT_FREQUENCY_HZ = 1.0
DEFAULT_WINDOW_SIZE_S = 600.0
DEFAULT_VISUALIZATION_FORMAT = "png"
DEFAULT_VISUALIZATION_DPI = 220

_HELPER_MODULES: dict[str, Any] = {}


@dataclass(frozen=True)
class SelectionWindow:
    percent: float
    percent_label: str
    source_run_dir: Path
    source_post_processed_dir: Path
    output_dir: Path
    original_started_at: str
    original_finished_at: str
    selected_started_at: str
    selected_finished_at: str
    source_start_utc: datetime
    source_end_utc: datetime
    selected_start_utc: datetime
    selected_end_utc: datetime
    original_duration_s: float
    cutoff_offset_s: float
    selected_duration_s: float


def _load_helper_module(cache_key: str, relative_path: str) -> Any:
    cached = _HELPER_MODULES.get(cache_key)
    if cached is not None:
        return cached

    module_path = (REPO_ROOT / relative_path).resolve()
    spec = importlib.util.spec_from_file_location(
        f"post_process_select_{cache_key}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _HELPER_MODULES[cache_key] = module
    return module


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
            "Select the last X percent of post-processed JSON outputs for one run or "
            "for every matching run under a root directory."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--run-dir",
        default=None,
        help="Run root directory containing post-processed/.",
    )
    source_group.add_argument(
        "--post-processed-dir",
        default=None,
        help="Explicit source post-processed directory.",
    )
    source_group.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory to recursively scan for run directories. Any directory "
            "with post-processed/global/trial-timing-summary.json will be processed."
        ),
    )
    parser.add_argument(
        "-x",
        "--percent",
        required=True,
        type=float,
        help="Keep the last X percent of the post-processed timeline.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional derived output directory. Default: "
            "<run-dir>/post-processed-<percent>."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory without deleting it first.",
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


def _parse_iso8601_utc(value: Any) -> datetime | None:
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
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _isoformat_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _round_s(value: float) -> float:
    return round(value, 6)


def _round_ms_from_seconds(value_s: float) -> float:
    return round(value_s * 1000.0, 3)


def _round_value(value: float) -> float:
    return round(value, 12)


def _normalize_percent(percent: float) -> float:
    if not math.isfinite(percent):
        raise ValueError(f"--percent must be finite: {percent!r}")
    if percent <= 0 or percent > 100:
        raise ValueError(f"--percent must be in the interval (0, 100]: {percent}")
    return float(percent)


def _percent_label(percent: float) -> str:
    text = f"{percent:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "_")


def _default_source_post_processed_dir(run_dir: Path) -> Path:
    return (run_dir / "post-processed").resolve()


def _default_output_dir(source_post_processed_dir: Path, *, percent_label: str) -> Path:
    run_dir = source_post_processed_dir.parent
    return (run_dir / f"{DEFAULT_OUTPUT_PREFIX}{percent_label}").resolve()


def discover_run_dirs_with_post_processed(root_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for timing_summary_path in root_dir.rglob("trial-timing-summary.json"):
        if not timing_summary_path.is_file():
            continue
        if timing_summary_path.parent.name != "global":
            continue
        if timing_summary_path.parent.parent.name != "post-processed":
            continue
        run_dirs.add(timing_summary_path.parent.parent.parent.resolve())
    return sorted(run_dirs)


def _relative_json_paths(source_dir: Path) -> set[str]:
    return {
        str(path.relative_to(source_dir))
        for path in source_dir.rglob("*.json")
        if path.is_file()
    }


def _relative_non_json_paths(source_dir: Path) -> list[str]:
    return sorted(
        str(path.relative_to(source_dir))
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() != ".json"
    )


def _stats_from_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg": None, "min": None, "max": None}
    return {
        "avg": _round_value(sum(values) / len(values)),
        "min": _round_value(min(values)),
        "max": _round_value(max(values)),
    }


def _duration_seconds(start_dt: datetime | None, end_dt: datetime | None) -> float | None:
    if start_dt is None or end_dt is None:
        return None
    return _round_s((end_dt - start_dt).total_seconds())


def _clip_range_to_window(
    *,
    start_s: float,
    end_s: float,
    window: SelectionWindow,
) -> tuple[float, float, float, float] | None:
    clipped_start_abs = max(start_s, window.cutoff_offset_s, 0.0)
    clipped_end_abs = min(end_s, window.original_duration_s)
    if clipped_end_abs <= clipped_start_abs:
        return None
    return (
        clipped_start_abs,
        clipped_end_abs,
        _round_s(clipped_start_abs - window.cutoff_offset_s),
        _round_s(clipped_end_abs - window.cutoff_offset_s),
    )


def _rebase_point_offset(
    offset_s: float,
    *,
    window: SelectionWindow,
) -> float | None:
    if offset_s < window.cutoff_offset_s or offset_s > window.original_duration_s:
        return None
    return _round_s(offset_s - window.cutoff_offset_s)


def _interpolate_power(points: list[tuple[float, float]], sample_time_s: float) -> float | None:
    if not points:
        return None
    if sample_time_s <= points[0][0]:
        return points[0][1]
    if sample_time_s >= points[-1][0]:
        return points[-1][1]
    for index in range(len(points) - 1):
        left_time, left_value = points[index]
        right_time, right_value = points[index + 1]
        if left_time <= sample_time_s <= right_time:
            if right_time <= left_time:
                return right_value
            ratio = (sample_time_s - left_time) / (right_time - left_time)
            return left_value + ((right_value - left_value) * ratio)
    return None


def _is_cancelled_status(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"cancelled", "canceled"}


def _extract_figure_paths_from_manifest_payload(payload: Any) -> list[Path]:
    figure_paths: list[Path] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "figure_path" and isinstance(child, str) and child.strip():
                    figure_paths.append(Path(child).expanduser().resolve())
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return figure_paths


class Selector:
    def __init__(
        self,
        *,
        source_post_processed_dir: Path,
        percent: float,
        output_dir: Path | None,
        overwrite: bool,
    ) -> None:
        self.source_post_processed_dir = source_post_processed_dir.expanduser().resolve()
        if not self.source_post_processed_dir.is_dir():
            raise ValueError(
                f"Source post-processed directory not found: {self.source_post_processed_dir}"
            )
        self.source_run_dir = self.source_post_processed_dir.parent
        self.percent = _normalize_percent(percent)
        self.percent_label = _percent_label(self.percent)
        self.output_dir = (
            output_dir.expanduser().resolve()
            if output_dir is not None
            else _default_output_dir(
                self.source_post_processed_dir,
                percent_label=self.percent_label,
            )
        )
        if self.output_dir == self.source_post_processed_dir:
            raise ValueError("--output-dir must be different from the source post-processed dir")
        if self.output_dir.exists() and not overwrite:
            raise ValueError(
                f"Output directory already exists; pass --overwrite to reuse it: {self.output_dir}"
            )

        self.source_json_paths = _relative_json_paths(self.source_post_processed_dir)
        self.source_non_json_paths = _relative_non_json_paths(self.source_post_processed_dir)
        self._payload_cache: dict[str, Any] = {}
        self._selected_global_payload: dict[str, Any] | None = None
        self._source_requests_payload: dict[str, Any] | None = None
        self._selected_requests_payload: dict[str, Any] | None = None
        self._selected_prefill_payloads: dict[str, dict[str, Any]] | None = None
        self._selected_power_payload: dict[str, Any] | None = None
        self.window = self._build_window()

        self.written_json_paths: set[str] = set()
        self.written_non_json_paths: set[str] = set()
        self.skipped_json_paths: list[str] = []
        self.skipped_non_json_paths: list[str] = []
        self.generated_visualization_manifests: list[str] = []
        self.skipped_visualizations: list[dict[str, Any]] = []

    def _load_source_json(self, relative_path: str) -> Any:
        cached = self._payload_cache.get(relative_path)
        if cached is not None:
            return cached
        path = self.source_post_processed_dir / relative_path
        payload = _read_json(path)
        self._payload_cache[relative_path] = payload
        return payload

    def _has_source_json(self, relative_path: str) -> bool:
        return relative_path in self.source_json_paths

    def _output_path(self, relative_path: str) -> Path:
        return (self.output_dir / relative_path).resolve()

    def _build_window(self) -> SelectionWindow:
        relative_path = DEFAULT_REQUIRED_DISCOVERY_RELATIVE_PATH
        if relative_path not in self.source_json_paths:
            raise ValueError(
                "Missing required input file: "
                f"{(self.source_post_processed_dir / relative_path).resolve()}"
            )
        payload = self._load_source_json(relative_path)
        if not isinstance(payload, dict):
            raise ValueError(
                "Global timing summary must be a JSON object: "
                f"{(self.source_post_processed_dir / relative_path).resolve()}"
            )

        source_start_utc = _parse_iso8601_utc(payload.get("experiment_started_at"))
        if source_start_utc is None:
            raise ValueError(
                "global/trial-timing-summary.json is missing a valid "
                "experiment_started_at timestamp"
            )

        total_duration_s = _float_or_none(payload.get("total_duration_s"))
        if total_duration_s is None or total_duration_s < 0:
            source_finish_utc = _parse_iso8601_utc(payload.get("experiment_finished_at"))
            total_duration_s = _duration_seconds(source_start_utc, source_finish_utc)
        if total_duration_s is None or total_duration_s < 0:
            raise ValueError(
                "global/trial-timing-summary.json is missing a valid total_duration_s"
            )

        cutoff_offset_s = _round_s(total_duration_s * (1.0 - (self.percent / 100.0)))
        selected_duration_s = _round_s(total_duration_s - cutoff_offset_s)
        source_end_utc = source_start_utc + timedelta(seconds=total_duration_s)
        selected_start_utc = source_start_utc + timedelta(seconds=cutoff_offset_s)
        selected_end_utc = source_end_utc

        return SelectionWindow(
            percent=self.percent,
            percent_label=self.percent_label,
            source_run_dir=self.source_run_dir,
            source_post_processed_dir=self.source_post_processed_dir,
            output_dir=self.output_dir,
            original_started_at=_isoformat_utc(source_start_utc) or "",
            original_finished_at=_isoformat_utc(source_end_utc) or "",
            selected_started_at=_isoformat_utc(selected_start_utc) or "",
            selected_finished_at=_isoformat_utc(selected_end_utc) or "",
            source_start_utc=source_start_utc,
            source_end_utc=source_end_utc,
            selected_start_utc=selected_start_utc,
            selected_end_utc=selected_end_utc,
            original_duration_s=_round_s(total_duration_s),
            cutoff_offset_s=cutoff_offset_s,
            selected_duration_s=selected_duration_s,
        )

    def _write_output(self, relative_path: str, payload: Any) -> None:
        _write_json(self._output_path(relative_path), payload)
        self.written_json_paths.add(relative_path)

    def _derived_llm_requests_path(self) -> str:
        return str(self._output_path("gateway/llm-requests/llm-requests.json"))

    def _derived_power_summary_path(self) -> str:
        return str(self._output_path("power/power-summary.json"))

    def _derived_prefill_timeseries_path(self) -> str:
        return str(
            self._output_path("prefill-concurrency/prefill-concurrency-timeseries.json")
        )

    def _derived_vllm_timeseries_path(self, base_dir: str) -> str:
        return str(self._output_path(f"{base_dir}/gauge-counter-timeseries.json"))

    def _has_output_file(self, relative_path: str) -> bool:
        return self._output_path(relative_path).is_file()

    def _has_output_dir(self, relative_path: str) -> bool:
        return self._output_path(relative_path).is_dir()

    def _record_visualization_manifest(self, manifest_path: Path) -> None:
        resolved_manifest_path = manifest_path.expanduser().resolve()
        try:
            relative_manifest_path = str(resolved_manifest_path.relative_to(self.output_dir))
        except ValueError:
            return
        if resolved_manifest_path.is_file():
            self.written_json_paths.add(relative_manifest_path)
            self.generated_visualization_manifests.append(relative_manifest_path)

        try:
            manifest_payload = _read_json(resolved_manifest_path)
        except Exception:
            return
        for figure_path in _extract_figure_paths_from_manifest_payload(manifest_payload):
            try:
                relative_figure_path = str(figure_path.relative_to(self.output_dir))
            except ValueError:
                continue
            if figure_path.is_file():
                self.written_non_json_paths.add(relative_figure_path)

    def _run_visualization_generator(
        self,
        *,
        name: str,
        module_cache_key: str,
        module_relative_path: str,
        function_name: str,
        kwargs: dict[str, Any],
    ) -> None:
        helper = _load_helper_module(module_cache_key, module_relative_path)
        generator = getattr(helper, function_name)
        manifest_path = generator(self.source_run_dir, **kwargs)
        if not isinstance(manifest_path, Path):
            manifest_path = Path(manifest_path)
        self._record_visualization_manifest(manifest_path)

    def _skip_visualization(self, name: str, reason: str) -> None:
        self.skipped_visualizations.append(
            {
                "name": name,
                "reason": reason,
            }
        )

    def generate_visualization_outputs(self) -> None:
        if self._has_output_file("job-throughput/job-throughput-timeseries.json"):
            self._run_visualization_generator(
                name="job-throughput",
                module_cache_key="visualization_job_throughput",
                module_relative_path=(
                    "post-process/visualization/job-throughput/generate_all_figures.py"
                ),
                function_name="generate_figure_for_run_dir",
                kwargs={
                    "timeseries_input_path": self._output_path(
                        "job-throughput/job-throughput-timeseries.json"
                    ),
                    "output_dir": self._output_path("visualization/job-throughput"),
                    "image_format": DEFAULT_VISUALIZATION_FORMAT,
                    "dpi": DEFAULT_VISUALIZATION_DPI,
                },
            )

        if self._has_output_file("agent-output-throughput/agent-output-throughput.json"):
            self._run_visualization_generator(
                name="agent-output-throughput",
                module_cache_key="visualization_agent_output_throughput",
                module_relative_path=(
                    "post-process/visualization/agent-output-throughput/generate_all_figures.py"
                ),
                function_name="generate_figures_for_run_dir",
                kwargs={
                    "agent_output_input_path": self._output_path(
                        "agent-output-throughput/agent-output-throughput.json"
                    ),
                    "output_dir": self._output_path(
                        "visualization/agent-output-throughput"
                    ),
                    "image_format": DEFAULT_VISUALIZATION_FORMAT,
                    "dpi": DEFAULT_VISUALIZATION_DPI,
                },
            )

        if self._has_output_file("job-concurrency/job-concurrency-timeseries.json"):
            self._run_visualization_generator(
                name="job-concurrency",
                module_cache_key="visualization_job_concurrency",
                module_relative_path=(
                    "post-process/visualization/job-concurrency/generate_all_figures.py"
                ),
                function_name="generate_figure_for_run_dir",
                kwargs={
                    "timeseries_input_path": self._output_path(
                        "job-concurrency/job-concurrency-timeseries.json"
                    ),
                    "output_dir": self._output_path("visualization/job-concurrency"),
                    "image_format": DEFAULT_VISUALIZATION_FORMAT,
                    "dpi": DEFAULT_VISUALIZATION_DPI,
                },
            )

        if self._has_output_file(
            "prefill-concurrency/prefill-concurrency-timeseries.json"
        ):
            self._run_visualization_generator(
                name="prefill-concurrency",
                module_cache_key="visualization_prefill_concurrency",
                module_relative_path=(
                    "post-process/visualization/prefill-concurrency/generate_all_figures.py"
                ),
                function_name="generate_figure_for_run_dir",
                kwargs={
                    "timeseries_input_path": self._output_path(
                        "prefill-concurrency/prefill-concurrency-timeseries.json"
                    ),
                    "output_dir": self._output_path("visualization/prefill-concurrency"),
                    "image_format": DEFAULT_VISUALIZATION_FORMAT,
                    "dpi": DEFAULT_VISUALIZATION_DPI,
                },
            )

        if self._has_output_file("power/power-summary.json"):
            self._run_visualization_generator(
                name="power",
                module_cache_key="visualization_power",
                module_relative_path="post-process/visualization/power/generate_all_figures.py",
                function_name="generate_figure_for_run_dir",
                kwargs={
                    "power_input_path": self._output_path("power/power-summary.json"),
                    "output_dir": self._output_path("visualization/power"),
                    "image_format": DEFAULT_VISUALIZATION_FORMAT,
                    "dpi": DEFAULT_VISUALIZATION_DPI,
                },
            )

        vllm_base_dir: str | None = None
        for candidate_base_dir in ("vllm-log", "vllm-metrics"):
            if self._has_output_file(f"{candidate_base_dir}/gauge-counter-timeseries.json") and self._has_output_file(
                f"{candidate_base_dir}/gauge-counter-timeseries.stats.json"
            ):
                vllm_base_dir = candidate_base_dir
                break
        if vllm_base_dir is not None:
            self._run_visualization_generator(
                name="vllm-metrics",
                module_cache_key="visualization_vllm_metrics",
                module_relative_path=(
                    "post-process/visualization/vllm-metrics/generate_all_figures.py"
                ),
                function_name="generate_figures_for_run_dir",
                kwargs={
                    "timeseries_input_path": self._output_path(
                        f"{vllm_base_dir}/gauge-counter-timeseries.json"
                    ),
                    "stats_input_path": self._output_path(
                        f"{vllm_base_dir}/gauge-counter-timeseries.stats.json"
                    ),
                    "output_dir": self._output_path("visualization/vllm-metrics"),
                    "image_format": DEFAULT_VISUALIZATION_FORMAT,
                    "dpi": DEFAULT_VISUALIZATION_DPI,
                },
            )

        if self._has_output_dir("gateway/stack"):
            helper = _load_helper_module(
                "visualization_gateway_stack",
                "post-process/visualization/gateway-stack/generate_all_figures.py",
            )
            required_input_names = [
                spec.get("input_name")
                for spec in getattr(helper, "METRIC_SPECS", [])
                if isinstance(spec, dict) and isinstance(spec.get("input_name"), str)
            ]
            if required_input_names and all(
                self._has_output_file(f"gateway/stack/{input_name}")
                for input_name in required_input_names
            ):
                self._run_visualization_generator(
                    name="gateway-stack",
                    module_cache_key="visualization_gateway_stack",
                    module_relative_path=(
                        "post-process/visualization/gateway-stack/generate_all_figures.py"
                    ),
                    function_name="generate_figures_for_run_dir",
                    kwargs={
                        "stack_input_dir": self._output_path("gateway/stack"),
                        "output_dir": self._output_path("visualization/gateway-stack"),
                        "image_format": DEFAULT_VISUALIZATION_FORMAT,
                        "dpi": DEFAULT_VISUALIZATION_DPI,
                    },
                )
            elif any(
                self._has_output_file(f"gateway/stack/{input_name}")
                for input_name in required_input_names
            ):
                self._skip_visualization(
                    "gateway-stack",
                    "Selected output is missing one or more required stacked histogram inputs",
                )

        if self._has_output_file("gateway/stack-context/context-usage-stacked-histogram.json"):
            self._run_visualization_generator(
                name="gateway-stack-context",
                module_cache_key="visualization_gateway_stack_context",
                module_relative_path=(
                    "post-process/visualization/gateway-stack-context/generate_all_figures.py"
                ),
                function_name="generate_figures_for_run_dir",
                kwargs={
                    "stack_context_input_dir": self._output_path("gateway/stack-context"),
                    "output_dir": self._output_path("visualization/gateway-stack-context"),
                    "image_format": DEFAULT_VISUALIZATION_FORMAT,
                    "dpi": DEFAULT_VISUALIZATION_DPI,
                },
            )

        if self._has_output_file("gateway/stack-kv/kv-usage-stacked-histogram.json"):
            self._run_visualization_generator(
                name="gateway-stack-kv",
                module_cache_key="visualization_gateway_stack_kv",
                module_relative_path=(
                    "post-process/visualization/gateway-stack-kv/generate_all_figures.py"
                ),
                function_name="generate_figures_for_run_dir",
                kwargs={
                    "stack_kv_input_dir": self._output_path("gateway/stack-kv"),
                    "output_dir": self._output_path("visualization/gateway-stack-kv"),
                    "image_format": DEFAULT_VISUALIZATION_FORMAT,
                    "dpi": DEFAULT_VISUALIZATION_DPI,
                },
            )

    def build_selected_global_payload(self) -> dict[str, Any]:
        if self._selected_global_payload is not None:
            return self._selected_global_payload

        source_payload = self._load_source_json("global/trial-timing-summary.json")
        if not isinstance(source_payload, dict):
            raise ValueError("global/trial-timing-summary.json must be a JSON object")

        def select_intervals(raw_items: Any) -> list[dict[str, Any]]:
            selected: list[dict[str, Any]] = []
            if not isinstance(raw_items, list):
                return selected
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                start_offset_s = _float_or_none(item.get("start_offset_s"))
                end_offset_s = _float_or_none(item.get("end_offset_s"))
                if start_offset_s is None or end_offset_s is None:
                    continue
                clipped = _clip_range_to_window(
                    start_s=start_offset_s,
                    end_s=end_offset_s,
                    window=self.window,
                )
                if clipped is None:
                    continue
                clipped_start_abs, clipped_end_abs, rebased_start_s, rebased_end_s = clipped
                updated = copy.deepcopy(item)
                updated["started_at"] = _isoformat_utc(
                    self.window.source_start_utc + timedelta(seconds=clipped_start_abs)
                )
                updated["finished_at"] = _isoformat_utc(
                    self.window.source_start_utc + timedelta(seconds=clipped_end_abs)
                )
                updated["start_offset_s"] = rebased_start_s
                updated["end_offset_s"] = rebased_end_s
                updated["duration_s"] = _round_s(rebased_end_s - rebased_start_s)
                selected.append(updated)
            selected.sort(
                key=lambda item: (
                    _float_or_none(item.get("start_offset_s"))
                    if _float_or_none(item.get("start_offset_s")) is not None
                    else float("inf"),
                    _float_or_none(item.get("end_offset_s"))
                    if _float_or_none(item.get("end_offset_s")) is not None
                    else float("inf"),
                    str(item.get("trial_id") or item.get("worker_id") or ""),
                )
            )
            return selected

        selected_trials = select_intervals(source_payload.get("trials"))
        selected_trails = select_intervals(source_payload.get("trails"))
        durations = [
            _float_or_none(trial.get("duration_s"))
            for trial in selected_trials
            if _float_or_none(trial.get("duration_s")) is not None
        ]
        stats = _stats_from_values([float(duration) for duration in durations])

        result = copy.deepcopy(source_payload)
        result["experiment_started_at"] = self.window.selected_started_at
        result["experiment_finished_at"] = self.window.selected_finished_at
        result["total_duration_s"] = self.window.selected_duration_s
        result["trial_count"] = len(selected_trials)
        result["trail_count"] = len(selected_trails)
        result["trial_duration_stats_s"] = stats
        result["trials"] = selected_trials
        result["trails"] = selected_trails

        self._selected_global_payload = result
        return result

    def _selected_trial_records(self) -> list[dict[str, Any]]:
        payload = self.build_selected_global_payload()
        raw_trials = payload.get("trials")
        if not isinstance(raw_trials, list):
            return []
        return [item for item in raw_trials if isinstance(item, dict)]

    def build_global_progress_payload(self) -> dict[str, Any]:
        source_payload = self._load_source_json("global-progress/replay-progress-summary.json")
        if not isinstance(source_payload, dict):
            raise ValueError("global-progress/replay-progress-summary.json must be a JSON object")

        selected_trials = sorted(
            self._selected_trial_records(),
            key=lambda item: (
                _float_or_none(item.get("end_offset_s"))
                if _float_or_none(item.get("end_offset_s")) is not None
                else float("inf"),
                str(item.get("trial_id") or ""),
            ),
        )
        milestone_step = _int_or_none(source_payload.get("milestone_step"))
        if milestone_step is None or milestone_step <= 0:
            milestone_step = DEFAULT_MILESTONE_STEP

        milestones: list[dict[str, Any]] = []
        finished_count = len(selected_trials)
        milestone_indices = list(range(milestone_step, finished_count + 1, milestone_step))
        if finished_count > 0 and (not milestone_indices or milestone_indices[-1] != finished_count):
            milestone_indices.append(finished_count)
        for replay_count in milestone_indices:
            finish_time_s = _float_or_none(selected_trials[replay_count - 1].get("end_offset_s"))
            milestones.append(
                {
                    "replay_count": replay_count,
                    "finish_time_s": finish_time_s,
                }
            )

        return {
            "source_run_dir": source_payload.get("source_run_dir"),
            "source_type": source_payload.get("source_type"),
            "experiment_started_at": self.window.selected_started_at,
            "replay_count": finished_count,
            "finished_replay_count": finished_count,
            "milestone_step": milestone_step,
            "milestones": milestones,
        }

    def build_job_throughput_payload(self) -> dict[str, Any]:
        source_payload = self._load_source_json("job-throughput/job-throughput-timeseries.json")
        if not isinstance(source_payload, dict):
            raise ValueError("job-throughput/job-throughput-timeseries.json must be a JSON object")

        helper = _load_helper_module(
            "job_throughput_extract",
            "post-process/job-throughput/extract_run.py",
        )
        timepoint_frequency_hz = _float_or_none(source_payload.get("timepoint_frequency_hz"))
        if timepoint_frequency_hz is None or timepoint_frequency_hz <= 0:
            timepoint_frequency_hz = DEFAULT_TIMEPOINT_FREQUENCY_HZ
        window_size_s = _float_or_none(source_payload.get("window_size_s"))
        if window_size_s is None or window_size_s <= 0:
            window_size_s = DEFAULT_WINDOW_SIZE_S

        selected_trials = self._selected_trial_records()
        completion_offsets_s = [
            float(end_offset_s)
            for item in selected_trials
            if (end_offset_s := _float_or_none(item.get("end_offset_s"))) is not None
        ]
        completion_offsets_s_excluding_cancelled = [
            float(end_offset_s)
            for item in selected_trials
            if (end_offset_s := _float_or_none(item.get("end_offset_s"))) is not None
            and not _is_cancelled_status(item.get("status"))
        ]

        throughput_points = helper._build_throughput_points(
            completion_offsets_s=completion_offsets_s,
            total_duration_s=self.window.selected_duration_s,
            timepoint_freq_hz=timepoint_frequency_hz,
            window_size_s=window_size_s,
        )
        throughput_points_excluding_cancelled = helper._build_throughput_points(
            completion_offsets_s=completion_offsets_s_excluding_cancelled,
            total_duration_s=self.window.selected_duration_s,
            timepoint_freq_hz=timepoint_frequency_hz,
            window_size_s=window_size_s,
        )

        return {
            "source_run_dir": source_payload.get("source_run_dir"),
            "source_type": source_payload.get("source_type"),
            "experiment_started_at": self.window.selected_started_at,
            "experiment_finished_at": self.window.selected_finished_at,
            "time_constraint_s": self.window.selected_duration_s,
            "service_failure_detected": bool(
                source_payload.get("service_failure_detected", False)
            ),
            "service_failure_cutoff_time_utc": source_payload.get("service_failure_cutoff_time_utc"),
            "replay_count": len(selected_trials),
            "finished_replay_count": len(completion_offsets_s),
            "finished_replay_count_excluding_cancelled": len(
                completion_offsets_s_excluding_cancelled
            ),
            "cancelled_finished_replay_count": len(completion_offsets_s)
            - len(completion_offsets_s_excluding_cancelled),
            "total_duration_s": self.window.selected_duration_s,
            "timepoint_frequency_hz": timepoint_frequency_hz,
            "timepoint_interval_s": _round_s(1.0 / timepoint_frequency_hz),
            "window_size_s": window_size_s,
            "window_width_s": _round_s(window_size_s * 2.0),
            "sample_count": len(throughput_points),
            "throughput_points": throughput_points,
            "throughput_points_excluding_cancelled": throughput_points_excluding_cancelled,
        }

    def build_job_concurrency_payload(self) -> dict[str, Any]:
        source_payload = self._load_source_json("job-concurrency/job-concurrency-timeseries.json")
        if not isinstance(source_payload, dict):
            raise ValueError("job-concurrency/job-concurrency-timeseries.json must be a JSON object")

        helper = _load_helper_module(
            "job_concurrency_extract",
            "post-process/job-concurrency/extract_run.py",
        )
        intervals = [
            {
                "job_id": item.get("trial_id") or item.get("worker_id"),
                "status": item.get("status"),
                "start_offset_s": item.get("start_offset_s"),
                "end_offset_s": item.get("end_offset_s"),
                "duration_s": item.get("duration_s"),
            }
            for item in self._selected_trial_records()
        ]
        concurrency_points = helper._build_concurrency_points(
            job_intervals=intervals,
            total_duration_s=self.window.selected_duration_s,
        )
        max_concurrency = max((point["concurrency"] for point in concurrency_points), default=0)
        avg_concurrency = (
            _round_value(
                sum(point["concurrency"] for point in concurrency_points)
                / len(concurrency_points)
            )
            if concurrency_points
            else 0.0
        )

        return {
            "source_run_dir": source_payload.get("source_run_dir"),
            "source_type": source_payload.get("source_type"),
            "experiment_started_at": self.window.selected_started_at,
            "experiment_finished_at": self.window.selected_finished_at,
            "time_constraint_s": self.window.selected_duration_s,
            "service_failure_detected": bool(
                source_payload.get("service_failure_detected", False)
            ),
            "service_failure_cutoff_time_utc": source_payload.get("service_failure_cutoff_time_utc"),
            "replay_count": len(intervals),
            "jobs_with_valid_range_count": len(intervals),
            "total_duration_s": self.window.selected_duration_s,
            "sample_count": len(concurrency_points),
            "max_concurrency": max_concurrency,
            "avg_concurrency": avg_concurrency,
            "concurrency_points": concurrency_points,
        }

    def build_agent_output_throughput_payload(self) -> dict[str, Any]:
        source_payload = self._load_source_json(
            "agent-output-throughput/agent-output-throughput.json"
        )
        if not isinstance(source_payload, dict):
            raise ValueError(
                "agent-output-throughput/agent-output-throughput.json must be a JSON object"
            )

        helper = _load_helper_module(
            "agent_output_throughput_extract",
            "post-process/agent-output-throughput/extract_run.py",
        )

        api_token_hash_by_run_id: dict[str, Any] = {}
        gateway_profile_id_by_run_id: dict[str, Any] = {}
        raw_agents = source_payload.get("agents")
        if isinstance(raw_agents, list):
            for agent in raw_agents:
                if not isinstance(agent, dict):
                    continue
                gateway_run_id = agent.get("gateway_run_id")
                if not isinstance(gateway_run_id, str) or not gateway_run_id:
                    continue
                api_token_hash_by_run_id[gateway_run_id] = agent.get("api_token_hash")
                gateway_profile_id_by_run_id[gateway_run_id] = agent.get(
                    "gateway_profile_id"
                )

        run_accumulator = helper._new_throughput_accumulator()
        agent_accumulators: dict[str, dict[str, Any]] = {}

        for record in self._selected_request_records():
            completion_tokens = _int_or_none(record.get("completion_tokens"))
            if completion_tokens is not None and completion_tokens < 0:
                completion_tokens = None
            request_duration_s = helper._extract_request_duration_s(record)
            helper._add_request_to_accumulator(
                run_accumulator,
                completion_tokens=completion_tokens,
                request_duration_s=request_duration_s,
            )

            gateway_run_id = record.get("gateway_run_id")
            if not isinstance(gateway_run_id, str) or not gateway_run_id:
                continue

            agent_payload = agent_accumulators.get(gateway_run_id)
            if agent_payload is None:
                gateway_profile_id = record.get("gateway_profile_id")
                if gateway_profile_id is None:
                    gateway_profile_id = gateway_profile_id_by_run_id.get(
                        gateway_run_id
                    )
                agent_payload = {
                    "gateway_run_id": gateway_run_id,
                    "gateway_profile_id": gateway_profile_id,
                    "api_token_hash": api_token_hash_by_run_id.get(gateway_run_id),
                    "_throughput_accumulator": helper._new_throughput_accumulator(),
                }
                agent_accumulators[gateway_run_id] = agent_payload

            helper._add_request_to_accumulator(
                agent_payload["_throughput_accumulator"],
                completion_tokens=completion_tokens,
                request_duration_s=request_duration_s,
            )

        agents: list[dict[str, Any]] = []
        for gateway_run_id in sorted(agent_accumulators):
            agent_payload = agent_accumulators[gateway_run_id]
            throughput_payload = helper._payload_from_accumulator(
                agent_payload.pop("_throughput_accumulator")
            )
            throughput_payload.update(
                {
                    "gateway_run_id": agent_payload["gateway_run_id"],
                    "gateway_profile_id": agent_payload.get("gateway_profile_id"),
                    "api_token_hash": agent_payload.get("api_token_hash"),
                }
            )
            agents.append(throughput_payload)

        throughput_values = [
            float(agent["output_throughput_tokens_per_s"])
            for agent in agents
            if isinstance(agent.get("output_throughput_tokens_per_s"), (int, float))
        ]

        result = helper._payload_from_accumulator(run_accumulator)
        result.update(
            {
                "source_run_dir": source_payload.get("source_run_dir"),
                "source_gateway_output_dir": source_payload.get(
                    "source_gateway_output_dir"
                ),
                "service_failure_detected": bool(
                    source_payload.get("service_failure_detected", False)
                ),
                "service_failure_cutoff_time_utc": source_payload.get(
                    "service_failure_cutoff_time_utc"
                ),
                "agent_count": len(agents),
                "agent_output_throughput_tokens_per_s_summary": helper._summarize_values(
                    throughput_values
                ),
                "agent_output_throughput_tokens_per_s_histogram": (
                    helper.build_agent_output_throughput_histogram(throughput_values)
                ),
                "agents": agents,
            }
        )
        return result

    def _load_source_requests_payload(self) -> dict[str, Any]:
        if self._source_requests_payload is not None:
            return self._source_requests_payload
        payload = self._load_source_json("gateway/llm-requests/llm-requests.json")
        if not isinstance(payload, dict):
            raise ValueError("gateway/llm-requests/llm-requests.json must be a JSON object")
        self._source_requests_payload = payload
        return payload

    def build_selected_requests_payload(self) -> dict[str, Any]:
        if self._selected_requests_payload is not None:
            return self._selected_requests_payload

        source_payload = self._load_source_requests_payload()
        raw_requests = source_payload.get("requests")
        if not isinstance(raw_requests, list):
            raise ValueError("gateway/llm-requests/llm-requests.json is missing requests[]")

        selected_requests: list[dict[str, Any]] = []
        for item in raw_requests:
            if not isinstance(item, dict):
                continue
            request_start_offset_s = _float_or_none(item.get("request_start_offset_s"))
            request_end_offset_s = _float_or_none(item.get("request_end_offset_s"))
            if request_start_offset_s is None or request_end_offset_s is None:
                continue
            clipped = _clip_range_to_window(
                start_s=request_start_offset_s,
                end_s=request_end_offset_s,
                window=self.window,
            )
            if clipped is None:
                continue
            clipped_start_abs, clipped_end_abs, rebased_start_s, rebased_end_s = clipped
            updated = copy.deepcopy(item)
            updated["request_start_time"] = _isoformat_utc(
                self.window.source_start_utc + timedelta(seconds=clipped_start_abs)
            )
            updated["request_end_time"] = _isoformat_utc(
                self.window.source_start_utc + timedelta(seconds=clipped_end_abs)
            )
            updated["request_start_offset_s"] = rebased_start_s
            updated["request_end_offset_s"] = rebased_end_s
            updated["request_end_to_run_end_s"] = _round_s(
                max(self.window.selected_duration_s - rebased_end_s, 0.0)
            )
            clipped_duration_s = _round_s(rebased_end_s - rebased_start_s)
            if "request_duration_ms" in updated:
                updated["request_duration_ms"] = _round_ms_from_seconds(clipped_duration_s)
            if "duration_ms" in updated:
                updated["duration_ms"] = _round_ms_from_seconds(clipped_duration_s)
            selected_requests.append(updated)

        selected_requests.sort(
            key=lambda item: (
                _parse_iso8601_utc(item.get("request_start_time")) or datetime.max.replace(tzinfo=timezone.utc),
                str(item.get("request_id") or ""),
            )
        )

        self._selected_requests_payload = {
            "source_run_dir": source_payload.get("source_run_dir"),
            "source_gateway_output_dir": source_payload.get("source_gateway_output_dir"),
            "service_failure_detected": bool(
                source_payload.get("service_failure_detected", False)
            ),
            "service_failure_cutoff_time_utc": source_payload.get("service_failure_cutoff_time_utc"),
            "request_count": len(selected_requests),
            "requests": selected_requests,
        }
        return self._selected_requests_payload

    def _selected_request_records(self) -> list[dict[str, Any]]:
        payload = self.build_selected_requests_payload()
        raw = payload.get("requests")
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def _selected_request_map(self) -> dict[tuple[str, str], dict[str, Any]]:
        mapping: dict[tuple[str, str], dict[str, Any]] = {}
        for item in self._selected_request_records():
            gateway_run_id = str(item.get("gateway_run_id") or "")
            request_id = str(item.get("request_id") or "")
            mapping[(gateway_run_id, request_id)] = item
        return mapping

    def build_llm_request_related_payloads(self) -> dict[str, dict[str, Any]]:
        selected_requests_payload = self.build_selected_requests_payload()
        selected_records = self._selected_request_records()
        source_payload = self._load_source_requests_payload()
        llm_helper = _load_helper_module(
            "gateway_llm_requests_extract",
            "post-process/gateway/llm-requests/extract_run.py",
        )

        base = {
            "source_run_dir": source_payload.get("source_run_dir"),
            "source_gateway_output_dir": source_payload.get("source_gateway_output_dir"),
            "service_failure_detected": bool(
                source_payload.get("service_failure_detected", False)
            ),
            "service_failure_cutoff_time_utc": source_payload.get("service_failure_cutoff_time_utc"),
        }

        stats_payload = {
            **base,
            "request_count": len(selected_records),
            **llm_helper.build_numeric_stats(selected_records),
        }
        stage_speed_summary = llm_helper.build_average_stage_speed_tokens_per_s(selected_records)
        stats_payload["average_stage_speed_tokens_per_s"] = stage_speed_summary
        speed_stats_payload = {
            **base,
            "request_count": len(selected_records),
            "average_stage_speed_tokens_per_s": stage_speed_summary,
        }
        status_payloads = llm_helper.build_stats_by_status_code(selected_records)
        longest_requests, shortest_requests = llm_helper.select_extreme_duration_requests(
            selected_records,
            limit=10,
        )

        payloads: dict[str, dict[str, Any]] = {
            "gateway/llm-requests/llm-requests.json": selected_requests_payload,
            "gateway/llm-requests/llm-request-stats.json": stats_payload,
            "gateway/llm-requests/llm-request-speed-stats.json": speed_stats_payload,
            "gateway/llm-requests/llm-requests-longest-10.json": {
                **base,
                "request_count": len(selected_records),
                "selected_count": len(longest_requests),
                "selection": "longest",
                "limit": 10,
                "duration_field": "request_duration_ms (fallback: duration_ms)",
                "requests": longest_requests,
            },
            "gateway/llm-requests/llm-requests-shortest-10.json": {
                **base,
                "request_count": len(selected_records),
                "selected_count": len(shortest_requests),
                "selection": "shortest",
                "limit": 10,
                "duration_field": "request_duration_ms (fallback: duration_ms)",
                "requests": shortest_requests,
            },
        }

        for relative_path in sorted(self.source_json_paths):
            if not relative_path.startswith("gateway/llm-requests/llm-requests-stats."):
                continue
            status_suffix = Path(relative_path).suffixes
            if len(status_suffix) < 2:
                continue
            status_key = Path(relative_path).stem.split(".")[-1]
            status_payload = status_payloads.get(
                status_key,
                {
                    "status_code": int(status_key),
                    "request_count": 0,
                    "metric_count": 0,
                    "metrics": {},
                },
            )
            payloads[relative_path] = {
                **base,
                **status_payload,
            }
        return payloads

    def build_usage_summary_payload(self) -> dict[str, Any]:
        source_payload = self._load_source_json("gateway/usage/usage-summary.json")
        if not isinstance(source_payload, dict):
            raise ValueError("gateway/usage/usage-summary.json must be a JSON object")

        usage_helper = _load_helper_module(
            "gateway_usage_extract",
            "post-process/gateway/usage/extract_run.py",
        )

        api_token_hash_by_run_id: dict[str, Any] = {}
        raw_agents = source_payload.get("agents")
        if isinstance(raw_agents, list):
            for agent in raw_agents:
                if not isinstance(agent, dict):
                    continue
                run_id = agent.get("gateway_run_id")
                if isinstance(run_id, str):
                    api_token_hash_by_run_id[run_id] = agent.get("api_token_hash")

        agent_accumulators: dict[str, dict[str, Any]] = {}
        run_accumulator = usage_helper._new_usage_accumulator()

        for record in self._selected_request_records():
            gateway_run_id = record.get("gateway_run_id")
            if not isinstance(gateway_run_id, str) or not gateway_run_id:
                continue
            agent_payload = agent_accumulators.get(gateway_run_id)
            if agent_payload is None:
                agent_payload = {
                    "gateway_run_id": gateway_run_id,
                    "gateway_profile_id": record.get("gateway_profile_id"),
                    "api_token_hash": api_token_hash_by_run_id.get(gateway_run_id),
                    "request_count": 0,
                    "_usage_accumulator": usage_helper._new_usage_accumulator(),
                }
                agent_accumulators[gateway_run_id] = agent_payload

            prompt_tokens = _int_or_none(record.get("prompt_tokens"))
            completion_tokens = _int_or_none(record.get("completion_tokens"))
            cached_tokens = _int_or_none(record.get("cached_tokens"))
            usage_helper._add_request_usage(
                agent_payload["_usage_accumulator"],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_prompt_tokens=cached_tokens,
            )
            usage_helper._add_request_usage(
                run_accumulator,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_prompt_tokens=cached_tokens,
            )
            agent_payload["request_count"] += 1

        agents: list[dict[str, Any]] = []
        request_length_totals: list[int] = []
        for gateway_run_id in sorted(agent_accumulators):
            agent_payload = agent_accumulators[gateway_run_id]
            usage_payload = usage_helper._usage_payload_from_accumulator(
                agent_payload.pop("_usage_accumulator")
            )
            max_request_length = usage_payload.get("max_request_length")
            if isinstance(max_request_length, int):
                request_length_totals.append(max_request_length)
            agents.append(
                {
                    "gateway_run_id": agent_payload["gateway_run_id"],
                    "gateway_profile_id": agent_payload.get("gateway_profile_id"),
                    "api_token_hash": agent_payload.get("api_token_hash"),
                    "request_count": agent_payload["request_count"],
                    "usage": usage_payload,
                }
            )

        run_usage = usage_helper._usage_payload_from_accumulator(run_accumulator)
        run_usage["avg_worker_max_request_length"] = (
            sum(request_length_totals) / len(request_length_totals)
            if request_length_totals
            else None
        )

        return {
            "source_run_dir": source_payload.get("source_run_dir"),
            "source_gateway_output_dir": source_payload.get("source_gateway_output_dir"),
            "agent_count": len(agents),
            "request_count": len(self._selected_request_records()),
            "usage": run_usage,
            "agents": agents,
        }

    def build_prefill_payloads(self) -> dict[str, dict[str, Any]]:
        if self._selected_prefill_payloads is not None:
            return self._selected_prefill_payloads

        source_payload = self._load_source_requests_payload()
        prefill_helper = _load_helper_module(
            "prefill_concurrency_extract",
            "post-process/prefill-concurrency/extract_run.py",
        )
        selected_request_map = self._selected_request_map()

        raw_requests = source_payload.get("requests")
        original_records = [item for item in raw_requests if isinstance(item, dict)] if isinstance(raw_requests, list) else []
        raw_activities = prefill_helper._build_prefill_activities(original_records)

        selected_activities: list[dict[str, Any]] = []
        for activity in raw_activities:
            prefill_start_offset_s = _float_or_none(activity.get("prefill_start_offset_s"))
            prefill_end_offset_s = _float_or_none(activity.get("prefill_end_offset_s"))
            if prefill_start_offset_s is None or prefill_end_offset_s is None:
                continue
            clipped = _clip_range_to_window(
                start_s=prefill_start_offset_s,
                end_s=prefill_end_offset_s,
                window=self.window,
            )
            if clipped is None:
                continue
            _clipped_start_abs, _clipped_end_abs, rebased_start_s, rebased_end_s = clipped
            updated = copy.deepcopy(activity)
            selected_request = selected_request_map.get(
                (
                    str(activity.get("gateway_run_id") or ""),
                    str(activity.get("request_id") or ""),
                )
            )
            if selected_request is not None:
                updated["request_start_offset_s"] = selected_request.get("request_start_offset_s")
                updated["request_end_offset_s"] = selected_request.get("request_end_offset_s")
            updated["prefill_start_offset_s"] = rebased_start_s
            updated["prefill_end_offset_s"] = rebased_end_s
            updated["prefill_duration_s"] = _round_s(rebased_end_s - rebased_start_s)
            request_start_offset_s = _float_or_none(updated.get("request_start_offset_s"))
            if request_start_offset_s is None:
                request_start_offset_s = 0.0
            updated["time_in_queue_s"] = _round_s(max(rebased_start_s - request_start_offset_s, 0.0))
            selected_activities.append(updated)

        tick_ms = DEFAULT_PREFILL_TICK_MS
        if self._has_source_json("prefill-concurrency/prefill-concurrency-timeseries.json"):
            source_timeseries = self._load_source_json(
                "prefill-concurrency/prefill-concurrency-timeseries.json"
            )
            if isinstance(source_timeseries, dict):
                parsed_tick_ms = _int_or_none(source_timeseries.get("tick_ms"))
                if parsed_tick_ms is not None and parsed_tick_ms > 0:
                    tick_ms = parsed_tick_ms
        tick_s = _round_s(tick_ms / 1000.0)

        selected_activities.sort(
            key=lambda activity: (
                _float_or_none(activity.get("prefill_start_offset_s"))
                if _float_or_none(activity.get("prefill_start_offset_s")) is not None
                else float("inf"),
                str(activity.get("request_id") or ""),
            )
        )

        concurrency_points = prefill_helper._build_prefill_concurrency_points(
            selected_activities,
            total_duration_s=self.window.selected_duration_s,
            tick_s=tick_s,
        )
        min_concurrency = min((point["concurrency"] for point in concurrency_points), default=0)
        max_concurrency = max((point["concurrency"] for point in concurrency_points), default=0)
        avg_concurrency = (
            _round_value(
                sum(point["concurrency"] for point in concurrency_points)
                / len(concurrency_points)
            )
            if concurrency_points
            else 0.0
        )

        base = {
            "source_run_dir": source_payload.get("source_run_dir"),
            "source_llm_requests_path": self._derived_llm_requests_path(),
            "source_gateway_output_dir": source_payload.get("source_gateway_output_dir"),
            "service_failure_detected": bool(
                source_payload.get("service_failure_detected", False)
            ),
            "service_failure_cutoff_time_utc": source_payload.get("service_failure_cutoff_time_utc"),
            "request_count": len(self._selected_request_records()),
            "prefill_activity_count": len(selected_activities),
            "total_duration_s": self.window.selected_duration_s,
            "tick_ms": tick_ms,
            "tick_s": tick_s,
        }

        self._selected_prefill_payloads = {
            "prefill-concurrency/prefill-activities.json": {
                **base,
                "activities": selected_activities,
            },
            "prefill-concurrency/prefill-concurrency-timeseries.json": {
                **base,
                "sample_count": len(concurrency_points),
                "concurrency_points": concurrency_points,
            },
            "prefill-concurrency/prefill-concurrency-stats.json": {
                **base,
                "sample_count": len(concurrency_points),
                "min_concurrency": min_concurrency,
                "max_concurrency": max_concurrency,
                "avg_concurrency": avg_concurrency,
            },
        }
        return self._selected_prefill_payloads

    def build_split_duration_payload(self) -> dict[str, Any]:
        source_payload = self._load_source_json("split/duration/duration-split-summary.json")
        if not isinstance(source_payload, dict):
            raise ValueError("split/duration/duration-split-summary.json must be a JSON object")
        split_helper = _load_helper_module(
            "split_duration_extract",
            "post-process/split/duration/extract_run.py",
        )

        grouped_requests: dict[str, list[dict[str, Any]]] = {}
        profile_by_run_id: dict[str, Any] = {}
        for record in self._selected_request_records():
            gateway_run_id = record.get("gateway_run_id")
            if not isinstance(gateway_run_id, str) or not gateway_run_id:
                continue
            grouped_requests.setdefault(gateway_run_id, []).append(record)
            profile_by_run_id[gateway_run_id] = record.get("gateway_profile_id")

        jobs: list[dict[str, Any]] = []
        for gateway_run_id in sorted(grouped_requests):
            records = grouped_requests[gateway_run_id]
            request_count_total = len(records)
            request_count_with_usage = 0
            prompt_tokens_total = 0
            decode_tokens_total = 0
            cached_prompt_tokens_total = 0
            max_request_length: int | None = None
            first_valid_request_start: datetime | None = None
            last_valid_request_end: datetime | None = None

            for record in records:
                prompt_tokens = _int_or_none(record.get("prompt_tokens"))
                decode_tokens = _int_or_none(record.get("completion_tokens"))
                cached_tokens = _int_or_none(record.get("cached_tokens"))
                if prompt_tokens is None or decode_tokens is None:
                    continue
                request_count_with_usage += 1
                prompt_tokens_total += prompt_tokens
                decode_tokens_total += decode_tokens
                if cached_tokens is not None:
                    cached_prompt_tokens_total += cached_tokens

                request_length = prompt_tokens + decode_tokens
                if max_request_length is None:
                    max_request_length = request_length
                else:
                    max_request_length = max(max_request_length, request_length)

                request_start = _parse_iso8601_utc(record.get("request_start_time"))
                request_end = _parse_iso8601_utc(record.get("request_end_time"))
                if request_start is not None and (
                    first_valid_request_start is None or request_start < first_valid_request_start
                ):
                    first_valid_request_start = request_start
                if request_end is not None and (
                    last_valid_request_end is None or request_end > last_valid_request_end
                ):
                    last_valid_request_end = request_end

            duration_s = _duration_seconds(first_valid_request_start, last_valid_request_end)
            jobs.append(
                {
                    "gateway_run_id": gateway_run_id,
                    "gateway_profile_id": profile_by_run_id.get(gateway_run_id),
                    "max_request_length": max_request_length,
                    "duration_s": duration_s,
                    "turn_count": request_count_with_usage,
                    "request_count_total": request_count_total,
                    "request_count_with_usage": request_count_with_usage,
                    "prompt_tokens": prompt_tokens_total,
                    "decode_tokens": decode_tokens_total,
                    "cached_prompt_tokens": cached_prompt_tokens_total,
                }
            )

        split_count = _int_or_none(source_payload.get("split_count"))
        if split_count is None or split_count <= 0:
            split_count = 10

        ranked_jobs = [
            job for job in jobs if _int_or_none(job.get("max_request_length")) is not None
        ]
        excluded_jobs = [
            job for job in jobs if _int_or_none(job.get("max_request_length")) is None
        ]
        ranked_jobs = split_helper._rank_jobs_by_max_request_length(
            ranked_jobs,
            split_count=split_count,
        )
        tables = split_helper._build_tables(ranked_jobs, split_count=split_count)

        return {
            "source_run_dir": source_payload.get("source_run_dir"),
            "source_gateway_output_dir": source_payload.get("source_gateway_output_dir"),
            "service_failure_detected": bool(
                source_payload.get("service_failure_detected", False)
            ),
            "service_failure_cutoff_time_utc": source_payload.get("service_failure_cutoff_time_utc"),
            "split_count": split_count,
            "bin_labels": split_helper._bin_labels(split_count),
            "job_count_total": len(jobs),
            "job_count": len(ranked_jobs),
            "job_count_excluded_no_token_usage": len(excluded_jobs),
            "metrics": source_payload.get("metrics") or list(split_helper.METRIC_NAMES),
            "tables": tables,
            "jobs": ranked_jobs,
            "excluded_jobs_no_token_usage": excluded_jobs,
        }

    def _select_vllm_timeseries_payload(self, base_dir: str) -> dict[str, Any]:
        relative_path = f"{base_dir}/gauge-counter-timeseries.json"
        source_payload = self._load_source_json(relative_path)
        if not isinstance(source_payload, dict):
            raise ValueError(f"{relative_path} must be a JSON object")

        metrics = source_payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{relative_path} is missing metrics{{}}")

        selected_metrics: dict[str, Any] = {}
        first_captured_at: datetime | None = None
        for metric_key, metric_payload in metrics.items():
            if not isinstance(metric_payload, dict):
                continue
            captured_at = metric_payload.get("captured_at")
            time_from_start_s = metric_payload.get("time_from_start_s")
            values = metric_payload.get("value")
            if (
                not isinstance(captured_at, list)
                or not isinstance(time_from_start_s, list)
                or not isinstance(values, list)
            ):
                selected_metrics[metric_key] = copy.deepcopy(metric_payload)
                continue
            if len(captured_at) != len(time_from_start_s) or len(captured_at) != len(values):
                selected_metrics[metric_key] = copy.deepcopy(metric_payload)
                continue

            kept_captured_at: list[Any] = []
            kept_time_from_start_s: list[float] = []
            kept_values: list[Any] = []
            for index, raw_offset in enumerate(time_from_start_s):
                offset_s = _float_or_none(raw_offset)
                if offset_s is None:
                    continue
                rebased_offset_s = _rebase_point_offset(offset_s, window=self.window)
                if rebased_offset_s is None:
                    continue
                kept_captured_at.append(captured_at[index])
                kept_time_from_start_s.append(rebased_offset_s)
                kept_values.append(values[index])
                captured_dt = _parse_iso8601_utc(captured_at[index])
                if captured_dt is not None and (
                    first_captured_at is None or captured_dt < first_captured_at
                ):
                    first_captured_at = captured_dt

            selected_metrics[metric_key] = {
                **copy.deepcopy(metric_payload),
                "captured_at": kept_captured_at,
                "time_from_start_s": kept_time_from_start_s,
                "value": kept_values,
            }

        selected_payload = copy.deepcopy(source_payload)
        selected_payload["first_captured_at"] = _isoformat_utc(first_captured_at)
        selected_payload["metrics"] = selected_metrics
        selected_payload["service_failure_detected"] = bool(
            source_payload.get("service_failure_detected", False)
        )
        selected_payload["service_failure_cutoff_time_utc"] = source_payload.get(
            "service_failure_cutoff_time_utc"
        )
        return selected_payload

    def build_vllm_payloads(self, base_dir: str) -> dict[str, dict[str, Any]]:
        timeseries_payload = self._select_vllm_timeseries_payload(base_dir)
        summary_helper = _load_helper_module(
            "vllm_metrics_summarize",
            "post-process/vllm-metrics/summarize_timeseries.py",
        )
        stats_payload = summary_helper.summarize_timeseries_payload(
            timeseries_payload,
            source_timeseries_path=self._output_path(f"{base_dir}/gauge-counter-timeseries.json"),
        )
        stats_payload["service_failure_detected"] = bool(
            timeseries_payload.get("service_failure_detected", False)
        )
        stats_payload["service_failure_cutoff_time_utc"] = timeseries_payload.get(
            "service_failure_cutoff_time_utc"
        )
        return {
            f"{base_dir}/gauge-counter-timeseries.json": timeseries_payload,
            f"{base_dir}/gauge-counter-timeseries.stats.json": stats_payload,
        }

    def build_power_summary_payload(self) -> dict[str, Any]:
        if self._selected_power_payload is not None:
            return self._selected_power_payload

        source_payload = self._load_source_json("power/power-summary.json")
        if not isinstance(source_payload, dict):
            raise ValueError("power/power-summary.json must be a JSON object")

        raw_points = source_payload.get("power_points")
        power_points_raw: list[tuple[float, float]] = []
        if isinstance(raw_points, list):
            for point in raw_points:
                if not isinstance(point, dict):
                    continue
                time_offset_s = _float_or_none(point.get("time_offset_s"))
                power_w = _float_or_none(point.get("power_w"))
                if time_offset_s is None or power_w is None:
                    continue
                power_points_raw.append((time_offset_s, power_w))
        power_points_raw.sort(key=lambda pair: pair[0])

        selected_points: list[tuple[float, float]] = []
        boundary_power = _interpolate_power(power_points_raw, self.window.cutoff_offset_s)
        if boundary_power is not None:
            selected_points.append((0.0, round(boundary_power, 6)))
        for time_offset_s, power_w in power_points_raw:
            rebased_offset_s = _rebase_point_offset(time_offset_s, window=self.window)
            if rebased_offset_s is None:
                continue
            selected_points.append((rebased_offset_s, round(power_w, 6)))

        deduped_points: list[tuple[float, float]] = []
        for time_offset_s, power_w in selected_points:
            if deduped_points and deduped_points[-1][0] == time_offset_s:
                deduped_points[-1] = (time_offset_s, power_w)
            else:
                deduped_points.append((time_offset_s, power_w))

        power_values = [power_w for _, power_w in deduped_points]
        avg_power_w = round(sum(power_values) / len(power_values), 6) if power_values else None
        min_power_w = round(min(power_values), 6) if power_values else None
        max_power_w = round(max(power_values), 6) if power_values else None

        total_energy_j = 0.0
        if len(deduped_points) >= 2:
            previous_time_s, previous_power_w = deduped_points[0]
            for time_s, power_w in deduped_points[1:]:
                delta_s = time_s - previous_time_s
                if delta_s > 0:
                    total_energy_j += ((previous_power_w + power_w) / 2.0) * delta_s
                previous_time_s = time_s
                previous_power_w = power_w
        total_energy_j = round(total_energy_j, 6)
        total_energy_kwh = round(total_energy_j / 3_600_000.0, 12)

        self._selected_power_payload = {
            "source_run_dir": source_payload.get("source_run_dir"),
            "source_type": source_payload.get("source_type"),
            "source_power_log_path": source_payload.get("source_power_log_path"),
            "experiment_started_at": self.window.selected_started_at,
            "experiment_finished_at": self.window.selected_finished_at,
            "time_constraint_s": self.window.selected_duration_s,
            "analysis_window_start_utc": self.window.selected_started_at,
            "analysis_window_end_utc": self.window.selected_finished_at,
            "service_failure_detected": bool(
                source_payload.get("service_failure_detected", False)
            ),
            "service_failure_cutoff_time_utc": source_payload.get("service_failure_cutoff_time_utc"),
            "power_log_found": bool(source_payload.get("power_log_found", False)),
            "power_sample_count": len(deduped_points),
            "power_stats_w": {
                "avg": avg_power_w,
                "min": min_power_w,
                "max": max_power_w,
            },
            "total_energy_j": total_energy_j,
            "total_energy_kwh": total_energy_kwh,
            "power_points": [
                {
                    "time_offset_s": _round_s(time_offset_s),
                    "power_w": power_w,
                }
                for time_offset_s, power_w in deduped_points
            ],
        }
        return self._selected_power_payload

    def build_power_sampling_payload(self) -> dict[str, Any]:
        source_payload = self._load_source_json("power-sampling/power-sampling-summary.json")
        if not isinstance(source_payload, dict):
            raise ValueError("power-sampling/power-sampling-summary.json must be a JSON object")

        power_sampling_helper = _load_helper_module(
            "power_sampling_extract",
            "post-process/power-sampling/extract_run.py",
        )
        power_summary_payload = self.build_power_summary_payload()
        prefill_payload = self.build_prefill_payloads()[
            "prefill-concurrency/prefill-concurrency-timeseries.json"
        ]
        power_points = power_sampling_helper._extract_power_points(power_summary_payload)
        prefill_points = power_sampling_helper._extract_prefill_points(prefill_payload)
        sample_times_s = [time_offset_s for time_offset_s, _ in prefill_points]
        sampled_power_w = power_sampling_helper._sample_power_for_times(
            power_points,
            sample_times_s,
        )

        power_by_concurrency: dict[int, list[float]] = {}
        for index, power_w in enumerate(sampled_power_w):
            concurrency = prefill_points[index][1]
            power_by_concurrency.setdefault(concurrency, []).append(power_w)

        concurrency_power_stats_w: dict[str, dict[str, Any]] = {}
        for concurrency in sorted(power_by_concurrency):
            concurrency_power_stats_w[str(concurrency)] = {
                "concurrency": concurrency,
                **power_sampling_helper._build_power_stats(power_by_concurrency[concurrency]),
            }

        non_zero_values = [
            power_w
            for concurrency, values in power_by_concurrency.items()
            if concurrency != 0
            for power_w in values
        ]

        return {
            "source_run_dir": source_payload.get("source_run_dir"),
            "source_power_summary_path": self._derived_power_summary_path(),
            "source_prefill_concurrency_timeseries_path": self._derived_prefill_timeseries_path(),
            "source_type": power_summary_payload.get("source_type"),
            "service_failure_detected": bool(
                power_summary_payload.get("service_failure_detected", False)
            ),
            "service_failure_cutoff_time_utc": power_summary_payload.get(
                "service_failure_cutoff_time_utc"
            ),
            "power_log_found": bool(power_summary_payload.get("power_log_found", False)),
            "request_count": _int_or_none(prefill_payload.get("request_count")),
            "prefill_activity_count": _int_or_none(prefill_payload.get("prefill_activity_count")),
            "total_duration_s": _float_or_none(prefill_payload.get("total_duration_s")),
            "tick_ms": _int_or_none(prefill_payload.get("tick_ms")),
            "tick_s": _float_or_none(prefill_payload.get("tick_s")),
            "prefill_tick_count": len(prefill_points),
            "power_point_count": len(power_points),
            "sampled_tick_count": len(sampled_power_w),
            "all_power_stats_w": power_sampling_helper._build_power_stats(sampled_power_w),
            "non_zero_power_stats_w": power_sampling_helper._build_power_stats(non_zero_values),
            "concurrency_power_stats_w": concurrency_power_stats_w,
            "sampling_method": {
                "interpolation": "linear",
                "outside_power_range": "clamp_to_nearest_endpoint",
            },
        }

    def _clip_range_entries(
        self,
        *,
        source_payload: dict[str, Any],
        sort_key: Any,
        input_request_count: int | None,
    ) -> dict[str, Any]:
        raw_entries = source_payload.get("entries")
        selected_entries: list[dict[str, Any]] = []
        if isinstance(raw_entries, list):
            for entry in raw_entries:
                if not isinstance(entry, dict):
                    continue
                range_start_s = _float_or_none(entry.get("range_start_s"))
                range_end_s = _float_or_none(entry.get("range_end_s"))
                if range_start_s is None or range_end_s is None:
                    continue
                clipped = _clip_range_to_window(
                    start_s=range_start_s,
                    end_s=range_end_s,
                    window=self.window,
                )
                if clipped is None:
                    continue
                _clipped_start_abs, _clipped_end_abs, rebased_start_s, rebased_end_s = clipped
                updated = copy.deepcopy(entry)
                updated["range_start_s"] = rebased_start_s
                updated["range_end_s"] = rebased_end_s
                range_duration_s = _round_s(rebased_end_s - rebased_start_s)
                updated["range_duration_s"] = range_duration_s
                avg_value_per_s = _float_or_none(updated.get("avg_value_per_s"))
                if avg_value_per_s is not None:
                    updated["total_value"] = _round_value(avg_value_per_s * range_duration_s)
                selected_entries.append(updated)
        selected_entries.sort(key=sort_key)

        payload = copy.deepcopy(source_payload)
        payload["source_llm_requests_path"] = self._derived_llm_requests_path()
        if input_request_count is not None:
            payload["input_request_count"] = input_request_count
        payload["entry_count"] = len(selected_entries)
        payload["entries"] = selected_entries
        return payload

    def build_stack_payloads(self, base_dir: str) -> dict[str, dict[str, Any]]:
        if base_dir == "gateway/stack":
            helper = _load_helper_module(
                "gateway_stack_extract",
                "post-process/gateway/stack/extract_run.py",
            )
            sort_key = lambda entry: (
                _float_or_none(entry.get("range_start_s"))
                if _float_or_none(entry.get("range_start_s")) is not None
                else float("inf"),
                str(entry.get("request_id") or ""),
            )
        elif base_dir == "gateway/stack-context":
            helper = _load_helper_module(
                "gateway_stack_context_extract",
                "post-process/gateway/stack-context/extract_run.py",
            )
            sort_key = lambda entry: (
                _float_or_none(entry.get("range_start_s"))
                if _float_or_none(entry.get("range_start_s")) is not None
                else float("inf"),
                str(entry.get("agent_key") or ""),
                str(entry.get("request_id") or ""),
            )
        elif base_dir == "gateway/stack-kv":
            helper = _load_helper_module(
                "gateway_stack_kv_extract",
                "post-process/gateway/stack-kv/extract_run.py",
            )
            sort_key = lambda entry: (
                _float_or_none(entry.get("range_start_s"))
                if _float_or_none(entry.get("range_start_s")) is not None
                else float("inf"),
                str(entry.get("request_id") or ""),
            )
        else:
            raise ValueError(f"Unsupported stack base dir: {base_dir}")

        selected_requests_count = (
            len(self._selected_request_records())
            if self._has_source_json("gateway/llm-requests/llm-requests.json")
            else None
        )

        payloads: dict[str, dict[str, Any]] = {}
        for relative_path in sorted(self.source_json_paths):
            if not relative_path.startswith(f"{base_dir}/"):
                continue
            filename = Path(relative_path).name
            if filename.endswith("-ranges.json"):
                source_payload = self._load_source_json(relative_path)
                if not isinstance(source_payload, dict):
                    raise ValueError(f"{relative_path} must be a JSON object")
                selected_ranges = self._clip_range_entries(
                    source_payload=source_payload,
                    sort_key=sort_key,
                    input_request_count=selected_requests_count,
                )
                payloads[relative_path] = selected_ranges

                histogram_filename = filename.replace("-ranges.json", "-stacked-histogram.json")
                histogram_relative_path = str(Path(relative_path).with_name(histogram_filename))
                if histogram_relative_path in self.source_json_paths:
                    histogram_source_payload = self._load_source_json(histogram_relative_path)
                    if not isinstance(histogram_source_payload, dict):
                        raise ValueError(f"{histogram_relative_path} must be a JSON object")
                    points = helper.build_stacked_histogram(selected_ranges.get("entries", []))
                    payloads[histogram_relative_path] = {
                        "source_run_dir": histogram_source_payload.get("source_run_dir"),
                        "source_gateway_output_dir": histogram_source_payload.get(
                            "source_gateway_output_dir"
                        ),
                        "source_llm_requests_path": self._derived_llm_requests_path(),
                        "service_failure_detected": bool(
                            histogram_source_payload.get("service_failure_detected", False)
                        ),
                        "service_failure_cutoff_time_utc": histogram_source_payload.get(
                            "service_failure_cutoff_time_utc"
                        ),
                        "input_request_count": selected_requests_count,
                        "metric": histogram_source_payload.get("metric"),
                        "bucket_width_s": histogram_source_payload.get("bucket_width_s", 1),
                        "point_count": len(points),
                        "points": points,
                    }
            elif filename.endswith("-stacked-histogram.json"):
                continue
        return payloads

    def build_service_failure_payload(self) -> dict[str, Any]:
        source_payload = self._load_source_json("service-failure/service-failure.json")
        if not isinstance(source_payload, dict):
            raise ValueError("service-failure/service-failure.json must be a JSON object")
        return copy.deepcopy(source_payload)

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self._has_source_json("service-failure/service-failure.json"):
            self._write_output(
                "service-failure/service-failure.json",
                self.build_service_failure_payload(),
            )

        self._write_output(
            "global/trial-timing-summary.json",
            self.build_selected_global_payload(),
        )

        if self._has_source_json("global-progress/replay-progress-summary.json"):
            self._write_output(
                "global-progress/replay-progress-summary.json",
                self.build_global_progress_payload(),
            )

        if self._has_source_json("job-throughput/job-throughput-timeseries.json"):
            self._write_output(
                "job-throughput/job-throughput-timeseries.json",
                self.build_job_throughput_payload(),
            )

        if self._has_source_json("job-concurrency/job-concurrency-timeseries.json"):
            self._write_output(
                "job-concurrency/job-concurrency-timeseries.json",
                self.build_job_concurrency_payload(),
            )

        if self._has_source_json("gateway/llm-requests/llm-requests.json"):
            llm_payloads = self.build_llm_request_related_payloads()
            for relative_path, payload in sorted(llm_payloads.items()):
                if relative_path in self.source_json_paths:
                    self._write_output(relative_path, payload)

        if self._has_source_json(
            "agent-output-throughput/agent-output-throughput.json"
        ) and self._has_source_json("gateway/llm-requests/llm-requests.json"):
            self._write_output(
                "agent-output-throughput/agent-output-throughput.json",
                self.build_agent_output_throughput_payload(),
            )

        if self._has_source_json("gateway/usage/usage-summary.json") and self._has_source_json(
            "gateway/llm-requests/llm-requests.json"
        ):
            self._write_output(
                "gateway/usage/usage-summary.json",
                self.build_usage_summary_payload(),
            )

        if (
            any(path.startswith("prefill-concurrency/") for path in self.source_json_paths)
            and self._has_source_json("gateway/llm-requests/llm-requests.json")
        ):
            for relative_path, payload in sorted(self.build_prefill_payloads().items()):
                if relative_path in self.source_json_paths:
                    self._write_output(relative_path, payload)

        if self._has_source_json("split/duration/duration-split-summary.json") and self._has_source_json(
            "gateway/llm-requests/llm-requests.json"
        ):
            self._write_output(
                "split/duration/duration-split-summary.json",
                self.build_split_duration_payload(),
            )

        for base_dir in ("vllm-log", "vllm-metrics"):
            if self._has_source_json(f"{base_dir}/gauge-counter-timeseries.json"):
                for relative_path, payload in sorted(self.build_vllm_payloads(base_dir).items()):
                    if relative_path in self.source_json_paths:
                        self._write_output(relative_path, payload)

        if self._has_source_json("power/power-summary.json"):
            self._write_output(
                "power/power-summary.json",
                self.build_power_summary_payload(),
            )

        if (
            self._has_source_json("power-sampling/power-sampling-summary.json")
            and self._has_source_json("power/power-summary.json")
            and self._has_source_json("prefill-concurrency/prefill-concurrency-timeseries.json")
            and self._has_source_json("gateway/llm-requests/llm-requests.json")
        ):
            self._write_output(
                "power-sampling/power-sampling-summary.json",
                self.build_power_sampling_payload(),
            )

        for base_dir in ("gateway/stack", "gateway/stack-context", "gateway/stack-kv"):
            has_stack_files = any(path.startswith(f"{base_dir}/") for path in self.source_json_paths)
            if has_stack_files and self._has_source_json("gateway/llm-requests/llm-requests.json"):
                for relative_path, payload in sorted(self.build_stack_payloads(base_dir).items()):
                    if relative_path in self.source_json_paths:
                        self._write_output(relative_path, payload)

        self.generate_visualization_outputs()

        skipped_json_paths = sorted(
            relative_path
            for relative_path in self.source_json_paths
            if relative_path not in self.written_json_paths
            and relative_path != DEFAULT_SELECTION_SUMMARY_NAME
        )
        self.skipped_json_paths = skipped_json_paths
        skipped_non_json_paths = sorted(
            relative_path
            for relative_path in self.source_non_json_paths
            if relative_path not in self.written_non_json_paths
        )
        self.skipped_non_json_paths = skipped_non_json_paths

        summary_payload = {
            "source_run_dir": str(self.source_run_dir),
            "source_post_processed_dir": str(self.source_post_processed_dir),
            "output_dir": str(self.output_dir),
            "selected_percent": self.percent,
            "window": {
                "original_started_at": self.window.original_started_at,
                "original_finished_at": self.window.original_finished_at,
                "selected_started_at": self.window.selected_started_at,
                "selected_finished_at": self.window.selected_finished_at,
                "original_duration_s": self.window.original_duration_s,
                "cutoff_offset_s": self.window.cutoff_offset_s,
                "selected_duration_s": self.window.selected_duration_s,
            },
            "written_json_files": sorted(self.written_json_paths),
            "written_non_json_files": sorted(self.written_non_json_paths),
            "generated_visualization_manifests": sorted(
                set(self.generated_visualization_manifests)
            ),
            "skipped_visualizations": self.skipped_visualizations,
            "skipped_json_files": skipped_json_paths,
            "skipped_non_json_files": skipped_non_json_paths,
        }
        self._write_output(DEFAULT_SELECTION_SUMMARY_NAME, summary_payload)
        return summary_payload


def select_post_processed(
    *,
    source_post_processed_dir: Path,
    percent: float,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    selector = Selector(
        source_post_processed_dir=source_post_processed_dir,
        percent=percent,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    return selector.run()


def select_post_processed_from_run_dir(
    *,
    run_dir: Path,
    percent: float,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    return select_post_processed(
        source_post_processed_dir=_default_source_post_processed_dir(resolved_run_dir),
        percent=percent,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def _select_run_dir_worker(
    task: tuple[str, float, bool],
) -> tuple[str, str | None, str | None]:
    run_dir_text, percent, overwrite = task
    run_dir = Path(run_dir_text).expanduser().resolve()
    try:
        summary_payload = select_post_processed_from_run_dir(
            run_dir=run_dir,
            percent=percent,
            overwrite=overwrite,
        )
    except Exception as exc:
        return (str(run_dir), None, str(exc))
    return (str(run_dir), str(summary_payload["output_dir"]), None)


def _resolve_source_dir(args: argparse.Namespace) -> Path:
    if args.post_processed_dir is not None:
        return Path(args.post_processed_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    return _default_source_post_processed_dir(run_dir)


def _run_root_dir_sequential(
    run_dirs: list[Path],
    *,
    percent: float,
    overwrite: bool,
) -> int:
    failure_count = 0
    for run_dir in run_dirs:
        try:
            summary_payload = select_post_processed_from_run_dir(
                run_dir=run_dir,
                percent=percent,
                overwrite=overwrite,
            )
            print(f"[done] {run_dir} -> {summary_payload['output_dir']}")
        except Exception as exc:
            failure_count += 1
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)
    return failure_count


def _run_root_dir_parallel(
    run_dirs: list[Path],
    *,
    percent: float,
    overwrite: bool,
    max_procs: int,
) -> int:
    failure_count = 0
    tasks = [(str(run_dir), percent, overwrite) for run_dir in run_dirs]
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        for run_dir_text, output_dir_text, error_text in executor.map(
            _select_run_dir_worker,
            tasks,
        ):
            if error_text is None:
                print(f"[done] {run_dir_text} -> {output_dir_text}")
            else:
                failure_count += 1
                print(f"[error] {run_dir_text}: {error_text}", file=sys.stderr)
    return failure_count


def _main_single_target(args: argparse.Namespace) -> int:
    if args.dry_run:
        raise ValueError("--dry-run can only be used with --root-dir")

    source_post_processed_dir = _resolve_source_dir(args)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else None
    )
    summary_payload = select_post_processed(
        source_post_processed_dir=source_post_processed_dir,
        percent=args.percent,
        output_dir=output_dir,
        overwrite=bool(args.overwrite),
    )
    print(summary_payload["output_dir"])
    return 0


def _main_root_dir(args: argparse.Namespace) -> int:
    if args.output_dir is not None:
        raise ValueError("--output-dir can only be used with --run-dir or --post-processed-dir")
    if args.max_procs <= 0:
        raise ValueError(f"--max-procs must be a positive integer: {args.max_procs}")

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    run_dirs = discover_run_dirs_with_post_processed(root_dir)
    print(f"Discovered {len(run_dirs)} run directories under {root_dir}")
    if not run_dirs:
        return 0
    if args.dry_run:
        for run_dir in run_dirs:
            print(str(run_dir))
        return 0

    worker_count = min(args.max_procs, len(run_dirs))
    print(f"Running selection with {worker_count} worker process(es)")
    if worker_count <= 1:
        failure_count = _run_root_dir_sequential(
            run_dirs,
            percent=args.percent,
            overwrite=bool(args.overwrite),
        )
    else:
        try:
            failure_count = _run_root_dir_parallel(
                run_dirs,
                percent=args.percent,
                overwrite=bool(args.overwrite),
                max_procs=worker_count,
            )
        except (PermissionError, OSError) as exc:
            print(
                f"[warn] Unable to start process pool ({exc}); falling back to sequential.",
                file=sys.stderr,
            )
            failure_count = _run_root_dir_sequential(
                run_dirs,
                percent=args.percent,
                overwrite=bool(args.overwrite),
            )

    if failure_count:
        print(
            f"Completed with {failure_count} failure(s) out of {len(run_dirs)} run directories.",
            file=sys.stderr,
        )
        return 1
    print(f"Completed selection for {len(run_dirs)} run directories.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.root_dir is not None:
        return _main_root_dir(args)
    return _main_single_target(args)


if __name__ == "__main__":
    raise SystemExit(main())
