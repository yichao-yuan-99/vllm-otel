from __future__ import annotations
from contextlib import redirect_stderr
from contextlib import redirect_stdout
import importlib.util
import io
import json
import sys
from pathlib import Path


def load_module() -> object:
    module_path = (
        Path(__file__).resolve().parents[1] / "select_post_processed.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "select_post_processed",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["select_post_processed"] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


class _FakeVisualizationModule:
    def __init__(
        self,
        *,
        manifest_name: str,
        figure_file_names: list[str],
        function_name: str,
        input_arg_name: str | None = None,
        stats_arg_name: str | None = None,
    ) -> None:
        self.DEFAULT_MANIFEST_NAME = manifest_name
        self._figure_file_names = list(figure_file_names)
        self._function_name = function_name
        self._input_arg_name = input_arg_name
        self._stats_arg_name = stats_arg_name
        setattr(self, function_name, self._generate)

    def _generate(
        self,
        run_dir: Path,
        *,
        output_dir: Path | None = None,
        image_format: str = "png",
        dpi: int = 220,
        **kwargs: object,
    ) -> Path:
        resolved_run_dir = Path(run_dir).expanduser().resolve()
        if output_dir is None:
            raise AssertionError("fake visualization generator requires output_dir")
        resolved_output_dir = Path(output_dir).expanduser().resolve()
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        figure_paths: list[Path] = []
        figures: list[dict[str, object]] = []
        for figure_file_name in self._figure_file_names:
            figure_path = resolved_output_dir / figure_file_name
            figure_path.write_bytes(b"fake-image")
            figure_paths.append(figure_path)
            figures.append(
                {
                    "figure_file_name": figure_file_name,
                    "figure_path": str(figure_path.resolve()),
                    "figure_generated": True,
                }
            )

        manifest: dict[str, object] = {
            "source_run_dir": str(resolved_run_dir),
            "output_dir": str(resolved_output_dir),
            "image_format": image_format,
            "dpi": dpi,
            "figure_count": len(figure_paths),
            "figures": figures,
            "figure_path": str(figure_paths[0].resolve()) if figure_paths else None,
        }
        if self._input_arg_name is not None and kwargs.get(self._input_arg_name) is not None:
            manifest[f"source_{self._input_arg_name}"] = str(
                Path(kwargs[self._input_arg_name]).expanduser().resolve()
            )
        if self._stats_arg_name is not None and kwargs.get(self._stats_arg_name) is not None:
            manifest[f"source_{self._stats_arg_name}"] = str(
                Path(kwargs[self._stats_arg_name]).expanduser().resolve()
            )

        manifest_path = resolved_output_dir / self.DEFAULT_MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path


def install_fake_visualization_modules(module: object) -> callable:
    original_loader = module._load_helper_module

    fake_modules = {
        "post-process/visualization/job-throughput/generate_all_figures.py": _FakeVisualizationModule(
            manifest_name="figures-manifest.json",
            figure_file_names=["job-throughput.png"],
            function_name="generate_figure_for_run_dir",
            input_arg_name="timeseries_input_path",
        ),
        "post-process/visualization/request-throughput/generate_all_figures.py": _FakeVisualizationModule(
            manifest_name="figures-manifest.json",
            figure_file_names=[
                "request-throughput.png",
                "request-throughput-status-200.png",
            ],
            function_name="generate_figure_for_run_dir",
            input_arg_name="timeseries_input_path",
        ),
        "post-process/visualization/agent-output-throughput/generate_all_figures.py": _FakeVisualizationModule(
            manifest_name="figures-manifest.json",
            figure_file_names=[
                "agent-output-throughput-histogram.png",
                "agent-output-throughput-vs-output-tokens.png",
            ],
            function_name="generate_figures_for_run_dir",
            input_arg_name="agent_output_input_path",
        ),
        "post-process/visualization/job-concurrency/generate_all_figures.py": _FakeVisualizationModule(
            manifest_name="figures-manifest.json",
            figure_file_names=["job-concurrency.png"],
            function_name="generate_figure_for_run_dir",
            input_arg_name="timeseries_input_path",
        ),
        "post-process/visualization/prefill-concurrency/generate_all_figures.py": _FakeVisualizationModule(
            manifest_name="figures-manifest.json",
            figure_file_names=["prefill-concurrency.png"],
            function_name="generate_figure_for_run_dir",
            input_arg_name="timeseries_input_path",
        ),
        "post-process/visualization/power/generate_all_figures.py": _FakeVisualizationModule(
            manifest_name="figures-manifest.json",
            figure_file_names=["gpu-power-over-time.png"],
            function_name="generate_figure_for_run_dir",
            input_arg_name="power_input_path",
        ),
        "post-process/visualization/vllm-metrics/generate_all_figures.py": _FakeVisualizationModule(
            manifest_name="figures-manifest.json",
            figure_file_names=["0001-metric-a.png"],
            function_name="generate_figures_for_run_dir",
            input_arg_name="timeseries_input_path",
            stats_arg_name="stats_input_path",
        ),
        "post-process/visualization/gateway-stack/generate_all_figures.py": _build_fake_gateway_stack_module(),
        "post-process/visualization/gateway-stack-context/generate_all_figures.py": _FakeVisualizationModule(
            manifest_name="figures-manifest.json",
            figure_file_names=["context-usage-stacked-histogram.png"],
            function_name="generate_figures_for_run_dir",
            input_arg_name="stack_context_input_dir",
        ),
        "post-process/visualization/gateway-stack-kv/generate_all_figures.py": _FakeVisualizationModule(
            manifest_name="figures-manifest.json",
            figure_file_names=["kv-usage-stacked-histogram.png"],
            function_name="generate_figures_for_run_dir",
            input_arg_name="stack_kv_input_dir",
        ),
        "post-process/visualization/gateway-ctx-aware/generate_all_figures.py": _FakeVisualizationModule(
            manifest_name="figures-manifest.json",
            figure_file_names=["ctx-aware-over-time.png"],
            function_name="generate_figure_for_run_dir",
            input_arg_name="timeseries_input_path",
        ),
        "post-process/visualization/gateway-slo-aware/generate_all_figures.py": _FakeVisualizationModule(
            manifest_name="figures-manifest.json",
            figure_file_names=[
                "slo-aware-over-time.png",
                "slo-aware-stored-throughput.png",
            ],
            function_name="generate_figure_for_run_dir",
            input_arg_name="slo_aware_input_path",
        ),
        "post-process/visualization/freq-control/generate_all_figures.py": _FakeVisualizationModule(
            manifest_name="figures-manifest.json",
            figure_file_names=["freq-control-over-time.png"],
            function_name="generate_figure_for_run_dir",
            input_arg_name="freq_control_input_path",
        ),
        "post-process/visualization/slo-decision/generate_all_figures.py": _FakeVisualizationModule(
            manifest_name="figures-manifest.json",
            figure_file_names=["slo-decision-over-time.png"],
            function_name="generate_figure_for_run_dir",
            input_arg_name="slo_decision_input_path",
        ),
        "post-process/visualization/stacked-per-agent/generate_all_figures.py": _build_fake_stacked_per_agent_module(),
    }

    def fake_loader(cache_key: str, relative_path: str) -> object:
        if relative_path in fake_modules:
            return fake_modules[relative_path]
        return original_loader(cache_key, relative_path)

    module._load_helper_module = fake_loader

    def restore() -> None:
        module._load_helper_module = original_loader

    return restore


def _build_fake_gateway_stack_module() -> object:
    module = _FakeVisualizationModule(
        manifest_name="figures-manifest.json",
        figure_file_names=["prompt-tokens-stacked-histogram.png"],
        function_name="generate_figures_for_run_dir",
        input_arg_name="stack_input_dir",
    )
    module.METRIC_SPECS = [
        {"input_name": "prompt-tokens-stacked-histogram.json"},
        {"input_name": "cached-tokens-stacked-histogram.json"},
        {"input_name": "compute-prompt-tokens-stacked-histogram.json"},
        {"input_name": "completion-tokens-stacked-histogram.json"},
        {"input_name": "compute-prompt-plus-completion-tokens-stacked-histogram.json"},
    ]
    return module


def _build_fake_stacked_per_agent_module() -> object:
    class _FakeStackedPerAgentModule:
        DEFAULT_MANIFEST_NAME = "figures-manifest.json"

        def generate_figures_for_run_dir(
            self,
            run_dir: Path,
            *,
            output_dir: Path | None = None,
            image_format: str = "png",
            dpi: int = 220,
            **kwargs: object,
        ) -> Path:
            resolved_run_dir = Path(run_dir).expanduser().resolve()
            if output_dir is None:
                raise AssertionError("fake stacked-per-agent generator requires output_dir")
            resolved_output_dir = Path(output_dir).expanduser().resolve()
            resolved_output_dir.mkdir(parents=True, exist_ok=True)

            materialized_path = (
                resolved_output_dir / "stacked-per-agent.window-120s.start-0.end-full.json"
            )
            materialized_path.write_text(
                json.dumps(
                    {
                        "source_run_dir": str(resolved_run_dir),
                        "window_size_s": 120.0,
                        "analysis_window_start_s": 0.0,
                        "analysis_window_end_s": None,
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )

            figure_path = (
                resolved_output_dir / "stacked-per-agent.window-120s.start-0.end-full.png"
            )
            figure_path.write_bytes(b"fake-image")

            manifest_path = resolved_output_dir / self.DEFAULT_MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(
                    {
                        "source_run_dir": str(resolved_run_dir),
                        "output_dir": str(resolved_output_dir),
                        "image_format": image_format,
                        "dpi": dpi,
                        "figure_generated": True,
                        "figure_file_name": figure_path.name,
                        "figure_path": str(figure_path.resolve()),
                        "materialized_file_name": materialized_path.name,
                        "materialized_data_path": str(materialized_path.resolve()),
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return manifest_path

    return _FakeStackedPerAgentModule()


def build_sample_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    post_processed_dir = run_dir / "post-processed"

    write_json(
        post_processed_dir / "service-failure" / "service-failure.json",
        {
            "detector_version": 3,
            "source_run_dir": str(run_dir),
            "source_sbatch_logs_dir": str(run_dir / "sbatch-logs"),
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
            "detected_at_utc": "2026-01-01T00:10:00Z",
        },
    )

    write_json(
        post_processed_dir / "global" / "trial-timing-summary.json",
        {
            "source_run_dir": str(run_dir),
            "source_type": "replay",
            "source_path": str(run_dir / "replay" / "summary.json"),
            "experiment_started_at": "2026-01-01T00:00:00Z",
            "experiment_finished_at": "2026-01-01T00:00:10Z",
            "total_duration_s": 10.0,
            "trial_count": 3,
            "trail_count": 3,
            "trial_duration_stats_s": {"avg": 2.333333, "min": 1.0, "max": 3.0},
            "trials": [
                {
                    "trial_id": "trial-a",
                    "status": "completed",
                    "started_at": "2026-01-01T00:00:01Z",
                    "finished_at": "2026-01-01T00:00:03Z",
                    "start_offset_s": 1.0,
                    "end_offset_s": 3.0,
                    "duration_s": 2.0,
                },
                {
                    "trial_id": "trial-b",
                    "status": "completed",
                    "started_at": "2026-01-01T00:00:04Z",
                    "finished_at": "2026-01-01T00:00:06Z",
                    "start_offset_s": 4.0,
                    "end_offset_s": 6.0,
                    "duration_s": 2.0,
                },
                {
                    "trial_id": "trial-c",
                    "status": "completed",
                    "started_at": "2026-01-01T00:00:07Z",
                    "finished_at": "2026-01-01T00:00:10Z",
                    "start_offset_s": 7.0,
                    "end_offset_s": 10.0,
                    "duration_s": 3.0,
                },
            ],
            "trails": [
                {
                    "trial_id": "trial-a",
                    "status": "completed",
                    "started_at": "2026-01-01T00:00:01Z",
                    "finished_at": "2026-01-01T00:00:03Z",
                    "start_offset_s": 1.0,
                    "end_offset_s": 3.0,
                    "duration_s": 2.0,
                },
                {
                    "trial_id": "trial-b",
                    "status": "completed",
                    "started_at": "2026-01-01T00:00:04Z",
                    "finished_at": "2026-01-01T00:00:06Z",
                    "start_offset_s": 4.0,
                    "end_offset_s": 6.0,
                    "duration_s": 2.0,
                },
                {
                    "trial_id": "trial-c",
                    "status": "completed",
                    "started_at": "2026-01-01T00:00:07Z",
                    "finished_at": "2026-01-01T00:00:10Z",
                    "start_offset_s": 7.0,
                    "end_offset_s": 10.0,
                    "duration_s": 3.0,
                },
            ],
        },
    )

    write_json(
        post_processed_dir / "global-progress" / "replay-progress-summary.json",
        {
            "source_run_dir": str(run_dir),
            "source_type": "replay",
            "experiment_started_at": "2026-01-01T00:00:00Z",
            "replay_count": 3,
            "finished_replay_count": 3,
            "milestone_step": 1,
            "milestones": [
                {"replay_count": 1, "finish_time_s": 3.0},
                {"replay_count": 2, "finish_time_s": 6.0},
                {"replay_count": 3, "finish_time_s": 10.0},
            ],
        },
    )

    write_json(
        post_processed_dir / "job-throughput" / "job-throughput-timeseries.json",
        {
            "source_run_dir": str(run_dir),
            "source_type": "replay",
            "experiment_started_at": "2026-01-01T00:00:00Z",
            "experiment_finished_at": "2026-01-01T00:00:10Z",
            "time_constraint_s": 10.0,
            "replay_count": 3,
            "finished_replay_count": 3,
            "finished_replay_count_excluding_cancelled": 3,
            "cancelled_finished_replay_count": 0,
            "total_duration_s": 10.0,
            "timepoint_frequency_hz": 1.0,
            "timepoint_interval_s": 1.0,
            "window_size_s": 1.0,
            "window_width_s": 2.0,
            "sample_count": 10,
            "throughput_points": [],
            "throughput_points_excluding_cancelled": [],
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "series_by_profile": {
                "profile-0": {
                    "gateway_profile_id": 0,
                    "replay_count": 3,
                    "finished_replay_count": 3,
                    "finished_replay_count_excluding_cancelled": 3,
                    "cancelled_finished_replay_count": 0,
                    "sample_count": 10,
                    "throughput_points": [],
                    "throughput_points_excluding_cancelled": [],
                }
            },
        },
    )

    write_json(
        post_processed_dir / "job-concurrency" / "job-concurrency-timeseries.json",
        {
            "source_run_dir": str(run_dir),
            "source_type": "replay",
            "experiment_started_at": "2026-01-01T00:00:00Z",
            "experiment_finished_at": "2026-01-01T00:00:10Z",
            "time_constraint_s": 10.0,
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "replay_count": 3,
            "jobs_with_valid_range_count": 3,
            "total_duration_s": 10.0,
            "sample_count": 10,
            "max_concurrency": 1,
            "avg_concurrency": 0.7,
            "concurrency_points": [],
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "series_by_profile": {
                "profile-0": {
                    "gateway_profile_id": 0,
                    "replay_count": 3,
                    "jobs_with_valid_range_count": 3,
                    "sample_count": 10,
                    "max_concurrency": 1,
                    "avg_concurrency": 0.7,
                    "concurrency_points": [],
                }
            },
        },
    )

    write_json(
        post_processed_dir / "agent-output-throughput" / "agent-output-throughput.json",
        {
            "source_run_dir": str(run_dir),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "agent_count": 3,
            "request_count": 3,
            "requests_with_output_tokens": 3,
            "requests_with_llm_request_duration": 3,
            "requests_with_output_tokens_and_llm_request_duration": 3,
            "output_tokens": 45,
            "completion_tokens": 45,
            "llm_request_duration_s": 5.0,
            "output_throughput_tokens_per_s": 9.0,
            "agent_output_throughput_tokens_per_s_summary": {
                "sample_count": 3,
                "avg": 9.166667,
                "min": 7.5,
                "max": 10.0,
                "std": 1.178511,
            },
            "agent_output_throughput_tokens_per_s_histogram": {
                "metric": "output_throughput_tokens_per_s",
                "bin_size": 1.0,
                "sample_count": 3,
                "bin_count": 4,
                "min": 7.5,
                "max": 10.0,
                "bins": [
                    {"bin_start": 7.0, "bin_end": 8.0, "count": 1},
                    {"bin_start": 8.0, "bin_end": 9.0, "count": 0},
                    {"bin_start": 9.0, "bin_end": 10.0, "count": 0},
                    {"bin_start": 10.0, "bin_end": 11.0, "count": 2},
                ],
            },
            "agents": [
                {
                    "gateway_run_id": "run-a",
                    "gateway_profile_id": 0,
                    "api_token_hash": "hash-a",
                    "replay_worker_status": "completed",
                    "replay_completed": True,
                    "request_count": 1,
                    "requests_with_output_tokens": 1,
                    "requests_with_llm_request_duration": 1,
                    "requests_with_output_tokens_and_llm_request_duration": 1,
                    "output_tokens": 10,
                    "completion_tokens": 10,
                    "llm_request_duration_s": 1.0,
                    "output_throughput_tokens_per_s": 10.0,
                },
                {
                    "gateway_run_id": "run-b",
                    "gateway_profile_id": 0,
                    "api_token_hash": "hash-b",
                    "replay_worker_status": "completed",
                    "replay_completed": True,
                    "request_count": 1,
                    "requests_with_output_tokens": 1,
                    "requests_with_llm_request_duration": 1,
                    "requests_with_output_tokens_and_llm_request_duration": 1,
                    "output_tokens": 20,
                    "completion_tokens": 20,
                    "llm_request_duration_s": 2.0,
                    "output_throughput_tokens_per_s": 10.0,
                },
                {
                    "gateway_run_id": "run-c",
                    "gateway_profile_id": 0,
                    "api_token_hash": "hash-c",
                    "replay_worker_status": "failed",
                    "replay_completed": False,
                    "request_count": 1,
                    "requests_with_output_tokens": 1,
                    "requests_with_llm_request_duration": 1,
                    "requests_with_output_tokens_and_llm_request_duration": 1,
                    "output_tokens": 15,
                    "completion_tokens": 15,
                    "llm_request_duration_s": 2.0,
                    "output_throughput_tokens_per_s": 7.5,
                },
            ],
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "series_by_profile": {
                "profile-0": {
                    "gateway_profile_id": 0,
                    "agent_count": 3,
                    "agents": [],
                }
            },
        },
    )

    requests_payload = {
        "source_run_dir": str(run_dir),
        "source_gateway_output_dir": str(run_dir / "gateway-output"),
        "service_failure_detected": False,
        "service_failure_cutoff_time_utc": None,
        "request_count": 3,
        "requests": [
            {
                "gateway_run_id": "run-a",
                "gateway_profile_id": 0,
                "api_token_hash": "hash-a",
                "trace_id": "trace-a",
                "request_id": "req-a",
                "request_start_time": "2026-01-01T00:00:01Z",
                "request_end_time": "2026-01-01T00:00:02Z",
                "request_start_offset_s": 1.0,
                "request_end_offset_s": 2.0,
                "request_end_to_run_end_s": 8.0,
                "request_duration_ms": 1000.0,
                "duration_ms": 1000.0,
                "status_code": 200,
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "cached_tokens": 40,
                "gen_ai.latency.time_in_queue": 0.1,
                "gen_ai.latency.time_in_model_prefill": 0.2,
                "gen_ai.latency.time_in_model_decode": 0.7,
                "gen_ai.latency.e2e": 0.9,
                "gen_ai.latency.time_to_first_token": 0.3,
            },
            {
                "gateway_run_id": "run-b",
                "gateway_profile_id": 0,
                "api_token_hash": "hash-b",
                "trace_id": "trace-b",
                "request_id": "req-b",
                "request_start_time": "2026-01-01T00:00:04Z",
                "request_end_time": "2026-01-01T00:00:06Z",
                "request_start_offset_s": 4.0,
                "request_end_offset_s": 6.0,
                "request_end_to_run_end_s": 4.0,
                "request_duration_ms": 2000.0,
                "duration_ms": 2000.0,
                "status_code": 200,
                "prompt_tokens": 120,
                "completion_tokens": 20,
                "total_tokens": 140,
                "cached_tokens": 30,
                "gen_ai.latency.time_in_queue": 0.1,
                "gen_ai.latency.time_in_model_prefill": 0.2,
                "gen_ai.latency.time_in_model_decode": 1.5,
                "gen_ai.latency.e2e": 1.9,
                "gen_ai.latency.time_to_first_token": 0.3,
            },
            {
                "gateway_run_id": "run-c",
                "gateway_profile_id": 0,
                "api_token_hash": "hash-c",
                "trace_id": "trace-c",
                "request_id": "req-c",
                "request_start_time": "2026-01-01T00:00:07Z",
                "request_end_time": "2026-01-01T00:00:09Z",
                "request_start_offset_s": 7.0,
                "request_end_offset_s": 9.0,
                "request_end_to_run_end_s": 1.0,
                "request_duration_ms": 2000.0,
                "duration_ms": 2000.0,
                "status_code": 499,
                "prompt_tokens": 90,
                "completion_tokens": 15,
                "total_tokens": 105,
                "cached_tokens": 20,
                "gen_ai.latency.time_in_queue": 0.2,
                "gen_ai.latency.time_in_model_prefill": 0.4,
                "gen_ai.latency.time_in_model_decode": 1.1,
                "gen_ai.latency.e2e": 1.8,
                "gen_ai.latency.time_to_first_token": 0.5,
            },
        ],
        "multi_profile": False,
        "port_profile_ids": [0],
    }
    write_json(
        post_processed_dir / "gateway" / "llm-requests" / "llm-requests.json",
        requests_payload,
    )
    write_json(
        post_processed_dir / "gateway" / "llm-requests" / "llm-request-stats.json",
        {"placeholder": True},
    )
    write_json(
        post_processed_dir / "gateway" / "llm-requests" / "llm-request-speed-stats.json",
        {"placeholder": True},
    )
    write_json(
        post_processed_dir / "gateway" / "llm-requests" / "llm-requests-longest-10.json",
        {"placeholder": True},
    )
    write_json(
        post_processed_dir / "gateway" / "llm-requests" / "llm-requests-shortest-10.json",
        {"placeholder": True},
    )
    write_json(
        post_processed_dir / "gateway" / "llm-requests" / "llm-requests-stats.200.json",
        {"placeholder": True},
    )
    write_json(
        post_processed_dir / "gateway" / "llm-requests" / "llm-requests-stats.499.json",
        {"placeholder": True},
    )
    write_json(
        post_processed_dir / "request-throughput" / "request-throughput-timeseries.json",
        {
            "source_run_dir": str(run_dir),
            "source_llm_requests_path": str(
                post_processed_dir / "gateway" / "llm-requests" / "llm-requests.json"
            ),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "total_duration_s": 10.0,
            "timepoint_frequency_hz": 1.0,
            "timepoint_interval_s": 1.0,
            "window_size_s": 1.0,
            "window_width_s": 2.0,
            "request_count": 3,
            "finished_request_count": 3,
            "finished_request_count_status_200": 2,
            "non_200_finished_request_count": 1,
            "sample_count": 10,
            "throughput_points": [],
            "throughput_points_status_200": [],
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "series_by_profile": {
                "profile-0": {
                    "gateway_profile_id": 0,
                    "request_count": 3,
                    "finished_request_count": 3,
                    "finished_request_count_status_200": 2,
                    "non_200_finished_request_count": 1,
                    "sample_count": 10,
                    "throughput_points": [],
                    "throughput_points_status_200": [],
                }
            },
        },
    )

    write_json(
        post_processed_dir / "gateway" / "usage" / "usage-summary.json",
        {
            "source_run_dir": str(run_dir),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "agent_count": 3,
            "request_count": 3,
            "usage": {},
            "agents": [
                {
                    "gateway_run_id": "run-a",
                    "gateway_profile_id": 0,
                    "api_token_hash": "hash-a",
                    "request_count": 1,
                    "usage": {},
                },
                {
                    "gateway_run_id": "run-b",
                    "gateway_profile_id": 0,
                    "api_token_hash": "hash-b",
                    "request_count": 1,
                    "usage": {},
                },
                {
                    "gateway_run_id": "run-c",
                    "gateway_profile_id": 0,
                    "api_token_hash": "hash-c",
                    "request_count": 1,
                    "usage": {},
                },
            ],
        },
    )

    write_json(
        post_processed_dir / "prefill-concurrency" / "prefill-activities.json",
        {
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "activities": [],
            "activities_by_profile": {},
        },
    )
    write_json(
        post_processed_dir / "prefill-concurrency" / "prefill-concurrency-timeseries.json",
        {
            "tick_ms": 10,
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "series_by_profile": {},
        },
    )
    write_json(
        post_processed_dir / "prefill-concurrency" / "prefill-concurrency-stats.json",
        {
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "series_by_profile": {},
        },
    )

    write_json(
        post_processed_dir / "split" / "duration" / "duration-split-summary.json",
        {
            "source_run_dir": str(run_dir),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "split_count": 2,
            "bin_labels": ["0-50%", "50-100%"],
            "metrics": [
                "duration_s",
                "turn_count",
                "prompt_tokens",
                "decode_tokens",
                "cached_prompt_tokens",
            ],
            "tables": {},
            "jobs": [],
            "excluded_jobs_no_token_usage": [],
        },
    )

    write_json(
        post_processed_dir / "vllm-log" / "gauge-counter-timeseries.json",
        {
            "source_run_dir": str(run_dir),
            "source_vllm_log_dir": str(run_dir / "vllm-log"),
            "cluster_mode": False,
            "port_profile_ids": [],
            "first_captured_at": "2026-01-01T00:00:00Z",
            "metric_count": 1,
            "metrics": {
                "metric-a": {
                    "name": "metric-a",
                    "sample_name": "metric-a",
                    "family": "metric-a",
                    "type": "gauge",
                    "help": "test",
                    "labels": {},
                    "captured_at": [
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T00:00:04Z",
                        "2026-01-01T00:00:05Z",
                        "2026-01-01T00:00:07Z",
                        "2026-01-01T00:00:09Z",
                    ],
                    "time_from_start_s": [0.0, 4.0, 5.0, 7.0, 9.0],
                    "value": [0.0, 4.0, 5.0, 7.0, 9.0],
                }
            },
        },
    )
    write_json(
        post_processed_dir / "vllm-log" / "gauge-counter-timeseries.stats.json",
        {"placeholder": True},
    )

    write_json(
        post_processed_dir / "power" / "power-summary.json",
        {
            "source_run_dir": str(run_dir),
            "source_type": "replay",
            "source_power_log_path": str(run_dir / "power" / "power-log.jsonl"),
            "experiment_started_at": "2026-01-01T00:00:00Z",
            "experiment_finished_at": "2026-01-01T00:00:10Z",
            "time_constraint_s": 10.0,
            "analysis_window_start_utc": "2026-01-01T00:00:00Z",
            "analysis_window_end_utc": "2026-01-01T00:00:10Z",
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "power_log_found": True,
            "power_sample_count": 4,
            "power_stats_w": {"avg": 140.0, "min": 100.0, "max": 180.0},
            "total_energy_j": 0.0,
            "total_energy_kwh": 0.0,
            "power_points": [
                {"time_offset_s": 4.0, "power_w": 100.0},
                {"time_offset_s": 5.0, "power_w": 120.0},
                {"time_offset_s": 7.0, "power_w": 160.0},
                {"time_offset_s": 9.0, "power_w": 180.0},
            ],
        },
    )

    write_json(
        post_processed_dir / "power-sampling" / "power-sampling-summary.json",
        {
            "source_run_dir": str(run_dir),
            "source_power_summary_path": str(
                post_processed_dir / "power" / "power-summary.json"
            ),
            "source_prefill_concurrency_timeseries_path": str(
                post_processed_dir
                / "prefill-concurrency"
                / "prefill-concurrency-timeseries.json"
            ),
            "source_type": "replay",
        },
    )

    write_json(
        post_processed_dir / "gateway" / "stack" / "prompt-tokens-ranges.json",
        {
            "source_run_dir": str(run_dir),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "source_llm_requests_path": str(
                post_processed_dir / "gateway" / "llm-requests" / "llm-requests.json"
            ),
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "input_request_count": 3,
            "metric": "prompt_tokens",
            "phase": "prefill",
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "entry_count": 2,
            "entries": [
                {
                    "gateway_run_id": "run-b",
                    "gateway_profile_id": 0,
                    "request_id": "req-b",
                    "trace_id": "trace-b",
                    "metric": "prompt_tokens",
                    "phase": "prefill",
                    "range_start_s": 4.5,
                    "range_end_s": 5.5,
                    "range_duration_s": 1.0,
                    "total_value": 100.0,
                    "avg_value_per_s": 100.0,
                },
                {
                    "gateway_run_id": "run-c",
                    "gateway_profile_id": 0,
                    "request_id": "req-c",
                    "trace_id": "trace-c",
                    "metric": "prompt_tokens",
                    "phase": "prefill",
                    "range_start_s": 7.0,
                    "range_end_s": 8.0,
                    "range_duration_s": 1.0,
                    "total_value": 20.0,
                    "avg_value_per_s": 20.0,
                },
            ],
            "entries_by_profile": {},
        },
    )
    write_json(
        post_processed_dir / "gateway" / "stack" / "prompt-tokens-stacked-histogram.json",
        {
            "source_run_dir": str(run_dir),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "source_llm_requests_path": str(
                post_processed_dir / "gateway" / "llm-requests" / "llm-requests.json"
            ),
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "input_request_count": 3,
            "metric": "prompt_tokens",
            "phase": "prefill",
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "bucket_width_s": 1,
            "point_count": 0,
            "points": [],
            "series_by_profile": {},
        },
    )
    write_json(
        post_processed_dir / "gateway" / "stack-context" / "context-usage-ranges.json",
        {
            "source_run_dir": str(run_dir),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "source_llm_requests_path": str(
                post_processed_dir / "gateway" / "llm-requests" / "llm-requests.json"
            ),
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "input_request_count": 3,
            "metric": "context_usage_tokens",
            "phase": "context",
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "entry_count": 2,
            "entries": [
                {
                    "gateway_run_id": "run-b",
                    "gateway_profile_id": 0,
                    "agent_key": "hash-b",
                    "request_id": "req-b",
                    "range_start_s": 4.5,
                    "range_end_s": 5.5,
                    "range_duration_s": 1.0,
                    "total_value": 300.0,
                    "avg_value_per_s": 300.0,
                },
                {
                    "gateway_run_id": "run-c",
                    "gateway_profile_id": 0,
                    "agent_key": "hash-c",
                    "request_id": "req-c",
                    "range_start_s": 7.0,
                    "range_end_s": 8.0,
                    "range_duration_s": 1.0,
                    "total_value": 180.0,
                    "avg_value_per_s": 180.0,
                },
            ],
            "entries_by_profile": {},
        },
    )
    write_json(
        post_processed_dir / "gateway" / "stack-context" / "context-usage-stacked-histogram.json",
        {
            "source_run_dir": str(run_dir),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "source_llm_requests_path": str(
                post_processed_dir / "gateway" / "llm-requests" / "llm-requests.json"
            ),
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "input_request_count": 3,
            "metric": "context_usage_tokens",
            "phase": "context",
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "bucket_width_s": 1,
            "point_count": 0,
            "points": [],
            "series_by_profile": {},
        },
    )
    write_json(
        post_processed_dir / "gateway" / "stack-kv" / "kv-usage-ranges.json",
        {
            "source_run_dir": str(run_dir),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "source_llm_requests_path": str(
                post_processed_dir / "gateway" / "llm-requests" / "llm-requests.json"
            ),
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "input_request_count": 3,
            "metric": "kv_usage_tokens",
            "phase": "request_lifetime",
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "entry_count": 2,
            "entries": [
                {
                    "gateway_run_id": "run-b",
                    "gateway_profile_id": 0,
                    "request_id": "req-b",
                    "range_start_s": 4.0,
                    "range_end_s": 6.0,
                    "range_duration_s": 2.0,
                    "total_value": 200.0,
                    "avg_value_per_s": 100.0,
                },
                {
                    "gateway_run_id": "run-c",
                    "gateway_profile_id": 0,
                    "request_id": "req-c",
                    "range_start_s": 7.0,
                    "range_end_s": 9.0,
                    "range_duration_s": 2.0,
                    "total_value": 160.0,
                    "avg_value_per_s": 80.0,
                },
            ],
            "entries_by_profile": {},
        },
    )
    write_json(
        post_processed_dir / "gateway" / "stack-kv" / "kv-usage-stacked-histogram.json",
        {
            "source_run_dir": str(run_dir),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "source_llm_requests_path": str(
                post_processed_dir / "gateway" / "llm-requests" / "llm-requests.json"
            ),
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "input_request_count": 3,
            "metric": "kv_usage_tokens",
            "phase": "request_lifetime",
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "bucket_width_s": 1,
            "point_count": 0,
            "points": [],
            "series_by_profile": {},
        },
    )
    write_json(
        post_processed_dir / "gateway" / "ctx-aware-log" / "ctx-aware-timeseries.json",
        {
            "source_run_dir": str(run_dir),
            "source_ctx_aware_log_path": str(run_dir / "gateway-output" / "job" / "ctx_aware_x.jsonl"),
            "selected_ctx_aware_log_file_name": "ctx_aware_x.jsonl",
            "ctx_aware_log_candidate_count": 1,
            "ctx_aware_log_candidates": [
                str(run_dir / "gateway-output" / "job" / "ctx_aware_x.jsonl")
            ],
            "started_at": "2026-01-01T00:00:04Z",
            "ended_at": "2026-01-01T00:00:08Z",
            "sample_count": 3,
            "duration_s": 4.0,
            "avg_sample_interval_s": 2.0,
            "metric_summaries": {},
            "samples": [
                {
                    "timestamp": "2026-01-01T00:00:04Z",
                    "second": 0.0,
                    "ongoing_agent_count": 1,
                    "pending_agent_count": 0,
                    "ongoing_effective_context_tokens": 10,
                    "pending_effective_context_tokens": 0,
                    "agents_turned_pending_due_to_context_threshold": 0,
                    "agents_turned_ongoing": 1,
                    "new_agents_added_as_pending": 0,
                    "new_agents_added_as_ongoing": 1,
                },
                {
                    "timestamp": "2026-01-01T00:00:06Z",
                    "second": 2.0,
                    "ongoing_agent_count": 1,
                    "pending_agent_count": 1,
                    "ongoing_effective_context_tokens": 20,
                    "pending_effective_context_tokens": 5,
                    "agents_turned_pending_due_to_context_threshold": 1,
                    "agents_turned_ongoing": 0,
                    "new_agents_added_as_pending": 0,
                    "new_agents_added_as_ongoing": 0,
                },
                {
                    "timestamp": "2026-01-01T00:00:08Z",
                    "second": 4.0,
                    "ongoing_agent_count": 2,
                    "pending_agent_count": 0,
                    "ongoing_effective_context_tokens": 30,
                    "pending_effective_context_tokens": 0,
                    "agents_turned_pending_due_to_context_threshold": 0,
                    "agents_turned_ongoing": 1,
                    "new_agents_added_as_pending": 0,
                    "new_agents_added_as_ongoing": 1,
                },
            ],
        },
    )
    write_json(
        post_processed_dir / "gateway" / "slo-aware-log" / "slo-aware-events.json",
        {
            "source_run_dir": str(run_dir),
            "source_type": "replay",
            "source_slo_aware_log_paths": [str(run_dir / "gateway-output" / "job" / "slo_aware.jsonl")],
            "experiment_started_at": "2026-01-01T00:00:00Z",
            "experiment_finished_at": "2026-01-01T00:00:10Z",
            "time_constraint_s": 10.0,
            "analysis_window_start_utc": "2026-01-01T00:00:00Z",
            "analysis_window_end_utc": "2026-01-01T00:00:10Z",
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "slo_aware_log_found": True,
            "events": [
                {
                    "timestamp_utc": "2026-01-01T00:00:04Z",
                    "time_offset_s": 4.0,
                    "event_type": "enter",
                    "api_token_hash": "hash-b",
                    "trace_id": "trace-b",
                    "wake_reason": None,
                    "resume_disposition": None,
                    "output_tokens_per_s": 5.0,
                    "slo_slack_s": 1.0,
                    "slo_target_tokens_per_s": 8.0,
                    "min_output_tokens_per_s": 4.0,
                    "avg_output_tokens_per_s": 5.0,
                    "ralexation_duration_s": 0.5,
                    "ralexation_until_utc": "2026-01-01T00:00:04.500000Z",
                },
                {
                    "timestamp_utc": "2026-01-01T00:00:07Z",
                    "time_offset_s": 7.0,
                    "event_type": "wake",
                    "api_token_hash": "hash-c",
                    "trace_id": "trace-c",
                    "wake_reason": "slack_recovered",
                    "resume_disposition": "resume",
                    "output_tokens_per_s": 7.5,
                    "slo_slack_s": 2.0,
                    "slo_target_tokens_per_s": 8.0,
                    "min_output_tokens_per_s": 6.0,
                    "avg_output_tokens_per_s": 7.0,
                    "ralexation_duration_s": 0.2,
                    "ralexation_until_utc": "2026-01-01T00:00:07.200000Z",
                },
            ],
        },
    )
    write_json(
        post_processed_dir / "freq-control" / "freq-control-summary.json",
        {
            "source_run_dir": str(run_dir),
            "source_type": "replay",
            "source_freq_control_log_dir_name": "freq-control",
            "source_query_log_paths": [str(run_dir / "freq-control" / "freq-controller.query.jsonl")],
            "source_decision_log_paths": [str(run_dir / "freq-control" / "freq-controller.decision.jsonl")],
            "source_control_error_log_paths": [str(run_dir / "freq-control" / "freq-controller.control-error.jsonl")],
            "experiment_started_at": "2026-01-01T00:00:00Z",
            "experiment_finished_at": "2026-01-01T00:00:10Z",
            "time_constraint_s": 10.0,
            "analysis_window_start_utc": "2026-01-01T00:00:00Z",
            "analysis_window_end_utc": "2026-01-01T00:00:10Z",
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "freq_control_log_found": True,
            "query_log_found": True,
            "decision_log_found": True,
            "control_error_log_found": True,
            "multi_profile": False,
            "port_profile_ids": [0],
            "series_keys": ["profile-0"],
            "linespace_policy_detected": False,
            "segmented_policy_detected": False,
            "query_points": [
                {
                    "timestamp_utc": "2026-01-01T00:00:04Z",
                    "time_offset_s": 4.0,
                    "phase": "active",
                    "job_active": True,
                    "context_usage": 0.4,
                    "error": None,
                    "port_profile_id": 0,
                },
                {
                    "timestamp_utc": "2026-01-01T00:00:08Z",
                    "time_offset_s": 8.0,
                    "phase": "pending",
                    "job_active": False,
                    "context_usage": 0.8,
                    "error": "slow",
                    "port_profile_id": 0,
                },
            ],
            "decision_points": [
                {
                    "timestamp_utc": "2026-01-01T00:00:06Z",
                    "time_offset_s": 6.0,
                    "changed": True,
                    "lower_bound": 0.2,
                    "upper_bound": 0.9,
                    "target_context_usage_threshold": 0.7,
                    "segment_count": 4,
                    "segment_width_context_usage": 0.1,
                    "low_freq_threshold": None,
                    "low_freq_cap_mhz": None,
                    "effective_min_frequency_mhz": 1000,
                    "window_context_usage": 0.65,
                    "current_frequency_mhz": 1200,
                    "target_frequency_mhz": 1300,
                    "port_profile_id": 0,
                }
            ],
            "control_error_points": [
                {
                    "timestamp_utc": "2026-01-01T00:00:07Z",
                    "time_offset_s": 7.0,
                    "port_profile_id": 0,
                }
            ],
        },
    )
    write_json(
        post_processed_dir / "slo-decision" / "slo-decision-summary.json",
        {
            "source_run_dir": str(run_dir),
            "source_type": "replay",
            "source_slo_decision_log_dir_name": "freq-control-linespace",
            "source_slo_decision_log_paths": [
                str(run_dir / "freq-control-linespace" / "freq-controller-ls.slo-decision.jsonl")
            ],
            "experiment_started_at": "2026-01-01T00:00:00Z",
            "experiment_finished_at": "2026-01-01T00:00:10Z",
            "time_constraint_s": 10.0,
            "analysis_window_start_utc": "2026-01-01T00:00:00Z",
            "analysis_window_end_utc": "2026-01-01T00:00:10Z",
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "slo_decision_log_found": True,
            "decision_points": [
                {
                    "timestamp_utc": "2026-01-01T00:00:04Z",
                    "time_offset_s": 4.0,
                    "changed": False,
                    "current_frequency_mhz": 1200,
                    "target_frequency_mhz": 1200,
                    "window_min_output_tokens_per_s": 6.0,
                    "target_output_throughput_tokens_per_s": 8.0,
                },
                {
                    "timestamp_utc": "2026-01-01T00:00:07Z",
                    "time_offset_s": 7.0,
                    "changed": True,
                    "current_frequency_mhz": 1200,
                    "target_frequency_mhz": 1300,
                    "window_min_output_tokens_per_s": 7.0,
                    "target_output_throughput_tokens_per_s": 8.0,
                },
            ],
        },
    )

    figures_dir = post_processed_dir / "visualization" / "job-throughput"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "job-throughput.png").write_bytes(b"png")

    return run_dir


def build_sample_root(tmp_path: Path) -> tuple[Path, list[Path]]:
    root_dir = tmp_path / "root"
    run_a = build_sample_run(root_dir / "suite-a")
    run_b = build_sample_run(root_dir / "suite-b" / "nested")

    ignored_summary = (
        root_dir
        / "ignored"
        / "run-x"
        / "post-processed-50"
        / "global"
        / "trial-timing-summary.json"
    )
    write_json(
        ignored_summary,
        {
            "experiment_started_at": "2026-01-01T00:00:00Z",
            "total_duration_s": 10.0,
        },
    )

    return root_dir, [run_a, run_b]


def test_select_post_processed_rewrites_supported_outputs(tmp_path: Path) -> None:
    module = load_module()
    restore = install_fake_visualization_modules(module)
    run_dir = build_sample_run(tmp_path)

    try:
        summary = module.select_post_processed(
            source_post_processed_dir=run_dir / "post-processed",
            percent=50,
        )
    finally:
        restore()

    output_dir = run_dir / "post-processed-50"
    assert summary["output_dir"] == str(output_dir)
    assert output_dir.is_dir()

    global_payload = read_json(output_dir / "global" / "trial-timing-summary.json")
    assert global_payload["experiment_started_at"] == "2026-01-01T00:00:05Z"
    assert global_payload["total_duration_s"] == 5.0
    assert [trial["trial_id"] for trial in global_payload["trials"]] == ["trial-b", "trial-c"]
    assert global_payload["trials"][0]["start_offset_s"] == 0.0
    assert global_payload["trials"][0]["end_offset_s"] == 1.0
    assert global_payload["trials"][1]["start_offset_s"] == 2.0
    assert global_payload["trials"][1]["end_offset_s"] == 5.0

    progress_payload = read_json(
        output_dir / "global-progress" / "replay-progress-summary.json"
    )
    assert progress_payload["replay_count"] == 2
    assert progress_payload["milestones"] == [
        {"replay_count": 1, "finish_time_s": 1.0},
        {"replay_count": 2, "finish_time_s": 5.0},
    ]

    throughput_payload = read_json(
        output_dir / "job-throughput" / "job-throughput-timeseries.json"
    )
    assert throughput_payload["finished_replay_count"] == 2
    assert throughput_payload["sample_count"] == 5
    assert throughput_payload["total_duration_s"] == 5.0
    assert throughput_payload["multi_profile"] is False
    assert throughput_payload["port_profile_ids"] == [0]
    assert throughput_payload["series_keys"] == ["profile-0"]
    assert throughput_payload["series_by_profile"]["profile-0"]["gateway_profile_id"] == 0
    assert throughput_payload["series_by_profile"]["profile-0"]["sample_count"] == 5

    concurrency_payload = read_json(
        output_dir / "job-concurrency" / "job-concurrency-timeseries.json"
    )
    assert concurrency_payload["sample_count"] == 5
    assert concurrency_payload["max_concurrency"] == 1
    assert concurrency_payload["multi_profile"] is False
    assert concurrency_payload["port_profile_ids"] == [0]
    assert concurrency_payload["series_keys"] == ["profile-0"]
    assert concurrency_payload["series_by_profile"]["profile-0"]["gateway_profile_id"] == 0
    assert concurrency_payload["concurrency_points"][:3] == [
        {"second": 0, "concurrency": 1},
        {"second": 1, "concurrency": 0},
        {"second": 2, "concurrency": 1},
    ]

    agent_output_payload = read_json(
        output_dir / "agent-output-throughput" / "agent-output-throughput.json"
    )
    assert agent_output_payload["agent_count"] == 2
    assert agent_output_payload["request_count"] == 2
    assert agent_output_payload["output_tokens"] == 35
    assert agent_output_payload["completion_tokens"] == 35
    assert agent_output_payload["llm_request_duration_s"] == 3.0
    assert agent_output_payload["output_throughput_tokens_per_s"] == 11.666667
    assert [agent["gateway_run_id"] for agent in agent_output_payload["agents"]] == [
        "run-b",
        "run-c",
    ]
    assert [agent["api_token_hash"] for agent in agent_output_payload["agents"]] == [
        "hash-b",
        "hash-c",
    ]
    assert [agent["gateway_profile_id"] for agent in agent_output_payload["agents"]] == [0, 0]
    assert [agent["replay_worker_status"] for agent in agent_output_payload["agents"]] == [
        "completed",
        "failed",
    ]
    assert [agent["replay_completed"] for agent in agent_output_payload["agents"]] == [
        True,
        False,
    ]
    assert agent_output_payload["multi_profile"] is False
    assert agent_output_payload["port_profile_ids"] == [0]
    assert agent_output_payload["series_keys"] == ["profile-0"]
    assert agent_output_payload["series_by_profile"]["profile-0"]["gateway_profile_id"] == 0
    assert agent_output_payload["series_by_profile"]["profile-0"]["agent_count"] == 2
    assert [
        agent["output_throughput_tokens_per_s"]
        for agent in agent_output_payload["agents"]
    ] == [20.0, 7.5]
    histogram_payload = agent_output_payload[
        "agent_output_throughput_tokens_per_s_histogram"
    ]
    assert histogram_payload["sample_count"] == 2
    assert histogram_payload["bin_size"] == 1.0
    assert histogram_payload["bins"][0] == {
        "bin_start": 7.0,
        "bin_end": 8.0,
        "count": 1,
    }
    assert histogram_payload["bins"][-1] == {
        "bin_start": 20.0,
        "bin_end": 21.0,
        "count": 1,
    }

    requests_payload = read_json(
        output_dir / "gateway" / "llm-requests" / "llm-requests.json"
    )
    assert requests_payload["request_count"] == 2
    assert [request["request_id"] for request in requests_payload["requests"]] == [
        "req-b",
        "req-c",
    ]
    assert requests_payload["requests"][0]["request_start_offset_s"] == 0.0
    assert requests_payload["requests"][0]["request_end_offset_s"] == 1.0
    assert requests_payload["requests"][0]["request_duration_ms"] == 1000.0
    assert requests_payload["requests"][1]["request_start_offset_s"] == 2.0
    assert requests_payload["requests"][1]["request_end_offset_s"] == 4.0
    assert requests_payload["multi_profile"] is False
    assert requests_payload["port_profile_ids"] == [0]

    request_throughput_payload = read_json(
        output_dir / "request-throughput" / "request-throughput-timeseries.json"
    )
    assert request_throughput_payload["request_count"] == 2
    assert request_throughput_payload["finished_request_count"] == 2
    assert request_throughput_payload["finished_request_count_status_200"] == 1
    assert request_throughput_payload["non_200_finished_request_count"] == 1
    assert request_throughput_payload["sample_count"] == 5
    assert request_throughput_payload["multi_profile"] is False
    assert request_throughput_payload["port_profile_ids"] == [0]
    assert request_throughput_payload["series_keys"] == ["profile-0"]
    assert request_throughput_payload["series_by_profile"]["profile-0"]["gateway_profile_id"] == 0
    assert request_throughput_payload["throughput_points"][0] == {
        "time_s": 0.0,
        "throughput_requests_per_s": 1.0,
    }
    assert request_throughput_payload["throughput_points_status_200"][-1] == {
        "time_s": 4.0,
        "throughput_requests_per_s": 0.0,
    }

    llm_stats_payload = read_json(
        output_dir / "gateway" / "llm-requests" / "llm-request-stats.json"
    )
    assert llm_stats_payload["request_count"] == 2
    assert llm_stats_payload["metrics"]["request_start_offset_s"]["min"] == 0.0
    assert llm_stats_payload["metrics"]["request_end_offset_s"]["max"] == 4.0

    usage_payload = read_json(output_dir / "gateway" / "usage" / "usage-summary.json")
    assert usage_payload["request_count"] == 2
    assert usage_payload["agent_count"] == 2
    assert usage_payload["service_failure_detected"] is False
    assert usage_payload["service_failure_cutoff_time_utc"] is None
    assert usage_payload["usage"]["prompt_tokens"] == 210
    assert usage_payload["usage"]["completion_tokens"] == 35
    assert [agent["api_token_hash"] for agent in usage_payload["agents"]] == [
        "hash-b",
        "hash-c",
    ]

    prefill_activity_payload = read_json(
        output_dir / "prefill-concurrency" / "prefill-activities.json"
    )
    assert prefill_activity_payload["request_count"] == 2
    assert prefill_activity_payload["prefill_activity_count"] == 1
    assert prefill_activity_payload["port_profile_ids"] == [0]
    assert prefill_activity_payload["series_keys"] == ["profile-0"]
    assert prefill_activity_payload["activities_by_profile"]["profile-0"]["gateway_profile_id"] == 0
    assert prefill_activity_payload["activities_by_profile"]["profile-0"]["prefill_activity_count"] == 1
    assert prefill_activity_payload["activities"][0]["request_id"] == "req-c"
    assert prefill_activity_payload["activities"][0]["prefill_start_offset_s"] == 2.2
    assert prefill_activity_payload["activities"][0]["prefill_end_offset_s"] == 2.6

    prefill_timeseries_payload = read_json(
        output_dir / "prefill-concurrency" / "prefill-concurrency-timeseries.json"
    )
    assert prefill_timeseries_payload["sample_count"] == 500
    non_zero_points = [
        point
        for point in prefill_timeseries_payload["concurrency_points"]
        if point["concurrency"] > 0
    ]
    assert non_zero_points[0]["time_offset_s"] == 2.2
    assert non_zero_points[-1]["time_offset_s"] == 2.59
    assert prefill_timeseries_payload["series_by_profile"]["profile-0"]["sample_count"] == 500

    prefill_stats_payload = read_json(
        output_dir / "prefill-concurrency" / "prefill-concurrency-stats.json"
    )
    assert prefill_stats_payload["sample_count"] == 500
    assert prefill_stats_payload["max_concurrency"] == 1
    assert prefill_stats_payload["series_by_profile"]["profile-0"]["gateway_profile_id"] == 0

    split_payload = read_json(output_dir / "split" / "duration" / "duration-split-summary.json")
    assert split_payload["job_count_total"] == 2
    assert split_payload["job_count"] == 2
    assert split_payload["tables"]["prompt_tokens"]["0-50%"]["count"] == 1
    assert split_payload["tables"]["prompt_tokens"]["50-100%"]["count"] == 1

    vllm_timeseries_payload = read_json(
        output_dir / "vllm-log" / "gauge-counter-timeseries.json"
    )
    metric_payload = vllm_timeseries_payload["metrics"]["metric-a"]
    assert metric_payload["time_from_start_s"] == [0.0, 2.0, 4.0]
    assert metric_payload["value"] == [5.0, 7.0, 9.0]

    vllm_stats_payload = read_json(
        output_dir / "vllm-log" / "gauge-counter-timeseries.stats.json"
    )
    assert vllm_stats_payload["source_timeseries_path"] == str(
        (output_dir / "vllm-log" / "gauge-counter-timeseries.json").resolve()
    )
    assert vllm_stats_payload["metrics"]["metric-a"]["sample_count"] == 3
    assert vllm_stats_payload["metrics"]["metric-a"]["avg"] == 7.0

    power_payload = read_json(output_dir / "power" / "power-summary.json")
    assert power_payload["experiment_started_at"] == "2026-01-01T00:00:05Z"
    assert power_payload["analysis_window_start_utc"] == "2026-01-01T00:00:05Z"
    assert power_payload["power_points"][0] == {"time_offset_s": 0.0, "power_w": 120.0}
    assert power_payload["power_points"][-1] == {"time_offset_s": 4.0, "power_w": 180.0}

    power_sampling_payload = read_json(
        output_dir / "power-sampling" / "power-sampling-summary.json"
    )
    assert power_sampling_payload["source_power_summary_path"] == str(
        (output_dir / "power" / "power-summary.json").resolve()
    )
    assert power_sampling_payload["source_prefill_concurrency_timeseries_path"] == str(
        (
            output_dir
            / "prefill-concurrency"
            / "prefill-concurrency-timeseries.json"
        ).resolve()
    )
    assert power_sampling_payload["request_count"] == 2
    assert power_sampling_payload["prefill_activity_count"] == 1
    assert power_sampling_payload["sampled_tick_count"] == 500

    stack_ranges_payload = read_json(
        output_dir / "gateway" / "stack" / "prompt-tokens-ranges.json"
    )
    assert stack_ranges_payload["source_llm_requests_path"] == str(
        (
            output_dir / "gateway" / "llm-requests" / "llm-requests.json"
        ).resolve()
    )
    assert stack_ranges_payload["input_request_count"] == 2
    assert stack_ranges_payload["entry_count"] == 2
    assert stack_ranges_payload["multi_profile"] is False
    assert stack_ranges_payload["port_profile_ids"] == [0]
    assert stack_ranges_payload["series_keys"] == ["profile-0"]
    assert stack_ranges_payload["entries"][0]["range_start_s"] == 0.0
    assert stack_ranges_payload["entries"][0]["range_end_s"] == 0.5
    assert stack_ranges_payload["entries"][0]["total_value"] == 50.0

    stack_hist_payload = read_json(
        output_dir / "gateway" / "stack" / "prompt-tokens-stacked-histogram.json"
    )
    assert stack_hist_payload["point_count"] == 3
    assert stack_hist_payload["points"] == [
        {"second": 0, "accumulated_value": 50.0},
        {"second": 1, "accumulated_value": 0.0},
        {"second": 2, "accumulated_value": 20.0},
    ]

    stack_context_ranges_payload = read_json(
        output_dir / "gateway" / "stack-context" / "context-usage-ranges.json"
    )
    assert stack_context_ranges_payload["entry_count"] == 2
    assert stack_context_ranges_payload["entries_by_profile"]["profile-0"][0][
        "request_id"
    ] == "req-b"
    assert stack_context_ranges_payload["entries_by_profile"]["profile-0"][1][
        "range_start_s"
    ] == 2.0

    stack_context_hist_payload = read_json(
        output_dir / "gateway" / "stack-context" / "context-usage-stacked-histogram.json"
    )
    assert stack_context_hist_payload["point_count"] == 3
    assert stack_context_hist_payload["series_by_profile"]["profile-0"]["gateway_profile_id"] == 0
    assert stack_context_hist_payload["points"] == [
        {"second": 0, "accumulated_value": 150.0},
        {"second": 1, "accumulated_value": 0.0},
        {"second": 2, "accumulated_value": 180.0},
    ]

    stack_kv_ranges_payload = read_json(
        output_dir / "gateway" / "stack-kv" / "kv-usage-ranges.json"
    )
    assert stack_kv_ranges_payload["entry_count"] == 2
    assert stack_kv_ranges_payload["entries_by_profile"]["profile-0"][1]["request_id"] == "req-c"
    assert stack_kv_ranges_payload["entries"][1]["avg_value_per_s"] == 80.0

    stack_kv_hist_payload = read_json(
        output_dir / "gateway" / "stack-kv" / "kv-usage-stacked-histogram.json"
    )
    assert stack_kv_hist_payload["point_count"] == 4
    assert stack_kv_hist_payload["series_by_profile"]["profile-0"]["gateway_profile_id"] == 0
    assert stack_kv_hist_payload["points"] == [
        {"second": 0, "accumulated_value": 100.0},
        {"second": 1, "accumulated_value": 0.0},
        {"second": 2, "accumulated_value": 80.0},
        {"second": 3, "accumulated_value": 80.0},
    ]

    ctx_aware_payload = read_json(
        output_dir / "gateway" / "ctx-aware-log" / "ctx-aware-timeseries.json"
    )
    assert ctx_aware_payload["started_at"] == "2026-01-01T00:00:05Z"
    assert ctx_aware_payload["sample_count"] == 2
    assert ctx_aware_payload["samples"][0]["second"] == 1.0
    assert ctx_aware_payload["samples"][1]["ongoing_agent_count"] == 2

    slo_aware_payload = read_json(
        output_dir / "gateway" / "slo-aware-log" / "slo-aware-events.json"
    )
    assert slo_aware_payload["slo_aware_event_count"] == 1
    assert slo_aware_payload["unique_agent_count"] == 1
    assert slo_aware_payload["events"][0]["api_token_hash"] == "hash-c"
    assert slo_aware_payload["events"][0]["time_offset_s"] == 2.0

    freq_control_payload = read_json(
        output_dir / "freq-control" / "freq-control-summary.json"
    )
    assert freq_control_payload["query_point_count"] == 1
    assert freq_control_payload["decision_point_count"] == 1
    assert freq_control_payload["control_error_point_count"] == 1
    assert freq_control_payload["port_profile_ids"] == [0]
    assert freq_control_payload["query_points"][0]["time_offset_s"] == 3.0
    assert freq_control_payload["decision_points"][0]["target_frequency_mhz"] == 1300

    slo_decision_payload = read_json(
        output_dir / "slo-decision" / "slo-decision-summary.json"
    )
    assert slo_decision_payload["slo_decision_point_count"] == 1
    assert slo_decision_payload["slo_decision_change_count"] == 1
    assert slo_decision_payload["decision_points"][0]["time_offset_s"] == 2.0
    assert slo_decision_payload["decision_points"][0]["target_frequency_mhz"] == 1300

    job_throughput_manifest = read_json(
        output_dir / "visualization" / "job-throughput" / "figures-manifest.json"
    )
    assert job_throughput_manifest["figure_count"] == 1
    assert Path(job_throughput_manifest["figure_path"]).name == "job-throughput.png"
    assert (output_dir / "visualization" / "job-throughput" / "job-throughput.png").is_file()

    agent_output_manifest = read_json(
        output_dir
        / "visualization"
        / "agent-output-throughput"
        / "figures-manifest.json"
    )
    assert agent_output_manifest["figure_count"] == 2
    assert [
        figure["figure_file_name"] for figure in agent_output_manifest["figures"]
    ] == [
        "agent-output-throughput-histogram.png",
        "agent-output-throughput-vs-output-tokens.png",
    ]
    assert (
        output_dir
        / "visualization"
        / "agent-output-throughput"
        / "agent-output-throughput-histogram.png"
    ).is_file()
    assert (
        output_dir
        / "visualization"
        / "agent-output-throughput"
        / "agent-output-throughput-vs-output-tokens.png"
    ).is_file()

    request_throughput_manifest = read_json(
        output_dir / "visualization" / "request-throughput" / "figures-manifest.json"
    )
    assert request_throughput_manifest["figure_count"] == 2
    assert [
        figure["figure_file_name"] for figure in request_throughput_manifest["figures"]
    ] == [
        "request-throughput.png",
        "request-throughput-status-200.png",
    ]

    ctx_aware_manifest = read_json(
        output_dir / "visualization" / "gateway-ctx-aware" / "figures-manifest.json"
    )
    assert ctx_aware_manifest["figure_count"] == 1
    assert Path(ctx_aware_manifest["figure_path"]).name == "ctx-aware-over-time.png"

    slo_aware_manifest = read_json(
        output_dir / "visualization" / "gateway-slo-aware" / "figures-manifest.json"
    )
    assert slo_aware_manifest["figure_count"] == 2
    assert [figure["figure_file_name"] for figure in slo_aware_manifest["figures"]] == [
        "slo-aware-over-time.png",
        "slo-aware-stored-throughput.png",
    ]

    freq_control_manifest = read_json(
        output_dir / "visualization" / "freq-control" / "figures-manifest.json"
    )
    assert freq_control_manifest["figure_count"] == 1
    assert Path(freq_control_manifest["figure_path"]).name == "freq-control-over-time.png"

    slo_decision_manifest = read_json(
        output_dir / "visualization" / "slo-decision" / "figures-manifest.json"
    )
    assert slo_decision_manifest["figure_count"] == 1
    assert Path(slo_decision_manifest["figure_path"]).name == "slo-decision-over-time.png"

    stacked_per_agent_manifest = read_json(
        output_dir / "visualization" / "stacked-per-agent" / "figures-manifest.json"
    )
    assert (
        Path(stacked_per_agent_manifest["materialized_data_path"]).name
        == "stacked-per-agent.window-120s.start-0.end-full.json"
    )
    assert (
        output_dir
        / "visualization"
        / "stacked-per-agent"
        / "stacked-per-agent.window-120s.start-0.end-full.png"
    ).is_file()

    selection_summary = read_json(output_dir / "selection-summary.json")
    assert (
        "visualization/job-throughput/job-throughput.png"
        in selection_summary["written_non_json_files"]
    )
    assert (
        "visualization/job-throughput/job-throughput.png"
        not in selection_summary["skipped_non_json_files"]
    )
    assert (
        "visualization/job-throughput/figures-manifest.json"
        in selection_summary["written_json_files"]
    )
    assert (
        "visualization/job-throughput/figures-manifest.json"
        in selection_summary["generated_visualization_manifests"]
    )
    assert (
        "visualization/agent-output-throughput/agent-output-throughput-histogram.png"
        in selection_summary["written_non_json_files"]
    )
    assert (
        "visualization/agent-output-throughput/agent-output-throughput-vs-output-tokens.png"
        in selection_summary["written_non_json_files"]
    )
    assert (
        "visualization/agent-output-throughput/figures-manifest.json"
        in selection_summary["written_json_files"]
    )
    assert (
        "visualization/agent-output-throughput/figures-manifest.json"
        in selection_summary["generated_visualization_manifests"]
    )
    assert (
        "visualization/request-throughput/request-throughput.png"
        in selection_summary["written_non_json_files"]
    )
    assert (
        "visualization/request-throughput/request-throughput-status-200.png"
        in selection_summary["written_non_json_files"]
    )
    assert (
        "visualization/request-throughput/figures-manifest.json"
        in selection_summary["generated_visualization_manifests"]
    )
    assert (
        "visualization/gateway-ctx-aware/ctx-aware-over-time.png"
        in selection_summary["written_non_json_files"]
    )
    assert (
        "visualization/gateway-ctx-aware/figures-manifest.json"
        in selection_summary["generated_visualization_manifests"]
    )
    assert (
        "visualization/gateway-slo-aware/slo-aware-over-time.png"
        in selection_summary["written_non_json_files"]
    )
    assert (
        "visualization/gateway-slo-aware/slo-aware-stored-throughput.png"
        in selection_summary["written_non_json_files"]
    )
    assert (
        "visualization/gateway-slo-aware/figures-manifest.json"
        in selection_summary["generated_visualization_manifests"]
    )
    assert (
        "visualization/freq-control/freq-control-over-time.png"
        in selection_summary["written_non_json_files"]
    )
    assert (
        "visualization/freq-control/figures-manifest.json"
        in selection_summary["generated_visualization_manifests"]
    )
    assert (
        "visualization/slo-decision/slo-decision-over-time.png"
        in selection_summary["written_non_json_files"]
    )
    assert (
        "visualization/slo-decision/figures-manifest.json"
        in selection_summary["generated_visualization_manifests"]
    )
    assert (
        "visualization/stacked-per-agent/stacked-per-agent.window-120s.start-0.end-full.json"
        in selection_summary["written_json_files"]
    )
    assert (
        "visualization/stacked-per-agent/stacked-per-agent.window-120s.start-0.end-full.png"
        in selection_summary["written_non_json_files"]
    )
    assert (
        "visualization/stacked-per-agent/figures-manifest.json"
        in selection_summary["generated_visualization_manifests"]
    )
    assert selection_summary["skipped_visualizations"] == [
        {
            "name": "gateway-stack",
            "reason": "Selected output is missing one or more required stacked histogram inputs",
        }
    ]
    assert selection_summary["skipped_json_files"] == []
    assert selection_summary["skipped_non_json_files"] == []
    assert (
        "global/trial-timing-summary.json"
        in selection_summary["written_json_files"]
    )


def test_select_post_processed_requires_overwrite_for_existing_output(tmp_path: Path) -> None:
    module = load_module()
    run_dir = build_sample_run(tmp_path)
    output_dir = run_dir / "post-processed-50"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        module.select_post_processed(
            source_post_processed_dir=run_dir / "post-processed",
            percent=50,
        )
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Expected select_post_processed to reject an existing output dir")


def test_discover_run_dirs_with_post_processed_scans_recursively(tmp_path: Path) -> None:
    module = load_module()
    root_dir, run_dirs = build_sample_root(tmp_path)

    discovered = module.discover_run_dirs_with_post_processed(root_dir)

    assert discovered == sorted(run_dirs)


def test_main_root_dir_dry_run_lists_runs_without_writing(tmp_path: Path) -> None:
    module = load_module()
    root_dir, run_dirs = build_sample_root(tmp_path)
    stdout = io.StringIO()

    with redirect_stdout(stdout):
        exit_code = module.main(
            [
                "--root-dir",
                str(root_dir),
                "--percent",
                "50",
                "--dry-run",
                "--max-procs",
                "1",
            ]
        )

    assert exit_code == 0
    output = stdout.getvalue()
    assert f"Discovered 2 run directories under {root_dir.resolve()}" in output
    for run_dir in run_dirs:
        assert str(run_dir.resolve()) in output
        assert not (run_dir / "post-processed-50").exists()


def test_main_root_dir_processes_all_runs_sequentially(tmp_path: Path) -> None:
    module = load_module()
    restore = install_fake_visualization_modules(module)
    root_dir, run_dirs = build_sample_root(tmp_path)
    stdout = io.StringIO()

    try:
        with redirect_stdout(stdout):
            exit_code = module.main(
                [
                    "--root-dir",
                    str(root_dir),
                    "--percent",
                    "50",
                    "--max-procs",
                    "1",
                ]
            )
    finally:
        restore()

    assert exit_code == 0
    output = stdout.getvalue()
    assert "Running selection with 1 worker process(es)" in output
    assert "Completed selection for 2 run directories." in output
    for run_dir in run_dirs:
        output_dir = run_dir / "post-processed-50"
        assert output_dir.is_dir()
        global_payload = read_json(output_dir / "global" / "trial-timing-summary.json")
        assert global_payload["experiment_started_at"] == "2026-01-01T00:00:05Z"
        assert (
            output_dir / "visualization" / "job-throughput" / "figures-manifest.json"
        ).is_file()


def test_main_root_dir_rejects_output_dir_override(tmp_path: Path) -> None:
    module = load_module()
    root_dir, _run_dirs = build_sample_root(tmp_path)

    try:
        module.main(
            [
                "--root-dir",
                str(root_dir),
                "--percent",
                "50",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    except ValueError as exc:
        assert "--output-dir can only be used with --run-dir or --post-processed-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output-dir is used with --root-dir")


def test_main_run_dir_rejects_dry_run(tmp_path: Path) -> None:
    module = load_module()
    run_dir = build_sample_run(tmp_path)
    stderr = io.StringIO()

    try:
        with redirect_stderr(stderr):
            module.main(
                [
                    "--run-dir",
                    str(run_dir),
                    "--percent",
                    "50",
                    "--dry-run",
                ]
            )
    except ValueError as exc:
        assert "--dry-run can only be used with --root-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --dry-run is used with --run-dir")
