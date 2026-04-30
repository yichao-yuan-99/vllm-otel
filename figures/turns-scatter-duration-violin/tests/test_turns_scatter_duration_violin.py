from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
MATERIALIZE_SCRIPT = (
    REPO_ROOT
    / "figures"
    / "turns-scatter-duration-violin"
    / "materialize_turns_scatter_duration_violin.py"
)
PLOT_SCRIPT = (
    REPO_ROOT
    / "figures"
    / "turns-scatter-duration-violin"
    / "plot_turns_scatter_duration_violin.py"
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _write_gateway_job(
    run_dir: Path,
    *,
    gateway_run_id: str,
    gateway_profile_id: int | None,
    request_count: int,
    request_window: tuple[str, str] | None = None,
    agent_window: tuple[str, str] | None = None,
    job_window: tuple[str, str] | None = None,
    manifest_window: tuple[str, str] | None = None,
) -> None:
    gateway_run_dir = (
        run_dir / "gateway-output" / f"profile-{gateway_profile_id}" / gateway_run_id
        if gateway_profile_id is not None
        else run_dir / "gateway-output" / gateway_run_id
    )
    gateway_run_dir.mkdir(parents=True, exist_ok=True)

    if request_window is not None:
        request_start, request_end = request_window
        _write_jsonl(
            gateway_run_dir / "requests" / "model_inference.jsonl",
            [
                {
                    "request_start_time": request_start,
                    "request_end_time": request_end,
                    "response": {
                        "usage": {
                            "prompt_tokens": 10 + index,
                            "completion_tokens": 5 + index,
                        }
                    },
                }
                for index in range(request_count)
            ],
        )

    lifecycle_records: list[dict[str, object]] = []
    if job_window is not None:
        lifecycle_records.append(
            {"event_type": "job_start", "timestamp": job_window[0]}
        )
    if agent_window is not None:
        lifecycle_records.append(
            {"event_type": "agent_start", "timestamp": agent_window[0]}
        )
        lifecycle_records.append(
            {"event_type": "agent_end", "timestamp": agent_window[1]}
        )
    if job_window is not None:
        lifecycle_records.append({"event_type": "job_end", "timestamp": job_window[1]})
    if lifecycle_records:
        _write_jsonl(
            gateway_run_dir / "events" / "lifecycle.jsonl",
            lifecycle_records,
        )

    if manifest_window is not None:
        _write_json(
            gateway_run_dir / "manifest.json",
            {
                "run_start_time": manifest_window[0],
                "run_end_time": manifest_window[1],
            },
        )


def _write_run_summary(
    root_dir: Path,
    *,
    benchmark: str,
    agent_type: str,
    run_dir_name: str,
    score: float,
    jobs: list[dict[str, object]],
    avg_turns_per_run: float | None = None,
) -> None:
    run_dir = root_dir / benchmark / agent_type / run_dir_name
    _write_json(
        run_dir / "run-stats" / "run-stats-summary.json",
        {
            "dataset": benchmark,
            "agent_type": agent_type,
            "score": score,
            "job_count": len(jobs),
            "avg_turns_per_run": avg_turns_per_run,
            "jobs": [
                {
                    "gateway_run_id": str(job["gateway_run_id"]),
                    "gateway_profile_id": job["gateway_profile_id"],
                    "request_count": int(job["request_count"]),
                }
                for job in jobs
            ],
        },
    )

    for job in jobs:
        _write_gateway_job(
            run_dir,
            gateway_run_id=str(job["gateway_run_id"]),
            gateway_profile_id=job["gateway_profile_id"],
            request_count=int(job["request_count"]),
            request_window=job.get("request_window"),
            agent_window=job.get("agent_window"),
            job_window=job.get("job_window"),
            manifest_window=job.get("manifest_window"),
        )


class TurnsScatterDurationViolinFigureTest(unittest.TestCase):
    def test_materialize_turns_scatter_duration_violin_builds_expected_panels(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results" / "qwen3-coder-30b"

            _write_run_summary(
                root_dir,
                benchmark="dabstep",
                agent_type="mini-swe-agent",
                run_dir_name="dabstep-20260306T180000Z",
                score=0.10,
                jobs=[
                    {
                        "gateway_run_id": "run-old-a",
                        "gateway_profile_id": 0,
                        "request_count": 9,
                        "agent_window": (
                            "2026-03-08T00:00:00.000Z",
                            "2026-03-08T00:00:01.000Z",
                        ),
                    }
                ],
            )
            _write_run_summary(
                root_dir,
                benchmark="dabstep",
                agent_type="mini-swe-agent",
                run_dir_name="dabstep-20260306T194929Z",
                score=0.20,
                avg_turns_per_run=14.0 / 3.0,
                jobs=[
                    {
                        "gateway_run_id": "run-agent",
                        "gateway_profile_id": 0,
                        "request_count": 2,
                        "agent_window": (
                            "2026-03-08T00:00:01.000Z",
                            "2026-03-08T00:00:03.500Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:00:00.000Z",
                            "2026-03-08T00:00:10.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:00:02.000Z",
                            "2026-03-08T00:00:07.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-job",
                        "gateway_profile_id": 1,
                        "request_count": 4,
                        "job_window": (
                            "2026-03-08T00:01:00.000Z",
                            "2026-03-08T00:01:09.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:01:01.000Z",
                            "2026-03-08T00:01:05.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-manifest",
                        "gateway_profile_id": 2,
                        "request_count": 8,
                        "manifest_window": (
                            "2026-03-08T00:02:00.000Z",
                            "2026-03-08T00:02:12.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:02:02.000Z",
                            "2026-03-08T00:02:05.000Z",
                        ),
                    },
                ],
            )
            _write_run_summary(
                root_dir,
                benchmark="dabstep",
                agent_type="terminus-2",
                run_dir_name="dabstep-20260306T215045Z",
                score=0.30,
                jobs=[
                    {
                        "gateway_run_id": "run-dab-terminus-a",
                        "gateway_profile_id": 0,
                        "request_count": 3,
                        "request_window": (
                            "2026-03-08T00:10:00.000Z",
                            "2026-03-08T00:10:07.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-dab-terminus-b",
                        "gateway_profile_id": 1,
                        "request_count": 6,
                        "agent_window": (
                            "2026-03-08T00:11:00.000Z",
                            "2026-03-08T00:11:04.000Z",
                        ),
                    },
                ],
            )
            _write_run_summary(
                root_dir,
                benchmark="swebench-verified",
                agent_type="mini-swe-agent",
                run_dir_name="swebench-verified-20260306T062226Z",
                score=0.40,
                jobs=[
                    {
                        "gateway_run_id": "run-swe-mini-a",
                        "gateway_profile_id": 0,
                        "request_count": 5,
                        "agent_window": (
                            "2026-03-08T00:20:00.000Z",
                            "2026-03-08T00:20:05.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-swe-mini-b",
                        "gateway_profile_id": 1,
                        "request_count": 10,
                        "agent_window": (
                            "2026-03-08T00:21:00.000Z",
                            "2026-03-08T00:21:10.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-swe-mini-c",
                        "gateway_profile_id": 2,
                        "request_count": 15,
                        "manifest_window": (
                            "2026-03-08T00:22:00.000Z",
                            "2026-03-08T00:22:20.000Z",
                        ),
                    },
                ],
            )
            _write_run_summary(
                root_dir,
                benchmark="swebench-verified",
                agent_type="terminus-2",
                run_dir_name="swebench-verified-20260306T082357Z",
                score=0.50,
                jobs=[
                    {
                        "gateway_run_id": "run-swe-term-a",
                        "gateway_profile_id": 0,
                        "request_count": 7,
                        "request_window": (
                            "2026-03-08T00:30:00.000Z",
                            "2026-03-08T00:30:14.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-swe-term-b",
                        "gateway_profile_id": 1,
                        "request_count": 14,
                        "request_window": (
                            "2026-03-08T00:31:00.000Z",
                            "2026-03-08T00:31:28.000Z",
                        ),
                    },
                ],
            )
            _write_run_summary(
                root_dir,
                benchmark="terminal-bench-2.0",
                agent_type="mini-swe-agent",
                run_dir_name="terminal-bench@2.0-20260306T163324Z",
                score=0.60,
                jobs=[
                    {
                        "gateway_run_id": "run-term-mini-a",
                        "gateway_profile_id": 0,
                        "request_count": 1,
                        "manifest_window": (
                            "2026-03-08T00:40:00.000Z",
                            "2026-03-08T00:40:50.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-term-mini-b",
                        "gateway_profile_id": 1,
                        "request_count": 9,
                        "manifest_window": (
                            "2026-03-08T00:41:00.000Z",
                            "2026-03-08T00:42:15.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-term-mini-c",
                        "gateway_profile_id": 2,
                        "request_count": 27,
                        "manifest_window": (
                            "2026-03-08T00:43:00.000Z",
                            "2026-03-08T00:44:40.000Z",
                        ),
                    },
                ],
            )
            _write_run_summary(
                root_dir,
                benchmark="terminal-bench-2.0",
                agent_type="terminus-2",
                run_dir_name="terminal-bench@2.0-20260306T174037Z",
                score=0.70,
                jobs=[
                    {
                        "gateway_run_id": "run-term-term-a",
                        "gateway_profile_id": 0,
                        "request_count": 9,
                        "manifest_window": (
                            "2026-03-08T00:50:00.000Z",
                            "2026-03-08T00:51:30.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-term-term-b",
                        "gateway_profile_id": 1,
                        "request_count": 18,
                        "manifest_window": (
                            "2026-03-08T00:52:00.000Z",
                            "2026-03-08T00:55:00.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-term-term-c",
                        "gateway_profile_id": 2,
                        "request_count": 36,
                        "manifest_window": (
                            "2026-03-08T00:56:00.000Z",
                            "2026-03-08T01:02:00.000Z",
                        ),
                    },
                ],
            )

            output_path = tmp_path / "turns-scatter-duration-violin.json"
            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--root-dir",
                    str(root_dir),
                    "--output",
                    str(output_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["figure_name"], "turns-scatter-duration-violin")
            self.assertEqual(payload["figure_orientation"], "horizontal")
            self.assertEqual(
                payload["benchmark_order"],
                ["dabstep", "swebench-verified", "terminal-bench-2.0"],
            )
            self.assertEqual(payload["agent_order"], ["mini-swe-agent", "terminus-2"])
            self.assertEqual(payload["panel_count"], 6)
            self.assertEqual(payload["total_turn_job_count"], 16)
            self.assertEqual(payload["total_duration_job_count"], 16)

            first_panel = payload["panels"][0]
            self.assertEqual(first_panel["benchmark"], "dabstep")
            self.assertEqual(first_panel["agent_type"], "mini-swe-agent")
            self.assertEqual(first_panel["run_dir_name"], "dabstep-20260306T194929Z")
            self.assertEqual(first_panel["candidate_run_count"], 2)
            self.assertEqual(first_panel["turns"], [2, 4, 8])
            self.assertEqual(first_panel["durations_s"], [2.5, 9.0, 12.0])
            self.assertAlmostEqual(first_panel["avg_turns_per_run_reported"], 14.0 / 3.0)
            self.assertAlmostEqual(first_panel["turn_stats"]["mean"], 14.0 / 3.0)
            self.assertAlmostEqual(first_panel["duration_stats"]["median"], 9.0)
            self.assertEqual(first_panel["duration_source_counts"]["lifecycle:agent"], 1)
            self.assertEqual(first_panel["duration_source_counts"]["lifecycle:job"], 1)
            self.assertEqual(first_panel["duration_source_counts"]["manifest:run"], 1)

            last_panel = payload["panels"][-1]
            self.assertEqual(last_panel["benchmark"], "terminal-bench-2.0")
            self.assertEqual(last_panel["agent_label"], "Terminus")
            self.assertEqual(last_panel["turns"], [9, 18, 36])
            self.assertEqual(last_panel["durations_s"], [90.0, 180.0, 360.0])
            self.assertAlmostEqual(last_panel["turn_stats"]["q1"], 13.5)
            self.assertAlmostEqual(last_panel["duration_stats"]["mean"], 210.0)

    def test_plot_turns_scatter_duration_violin_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results" / "qwen3-coder-30b"

            for benchmark, mini_jobs, terminus_jobs in [
                (
                    "dabstep",
                    [
                        {
                            "gateway_run_id": "dab-mini-a",
                            "gateway_profile_id": 0,
                            "request_count": 2,
                            "agent_window": (
                                "2026-03-08T00:00:00.000Z",
                                "2026-03-08T00:00:02.000Z",
                            ),
                        },
                        {
                            "gateway_run_id": "dab-mini-b",
                            "gateway_profile_id": 1,
                            "request_count": 4,
                            "manifest_window": (
                                "2026-03-08T00:01:00.000Z",
                                "2026-03-08T00:01:06.000Z",
                            ),
                        },
                    ],
                    [
                        {
                            "gateway_run_id": "dab-term-a",
                            "gateway_profile_id": 0,
                            "request_count": 3,
                            "request_window": (
                                "2026-03-08T00:10:00.000Z",
                                "2026-03-08T00:10:05.000Z",
                            ),
                        },
                        {
                            "gateway_run_id": "dab-term-b",
                            "gateway_profile_id": 1,
                            "request_count": 6,
                            "agent_window": (
                                "2026-03-08T00:11:00.000Z",
                                "2026-03-08T00:11:08.000Z",
                            ),
                        },
                    ],
                ),
                (
                    "swebench-verified",
                    [
                        {
                            "gateway_run_id": "swe-mini-a",
                            "gateway_profile_id": 0,
                            "request_count": 5,
                            "agent_window": (
                                "2026-03-08T00:20:00.000Z",
                                "2026-03-08T00:20:10.000Z",
                            ),
                        },
                        {
                            "gateway_run_id": "swe-mini-b",
                            "gateway_profile_id": 1,
                            "request_count": 10,
                            "manifest_window": (
                                "2026-03-08T00:21:00.000Z",
                                "2026-03-08T00:21:25.000Z",
                            ),
                        },
                    ],
                    [
                        {
                            "gateway_run_id": "swe-term-a",
                            "gateway_profile_id": 0,
                            "request_count": 7,
                            "request_window": (
                                "2026-03-08T00:30:00.000Z",
                                "2026-03-08T00:30:16.000Z",
                            ),
                        },
                        {
                            "gateway_run_id": "swe-term-b",
                            "gateway_profile_id": 1,
                            "request_count": 14,
                            "manifest_window": (
                                "2026-03-08T00:31:00.000Z",
                                "2026-03-08T00:31:40.000Z",
                            ),
                        },
                    ],
                ),
                (
                    "terminal-bench-2.0",
                    [
                        {
                            "gateway_run_id": "tb-mini-a",
                            "gateway_profile_id": 0,
                            "request_count": 1,
                            "manifest_window": (
                                "2026-03-08T00:40:00.000Z",
                                "2026-03-08T00:40:50.000Z",
                            ),
                        },
                        {
                            "gateway_run_id": "tb-mini-b",
                            "gateway_profile_id": 1,
                            "request_count": 9,
                            "manifest_window": (
                                "2026-03-08T00:41:00.000Z",
                                "2026-03-08T00:42:30.000Z",
                            ),
                        },
                    ],
                    [
                        {
                            "gateway_run_id": "tb-term-a",
                            "gateway_profile_id": 0,
                            "request_count": 9,
                            "manifest_window": (
                                "2026-03-08T00:50:00.000Z",
                                "2026-03-08T00:51:30.000Z",
                            ),
                        },
                        {
                            "gateway_run_id": "tb-term-b",
                            "gateway_profile_id": 1,
                            "request_count": 18,
                            "manifest_window": (
                                "2026-03-08T00:52:00.000Z",
                                "2026-03-08T00:55:00.000Z",
                            ),
                        },
                    ],
                ),
            ]:
                _write_run_summary(
                    root_dir,
                    benchmark=benchmark,
                    agent_type="mini-swe-agent",
                    run_dir_name=f"{benchmark}-20260306T100000Z",
                    score=0.2,
                    jobs=mini_jobs,
                )
                _write_run_summary(
                    root_dir,
                    benchmark=benchmark,
                    agent_type="terminus-2",
                    run_dir_name=f"{benchmark}-20260306T110000Z",
                    score=0.3,
                    jobs=terminus_jobs,
                )

            input_path = tmp_path / "turns-scatter-duration-violin.json"
            output_path = tmp_path / "turns-scatter-duration-violin.png"
            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--root-dir",
                    str(root_dir),
                    "--output",
                    str(input_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PLOT_SCRIPT),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
