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
    REPO_ROOT / "figures" / "turns-duration" / "materialize_turns_duration.py"
)
PLOT_SCRIPT = REPO_ROOT / "figures" / "turns-duration" / "plot_turns_duration.py"


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
) -> None:
    run_dir = root_dir / benchmark / agent_type / run_dir_name
    _write_json(
        run_dir / "run-stats" / "run-stats-summary.json",
        {
            "dataset": benchmark,
            "agent_type": agent_type,
            "score": score,
            "job_count": len(jobs),
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


class TurnsDurationFigureTest(unittest.TestCase):
    def test_materialize_turns_duration_builds_expected_panels(self) -> None:
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
                        "request_count": 1,
                        "agent_window": (
                            "2026-03-08T00:00:00.000Z",
                            "2026-03-08T00:00:01.000Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:00:00.000Z",
                            "2026-03-08T00:00:03.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:00:00.500Z",
                            "2026-03-08T00:00:01.500Z",
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
                        "request_count": 1,
                        "manifest_window": (
                            "2026-03-08T00:02:00.000Z",
                            "2026-03-08T00:02:12.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:02:02.000Z",
                            "2026-03-08T00:02:05.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-requests",
                        "gateway_profile_id": 3,
                        "request_count": 3,
                        "request_window": (
                            "2026-03-08T00:03:03.000Z",
                            "2026-03-08T00:03:10.000Z",
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
                        "request_count": 2,
                        "agent_window": (
                            "2026-03-08T00:10:00.000Z",
                            "2026-03-08T00:10:03.000Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:10:00.000Z",
                            "2026-03-08T00:10:04.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:10:00.500Z",
                            "2026-03-08T00:10:03.500Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-dab-terminus-b",
                        "gateway_profile_id": 1,
                        "request_count": 3,
                        "agent_window": (
                            "2026-03-08T00:11:00.000Z",
                            "2026-03-08T00:11:06.000Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:11:00.000Z",
                            "2026-03-08T00:11:07.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:11:01.000Z",
                            "2026-03-08T00:11:05.000Z",
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
                        "request_count": 2,
                        "agent_window": (
                            "2026-03-08T00:20:00.000Z",
                            "2026-03-08T00:20:05.000Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:20:00.000Z",
                            "2026-03-08T00:20:06.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:20:01.000Z",
                            "2026-03-08T00:20:04.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-swe-mini-b",
                        "gateway_profile_id": 1,
                        "request_count": 3,
                        "agent_window": (
                            "2026-03-08T00:21:00.000Z",
                            "2026-03-08T00:21:10.000Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:21:00.000Z",
                            "2026-03-08T00:21:11.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:21:02.000Z",
                            "2026-03-08T00:21:08.000Z",
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
                        "gateway_run_id": "run-swe-terminus-a",
                        "gateway_profile_id": 0,
                        "request_count": 2,
                        "agent_window": (
                            "2026-03-08T00:30:00.000Z",
                            "2026-03-08T00:30:07.000Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:30:00.000Z",
                            "2026-03-08T00:30:08.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:30:01.000Z",
                            "2026-03-08T00:30:06.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-swe-terminus-b",
                        "gateway_profile_id": 1,
                        "request_count": 3,
                        "agent_window": (
                            "2026-03-08T00:31:00.000Z",
                            "2026-03-08T00:31:14.000Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:31:00.000Z",
                            "2026-03-08T00:31:15.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:31:02.000Z",
                            "2026-03-08T00:31:10.000Z",
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
                        "gateway_run_id": "run-tb-mini-a",
                        "gateway_profile_id": 0,
                        "request_count": 1,
                        "agent_window": (
                            "2026-03-08T00:40:00.000Z",
                            "2026-03-08T00:40:09.000Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:40:00.000Z",
                            "2026-03-08T00:40:10.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:40:02.000Z",
                            "2026-03-08T00:40:08.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-tb-mini-b",
                        "gateway_profile_id": 1,
                        "request_count": 2,
                        "agent_window": (
                            "2026-03-08T00:41:00.000Z",
                            "2026-03-08T00:41:18.000Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:41:00.000Z",
                            "2026-03-08T00:41:19.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:41:01.000Z",
                            "2026-03-08T00:41:13.000Z",
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
                        "gateway_run_id": "run-tb-terminus-a",
                        "gateway_profile_id": 0,
                        "request_count": 2,
                        "agent_window": (
                            "2026-03-08T00:50:00.000Z",
                            "2026-03-08T00:50:09.000Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:50:00.000Z",
                            "2026-03-08T00:50:10.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:50:02.000Z",
                            "2026-03-08T00:50:07.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-tb-terminus-b",
                        "gateway_profile_id": 1,
                        "request_count": 3,
                        "agent_window": (
                            "2026-03-08T00:51:00.000Z",
                            "2026-03-08T00:51:18.000Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:51:00.000Z",
                            "2026-03-08T00:51:19.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:51:02.000Z",
                            "2026-03-08T00:51:16.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-tb-terminus-c",
                        "gateway_profile_id": 2,
                        "request_count": 4,
                        "agent_window": (
                            "2026-03-08T00:52:00.000Z",
                            "2026-03-08T00:52:36.000Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:52:00.000Z",
                            "2026-03-08T00:52:37.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:52:03.000Z",
                            "2026-03-08T00:52:28.000Z",
                        ),
                    },
                    {
                        "gateway_run_id": "run-tb-terminus-d",
                        "gateway_profile_id": 3,
                        "request_count": 5,
                        "agent_window": (
                            "2026-03-08T00:53:00.000Z",
                            "2026-03-08T00:54:12.000Z",
                        ),
                        "job_window": (
                            "2026-03-08T00:53:00.000Z",
                            "2026-03-08T00:54:13.000Z",
                        ),
                        "request_window": (
                            "2026-03-08T00:53:04.000Z",
                            "2026-03-08T00:53:50.000Z",
                        ),
                    },
                ],
            )

            output_path = tmp_path / "turns-duration.json"
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
            self.assertEqual(payload["figure_name"], "turns-duration")
            self.assertEqual(payload["metric_name"], "agent_duration_s")
            self.assertEqual(
                payload["metric_label"],
                "Agent duration per job (s)",
            )
            self.assertEqual(
                payload["benchmark_order"],
                ["dabstep", "swebench-verified", "terminal-bench-2.0"],
            )
            self.assertEqual(payload["agent_order"], ["mini-swe-agent", "terminus-2"])
            self.assertEqual(payload["panel_count"], 6)
            self.assertEqual(payload["total_job_count"], 16)

            first_panel = payload["panels"][0]
            self.assertEqual(first_panel["benchmark"], "dabstep")
            self.assertEqual(first_panel["agent_type"], "mini-swe-agent")
            self.assertEqual(first_panel["run_dir_name"], "dabstep-20260306T194929Z")
            self.assertEqual(first_panel["candidate_run_count"], 2)
            self.assertEqual(first_panel["durations_s"], [2.5, 9.0, 12.0, 7.0])
            self.assertEqual(
                first_panel["duration_source_counts"],
                {
                    "lifecycle:agent": 1,
                    "lifecycle:job": 1,
                    "manifest:run": 1,
                    "requests:window": 1,
                },
            )
            self.assertAlmostEqual(first_panel["avg_agent_duration_s_computed"], 7.625)
            self.assertEqual(first_panel["stats"]["sample_count"], 4)
            self.assertAlmostEqual(first_panel["stats"]["min"], 2.5)
            self.assertAlmostEqual(first_panel["stats"]["max"], 12.0)
            self.assertAlmostEqual(first_panel["stats"]["median"], 8.0)

            last_panel = payload["panels"][-1]
            self.assertEqual(last_panel["benchmark"], "terminal-bench-2.0")
            self.assertEqual(last_panel["agent_label"], "Terminus")
            self.assertEqual(last_panel["durations_s"], [9.0, 18.0, 36.0, 72.0])
            self.assertAlmostEqual(last_panel["stats"]["mean"], 33.75)
            self.assertAlmostEqual(last_panel["stats"]["q1"], 15.75)
            self.assertAlmostEqual(last_panel["stats"]["q3"], 45.0)

    def test_plot_turns_duration_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results" / "qwen3-coder-30b"

            for benchmark, mini_durations, terminus_durations in [
                ("dabstep", [2.0, 4.0, 8.0], [3.0, 6.0, 12.0]),
                ("swebench-verified", [5.0, 10.0, 15.0], [7.0, 14.0, 21.0]),
                ("terminal-bench-2.0", [9.0, 18.0, 27.0], [9.0, 18.0, 36.0]),
            ]:
                _write_run_summary(
                    root_dir,
                    benchmark=benchmark,
                    agent_type="mini-swe-agent",
                    run_dir_name=f"{benchmark}-20260306T100000Z",
                    score=0.2,
                    jobs=[
                        {
                            "gateway_run_id": f"{benchmark}-mini-{index}",
                            "gateway_profile_id": index,
                            "request_count": index + 1,
                            "agent_window": (
                                f"2026-03-08T00:{index:02d}:00.000Z",
                                f"2026-03-08T00:{index:02d}:{int(duration):02d}.000Z",
                            ),
                            "job_window": (
                                f"2026-03-08T00:{index:02d}:00.000Z",
                                f"2026-03-08T00:{index:02d}:{int(duration + 1):02d}.000Z",
                            ),
                            "request_window": (
                                f"2026-03-08T00:{index:02d}:01.000Z",
                                f"2026-03-08T00:{index:02d}:{max(1, int(duration - 1)):02d}.000Z",
                            ),
                        }
                        for index, duration in enumerate(mini_durations)
                    ],
                )
                _write_run_summary(
                    root_dir,
                    benchmark=benchmark,
                    agent_type="terminus-2",
                    run_dir_name=f"{benchmark}-20260306T110000Z",
                    score=0.3,
                    jobs=[
                        {
                            "gateway_run_id": f"{benchmark}-terminus-{index}",
                            "gateway_profile_id": index,
                            "request_count": index + 1,
                            "agent_window": (
                                f"2026-03-08T01:{index:02d}:00.000Z",
                                f"2026-03-08T01:{index:02d}:{int(duration):02d}.000Z",
                            ),
                            "job_window": (
                                f"2026-03-08T01:{index:02d}:00.000Z",
                                f"2026-03-08T01:{index:02d}:{int(duration + 1):02d}.000Z",
                            ),
                            "request_window": (
                                f"2026-03-08T01:{index:02d}:01.000Z",
                                f"2026-03-08T01:{index:02d}:{max(1, int(duration - 1)):02d}.000Z",
                            ),
                        }
                        for index, duration in enumerate(terminus_durations)
                    ],
                )

            input_path = tmp_path / "turns-duration.json"
            output_path = tmp_path / "turns-duration.png"
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
