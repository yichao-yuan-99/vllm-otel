from __future__ import annotations

import asyncio
import json
from pathlib import Path
from random import Random
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CON_DRIVER_SRC = PROJECT_ROOT / "con-driver" / "src"
if str(CON_DRIVER_SRC) not in sys.path:
    sys.path.insert(0, str(CON_DRIVER_SRC))

from con_driver.backends.base import TrialBackend
from con_driver.models import TaskCandidate
from con_driver.patterns import EagerArrivalPattern
from con_driver.scheduler import (
    ConcurrentDriver,
    GatewayModeConfig,
    LaunchProfileConfig,
    SchedulerConfig,
    VLLMLogConfig,
)


class _FakeBackend(TrialBackend):
    def __init__(self, *, size: int, task_root: Path) -> None:
        self._tasks = [
            TaskCandidate(dataset="swebench-verified", path=(task_root / f"task-{index:04d}"))
            for index in range(size)
        ]

    def dataset_cache_root(self) -> Path:
        return Path("unused-cache-root")

    async def prepare_task_pool(
        self,
        *,
        pool_specs: list[str],
        datasets_root: Path,
    ) -> list[TaskCandidate]:
        return list(self._tasks)

    def build_launch_command(
        self,
        *,
        task: TaskCandidate,
        trial_id: str,
        trials_dir: Path,
        runtime_forwarded_args: list[str] | None = None,
    ) -> list[str]:
        command = ["echo", str(task.path)]
        if runtime_forwarded_args:
            command.extend(runtime_forwarded_args)
        return command


def _run_subset(
    *,
    backend: TrialBackend,
    results_dir: Path,
    subset_start: int,
    subset_end: int,
) -> set[str]:
    driver = ConcurrentDriver(
        backend=backend,
        arrival_pattern=EagerArrivalPattern(),
        rng=Random(1234),
        config=SchedulerConfig(
            max_concurrent=8,
            n_task=250,
            results_dir=results_dir,
            dry_run=True,
            sample_with_replacement=False,
            task_subset_start=subset_start,
            task_subset_end=subset_end,
        ),
    )
    summary = asyncio.run(driver.run(pool_specs=["swebench-verified"]))
    results_path = summary.results_dir / "meta" / "results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return {str(item["task_path"]) for item in payload}


def test_task_subsets_cover_full_pool_without_overlap(tmp_path: Path) -> None:
    backend = _FakeBackend(size=500, task_root=tmp_path / "tasks")
    shard_a = _run_subset(
        backend=backend,
        results_dir=tmp_path / "out",
        subset_start=0,
        subset_end=250,
    )
    shard_b = _run_subset(
        backend=backend,
        results_dir=tmp_path / "out",
        subset_start=250,
        subset_end=500,
    )

    assert len(shard_a) == 250
    assert len(shard_b) == 250
    assert shard_a.isdisjoint(shard_b)
    assert len(shard_a | shard_b) == 500


def test_task_subset_end_out_of_range_is_rejected(tmp_path: Path) -> None:
    backend = _FakeBackend(size=10, task_root=tmp_path / "tasks")
    driver = ConcurrentDriver(
        backend=backend,
        arrival_pattern=EagerArrivalPattern(),
        rng=Random(1),
        config=SchedulerConfig(
            max_concurrent=1,
            n_task=1,
            results_dir=tmp_path / "out",
            dry_run=True,
            sample_with_replacement=False,
            task_subset_start=0,
            task_subset_end=11,
        ),
    )

    with pytest.raises(ValueError, match="task_subset_end is out of range"):
        asyncio.run(driver.run(pool_specs=["swebench-verified"]))


def test_live_profile_selection_fills_low_to_high_profile_id(tmp_path: Path) -> None:
    backend = _FakeBackend(size=10, task_root=tmp_path / "tasks")
    driver = ConcurrentDriver(
        backend=backend,
        arrival_pattern=EagerArrivalPattern(),
        rng=Random(1),
        config=SchedulerConfig(
            max_concurrent=3,
            n_task=3,
            results_dir=tmp_path / "out",
            dry_run=True,
            sample_with_replacement=False,
            launch_profiles=[
                LaunchProfileConfig(
                    port_profile_id=2,
                    max_concurrent=1,
                    forwarded_args=[],
                    launch_env={},
                    gateway_base_url="http://127.0.0.1:38171",
                ),
                LaunchProfileConfig(
                    port_profile_id=1,
                    max_concurrent=2,
                    forwarded_args=[],
                    launch_env={},
                    gateway_base_url="http://127.0.0.1:28171",
                ),
            ],
        ),
    )

    selected = driver._select_launch_profile_for_live(active={})
    assert selected is not None
    assert selected.port_profile_id == 1

    active = {
        "a": SimpleNamespace(launch_profile_id=1),
        "b": SimpleNamespace(launch_profile_id=1),
    }
    selected = driver._select_launch_profile_for_live(active=active)
    assert selected is not None
    assert selected.port_profile_id == 2

    active["c"] = SimpleNamespace(launch_profile_id=2)
    assert driver._select_launch_profile_for_live(active=active) is None


def test_manifest_marks_cluster_mode_when_launch_profiles_are_configured(tmp_path: Path) -> None:
    backend = _FakeBackend(size=4, task_root=tmp_path / "tasks")
    driver = ConcurrentDriver(
        backend=backend,
        arrival_pattern=EagerArrivalPattern(),
        rng=Random(7),
        config=SchedulerConfig(
            max_concurrent=2,
            n_task=2,
            results_dir=tmp_path / "out",
            dry_run=True,
            sample_with_replacement=False,
            launch_profiles=[
                LaunchProfileConfig(
                    port_profile_id=0,
                    max_concurrent=1,
                    forwarded_args=[],
                    launch_env={},
                    gateway_base_url="http://127.0.0.1:11457",
                ),
                LaunchProfileConfig(
                    port_profile_id=1,
                    max_concurrent=1,
                    forwarded_args=[],
                    launch_env={},
                    gateway_base_url="http://127.0.0.1:21457",
                ),
            ],
        ),
    )

    summary = asyncio.run(driver.run(pool_specs=["swebench-verified"]))
    manifest_path = summary.results_dir / "meta" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["cluster_mode"] is True
    assert manifest["launch_mode"] == "cluster"


def test_manifest_records_cluster_vllm_log_endpoints(tmp_path: Path) -> None:
    backend = _FakeBackend(size=4, task_root=tmp_path / "tasks")
    driver = ConcurrentDriver(
        backend=backend,
        arrival_pattern=EagerArrivalPattern(),
        rng=Random(7),
        config=SchedulerConfig(
            max_concurrent=2,
            n_task=2,
            results_dir=tmp_path / "out",
            dry_run=True,
            sample_with_replacement=False,
            launch_profiles=[
                LaunchProfileConfig(
                    port_profile_id=0,
                    max_concurrent=1,
                    forwarded_args=[],
                    launch_env={},
                    vllm_log_endpoint="http://127.0.0.1:11451/metrics",
                ),
                LaunchProfileConfig(
                    port_profile_id=1,
                    max_concurrent=1,
                    forwarded_args=[],
                    launch_env={},
                    vllm_log_endpoint="http://127.0.0.1:24123/metrics",
                ),
            ],
            vllm_log=VLLMLogConfig(
                endpoint="http://127.0.0.1:11451/metrics",
                interval_s=1.0,
                timeout_s=5.0,
            ),
        ),
    )

    summary = asyncio.run(driver.run(pool_specs=["swebench-verified"]))
    manifest_path = summary.results_dir / "meta" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["vllm_log_enabled"] is True
    assert manifest["vllm_log_endpoint"] == "http://127.0.0.1:11451/metrics"
    assert manifest["vllm_log_endpoints"] == [
        "http://127.0.0.1:11451/metrics",
        "http://127.0.0.1:24123/metrics",
    ]


def test_gateway_agent_token_does_not_append_real_key(tmp_path: Path) -> None:
    backend = _FakeBackend(size=1, task_root=tmp_path / "tasks")
    driver = ConcurrentDriver(
        backend=backend,
        arrival_pattern=EagerArrivalPattern(),
        rng=Random(7),
        config=SchedulerConfig(
            max_concurrent=1,
            n_task=1,
            results_dir=tmp_path / "out",
            dry_run=True,
            sample_with_replacement=False,
            gateway=GatewayModeConfig(
                base_url="http://127.0.0.1:18171",
                job_output_root="gateway-output",
                timeout_s=3600.0,
            ),
        ),
    )

    token = driver._make_agent_api_token(launch_index=0, trial_id="trial-abc")
    assert token.startswith("condrv_run-")
    assert token.endswith("_0000_trial-abc")
    assert "@@" not in token


def test_cluster_mode_starts_one_vllm_monitor_per_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend(size=1, task_root=tmp_path / "tasks")
    driver = ConcurrentDriver(
        backend=backend,
        arrival_pattern=EagerArrivalPattern(),
        rng=Random(7),
        config=SchedulerConfig(
            max_concurrent=1,
            n_task=1,
            results_dir=tmp_path / "out",
            dry_run=False,
            sample_with_replacement=False,
            launch_profiles=[
                LaunchProfileConfig(
                    port_profile_id=0,
                    max_concurrent=1,
                    forwarded_args=[],
                    launch_env={},
                    vllm_log_endpoint="http://127.0.0.1:11451/metrics",
                ),
                LaunchProfileConfig(
                    port_profile_id=1,
                    max_concurrent=1,
                    forwarded_args=[],
                    launch_env={},
                    vllm_log_endpoint="http://127.0.0.1:24123/metrics",
                ),
            ],
            vllm_log=VLLMLogConfig(
                endpoint="http://127.0.0.1:11451/metrics",
                interval_s=1.0,
                timeout_s=5.0,
            ),
        ),
    )

    started_monitors: list[dict[str, object]] = []

    async def _fake_start_vllm_monitor(*, target):  # type: ignore[no-untyped-def]
        stdout_log = target.log_dir / "monitor.stdout.log"
        stderr_log = target.log_dir / "monitor.stderr.log"
        started_monitors.append(
            {
                "port_profile_id": target.port_profile_id,
                "endpoint": target.endpoint,
                "log_dir": target.log_dir,
            }
        )
        return SimpleNamespace(
            port_profile_id=target.port_profile_id,
            endpoint=target.endpoint,
            log_dir=target.log_dir,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )

    async def _fake_stop_vllm_monitor(_monitor):  # type: ignore[no-untyped-def]
        return 0

    async def _fake_run_live(*, launch_plan, progress, progress_task_id):  # type: ignore[no-untyped-def]
        return []

    monkeypatch.setattr(driver, "_start_vllm_monitor", _fake_start_vllm_monitor)
    monkeypatch.setattr(driver, "_stop_vllm_monitor", _fake_stop_vllm_monitor)
    monkeypatch.setattr(driver, "_run_live", _fake_run_live)

    summary = asyncio.run(driver.run(pool_specs=["swebench-verified"]))
    manifest_path = summary.results_dir / "meta" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert [entry["port_profile_id"] for entry in started_monitors] == [0, 1]
    assert [entry["endpoint"] for entry in started_monitors] == [
        "http://127.0.0.1:11451/metrics",
        "http://127.0.0.1:24123/metrics",
    ]
    assert manifest["vllm_log_monitor_return_code"] is None
    assert manifest["vllm_log_monitor_return_codes"] == [
        {
            "port_profile_id": 0,
            "endpoint": "http://127.0.0.1:11451/metrics",
            "return_code": 0,
        },
        {
            "port_profile_id": 1,
            "endpoint": "http://127.0.0.1:24123/metrics",
            "return_code": 0,
        },
    ]


def test_cluster_mode_starts_and_ends_job_for_each_gateway_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend(size=1, task_root=tmp_path / "tasks")
    driver = ConcurrentDriver(
        backend=backend,
        arrival_pattern=EagerArrivalPattern(),
        rng=Random(7),
        config=SchedulerConfig(
            max_concurrent=2,
            n_task=1,
            results_dir=tmp_path / "out",
            dry_run=False,
            sample_with_replacement=False,
            launch_profiles=[
                LaunchProfileConfig(
                    port_profile_id=0,
                    max_concurrent=1,
                    forwarded_args=[],
                    launch_env={},
                    gateway_base_url="http://127.0.0.1:18171",
                ),
                LaunchProfileConfig(
                    port_profile_id=1,
                    max_concurrent=1,
                    forwarded_args=[],
                    launch_env={},
                    gateway_base_url="http://127.0.0.1:28171",
                ),
            ],
            gateway=GatewayModeConfig(
                base_url="http://127.0.0.1:18171",
                job_output_root="gateway-output",
                timeout_s=5.0,
            ),
        ),
    )

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.status_code = 200
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self) -> dict[str, object]:
            return self._payload

    calls: list[tuple[str, dict[str, object], float]] = []

    def _fake_post(url: str, *, json: dict[str, object], timeout: float) -> _FakeResponse:
        calls.append((url, dict(json), timeout))
        return _FakeResponse({"ok": True, "url": url})

    monkeypatch.setattr("con_driver.scheduler.requests.post", _fake_post)

    async def _fake_run_live(*, launch_plan, progress, progress_task_id):  # type: ignore[no-untyped-def]
        return []

    monkeypatch.setattr(driver, "_run_live", _fake_run_live)

    summary = asyncio.run(driver.run(pool_specs=["swebench-verified"]))
    manifest_path = summary.results_dir / "meta" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    start_calls = [call for call in calls if call[0].endswith("/job/start")]
    end_calls = [call for call in calls if call[0].endswith("/job/end")]
    assert [url for url, _, _ in start_calls] == [
        "http://127.0.0.1:18171/job/start",
        "http://127.0.0.1:28171/job/start",
    ]
    assert [payload.get("output_location") for _, payload, _ in start_calls] == [
        str((summary.results_dir / "gateway-output" / "profile-0").resolve()),
        str((summary.results_dir / "gateway-output" / "profile-1").resolve()),
    ]
    assert [url for url, _, _ in end_calls] == [
        "http://127.0.0.1:18171/job/end",
        "http://127.0.0.1:28171/job/end",
    ]
    assert all(payload == {"status": "completed"} for _, payload, _ in end_calls)

    assert manifest["gateway_job_start_status"] == "ok"
    assert manifest["gateway_job_started_base_urls"] == [
        "http://127.0.0.1:18171",
        "http://127.0.0.1:28171",
    ]
    assert manifest["gateway_output_locations"] == [
        {
            "port_profile_id": 0,
            "base_url": "http://127.0.0.1:18171",
            "output_location": str((summary.results_dir / "gateway-output" / "profile-0").resolve()),
            "output_location_relative_to_results_dir": "gateway-output/profile-0",
        },
        {
            "port_profile_id": 1,
            "base_url": "http://127.0.0.1:28171",
            "output_location": str((summary.results_dir / "gateway-output" / "profile-1").resolve()),
            "output_location_relative_to_results_dir": "gateway-output/profile-1",
        },
    ]
    assert manifest["gateway_job_end_status"] == "completed"
    assert manifest["gateway_job_ended_base_urls"] == [
        "http://127.0.0.1:18171",
        "http://127.0.0.1:28171",
    ]


def test_gateway_wrapper_receives_resolved_agent_name(tmp_path: Path) -> None:
    backend = _FakeBackend(size=1, task_root=tmp_path / "tasks")
    driver = ConcurrentDriver(
        backend=backend,
        arrival_pattern=EagerArrivalPattern(),
        rng=Random(7),
        config=SchedulerConfig(
            max_concurrent=1,
            n_task=1,
            results_dir=tmp_path / "out",
            dry_run=True,
            sample_with_replacement=False,
            gateway=GatewayModeConfig(
                base_url="http://127.0.0.1:18171",
                job_output_root="gateway-output",
                timeout_s=5.0,
            ),
            effective_config={"resolved_agent_name": "mini-swe-agent"},
        ),
    )

    wrapped = driver._wrap_with_gateway_agent_wrapper(
        command=["harbor", "trials", "start"],
        api_token="token-123",
        gateway_base_url="http://127.0.0.1:18171",
        resolved_agent_name="mini-swe-agent",
    )

    assert "--agent-name" in wrapped
    idx = wrapped.index("--agent-name")
    assert wrapped[idx + 1] == "mini-swe-agent"
