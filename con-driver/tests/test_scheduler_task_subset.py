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
from con_driver.scheduler import ConcurrentDriver, LaunchProfileConfig, SchedulerConfig


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
