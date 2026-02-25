"""Shared models for the concurrent driver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskCandidate:
    """A launchable local task path and its source dataset."""

    dataset: str
    path: Path


@dataclass(frozen=True)
class LaunchRequest:
    """A single launch request produced by the scheduler."""

    launch_index: int
    trial_id: str
    task: TaskCandidate
    trials_dir: Path
    stdout_log: Path
    stderr_log: Path
    command: list[str]


@dataclass(frozen=True)
class LaunchResult:
    """Result metadata for one launched trial process."""

    launch_index: int
    trial_id: str
    dataset: str
    task_path: str
    command: list[str]
    started_at: str
    finished_at: str
    duration_s: float
    return_code: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "launch_index": self.launch_index,
            "trial_id": self.trial_id,
            "dataset": self.dataset,
            "task_path": self.task_path,
            "command": self.command,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "return_code": self.return_code,
            "status": self.status,
        }


@dataclass(frozen=True)
class RunSummary:
    """Top-level run summary written at the end of execution."""

    run_id: str
    total_requested: int
    launched: int
    succeeded: int
    failed: int
    dry_run: bool
    results_dir: Path
    manifest_path: Path
    events_path: Path
