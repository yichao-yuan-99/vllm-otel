"""Backend interface for dataset prep and command construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from con_driver.models import TaskCandidate


class TrialBackend(ABC):
    """Abstraction for trial launcher backends."""

    @abstractmethod
    def dataset_cache_root(self) -> Path:
        """Return the shared dataset cache root used by this backend."""

    @abstractmethod
    async def prepare_task_pool(
        self,
        *,
        pool_specs: Sequence[str],
        datasets_root: Path,
    ) -> list[TaskCandidate]:
        """Download/resolve datasets and return all launchable task paths."""

    @abstractmethod
    def build_launch_command(
        self,
        *,
        task: TaskCandidate,
        trial_id: str,
        trials_dir: Path,
    ) -> list[str]:
        """Build a command that launches one trial for the given task."""
