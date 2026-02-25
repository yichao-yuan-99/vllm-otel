"""Harbor backend for concurrent trial launching."""

from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from con_driver.backends.base import TrialBackend
from con_driver.models import TaskCandidate


def _safe_dataset_dir_name(dataset_spec: str) -> str:
    safe = []
    for char in dataset_spec:
        if char.isalnum() or char in {"-", "_", ".", "@"}:
            safe.append(char)
        else:
            safe.append("_")
    name = "".join(safe).strip("_")
    return name or "dataset"


def _is_task_dir(path: Path) -> bool:
    """Best-effort local task validation for Harbor tasks."""
    return (
        (path / "task.toml").is_file()
        and (path / "instruction.md").is_file()
        and (path / "environment").is_dir()
    )


def _discover_tasks(root: Path) -> list[Path]:
    tasks: list[Path] = []
    seen: set[Path] = set()
    for task_toml in sorted(root.rglob("task.toml")):
        candidate = task_toml.parent.resolve()
        if candidate in seen:
            continue
        if _is_task_dir(candidate):
            seen.add(candidate)
            tasks.append(candidate)
    return tasks


@dataclass(frozen=True)
class HarborBackendConfig:
    harbor_bin: list[str]
    harbor_args: list[str]


class HarborBackend(TrialBackend):
    """Backend that delegates dataset/trial operations to `harbor` CLI."""

    _INTERRUPT_GRACE_SEC = 120.0
    _TERMINATE_GRACE_SEC = 5.0

    def __init__(self, config: HarborBackendConfig):
        self._config = config

    async def prepare_task_pool(
        self,
        *,
        pool_specs: Sequence[str],
        datasets_root: Path,
    ) -> list[TaskCandidate]:
        datasets_root.mkdir(parents=True, exist_ok=True)

        tasks: list[TaskCandidate] = []
        for dataset_spec in pool_specs:
            output_dir = datasets_root / _safe_dataset_dir_name(dataset_spec)
            output_dir.mkdir(parents=True, exist_ok=True)

            command = [
                *self._config.harbor_bin,
                "datasets",
                "download",
                dataset_spec,
                "--output-dir",
                str(output_dir),
            ]
            await self._run_checked(command)

            discovered = _discover_tasks(output_dir)
            if not discovered:
                raise RuntimeError(
                    f"No tasks found after downloading dataset '{dataset_spec}' into "
                    f"{output_dir}"
                )

            for task_path in discovered:
                tasks.append(TaskCandidate(dataset=dataset_spec, path=task_path))

        if not tasks:
            raise RuntimeError("Task pool is empty after preparing all datasets")
        return tasks

    def build_launch_command(
        self,
        *,
        task: TaskCandidate,
        trial_id: str,
        trials_dir: Path,
    ) -> list[str]:
        return [
            *self._config.harbor_bin,
            "trials",
            "start",
            "-p",
            str(task.path),
            *self._config.harbor_args,
            "--trial-name",
            trial_id,
            "--trials-dir",
            str(trials_dir),
        ]

    async def _run_checked(self, command: list[str]) -> None:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate()
        except BaseException as exc:
            if process.returncode is None:
                process.send_signal(signal.SIGINT)
                if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
                    try:
                        await asyncio.wait_for(
                            process.wait(), timeout=self._INTERRUPT_GRACE_SEC
                        )
                    except asyncio.TimeoutError:
                        process.terminate()
                        try:
                            await asyncio.wait_for(
                                process.wait(), timeout=self._TERMINATE_GRACE_SEC
                            )
                        except asyncio.TimeoutError:
                            process.kill()
                            await process.wait()
                else:
                    try:
                        await asyncio.wait_for(
                            process.wait(), timeout=self._TERMINATE_GRACE_SEC
                        )
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
            raise

        if process.returncode != 0:
            raise RuntimeError(
                "Command failed with non-zero exit code "
                f"{process.returncode}: {' '.join(command)}\n"
                f"stdout:\n{stdout.decode('utf-8', errors='replace')}\n"
                f"stderr:\n{stderr.decode('utf-8', errors='replace')}"
            )
