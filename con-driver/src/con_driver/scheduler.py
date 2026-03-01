"""Core scheduler for concurrent trial launching."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import TextIO
from uuid import uuid4

import requests
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from con_driver.backends.base import TrialBackend
from con_driver.models import LaunchRequest, LaunchResult, RunSummary, TaskCandidate
from con_driver.patterns import ArrivalPattern


@dataclass(frozen=True)
class SchedulerConfig:
    max_concurrent: int
    n_task: int
    results_dir: Path
    dry_run: bool = False
    sample_with_replacement: bool = True
    effective_config: dict[str, object] | None = None
    vllm_log: "VLLMLogConfig | None" = None
    gateway: "GatewayModeConfig | None" = None
    launch_env: dict[str, str] | None = None


@dataclass(frozen=True)
class VLLMLogConfig:
    endpoint: str
    interval_s: float = 1.0
    timeout_s: float = 5.0


@dataclass(frozen=True)
class GatewayModeConfig:
    base_url: str
    job_output_root: str
    timeout_s: float = 3600.0


@dataclass
class _ActiveLaunch:
    request: LaunchRequest
    process: asyncio.subprocess.Process
    wait_task: asyncio.Task[int]
    started_at: datetime
    stdout_handle: TextIO
    stderr_handle: TextIO


@dataclass
class _VLLMMonitorProcess:
    process: asyncio.subprocess.Process
    wait_task: asyncio.Task[int]
    stdout_handle: TextIO
    stderr_handle: TextIO
    stdout_log: Path
    stderr_log: Path


@dataclass(frozen=True)
class _LaunchPlanItem:
    launch_index: int
    task: TaskCandidate


class ConcurrentDriver:
    """Driver that samples tasks and launches backend commands concurrently."""
    _INTERRUPT_GRACE_SEC = 120.0
    _TERMINATE_GRACE_SEC = 5.0
    _MONITOR_INTERRUPT_GRACE_SEC = 15.0
    _MONITOR_TERMINATE_GRACE_SEC = 5.0
    _OUTPUT_MARKER_NAME = "CON_DRIVER_OUTPUT"
    _GATEWAY_AGENT_WRAPPER_MODULE = "con_driver.gateway_wrapper"

    def __init__(
        self,
        *,
        backend: TrialBackend,
        arrival_pattern: ArrivalPattern,
        rng: Random,
        config: SchedulerConfig,
    ):
        if config.max_concurrent <= 0:
            raise ValueError("max_concurrent must be > 0")
        if config.n_task <= 0:
            raise ValueError("n_task must be > 0")
        if config.vllm_log is not None:
            if config.vllm_log.interval_s <= 0:
                raise ValueError("vllm_log.interval_s must be > 0")
            if config.vllm_log.timeout_s <= 0:
                raise ValueError("vllm_log.timeout_s must be > 0")
        if config.gateway is not None:
            if config.gateway.timeout_s <= 0:
                raise ValueError("gateway.timeout_s must be > 0")
            if not config.gateway.base_url.strip():
                raise ValueError("gateway.base_url cannot be empty")
            if not config.gateway.job_output_root.strip():
                raise ValueError("gateway.job_output_root cannot be empty")
            normalized_subdir = config.gateway.job_output_root.strip()
            if normalized_subdir in {".", "./"}:
                raise ValueError("gateway.job_output_root must be a subdirectory path")
            if Path(normalized_subdir).is_absolute():
                raise ValueError("gateway.job_output_root must be a relative path")

        self._backend = backend
        self._arrival_pattern = arrival_pattern
        self._rng = rng
        self._config = config

        self._run_id = self._make_run_id()
        self._target_dir = config.results_dir.resolve()
        self._datasets_dir = self._backend.dataset_cache_root().resolve()
        self._set_results_dir(self._target_dir)

    async def run(self, *, pool_specs: list[str]) -> RunSummary:
        run_started_at = _utc_now_iso()
        run_started_monotonic = time.monotonic()
        self._target_dir.mkdir(parents=True, exist_ok=True)
        self._datasets_dir.mkdir(parents=True, exist_ok=True)

        task_pool = await self._backend.prepare_task_pool(
            pool_specs=pool_specs,
            datasets_root=self._datasets_dir,
        )
        dataset_mode = self._is_dataset_mode(pool_specs=pool_specs, task_pool=task_pool)
        dataset_mode_dataset = next(iter({task.dataset for task in task_pool}), None)
        if dataset_mode and dataset_mode_dataset is not None:
            run_dir = self._target_dir / self._make_dataset_mode_dir_name(dataset_mode_dataset)
        else:
            run_dir = self._target_dir / self._make_job_dir_name()
        self._set_results_dir(run_dir)

        self._prepare_directories()
        self._write_output_marker()
        self._write_effective_config_toml()
        # Reset per-run artifacts if the output directory is reused.
        self._events_path.write_text("", encoding="utf-8")

        manifest_payload: dict[str, object] = {
            "run_id": self._run_id,
            "started_at": run_started_at,
            "pool_specs": pool_specs,
            "max_concurrent": self._config.max_concurrent,
            "n_task": self._config.n_task,
            "dry_run": self._config.dry_run,
            "sample_with_replacement": self._config.sample_with_replacement,
            "arrival_pattern": self._arrival_pattern.describe(),
            "target_results_dir": str(self._target_dir),
            "results_dir": str(self._results_dir),
            "datasets_dir": str(self._datasets_dir),
            "dataset_mode": dataset_mode,
            "run_mode": "dataset" if dataset_mode else "job",
            "dataset_mode_dataset": dataset_mode_dataset if dataset_mode else None,
            "dataset_mode_dataset_size": len(task_pool) if dataset_mode else None,
            "finished_at": None,
            "run_duration_s": None,
            "reward_avg": None,
            "vllm_log_enabled": self._config.vllm_log is not None,
            "vllm_log_dir": (
                str(self._vllm_log_dir) if self._config.vllm_log is not None else None
            ),
            "vllm_log_endpoint": (
                self._config.vllm_log.endpoint
                if self._config.vllm_log is not None
                else None
            ),
            "vllm_log_interval_s": (
                self._config.vllm_log.interval_s
                if self._config.vllm_log is not None
                else None
            ),
            "vllm_log_timeout_s": (
                self._config.vllm_log.timeout_s
                if self._config.vllm_log is not None
                else None
            ),
            "gateway_enabled": self._config.gateway is not None,
            "gateway_base_url": (
                self._config.gateway.base_url
                if self._config.gateway is not None
                else None
            ),
            "gateway_job_output_subdir": (
                self._config.gateway.job_output_root
                if self._config.gateway is not None
                else None
            ),
            "gateway_job_output_root": (
                self._config.gateway.job_output_root
                if self._config.gateway is not None
                else None
            ),
            "gateway_output_location": None,
            "gateway_output_relative_to_results_dir": None,
            "gateway_job_start_status": None,
            "gateway_job_end_status": None,
        }
        self._write_manifest(manifest_payload)

        self._append_event(
            {
                "event": "task_pool_ready",
                "task_pool_size": len(task_pool),
                "datasets": sorted({task.dataset for task in task_pool}),
                "dataset_mode": dataset_mode,
            }
        )

        launch_plan = self._build_launch_plan(task_pool=task_pool)
        pattern_name = str(self._arrival_pattern.describe().get("name", "")).strip().lower()
        self._append_event(
            {
                "event": "sampling_plan_ready",
                "sample_size": len(launch_plan),
                "sample_with_replacement": self._config.sample_with_replacement,
                "pattern": pattern_name,
            }
        )

        if dataset_mode and not self._config.dry_run and dataset_mode_dataset is not None:
            print(
                "Dataset mode: "
                f"dataset={dataset_mode_dataset} "
                f"dataset_size={len(task_pool)} "
                f"results_dir={self._results_dir}",
                flush=True,
            )

        monitor: _VLLMMonitorProcess | None = None
        monitor_exit_code: int | None = None
        results: list[LaunchResult] = []
        run_exception: BaseException | None = None
        gateway_job_started = False
        gateway_job_status: str | None = None
        gateway_output_location: str | None = None
        try:
            if self._config.gateway is not None and not self._config.dry_run:
                gateway_output_location = self._build_gateway_output_location()
                gateway_output_relative = str(
                    Path(gateway_output_location).resolve().relative_to(self._results_dir)
                )
                self._gateway_job_start(output_location=gateway_output_location)
                gateway_job_started = True
                manifest_payload["gateway_output_location"] = gateway_output_location
                manifest_payload["gateway_output_relative_to_results_dir"] = (
                    gateway_output_relative
                )
                manifest_payload["gateway_job_start_status"] = "ok"
                self._write_manifest(manifest_payload)
                self._append_event(
                    {
                        "event": "gateway_job_start",
                        "base_url": self._config.gateway.base_url,
                        "output_location": gateway_output_location,
                        "output_location_relative_to_results_dir": gateway_output_relative,
                        "status": "ok",
                    }
                )

            if self._config.vllm_log is not None and not self._config.dry_run:
                monitor = await self._start_vllm_monitor()
                manifest_payload.update(
                    {
                        "vllm_log_stdout": str(monitor.stdout_log),
                        "vllm_log_stderr": str(monitor.stderr_log),
                    }
                )
                self._write_manifest(manifest_payload)
                self._append_event(
                    {
                        "event": "vllm_log_started",
                        "endpoint": self._config.vllm_log.endpoint,
                        "interval_s": self._config.vllm_log.interval_s,
                        "timeout_s": self._config.vllm_log.timeout_s,
                        "log_dir": str(self._vllm_log_dir),
                        "stdout_log": str(monitor.stdout_log),
                        "stderr_log": str(monitor.stderr_log),
                    }
                )

            with self._create_progress() as progress:
                progress_task_id = progress.add_task(
                    "dry-run" if self._config.dry_run else "running",
                    total=len(launch_plan),
                    launched=0,
                    active=0,
                )
                if self._config.dry_run:
                    results = self._run_dry(
                        launch_plan=launch_plan,
                        progress=progress,
                        progress_task_id=progress_task_id,
                    )
                else:
                    results = await self._run_live(
                        launch_plan=launch_plan,
                        progress=progress,
                        progress_task_id=progress_task_id,
                    )
        except BaseException as exc:
            run_exception = exc
            if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
                gateway_job_status = "interrupted"
            else:
                gateway_job_status = "failed"
            raise
        finally:
            if monitor is not None:
                monitor_exit_code = await self._stop_vllm_monitor(monitor)
                self._append_event(
                    {
                        "event": "vllm_log_stopped",
                        "return_code": monitor_exit_code,
                    }
                )
            if gateway_job_started and self._config.gateway is not None and not self._config.dry_run:
                final_status = gateway_job_status
                if final_status is None:
                    failed_count = sum(
                        1 for result in results if result.status not in {"succeeded", "dry-run"}
                    )
                    final_status = "completed" if failed_count == 0 else "failed"
                try:
                    gateway_end_response = self._gateway_job_end(status=final_status)
                    manifest_payload["gateway_job_end_status"] = final_status
                    self._write_manifest(manifest_payload)
                    self._append_event(
                        {
                            "event": "gateway_job_end",
                            "status": final_status,
                            "response": gateway_end_response,
                        }
                    )
                except Exception as exc:
                    self._append_event(
                        {
                            "event": "gateway_job_end_failed",
                            "status": final_status,
                            "error": str(exc),
                        }
                    )
                    if run_exception is None:
                        raise

        succeeded = sum(1 for result in results if result.status in {"succeeded", "dry-run"})
        failed = len(results) - succeeded

        self._results_path.write_text(
            json.dumps([result.to_dict() for result in results], indent=2),
            encoding="utf-8",
        )

        reward_avg = self._compute_reward_avg(results)
        finished_at = _utc_now_iso()
        run_duration_s = max(time.monotonic() - run_started_monotonic, 0.0)
        self._append_event(
            {
                "event": "run_complete",
                "finished_at": finished_at,
                "run_duration_s": run_duration_s,
                "launched": len(results),
                "succeeded": succeeded,
                "failed": failed,
                "reward_avg": reward_avg,
                "vllm_log_monitor_return_code": monitor_exit_code,
                "results_json": str(self._results_path),
            }
        )
        manifest_payload.update(
            {
                "finished_at": finished_at,
                "run_duration_s": run_duration_s,
                "launched": len(results),
                "succeeded": succeeded,
                "failed": failed,
                "reward_avg": reward_avg,
                "vllm_log_monitor_return_code": monitor_exit_code,
                "results_json": str(self._results_path),
            }
        )
        self._write_manifest(manifest_payload)

        return RunSummary(
            run_id=self._run_id,
            total_requested=self._config.n_task,
            launched=len(results),
            succeeded=succeeded,
            failed=failed,
            dry_run=self._config.dry_run,
            results_dir=self._results_dir,
            manifest_path=self._manifest_path,
            events_path=self._events_path,
        )

    def _run_dry(
        self,
        *,
        launch_plan: list[_LaunchPlanItem],
        progress: Progress,
        progress_task_id: TaskID,
    ) -> list[LaunchResult]:
        results: list[LaunchResult] = []
        for item in launch_plan:
            request = self._build_request(launch_index=item.launch_index, task=item.task)
            now = _utc_now_iso()
            event_payload: dict[str, object] = {
                "event": "launch_dry_run",
                "launch_index": request.launch_index,
                "trial_id": request.trial_id,
                "dataset": request.task.dataset,
                "task_path": str(request.task.path),
                "command": request.command,
            }
            self._append_event(
                event_payload
            )
            results.append(
                LaunchResult(
                    launch_index=request.launch_index,
                    trial_id=request.trial_id,
                    dataset=request.task.dataset,
                    task_path=str(request.task.path),
                    command=request.command,
                    started_at=now,
                    finished_at=now,
                    duration_s=0.0,
                    return_code=0,
                    status="dry-run",
                )
            )
            progress.update(
                progress_task_id,
                advance=1,
                launched=item.launch_index + 1,
                active=0,
            )
        return results

    async def _run_live(
        self,
        *,
        launch_plan: list[_LaunchPlanItem],
        progress: Progress,
        progress_task_id: TaskID,
    ) -> list[LaunchResult]:
        active: dict[str, _ActiveLaunch] = {}
        results: list[LaunchResult] = []
        next_launch_at = time.monotonic()

        try:
            for item in launch_plan:
                launch_index = item.launch_index
                while len(active) >= self._config.max_concurrent:
                    await self._collect_completed(
                        active=active,
                        results=results,
                        progress=progress,
                        progress_task_id=progress_task_id,
                        launched=launch_index,
                    )

                delay = next_launch_at - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)

                request = self._build_request(launch_index=launch_index, task=item.task)
                active_launch = await self._start_launch(request)
                active[request.trial_id] = active_launch
                progress.update(
                    progress_task_id,
                    launched=launch_index + 1,
                    active=len(active),
                )

                event_payload: dict[str, object] = {
                    "event": "launch",
                    "launch_index": request.launch_index,
                    "trial_id": request.trial_id,
                    "dataset": request.task.dataset,
                    "task_path": str(request.task.path),
                    "stdout_log": str(request.stdout_log),
                    "stderr_log": str(request.stderr_log),
                    "command": request.command,
                }
                self._append_event(
                    event_payload
                )

                next_launch_at = time.monotonic() + self._arrival_pattern.next_delay_s()

            while active:
                await self._collect_completed(
                    active=active,
                    results=results,
                    progress=progress,
                    progress_task_id=progress_task_id,
                    launched=len(launch_plan),
                )

            return results
        except BaseException as exc:
            await self._terminate_active(active, reason=exc)
            raise

    async def _start_launch(self, request: LaunchRequest) -> _ActiveLaunch:
        request.stdout_log.parent.mkdir(parents=True, exist_ok=True)
        request.stderr_log.parent.mkdir(parents=True, exist_ok=True)

        stdout_handle = request.stdout_log.open("w", encoding="utf-8")
        stderr_handle = request.stderr_log.open("w", encoding="utf-8")

        try:
            launch_env = None
            if self._config.launch_env:
                launch_env = os.environ.copy()
                launch_env.update(self._config.launch_env)
            process = await asyncio.create_subprocess_exec(
                *request.command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=launch_env,
            )
        except Exception:
            stdout_handle.close()
            stderr_handle.close()
            raise

        return _ActiveLaunch(
            request=request,
            process=process,
            wait_task=asyncio.create_task(process.wait()),
            started_at=datetime.now(UTC),
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
        )

    async def _start_vllm_monitor(self) -> _VLLMMonitorProcess:
        if self._config.vllm_log is None:
            raise RuntimeError("vLLM monitor config is not enabled.")

        self._vllm_log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = self._vllm_log_dir / "monitor.stdout.log"
        stderr_log = self._vllm_log_dir / "monitor.stderr.log"
        stdout_handle = stdout_log.open("w", encoding="utf-8")
        stderr_handle = stderr_log.open("w", encoding="utf-8")

        command = [
            sys.executable,
            "-m",
            "con_driver.vllm_metrics_monitor",
            "--endpoint",
            self._config.vllm_log.endpoint,
            "--output-dir",
            str(self._vllm_log_dir),
            "--interval-s",
            str(self._config.vllm_log.interval_s),
            "--timeout-s",
            str(self._config.vllm_log.timeout_s),
            "--block-size",
            "100",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
        except Exception:
            stdout_handle.close()
            stderr_handle.close()
            raise

        return _VLLMMonitorProcess(
            process=process,
            wait_task=asyncio.create_task(process.wait()),
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )

    async def _stop_vllm_monitor(self, monitor: _VLLMMonitorProcess) -> int:
        process = monitor.process
        wait_task = monitor.wait_task

        try:
            if process.returncode is None:
                process.send_signal(signal.SIGINT)
                try:
                    await asyncio.wait_for(wait_task, timeout=self._MONITOR_INTERRUPT_GRACE_SEC)
                except asyncio.TimeoutError:
                    if process.returncode is None:
                        process.terminate()
                    try:
                        await asyncio.wait_for(
                            wait_task, timeout=self._MONITOR_TERMINATE_GRACE_SEC
                        )
                    except asyncio.TimeoutError:
                        if process.returncode is None:
                            process.kill()
                        await wait_task
            else:
                await wait_task
        finally:
            monitor.stdout_handle.close()
            monitor.stderr_handle.close()

        return process.returncode if process.returncode is not None else 1

    async def _collect_completed(
        self,
        *,
        active: dict[str, _ActiveLaunch],
        results: list[LaunchResult],
        progress: Progress,
        progress_task_id: TaskID,
        launched: int,
    ) -> None:
        if not active:
            return

        done, _ = await asyncio.wait(
            [launch.wait_task for launch in active.values()],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for done_task in done:
            launch = next(item for item in active.values() if item.wait_task is done_task)
            return_code = done_task.result()
            finished_at = datetime.now(UTC)
            duration_s = (finished_at - launch.started_at).total_seconds()
            status = "succeeded" if return_code == 0 else "failed"

            launch.stdout_handle.close()
            launch.stderr_handle.close()
            active.pop(launch.request.trial_id)

            result = LaunchResult(
                launch_index=launch.request.launch_index,
                trial_id=launch.request.trial_id,
                dataset=launch.request.task.dataset,
                task_path=str(launch.request.task.path),
                command=launch.request.command,
                started_at=launch.started_at.isoformat(),
                finished_at=finished_at.isoformat(),
                duration_s=duration_s,
                return_code=return_code,
                status=status,
            )
            results.append(result)

            self._append_event(
                {
                    "event": "complete",
                    "launch_index": result.launch_index,
                    "trial_id": result.trial_id,
                    "status": result.status,
                    "return_code": result.return_code,
                    "duration_s": result.duration_s,
                }
            )
            progress.update(
                progress_task_id,
                advance=1,
                launched=launched,
                active=len(active),
            )

    async def _terminate_active(
        self,
        active: dict[str, _ActiveLaunch],
        *,
        reason: BaseException | None = None,
    ) -> None:
        if not active:
            return

        use_interrupt_signal = isinstance(reason, (KeyboardInterrupt, asyncio.CancelledError))
        first_signal = signal.SIGINT if use_interrupt_signal else signal.SIGTERM
        first_signal_name = "SIGINT" if use_interrupt_signal else "SIGTERM"
        interrupt_broadcast_started = time.monotonic()
        if use_interrupt_signal:
            print(
                "Ctrl-C captured. Broadcasting SIGINT to "
                f"{len(active)} active subprocesses.",
                flush=True,
            )

        self._append_event(
            {
                "event": "subprocess_signal_broadcast",
                "signal": first_signal_name,
                "active_processes": len(active),
            }
        )

        for launch in active.values():
            if launch.process.returncode is None:
                launch.process.send_signal(first_signal)

        wait_tasks = [launch.wait_task for launch in active.values()]
        if wait_tasks:
            finished = await self._wait_for_subprocesses(
                wait_tasks=wait_tasks,
                timeout_s=self._INTERRUPT_GRACE_SEC,
                description="Waiting for subprocesses after Ctrl-C",
                since_start_monotonic=interrupt_broadcast_started,
                with_timing_bar=use_interrupt_signal,
            )
            if not finished:
                if use_interrupt_signal:
                    elapsed = time.monotonic() - interrupt_broadcast_started
                    print(
                        "SIGINT grace timeout reached "
                        f"({elapsed:.1f}s since Ctrl-C). Escalating to SIGTERM.",
                        flush=True,
                    )
                    self._append_event(
                        {
                            "event": "subprocess_signal_escalation",
                            "signal": "SIGTERM",
                            "remaining_processes": sum(
                                1
                                for launch in active.values()
                                if launch.process.returncode is None
                            ),
                        }
                    )
                    for launch in active.values():
                        if launch.process.returncode is None:
                            launch.process.terminate()

                    finished_after_term = await self._wait_for_subprocesses(
                        wait_tasks=wait_tasks,
                        timeout_s=self._TERMINATE_GRACE_SEC,
                        description="Waiting for subprocesses after SIGTERM",
                        since_start_monotonic=interrupt_broadcast_started,
                        with_timing_bar=True,
                    )
                    if not finished_after_term:
                        elapsed = time.monotonic() - interrupt_broadcast_started
                        print(
                            "SIGTERM grace timeout reached "
                            f"({elapsed:.1f}s since Ctrl-C). Escalating to SIGKILL.",
                            flush=True,
                        )
                        self._append_event(
                            {
                                "event": "subprocess_signal_escalation",
                                "signal": "SIGKILL",
                                "remaining_processes": sum(
                                    1
                                    for launch in active.values()
                                    if launch.process.returncode is None
                                ),
                            }
                        )
                        for launch in active.values():
                            if launch.process.returncode is None:
                                launch.process.kill()
                        await asyncio.gather(*wait_tasks, return_exceptions=True)
                else:
                    self._append_event(
                        {
                            "event": "subprocess_signal_escalation",
                            "signal": "SIGKILL",
                            "remaining_processes": sum(
                                1 for launch in active.values() if launch.process.returncode is None
                            ),
                        }
                    )
                    for launch in active.values():
                        if launch.process.returncode is None:
                            launch.process.kill()
                    await asyncio.gather(*wait_tasks, return_exceptions=True)
        if use_interrupt_signal:
            elapsed = time.monotonic() - interrupt_broadcast_started
            print(
                "All active subprocesses exited after Ctrl-C "
                f"(elapsed {elapsed:.1f}s).",
                flush=True,
            )

        for launch in active.values():
            launch.stdout_handle.close()
            launch.stderr_handle.close()

        active.clear()

    async def _wait_for_subprocesses(
        self,
        *,
        wait_tasks: list[asyncio.Task[int]],
        timeout_s: float,
        description: str,
        since_start_monotonic: float,
        with_timing_bar: bool,
    ) -> bool:
        if not wait_tasks:
            return True
        if not with_timing_bar:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*wait_tasks, return_exceptions=True),
                    timeout=timeout_s,
                )
                return True
            except asyncio.TimeoutError:
                return False

        phase_started = time.monotonic()
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]{task.description}[/bold yellow]"),
            BarColumn(),
            TextColumn("{task.completed:.1f}/{task.total:.1f}s"),
            TextColumn(
                "remaining={task.fields[remaining]} "
                "since_ctrl_c={task.fields[since_ctrl_c]}"
            ),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            progress_task = progress.add_task(
                description,
                total=timeout_s,
                completed=0.0,
                remaining=sum(1 for task in wait_tasks if not task.done()),
                since_ctrl_c="0.0s",
            )
            while True:
                pending = [task for task in wait_tasks if not task.done()]
                now = time.monotonic()
                phase_elapsed = now - phase_started
                since_ctrl_c = now - since_start_monotonic

                progress.update(
                    progress_task,
                    completed=min(phase_elapsed, timeout_s),
                    remaining=len(pending),
                    since_ctrl_c=f"{since_ctrl_c:.1f}s",
                )

                if not pending:
                    return True
                if phase_elapsed >= timeout_s:
                    return False

                await asyncio.wait(
                    pending,
                    timeout=min(0.2, timeout_s - phase_elapsed),
                    return_when=asyncio.FIRST_COMPLETED,
                )

    def _build_request(
        self,
        *,
        launch_index: int,
        task: TaskCandidate,
    ) -> LaunchRequest:
        trial_id = self._make_trial_id(launch_index)
        trials_dir = self._trials_root

        command = self._backend.build_launch_command(
            task=task,
            trial_id=trial_id,
            trials_dir=trials_dir,
        )
        if self._config.gateway is not None:
            api_token = self._make_agent_api_token(
                launch_index=launch_index,
                trial_id=trial_id,
            )
            command = [
                *command,
                "--agent-kwarg",
                f"api_key={api_token}",
            ]
            if not self._config.dry_run:
                command = self._wrap_with_gateway_agent_wrapper(
                    command=command,
                    api_token=api_token,
                )

        return LaunchRequest(
            launch_index=launch_index,
            trial_id=trial_id,
            task=task,
            trials_dir=trials_dir,
            stdout_log=self._logs_dir / f"{trial_id}.stdout.log",
            stderr_log=self._logs_dir / f"{trial_id}.stderr.log",
            command=command,
        )

    def _wrap_with_gateway_agent_wrapper(
        self,
        *,
        command: list[str],
        api_token: str,
    ) -> list[str]:
        if self._config.gateway is None:
            raise RuntimeError("Gateway wrapper requested but gateway config is disabled")
        return [
            sys.executable,
            "-m",
            self._GATEWAY_AGENT_WRAPPER_MODULE,
            "--gateway-url",
            self._config.gateway.base_url,
            "--api-token",
            api_token,
            "--timeout-s",
            str(self._config.gateway.timeout_s),
            "--",
            *command,
        ]

    def _make_agent_api_token(self, *, launch_index: int, trial_id: str) -> str:
        return f"condrv_{self._run_id}_{launch_index:04d}_{trial_id}"

    def _build_launch_plan(
        self,
        *,
        task_pool: list[TaskCandidate],
    ) -> list[_LaunchPlanItem]:
        sampled_tasks = self._build_sampling_plan(task_pool)
        return [
            _LaunchPlanItem(launch_index=index, task=task)
            for index, task in enumerate(sampled_tasks)
        ]

    def _build_sampling_plan(self, task_pool: list[TaskCandidate]) -> list[TaskCandidate]:
        if self._config.sample_with_replacement:
            return [self._rng.choice(task_pool) for _ in range(self._config.n_task)]

        if self._config.n_task > len(task_pool):
            raise ValueError(
                "Sampling without replacement requested but n_task exceeds pool size: "
                f"n_task={self._config.n_task}, pool_size={len(task_pool)}"
            )

        shuffled = list(task_pool)
        self._rng.shuffle(shuffled)
        return shuffled[: self._config.n_task]

    def _prepare_directories(self) -> None:
        self._target_dir.mkdir(parents=True, exist_ok=True)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._datasets_dir.mkdir(parents=True, exist_ok=True)
        self._trials_root.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir.mkdir(parents=True, exist_ok=True)

    def _set_results_dir(self, path: Path) -> None:
        self._results_dir = path.resolve()
        self._trials_root = self._results_dir / "trials"
        self._logs_dir = self._results_dir / "logs"
        self._meta_dir = self._results_dir / "meta"
        self._vllm_log_dir = self._results_dir / "vllm-log"
        self._manifest_path = self._meta_dir / "run_manifest.json"
        self._events_path = self._meta_dir / "events.jsonl"
        self._results_path = self._meta_dir / "results.json"

    def _build_gateway_output_location(self) -> str:
        if self._config.gateway is None:
            raise RuntimeError("Gateway output location requested but gateway config is disabled")
        subdir = self._config.gateway.job_output_root.strip()
        if not subdir or subdir in {".", "./"}:
            raise ValueError("gateway_job_output_root must be a non-empty subdirectory path")
        subdir_path = Path(subdir)
        if subdir_path.is_absolute():
            raise ValueError("gateway_job_output_root must be a relative subdirectory path")
        output_path = (self._results_dir / subdir_path).resolve()
        try:
            output_path.relative_to(self._results_dir)
        except ValueError as exc:
            raise ValueError(
                "gateway_job_output_root must stay within the con-driver run directory"
            ) from exc
        if output_path == self._results_dir:
            raise ValueError("gateway_job_output_root must be a subdirectory, not run root")
        return str(output_path)

    def _gateway_job_start(self, *, output_location: str) -> dict[str, object]:
        if self._config.gateway is None:
            raise RuntimeError("Gateway job/start requested but gateway config is disabled")
        endpoint = f"{self._config.gateway.base_url.rstrip('/')}/job/start"
        try:
            response = requests.post(
                endpoint,
                json={"output_location": output_location},
                timeout=self._config.gateway.timeout_s,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to call gateway /job/start: {exc}") from exc
        if response.status_code >= 300:
            raise RuntimeError(
                "Gateway /job/start returned non-success "
                f"{response.status_code}: {response.text}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Gateway /job/start response must be a JSON object")
        return payload

    def _gateway_job_end(self, *, status: str) -> dict[str, object]:
        if self._config.gateway is None:
            raise RuntimeError("Gateway job/end requested but gateway config is disabled")
        endpoint = f"{self._config.gateway.base_url.rstrip('/')}/job/end"
        try:
            response = requests.post(
                endpoint,
                json={"status": status},
                timeout=self._config.gateway.timeout_s,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to call gateway /job/end: {exc}") from exc
        if response.status_code >= 300:
            raise RuntimeError(
                "Gateway /job/end returned non-success "
                f"{response.status_code}: {response.text}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Gateway /job/end response must be a JSON object")
        return payload

    def _write_output_marker(self) -> None:
        marker_path = self._results_dir / self._OUTPUT_MARKER_NAME
        marker_path.write_text(
            "\n".join(
                [
                    "This directory is managed by con-driver.",
                    f"updated_at = {_utc_now_iso()}",
                    f"run_id = {self._run_id}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_effective_config_toml(self) -> None:
        if self._config.effective_config is None:
            return

        payload = self._config.effective_config
        source_config = payload.get("source_config")
        backend_name = str(payload.get("backend", "harbor"))
        pool_raw = str(payload.get("pool_raw", ""))
        pool_specs = payload.get("pool_specs", [])
        pattern = str(payload.get("pattern", ""))
        pattern_args_tokens = payload.get("pattern_args_tokens", [])
        max_concurrent = int(payload.get("max_concurrent", self._config.max_concurrent))
        n_task = int(payload.get("n_task", self._config.n_task))
        dry_run = bool(payload.get("dry_run", self._config.dry_run))
        sample_without_replacement = bool(
            payload.get("sample_without_replacement", not self._config.sample_with_replacement)
        )
        seed = payload.get("seed")
        harbor_bin_tokens = payload.get("harbor_bin_tokens", [])
        forwarded_args = payload.get("forwarded_args", [])
        vllm_log_enabled = bool(payload.get("vllm_log_enabled", False))
        vllm_log_endpoint_value = payload.get(
            "vllm_log_endpoint",
            self._config.vllm_log.endpoint if self._config.vllm_log is not None else "",
        )
        vllm_log_endpoint = (
            vllm_log_endpoint_value if isinstance(vllm_log_endpoint_value, str) else ""
        )
        vllm_log_interval_s = float(
            payload.get(
                "vllm_log_interval_s",
                self._config.vllm_log.interval_s if self._config.vllm_log is not None else 1.0,
            )
        )
        vllm_log_timeout_s = float(
            payload.get(
                "vllm_log_timeout_s",
                self._config.vllm_log.timeout_s if self._config.vllm_log is not None else 5.0,
            )
        )
        gateway_enabled = bool(payload.get("gateway_enabled", self._config.gateway is not None))
        gateway_url_value = payload.get(
            "gateway_url",
            self._config.gateway.base_url if self._config.gateway is not None else "",
        )
        gateway_url = gateway_url_value if isinstance(gateway_url_value, str) else ""
        gateway_job_output_root_value = payload.get(
            "gateway_job_output_subdir",
            payload.get(
                "gateway_job_output_root",
                self._config.gateway.job_output_root if self._config.gateway is not None else "",
            ),
        )
        gateway_job_output_root = (
            gateway_job_output_root_value
            if isinstance(gateway_job_output_root_value, str)
            else ""
        )
        gateway_timeout_s = float(
            payload.get(
                "gateway_timeout_s",
                self._config.gateway.timeout_s if self._config.gateway is not None else 30.0,
            )
        )
        port_profile_id = payload.get("port_profile_id")
        resolved_agent_name = payload.get("resolved_agent_name")
        resolved_model_name = payload.get("resolved_model_name")
        resolved_model_context_window = payload.get("resolved_model_context_window")
        agent_base_url = payload.get("agent_base_url")

        def _toml_quote(value: str) -> str:
            return json.dumps(value)

        def _toml_list(values: object) -> str:
            if not isinstance(values, list):
                return "[]"
            quoted = ", ".join(_toml_quote(str(item)) for item in values)
            return f"[{quoted}]"

        lines = [
            "# Auto-generated by con-driver.",
            f"# generated_at = {_toml_quote(_utc_now_iso())}",
            "",
            "[run]",
            f"pool = {_toml_quote(pool_raw)}",
            f"pool_specs = {_toml_list(pool_specs)}",
            f"pattern = {_toml_quote(pattern)}",
            f"pattern_args = {_toml_list(pattern_args_tokens)}",
            f"max_concurrent = {max_concurrent}",
            f"n_task = {n_task}",
            f"dry_run = {'true' if dry_run else 'false'}",
            (
                "sample_without_replacement = "
                f"{'true' if sample_without_replacement else 'false'}"
            ),
            f"target_results_dir = {_toml_quote(str(self._target_dir.resolve()))}",
            f"results_dir = {_toml_quote(str(self._results_dir.resolve()))}",
        ]
        if isinstance(source_config, str) and source_config:
            lines.append(f"source_config = {_toml_quote(source_config)}")
        if isinstance(seed, int) and not isinstance(seed, bool):
            lines.append(f"seed = {seed}")

        lines.extend(
            [
                "",
                "[backend]",
                f"name = {_toml_quote(backend_name)}",
                f"harbor_bin = {_toml_list(harbor_bin_tokens)}",
                f"forwarded_args = {_toml_list(forwarded_args)}",
            ]
        )
        if (
            port_profile_id is not None
            or isinstance(resolved_agent_name, str)
            or isinstance(resolved_model_name, str)
            or isinstance(agent_base_url, str)
        ):
            lines.extend(
                [
                    "",
                    "[runtime]",
                ]
            )
            if port_profile_id is not None:
                lines.append(f"port_profile_id = {_toml_quote(str(port_profile_id))}")
            if isinstance(resolved_agent_name, str) and resolved_agent_name:
                lines.append(f"agent = {_toml_quote(resolved_agent_name)}")
            if isinstance(resolved_model_name, str) and resolved_model_name:
                lines.append(f"served_model_name = {_toml_quote(resolved_model_name)}")
            if isinstance(resolved_model_context_window, int):
                lines.append(f"context_window = {resolved_model_context_window}")
            if isinstance(agent_base_url, str) and agent_base_url:
                lines.append(f"agent_base_url = {_toml_quote(agent_base_url)}")
        lines.extend(
            [
                "",
                "[vllm_log]",
                f"enabled = {'true' if vllm_log_enabled else 'false'}",
                f"endpoint = {_toml_quote(vllm_log_endpoint)}",
                f"interval_s = {vllm_log_interval_s}",
                f"timeout_s = {vllm_log_timeout_s}",
            ]
        )
        lines.extend(
            [
                "",
                "[gateway]",
                f"enabled = {'true' if gateway_enabled else 'false'}",
                f"url = {_toml_quote(gateway_url)}",
                f"job_output_subdir = {_toml_quote(gateway_job_output_root)}",
                f"job_output_root = {_toml_quote(gateway_job_output_root)}",
                f"timeout_s = {gateway_timeout_s}",
            ]
        )

        (self._meta_dir / "config.toml").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def _is_dataset_mode(self, *, pool_specs: list[str], task_pool: list[TaskCandidate]) -> bool:
        if len(pool_specs) != 1:
            return False
        if len({task.dataset for task in task_pool}) != 1:
            return False
        if self._config.sample_with_replacement:
            return False
        if self._config.n_task != len(task_pool):
            return False
        pattern_name = str(self._arrival_pattern.describe().get("name", "")).strip().lower()
        return pattern_name == "eager"

    def _make_dataset_mode_dir_name(self, dataset_name: str) -> str:
        safe_dataset = self._safe_dir_name(dataset_name)
        return self._make_unique_run_dir_name(prefix=safe_dataset)

    def _make_job_dir_name(self) -> str:
        return self._make_unique_run_dir_name(prefix="job")

    def _make_unique_run_dir_name(self, *, prefix: str) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        base_name = f"{prefix}-{stamp}"
        candidate = base_name
        suffix = 2
        while (self._target_dir / candidate).exists():
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        return candidate

    def _safe_dir_name(self, raw: str) -> str:
        chars: list[str] = []
        for char in raw:
            if char.isalnum() or char in {"-", "_", ".", "@"}:
                chars.append(char)
            else:
                chars.append("_")
        normalized = "".join(chars).strip("_")
        return normalized or "dataset"

    def _compute_reward_avg(self, results: list[LaunchResult]) -> float | None:
        rewards: list[float] = []
        for result in results:
            result_path = self._trials_root / result.trial_id / "result.json"
            if not result_path.is_file():
                continue

            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            verifier = payload.get("verifier_result")
            if not isinstance(verifier, dict):
                continue
            rewards_payload = verifier.get("rewards")
            if not isinstance(rewards_payload, dict):
                continue
            reward = rewards_payload.get("reward")
            if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                continue
            rewards.append(float(reward))

        if not rewards:
            return None
        return sum(rewards) / len(rewards)

    def _create_progress(self) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("launched={task.fields[launched]} active={task.fields[active]}"),
            TimeElapsedColumn(),
            transient=False,
        )

    def _write_manifest(self, payload: dict[str, object]) -> None:
        self._manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _append_event(self, payload: dict[str, object]) -> None:
        event = {"time": _utc_now_iso(), **payload}
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def _make_trial_id(self, launch_index: int) -> str:
        return f"trial-{launch_index:04d}-{uuid4().hex[:8]}"

    def _make_run_id(self) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"run-{stamp}-{uuid4().hex[:6]}"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
