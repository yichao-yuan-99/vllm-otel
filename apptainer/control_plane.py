#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared control-plane logic for the Apptainer HPC server/client workflow."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import textwrap
import threading
import time
from typing import Any, Callable
import tomllib
from urllib import error as urlerror
from urllib import request as urlrequest


DEFAULT_JAEGER_IMAGE = "docker://jaegertracing/all-in-one:1.57"
DEFAULT_VLLM_IMAGE = "docker://yichaoyuan/vllm-openai-otel:v0.14.1-otel-lp-rocm"
DEFAULT_JAEGER_SIF_NAME = "jaeger-all-in-one-1.57.sif"
DEFAULT_VLLM_SIF_NAME = "vllm-openai-otel-v0.14.1-otel-lp-rocm.sif"
DEFAULT_STOP_WAIT_TIMEOUT_SECONDS = 180
DEFAULT_STOP_POLL_INTERVAL_SECONDS = 2
DEFAULT_TEST_PREFLIGHT_TIMEOUT_SECONDS = 30
DEFAULT_TEST_PREFLIGHT_POLL_INTERVAL_SECONDS = 2
DEFAULT_WAIT_UP_POLL_INTERVAL_SECONDS = 2


class ControlPlaneError(RuntimeError):
    """Typed error raised by control-plane actions."""

    def __init__(
        self,
        *,
        message: str,
        code: int = 1,
        http_status: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.http_status = http_status
        self.details = details or {}


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    gpus_per_node: int
    gpu_memory_gb: float
    total_vram_gb: float
    max_time: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    vllm_model_name: str
    served_model_name: str
    weight_vram_gb: float
    extra_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeConfig:
    repo_root: Path
    app_dir: Path
    host: str
    port: int
    run_dir: Path
    log_dir: Path
    state_file: Path
    login_host: str
    job_name: str
    job_nodes: int
    service_port: int
    jaeger_otlp_port: int
    jaeger_ui_port: int
    startup_timeout: int
    ssh_options: list[str]
    partitions: dict[str, PartitionSpec]
    models: dict[str, ModelSpec]
    env: dict[str, str]
    apptainer_imgs: Path
    jaeger_image: str
    vllm_image: str
    jaeger_sif: Path
    vllm_sif: Path


@dataclass(frozen=True)
class CommandResult:
    code: int
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActiveJob:
    job_id: str
    partition: str
    model: str
    submitted_at: str
    tensor_parallel_size: int
    service_port: int
    jaeger_otlp_port: int
    jaeger_ui_port: int
    sbatch_script: str
    slurm_out_log: str
    slurm_err_log: str
    jaeger_log: str
    vllm_log: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActiveJob":
        return cls(
            job_id=str(payload["job_id"]),
            partition=str(payload["partition"]),
            model=str(payload["model"]),
            submitted_at=str(payload["submitted_at"]),
            tensor_parallel_size=int(payload["tensor_parallel_size"]),
            service_port=int(payload["service_port"]),
            jaeger_otlp_port=int(payload["jaeger_otlp_port"]),
            jaeger_ui_port=int(payload["jaeger_ui_port"]),
            sbatch_script=str(payload["sbatch_script"]),
            slurm_out_log=str(payload["slurm_out_log"]),
            slurm_err_log=str(payload["slurm_err_log"]),
            jaeger_log=str(payload["jaeger_log"]),
            vllm_log=str(payload["vllm_log"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "partition": self.partition,
            "model": self.model,
            "submitted_at": self.submitted_at,
            "tensor_parallel_size": self.tensor_parallel_size,
            "service_port": self.service_port,
            "jaeger_otlp_port": self.jaeger_otlp_port,
            "jaeger_ui_port": self.jaeger_ui_port,
            "sbatch_script": self.sbatch_script,
            "slurm_out_log": self.slurm_out_log,
            "slurm_err_log": self.slurm_err_log,
            "jaeger_log": self.jaeger_log,
            "vllm_log": self.vllm_log,
        }


class ControlPlane:
    """Main API used by the HTTP server and CLI tests."""

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path.resolve()
        self._cfg = load_runtime_config(self._config_path)
        self._lock = threading.Lock()
        self._start_progress_lock = threading.Lock()
        self._stop_progress_lock = threading.Lock()
        self._test_progress_lock = threading.Lock()
        self._start_progress: dict[str, Any] = {
            "status": "idle",
            "phase": None,
            "message": "no start command has been run yet",
            "job_id": None,
            "started_at": None,
            "finished_at": None,
            "updated_at": _utc_now_iso(),
        }
        self._stop_progress: dict[str, Any] = {
            "status": "idle",
            "phase": None,
            "message": "no stop command has been run yet",
            "job_id": None,
            "started_at": None,
            "finished_at": None,
            "updated_at": _utc_now_iso(),
        }
        self._test_progress: dict[str, Any] = {
            "status": "idle",
            "phase": None,
            "message": "no test has been run yet",
            "job_id": None,
            "started_at": None,
            "finished_at": None,
            "updated_at": _utc_now_iso(),
        }

        self._cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self._cfg.log_dir.mkdir(parents=True, exist_ok=True)
        self._cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._archive_previous_artifacts()

    @property
    def config(self) -> RuntimeConfig:
        return self._cfg

    def pull(self) -> CommandResult:
        with self._lock:
            self._require_command("apptainer")
            self._cfg.apptainer_imgs.mkdir(parents=True, exist_ok=True)

            actions: list[str] = []
            if self._cfg.jaeger_sif.exists():
                actions.append(f"skip existing {self._cfg.jaeger_sif}")
            else:
                self._run_checked(
                    [
                        "apptainer",
                        "pull",
                        str(self._cfg.jaeger_sif),
                        self._cfg.jaeger_image,
                    ],
                    timeout_seconds=60 * 60,
                )
                actions.append(f"pulled {self._cfg.jaeger_image} -> {self._cfg.jaeger_sif}")

            if self._cfg.vllm_sif.exists():
                actions.append(f"skip existing {self._cfg.vllm_sif}")
            else:
                self._run_checked(
                    ["apptainer", "pull", str(self._cfg.vllm_sif), self._cfg.vllm_image],
                    timeout_seconds=2 * 60 * 60,
                )
                actions.append(f"pulled {self._cfg.vllm_image} -> {self._cfg.vllm_sif}")

            return CommandResult(
                code=0,
                message="pull complete",
                data={
                    "actions": actions,
                    "jaeger_sif": str(self._cfg.jaeger_sif),
                    "vllm_sif": str(self._cfg.vllm_sif),
                },
            )

    def start(self, *, partition: str, model: str, block: bool = False) -> CommandResult:
        start_started_at = _utc_now_iso()
        self._set_start_progress(
            status="running",
            phase="validate",
            message=f"validating start request partition={partition} model={model}",
            job_id=None,
            started_at=start_started_at,
            finished_at=None,
        )

        try:
            with self._lock:
                self._require_command("sbatch")
                self._require_command("squeue")
                self._ensure_sif_files()

                state = self._load_state()
                state_changed, active, job_status = self._refresh_active_job(state)
                if state_changed:
                    self._save_state(state)

                if active is not None and job_status is not None:
                    raise ControlPlaneError(
                        message=(
                            f"active job already exists: {active.job_id} ({job_status}). "
                            "Stop it first before launching another."
                        ),
                        code=11,
                        http_status=409,
                        details={"active_job": active.to_dict(), "job_status": job_status},
                    )

                partition_spec = self._cfg.partitions.get(partition)
                if partition_spec is None:
                    raise ControlPlaneError(
                        message=f"unknown partition '{partition}'",
                        code=12,
                        http_status=400,
                        details={"allowed_partitions": sorted(self._cfg.partitions.keys())},
                    )

                model_spec = self._cfg.models.get(model)
                if model_spec is None:
                    raise ControlPlaneError(
                        message=f"unknown model '{model}'",
                        code=13,
                        http_status=400,
                        details={"allowed_models": sorted(self._cfg.models.keys())},
                    )

                max_weight = partition_spec.total_vram_gb * 0.75
                if model_spec.weight_vram_gb > max_weight:
                    raise ControlPlaneError(
                        message=(
                            f"model '{model}' requires {model_spec.weight_vram_gb:.1f} GB which exceeds "
                            f"75% of partition '{partition}' VRAM ({max_weight:.1f} GB)"
                        ),
                        code=14,
                        http_status=422,
                        details={
                            "model_weight_vram_gb": model_spec.weight_vram_gb,
                            "partition_total_vram_gb": partition_spec.total_vram_gb,
                            "max_allowed_weight_vram_gb": max_weight,
                        },
                    )

                self._set_start_progress(
                    status="running",
                    phase="submit",
                    message=f"writing sbatch script and submitting to partition {partition_spec.name}",
                    job_id=None,
                    started_at=start_started_at,
                    finished_at=None,
                )
                self._ensure_port_available_on_login(self._cfg.service_port)
                script_path = self._write_sbatch_script(
                    partition_spec=partition_spec,
                    model_spec=model_spec,
                )
                sbatch_result = self._run_checked(["sbatch", str(script_path)], timeout_seconds=120)
                job_id = _extract_sbatch_job_id(f"{sbatch_result.stdout}\n{sbatch_result.stderr}")

                self._set_start_progress(
                    status="running",
                    phase="record",
                    message=f"submitted job {job_id}; recording active state",
                    job_id=job_id,
                    started_at=start_started_at,
                    finished_at=None,
                )

                slurm_out_log = str(self._cfg.log_dir / f"slurm.{job_id}.out")
                slurm_err_log = str(self._cfg.log_dir / f"slurm.{job_id}.err")
                jaeger_log = str(self._cfg.log_dir / f"jaeger.{job_id}.log")
                vllm_log = str(self._cfg.log_dir / f"vllm.{job_id}.log")

                new_active = ActiveJob(
                    job_id=job_id,
                    partition=partition_spec.name,
                    model=model_spec.name,
                    submitted_at=_utc_now_iso(),
                    tensor_parallel_size=partition_spec.gpus_per_node,
                    service_port=self._cfg.service_port,
                    jaeger_otlp_port=self._cfg.jaeger_otlp_port,
                    jaeger_ui_port=self._cfg.jaeger_ui_port,
                    sbatch_script=str(script_path),
                    slurm_out_log=slurm_out_log,
                    slurm_err_log=slurm_err_log,
                    jaeger_log=jaeger_log,
                    vllm_log=vllm_log,
                )
                state["active_job"] = new_active.to_dict()
                self._save_state(state)

                result_data = {
                    "job_id": job_id,
                    "partition": partition_spec.name,
                    "model": model_spec.name,
                    "time_limit": partition_spec.max_time,
                    "tensor_parallel_size": partition_spec.gpus_per_node,
                    "service_port": self._cfg.service_port,
                    "jaeger_otlp_port": self._cfg.jaeger_otlp_port,
                    "jaeger_ui_port": self._cfg.jaeger_ui_port,
                    "sbatch_script": str(script_path),
                    "blocked": block,
                }

            if not block:
                finished_at = _utc_now_iso()
                self._set_start_progress(
                    status="succeeded",
                    phase="done",
                    message=f"submitted job {job_id}",
                    job_id=job_id,
                    started_at=start_started_at,
                    finished_at=finished_at,
                )
                return CommandResult(
                    code=0,
                    message=f"submitted job {job_id}",
                    data=result_data,
                )

            self._set_start_progress(
                status="running",
                phase="wait_services",
                message=f"submitted job {job_id}; waiting for vLLM and Jaeger readiness",
                job_id=job_id,
                started_at=start_started_at,
                finished_at=None,
            )

            last_wait_message: str | None = None

            def _on_wait_snapshot(snapshot: dict[str, Any]) -> None:
                nonlocal last_wait_message
                active_status = snapshot.get("active_job_status")
                vllm_ok = bool((snapshot.get("vllm") or {}).get("ok"))
                jaeger_ok = bool((snapshot.get("jaeger") or {}).get("ok"))
                message = (
                    f"waiting for services: slurm_status={active_status}, "
                    f"vllm_up={vllm_ok}, jaeger_up={jaeger_ok}"
                )
                if message == last_wait_message:
                    return
                last_wait_message = message
                self._set_start_progress(
                    status="running",
                    phase="wait_services",
                    message=message,
                    job_id=job_id,
                    started_at=start_started_at,
                    finished_at=None,
                )

            readiness = self._wait_for_services_up(
                timeout_seconds=self._cfg.startup_timeout,
                poll_interval_seconds=DEFAULT_WAIT_UP_POLL_INTERVAL_SECONDS,
                expected_job_id=job_id,
                progress_callback=_on_wait_snapshot,
            )
            result_data["readiness"] = readiness
            result_data["waited_seconds"] = readiness.get("waited_seconds")

            finished_at = _utc_now_iso()
            self._set_start_progress(
                status="succeeded",
                phase="done",
                message=f"submitted job {job_id} and services are up",
                job_id=job_id,
                started_at=start_started_at,
                finished_at=finished_at,
            )
            return CommandResult(
                code=0,
                message=f"submitted job {job_id} and services are up",
                data=result_data,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_start_progress(
                status="failed",
                phase="failed",
                message=str(exc),
                job_id=self._current_start_job_id(),
                started_at=start_started_at,
                finished_at=_utc_now_iso(),
            )
            raise

    def stop(self, *, reason: str = "stopped", block: bool = False) -> CommandResult:
        stop_started_at = _utc_now_iso()
        self._set_stop_progress(
            status="running",
            phase="validate",
            message="validating active job before stop",
            job_id=None,
            started_at=stop_started_at,
            finished_at=None,
        )

        try:
            with self._lock:
                self._require_command("squeue")
                self._require_command("scancel")

                state = self._load_state()
                active_raw = state.get("active_job")
                if not isinstance(active_raw, dict):
                    raise ControlPlaneError(
                        message="no active job",
                        code=21,
                        http_status=404,
                    )

                active = ActiveJob.from_dict(active_raw)
                self._set_stop_progress(
                    status="running",
                    phase="validate",
                    message=f"loaded active job {active.job_id}",
                    job_id=active.job_id,
                    started_at=stop_started_at,
                    finished_at=None,
                )

                slurm_user = self._slurm_user()
                status = self._slurm_job_status(active.job_id)
                if status is not None:
                    self._set_stop_progress(
                        status="running",
                        phase="cancel",
                        message=f"sending scancel for job {active.job_id} (status={status})",
                        job_id=active.job_id,
                        started_at=stop_started_at,
                        finished_at=None,
                    )
                    cancel = self._run(
                        ["scancel", "-u", slurm_user, active.job_id],
                        timeout_seconds=60,
                    )
                    if cancel.returncode != 0:
                        stderr = cancel.stderr.strip()
                        if "Invalid job id specified" not in stderr:
                            raise ControlPlaneError(
                                message=f"failed to stop job {active.job_id}",
                                code=22,
                                http_status=500,
                                details={
                                    "stderr": _truncate_text(stderr),
                                    "stdout": _truncate_text(cancel.stdout.strip()),
                                },
                            )

                if not block:
                    if status is None:
                        self._archive_active_job(state, reason=reason)
                        self._save_state(state)
                        final_status = "not_found"
                    else:
                        active_raw["stop_requested_at"] = _utc_now_iso()
                        state["active_job"] = active_raw
                        self._save_state(state)
                        final_status = "cancelling"

                    finished_at = _utc_now_iso()
                    self._set_stop_progress(
                        status="succeeded",
                        phase="done",
                        message=f"stop requested for job {active.job_id} (final_status={final_status})",
                        job_id=active.job_id,
                        started_at=stop_started_at,
                        finished_at=finished_at,
                    )
                    return CommandResult(
                        code=0,
                        message=f"stop requested for job {active.job_id}",
                        data={
                            "job_id": active.job_id,
                            "previous_status": status or "not_found",
                            "final_status": final_status,
                            "waited_seconds": 0.0,
                            "slurm_user": slurm_user,
                            "blocked": False,
                            "stop_requested_at": active_raw.get("stop_requested_at"),
                        },
                    )

                self._set_stop_progress(
                    status="running",
                    phase="wait_slurm",
                    message=f"waiting for job {active.job_id} to disappear from squeue",
                    job_id=active.job_id,
                    started_at=stop_started_at,
                    finished_at=None,
                )

                # Block until the job disappears from Slurm so callers know stop is complete.
                wait_started_at = time.monotonic()
                deadline = wait_started_at + DEFAULT_STOP_WAIT_TIMEOUT_SECONDS
                last_seen_status: str | None = None
                while True:
                    current_status = self._slurm_job_status(active.job_id)
                    if current_status is None:
                        break
                    if current_status != last_seen_status:
                        last_seen_status = current_status
                        self._set_stop_progress(
                            status="running",
                            phase="wait_slurm",
                            message=f"job {active.job_id} still active (status={current_status})",
                            job_id=active.job_id,
                            started_at=stop_started_at,
                            finished_at=None,
                        )
                    if time.monotonic() >= deadline:
                        raise ControlPlaneError(
                            message=(
                                f"timed out waiting for job {active.job_id} to disappear "
                                f"(last_status={current_status})"
                            ),
                            code=23,
                            http_status=504,
                            details={
                                "job_id": active.job_id,
                                "last_status": current_status,
                                "wait_timeout_seconds": DEFAULT_STOP_WAIT_TIMEOUT_SECONDS,
                                "slurm_user": slurm_user,
                            },
                        )
                    time.sleep(DEFAULT_STOP_POLL_INTERVAL_SECONDS)

                waited_seconds = round(time.monotonic() - wait_started_at, 3)
                self._archive_active_job(state, reason=reason)
                self._save_state(state)

                finished_at = _utc_now_iso()
                self._set_stop_progress(
                    status="succeeded",
                    phase="done",
                    message=f"stopped job {active.job_id}",
                    job_id=active.job_id,
                    started_at=stop_started_at,
                    finished_at=finished_at,
                )

                return CommandResult(
                    code=0,
                    message=f"stopped job {active.job_id}",
                    data={
                        "job_id": active.job_id,
                        "previous_status": status or "not_found",
                        "final_status": "not_found",
                        "waited_seconds": waited_seconds,
                        "slurm_user": slurm_user,
                        "blocked": True,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            self._set_stop_progress(
                status="failed",
                phase="failed",
                message=str(exc),
                job_id=self._current_stop_job_id(),
                started_at=stop_started_at,
                finished_at=_utc_now_iso(),
            )
            raise

    def stop_poll(self) -> CommandResult:
        with self._lock:
            state = self._load_state()
            active_raw = state.get("active_job")
            if not isinstance(active_raw, dict):
                return CommandResult(
                    code=0,
                    message="no active job",
                    data={
                        "done": True,
                        "job_id": None,
                        "job_status": "not_found",
                        "stop_requested_at": None,
                    },
                )

            active = ActiveJob.from_dict(active_raw)
            stop_requested_at = active_raw.get("stop_requested_at")
            status = self._slurm_job_status(active.job_id)
            if status is None:
                reason = (
                    "stopped"
                    if isinstance(stop_requested_at, str) and stop_requested_at
                    else "finished"
                )
                self._archive_active_job(state, reason=reason)
                self._save_state(state)
                return CommandResult(
                    code=0,
                    message=f"job {active.job_id} is no longer in squeue",
                    data={
                        "done": True,
                        "job_id": active.job_id,
                        "job_status": "not_found",
                        "reason": reason,
                        "stop_requested_at": stop_requested_at,
                    },
                )

            return CommandResult(
                code=0,
                message=f"job {active.job_id} is still active",
                data={
                    "done": False,
                    "job_id": active.job_id,
                    "job_status": status,
                    "stop_requested_at": stop_requested_at,
                },
            )

    def start_status(self) -> CommandResult:
        with self._start_progress_lock:
            progress = dict(self._start_progress)
        return CommandResult(code=0, message="start status", data=progress)

    def stop_status(self) -> CommandResult:
        with self._stop_progress_lock:
            progress = dict(self._stop_progress)
        return CommandResult(code=0, message="stop status", data=progress)

    def logs(self, *, lines: int = 200) -> CommandResult:
        with self._lock:
            if lines <= 0:
                raise ControlPlaneError(
                    message="lines must be > 0",
                    code=31,
                    http_status=400,
                )

            state = self._load_state()
            active_raw = state.get("active_job")
            if not isinstance(active_raw, dict):
                raise ControlPlaneError(
                    message="no active job",
                    code=32,
                    http_status=404,
                )
            active = ActiveJob.from_dict(active_raw)

            logs = {
                "slurm_out": _tail_file(Path(active.slurm_out_log), lines=lines),
                "slurm_err": _tail_file(Path(active.slurm_err_log), lines=lines),
                "jaeger": _tail_file(Path(active.jaeger_log), lines=lines),
                "vllm": _tail_file(Path(active.vllm_log), lines=lines),
            }

            status = self._slurm_job_status(active.job_id)
            if status is None:
                self._archive_active_job(state, reason="finished")
                self._save_state(state)

            return CommandResult(
                code=0,
                message="log snapshot collected",
                data={
                    "job_id": active.job_id,
                    "job_status": status or "not_found",
                    "lines": lines,
                    "logs": logs,
                },
            )

    def test(self) -> CommandResult:
        test_started_at = _utc_now_iso()
        self._set_test_progress(
            status="running",
            phase="validate",
            message="validating active job before smoke tests",
            job_id=None,
            started_at=test_started_at,
            finished_at=None,
        )

        try:
            with self._lock:
                self._require_command("apptainer")
                self._ensure_sif_files()

                state = self._load_state()
                active_raw = state.get("active_job")
                if not isinstance(active_raw, dict):
                    raise ControlPlaneError(
                        message="no active job",
                        code=41,
                        http_status=404,
                    )

                active = ActiveJob.from_dict(active_raw)
                self._set_test_progress(
                    status="running",
                    phase="validate",
                    message=f"validating job {active.job_id}",
                    job_id=active.job_id,
                    started_at=test_started_at,
                    finished_at=None,
                )

                job_status = self._slurm_job_status(active.job_id)
                if job_status is None:
                    self._archive_active_job(state, reason="finished")
                    self._save_state(state)
                    raise ControlPlaneError(
                        message="active job is no longer running",
                        code=42,
                        http_status=409,
                    )
                if job_status not in {"RUNNING", "COMPLETING"}:
                    raise ControlPlaneError(
                        message=f"active job status is {job_status}; wait until RUNNING before testing",
                        code=43,
                        http_status=409,
                    )

                model_key = active.model
                model_spec = self._cfg.models.get(model_key)
                if model_spec is None:
                    raise ControlPlaneError(
                        message=f"model config '{model_key}' is missing",
                        code=44,
                        http_status=500,
                    )

                shared_env = {
                    "PYTHONNOUSERSITE": "1",
                    "AITER_JIT_DIR": self._cfg.env.get(
                        "AITER_JIT_DIR",
                        f"{os.environ.get('TMPDIR', '/tmp')}/vllm-aiter-jit-{os.environ.get('USER', 'user')}",
                    ),
                    "XDG_CACHE_HOME": self._cfg.env.get(
                        "XDG_CACHE_HOME",
                        f"{os.environ.get('TMPDIR', '/tmp')}/vllm-runtime-{os.environ.get('USER', 'user')}/xdg-cache",
                    ),
                    "VLLM_CACHE_ROOT": self._cfg.env.get(
                        "VLLM_CACHE_ROOT",
                        f"{os.environ.get('TMPDIR', '/tmp')}/vllm-runtime-{os.environ.get('USER', 'user')}/xdg-cache/vllm",
                    ),
                    "VLLM_BASE_URL": f"http://127.0.0.1:{active.service_port}",
                    "VLLM_TEST_MODEL_NAME": model_spec.served_model_name,
                    "VLLM_STARTUP_TIMEOUT": str(self._cfg.startup_timeout),
                }

                otel_env = {
                    **shared_env,
                    "OTEL_SERVICE_NAME": self._cfg.env.get("OTEL_TEST_SERVICE_NAME", "otel-smoke-client"),
                    "OTEL_EXPORTER_OTLP_TRACES_INSECURE": "true",
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": f"grpc://127.0.0.1:{active.jaeger_otlp_port}",
                }
                force_env = {
                    **shared_env,
                    "VLLM_FORCE_SEQUENCE_TEST_TEXT": self._cfg.env.get(
                        "VLLM_FORCE_SEQUENCE_TEST_TEXT", "FORCE_SEQ_SMOKE_OK"
                    ),
                }

                self._set_test_progress(
                    status="running",
                    phase="preflight",
                    message="checking vLLM endpoint reachability before smoke tests",
                    job_id=active.job_id,
                    started_at=test_started_at,
                    finished_at=None,
                )
                self._wait_for_vllm_preflight_or_raise(
                    base_url=shared_env["VLLM_BASE_URL"],
                    timeout_seconds=DEFAULT_TEST_PREFLIGHT_TIMEOUT_SECONDS,
                )

                self._set_test_progress(
                    status="running",
                    phase="otel",
                    message="running OTEL smoke test",
                    job_id=active.job_id,
                    started_at=test_started_at,
                    finished_at=None,
                )
                otel_cmd = self._apptainer_exec_python(
                    script_path=self._cfg.repo_root / "docker/tests/dummy_otel_client.py",
                    extra_env=otel_env,
                )
                otel_result = self._run_checked(otel_cmd, timeout_seconds=30 * 60)

                self._set_test_progress(
                    status="running",
                    phase="force_sequence",
                    message="running force-sequence smoke test",
                    job_id=active.job_id,
                    started_at=test_started_at,
                    finished_at=None,
                )
                force_cmd = self._apptainer_exec_python(
                    script_path=self._cfg.repo_root / "docker/tests/force_sequence_smoke_test.py",
                    extra_env=force_env,
                )
                force_result = self._run_checked(force_cmd, timeout_seconds=30 * 60)

                finished_at = _utc_now_iso()
                self._set_test_progress(
                    status="passed",
                    phase="done",
                    message=f"both smoke tests passed for job {active.job_id}",
                    job_id=active.job_id,
                    started_at=test_started_at,
                    finished_at=finished_at,
                )
                return CommandResult(
                    code=0,
                    message="both smoke tests passed",
                    data={
                        "job_id": active.job_id,
                        "otel_stdout": _truncate_text(otel_result.stdout.strip(), max_chars=8000),
                        "otel_stderr": _truncate_text(otel_result.stderr.strip(), max_chars=4000),
                        "force_stdout": _truncate_text(force_result.stdout.strip(), max_chars=8000),
                        "force_stderr": _truncate_text(force_result.stderr.strip(), max_chars=4000),
                    },
                )
        except Exception as exc:  # noqa: BLE001
            self._set_test_progress(
                status="failed",
                phase="failed",
                message=str(exc),
                job_id=self._current_test_job_id(),
                started_at=test_started_at,
                finished_at=_utc_now_iso(),
            )
            raise

    def test_status(self) -> CommandResult:
        with self._test_progress_lock:
            progress = dict(self._test_progress)
        return CommandResult(code=0, message="test status", data=progress)

    def up(self) -> CommandResult:
        with self._lock:
            readiness = self._collect_readiness_snapshot()
        message = "services are up" if readiness["ready"] else "services are not ready"
        return CommandResult(code=0, message=message, data=readiness)

    def wait_up(
        self,
        *,
        timeout_seconds: int | None = None,
        poll_interval_seconds: float = DEFAULT_WAIT_UP_POLL_INTERVAL_SECONDS,
    ) -> CommandResult:
        if timeout_seconds is None:
            timeout_seconds = self._cfg.startup_timeout
        if timeout_seconds <= 0:
            raise ControlPlaneError(
                message="timeout_seconds must be > 0",
                code=61,
                http_status=400,
            )
        if poll_interval_seconds <= 0:
            raise ControlPlaneError(
                message="poll_interval_seconds must be > 0",
                code=62,
                http_status=400,
            )

        snapshot = self._wait_for_services_up(
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            expected_job_id=None,
        )
        return CommandResult(code=0, message="services are up", data=snapshot)

    def status(self) -> CommandResult:
        with self._lock:
            state = self._load_state()
            state_changed, active, job_status = self._refresh_active_job(state)
            if state_changed:
                self._save_state(state)

            return CommandResult(
                code=0,
                message="status",
                data={
                    "server": {
                        "host": self._cfg.host,
                        "port": self._cfg.port,
                        "config_path": str(self._config_path),
                    },
                    "active_job": active.to_dict() if active is not None else None,
                    "active_job_status": job_status,
                    "allowed_partitions": {
                        key: {
                            "gpus_per_node": value.gpus_per_node,
                            "gpu_memory_gb": value.gpu_memory_gb,
                            "total_vram_gb": value.total_vram_gb,
                            "max_time": value.max_time,
                        }
                        for key, value in self._cfg.partitions.items()
                    },
                    "allowed_models": {
                        key: {
                            "vllm_model_name": value.vllm_model_name,
                            "served_model_name": value.served_model_name,
                            "weight_vram_gb": value.weight_vram_gb,
                            "extra_args": value.extra_args,
                        }
                        for key, value in self._cfg.models.items()
                    },
                },
            )

    def _command_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self._cfg.env)
        return env

    def _require_command(self, command_name: str) -> None:
        if shutil.which(command_name) is None:
            raise ControlPlaneError(
                message=f"required command not found: {command_name}",
                code=90,
                http_status=500,
            )

    def _ensure_sif_files(self) -> None:
        missing: list[str] = []
        if not self._cfg.jaeger_sif.exists():
            missing.append(str(self._cfg.jaeger_sif))
        if not self._cfg.vllm_sif.exists():
            missing.append(str(self._cfg.vllm_sif))
        if missing:
            raise ControlPlaneError(
                message="missing SIF files; run pull first",
                code=91,
                http_status=400,
                details={"missing": missing},
            )

    def _load_state(self) -> dict[str, Any]:
        if not self._cfg.state_file.exists():
            return {"active_job": None, "history": []}
        try:
            raw = json.loads(self._cfg.state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ControlPlaneError(
                message=f"state file is not valid JSON: {self._cfg.state_file}",
                code=92,
                http_status=500,
                details={"error": str(exc)},
            ) from exc

        if not isinstance(raw, dict):
            return {"active_job": None, "history": []}
        if "history" not in raw or not isinstance(raw["history"], list):
            raw["history"] = []
        if "active_job" not in raw:
            raw["active_job"] = None
        return raw

    def _save_state(self, state: dict[str, Any]) -> None:
        self._cfg.state_file.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _archive_active_job(self, state: dict[str, Any], *, reason: str) -> None:
        active_raw = state.get("active_job")
        if not isinstance(active_raw, dict):
            state["active_job"] = None
            return
        snapshot = dict(active_raw)
        snapshot["ended_at"] = _utc_now_iso()
        snapshot["end_reason"] = reason
        history = state.setdefault("history", [])
        if isinstance(history, list):
            history.append(snapshot)
        state["active_job"] = None

    def _archive_previous_artifacts(self) -> None:
        """Move stale run/log files into run/old and logs/old on server startup."""
        run_old_dir = self._cfg.run_dir / "old"
        logs_old_dir = self._cfg.log_dir / "old"
        run_old_dir.mkdir(parents=True, exist_ok=True)
        logs_old_dir.mkdir(parents=True, exist_ok=True)

        protected_paths: set[Path] = {self._cfg.state_file.resolve()}
        state: dict[str, Any]
        try:
            state = self._load_state()
        except ControlPlaneError:
            state = {"active_job": None}

        active_raw = state.get("active_job")
        if isinstance(active_raw, dict):
            for key in ("sbatch_script", "slurm_out_log", "slurm_err_log", "jaeger_log", "vllm_log"):
                raw_path = active_raw.get(key)
                if isinstance(raw_path, str) and raw_path:
                    protected_paths.add(Path(raw_path).expanduser().resolve())

        self._move_dir_contents_to_old(
            base_dir=self._cfg.run_dir,
            old_dir=run_old_dir,
            protected_paths=protected_paths,
        )
        self._move_dir_contents_to_old(
            base_dir=self._cfg.log_dir,
            old_dir=logs_old_dir,
            protected_paths=protected_paths,
        )

    def _move_dir_contents_to_old(
        self,
        *,
        base_dir: Path,
        old_dir: Path,
        protected_paths: set[Path],
    ) -> None:
        for item in base_dir.iterdir():
            if item.name == "old":
                continue
            try:
                resolved_item = item.resolve()
            except FileNotFoundError:
                continue
            if resolved_item in protected_paths:
                continue

            target = old_dir / item.name
            if target.exists():
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                suffix = 1
                while True:
                    candidate = old_dir / f"{item.name}.{timestamp}.{suffix}"
                    if not candidate.exists():
                        target = candidate
                        break
                    suffix += 1
            shutil.move(str(item), str(target))

    def _refresh_active_job(
        self, state: dict[str, Any]
    ) -> tuple[bool, ActiveJob | None, str | None]:
        active_raw = state.get("active_job")
        if not isinstance(active_raw, dict):
            return (False, None, None)

        active = ActiveJob.from_dict(active_raw)
        job_status = self._slurm_job_status(active.job_id)
        if job_status is None:
            stop_requested_at = active_raw.get("stop_requested_at")
            reason = (
                "stopped"
                if isinstance(stop_requested_at, str) and stop_requested_at
                else "finished"
            )
            self._archive_active_job(state, reason=reason)
            return (True, None, None)
        return (False, active, job_status)

    def _slurm_job_status(self, job_id: str) -> str | None:
        self._require_command("squeue")
        slurm_user = self._slurm_user()
        result = self._run(
            [
                "squeue",
                "-u",
                slurm_user,
                "--noheader",
                "--jobs",
                str(job_id),
                "--format",
                "%T",
            ],
            timeout_seconds=30,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "Invalid job id specified" in stderr:
                return None
            raise ControlPlaneError(
                message=f"failed to query slurm job status for {job_id}",
                code=93,
                http_status=500,
                details={"stderr": _truncate_text(stderr), "stdout": _truncate_text(result.stdout.strip())},
            )

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        return lines[0]

    def _slurm_user(self) -> str:
        user = (
            self._cfg.env.get("USER")
            or self._cfg.env.get("LOGNAME")
            or os.environ.get("USER")
            or os.environ.get("LOGNAME")
        )
        if user:
            return user
        raise ControlPlaneError(
            message="cannot determine Slurm user; set USER in server environment",
            code=97,
            http_status=500,
        )

    def _ensure_port_available_on_login(self, port: int) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise ControlPlaneError(
                    message=(
                        f"service_port {port} is already in use on login node; "
                        "choose a different cluster.service_port"
                    ),
                    code=15,
                    http_status=409,
                    details={"service_port": port, "error": str(exc)},
                ) from exc

    def _wait_for_vllm_preflight_or_raise(self, *, base_url: str, timeout_seconds: int) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error = "vLLM preflight not started"
        models_url = f"{base_url.rstrip('/')}/v1/models"

        while time.monotonic() < deadline:
            try:
                req = urlrequest.Request(models_url, method="GET")
                with urlrequest.urlopen(req, timeout=5) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    if response.status == 200:
                        return
                    last_error = f"HTTP {response.status}: {_truncate_text(body, max_chars=200)}"
            except urlerror.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {_truncate_text(body, max_chars=200)}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            time.sleep(DEFAULT_TEST_PREFLIGHT_POLL_INTERVAL_SECONDS)

        raise ControlPlaneError(
            message=(
                f"vLLM preflight failed at {models_url} after {timeout_seconds}s; "
                f"last_error={last_error}"
            ),
            code=45,
            http_status=409,
            details={"models_url": models_url, "timeout_seconds": timeout_seconds, "last_error": last_error},
        )

    def _collect_readiness_snapshot(self) -> dict[str, Any]:
        state = self._load_state()
        state_changed, active, job_status = self._refresh_active_job(state)
        if state_changed:
            self._save_state(state)

        vllm_url = f"http://127.0.0.1:{self._cfg.service_port}/v1/models"
        jaeger_url = f"http://127.0.0.1:{self._cfg.jaeger_ui_port}"

        vllm_probe = self._probe_http_url(vllm_url)
        jaeger_probe = self._probe_http_url(jaeger_url)

        active_job_payload = active.to_dict() if active is not None else None
        active_running = job_status in {"RUNNING", "COMPLETING"}
        ready = bool(active_running and vllm_probe["ok"] and jaeger_probe["ok"])

        return {
            "ready": ready,
            "active_job": active_job_payload,
            "active_job_status": job_status,
            "vllm": vllm_probe,
            "jaeger": jaeger_probe,
        }

    def _wait_for_services_up(
        self,
        *,
        timeout_seconds: int,
        poll_interval_seconds: float,
        expected_job_id: str | None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + timeout_seconds
        last_snapshot: dict[str, Any] | None = None

        while True:
            with self._lock:
                snapshot = self._collect_readiness_snapshot()
            last_snapshot = snapshot
            if progress_callback is not None:
                progress_callback(snapshot)

            if expected_job_id is not None:
                active_job = snapshot.get("active_job")
                active_job_id = active_job.get("job_id") if isinstance(active_job, dict) else None
                if active_job_id != expected_job_id:
                    raise ControlPlaneError(
                        message=(
                            "active job changed while waiting for readiness "
                            f"(expected {expected_job_id}, got {active_job_id})"
                        ),
                        code=64,
                        http_status=409,
                        details={"expected_job_id": expected_job_id, "last_snapshot": snapshot},
                    )

            if snapshot["ready"]:
                waited_seconds = round(time.monotonic() - started, 3)
                snapshot["waited_seconds"] = waited_seconds
                return snapshot

            if time.monotonic() >= deadline:
                raise ControlPlaneError(
                    message=f"timed out waiting for services to be up after {timeout_seconds}s",
                    code=63,
                    http_status=408,
                    details={
                        "timeout_seconds": timeout_seconds,
                        "poll_interval_seconds": poll_interval_seconds,
                        "last_snapshot": last_snapshot,
                    },
                )

            time.sleep(poll_interval_seconds)

    def _probe_http_url(self, url: str) -> dict[str, Any]:
        try:
            req = urlrequest.Request(url, method="GET")
            with urlrequest.urlopen(req, timeout=5) as response:
                return {
                    "ok": True,
                    "url": url,
                    "http_status": int(response.status),
                    "error": None,
                }
        except urlerror.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return {
                "ok": False,
                "url": url,
                "http_status": int(exc.code),
                "error": _truncate_text(body.strip(), max_chars=200),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "url": url,
                "http_status": None,
                "error": str(exc),
            }

    def _write_sbatch_script(self, *, partition_spec: PartitionSpec, model_spec: ModelSpec) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_partition = _safe_token(partition_spec.name)
        safe_model = _safe_token(model_spec.name)
        script_path = self._cfg.run_dir / f"sbatch-{timestamp}-{safe_partition}-{safe_model}.sh"

        visible_devices = ",".join(str(idx) for idx in range(partition_spec.gpus_per_node))
        slurm_out = self._cfg.log_dir / "slurm.%j.out"
        slurm_err = self._cfg.log_dir / "slurm.%j.err"

        tmp_root = os.environ.get("TMPDIR", "/tmp")
        user_name = os.environ.get("USER", "user")
        aiter_jit_dir = self._cfg.env.get("AITER_JIT_DIR", f"{tmp_root}/vllm-aiter-jit-{user_name}")
        runtime_root = self._cfg.env.get("VLLM_RUNTIME_ROOT", f"{tmp_root}/vllm-runtime-{user_name}")
        xdg_cache_home = self._cfg.env.get("XDG_CACHE_HOME", f"{runtime_root}/xdg-cache")
        vllm_cache_root = self._cfg.env.get("VLLM_CACHE_ROOT", f"{xdg_cache_home}/vllm")

        ssh_options = " ".join(shlex.quote(opt) for opt in self._cfg.ssh_options)
        extra_args = " ".join(shlex.quote(arg) for arg in model_spec.extra_args)

        script = textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            #SBATCH --job-name={_safe_token(self._cfg.job_name)}
            #SBATCH --output={slurm_out}
            #SBATCH --error={slurm_err}
            #SBATCH --nodes={self._cfg.job_nodes}
            #SBATCH --time={partition_spec.max_time}
            #SBATCH --partition={partition_spec.name}

            set -euo pipefail

            echo "Job ${{SLURM_JOB_ID}} starting on $(hostname) at $(date)"

            LOGIN_HOST={shlex.quote(self._cfg.login_host)}
            VLLM_SERVICE_PORT={self._cfg.service_port}
            JAEGER_OTLP_PORT={self._cfg.jaeger_otlp_port}
            JAEGER_UI_PORT={self._cfg.jaeger_ui_port}

            JAEGER_SIF={shlex.quote(str(self._cfg.jaeger_sif))}
            VLLM_SIF={shlex.quote(str(self._cfg.vllm_sif))}

            VLLM_MODEL_NAME={shlex.quote(model_spec.vllm_model_name)}
            VLLM_SERVED_MODEL_NAME={shlex.quote(model_spec.served_model_name)}
            VLLM_TENSOR_PARALLEL_SIZE={partition_spec.gpus_per_node}
            VLLM_VISIBLE_DEVICES={shlex.quote(visible_devices)}

            VLLM_APPTAINER_HOME={shlex.quote(self._cfg.env.get('VLLM_APPTAINER_HOME', ''))}
            HF_HOME={shlex.quote(self._cfg.env.get('HF_HOME', ''))}
            HF_HUB_CACHE={shlex.quote(self._cfg.env.get('HF_HUB_CACHE', ''))}
            HF_TOKEN="${{HF_TOKEN:-}}"

            AITER_JIT_DIR={shlex.quote(aiter_jit_dir)}
            XDG_CACHE_HOME={shlex.quote(xdg_cache_home)}
            VLLM_CACHE_ROOT={shlex.quote(vllm_cache_root)}

            OTEL_SERVICE_NAME={shlex.quote(self._cfg.env.get('OTEL_SERVICE_NAME', 'vllm-server'))}
            OTEL_EXPORTER_OTLP_TRACES_INSECURE={shlex.quote(self._cfg.env.get('OTEL_EXPORTER_OTLP_TRACES_INSECURE', 'true'))}
            OTEL_EXPORTER_OTLP_TRACES_ENDPOINT={shlex.quote(f'grpc://127.0.0.1:{self._cfg.jaeger_otlp_port}')}
            VLLM_COLLECT_DETAILED_TRACES={shlex.quote(self._cfg.env.get('VLLM_COLLECT_DETAILED_TRACES', 'all'))}
            VLLM_LOGITS_PROCESSORS={shlex.quote(self._cfg.env.get('VLLM_LOGITS_PROCESSORS', 'forceSeq.force_sequence_logits_processor:ForceSequenceAdapter'))}
            VLLM_FORCE_SEQ_TRUST_REMOTE_CODE={shlex.quote(self._cfg.env.get('VLLM_FORCE_SEQ_TRUST_REMOTE_CODE', 'false'))}

            JOB_LOG_DIR={shlex.quote(str(self._cfg.log_dir))}
            mkdir -p "${{JOB_LOG_DIR}}" "${{AITER_JIT_DIR}}" "${{XDG_CACHE_HOME}}" "${{VLLM_CACHE_ROOT}}"

            JAEGER_LOG="${{JOB_LOG_DIR}}/jaeger.${{SLURM_JOB_ID}}.log"
            VLLM_LOG="${{JOB_LOG_DIR}}/vllm.${{SLURM_JOB_ID}}.log"

            SSH_OPTIONS=({ssh_options})
            VLLM_EXTRA_ARGS=({extra_args})

            TUNNEL_PIDS=()
            start_tunnel() {{
              local remote_port="$1"
              local local_port="$2"
              ssh "${{SSH_OPTIONS[@]}}" -N -R "${{remote_port}}:127.0.0.1:${{local_port}}" "${{LOGIN_HOST}}" &
              TUNNEL_PIDS+=("$!")
            }}

            APPTAINER_HOME_ARGS=()
            if [[ -n "${{VLLM_APPTAINER_HOME}}" ]]; then
              mkdir -p "${{VLLM_APPTAINER_HOME}}"
              APPTAINER_HOME_ARGS=(-H "${{VLLM_APPTAINER_HOME}}")
            fi

            BIND_ARGS=()
            if [[ -n "${{HF_HOME}}" ]]; then
              BIND_ARGS+=(--bind "${{HF_HOME}}:${{HF_HOME}}")
            fi
            if [[ -n "${{HF_HUB_CACHE}}" ]]; then
              BIND_ARGS+=(--bind "${{HF_HUB_CACHE}}:${{HF_HUB_CACHE}}")
            fi

            cleanup() {{
              set +e
              if [[ -n "${{VLLM_PID:-}}" ]]; then kill "${{VLLM_PID}}" >/dev/null 2>&1 || true; fi
              if [[ -n "${{JAEGER_PID:-}}" ]]; then kill "${{JAEGER_PID}}" >/dev/null 2>&1 || true; fi
              for pid in "${{TUNNEL_PIDS[@]}}"; do
                kill "${{pid}}" >/dev/null 2>&1 || true
              done
            }}
            trap cleanup EXIT INT TERM

            # Reverse tunnels from compute node to login node.
            start_tunnel "${{VLLM_SERVICE_PORT}}" "${{VLLM_SERVICE_PORT}}"
            start_tunnel "${{JAEGER_OTLP_PORT}}" "${{JAEGER_OTLP_PORT}}"
            start_tunnel "${{JAEGER_UI_PORT}}" "${{JAEGER_UI_PORT}}"

            apptainer run \
              --cleanenv \
              "${{APPTAINER_HOME_ARGS[@]}}" \
              --env COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
              "${{JAEGER_SIF}}" \
              >"${{JAEGER_LOG}}" 2>&1 &
            JAEGER_PID=$!

            VLLM_CMD=(
              /opt/vllm-plugins/vllm_entrypoint.sh
              --model "${{VLLM_MODEL_NAME}}"
              --served-model-name "${{VLLM_SERVED_MODEL_NAME}}"
              --port "${{VLLM_SERVICE_PORT}}"
              --tensor-parallel-size "${{VLLM_TENSOR_PARALLEL_SIZE}}"
              --otlp-traces-endpoint "${{OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}}"
              --collect-detailed-traces "${{VLLM_COLLECT_DETAILED_TRACES}}"
              --enable-prompt-tokens-details
              --logits-processors "${{VLLM_LOGITS_PROCESSORS}}"
            )
            if [[ ${{#VLLM_EXTRA_ARGS[@]}} -gt 0 ]]; then
              VLLM_CMD+=("${{VLLM_EXTRA_ARGS[@]}}")
            fi

            apptainer exec \
              --rocm \
              --cleanenv \
              "${{APPTAINER_HOME_ARGS[@]}}" \
              "${{BIND_ARGS[@]}}" \
              --env PYTHONNOUSERSITE=1 \
              --env AITER_JIT_DIR="${{AITER_JIT_DIR}}" \
              --env XDG_CACHE_HOME="${{XDG_CACHE_HOME}}" \
              --env VLLM_CACHE_ROOT="${{VLLM_CACHE_ROOT}}" \
              --env HF_HOME="${{HF_HOME}}" \
              --env HF_HUB_CACHE="${{HF_HUB_CACHE}}" \
              --env HF_TOKEN="${{HF_TOKEN}}" \
              --env OTEL_SERVICE_NAME="${{OTEL_SERVICE_NAME}}" \
              --env OTEL_EXPORTER_OTLP_TRACES_INSECURE="${{OTEL_EXPORTER_OTLP_TRACES_INSECURE}}" \
              --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${{OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}}" \
              --env HIP_VISIBLE_DEVICES="${{VLLM_VISIBLE_DEVICES}}" \
              --env ROCR_VISIBLE_DEVICES="${{VLLM_VISIBLE_DEVICES}}" \
              --env VLLM_MODEL_NAME="${{VLLM_MODEL_NAME}}" \
              --env VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${{VLLM_FORCE_SEQ_TRUST_REMOTE_CODE}}" \
              "${{VLLM_SIF}}" \
              "${{VLLM_CMD[@]}}" \
              >"${{VLLM_LOG}}" 2>&1 &
            VLLM_PID=$!

            wait -n "${{JAEGER_PID}}" "${{VLLM_PID}}"
            EXIT_CODE=$?
            echo "One service exited with code ${{EXIT_CODE}} at $(date)."
            exit "${{EXIT_CODE}}"
            """
        )

        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o750)
        return script_path

    def _apptainer_exec_python(self, *, script_path: Path, extra_env: dict[str, str]) -> list[str]:
        cmd = [
            "apptainer",
            "exec",
            "--cleanenv",
            "--bind",
            f"{self._cfg.repo_root}:/workspace:ro",
        ]

        for key, value in extra_env.items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.extend(
            [
                str(self._cfg.vllm_sif),
                "python3",
                f"/workspace/{script_path.relative_to(self._cfg.repo_root)}",
            ]
        )
        return cmd

    def _run(
        self,
        command: list[str],
        *,
        timeout_seconds: int,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                cwd=self._cfg.repo_root,
                env=self._command_env(),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ControlPlaneError(
                message=f"required command not found: {command[0]}",
                code=96,
                http_status=500,
            ) from exc

    def _run_checked(
        self,
        command: list[str],
        *,
        timeout_seconds: int,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self._run(command, timeout_seconds=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise ControlPlaneError(
                message=f"command timed out: {shlex.join(command)}",
                code=94,
                http_status=500,
                details={"timeout_seconds": timeout_seconds},
            ) from exc

        if result.returncode != 0:
            raise ControlPlaneError(
                message=f"command failed: {shlex.join(command)}",
                code=95,
                http_status=500,
                details={
                    "returncode": result.returncode,
                    "stdout": _truncate_text(result.stdout.strip()),
                    "stderr": _truncate_text(result.stderr.strip()),
                },
            )
        return result

    def _set_start_progress(
        self,
        *,
        status: str,
        phase: str | None,
        message: str,
        job_id: str | None,
        started_at: str | None,
        finished_at: str | None,
    ) -> None:
        with self._start_progress_lock:
            self._start_progress = {
                "status": status,
                "phase": phase,
                "message": message,
                "job_id": job_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "updated_at": _utc_now_iso(),
            }

    def _current_start_job_id(self) -> str | None:
        with self._start_progress_lock:
            value = self._start_progress.get("job_id")
        if isinstance(value, str) and value:
            return value
        return None

    def _set_stop_progress(
        self,
        *,
        status: str,
        phase: str | None,
        message: str,
        job_id: str | None,
        started_at: str | None,
        finished_at: str | None,
    ) -> None:
        with self._stop_progress_lock:
            self._stop_progress = {
                "status": status,
                "phase": phase,
                "message": message,
                "job_id": job_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "updated_at": _utc_now_iso(),
            }

    def _current_stop_job_id(self) -> str | None:
        with self._stop_progress_lock:
            value = self._stop_progress.get("job_id")
        if isinstance(value, str) and value:
            return value
        return None

    def _set_test_progress(
        self,
        *,
        status: str,
        phase: str | None,
        message: str,
        job_id: str | None,
        started_at: str | None,
        finished_at: str | None,
    ) -> None:
        with self._test_progress_lock:
            self._test_progress = {
                "status": status,
                "phase": phase,
                "message": message,
                "job_id": job_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "updated_at": _utc_now_iso(),
            }

    def _current_test_job_id(self) -> str | None:
        with self._test_progress_lock:
            value = self._test_progress.get("job_id")
        if isinstance(value, str) and value:
            return value
        return None


def load_runtime_config(config_path: Path) -> RuntimeConfig:
    if not config_path.exists():
        raise ControlPlaneError(
            message=f"config file not found: {config_path}",
            code=100,
            http_status=500,
        )

    app_dir = Path(__file__).resolve().parent
    repo_root = app_dir.parent

    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ControlPlaneError(
            message=f"failed to parse config: {config_path}",
            code=101,
            http_status=500,
            details={"error": str(exc)},
        ) from exc

    if not isinstance(raw, dict):
        raise ControlPlaneError(
            message=f"config root must be a table: {config_path}",
            code=102,
            http_status=500,
        )

    server_table = _require_table(raw, "server")
    cluster_table = _require_table(raw, "cluster")

    merged_env = dict(os.environ)

    host = str(server_table.get("host", "0.0.0.0"))
    port = int(server_table.get("port", 23971))

    run_dir = _resolve_path(repo_root, str(cluster_table.get("run_dir", "apptainer/run")))
    log_dir = _resolve_path(repo_root, str(cluster_table.get("log_dir", "apptainer/logs")))
    state_file = _resolve_path(repo_root, str(cluster_table.get("state_file", "apptainer/run/control_state.json")))

    login_host = str(cluster_table.get("login_host", "login"))
    job_name = str(cluster_table.get("job_name", "vllm_job"))
    job_nodes = int(cluster_table.get("job_nodes", 1))

    service_port = int(cluster_table.get("service_port", int(merged_env.get("VLLM_SERVICE_PORT", "11451"))))
    jaeger_otlp_port = int(cluster_table.get("jaeger_otlp_port", 4317))
    jaeger_ui_port = int(cluster_table.get("jaeger_ui_port", 16686))
    startup_timeout = int(cluster_table.get("startup_timeout", int(merged_env.get("VLLM_STARTUP_TIMEOUT", "900"))))

    ssh_options_raw = cluster_table.get(
        "ssh_options",
        [
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
        ],
    )
    if not isinstance(ssh_options_raw, list) or not all(
        isinstance(value, str) for value in ssh_options_raw
    ):
        raise ControlPlaneError(
            message="cluster.ssh_options must be an array of strings",
            code=103,
            http_status=500,
        )
    ssh_options = list(ssh_options_raw)

    partitions_table = raw.get("partition")
    if not isinstance(partitions_table, dict) or not partitions_table:
        raise ControlPlaneError(
            message="config must define [partition.<name>] tables",
            code=104,
            http_status=500,
        )

    partitions: dict[str, PartitionSpec] = {}
    for name, value in partitions_table.items():
        if not isinstance(value, dict):
            raise ControlPlaneError(
                message=f"partition.{name} must be a table",
                code=105,
                http_status=500,
            )
        partitions[name] = PartitionSpec(
            name=name,
            gpus_per_node=int(value["gpus_per_node"]),
            gpu_memory_gb=float(value["gpu_memory_gb"]),
            total_vram_gb=float(value["total_vram_gb"]),
            max_time=str(value.get("max_time", "04:00:00")),
        )

    models_table = raw.get("models")
    if not isinstance(models_table, dict) or not models_table:
        raise ControlPlaneError(
            message="config must define [models.<name>] tables",
            code=106,
            http_status=500,
        )

    models: dict[str, ModelSpec] = {}
    for name, value in models_table.items():
        if not isinstance(value, dict):
            raise ControlPlaneError(
                message=f"models.{name} must be a table",
                code=107,
                http_status=500,
            )

        extra_args_raw = value.get("extra_args", [])
        if not isinstance(extra_args_raw, list) or not all(
            isinstance(item, str) for item in extra_args_raw
        ):
            raise ControlPlaneError(
                message=f"models.{name}.extra_args must be a string array",
                code=108,
                http_status=500,
            )

        models[name] = ModelSpec(
            name=name,
            vllm_model_name=str(value["vllm_model_name"]),
            served_model_name=str(value["served_model_name"]),
            weight_vram_gb=float(value["weight_vram_gb"]),
            extra_args=list(extra_args_raw),
        )

    apptainer_imgs = Path(
        _expand_vars(merged_env.get("APPTAINER_IMGS", f"{Path.home()}/apptainer-images"), merged_env)
    ).expanduser()
    jaeger_image = merged_env.get("JAEGER_IMAGE", DEFAULT_JAEGER_IMAGE)
    vllm_image = merged_env.get("VLLM_IMAGE", DEFAULT_VLLM_IMAGE)

    jaeger_sif_raw = merged_env.get("JAEGER_SIF", str(apptainer_imgs / DEFAULT_JAEGER_SIF_NAME))
    vllm_sif_raw = merged_env.get("VLLM_SIF", str(apptainer_imgs / DEFAULT_VLLM_SIF_NAME))

    jaeger_sif = _resolve_path(repo_root, _expand_vars(jaeger_sif_raw, merged_env))
    vllm_sif = _resolve_path(repo_root, _expand_vars(vllm_sif_raw, merged_env))

    return RuntimeConfig(
        repo_root=repo_root,
        app_dir=app_dir,
        host=host,
        port=port,
        run_dir=run_dir,
        log_dir=log_dir,
        state_file=state_file,
        login_host=login_host,
        job_name=job_name,
        job_nodes=job_nodes,
        service_port=service_port,
        jaeger_otlp_port=jaeger_otlp_port,
        jaeger_ui_port=jaeger_ui_port,
        startup_timeout=startup_timeout,
        ssh_options=ssh_options,
        partitions=partitions,
        models=models,
        env=merged_env,
        apptainer_imgs=apptainer_imgs,
        jaeger_image=jaeger_image,
        vllm_image=vllm_image,
        jaeger_sif=jaeger_sif,
        vllm_sif=vllm_sif,
    )


def command_result_payload(result: CommandResult) -> dict[str, Any]:
    return {
        "ok": result.code == 0,
        "code": result.code,
        "message": result.message,
        "data": result.data,
    }


def error_payload(exc: ControlPlaneError) -> dict[str, Any]:
    return {
        "ok": False,
        "code": exc.code,
        "message": exc.message,
        "details": exc.details,
    }


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _expand_vars(value: str, env: dict[str, str] | None = None) -> str:
    env_map = env or os.environ

    def repl(match: re.Match[str]) -> str:
        key = match.group("brace") or match.group("plain")
        return env_map.get(key, os.environ.get(key, ""))

    pattern = re.compile(r"\$\{(?P<brace>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)")
    return pattern.sub(repl, value)


def _require_table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    table = raw.get(key)
    if not isinstance(table, dict):
        raise ControlPlaneError(
            message=f"config missing [{key}] table",
            code=110,
            http_status=500,
        )
    return table


def _extract_sbatch_job_id(stdout: str) -> str:
    match = re.search(r"Submitted\s+batch\s+job\s+(\d+)", stdout)
    if match is None:
        raise ControlPlaneError(
            message="unable to parse sbatch job id",
            code=111,
            http_status=500,
            details={"sbatch_stdout": _truncate_text(stdout.strip())},
        )
    return match.group(1)


def _truncate_text(text: str, *, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n...<truncated {len(text) - max_chars} chars>"


def _tail_file(path: Path, *, lines: int) -> str:
    if not path.exists():
        return f"(missing log file: {path})"
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        recent = deque(handle, maxlen=lines)
    return "".join(recent)
