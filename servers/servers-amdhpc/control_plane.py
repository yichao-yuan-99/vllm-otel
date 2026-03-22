#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared control-plane logic for the Apptainer HPC server/client workflow."""

from __future__ import annotations

import base64
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

try:
    from .port_profiles import PortProfile as AMDHPCPortProfile, load_port_profiles
except ImportError:  # pragma: no cover
    from port_profiles import PortProfile as AMDHPCPortProfile, load_port_profiles  # type: ignore[no-redef]


DEFAULT_JAEGER_IMAGE = "docker://jaegertracing/all-in-one:1.57"
DEFAULT_VLLM_IMAGE = "docker://yichaoyuan/vllm-openai-otel:v0.16.0-otel-lp-rocm"
DEFAULT_JAEGER_QUERY_PORT = 16686
DEFAULT_JAEGER_OTLP_GRPC_PORT = 4317
DEFAULT_JAEGER_PULL_TIMEOUT_SECONDS = 60 * 60
DEFAULT_VLLM_PULL_TIMEOUT_SECONDS = 2 * 60 * 60
DEFAULT_STOP_WAIT_TIMEOUT_SECONDS = 180
DEFAULT_STOP_POLL_INTERVAL_SECONDS = 2
DEFAULT_WAIT_UP_POLL_INTERVAL_SECONDS = 2
DEFAULT_SQUEUE_MISS_GRACE_SECONDS = 30


def _encode_model_extra_args(extra_args: list[str]) -> str:
    payload = json.dumps(extra_args, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def _infer_sif_name_from_image(image: str) -> str:
    normalized = image.strip()
    if not normalized:
        raise ValueError("image reference must be non-empty")
    if "://" in normalized:
        normalized = normalized.split("://", 1)[1]
    token = normalized.rsplit("/", 1)[-1]
    token = token.replace(":", "-")
    return f"{token}.sif"


def _effective_vllm_extra_args(*, extra_args: list[str], gpus_per_node: int) -> list[str]:
    normalized_args: list[str] = []
    skip_next = False

    for index, arg in enumerate(extra_args):
        if skip_next:
            skip_next = False
            continue

        normalized = arg.strip().lower()
        if normalized == "--trust-remote-code":
            continue
        if normalized.startswith("--trust-remote-code="):
            continue
        if normalized == "--distributed_executor_backend":
            if index + 1 < len(extra_args):
                skip_next = True
            continue
        if normalized.startswith("--distributed_executor_backend="):
            continue
        normalized_args.append(arg)

    normalized_args.append("--trust-remote-code")
    if gpus_per_node > 1:
        normalized_args.extend(["--distributed_executor_backend", "ray"])
    return normalized_args


def _compute_even_group_gpu_split(
    *,
    gpus_per_node: int,
    gpu_memory_gb: float,
    group_size: int,
) -> tuple[int, float, list[str]]:
    if gpus_per_node <= 0:
        raise ValueError(f"gpus_per_node must be > 0 (got {gpus_per_node})")
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0 (got {group_size})")
    if group_size > gpus_per_node:
        raise ValueError(
            f"group_size={group_size} exceeds gpus_per_node={gpus_per_node}; "
            "single-node grouped launch requires group_size <= gpus_per_node"
        )
    if gpus_per_node % group_size != 0:
        raise ValueError(
            f"group_size={group_size} does not evenly divide gpus_per_node={gpus_per_node}; "
            "choose a profile count that evenly divides node GPUs"
        )

    gpus_per_profile = gpus_per_node // group_size
    total_vram_per_profile_gb = gpu_memory_gb * gpus_per_profile
    visible_devices_by_profile = []
    for worker_index in range(group_size):
        start_gpu = worker_index * gpus_per_profile
        gpu_ids = [str(start_gpu + offset) for offset in range(gpus_per_profile)]
        visible_devices_by_profile.append(",".join(gpu_ids))

    return gpus_per_profile, total_vram_per_profile_gb, visible_devices_by_profile


def _normalize_service_extra_env(extra_env: dict[str, str] | None) -> dict[str, str]:
    if not extra_env:
        return {}

    normalized: dict[str, str] = {}
    for raw_key, raw_value in extra_env.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("environment variable key cannot be empty")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(
                f"invalid environment variable key '{key}'; "
                "must match [A-Za-z_][A-Za-z0-9_]*"
            )
        value = str(raw_value)
        if "\x00" in value:
            raise ValueError(f"environment variable '{key}' contains NUL byte")
        normalized[key] = value

    return dict(sorted(normalized.items()))


def _render_apptainer_extra_env_flags(*, extra_env: dict[str, str], indent: str) -> str:
    if not extra_env:
        return ""
    lines = [
        f"{indent}--env {shlex.quote(f'{key}={value}')} \\"
        for key, value in extra_env.items()
    ]
    return "\n".join(lines) + "\n"


def _apply_lmcache_option(
    *,
    extra_env: dict[str, str],
    lmcache_max_local_cpu_size: str | None,
) -> tuple[dict[str, str], bool]:
    normalized = dict(extra_env)
    if lmcache_max_local_cpu_size is None:
        return dict(sorted(normalized.items())), False

    size_value = str(lmcache_max_local_cpu_size).strip()
    if not size_value:
        raise ValueError("lmcache size must be non-empty")

    existing = normalized.get("LMCACHE_MAX_LOCAL_CPU_SIZE")
    if existing is not None and existing != size_value:
        raise ValueError(
            "LMCACHE_MAX_LOCAL_CPU_SIZE is already set in extra_env with a different value"
        )

    normalized["LMCACHE_MAX_LOCAL_CPU_SIZE"] = size_value
    return dict(sorted(normalized.items())), True


def _normalize_local_mode_script(local_mode_script: str | None) -> str | None:
    if local_mode_script is None:
        return None
    normalized = str(local_mode_script).strip()
    if not normalized:
        raise ValueError("local_mode_script must be non-empty")
    if "\x00" in normalized:
        raise ValueError("local_mode_script cannot contain NUL byte")
    return normalized


def _normalize_user_extra_vllm_args(extra_vllm_args: list[str] | None) -> list[str]:
    if extra_vllm_args is None:
        return []
    if not isinstance(extra_vllm_args, list):
        raise ValueError("extra_vllm_args must be a list of strings")
    if not extra_vllm_args:
        return []

    normalized: list[str] = []
    for index, value in enumerate(extra_vllm_args):
        if not isinstance(value, str):
            raise ValueError(f"extra_vllm_args[{index}] must be a string")
        if not value.strip():
            raise ValueError(f"extra_vllm_args[{index}] cannot be empty")
        if "\x00" in value:
            raise ValueError(f"extra_vllm_args[{index}] cannot contain NUL byte")
        normalized.append(value)
    return normalized


def _effective_partition_vllm_sif(
    *,
    partition_vllm_sif: Path | None,
    default_vllm_sif: Path,
) -> Path:
    if partition_vllm_sif is not None and partition_vllm_sif.exists():
        return partition_vllm_sif
    return default_vllm_sif


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
    vllm_sif: Path | None = None


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
    job_name_prefix: str
    job_nodes: int
    service_port: int
    jaeger_otlp_port: int
    jaeger_ui_port: int
    startup_timeout: int
    startup_timeout_after_running: bool
    stop_wait_timeout_seconds: int
    stop_poll_interval_seconds: float
    wait_up_poll_interval_seconds: float
    ssh_options: list[str]
    port_profiles: dict[int, AMDHPCPortProfile]
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
    port_profile_id: int
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
    extra_env: dict[str, str] = field(default_factory=dict)
    group_name: str | None = None
    group_profiles: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActiveJob":
        return cls(
            port_profile_id=int(payload.get("port_profile_id", 0)),
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
            extra_env=(
                {
                    str(key): str(value)
                    for key, value in payload.get("extra_env", {}).items()
                    if isinstance(key, str)
                }
                if isinstance(payload.get("extra_env"), dict)
                else {}
            ),
            group_name=(
                str(payload.get("group_name"))
                if isinstance(payload.get("group_name"), str) and str(payload.get("group_name")).strip()
                else None
            ),
            group_profiles=(
                [
                    int(item)
                    for item in payload.get("group_profiles", [])
                    if isinstance(item, int) and not isinstance(item, bool)
                ]
                if isinstance(payload.get("group_profiles"), list)
                else []
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "port_profile_id": self.port_profile_id,
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
            "extra_env": dict(self.extra_env),
            "group_name": self.group_name,
            "group_profiles": list(self.group_profiles),
        }


class ControlPlane:
    """Main API used by the HTTP server and CLI."""

    def __init__(self, config_path: Path, *, archive_previous_artifacts: bool = True) -> None:
        self._config_path = config_path.resolve()
        self._cfg = load_runtime_config(self._config_path)
        self._lock = threading.Lock()
        self._start_progress_lock = threading.Lock()
        self._stop_progress_lock = threading.Lock()
        self._start_progress: dict[int, dict[str, Any]] = {}
        self._stop_progress: dict[int, dict[str, Any]] = {}
        self._cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self._cfg.log_dir.mkdir(parents=True, exist_ok=True)
        self._cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
        if archive_previous_artifacts:
            self._archive_previous_artifacts()

    @property
    def config(self) -> RuntimeConfig:
        return self._cfg

    def validate_startup_requirements(self) -> list[str]:
        with self._lock:
            return self._prepare_sif_files()

    def start(
        self,
        *,
        port_profile_id: int,
        partition: str,
        model: str,
        block: bool = False,
        extra_env: dict[str, str] | None = None,
        lmcache_max_local_cpu_size: str | None = None,
        extra_vllm_args: list[str] | None = None,
    ) -> CommandResult:
        port_profile = self._require_port_profile(port_profile_id)
        try:
            normalized_extra_env = _normalize_service_extra_env(extra_env)
            normalized_extra_env, lmcache_enabled = _apply_lmcache_option(
                extra_env=normalized_extra_env,
                lmcache_max_local_cpu_size=lmcache_max_local_cpu_size,
            )
            normalized_extra_vllm_args = _normalize_user_extra_vllm_args(extra_vllm_args)
        except ValueError as exc:
            raise ControlPlaneError(
                message=f"invalid start.extra_env/lmcache/extra_vllm_args: {exc}",
                code=122,
                http_status=400,
            ) from exc
        start_started_at = _utc_now_iso()
        self._set_start_progress(
            port_profile_id=port_profile.profile_id,
            status="running",
            phase="validate",
            message=(
                f"validating start request port_profile={port_profile.profile_id} "
                f"partition={partition} model={model}"
            ),
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
                state_changed, active, job_status = self._refresh_active_job(
                    state,
                    port_profile_id=port_profile.profile_id,
                )
                if state_changed:
                    self._save_state(state)
                self._ensure_single_profile_command_allowed(
                    state,
                    port_profile_id=port_profile.profile_id,
                    command_name="start",
                )

                if active is not None and job_status is not None:
                    raise ControlPlaneError(
                        message=(
                            f"active job already exists for port profile {port_profile.profile_id}: "
                            f"{active.job_id} ({job_status}). Stop it first before launching another."
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
                effective_vllm_sif = self._ensure_partition_vllm_sif_file(partition_spec)

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
                    port_profile_id=port_profile.profile_id,
                    status="running",
                    phase="submit",
                    message=(
                        f"writing sbatch script and submitting to partition {partition_spec.name} "
                        f"for port profile {port_profile.profile_id}"
                    ),
                    job_id=None,
                    started_at=start_started_at,
                    finished_at=None,
                )
                self._ensure_profile_ports_available_on_login(port_profile)
                script_path = self._write_sbatch_script(
                    partition_spec=partition_spec,
                    model_spec=model_spec,
                    port_profile=port_profile,
                    extra_env=normalized_extra_env,
                    lmcache_enabled=lmcache_enabled,
                    extra_vllm_args=normalized_extra_vllm_args,
                )
                sbatch_result = self._run_checked(["sbatch", str(script_path)], timeout_seconds=120)
                job_id = _extract_sbatch_job_id(f"{sbatch_result.stdout}\n{sbatch_result.stderr}")

                self._set_start_progress(
                    port_profile_id=port_profile.profile_id,
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
                    port_profile_id=port_profile.profile_id,
                    job_id=job_id,
                    partition=partition_spec.name,
                    model=model_spec.name,
                    submitted_at=_utc_now_iso(),
                    tensor_parallel_size=partition_spec.gpus_per_node,
                    service_port=port_profile.vllm_port,
                    jaeger_otlp_port=port_profile.jaeger_otlp_port,
                    jaeger_ui_port=port_profile.jaeger_api_port,
                    sbatch_script=str(script_path),
                    slurm_out_log=slurm_out_log,
                    slurm_err_log=slurm_err_log,
                    jaeger_log=jaeger_log,
                    vllm_log=vllm_log,
                    extra_env=normalized_extra_env,
                )
                self._set_active_job(state, port_profile_id=port_profile.profile_id, active_job=new_active.to_dict())
                self._save_state(state)

                result_data = {
                    "port_profile": port_profile.profile_id,
                    "job_id": job_id,
                    "partition": partition_spec.name,
                    "model": model_spec.name,
                    "time_limit": partition_spec.max_time,
                    "tensor_parallel_size": partition_spec.gpus_per_node,
                    "service_port": port_profile.vllm_port,
                    "jaeger_otlp_port": port_profile.jaeger_otlp_port,
                    "jaeger_ui_port": port_profile.jaeger_api_port,
                    "vllm_sif": str(effective_vllm_sif),
                    "sbatch_script": str(script_path),
                    "extra_env": dict(normalized_extra_env),
                    "extra_vllm_args": list(normalized_extra_vllm_args),
                    "blocked": block,
                }

            if not block:
                finished_at = _utc_now_iso()
                self._set_start_progress(
                    port_profile_id=port_profile.profile_id,
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
                port_profile_id=port_profile.profile_id,
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
                vllm_probe = snapshot.get("vllm") if isinstance(snapshot.get("vllm"), dict) else {}
                jaeger_probe = snapshot.get("jaeger") if isinstance(snapshot.get("jaeger"), dict) else {}

                def _probe_status(name: str, probe: dict[str, Any]) -> str:
                    ok = bool(probe.get("ok"))
                    http_status = probe.get("http_status")
                    if ok:
                        return f"{name}=up(http={http_status})"
                    error_raw = probe.get("error")
                    error_text = _truncate_text(str(error_raw), max_chars=120) if error_raw else "unknown"
                    return f"{name}=down(http={http_status}, error={error_text})"

                message = (
                    f"waiting for services: slurm_status={active_status}, "
                    f"{_probe_status('vllm', vllm_probe)}, "
                    f"{_probe_status('jaeger', jaeger_probe)}"
                )
                if message == last_wait_message:
                    return
                last_wait_message = message
                self._set_start_progress(
                    port_profile_id=port_profile.profile_id,
                    status="running",
                    phase="wait_services",
                    message=message,
                    job_id=job_id,
                    started_at=start_started_at,
                    finished_at=None,
                )

            readiness = self._wait_for_services_up(
                port_profile_id=port_profile.profile_id,
                timeout_seconds=self._cfg.startup_timeout,
                poll_interval_seconds=self._cfg.wait_up_poll_interval_seconds,
                expected_job_id=job_id,
                defer_timeout_until_running=self._cfg.startup_timeout_after_running,
                progress_callback=_on_wait_snapshot,
            )
            result_data["readiness"] = readiness
            result_data["waited_seconds"] = readiness.get("waited_seconds")

            finished_at = _utc_now_iso()
            self._set_start_progress(
                port_profile_id=port_profile.profile_id,
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
                port_profile_id=port_profile.profile_id,
                status="failed",
                phase="failed",
                message=str(exc),
                job_id=self._current_start_job_id(port_profile.profile_id),
                started_at=start_started_at,
                finished_at=_utc_now_iso(),
            )
            raise

    def start_group(
        self,
        *,
        group_name: str,
        port_profile_ids: list[int],
        partition: str,
        model: str,
        block: bool = True,
        extra_env: dict[str, str] | None = None,
        lmcache_max_local_cpu_size: str | None = None,
        extra_vllm_args: list[str] | None = None,
        launch_mode: str = "single-node-split",
    ) -> CommandResult:
        try:
            normalized_extra_env = _normalize_service_extra_env(extra_env)
            normalized_extra_env, lmcache_enabled = _apply_lmcache_option(
                extra_env=normalized_extra_env,
                lmcache_max_local_cpu_size=lmcache_max_local_cpu_size,
            )
            normalized_extra_vllm_args = _normalize_user_extra_vllm_args(extra_vllm_args)
        except ValueError as exc:
            raise ControlPlaneError(
                message=f"invalid group/start.extra_env/lmcache/extra_vllm_args: {exc}",
                code=123,
                http_status=400,
            ) from exc
        normalized_group_name = group_name.strip()
        if not normalized_group_name:
            raise ControlPlaneError(
                message="group_name must be non-empty",
                code=71,
                http_status=400,
            )
        if not port_profile_ids:
            raise ControlPlaneError(
                message="profile_list must contain at least one profile id",
                code=72,
                http_status=400,
            )
        if any(isinstance(value, bool) for value in port_profile_ids):
            raise ControlPlaneError(
                message="profile_list values must be integers",
                code=73,
                http_status=400,
            )

        normalized_profile_ids = [int(value) for value in port_profile_ids]
        if len(set(normalized_profile_ids)) != len(normalized_profile_ids):
            raise ControlPlaneError(
                message="profile_list cannot contain duplicate profile ids",
                code=74,
                http_status=400,
            )

        normalized_launch_mode_raw = str(launch_mode).strip().lower().replace("_", "-")
        if normalized_launch_mode_raw in {"single-node-split", "split", "single-node"}:
            normalized_launch_mode = "single-node-split"
        elif normalized_launch_mode_raw in {
            "multi-node",
            "many",
            "many-node",
            "one-profile-per-node",
            "multi-node-one-profile",
        }:
            normalized_launch_mode = "multi-node-one-profile"
        else:
            raise ControlPlaneError(
                message=(
                    "launch_mode must be one of: single-node-split, split, single-node, "
                    "multi-node, many, one-profile-per-node"
                ),
                code=125,
                http_status=400,
                details={"launch_mode": launch_mode},
            )

        port_profiles = [self._require_port_profile(profile_id) for profile_id in normalized_profile_ids]
        start_started_at = _utc_now_iso()
        for profile in port_profiles:
            self._set_start_progress(
                port_profile_id=profile.profile_id,
                status="running",
                phase="validate",
                message=(
                    f"validating grouped start request group={normalized_group_name} "
                    f"partition={partition} model={model} launch_mode={normalized_launch_mode}"
                ),
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
                state_changed, _ = self._refresh_all_active_jobs(state)
                if state_changed:
                    self._save_state(state)

                group_active_entries = self._active_group_entries(
                    state,
                    group_name=normalized_group_name,
                )
                if group_active_entries:
                    raise ControlPlaneError(
                        message=f"group '{normalized_group_name}' already has active services",
                        code=75,
                        http_status=409,
                        details={
                            "group_name": normalized_group_name,
                            "active_profiles": [entry[0] for entry in group_active_entries],
                        },
                    )

                conflicting_profiles: list[dict[str, Any]] = []
                for profile in port_profiles:
                    active_raw = self._active_job_raw(state, port_profile_id=profile.profile_id)
                    if not isinstance(active_raw, dict):
                        continue
                    active = ActiveJob.from_dict(active_raw)
                    conflicting_profiles.append(
                        {
                            "port_profile": profile.profile_id,
                            "job_id": active.job_id,
                            "group_name": active.group_name,
                        }
                    )
                if conflicting_profiles:
                    raise ControlPlaneError(
                        message=(
                            "cannot start grouped services because one or more profiles "
                            "already have active jobs"
                        ),
                        code=76,
                        http_status=409,
                        details={"conflicts": conflicting_profiles},
                    )

                partition_spec = self._cfg.partitions.get(partition)
                if partition_spec is None:
                    raise ControlPlaneError(
                        message=f"unknown partition '{partition}'",
                        code=12,
                        http_status=400,
                        details={"allowed_partitions": sorted(self._cfg.partitions.keys())},
                    )
                effective_vllm_sif = self._ensure_partition_vllm_sif_file(partition_spec)

                model_spec = self._cfg.models.get(model)
                if model_spec is None:
                    raise ControlPlaneError(
                        message=f"unknown model '{model}'",
                        code=13,
                        http_status=400,
                        details={"allowed_models": sorted(self._cfg.models.keys())},
                    )

                gpus_per_profile: int
                total_vram_per_profile_gb: float
                if normalized_launch_mode == "single-node-split":
                    try:
                        (
                            gpus_per_profile,
                            total_vram_per_profile_gb,
                            visible_devices_by_profile,
                        ) = _compute_even_group_gpu_split(
                            gpus_per_node=partition_spec.gpus_per_node,
                            gpu_memory_gb=partition_spec.gpu_memory_gb,
                            group_size=len(port_profiles),
                        )
                    except ValueError as exc:
                        raise ControlPlaneError(
                            message=f"invalid grouped GPU split for partition '{partition}': {exc}",
                            code=120,
                            http_status=422,
                            details={
                                "partition": partition,
                                "gpus_per_node": partition_spec.gpus_per_node,
                                "gpu_memory_gb": partition_spec.gpu_memory_gb,
                                "group_size": len(port_profiles),
                                "profile_list": list(normalized_profile_ids),
                                "launch_mode": normalized_launch_mode,
                            },
                        ) from exc
                    max_weight = total_vram_per_profile_gb * 0.75
                    if model_spec.weight_vram_gb > max_weight:
                        raise ControlPlaneError(
                            message=(
                                f"model '{model}' requires {model_spec.weight_vram_gb:.1f} GB which exceeds "
                                "75% of per-profile VRAM for grouped launch "
                                f"({max_weight:.1f} GB; group_size={len(port_profiles)}, "
                                f"gpus_per_profile={gpus_per_profile})"
                            ),
                            code=14,
                            http_status=422,
                            details={
                                "model_weight_vram_gb": model_spec.weight_vram_gb,
                                "partition_total_vram_gb": partition_spec.total_vram_gb,
                                "group_total_vram_per_profile_gb": total_vram_per_profile_gb,
                                "group_size": len(port_profiles),
                                "gpus_per_profile": gpus_per_profile,
                                "max_allowed_weight_vram_gb": max_weight,
                                "launch_mode": normalized_launch_mode,
                            },
                        )
                else:
                    gpus_per_profile = partition_spec.gpus_per_node
                    total_vram_per_profile_gb = partition_spec.total_vram_gb
                    max_weight = total_vram_per_profile_gb * 0.75
                    if model_spec.weight_vram_gb > max_weight:
                        raise ControlPlaneError(
                            message=(
                                f"model '{model}' requires {model_spec.weight_vram_gb:.1f} GB which exceeds "
                                "75% of per-node VRAM for multi-node launch "
                                f"({max_weight:.1f} GB)"
                            ),
                            code=14,
                            http_status=422,
                            details={
                                "model_weight_vram_gb": model_spec.weight_vram_gb,
                                "partition_total_vram_gb": partition_spec.total_vram_gb,
                                "group_total_vram_per_profile_gb": total_vram_per_profile_gb,
                                "group_size": len(port_profiles),
                                "gpus_per_profile": gpus_per_profile,
                                "max_allowed_weight_vram_gb": max_weight,
                                "launch_mode": normalized_launch_mode,
                            },
                        )

                for profile in port_profiles:
                    self._ensure_profile_ports_available_on_login(profile)

                if normalized_launch_mode == "single-node-split":
                    script_path = self._write_group_sbatch_script(
                        partition_spec=partition_spec,
                        model_spec=model_spec,
                        port_profiles=port_profiles,
                        group_name=normalized_group_name,
                        gpus_per_profile=gpus_per_profile,
                        visible_devices_by_profile=visible_devices_by_profile,
                        extra_env=normalized_extra_env,
                        lmcache_enabled=lmcache_enabled,
                        extra_vllm_args=normalized_extra_vllm_args,
                    )
                else:
                    script_path = self._write_group_multi_node_sbatch_script(
                        partition_spec=partition_spec,
                        model_spec=model_spec,
                        port_profiles=port_profiles,
                        group_name=normalized_group_name,
                        extra_env=normalized_extra_env,
                        lmcache_enabled=lmcache_enabled,
                        extra_vllm_args=normalized_extra_vllm_args,
                    )
                sbatch_result = self._run_checked(["sbatch", str(script_path)], timeout_seconds=120)
                job_id = _extract_sbatch_job_id(f"{sbatch_result.stdout}\n{sbatch_result.stderr}")

                for profile in port_profiles:
                    slurm_out_log = str(self._cfg.log_dir / f"slurm.{job_id}.out")
                    slurm_err_log = str(self._cfg.log_dir / f"slurm.{job_id}.err")
                    jaeger_log = str(self._cfg.log_dir / f"jaeger.{job_id}.p{profile.profile_id}.log")
                    vllm_log = str(self._cfg.log_dir / f"vllm.{job_id}.p{profile.profile_id}.log")
                    new_active = ActiveJob(
                        port_profile_id=profile.profile_id,
                        job_id=job_id,
                        partition=partition_spec.name,
                        model=model_spec.name,
                        submitted_at=_utc_now_iso(),
                        tensor_parallel_size=gpus_per_profile,
                        service_port=profile.vllm_port,
                        jaeger_otlp_port=profile.jaeger_otlp_port,
                        jaeger_ui_port=profile.jaeger_api_port,
                        sbatch_script=str(script_path),
                        slurm_out_log=slurm_out_log,
                        slurm_err_log=slurm_err_log,
                        jaeger_log=jaeger_log,
                        vllm_log=vllm_log,
                        extra_env=normalized_extra_env,
                        group_name=normalized_group_name,
                        group_profiles=list(normalized_profile_ids),
                    )
                    self._set_active_job(
                        state,
                        port_profile_id=profile.profile_id,
                        active_job=new_active.to_dict(),
                    )

                self._save_state(state)
                result_data = {
                    "group_name": normalized_group_name,
                    "profile_list": list(normalized_profile_ids),
                    "launch_mode": normalized_launch_mode,
                    "job_id": job_id,
                    "partition": partition_spec.name,
                    "model": model_spec.name,
                    "time_limit": partition_spec.max_time,
                    "tensor_parallel_size": gpus_per_profile,
                    "gpus_per_profile": gpus_per_profile,
                    "total_vram_per_profile_gb": total_vram_per_profile_gb,
                    "vllm_sif": str(effective_vllm_sif),
                    "extra_env": dict(normalized_extra_env),
                    "extra_vllm_args": list(normalized_extra_vllm_args),
                    "blocked": bool(block),
                    "sbatch_script": str(script_path),
                }

            if not block:
                finished_at = _utc_now_iso()
                for profile in port_profiles:
                    self._set_start_progress(
                        port_profile_id=profile.profile_id,
                        status="succeeded",
                        phase="done",
                        message=(
                            f"submitted grouped job {job_id} "
                            f"(group={normalized_group_name})"
                        ),
                        job_id=job_id,
                        started_at=start_started_at,
                        finished_at=finished_at,
                    )
                return CommandResult(
                    code=0,
                    message=f"submitted grouped job {job_id}",
                    data=result_data,
                )

            for profile in port_profiles:
                self._set_start_progress(
                    port_profile_id=profile.profile_id,
                    status="running",
                    phase="wait_services",
                    message=(
                        f"submitted grouped job {job_id}; waiting for grouped service readiness "
                        f"(group={normalized_group_name})"
                    ),
                    job_id=job_id,
                    started_at=start_started_at,
                    finished_at=None,
                )

            last_wait_messages_by_profile: dict[int, str] = {}

            def _on_group_wait_snapshot(snapshot: dict[str, Any]) -> None:
                profiles_payload = snapshot.get("profiles")
                if not isinstance(profiles_payload, list):
                    return

                for profile_snapshot in profiles_payload:
                    if not isinstance(profile_snapshot, dict):
                        continue
                    profile_id_raw = profile_snapshot.get("port_profile")
                    if not isinstance(profile_id_raw, int):
                        continue

                    active_status = profile_snapshot.get("active_job_status")
                    vllm_probe = (
                        profile_snapshot.get("vllm")
                        if isinstance(profile_snapshot.get("vllm"), dict)
                        else {}
                    )
                    jaeger_probe = (
                        profile_snapshot.get("jaeger")
                        if isinstance(profile_snapshot.get("jaeger"), dict)
                        else {}
                    )

                    def _probe_status(name: str, probe: dict[str, Any]) -> str:
                        ok = bool(probe.get("ok"))
                        http_status = probe.get("http_status")
                        if ok:
                            return f"{name}=up(http={http_status})"
                        error_raw = probe.get("error")
                        error_text = _truncate_text(str(error_raw), max_chars=120) if error_raw else "unknown"
                        return f"{name}=down(http={http_status}, error={error_text})"

                    message = (
                        f"group wait: slurm_status={active_status}, "
                        f"{_probe_status('vllm', vllm_probe)}, "
                        f"{_probe_status('jaeger', jaeger_probe)}"
                    )
                    if last_wait_messages_by_profile.get(profile_id_raw) == message:
                        continue
                    last_wait_messages_by_profile[profile_id_raw] = message
                    self._set_start_progress(
                        port_profile_id=profile_id_raw,
                        status="running",
                        phase="wait_services",
                        message=message,
                        job_id=job_id,
                        started_at=start_started_at,
                        finished_at=None,
                    )

            readiness = self._wait_for_group_services_up(
                group_name=normalized_group_name,
                port_profile_ids=normalized_profile_ids,
                timeout_seconds=self._cfg.startup_timeout,
                poll_interval_seconds=self._cfg.wait_up_poll_interval_seconds,
                expected_job_id=job_id,
                defer_timeout_until_running=self._cfg.startup_timeout_after_running,
                progress_callback=_on_group_wait_snapshot,
            )
            result_data["readiness"] = readiness
            result_data["waited_seconds"] = readiness.get("waited_seconds")

            finished_at = _utc_now_iso()
            for profile in port_profiles:
                self._set_start_progress(
                    port_profile_id=profile.profile_id,
                    status="succeeded",
                    phase="done",
                    message=(
                        f"submitted grouped job {job_id} and services are up "
                        f"(group={normalized_group_name})"
                    ),
                    job_id=job_id,
                    started_at=start_started_at,
                    finished_at=finished_at,
                )
            return CommandResult(
                code=0,
                message=f"group '{normalized_group_name}' services are up",
                data=result_data,
            )
        except Exception as exc:  # noqa: BLE001
            for profile in port_profiles:
                self._set_start_progress(
                    port_profile_id=profile.profile_id,
                    status="failed",
                    phase="failed",
                    message=str(exc),
                    job_id=self._current_start_job_id(profile.profile_id),
                    started_at=start_started_at,
                    finished_at=_utc_now_iso(),
                )
            raise

    def start_many(
        self,
        *,
        group_name: str,
        port_profile_ids: list[int],
        partition: str,
        model: str,
        block: bool = True,
        extra_env: dict[str, str] | None = None,
        lmcache_max_local_cpu_size: str | None = None,
        extra_vllm_args: list[str] | None = None,
    ) -> CommandResult:
        return self.start_group(
            group_name=group_name,
            port_profile_ids=port_profile_ids,
            partition=partition,
            model=model,
            block=block,
            extra_env=extra_env,
            lmcache_max_local_cpu_size=lmcache_max_local_cpu_size,
            extra_vllm_args=extra_vllm_args,
            launch_mode="multi-node-one-profile",
        )

    def render_start_sbatch(
        self,
        *,
        port_profile_id: int,
        partition: str,
        model: str,
        extra_env: dict[str, str] | None = None,
        lmcache_max_local_cpu_size: str | None = None,
        extra_vllm_args: list[str] | None = None,
        no_async_scheduling: bool = False,
        local_mode_script: str | None = None,
        check_port_availability: bool = False,
    ) -> CommandResult:
        port_profile = self._require_port_profile(port_profile_id)
        try:
            normalized_extra_env = _normalize_service_extra_env(extra_env)
            normalized_extra_env, lmcache_enabled = _apply_lmcache_option(
                extra_env=normalized_extra_env,
                lmcache_max_local_cpu_size=lmcache_max_local_cpu_size,
            )
            normalized_extra_vllm_args = _normalize_user_extra_vllm_args(extra_vllm_args)
            normalized_local_mode_script = _normalize_local_mode_script(local_mode_script)
        except ValueError as exc:
            raise ControlPlaneError(
                message=f"invalid render/start.extra_env/lmcache/extra_vllm_args/local_mode: {exc}",
                code=122,
                http_status=400,
            ) from exc

        with self._lock:
            partition_spec = self._cfg.partitions.get(partition)
            if partition_spec is None:
                raise ControlPlaneError(
                    message=f"unknown partition '{partition}'",
                    code=12,
                    http_status=400,
                    details={"allowed_partitions": sorted(self._cfg.partitions.keys())},
                )
            # Render-only mode should not require local SIF files to exist yet.
            effective_vllm_sif = self._effective_vllm_sif_path(partition_spec)

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

            if check_port_availability:
                self._ensure_profile_ports_available_on_login(port_profile)

            if normalized_local_mode_script is not None:
                gateway_port, gateway_parse_port = self._gateway_ports_for_profile(
                    port_profile_id=port_profile.profile_id
                )
                script_path = self._write_local_mode_sbatch_script(
                    partition_spec=partition_spec,
                    model_spec=model_spec,
                    port_profile=port_profile,
                    gateway_port=gateway_port,
                    gateway_parse_port=gateway_parse_port,
                    local_mode_script=normalized_local_mode_script,
                    extra_env=normalized_extra_env,
                    lmcache_enabled=lmcache_enabled,
                    extra_vllm_args=normalized_extra_vllm_args,
                    no_async_scheduling=no_async_scheduling,
                )
            else:
                script_path = self._write_sbatch_script(
                    partition_spec=partition_spec,
                    model_spec=model_spec,
                    port_profile=port_profile,
                    extra_env=normalized_extra_env,
                    lmcache_enabled=lmcache_enabled,
                    extra_vllm_args=normalized_extra_vllm_args,
                    no_async_scheduling=no_async_scheduling,
                )

        return CommandResult(
            code=0,
            message=f"rendered sbatch script for port profile {port_profile.profile_id}",
            data={
                "port_profile": port_profile.profile_id,
                "partition": partition_spec.name,
                "model": model_spec.name,
                "time_limit": partition_spec.max_time,
                "tensor_parallel_size": partition_spec.gpus_per_node,
                "service_port": port_profile.vllm_port,
                "jaeger_otlp_port": port_profile.jaeger_otlp_port,
                "jaeger_ui_port": port_profile.jaeger_api_port,
                "vllm_sif": str(effective_vllm_sif),
                "lmcache_port": port_profile.lmcache_port,
                "sbatch_script": str(script_path),
                "extra_env": dict(normalized_extra_env),
                "extra_vllm_args": list(normalized_extra_vllm_args),
                "lmcache_enabled": lmcache_enabled,
                "no_async_scheduling": bool(no_async_scheduling),
                "local_mode_enabled": normalized_local_mode_script is not None,
                "local_mode_script": normalized_local_mode_script,
                "validated_ports_available_on_login": bool(check_port_availability),
            },
        )

    def render_start_group_sbatch(
        self,
        *,
        group_name: str,
        port_profile_ids: list[int],
        partition: str,
        model: str,
        extra_env: dict[str, str] | None = None,
        lmcache_max_local_cpu_size: str | None = None,
        extra_vllm_args: list[str] | None = None,
        no_async_scheduling: bool = False,
        local_mode_script: str | None = None,
        check_port_availability: bool = False,
    ) -> CommandResult:
        try:
            normalized_extra_env = _normalize_service_extra_env(extra_env)
            normalized_extra_env, lmcache_enabled = _apply_lmcache_option(
                extra_env=normalized_extra_env,
                lmcache_max_local_cpu_size=lmcache_max_local_cpu_size,
            )
            normalized_extra_vllm_args = _normalize_user_extra_vllm_args(extra_vllm_args)
            normalized_local_mode_script = _normalize_local_mode_script(local_mode_script)
        except ValueError as exc:
            raise ControlPlaneError(
                message=f"invalid render/group-start.extra_env/lmcache/extra_vllm_args/local_mode: {exc}",
                code=123,
                http_status=400,
            ) from exc

        normalized_group_name = group_name.strip()
        if not normalized_group_name:
            raise ControlPlaneError(
                message="group_name must be non-empty",
                code=71,
                http_status=400,
            )
        if not port_profile_ids:
            raise ControlPlaneError(
                message="profile_list must contain at least one profile id",
                code=72,
                http_status=400,
            )
        if any(isinstance(value, bool) for value in port_profile_ids):
            raise ControlPlaneError(
                message="profile_list values must be integers",
                code=73,
                http_status=400,
            )

        normalized_profile_ids = [int(value) for value in port_profile_ids]
        if len(set(normalized_profile_ids)) != len(normalized_profile_ids):
            raise ControlPlaneError(
                message="profile_list cannot contain duplicate profile ids",
                code=74,
                http_status=400,
            )

        port_profiles = [self._require_port_profile(profile_id) for profile_id in normalized_profile_ids]
        with self._lock:
            partition_spec = self._cfg.partitions.get(partition)
            if partition_spec is None:
                raise ControlPlaneError(
                    message=f"unknown partition '{partition}'",
                    code=12,
                    http_status=400,
                    details={"allowed_partitions": sorted(self._cfg.partitions.keys())},
                )
            # Render-only mode should not require local SIF files to exist yet.
            effective_vllm_sif = self._effective_vllm_sif_path(partition_spec)

            model_spec = self._cfg.models.get(model)
            if model_spec is None:
                raise ControlPlaneError(
                    message=f"unknown model '{model}'",
                    code=13,
                    http_status=400,
                    details={"allowed_models": sorted(self._cfg.models.keys())},
                )

            try:
                (
                    gpus_per_profile,
                    total_vram_per_profile_gb,
                    visible_devices_by_profile,
                ) = _compute_even_group_gpu_split(
                    gpus_per_node=partition_spec.gpus_per_node,
                    gpu_memory_gb=partition_spec.gpu_memory_gb,
                    group_size=len(port_profiles),
                )
            except ValueError as exc:
                raise ControlPlaneError(
                    message=f"invalid grouped GPU split for partition '{partition}': {exc}",
                    code=120,
                    http_status=422,
                    details={
                        "partition": partition,
                        "gpus_per_node": partition_spec.gpus_per_node,
                        "gpu_memory_gb": partition_spec.gpu_memory_gb,
                        "group_size": len(port_profiles),
                        "profile_list": list(normalized_profile_ids),
                    },
                ) from exc

            max_weight = total_vram_per_profile_gb * 0.75
            if model_spec.weight_vram_gb > max_weight:
                raise ControlPlaneError(
                    message=(
                        f"model '{model}' requires {model_spec.weight_vram_gb:.1f} GB which exceeds "
                        "75% of per-profile VRAM for grouped launch "
                        f"({max_weight:.1f} GB; group_size={len(port_profiles)}, "
                        f"gpus_per_profile={gpus_per_profile})"
                    ),
                    code=14,
                    http_status=422,
                    details={
                        "model_weight_vram_gb": model_spec.weight_vram_gb,
                        "partition_total_vram_gb": partition_spec.total_vram_gb,
                        "group_total_vram_per_profile_gb": total_vram_per_profile_gb,
                        "group_size": len(port_profiles),
                        "gpus_per_profile": gpus_per_profile,
                        "max_allowed_weight_vram_gb": max_weight,
                    },
                )

            if check_port_availability:
                for profile in port_profiles:
                    self._ensure_profile_ports_available_on_login(profile)

            script_path = self._write_group_sbatch_script(
                partition_spec=partition_spec,
                model_spec=model_spec,
                port_profiles=port_profiles,
                group_name=normalized_group_name,
                gpus_per_profile=gpus_per_profile,
                visible_devices_by_profile=visible_devices_by_profile,
                extra_env=normalized_extra_env,
                lmcache_enabled=lmcache_enabled,
                extra_vllm_args=normalized_extra_vllm_args,
                no_async_scheduling=no_async_scheduling,
                local_mode_script=normalized_local_mode_script,
            )

        return CommandResult(
            code=0,
            message=f"rendered grouped sbatch script for group '{normalized_group_name}'",
            data={
                "group_name": normalized_group_name,
                "profile_list": list(normalized_profile_ids),
                "partition": partition_spec.name,
                "model": model_spec.name,
                "time_limit": partition_spec.max_time,
                "tensor_parallel_size": gpus_per_profile,
                "gpus_per_profile": gpus_per_profile,
                "total_vram_per_profile_gb": total_vram_per_profile_gb,
                "vllm_sif": str(effective_vllm_sif),
                "visible_devices_by_profile": list(visible_devices_by_profile),
                "sbatch_script": str(script_path),
                "extra_env": dict(normalized_extra_env),
                "extra_vllm_args": list(normalized_extra_vllm_args),
                "lmcache_enabled": lmcache_enabled,
                "no_async_scheduling": bool(no_async_scheduling),
                "local_mode_enabled": normalized_local_mode_script is not None,
                "local_mode_script": normalized_local_mode_script,
                "validated_ports_available_on_login": bool(check_port_availability),
                "profiles": [
                    {
                        "port_profile": profile.profile_id,
                        "service_port": profile.vllm_port,
                        "jaeger_otlp_port": profile.jaeger_otlp_port,
                        "jaeger_ui_port": profile.jaeger_api_port,
                        "lmcache_port": profile.lmcache_port,
                    }
                    for profile in port_profiles
                ],
            },
        )

    def stop_group(
        self,
        *,
        group_name: str,
        reason: str = "stopped",
        block: bool = True,
    ) -> CommandResult:
        normalized_group_name = group_name.strip()
        if not normalized_group_name:
            raise ControlPlaneError(
                message="group_name must be non-empty",
                code=77,
                http_status=400,
            )

        stop_started_at = _utc_now_iso()
        try:
            with self._lock:
                self._require_command("squeue")
                self._require_command("scancel")

                state = self._load_state()
                state_changed, _ = self._refresh_all_active_jobs(state)
                if state_changed:
                    self._save_state(state)

                group_entries = self._active_group_entries(
                    state,
                    group_name=normalized_group_name,
                )
                if not group_entries:
                    raise ControlPlaneError(
                        message=f"no active services for group '{normalized_group_name}'",
                        code=78,
                        http_status=404,
                    )

                group_profile_ids = [entry[0] for entry in group_entries]
                job_ids = {entry[1].job_id for entry in group_entries}
                if len(job_ids) != 1:
                    raise ControlPlaneError(
                        message=(
                            f"active group '{normalized_group_name}' has inconsistent job IDs; "
                            "cannot stop safely"
                        ),
                        code=79,
                        http_status=409,
                        details={
                            "group_name": normalized_group_name,
                            "job_ids": sorted(job_ids),
                            "profiles": group_profile_ids,
                        },
                    )

                job_id = next(iter(job_ids))
                for profile_id in group_profile_ids:
                    self._set_stop_progress(
                        port_profile_id=profile_id,
                        status="running",
                        phase="validate",
                        message=(
                            f"validating grouped stop request group={normalized_group_name} "
                            f"job={job_id}"
                        ),
                        job_id=job_id,
                        started_at=stop_started_at,
                        finished_at=None,
                    )

                slurm_user = self._slurm_user()
                status = self._slurm_job_status(job_id)
                if status is not None:
                    cancel = self._run(
                        ["scancel", "-u", slurm_user, job_id],
                        timeout_seconds=60,
                    )
                    if cancel.returncode != 0:
                        stderr = cancel.stderr.strip()
                        if "Invalid job id specified" not in stderr:
                            raise ControlPlaneError(
                                message=f"failed to stop grouped job {job_id}",
                                code=80,
                                http_status=500,
                                details={
                                    "stderr": _truncate_text(stderr),
                                    "stdout": _truncate_text(cancel.stdout.strip()),
                                },
                            )

                if not block:
                    if status is None:
                        for profile_id in group_profile_ids:
                            self._archive_active_job(state, port_profile_id=profile_id, reason=reason)
                        final_status = "not_found"
                    else:
                        for profile_id, active_raw in self._active_group_entries_raw(
                            state,
                            group_name=normalized_group_name,
                        ):
                            active_payload = dict(active_raw)
                            active_payload["stop_requested_at"] = _utc_now_iso()
                            self._set_active_job(
                                state,
                                port_profile_id=profile_id,
                                active_job=active_payload,
                            )
                        final_status = "cancelling"

                    self._save_state(state)
                    finished_at = _utc_now_iso()
                    for profile_id in group_profile_ids:
                        self._set_stop_progress(
                            port_profile_id=profile_id,
                            status="succeeded",
                            phase="done",
                            message=(
                                f"group stop requested for job {job_id} "
                                f"(group={normalized_group_name}, final_status={final_status})"
                            ),
                            job_id=job_id,
                            started_at=stop_started_at,
                            finished_at=finished_at,
                        )
                    return CommandResult(
                        code=0,
                        message=f"group stop requested for job {job_id}",
                        data={
                            "group_name": normalized_group_name,
                            "profile_list": group_profile_ids,
                            "job_id": job_id,
                            "previous_status": status or "not_found",
                            "final_status": final_status,
                            "waited_seconds": 0.0,
                            "slurm_user": slurm_user,
                            "blocked": False,
                        },
                    )

                wait_started_at = time.monotonic()
                deadline = wait_started_at + self._cfg.stop_wait_timeout_seconds
                last_seen_status: str | None = None
                while True:
                    current_status = self._slurm_job_status(job_id)
                    if current_status is None:
                        break
                    if current_status != last_seen_status:
                        last_seen_status = current_status
                        for profile_id in group_profile_ids:
                            self._set_stop_progress(
                                port_profile_id=profile_id,
                                status="running",
                                phase="wait_slurm",
                                message=(
                                    f"group job {job_id} still active "
                                    f"(group={normalized_group_name}, status={current_status})"
                                ),
                                job_id=job_id,
                                started_at=stop_started_at,
                                finished_at=None,
                            )
                    if time.monotonic() >= deadline:
                        raise ControlPlaneError(
                            message=(
                                f"timed out waiting for grouped job {job_id} to disappear "
                                f"(group={normalized_group_name}, last_status={current_status})"
                            ),
                            code=81,
                            http_status=504,
                            details={
                                "group_name": normalized_group_name,
                                "job_id": job_id,
                                "last_status": current_status,
                                "wait_timeout_seconds": self._cfg.stop_wait_timeout_seconds,
                                "slurm_user": slurm_user,
                            },
                        )
                    time.sleep(self._cfg.stop_poll_interval_seconds)

                waited_seconds = round(time.monotonic() - wait_started_at, 3)
                for profile_id in group_profile_ids:
                    self._archive_active_job(state, port_profile_id=profile_id, reason=reason)
                self._save_state(state)

                finished_at = _utc_now_iso()
                for profile_id in group_profile_ids:
                    self._set_stop_progress(
                        port_profile_id=profile_id,
                        status="succeeded",
                        phase="done",
                        message=f"stopped grouped job {job_id} (group={normalized_group_name})",
                        job_id=job_id,
                        started_at=stop_started_at,
                        finished_at=finished_at,
                    )

                return CommandResult(
                    code=0,
                    message=f"stopped grouped job {job_id}",
                    data={
                        "group_name": normalized_group_name,
                        "profile_list": group_profile_ids,
                        "job_id": job_id,
                        "previous_status": status or "not_found",
                        "final_status": "not_found",
                        "waited_seconds": waited_seconds,
                        "slurm_user": slurm_user,
                        "blocked": True,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                state = self._load_state()
                group_entries = self._active_group_entries(
                    state,
                    group_name=normalized_group_name,
                )
            for profile_id, _ in group_entries:
                self._set_stop_progress(
                    port_profile_id=profile_id,
                    status="failed",
                    phase="failed",
                    message=str(exc),
                    job_id=self._current_stop_job_id(profile_id),
                    started_at=stop_started_at,
                    finished_at=_utc_now_iso(),
                )
            raise

    def stop(
        self,
        *,
        port_profile_id: int,
        reason: str = "stopped",
        block: bool = False,
        allow_group: bool = False,
    ) -> CommandResult:
        port_profile = self._require_port_profile(port_profile_id)
        stop_started_at = _utc_now_iso()
        self._set_stop_progress(
            port_profile_id=port_profile.profile_id,
            status="running",
            phase="validate",
            message=f"validating active job before stop for port profile {port_profile.profile_id}",
            job_id=None,
            started_at=stop_started_at,
            finished_at=None,
        )

        try:
            with self._lock:
                self._require_command("squeue")
                self._require_command("scancel")

                state = self._load_state()
                if not allow_group:
                    self._ensure_single_profile_command_allowed(
                        state,
                        port_profile_id=port_profile.profile_id,
                        command_name="stop",
                    )
                active_raw = self._active_job_raw(state, port_profile_id=port_profile.profile_id)
                if not isinstance(active_raw, dict):
                    raise ControlPlaneError(
                        message=f"no active job for port profile {port_profile.profile_id}",
                        code=21,
                        http_status=404,
                    )

                active = ActiveJob.from_dict(active_raw)
                self._set_stop_progress(
                    port_profile_id=port_profile.profile_id,
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
                        port_profile_id=port_profile.profile_id,
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
                        self._archive_active_job(state, port_profile_id=port_profile.profile_id, reason=reason)
                        self._save_state(state)
                        final_status = "not_found"
                    else:
                        active_raw["stop_requested_at"] = _utc_now_iso()
                        self._set_active_job(
                            state,
                            port_profile_id=port_profile.profile_id,
                            active_job=active_raw,
                        )
                        self._save_state(state)
                        final_status = "cancelling"

                    finished_at = _utc_now_iso()
                    self._set_stop_progress(
                        port_profile_id=port_profile.profile_id,
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
                            "port_profile": port_profile.profile_id,
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
                    port_profile_id=port_profile.profile_id,
                    status="running",
                    phase="wait_slurm",
                    message=f"waiting for job {active.job_id} to disappear from squeue",
                    job_id=active.job_id,
                    started_at=stop_started_at,
                    finished_at=None,
                )

                wait_started_at = time.monotonic()
                deadline = wait_started_at + self._cfg.stop_wait_timeout_seconds
                last_seen_status: str | None = None
                while True:
                    current_status = self._slurm_job_status(active.job_id)
                    if current_status is None:
                        break
                    if current_status != last_seen_status:
                        last_seen_status = current_status
                        self._set_stop_progress(
                            port_profile_id=port_profile.profile_id,
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
                                "wait_timeout_seconds": self._cfg.stop_wait_timeout_seconds,
                                "slurm_user": slurm_user,
                            },
                        )
                    time.sleep(self._cfg.stop_poll_interval_seconds)

                waited_seconds = round(time.monotonic() - wait_started_at, 3)
                self._archive_active_job(state, port_profile_id=port_profile.profile_id, reason=reason)
                self._save_state(state)

                finished_at = _utc_now_iso()
                self._set_stop_progress(
                    port_profile_id=port_profile.profile_id,
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
                        "port_profile": port_profile.profile_id,
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
                port_profile_id=port_profile.profile_id,
                status="failed",
                phase="failed",
                message=str(exc),
                job_id=self._current_stop_job_id(port_profile.profile_id),
                started_at=stop_started_at,
                finished_at=_utc_now_iso(),
            )
            raise

    def stop_poll(self, *, port_profile_id: int, allow_group: bool = False) -> CommandResult:
        with self._lock:
            state = self._load_state()
            if not allow_group:
                self._ensure_single_profile_command_allowed(
                    state,
                    port_profile_id=port_profile_id,
                    command_name="stop-poll",
                )
            active_raw = self._active_job_raw(state, port_profile_id=port_profile_id)
            if not isinstance(active_raw, dict):
                return CommandResult(
                    code=0,
                    message=f"no active job for port profile {port_profile_id}",
                    data={
                        "port_profile": port_profile_id,
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
                self._archive_active_job(state, port_profile_id=port_profile_id, reason=reason)
                self._save_state(state)
                return CommandResult(
                    code=0,
                    message=f"job {active.job_id} is no longer in squeue",
                    data={
                        "port_profile": port_profile_id,
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
                    "port_profile": port_profile_id,
                    "done": False,
                    "job_id": active.job_id,
                    "job_status": status,
                    "stop_requested_at": stop_requested_at,
                },
            )

    def start_status(self, *, port_profile_id: int) -> CommandResult:
        return self._progress_status(
            kind="start",
            port_profile_id=port_profile_id,
            idle_message="no start command has been run yet",
        )

    def stop_status(self, *, port_profile_id: int) -> CommandResult:
        return self._progress_status(
            kind="stop",
            port_profile_id=port_profile_id,
            idle_message="no stop command has been run yet",
        )

    def logs(self, *, port_profile_id: int, lines: int = 200) -> CommandResult:
        with self._lock:
            if lines <= 0:
                raise ControlPlaneError(
                    message="lines must be > 0",
                    code=31,
                    http_status=400,
                )

            state = self._load_state()
            self._ensure_single_profile_command_allowed(
                state,
                port_profile_id=port_profile_id,
                command_name="logs",
            )
            active_raw = self._active_job_raw(state, port_profile_id=port_profile_id)
            if not isinstance(active_raw, dict):
                raise ControlPlaneError(
                    message=f"no active job for port profile {port_profile_id}",
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
                self._archive_active_job(state, port_profile_id=port_profile_id, reason="finished")
                self._save_state(state)

            return CommandResult(
                code=0,
                message="log snapshot collected",
                data={
                    "port_profile": port_profile_id,
                    "job_id": active.job_id,
                    "job_status": status or "not_found",
                    "lines": lines,
                    "logs": logs,
                },
            )

    def up(self, *, port_profile_id: int) -> CommandResult:
        with self._lock:
            readiness = self._collect_readiness_snapshot(port_profile_id=port_profile_id)
        message = "services are up" if readiness["ready"] else "services are not ready"
        return CommandResult(code=0, message=message, data=readiness)

    def wait_up(
        self,
        *,
        port_profile_id: int,
        timeout_seconds: int | None = None,
        poll_interval_seconds: float | None = None,
        defer_timeout_until_running: bool | None = None,
    ) -> CommandResult:
        if timeout_seconds is None:
            timeout_seconds = self._cfg.startup_timeout
        if poll_interval_seconds is None:
            poll_interval_seconds = self._cfg.wait_up_poll_interval_seconds
        if defer_timeout_until_running is None:
            defer_timeout_until_running = self._cfg.startup_timeout_after_running
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
            port_profile_id=port_profile_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            expected_job_id=None,
            defer_timeout_until_running=defer_timeout_until_running,
        )
        return CommandResult(code=0, message="services are up", data=snapshot)

    def status(self, *, port_profile_id: int) -> CommandResult:
        with self._lock:
            state = self._load_state()
            state_changed, active_jobs = self._refresh_all_active_jobs(state)
            if state_changed:
                self._save_state(state)

            selected = active_jobs.get(port_profile_id)
            active = selected[0] if selected is not None else None
            job_status = selected[1] if selected is not None else None
            active_groups: dict[str, dict[str, Any]] = {}
            for profile_id, (profile_active, profile_status) in active_jobs.items():
                if not profile_active.group_name:
                    continue
                group_payload = active_groups.setdefault(
                    profile_active.group_name,
                    {
                        "group_name": profile_active.group_name,
                        "job_id": profile_active.job_id,
                        "profiles": [],
                        "active_job_status_by_profile": {},
                    },
                )
                profiles_payload = group_payload.get("profiles")
                if isinstance(profiles_payload, list):
                    profiles_payload.append(profile_id)
                status_payload = group_payload.get("active_job_status_by_profile")
                if isinstance(status_payload, dict):
                    status_payload[str(profile_id)] = profile_status

            return CommandResult(
                code=0,
                message="status",
                data={
                    "server": {
                        "host": self._cfg.host,
                        "port": self._cfg.port,
                        "config_path": str(self._config_path),
                    },
                    "port_profile": port_profile_id,
                    "active_job": active.to_dict() if active is not None else None,
                    "active_job_status": job_status,
                    "active_jobs": {
                        str(profile_id): {
                            "active_job": profile_active.to_dict(),
                            "active_job_status": profile_status,
                        }
                        for profile_id, (profile_active, profile_status) in active_jobs.items()
                    },
                    "active_groups": {
                        group_name: payload
                        for group_name, payload in active_groups.items()
                    },
                    "allowed_port_profiles": {
                        str(profile_id): {
                            "label": profile.label,
                            "service_port": profile.vllm_port,
                            "jaeger_otlp_port": profile.jaeger_otlp_port,
                            "jaeger_ui_port": profile.jaeger_api_port,
                        }
                        for profile_id, profile in self._cfg.port_profiles.items()
                    },
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

    def group_status(self, *, group_name: str) -> CommandResult:
        normalized_group_name = group_name.strip()
        if not normalized_group_name:
            raise ControlPlaneError(
                message="group_name must be non-empty",
                code=87,
                http_status=400,
            )

        with self._lock:
            state = self._load_state()
            state_changed, active_jobs = self._refresh_all_active_jobs(state)
            if state_changed:
                self._save_state(state)

            group_entries = [
                (profile_id, active, status)
                for profile_id, (active, status) in active_jobs.items()
                if active.group_name == normalized_group_name
            ]
            group_entries.sort(key=lambda item: item[0])
            if not group_entries:
                raise ControlPlaneError(
                    message=f"no active services for group '{normalized_group_name}'",
                    code=78,
                    http_status=404,
                )

            job_ids = sorted({entry[1].job_id for entry in group_entries})
            return CommandResult(
                code=0,
                message="group status",
                data={
                    "group_name": normalized_group_name,
                    "job_ids": job_ids,
                    "profiles": [
                        {
                            "port_profile": profile_id,
                            "active_job": active.to_dict(),
                            "active_job_status": status,
                        }
                        for profile_id, active, status in group_entries
                    ],
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

    def _prepare_sif_files(self) -> list[str]:
        self._require_command("apptainer")
        actions: list[str] = []
        actions.append(
            self._pull_image_if_missing(
                name="jaeger",
                image=self._cfg.jaeger_image,
                sif_path=self._cfg.jaeger_sif,
                timeout_seconds=DEFAULT_JAEGER_PULL_TIMEOUT_SECONDS,
            )
        )
        if self._requires_default_vllm_sif():
            actions.append(
                self._pull_image_if_missing(
                    name="vllm",
                    image=self._cfg.vllm_image,
                    sif_path=self._cfg.vllm_sif,
                    timeout_seconds=DEFAULT_VLLM_PULL_TIMEOUT_SECONDS,
                )
            )
        else:
            actions.append(
                "skipped pulling default vllm image because all partitions have existing sif_img overrides"
            )
        self._ensure_sif_files()
        return actions

    def _pull_image_if_missing(
        self,
        *,
        name: str,
        image: str,
        sif_path: Path,
        timeout_seconds: int,
    ) -> str:
        if sif_path.exists():
            return f"found existing {name} SIF: {sif_path}"

        sif_path.parent.mkdir(parents=True, exist_ok=True)
        result = self._run(["apptainer", "pull", str(sif_path), image], timeout_seconds=timeout_seconds)
        if result.returncode != 0:
            raise ControlPlaneError(
                message=f"failed to pull {name} image",
                code=91,
                http_status=500,
                details={
                    "image": image,
                    "sif_path": str(sif_path),
                    "stdout": _truncate_text(result.stdout.strip()),
                    "stderr": _truncate_text(result.stderr.strip()),
                },
            )
        return f"pulled {name}: {image} -> {sif_path}"

    def _ensure_sif_files(self) -> None:
        missing: list[str] = []
        if not self._cfg.jaeger_sif.exists():
            missing.append(str(self._cfg.jaeger_sif))
        if self._requires_default_vllm_sif() and not self._cfg.vllm_sif.exists():
            missing.append(str(self._cfg.vllm_sif))
        if missing:
            raise ControlPlaneError(
                message="missing required SIF image files",
                code=91,
                http_status=400,
                details={
                    "missing": missing,
                },
            )

    def _requires_default_vllm_sif(self) -> bool:
        for partition_spec in self._cfg.partitions.values():
            if not self._partition_sif_override_exists(partition_spec):
                return True
        return False

    def _partition_sif_override_exists(self, partition_spec: PartitionSpec) -> bool:
        partition_vllm_sif = partition_spec.vllm_sif
        return partition_vllm_sif is not None and partition_vllm_sif.exists()

    def _effective_vllm_sif_path(self, partition_spec: PartitionSpec) -> Path:
        return _effective_partition_vllm_sif(
            partition_vllm_sif=partition_spec.vllm_sif,
            default_vllm_sif=self._cfg.vllm_sif,
        )

    def _ensure_partition_vllm_sif_file(self, partition_spec: PartitionSpec) -> Path:
        effective_vllm_sif = self._effective_vllm_sif_path(partition_spec)
        if not effective_vllm_sif.exists():
            raise ControlPlaneError(
                message=(
                    "missing required VLLM SIF image file for "
                    f"partition '{partition_spec.name}'"
                ),
                code=91,
                http_status=400,
                details={
                    "partition": partition_spec.name,
                    "missing_vllm_sif": str(effective_vllm_sif),
                    "partition_sif_img": (
                        str(partition_spec.vllm_sif) if partition_spec.vllm_sif is not None else None
                    ),
                    "default_vllm_sif": str(self._cfg.vllm_sif),
                },
            )
        return effective_vllm_sif

    def _empty_progress(self, *, message: str) -> dict[str, Any]:
        return {
            "status": "idle",
            "phase": None,
            "message": message,
            "job_id": None,
            "started_at": None,
            "finished_at": None,
            "updated_at": _utc_now_iso(),
        }

    def _progress_state(
        self,
        kind: str,
    ) -> tuple[threading.Lock, dict[int, dict[str, Any]]]:
        if kind == "start":
            return self._start_progress_lock, self._start_progress
        if kind == "stop":
            return self._stop_progress_lock, self._stop_progress
        raise ValueError(f"unknown progress kind: {kind}")

    def _progress_status(
        self,
        *,
        kind: str,
        port_profile_id: int,
        idle_message: str,
    ) -> CommandResult:
        lock, progress_store = self._progress_state(kind)
        with lock:
            progress = dict(
                progress_store.get(
                    port_profile_id,
                    self._empty_progress(message=idle_message),
                )
            )
        progress["port_profile"] = port_profile_id
        return CommandResult(code=0, message=f"{kind} status", data=progress)

    def _load_state(self) -> dict[str, Any]:
        if not self._cfg.state_file.exists():
            return {"active_jobs": {}, "history": []}
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
            return {"active_jobs": {}, "history": []}
        if "history" not in raw or not isinstance(raw["history"], list):
            raw["history"] = []
        if "active_jobs" not in raw or not isinstance(raw["active_jobs"], dict):
            migrated: dict[str, Any] = {}
            legacy_active = raw.get("active_job")
            if isinstance(legacy_active, dict):
                legacy_snapshot = dict(legacy_active)
                legacy_snapshot.setdefault("port_profile_id", 0)
                migrated[str(int(legacy_snapshot["port_profile_id"]))] = legacy_snapshot
            raw["active_jobs"] = migrated
        for entry in raw["history"]:
            if isinstance(entry, dict):
                entry.setdefault("port_profile_id", 0)
        raw.pop("active_job", None)
        return raw

    def _save_state(self, state: dict[str, Any]) -> None:
        self._cfg.state_file.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _active_jobs_table(self, state: dict[str, Any]) -> dict[str, Any]:
        active_jobs = state.get("active_jobs")
        if not isinstance(active_jobs, dict):
            active_jobs = {}
            state["active_jobs"] = active_jobs
        return active_jobs

    def _active_job_raw(self, state: dict[str, Any], *, port_profile_id: int) -> dict[str, Any] | None:
        active_jobs = self._active_jobs_table(state)
        active_raw = active_jobs.get(str(port_profile_id))
        if not isinstance(active_raw, dict):
            return None
        return active_raw

    def _set_active_job(
        self,
        state: dict[str, Any],
        *,
        port_profile_id: int,
        active_job: dict[str, Any],
    ) -> None:
        active_jobs = self._active_jobs_table(state)
        payload = dict(active_job)
        payload["port_profile_id"] = int(payload.get("port_profile_id", port_profile_id))
        active_jobs[str(port_profile_id)] = payload

    def _archive_active_job(self, state: dict[str, Any], *, port_profile_id: int, reason: str) -> None:
        active_jobs = self._active_jobs_table(state)
        active_raw = active_jobs.get(str(port_profile_id))
        if not isinstance(active_raw, dict):
            active_jobs.pop(str(port_profile_id), None)
            return
        snapshot = dict(active_raw)
        snapshot["ended_at"] = _utc_now_iso()
        snapshot["end_reason"] = reason
        history = state.setdefault("history", [])
        if isinstance(history, list):
            history.append(snapshot)
        active_jobs.pop(str(port_profile_id), None)

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
            state = {"active_jobs": {}}
        else:
            # Refresh persisted active jobs first so stale entries from a previous
            # server run do not keep old artifacts protected from archiving.
            try:
                state_changed, _ = self._refresh_all_active_jobs(state)
                if state_changed:
                    self._save_state(state)
            except ControlPlaneError:
                # Archiving should remain best-effort; failures here should not
                # block server startup.
                pass

        for active_raw in self._active_jobs_table(state).values():
            if not isinstance(active_raw, dict):
                continue
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
        self,
        state: dict[str, Any],
        *,
        port_profile_id: int,
    ) -> tuple[bool, ActiveJob | None, str | None]:
        active_raw = self._active_job_raw(state, port_profile_id=port_profile_id)
        if not isinstance(active_raw, dict):
            return (False, None, None)

        active = ActiveJob.from_dict(active_raw)
        job_status = self._slurm_job_status(active.job_id)
        missing_in_squeue_key = "missing_in_squeue_since"
        if job_status is None:
            stop_requested_at = active_raw.get("stop_requested_at")
            if isinstance(stop_requested_at, str) and stop_requested_at:
                self._archive_active_job(state, port_profile_id=port_profile_id, reason="stopped")
                return (True, None, None)

            missing_since_raw = active_raw.get(missing_in_squeue_key)
            missing_since = (
                _parse_iso_datetime(missing_since_raw)
                if isinstance(missing_since_raw, str) and missing_since_raw
                else None
            )
            if missing_since is None:
                active_payload = dict(active_raw)
                active_payload[missing_in_squeue_key] = _utc_now_iso()
                self._set_active_job(
                    state,
                    port_profile_id=port_profile_id,
                    active_job=active_payload,
                )
                return (True, active, "UNKNOWN")

            elapsed_seconds = max((datetime.now(timezone.utc) - missing_since).total_seconds(), 0.0)
            if elapsed_seconds < DEFAULT_SQUEUE_MISS_GRACE_SECONDS:
                return (False, active, "UNKNOWN")

            self._archive_active_job(state, port_profile_id=port_profile_id, reason="finished")
            return (True, None, None)

        if missing_in_squeue_key in active_raw:
            active_payload = dict(active_raw)
            active_payload.pop(missing_in_squeue_key, None)
            self._set_active_job(
                state,
                port_profile_id=port_profile_id,
                active_job=active_payload,
            )
            return (True, active, job_status)

        return (False, active, job_status)

    def _refresh_all_active_jobs(
        self,
        state: dict[str, Any],
    ) -> tuple[bool, dict[int, tuple[ActiveJob, str | None]]]:
        state_changed = False
        active_jobs: dict[int, tuple[ActiveJob, str | None]] = {}
        for profile_key in sorted(self._active_jobs_table(state).keys(), key=int):
            state_changed_for_profile, active, job_status = self._refresh_active_job(
                state,
                port_profile_id=int(profile_key),
            )
            state_changed = state_changed or state_changed_for_profile
            if active is not None:
                active_jobs[int(profile_key)] = (active, job_status)
        return state_changed, active_jobs

    def _active_group_entries_raw(
        self,
        state: dict[str, Any],
        *,
        group_name: str,
    ) -> list[tuple[int, dict[str, Any]]]:
        normalized_group_name = group_name.strip()
        entries: list[tuple[int, dict[str, Any]]] = []
        for profile_key, active_raw in self._active_jobs_table(state).items():
            if not isinstance(active_raw, dict):
                continue
            active_group_name = active_raw.get("group_name")
            if not isinstance(active_group_name, str) or active_group_name.strip() != normalized_group_name:
                continue
            entries.append((int(profile_key), active_raw))
        entries.sort(key=lambda item: item[0])
        return entries

    def _active_group_entries(
        self,
        state: dict[str, Any],
        *,
        group_name: str,
    ) -> list[tuple[int, ActiveJob]]:
        return [
            (profile_id, ActiveJob.from_dict(active_raw))
            for profile_id, active_raw in self._active_group_entries_raw(state, group_name=group_name)
        ]

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

    def _require_port_profile(self, port_profile_id: int) -> AMDHPCPortProfile:
        if isinstance(port_profile_id, bool):
            raise ControlPlaneError(
                message="port_profile must be an integer",
                code=16,
                http_status=400,
            )
        profile = self._cfg.port_profiles.get(int(port_profile_id))
        if profile is None:
            raise ControlPlaneError(
                message=f"unknown port profile '{port_profile_id}'",
                code=17,
                http_status=400,
                details={"allowed_port_profiles": sorted(self._cfg.port_profiles.keys())},
            )
        return profile

    def _ensure_single_profile_command_allowed(
        self,
        state: dict[str, Any],
        *,
        port_profile_id: int,
        command_name: str,
    ) -> None:
        active_raw = self._active_job_raw(state, port_profile_id=port_profile_id)
        if not isinstance(active_raw, dict):
            return
        group_name = active_raw.get("group_name")
        if not isinstance(group_name, str) or not group_name.strip():
            return
        group_profiles_raw = active_raw.get("group_profiles")
        group_profiles = (
            [
                int(value)
                for value in group_profiles_raw
                if isinstance(value, int) and not isinstance(value, bool)
            ]
            if isinstance(group_profiles_raw, list)
            else []
        )
        raise ControlPlaneError(
            message=(
                f"{command_name} for a single --port-profile is not allowed while "
                f"profile {port_profile_id} belongs to group '{group_name}'. "
                "Use group commands instead."
            ),
            code=82,
            http_status=409,
            details={
                "port_profile": port_profile_id,
                "group_name": group_name,
                "group_profiles": group_profiles,
            },
        )

    def _ensure_port_available_on_login(self, port: int, *, label: str) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise ControlPlaneError(
                    message=(
                        f"{label} {port} is already in use on login node; "
                        "choose a different port profile"
                    ),
                    code=15,
                    http_status=409,
                    details={"label": label, "port": port, "error": str(exc)},
                ) from exc

    def _ensure_profile_ports_available_on_login(self, port_profile: AMDHPCPortProfile) -> None:
        self._ensure_port_available_on_login(port_profile.vllm_port, label="service_port")
        self._ensure_port_available_on_login(port_profile.jaeger_otlp_port, label="jaeger_otlp_port")
        self._ensure_port_available_on_login(port_profile.jaeger_api_port, label="jaeger_ui_port")

    def _gateway_ports_for_profile(self, *, port_profile_id: int) -> tuple[int, int]:
        config_path = (self._cfg.repo_root / "configs" / "port_profiles.toml").resolve()
        try:
            payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ControlPlaneError(
                message=f"port profiles config not found: {config_path}",
                code=124,
                http_status=500,
            ) from exc
        except tomllib.TOMLDecodeError as exc:
            raise ControlPlaneError(
                message=f"failed to parse port profiles config: {config_path}",
                code=124,
                http_status=500,
                details={"error": str(exc)},
            ) from exc

        profiles = payload.get("profiles")
        if not isinstance(profiles, dict):
            raise ControlPlaneError(
                message=f"port profiles config is missing [profiles]: {config_path}",
                code=124,
                http_status=500,
            )

        raw_profile = profiles.get(str(port_profile_id))
        if not isinstance(raw_profile, dict):
            raise ControlPlaneError(
                message=f"missing port profile {port_profile_id} in {config_path}",
                code=124,
                http_status=500,
            )

        gateway_port_raw = raw_profile.get("gateway_port")
        gateway_parse_port_raw = raw_profile.get("gateway_parse_port")
        if isinstance(gateway_port_raw, bool) or not isinstance(gateway_port_raw, int):
            raise ControlPlaneError(
                message=f"profiles.{port_profile_id}.gateway_port must be an integer in {config_path}",
                code=124,
                http_status=500,
            )
        if isinstance(gateway_parse_port_raw, bool) or not isinstance(gateway_parse_port_raw, int):
            raise ControlPlaneError(
                message=(
                    f"profiles.{port_profile_id}.gateway_parse_port must be an integer in "
                    f"{config_path}"
                ),
                code=124,
                http_status=500,
            )

        gateway_port = int(gateway_port_raw)
        gateway_parse_port = int(gateway_parse_port_raw)
        for key, value in (
            ("gateway_port", gateway_port),
            ("gateway_parse_port", gateway_parse_port),
        ):
            if value < 1 or value > 65535:
                raise ControlPlaneError(
                    message=(
                        f"profiles.{port_profile_id}.{key} must be in range 1..65535 in "
                        f"{config_path}"
                    ),
                    code=124,
                    http_status=500,
                )

        return gateway_port, gateway_parse_port

    def _collect_readiness_snapshot(self, *, port_profile_id: int) -> dict[str, Any]:
        port_profile = self._require_port_profile(port_profile_id)
        state = self._load_state()
        self._ensure_single_profile_command_allowed(
            state,
            port_profile_id=port_profile.profile_id,
            command_name="up/wait-up",
        )
        state_changed, active, job_status = self._refresh_active_job(state, port_profile_id=port_profile.profile_id)
        if state_changed:
            self._save_state(state)

        vllm_url = f"http://127.0.0.1:{port_profile.vllm_port}/v1/models"
        jaeger_url = f"http://127.0.0.1:{port_profile.jaeger_api_port}"

        vllm_probe = self._probe_http_url(vllm_url)
        jaeger_probe = self._probe_http_url(jaeger_url)

        active_job_payload = active.to_dict() if active is not None else None
        active_running = job_status in {"RUNNING", "COMPLETING"}
        ready = bool(active_running and vllm_probe["ok"] and jaeger_probe["ok"])

        return {
            "port_profile": port_profile.profile_id,
            "ready": ready,
            "active_job": active_job_payload,
            "active_job_status": job_status,
            "vllm": vllm_probe,
            "jaeger": jaeger_probe,
        }

    def _wait_for_services_up(
        self,
        *,
        port_profile_id: int,
        timeout_seconds: int,
        poll_interval_seconds: float,
        expected_job_id: str | None,
        defer_timeout_until_running: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        timeout_started = started if not defer_timeout_until_running else None
        timeout_started_at_iso = _utc_now_iso() if timeout_started is not None else None
        deadline = (timeout_started + timeout_seconds) if timeout_started is not None else None
        last_snapshot: dict[str, Any] | None = None

        while True:
            with self._lock:
                snapshot = self._collect_readiness_snapshot(port_profile_id=port_profile_id)
            last_snapshot = snapshot

            status_raw = snapshot.get("active_job_status")
            status_text = status_raw if isinstance(status_raw, str) else None
            if (
                defer_timeout_until_running
                and deadline is None
                and (
                    _is_slurm_running_state(status_text)
                    or (
                        expected_job_id is None
                        and not _is_slurm_pending_state(status_text)
                    )
                )
            ):
                timeout_started = time.monotonic()
                timeout_started_at_iso = _utc_now_iso()
                deadline = timeout_started + timeout_seconds

            snapshot["defer_timeout_until_running"] = defer_timeout_until_running
            snapshot["timeout_started"] = deadline is not None
            snapshot["timeout_status_gate"] = "RUNNING"
            snapshot["timeout_started_at_status"] = status_text
            snapshot["startup_timeout_started_at"] = timeout_started_at_iso

            if progress_callback is not None:
                progress_callback(snapshot)

            if expected_job_id is not None:
                active_job = snapshot.get("active_job")
                active_job_id = active_job.get("job_id") if isinstance(active_job, dict) else None
                if active_job_id != expected_job_id:
                    if active_job_id is None:
                        raise ControlPlaneError(
                            message=(
                                f"job {expected_job_id} disappeared while waiting for readiness; "
                                "it may have been cancelled or exited"
                            ),
                            code=65,
                            http_status=409,
                            details={"expected_job_id": expected_job_id, "last_snapshot": snapshot},
                        )
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
                if timeout_started is not None:
                    snapshot["startup_waited_seconds"] = round(
                        max(time.monotonic() - timeout_started, 0.0),
                        3,
                    )
                else:
                    snapshot["startup_waited_seconds"] = 0.0
                return snapshot

            if deadline is not None and time.monotonic() >= deadline:
                raise ControlPlaneError(
                    message=f"timed out waiting for services to be up after {timeout_seconds}s",
                    code=63,
                    http_status=408,
                    details={
                        "timeout_seconds": timeout_seconds,
                        "poll_interval_seconds": poll_interval_seconds,
                        "defer_timeout_until_running": defer_timeout_until_running,
                        "timeout_started_at_status": status_text,
                        "last_snapshot": last_snapshot,
                    },
                )

            time.sleep(poll_interval_seconds)

    def _collect_group_readiness_snapshot(
        self,
        *,
        group_name: str,
        port_profile_ids: list[int],
        expected_job_id: str | None,
    ) -> dict[str, Any]:
        state = self._load_state()
        profiles_payload: list[dict[str, Any]] = []
        state_changed = False
        all_ready = True
        all_running = True

        for port_profile_id in port_profile_ids:
            state_changed_for_profile, active, job_status = self._refresh_active_job(
                state,
                port_profile_id=port_profile_id,
            )
            state_changed = state_changed or state_changed_for_profile
            if active is None:
                latest_history_entry: dict[str, Any] | None = None
                history = state.get("history")
                if isinstance(history, list):
                    for entry in reversed(history):
                        if not isinstance(entry, dict):
                            continue
                        raw_profile_id = entry.get("port_profile_id")
                        if isinstance(raw_profile_id, int) and raw_profile_id == port_profile_id:
                            latest_history_entry = dict(entry)
                            break
                raise ControlPlaneError(
                    message=(
                        f"group '{group_name}' profile {port_profile_id} no longer has an active job"
                    ),
                    code=83,
                    http_status=409,
                    details={
                        "group_name": group_name,
                        "port_profile": port_profile_id,
                        "latest_history_entry": latest_history_entry,
                    },
                )
            if active.group_name != group_name:
                raise ControlPlaneError(
                    message=(
                        f"profile {port_profile_id} active job does not belong to "
                        f"group '{group_name}'"
                    ),
                    code=84,
                    http_status=409,
                    details={
                        "port_profile": port_profile_id,
                        "active_group_name": active.group_name,
                        "expected_group_name": group_name,
                    },
                )
            if expected_job_id is not None and active.job_id != expected_job_id:
                raise ControlPlaneError(
                    message=(
                        f"profile {port_profile_id} active job changed while waiting for "
                        f"group '{group_name}' readiness"
                    ),
                    code=85,
                    http_status=409,
                    details={
                        "port_profile": port_profile_id,
                        "expected_job_id": expected_job_id,
                        "active_job_id": active.job_id,
                    },
                )

            port_profile = self._require_port_profile(port_profile_id)
            vllm_url = f"http://127.0.0.1:{port_profile.vllm_port}/v1/models"
            jaeger_url = f"http://127.0.0.1:{port_profile.jaeger_api_port}"
            vllm_probe = self._probe_http_url(vllm_url)
            jaeger_probe = self._probe_http_url(jaeger_url)

            active_running = _is_slurm_running_state(job_status)
            profile_ready = bool(active_running and vllm_probe["ok"] and jaeger_probe["ok"])
            all_running = bool(all_running and active_running)
            all_ready = bool(all_ready and profile_ready)
            profiles_payload.append(
                {
                    "port_profile": port_profile_id,
                    "group_name": group_name,
                    "ready": profile_ready,
                    "active_job": active.to_dict(),
                    "active_job_status": job_status,
                    "vllm": vllm_probe,
                    "jaeger": jaeger_probe,
                }
            )

        if state_changed:
            self._save_state(state)

        return {
            "group_name": group_name,
            "ready": all_ready,
            "all_running": all_running,
            "profiles": profiles_payload,
        }

    def _wait_for_group_services_up(
        self,
        *,
        group_name: str,
        port_profile_ids: list[int],
        timeout_seconds: int,
        poll_interval_seconds: float,
        expected_job_id: str | None,
        defer_timeout_until_running: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        timeout_started_at: float | None = None
        timeout_started_at_status: str | None = None
        last_snapshot: dict[str, Any] | None = None

        while True:
            with self._lock:
                snapshot = self._collect_group_readiness_snapshot(
                    group_name=group_name,
                    port_profile_ids=port_profile_ids,
                    expected_job_id=expected_job_id,
                )

            waited_seconds = round(time.monotonic() - started, 3)
            snapshot["waited_seconds"] = waited_seconds
            last_snapshot = snapshot
            if progress_callback is not None:
                progress_callback(snapshot)

            if snapshot["ready"]:
                return snapshot

            all_running = bool(snapshot.get("all_running"))
            if timeout_started_at is None:
                if defer_timeout_until_running:
                    if all_running:
                        timeout_started_at = time.monotonic()
                        timeout_started_at_status = "all_running"
                else:
                    timeout_started_at = started
                    timeout_started_at_status = "timeout_from_submit"

            if timeout_started_at is not None:
                elapsed_from_timeout_start = time.monotonic() - timeout_started_at
                if elapsed_from_timeout_start > timeout_seconds:
                    raise ControlPlaneError(
                        message=(
                            f"timed out waiting for grouped services to become ready "
                            f"(group={group_name}, waited={elapsed_from_timeout_start:.1f}s)"
                        ),
                        code=86,
                        http_status=504,
                        details={
                            "group_name": group_name,
                            "timeout_seconds": timeout_seconds,
                            "poll_interval_seconds": poll_interval_seconds,
                            "defer_timeout_until_running": defer_timeout_until_running,
                            "timeout_started_at_status": timeout_started_at_status,
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

    def _write_group_multi_node_sbatch_script(
        self,
        *,
        partition_spec: PartitionSpec,
        model_spec: ModelSpec,
        port_profiles: list[AMDHPCPortProfile],
        group_name: str,
        extra_env: dict[str, str],
        lmcache_enabled: bool,
        extra_vllm_args: list[str],
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_partition = _safe_token(partition_spec.name)
        safe_model = _safe_token(model_spec.name)
        effective_vllm_sif = self._effective_vllm_sif_path(partition_spec)
        safe_group_name = _safe_token(group_name)
        script_path = (
            self._cfg.run_dir
            / f"sbatch-{timestamp}-gm{safe_group_name}-{safe_partition}-{safe_model}.sh"
        )

        group_size = len(port_profiles)
        profile_ids_csv = ",".join(str(profile.profile_id) for profile in port_profiles)
        vllm_ports_csv = ",".join(str(profile.vllm_port) for profile in port_profiles)
        jaeger_otlp_ports_csv = ",".join(str(profile.jaeger_otlp_port) for profile in port_profiles)
        jaeger_ui_ports_csv = ",".join(str(profile.jaeger_api_port) for profile in port_profiles)
        lmcache_ports_csv = ",".join(str(profile.lmcache_port) for profile in port_profiles)

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
        effective_extra_args = _effective_vllm_extra_args(
            extra_args=[*model_spec.extra_args, *extra_vllm_args],
            gpus_per_node=partition_spec.gpus_per_node,
        )
        encoded_extra_args = _encode_model_extra_args(effective_extra_args)
        force_seq_trust_remote_code = self._cfg.env.get("VLLM_FORCE_SEQ_TRUST_REMOTE_CODE")
        if force_seq_trust_remote_code is None:
            force_seq_trust_remote_code = "true"
        extra_apptainer_env_flags = _render_apptainer_extra_env_flags(
            extra_env=extra_env,
            indent="                ",
        )
        lmcache_kv_transfer_args = ""
        if lmcache_enabled:
            lmcache_kv_transfer_args = (
                "                --kv-transfer-config\n"
                "                '{\"kv_connector\":\"LMCacheConnectorV1\", \"kv_role\":\"kv_both\"}'\n"
            )

        script = textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            #SBATCH --job-name={_safe_token(f'{self._cfg.job_name_prefix}m_{safe_group_name}')}
            #SBATCH --output={slurm_out}
            #SBATCH --error={slurm_err}
            #SBATCH --nodes={group_size}
            #SBATCH --ntasks={group_size}
            #SBATCH --ntasks-per-node=1
            #SBATCH --time={partition_spec.max_time}
            #SBATCH --partition={partition_spec.name}

            set -euo pipefail

            echo "Multi-node grouped job ${{SLURM_JOB_ID}} (group={shlex.quote(group_name)}) starting at $(date)"
            echo "SLURM_JOB_NODELIST=${{SLURM_JOB_NODELIST:-unknown}}"

            LOGIN_HOST={shlex.quote(self._cfg.login_host)}
            GROUP_NAME={shlex.quote(group_name)}
            GROUP_SIZE={group_size}
            GROUP_PROFILE_IDS_CSV={shlex.quote(profile_ids_csv)}
            GROUP_VLLM_PORTS_CSV={shlex.quote(vllm_ports_csv)}
            GROUP_JAEGER_OTLP_LOGIN_PORTS_CSV={shlex.quote(jaeger_otlp_ports_csv)}
            GROUP_JAEGER_UI_LOGIN_PORTS_CSV={shlex.quote(jaeger_ui_ports_csv)}
            GROUP_LMCACHE_INTERNAL_API_SERVER_PORT_START_CSV={shlex.quote(lmcache_ports_csv)}
            GROUP_VLLM_TENSOR_PARALLEL_SIZE={partition_spec.gpus_per_node}
            GROUP_VLLM_VISIBLE_DEVICES={shlex.quote(visible_devices)}
            JAEGER_OTLP_LOCAL_PORT={DEFAULT_JAEGER_OTLP_GRPC_PORT}
            JAEGER_UI_LOCAL_PORT={DEFAULT_JAEGER_QUERY_PORT}

            JAEGER_SIF={shlex.quote(str(self._cfg.jaeger_sif))}
            VLLM_SIF={shlex.quote(str(effective_vllm_sif))}

            VLLM_MODEL_NAME={shlex.quote(model_spec.vllm_model_name)}
            VLLM_SERVED_MODEL_NAME={shlex.quote(model_spec.served_model_name)}

            VLLM_APPTAINER_HOME={shlex.quote(self._cfg.env.get('VLLM_APPTAINER_HOME', ''))}
            HF_HOME={shlex.quote(self._cfg.env.get('HF_HOME', ''))}
            HF_HUB_CACHE={shlex.quote(self._cfg.env.get('HF_HUB_CACHE', ''))}
            HF_HUB_OFFLINE=1
            TRANSFORMERS_OFFLINE=1
            HF_TOKEN="${{HF_TOKEN:-}}"
            export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

            AITER_JIT_DIR={shlex.quote(aiter_jit_dir)}
            XDG_CACHE_HOME={shlex.quote(xdg_cache_home)}
            VLLM_CACHE_ROOT={shlex.quote(vllm_cache_root)}

            OTEL_SERVICE_NAME={shlex.quote(self._cfg.env.get('OTEL_SERVICE_NAME', 'vllm-server'))}
            OTEL_EXPORTER_OTLP_TRACES_INSECURE={shlex.quote(self._cfg.env.get('OTEL_EXPORTER_OTLP_TRACES_INSECURE', 'true'))}
            OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="grpc://127.0.0.1:${{JAEGER_OTLP_LOCAL_PORT}}"
            VLLM_COLLECT_DETAILED_TRACES={shlex.quote(self._cfg.env.get('VLLM_COLLECT_DETAILED_TRACES', 'all'))}
            VLLM_LOGITS_PROCESSORS={shlex.quote(self._cfg.env.get('VLLM_LOGITS_PROCESSORS', 'forceSeq.force_sequence_logits_processor:ForceSequenceAdapter'))}
            VLLM_MODEL_EXTRA_ARGS_B64={shlex.quote(encoded_extra_args)}
            VLLM_FORCE_SEQ_TRUST_REMOTE_CODE={shlex.quote(force_seq_trust_remote_code)}

            REPO_ROOT={shlex.quote(str(self._cfg.repo_root))}
            JOB_LOG_DIR={shlex.quote(str(self._cfg.log_dir))}
            mkdir -p "${{JOB_LOG_DIR}}" "${{AITER_JIT_DIR}}" "${{XDG_CACHE_HOME}}" "${{VLLM_CACHE_ROOT}}"

            run_many_worker() {{
              set -euo pipefail
              local SSH_OPTIONS=({ssh_options})

              local index="${{SLURM_PROCID:-${{SLURM_NODEID:-}}}}"
              if [[ -z "${{index}}" ]]; then
                echo "Missing SLURM_PROCID/SLURM_NODEID for many worker." >&2
                exit 92
              fi

              IFS=',' read -r -a PROFILE_IDS <<<"${{GROUP_PROFILE_IDS_CSV}}"
              IFS=',' read -r -a VLLM_PORTS <<<"${{GROUP_VLLM_PORTS_CSV}}"
              IFS=',' read -r -a JAEGER_OTLP_PORTS <<<"${{GROUP_JAEGER_OTLP_LOGIN_PORTS_CSV}}"
              IFS=',' read -r -a JAEGER_UI_PORTS <<<"${{GROUP_JAEGER_UI_LOGIN_PORTS_CSV}}"
              IFS=',' read -r -a LMCACHE_PORT_STARTS <<<"${{GROUP_LMCACHE_INTERNAL_API_SERVER_PORT_START_CSV}}"

              if (( index < 0 || index >= GROUP_SIZE )); then
                echo "Worker index ${{index}} out of range for group size ${{GROUP_SIZE}}." >&2
                exit 93
              fi

              if (( ${{#PROFILE_IDS[@]}} != GROUP_SIZE )) || (( ${{#VLLM_PORTS[@]}} != GROUP_SIZE )) || (( ${{#JAEGER_OTLP_PORTS[@]}} != GROUP_SIZE )) || (( ${{#JAEGER_UI_PORTS[@]}} != GROUP_SIZE )) || (( ${{#LMCACHE_PORT_STARTS[@]}} != GROUP_SIZE )); then
                echo "Grouped CSV payload length mismatch (expected ${{GROUP_SIZE}} entries)." >&2
                exit 94
              fi

              local PROFILE_ID="${{PROFILE_IDS[${{index}}]}}"
              local VLLM_SERVICE_PORT="${{VLLM_PORTS[${{index}}]}}"
              local JAEGER_OTLP_LOGIN_PORT="${{JAEGER_OTLP_PORTS[${{index}}]}}"
              local JAEGER_UI_LOGIN_PORT="${{JAEGER_UI_PORTS[${{index}}]}}"
              local LMCACHE_INTERNAL_API_SERVER_PORT_START="${{LMCACHE_PORT_STARTS[${{index}}]}}"
              local JAEGER_LOG="${{JOB_LOG_DIR}}/jaeger.${{SLURM_JOB_ID}}.p${{PROFILE_ID}}.log"
              local VLLM_LOG="${{JOB_LOG_DIR}}/vllm.${{SLURM_JOB_ID}}.p${{PROFILE_ID}}.log"

              local APPTAINER_HOME_ARGS=()
              if [[ -n "${{VLLM_APPTAINER_HOME}}" ]]; then
                mkdir -p "${{VLLM_APPTAINER_HOME}}"
                APPTAINER_HOME_ARGS=(-H "${{VLLM_APPTAINER_HOME}}")
              fi

              local BIND_ARGS=()
              if [[ -n "${{HF_HOME}}" ]]; then
                BIND_ARGS+=(--bind "${{HF_HOME}}:${{HF_HOME}}")
              fi
              if [[ -n "${{HF_HUB_CACHE}}" ]]; then
                BIND_ARGS+=(--bind "${{HF_HUB_CACHE}}:${{HF_HUB_CACHE}}")
              fi

              local TUNNEL_PIDS=()
              start_tunnel() {{
                local remote_port="$1"
                local local_port="$2"
                local retries=3
                local attempt=0
                while true; do
                  ssh "${{SSH_OPTIONS[@]}}" -N -R "${{remote_port}}:127.0.0.1:${{local_port}}" "${{LOGIN_HOST}}" &
                  local tunnel_pid=$!
                  sleep 1
                  if kill -0 "${{tunnel_pid}}" >/dev/null 2>&1; then
                    TUNNEL_PIDS+=("${{tunnel_pid}}")
                    return 0
                  fi

                  wait "${{tunnel_pid}}" >/dev/null 2>&1 || true
                  attempt=$((attempt + 1))
                  if (( attempt >= retries )); then
                    echo "Reverse tunnel failed after ${{retries}} attempts (group=${{GROUP_NAME}} profile=${{PROFILE_ID}} remote=${{remote_port}} local=${{local_port}})." >&2
                    return 1
                  fi
                  echo "Reverse tunnel startup failed (attempt ${{attempt}}/${{retries}}) for group=${{GROUP_NAME}} profile=${{PROFILE_ID}}; retrying." >&2
                  sleep 1
                done
              }}

              verify_tunnels_alive() {{
                local failed=0
                for pid in "${{TUNNEL_PIDS[@]}}"; do
                  if ! kill -0 "${{pid}}" >/dev/null 2>&1; then
                    echo "Reverse tunnel process exited before startup completed (pid=${{pid}})." >&2
                    failed=1
                  fi
                done
                return "${{failed}}"
              }}

              cleanup_worker() {{
                set +e
                if [[ -n "${{VLLM_PID:-}}" ]]; then kill "${{VLLM_PID}}" >/dev/null 2>&1 || true; fi
                if [[ -n "${{JAEGER_PID:-}}" ]]; then kill "${{JAEGER_PID}}" >/dev/null 2>&1 || true; fi
                for pid in "${{TUNNEL_PIDS[@]}}"; do
                  kill "${{pid}}" >/dev/null 2>&1 || true
                done
              }}
              trap cleanup_worker EXIT INT TERM

              echo "group=${{GROUP_NAME}} profile=${{PROFILE_ID}} node=$(hostname) proc=${{index}} vllm_port=${{VLLM_SERVICE_PORT}}"

              start_tunnel "${{VLLM_SERVICE_PORT}}" "${{VLLM_SERVICE_PORT}}"
              start_tunnel "${{JAEGER_OTLP_LOGIN_PORT}}" "${{JAEGER_OTLP_LOCAL_PORT}}"
              start_tunnel "${{JAEGER_UI_LOGIN_PORT}}" "${{JAEGER_UI_LOCAL_PORT}}"
              sleep 1
              if ! verify_tunnels_alive; then
                echo "One or more reverse tunnels failed to establish. Aborting startup." >&2
                exit 72
              fi

              apptainer run \
                --cleanenv \
                "${{APPTAINER_HOME_ARGS[@]}}" \
                --env COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
                "${{JAEGER_SIF}}" \
                >"${{JAEGER_LOG}}" 2>&1 &
              JAEGER_PID=$!

              local VLLM_VISIBLE_DEVICES="${{GROUP_VLLM_VISIBLE_DEVICES}}"
              local VLLM_HIP_VISIBLE_DEVICES="0"
              if [[ "${{VLLM_VISIBLE_DEVICES}}" == *,* ]]; then
                IFS=',' read -r -a VLLM_VISIBLE_DEVICE_LIST <<<"${{VLLM_VISIBLE_DEVICES}}"
                mapped_devices=()
                for i in "${{!VLLM_VISIBLE_DEVICE_LIST[@]}}"; do
                  mapped_devices+=("${{i}}")
                done
                IFS=',' VLLM_HIP_VISIBLE_DEVICES="${{mapped_devices[*]}}"
              fi

              local OTEL_SERVICE_NAME_WORKER="${{OTEL_SERVICE_NAME}}-m${{GROUP_NAME}}-p${{PROFILE_ID}}"
              VLLM_CMD=(
                /opt/vllm-plugins/vllm_entrypoint.sh
                --model "${{VLLM_MODEL_NAME}}"
                --served-model-name "${{VLLM_SERVED_MODEL_NAME}}"
                --port "${{VLLM_SERVICE_PORT}}"
                --tensor-parallel-size "${{GROUP_VLLM_TENSOR_PARALLEL_SIZE}}"
                --otlp-traces-endpoint "${{OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}}"
                --collect-detailed-traces "${{VLLM_COLLECT_DETAILED_TRACES}}"
                --enable-prompt-tokens-details
                --logits-processors "${{VLLM_LOGITS_PROCESSORS}}"
{lmcache_kv_transfer_args}              )

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
                --env HF_HUB_OFFLINE="${{HF_HUB_OFFLINE}}" \
                --env TRANSFORMERS_OFFLINE="${{TRANSFORMERS_OFFLINE}}" \
                --env HF_TOKEN="${{HF_TOKEN}}" \
                --env OTEL_SERVICE_NAME="${{OTEL_SERVICE_NAME_WORKER}}" \
                --env OTEL_EXPORTER_OTLP_TRACES_INSECURE="${{OTEL_EXPORTER_OTLP_TRACES_INSECURE}}" \
                --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${{OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}}" \
                --env HIP_VISIBLE_DEVICES="${{VLLM_HIP_VISIBLE_DEVICES}}" \
                --env ROCR_VISIBLE_DEVICES="${{VLLM_VISIBLE_DEVICES}}" \
                --env VLLM_MODEL_NAME="${{VLLM_MODEL_NAME}}" \
                --env VLLM_MODEL_EXTRA_ARGS_B64="${{VLLM_MODEL_EXTRA_ARGS_B64}}" \
                --env VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${{VLLM_FORCE_SEQ_TRUST_REMOTE_CODE}}" \
{extra_apptainer_env_flags}                --env LMCACHE_INTERNAL_API_SERVER_ENABLED=1 \
                --env PYTHONHASHSEED=0 \
                --env LMCACHE_INTERNAL_API_SERVER_PORT_START="${{LMCACHE_INTERNAL_API_SERVER_PORT_START}}" \
                "${{VLLM_SIF}}" \
                "${{VLLM_CMD[@]}}" \
                >"${{VLLM_LOG}}" 2>&1 &
              VLLM_PID=$!

              WAIT_PIDS=("${{JAEGER_PID}}" "${{VLLM_PID}}" "${{TUNNEL_PIDS[@]}}")
              wait -n "${{WAIT_PIDS[@]}}"
              EXIT_CODE=$?
              echo "group=${{GROUP_NAME}} profile=${{PROFILE_ID}} process exited with code ${{EXIT_CODE}} at $(date)."
              exit "${{EXIT_CODE}}"
            }}

            export LOGIN_HOST GROUP_NAME GROUP_SIZE
            export GROUP_PROFILE_IDS_CSV GROUP_VLLM_PORTS_CSV
            export GROUP_JAEGER_OTLP_LOGIN_PORTS_CSV GROUP_JAEGER_UI_LOGIN_PORTS_CSV
            export GROUP_LMCACHE_INTERNAL_API_SERVER_PORT_START_CSV
            export GROUP_VLLM_TENSOR_PARALLEL_SIZE GROUP_VLLM_VISIBLE_DEVICES
            export JAEGER_OTLP_LOCAL_PORT JAEGER_UI_LOCAL_PORT
            export JAEGER_SIF VLLM_SIF VLLM_MODEL_NAME VLLM_SERVED_MODEL_NAME
            export VLLM_APPTAINER_HOME HF_HOME HF_HUB_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_TOKEN
            export AITER_JIT_DIR XDG_CACHE_HOME VLLM_CACHE_ROOT
            export OTEL_SERVICE_NAME OTEL_EXPORTER_OTLP_TRACES_INSECURE OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
            export VLLM_COLLECT_DETAILED_TRACES VLLM_LOGITS_PROCESSORS
            export VLLM_MODEL_EXTRA_ARGS_B64 VLLM_FORCE_SEQ_TRUST_REMOTE_CODE
            export REPO_ROOT JOB_LOG_DIR
            export -f run_many_worker

            srun \
              --nodes="${{GROUP_SIZE}}" \
              --ntasks="${{GROUP_SIZE}}" \
              --ntasks-per-node=1 \
              --kill-on-bad-exit=1 \
              bash -lc 'run_many_worker'
            """
        )

        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o750)
        return script_path

    def _write_group_sbatch_script(
        self,
        *,
        partition_spec: PartitionSpec,
        model_spec: ModelSpec,
        port_profiles: list[AMDHPCPortProfile],
        group_name: str,
        gpus_per_profile: int,
        visible_devices_by_profile: list[str],
        extra_env: dict[str, str],
        lmcache_enabled: bool,
        extra_vllm_args: list[str],
        no_async_scheduling: bool = False,
        local_mode_script: str | None = None,
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_partition = _safe_token(partition_spec.name)
        safe_model = _safe_token(model_spec.name)
        effective_vllm_sif = self._effective_vllm_sif_path(partition_spec)
        safe_group_name = _safe_token(group_name)
        script_path = (
            self._cfg.run_dir
            / f"sbatch-{timestamp}-g{safe_group_name}-{safe_partition}-{safe_model}.sh"
        )

        profile_ids_csv = ",".join(str(profile.profile_id) for profile in port_profiles)
        vllm_ports_csv = ",".join(str(profile.vllm_port) for profile in port_profiles)
        jaeger_otlp_ports_csv = ",".join(str(profile.jaeger_otlp_port) for profile in port_profiles)
        jaeger_ui_ports_csv = ",".join(str(profile.jaeger_api_port) for profile in port_profiles)
        lmcache_ports_csv = ",".join(str(profile.lmcache_port) for profile in port_profiles)
        gateway_ports: list[int] = []
        gateway_parse_ports: list[int] = []
        for profile in port_profiles:
            gateway_port, gateway_parse_port = self._gateway_ports_for_profile(
                port_profile_id=profile.profile_id
            )
            gateway_ports.append(gateway_port)
            gateway_parse_ports.append(gateway_parse_port)
        gateway_ports_csv = ",".join(str(port) for port in gateway_ports)
        gateway_parse_ports_csv = ",".join(str(port) for port in gateway_parse_ports)
        group_size = len(port_profiles)
        if len(visible_devices_by_profile) != group_size:
            raise ControlPlaneError(
                message=(
                    "visible_devices_by_profile must match group size "
                    f"(expected {group_size}, got {len(visible_devices_by_profile)})"
                ),
                code=121,
                http_status=500,
            )
        group_visible_devices_semicolon = ";".join(visible_devices_by_profile)
        slurm_out = self._cfg.log_dir / "slurm.%j.out"
        slurm_err = self._cfg.log_dir / "slurm.%j.err"

        tmp_root = os.environ.get("TMPDIR", "/tmp")
        user_name = os.environ.get("USER", "user")
        aiter_jit_dir = self._cfg.env.get("AITER_JIT_DIR", f"{tmp_root}/vllm-aiter-jit-{user_name}")
        runtime_root = self._cfg.env.get("VLLM_RUNTIME_ROOT", f"{tmp_root}/vllm-runtime-{user_name}")
        xdg_cache_home = self._cfg.env.get("XDG_CACHE_HOME", f"{runtime_root}/xdg-cache")
        vllm_cache_root = self._cfg.env.get("VLLM_CACHE_ROOT", f"{xdg_cache_home}/vllm")

        ssh_options = " ".join(shlex.quote(opt) for opt in self._cfg.ssh_options)
        effective_extra_args = _effective_vllm_extra_args(
            extra_args=[*model_spec.extra_args, *extra_vllm_args],
            gpus_per_node=gpus_per_profile,
        )
        encoded_extra_args = _encode_model_extra_args(effective_extra_args)
        force_seq_trust_remote_code = self._cfg.env.get("VLLM_FORCE_SEQ_TRUST_REMOTE_CODE")
        if force_seq_trust_remote_code is None:
            force_seq_trust_remote_code = "true"
        extra_apptainer_env_flags = _render_apptainer_extra_env_flags(
            extra_env=extra_env,
            indent="                ",
        )
        lmcache_kv_transfer_args = ""
        if lmcache_enabled:
            lmcache_kv_transfer_args = (
                "                --kv-transfer-config\n"
                "                '{\"kv_connector\":\"LMCacheConnectorV1\", \"kv_role\":\"kv_both\"}'\n"
            )
        no_async_scheduling_args = ""
        if no_async_scheduling:
            no_async_scheduling_args = "                --no-async-scheduling\n"
        local_mode_script_raw = local_mode_script if local_mode_script is not None else ""
        local_mode_script_quoted = shlex.quote(local_mode_script_raw)
        gateway_config_default = self._cfg.repo_root / "gateway" / "config.toml"
        gateway_config_fallback = self._cfg.repo_root / "gateway" / "config.example.toml"
        gateway_venv_dir_default = self._cfg.repo_root / ".venv"

        script = textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            #SBATCH --job-name={_safe_token(f'{self._cfg.job_name_prefix}g_{safe_group_name}')}
            #SBATCH --output={slurm_out}
            #SBATCH --error={slurm_err}
            #SBATCH --nodes=1
            #SBATCH --ntasks={group_size}
            #SBATCH --ntasks-per-node={group_size}
            #SBATCH --time={partition_spec.max_time}
            #SBATCH --partition={partition_spec.name}

            set -euo pipefail

            echo "Grouped job ${{SLURM_JOB_ID}} (group={shlex.quote(group_name)}) starting at $(date)"
            echo "SLURM_JOB_NODELIST=${{SLURM_JOB_NODELIST:-unknown}}"

            LOGIN_HOST={shlex.quote(self._cfg.login_host)}
            GROUP_NAME={shlex.quote(group_name)}
            GROUP_SIZE={group_size}
            GROUP_PROFILE_IDS_CSV={shlex.quote(profile_ids_csv)}
            GROUP_VLLM_PORTS_CSV={shlex.quote(vllm_ports_csv)}
            GROUP_GATEWAY_PORTS_CSV={shlex.quote(gateway_ports_csv)}
            GROUP_GATEWAY_PARSE_PORTS_CSV={shlex.quote(gateway_parse_ports_csv)}
            GROUP_JAEGER_OTLP_LOGIN_PORTS_CSV={shlex.quote(jaeger_otlp_ports_csv)}
            GROUP_JAEGER_UI_LOGIN_PORTS_CSV={shlex.quote(jaeger_ui_ports_csv)}
            GROUP_LMCACHE_INTERNAL_API_SERVER_PORT_START_CSV={shlex.quote(lmcache_ports_csv)}
            GROUP_VLLM_VISIBLE_DEVICES_SEMICOLON={shlex.quote(group_visible_devices_semicolon)}
            GROUP_VLLM_TENSOR_PARALLEL_SIZE={gpus_per_profile}
            JAEGER_OTLP_LOCAL_PORT={DEFAULT_JAEGER_OTLP_GRPC_PORT}
            JAEGER_UI_LOCAL_PORT={DEFAULT_JAEGER_QUERY_PORT}

            JAEGER_SIF={shlex.quote(str(self._cfg.jaeger_sif))}
            VLLM_SIF={shlex.quote(str(effective_vllm_sif))}

            VLLM_MODEL_NAME={shlex.quote(model_spec.vllm_model_name)}
            VLLM_SERVED_MODEL_NAME={shlex.quote(model_spec.served_model_name)}

            VLLM_APPTAINER_HOME={shlex.quote(self._cfg.env.get('VLLM_APPTAINER_HOME', ''))}
            HF_HOME={shlex.quote(self._cfg.env.get('HF_HOME', ''))}
            HF_HUB_CACHE={shlex.quote(self._cfg.env.get('HF_HUB_CACHE', ''))}
            HF_HUB_OFFLINE=1
            TRANSFORMERS_OFFLINE=1
            HF_TOKEN="${{HF_TOKEN:-}}"
            export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

            AITER_JIT_DIR={shlex.quote(aiter_jit_dir)}
            XDG_CACHE_HOME={shlex.quote(xdg_cache_home)}
            VLLM_CACHE_ROOT={shlex.quote(vllm_cache_root)}

            OTEL_SERVICE_NAME={shlex.quote(self._cfg.env.get('OTEL_SERVICE_NAME', 'vllm-server'))}
            OTEL_EXPORTER_OTLP_TRACES_INSECURE={shlex.quote(self._cfg.env.get('OTEL_EXPORTER_OTLP_TRACES_INSECURE', 'true'))}
            OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="grpc://127.0.0.1:${{JAEGER_OTLP_LOCAL_PORT}}"
            VLLM_COLLECT_DETAILED_TRACES={shlex.quote(self._cfg.env.get('VLLM_COLLECT_DETAILED_TRACES', 'all'))}
            VLLM_LOGITS_PROCESSORS={shlex.quote(self._cfg.env.get('VLLM_LOGITS_PROCESSORS', 'forceSeq.force_sequence_logits_processor:ForceSequenceAdapter'))}
            VLLM_MODEL_EXTRA_ARGS_B64={shlex.quote(encoded_extra_args)}
            VLLM_FORCE_SEQ_TRUST_REMOTE_CODE={shlex.quote(force_seq_trust_remote_code)}
            GATEWAY_CONFIG_DEFAULT={shlex.quote(str(gateway_config_default))}
            GATEWAY_CONFIG_FALLBACK={shlex.quote(str(gateway_config_fallback))}
            GATEWAY_VENV_DIR="${{GATEWAY_VENV_DIR:-{shlex.quote(str(gateway_venv_dir_default))}}}"
            GATEWAY_HOST="${{GATEWAY_HOST:-127.0.0.1}}"
            GATEWAY_SKIP_INSTALL="${{GATEWAY_SKIP_INSTALL:-1}}"
            GROUP_GATEWAY_CONFIG="${{GROUP_GATEWAY_CONFIG:-}}"

            REPO_ROOT={shlex.quote(str(self._cfg.repo_root))}
            JOB_LOG_DIR={shlex.quote(str(self._cfg.log_dir))}
            mkdir -p "${{JOB_LOG_DIR}}" "${{AITER_JIT_DIR}}" "${{XDG_CACHE_HOME}}" "${{VLLM_CACHE_ROOT}}"
            GROUP_LOCAL_MODE_SCRIPT={local_mode_script_quoted}
            GROUP_LOCAL_MODE_LOG="${{JOB_LOG_DIR}}/group-local-mode.${{SLURM_JOB_ID}}.log"
            JAEGER_LOG_SHARED="${{JOB_LOG_DIR}}/jaeger.${{SLURM_JOB_ID}}.shared.log"

            run_group_worker() {{
              set -euo pipefail
              local SSH_OPTIONS=({ssh_options})

              local index="${{SLURM_PROCID:-${{SLURM_NODEID:-}}}}"
              if [[ -z "${{index}}" ]]; then
                echo "Missing SLURM_PROCID/SLURM_NODEID for grouped worker." >&2
                exit 92
              fi

              IFS=',' read -r -a PROFILE_IDS <<<"${{GROUP_PROFILE_IDS_CSV}}"
              IFS=',' read -r -a VLLM_PORTS <<<"${{GROUP_VLLM_PORTS_CSV}}"
              IFS=',' read -r -a GATEWAY_PORTS <<<"${{GROUP_GATEWAY_PORTS_CSV}}"
              IFS=',' read -r -a GATEWAY_PARSE_PORTS <<<"${{GROUP_GATEWAY_PARSE_PORTS_CSV}}"
              IFS=',' read -r -a JAEGER_OTLP_PORTS <<<"${{GROUP_JAEGER_OTLP_LOGIN_PORTS_CSV}}"
              IFS=',' read -r -a JAEGER_UI_PORTS <<<"${{GROUP_JAEGER_UI_LOGIN_PORTS_CSV}}"
              IFS=',' read -r -a LMCACHE_PORT_STARTS <<<"${{GROUP_LMCACHE_INTERNAL_API_SERVER_PORT_START_CSV}}"
              IFS=';' read -r -a VLLM_VISIBLE_DEVICES_BY_PROFILE <<<"${{GROUP_VLLM_VISIBLE_DEVICES_SEMICOLON}}"

              if [[ "${{index}}" -lt 0 || "${{index}}" -ge "${{#PROFILE_IDS[@]}}" ]]; then
                echo "SLURM_PROCID out of range for grouped profile mapping: index=${{index}}" >&2
                exit 93
              fi
              if [[ "${{index}}" -ge "${{#VLLM_VISIBLE_DEVICES_BY_PROFILE[@]}}" ]]; then
                echo "SLURM_PROCID out of range for grouped visible-devices mapping: index=${{index}}" >&2
                exit 94
              fi
              if [[ "${{index}}" -ge "${{#GATEWAY_PORTS[@]}}" || "${{index}}" -ge "${{#GATEWAY_PARSE_PORTS[@]}}" ]]; then
                echo "SLURM_PROCID out of range for grouped gateway-port mapping: index=${{index}}" >&2
                exit 96
              fi

              local PROFILE_ID="${{PROFILE_IDS[${{index}}]}}"
              local VLLM_SERVICE_PORT="${{VLLM_PORTS[${{index}}]}}"
              local GATEWAY_PORT="${{GATEWAY_PORTS[${{index}}]}}"
              local GATEWAY_PARSE_PORT="${{GATEWAY_PARSE_PORTS[${{index}}]}}"
              local JAEGER_OTLP_LOGIN_PORT="${{JAEGER_OTLP_PORTS[${{index}}]}}"
              local JAEGER_UI_LOGIN_PORT="${{JAEGER_UI_PORTS[${{index}}]}}"
              local LMCACHE_INTERNAL_API_SERVER_PORT_START="${{LMCACHE_PORT_STARTS[${{index}}]}}"
              local VLLM_VISIBLE_DEVICES="${{VLLM_VISIBLE_DEVICES_BY_PROFILE[${{index}}]}}"
              local OTEL_SERVICE_NAME_WORKER="${{OTEL_SERVICE_NAME}}-g${{GROUP_NAME}}-p${{PROFILE_ID}}"
              local VLLM_HIP_VISIBLE_DEVICES="0"
              if [[ "${{VLLM_VISIBLE_DEVICES}}" == *,* ]]; then
                IFS=',' read -r -a VLLM_VISIBLE_DEVICE_LIST <<<"${{VLLM_VISIBLE_DEVICES}}"
                local mapped_devices=()
                for i in "${{!VLLM_VISIBLE_DEVICE_LIST[@]}}"; do
                  mapped_devices+=("${{i}}")
                done
                IFS=',' VLLM_HIP_VISIBLE_DEVICES="${{mapped_devices[*]}}"
              fi
              local VLLM_TENSOR_PARALLEL_SIZE="${{GROUP_VLLM_TENSOR_PARALLEL_SIZE}}"

              local VLLM_LOG="${{JOB_LOG_DIR}}/vllm.${{SLURM_JOB_ID}}.p${{PROFILE_ID}}.log"
              local GATEWAY_LOG="${{JOB_LOG_DIR}}/gateway.${{SLURM_JOB_ID}}.p${{PROFILE_ID}}.log"

              echo "group=${{GROUP_NAME}} profile=${{PROFILE_ID}} node=$(hostname) proc=${{index}} vllm_port=${{VLLM_SERVICE_PORT}}"

              local TUNNEL_PID=""
              local REVERSE_TUNNELS_ENABLED=1
              if [[ -n "${{GROUP_LOCAL_MODE_SCRIPT:-}}" ]]; then
                REVERSE_TUNNELS_ENABLED=0
              fi
              start_reverse_tunnels() {{
                local retries=5
                local attempt=1
                local stagger_seconds=$(( (index % 4) + 1 ))
                sleep "${{stagger_seconds}}"

                while [[ "${{attempt}}" -le "${{retries}}" ]]; do
                  ssh "${{SSH_OPTIONS[@]}}" -N \
                    -R "${{VLLM_SERVICE_PORT}}:127.0.0.1:${{VLLM_SERVICE_PORT}}" \
                    -R "${{JAEGER_OTLP_LOGIN_PORT}}:127.0.0.1:${{JAEGER_OTLP_LOCAL_PORT}}" \
                    -R "${{JAEGER_UI_LOGIN_PORT}}:127.0.0.1:${{JAEGER_UI_LOCAL_PORT}}" \
                    "${{LOGIN_HOST}}" &
                  TUNNEL_PID="$!"

                  sleep 1
                  if kill -0 "${{TUNNEL_PID}}" >/dev/null 2>&1; then
                    return 0
                  fi

                  wait "${{TUNNEL_PID}}" || true
                  echo "Reverse tunnel startup failed (attempt ${{attempt}}/${{retries}}) for group=${{GROUP_NAME}} profile=${{PROFILE_ID}}; retrying." >&2
                  attempt=$((attempt + 1))
                  sleep 2
                done

                return 1
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
                if [[ -n "${{GATEWAY_PID:-}}" ]]; then kill "${{GATEWAY_PID}}" >/dev/null 2>&1 || true; fi
                if [[ -n "${{TUNNEL_PID:-}}" ]]; then kill "${{TUNNEL_PID}}" >/dev/null 2>&1 || true; fi
              }}
              trap cleanup EXIT INT TERM

              if [[ "${{REVERSE_TUNNELS_ENABLED}}" -eq 1 ]]; then
                if ! start_reverse_tunnels; then
                  echo "One or more reverse tunnels failed to establish. Aborting startup." >&2
                  exit 72
                fi
              else
                echo "group=${{GROUP_NAME}} profile=${{PROFILE_ID}} local-mode detected; skipping reverse tunnel setup."
              fi

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
{lmcache_kv_transfer_args}{no_async_scheduling_args}              )

              echo "group=${{GROUP_NAME}} profile=${{PROFILE_ID}} proc=${{index}} OTEL_SERVICE_NAME=${{OTEL_SERVICE_NAME_WORKER}} HIP_VISIBLE_DEVICES=${{VLLM_HIP_VISIBLE_DEVICES}} ROCR_VISIBLE_DEVICES=${{VLLM_VISIBLE_DEVICES}}"
              {{
                local cpu_count="unknown"
                if command -v nproc >/dev/null 2>&1; then
                  cpu_count="$(nproc)"
                fi
                local cpu_ids="unknown"
                if [[ -r /proc/self/status ]]; then
                  cpu_ids="$(awk '/^Cpus_allowed_list:/ {{print $2}}' /proc/self/status)"
                fi
                echo "group=${{GROUP_NAME}} profile=${{PROFILE_ID}} proc=${{index}} cpu_count=${{cpu_count}} cpu_ids=${{cpu_ids}}"
                echo "group=${{GROUP_NAME}} profile=${{PROFILE_ID}} proc=${{index}} running rocm-smi before vLLM startup"
                if command -v rocm-smi >/dev/null 2>&1; then
                  rocm-smi || true
                else
                  echo "rocm-smi not found in PATH"
                fi
              }} >"${{VLLM_LOG}}" 2>&1

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
                --env HF_HUB_OFFLINE="${{HF_HUB_OFFLINE}}" \
                --env TRANSFORMERS_OFFLINE="${{TRANSFORMERS_OFFLINE}}" \
                --env HF_TOKEN="${{HF_TOKEN}}" \
                --env OTEL_SERVICE_NAME="${{OTEL_SERVICE_NAME_WORKER}}" \
                --env OTEL_EXPORTER_OTLP_TRACES_INSECURE="${{OTEL_EXPORTER_OTLP_TRACES_INSECURE}}" \
                --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${{OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}}" \
                --env HIP_VISIBLE_DEVICES="${{VLLM_HIP_VISIBLE_DEVICES}}" \
                --env ROCR_VISIBLE_DEVICES="${{VLLM_VISIBLE_DEVICES}}" \
                --env VLLM_MODEL_NAME="${{VLLM_MODEL_NAME}}" \
                --env VLLM_MODEL_EXTRA_ARGS_B64="${{VLLM_MODEL_EXTRA_ARGS_B64}}" \
                --env VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${{VLLM_FORCE_SEQ_TRUST_REMOTE_CODE}}" \
{extra_apptainer_env_flags}                --env LMCACHE_INTERNAL_API_SERVER_ENABLED=1 \
                --env PYTHONHASHSEED=0 \
                --env LMCACHE_INTERNAL_API_SERVER_PORT_START="${{LMCACHE_INTERNAL_API_SERVER_PORT_START}}" \
                "${{VLLM_SIF}}" \
                "${{VLLM_CMD[@]}}" \
                >>"${{VLLM_LOG}}" 2>&1 &
              VLLM_PID=$!

              local GATEWAY_PID=""
              if [[ -n "${{GROUP_LOCAL_MODE_SCRIPT:-}}" ]]; then
                local GATEWAY_CONFIG_PATH=""
                if [[ -n "${{GROUP_GATEWAY_CONFIG:-}}" ]]; then
                  GATEWAY_CONFIG_PATH="${{GROUP_GATEWAY_CONFIG}}"
                elif [[ -f "${{GATEWAY_CONFIG_DEFAULT}}" ]]; then
                  GATEWAY_CONFIG_PATH="${{GATEWAY_CONFIG_DEFAULT}}"
                else
                  GATEWAY_CONFIG_PATH="${{GATEWAY_CONFIG_FALLBACK}}"
                fi

                local GATEWAY_PYTHON="python3"
                if [[ -x "${{GATEWAY_VENV_DIR}}/bin/python" ]]; then
                  GATEWAY_PYTHON="${{GATEWAY_VENV_DIR}}/bin/python"
                fi

                local GATEWAY_CMD=(
                  "${{GATEWAY_PYTHON}}"
                  -m gateway
                  start
                  --config "${{GATEWAY_CONFIG_PATH}}"
                  --host "${{GATEWAY_HOST}}"
                  --venv-dir "${{GATEWAY_VENV_DIR}}"
                  --port-profile-id "${{PROFILE_ID}}"
                )
                if [[ "${{GATEWAY_SKIP_INSTALL}}" == "1" ]]; then
                  GATEWAY_CMD+=(--skip-install)
                fi

                local GATEWAY_JAEGER_API_BASE_URL="http://127.0.0.1:${{JAEGER_UI_LOCAL_PORT}}/api/traces"
                local GATEWAY_OTLP_TRACES_ENDPOINT="grpc://127.0.0.1:${{JAEGER_OTLP_LOCAL_PORT}}"

                echo "Starting gateway command (group=${{GROUP_NAME}} profile=${{PROFILE_ID}}): ${{GATEWAY_CMD[*]}}"
                echo "group=${{GROUP_NAME}} profile=${{PROFILE_ID}} gateway trace endpoints: jaeger_api=${{GATEWAY_JAEGER_API_BASE_URL}} otlp=${{GATEWAY_OTLP_TRACES_ENDPOINT}}"
                GATEWAY_JAEGER_API_BASE_URL_OVERRIDE="${{GATEWAY_JAEGER_API_BASE_URL}}" \
                GATEWAY_OTLP_TRACES_ENDPOINT_OVERRIDE="${{GATEWAY_OTLP_TRACES_ENDPOINT}}" \
                  "${{GATEWAY_CMD[@]}}" >"${{GATEWAY_LOG}}" 2>&1 &
                GATEWAY_PID=$!
              else
                echo "group=${{GROUP_NAME}} profile=${{PROFILE_ID}} gateway startup skipped (no grouped workload mode requested)."
              fi

              WAIT_PIDS=("${{VLLM_PID}}")
              if [[ -n "${{GATEWAY_PID}}" ]]; then
                WAIT_PIDS+=("${{GATEWAY_PID}}")
              fi
              if [[ -n "${{TUNNEL_PID}}" ]]; then
                WAIT_PIDS+=("${{TUNNEL_PID}}")
              fi
              wait -n "${{WAIT_PIDS[@]}}"
              EXIT_CODE=$?
              echo "group=${{GROUP_NAME}} profile=${{PROFILE_ID}} process exited with code ${{EXIT_CODE}} at $(date)."
              exit "${{EXIT_CODE}}"
            }}

            probe_http_url() {{
              local url="$1"
              python3 -c "import sys, urllib.request; req = urllib.request.Request(sys.argv[1], method='GET'); resp = urllib.request.urlopen(req, timeout=3); sys.exit(0 if int(resp.status) == 200 else 1)" "$url" >/dev/null 2>&1
            }}

            probe_tcp_port() {{
              local host="$1"
              local port="$2"
              python3 -c "import socket, sys; sock = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2); sock.close()" "$host" "$port" >/dev/null 2>&1
            }}

            wait_for_group_vllm_ready() {{
              local timeout_seconds="${{SBATCH_ORCHESTRATOR_READY_TIMEOUT_SECONDS:-900}}"
              local poll_interval_seconds="${{SBATCH_ORCHESTRATOR_READY_POLL_INTERVAL_SECONDS:-2}}"
              local deadline=$((SECONDS + timeout_seconds))
              IFS=',' read -r -a VLLM_PORTS <<<"${{GROUP_VLLM_PORTS_CSV}}"
              IFS=',' read -r -a GATEWAY_PORTS <<<"${{GROUP_GATEWAY_PORTS_CSV}}"
              IFS=',' read -r -a GATEWAY_PARSE_PORTS <<<"${{GROUP_GATEWAY_PARSE_PORTS_CSV}}"

              while (( SECONDS < deadline )); do
                if [[ -n "${{SRUN_PID:-}}" ]] && ! kill -0 "${{SRUN_PID}}" >/dev/null 2>&1; then
                  echo "Grouped worker step exited before orchestrator readiness checks completed." >&2
                  return 1
                fi

                local all_ready=1
                for idx in "${{!VLLM_PORTS[@]}}"; do
                  local vllm_port="${{VLLM_PORTS[${{idx}}]}}"
                  local gateway_port="${{GATEWAY_PORTS[${{idx}}]}}"
                  local gateway_parse_port="${{GATEWAY_PARSE_PORTS[${{idx}}]}}"
                  if ! probe_http_url "http://127.0.0.1:${{vllm_port}}/v1/models"; then
                    all_ready=0
                    break
                  fi
                  if ! probe_tcp_port "127.0.0.1" "${{gateway_port}}"; then
                    all_ready=0
                    break
                  fi
                  if [[ "${{gateway_parse_port}}" -ne "${{gateway_port}}" ]] && ! probe_tcp_port "127.0.0.1" "${{gateway_parse_port}}"; then
                    all_ready=0
                    break
                  fi
                done

                if [[ "${{all_ready}}" -eq 1 ]]; then
                  return 0
                fi
                sleep "${{poll_interval_seconds}}"
              done
              return 1
            }}

            start_shared_jaeger() {{
              local APPTAINER_HOME_ARGS=()
              if [[ -n "${{VLLM_APPTAINER_HOME}}" ]]; then
                mkdir -p "${{VLLM_APPTAINER_HOME}}"
                APPTAINER_HOME_ARGS=(-H "${{VLLM_APPTAINER_HOME}}")
              fi

              apptainer run \
                --cleanenv \
                "${{APPTAINER_HOME_ARGS[@]}}" \
                --env COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
                "${{JAEGER_SIF}}" \
                >"${{JAEGER_LOG_SHARED}}" 2>&1 &
              JAEGER_PID=$!

              sleep 1
              if ! kill -0 "${{JAEGER_PID}}" >/dev/null 2>&1; then
                wait "${{JAEGER_PID}}" || true
                echo "Shared Jaeger failed to start. See ${{JAEGER_LOG_SHARED}}" >&2
                exit 95
              fi
              echo "Started shared Jaeger for group=${{GROUP_NAME}} pid=${{JAEGER_PID}} log=${{JAEGER_LOG_SHARED}}"
            }}

            cleanup_group_main() {{
              set +e
              if [[ -n "${{ORCHESTRATOR_PID:-}}" ]] && kill -0 "${{ORCHESTRATOR_PID}}" >/dev/null 2>&1; then
                kill "${{ORCHESTRATOR_PID}}" >/dev/null 2>&1 || true
                wait "${{ORCHESTRATOR_PID}}" >/dev/null 2>&1 || true
              fi
              if [[ -n "${{SRUN_PID:-}}" ]] && kill -0 "${{SRUN_PID}}" >/dev/null 2>&1; then
                kill "${{SRUN_PID}}" >/dev/null 2>&1 || true
                wait "${{SRUN_PID}}" >/dev/null 2>&1 || true
              fi
              if [[ -n "${{JAEGER_PID:-}}" ]] && kill -0 "${{JAEGER_PID}}" >/dev/null 2>&1; then
                kill "${{JAEGER_PID}}" >/dev/null 2>&1 || true
                wait "${{JAEGER_PID}}" >/dev/null 2>&1 || true
              fi
            }}

            export LOGIN_HOST GROUP_NAME GROUP_SIZE
            export GROUP_PROFILE_IDS_CSV GROUP_VLLM_PORTS_CSV GROUP_GATEWAY_PORTS_CSV GROUP_GATEWAY_PARSE_PORTS_CSV
            export GROUP_JAEGER_OTLP_LOGIN_PORTS_CSV GROUP_JAEGER_UI_LOGIN_PORTS_CSV
            export GROUP_LMCACHE_INTERNAL_API_SERVER_PORT_START_CSV
            export GROUP_VLLM_VISIBLE_DEVICES_SEMICOLON GROUP_VLLM_TENSOR_PARALLEL_SIZE
            export JAEGER_OTLP_LOCAL_PORT JAEGER_UI_LOCAL_PORT JAEGER_SIF VLLM_SIF
            export VLLM_MODEL_NAME VLLM_SERVED_MODEL_NAME
            export VLLM_APPTAINER_HOME HF_HOME HF_HUB_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_TOKEN
            export AITER_JIT_DIR XDG_CACHE_HOME VLLM_CACHE_ROOT
            export OTEL_SERVICE_NAME OTEL_EXPORTER_OTLP_TRACES_INSECURE OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
            export VLLM_COLLECT_DETAILED_TRACES VLLM_LOGITS_PROCESSORS
            export VLLM_MODEL_EXTRA_ARGS_B64 VLLM_FORCE_SEQ_TRUST_REMOTE_CODE
            export GATEWAY_CONFIG_DEFAULT GATEWAY_CONFIG_FALLBACK GATEWAY_VENV_DIR GATEWAY_HOST GATEWAY_SKIP_INSTALL GROUP_GATEWAY_CONFIG
            export JOB_LOG_DIR GROUP_LOCAL_MODE_SCRIPT GROUP_LOCAL_MODE_LOG JAEGER_LOG_SHARED
            export -f run_group_worker
            trap cleanup_group_main EXIT INT TERM
            start_shared_jaeger
            srun \
              --nodes=1 \
              --ntasks="${{GROUP_SIZE}}" \
              --ntasks-per-node="${{GROUP_SIZE}}" \
              --kill-on-bad-exit=1 \
              bash -lc 'run_group_worker' &
            SRUN_PID=$!

            if [[ -n "${{GROUP_LOCAL_MODE_SCRIPT}}" ]]; then
              echo "Detected grouped local-mode script; waiting for grouped vLLM readiness."
              if ! wait_for_group_vllm_ready; then
                echo "Timed out waiting for grouped vLLM readiness before running grouped local-mode script." >&2
                exit 98
              fi

              echo "Executing grouped local-mode script: ${{GROUP_LOCAL_MODE_SCRIPT}}"
              set +e
              bash -lc "${{GROUP_LOCAL_MODE_SCRIPT}}" >"${{GROUP_LOCAL_MODE_LOG}}" 2>&1 &
              ORCHESTRATOR_PID=$!
              set -e
            else
              wait "${{SRUN_PID}}"
              exit $?
            fi

            set +e
            wait -n "${{SRUN_PID}}" "${{ORCHESTRATOR_PID}}"
            FIRST_EXIT_CODE=$?
            set -e

            if ! kill -0 "${{ORCHESTRATOR_PID}}" >/dev/null 2>&1; then
              ORCHESTRATOR_EXIT_CODE="${{FIRST_EXIT_CODE}}"
              set +e
              wait "${{ORCHESTRATOR_PID}}"
              WAIT_ORCHESTRATOR_EXIT_CODE=$?
              set -e
              if [[ "${{WAIT_ORCHESTRATOR_EXIT_CODE}}" -ne 127 ]]; then
                ORCHESTRATOR_EXIT_CODE="${{WAIT_ORCHESTRATOR_EXIT_CODE}}"
              fi
              if [[ "${{ORCHESTRATOR_EXIT_CODE}}" -ne 0 ]]; then
                echo "Grouped workload command failed with exit code ${{ORCHESTRATOR_EXIT_CODE}}." >&2
                exit "${{ORCHESTRATOR_EXIT_CODE}}"
              fi

              echo "Grouped workload command finished successfully; stopping grouped workers."
              if kill -0 "${{SRUN_PID}}" >/dev/null 2>&1; then
                kill "${{SRUN_PID}}" >/dev/null 2>&1 || true
              fi
              wait "${{SRUN_PID}}" >/dev/null 2>&1 || true
              exit 0
            fi

            echo "Grouped worker step exited before grouped workload command completion." >&2
            if kill -0 "${{ORCHESTRATOR_PID}}" >/dev/null 2>&1; then
              kill "${{ORCHESTRATOR_PID}}" >/dev/null 2>&1 || true
            fi
            wait "${{ORCHESTRATOR_PID}}" >/dev/null 2>&1 || true
            wait "${{SRUN_PID}}" >/dev/null 2>&1 || true
            exit "${{FIRST_EXIT_CODE}}"
            """
        )

        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o750)
        return script_path

    def _write_sbatch_script(
        self,
        *,
        partition_spec: PartitionSpec,
        model_spec: ModelSpec,
        port_profile: AMDHPCPortProfile,
        extra_env: dict[str, str],
        lmcache_enabled: bool,
        extra_vllm_args: list[str],
        no_async_scheduling: bool = False,
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_partition = _safe_token(partition_spec.name)
        safe_model = _safe_token(model_spec.name)
        effective_vllm_sif = self._effective_vllm_sif_path(partition_spec)
        script_path = (
            self._cfg.run_dir
            / f"sbatch-{timestamp}-p{port_profile.profile_id}-{safe_partition}-{safe_model}.sh"
        )

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
        effective_extra_args = _effective_vllm_extra_args(
            extra_args=[*model_spec.extra_args, *extra_vllm_args],
            gpus_per_node=partition_spec.gpus_per_node,
        )
        encoded_extra_args = _encode_model_extra_args(effective_extra_args)
        force_seq_trust_remote_code = self._cfg.env.get("VLLM_FORCE_SEQ_TRUST_REMOTE_CODE")
        if force_seq_trust_remote_code is None:
            force_seq_trust_remote_code = "true"
        extra_apptainer_env_flags = _render_apptainer_extra_env_flags(
            extra_env=extra_env,
            indent="              ",
        )
        lmcache_kv_transfer_args = ""
        if lmcache_enabled:
            lmcache_kv_transfer_args = (
                "              --kv-transfer-config\n"
                "              '{\"kv_connector\":\"LMCacheConnectorV1\", \"kv_role\":\"kv_both\"}'\n"
            )
        no_async_scheduling_args = ""
        if no_async_scheduling:
            no_async_scheduling_args = "              --no-async-scheduling\n"

        script = textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            #SBATCH --job-name={_safe_token(f'{self._cfg.job_name_prefix}{port_profile.profile_id}')}
            #SBATCH --output={slurm_out}
            #SBATCH --error={slurm_err}
            #SBATCH --nodes={self._cfg.job_nodes}
            #SBATCH --time={partition_spec.max_time}
            #SBATCH --partition={partition_spec.name}

            set -euo pipefail

            echo "Job ${{SLURM_JOB_ID}} starting on $(hostname) at $(date)"

            LOGIN_HOST={shlex.quote(self._cfg.login_host)}
            VLLM_SERVICE_PORT={port_profile.vllm_port}
            JAEGER_OTLP_LOGIN_PORT={port_profile.jaeger_otlp_port}
            JAEGER_UI_LOGIN_PORT={port_profile.jaeger_api_port}
            LMCACHE_INTERNAL_API_SERVER_PORT_START={port_profile.lmcache_port}
            JAEGER_OTLP_LOCAL_PORT={DEFAULT_JAEGER_OTLP_GRPC_PORT}
            JAEGER_UI_LOCAL_PORT={DEFAULT_JAEGER_QUERY_PORT}

            JAEGER_SIF={shlex.quote(str(self._cfg.jaeger_sif))}
            VLLM_SIF={shlex.quote(str(effective_vllm_sif))}

            VLLM_MODEL_NAME={shlex.quote(model_spec.vllm_model_name)}
            VLLM_SERVED_MODEL_NAME={shlex.quote(model_spec.served_model_name)}
            VLLM_TENSOR_PARALLEL_SIZE={partition_spec.gpus_per_node}
            VLLM_VISIBLE_DEVICES={shlex.quote(visible_devices)}

            VLLM_APPTAINER_HOME={shlex.quote(self._cfg.env.get('VLLM_APPTAINER_HOME', ''))}
            HF_HOME={shlex.quote(self._cfg.env.get('HF_HOME', ''))}
            HF_HUB_CACHE={shlex.quote(self._cfg.env.get('HF_HUB_CACHE', ''))}
            HF_HUB_OFFLINE=1
            TRANSFORMERS_OFFLINE=1
            HF_TOKEN="${{HF_TOKEN:-}}"
            export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

            AITER_JIT_DIR={shlex.quote(aiter_jit_dir)}
            XDG_CACHE_HOME={shlex.quote(xdg_cache_home)}
            VLLM_CACHE_ROOT={shlex.quote(vllm_cache_root)}

            OTEL_SERVICE_NAME={shlex.quote(self._cfg.env.get('OTEL_SERVICE_NAME', 'vllm-server'))}
            OTEL_EXPORTER_OTLP_TRACES_INSECURE={shlex.quote(self._cfg.env.get('OTEL_EXPORTER_OTLP_TRACES_INSECURE', 'true'))}
            OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="grpc://127.0.0.1:${{JAEGER_OTLP_LOCAL_PORT}}"
            VLLM_COLLECT_DETAILED_TRACES={shlex.quote(self._cfg.env.get('VLLM_COLLECT_DETAILED_TRACES', 'all'))}
            VLLM_LOGITS_PROCESSORS={shlex.quote(self._cfg.env.get('VLLM_LOGITS_PROCESSORS', 'forceSeq.force_sequence_logits_processor:ForceSequenceAdapter'))}
            VLLM_MODEL_EXTRA_ARGS_B64={shlex.quote(encoded_extra_args)}
            VLLM_FORCE_SEQ_TRUST_REMOTE_CODE={shlex.quote(force_seq_trust_remote_code)}

            JOB_LOG_DIR={shlex.quote(str(self._cfg.log_dir))}
            mkdir -p "${{JOB_LOG_DIR}}" "${{AITER_JIT_DIR}}" "${{XDG_CACHE_HOME}}" "${{VLLM_CACHE_ROOT}}"

            JAEGER_LOG="${{JOB_LOG_DIR}}/jaeger.${{SLURM_JOB_ID}}.log"
            VLLM_LOG="${{JOB_LOG_DIR}}/vllm.${{SLURM_JOB_ID}}.log"

            SSH_OPTIONS=({ssh_options})

            TUNNEL_PIDS=()
            start_tunnel() {{
              local remote_port="$1"
              local local_port="$2"
              ssh "${{SSH_OPTIONS[@]}}" -N -R "${{remote_port}}:127.0.0.1:${{local_port}}" "${{LOGIN_HOST}}" &
              TUNNEL_PIDS+=("$!")
            }}

            verify_tunnels_alive() {{
              local failed=0
              for pid in "${{TUNNEL_PIDS[@]}}"; do
                if ! kill -0 "${{pid}}" >/dev/null 2>&1; then
                  echo "Reverse tunnel process exited before startup completed (pid=${{pid}})." >&2
                  failed=1
                fi
              done
              return "${{failed}}"
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
            start_tunnel "${{JAEGER_OTLP_LOGIN_PORT}}" "${{JAEGER_OTLP_LOCAL_PORT}}"
            start_tunnel "${{JAEGER_UI_LOGIN_PORT}}" "${{JAEGER_UI_LOCAL_PORT}}"
            sleep 1
            if ! verify_tunnels_alive; then
              echo "One or more reverse tunnels failed to establish. Aborting startup." >&2
              exit 72
            fi

            apptainer run \
              --cleanenv \
              "${{APPTAINER_HOME_ARGS[@]}}" \
              --env COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
              "${{JAEGER_SIF}}" \
              >"${{JAEGER_LOG}}" 2>&1 &
            JAEGER_PID=$!

            VLLM_HIP_VISIBLE_DEVICES="0"
            if [[ "${{VLLM_VISIBLE_DEVICES}}" == *,* ]]; then
              IFS=',' read -r -a VLLM_VISIBLE_DEVICE_LIST <<<"${{VLLM_VISIBLE_DEVICES}}"
              mapped_devices=()
              for i in "${{!VLLM_VISIBLE_DEVICE_LIST[@]}}"; do
                mapped_devices+=("${{i}}")
              done
              IFS=',' VLLM_HIP_VISIBLE_DEVICES="${{mapped_devices[*]}}"
            fi

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
{lmcache_kv_transfer_args}{no_async_scheduling_args}            )

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
              --env HF_HUB_OFFLINE="${{HF_HUB_OFFLINE}}" \
              --env TRANSFORMERS_OFFLINE="${{TRANSFORMERS_OFFLINE}}" \
              --env HF_TOKEN="${{HF_TOKEN}}" \
              --env OTEL_SERVICE_NAME="${{OTEL_SERVICE_NAME}}" \
              --env OTEL_EXPORTER_OTLP_TRACES_INSECURE="${{OTEL_EXPORTER_OTLP_TRACES_INSECURE}}" \
              --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${{OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}}" \
              --env HIP_VISIBLE_DEVICES="${{VLLM_HIP_VISIBLE_DEVICES}}" \
              --env ROCR_VISIBLE_DEVICES="${{VLLM_VISIBLE_DEVICES}}" \
              --env VLLM_MODEL_NAME="${{VLLM_MODEL_NAME}}" \
              --env VLLM_MODEL_EXTRA_ARGS_B64="${{VLLM_MODEL_EXTRA_ARGS_B64}}" \
              --env VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${{VLLM_FORCE_SEQ_TRUST_REMOTE_CODE}}" \
{extra_apptainer_env_flags}              --env LMCACHE_INTERNAL_API_SERVER_ENABLED=1 \
              --env PYTHONHASHSEED=0 \
              --env LMCACHE_INTERNAL_API_SERVER_PORT_START="${{LMCACHE_INTERNAL_API_SERVER_PORT_START}}" \
              "${{VLLM_SIF}}" \
              "${{VLLM_CMD[@]}}" \
              >"${{VLLM_LOG}}" 2>&1 &
            VLLM_PID=$!

            WAIT_PIDS=("${{JAEGER_PID}}" "${{VLLM_PID}}" "${{TUNNEL_PIDS[@]}}")
            wait -n "${{WAIT_PIDS[@]}}"
            EXIT_CODE=$?
            echo "One service/tunnel process exited with code ${{EXIT_CODE}} at $(date)."
            exit "${{EXIT_CODE}}"
            """
        )

        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o750)
        return script_path

    def _write_local_mode_sbatch_script(
        self,
        *,
        partition_spec: PartitionSpec,
        model_spec: ModelSpec,
        port_profile: AMDHPCPortProfile,
        gateway_port: int,
        gateway_parse_port: int,
        local_mode_script: str,
        extra_env: dict[str, str],
        lmcache_enabled: bool,
        extra_vllm_args: list[str],
        no_async_scheduling: bool = False,
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_partition = _safe_token(partition_spec.name)
        safe_model = _safe_token(model_spec.name)
        effective_vllm_sif = self._effective_vllm_sif_path(partition_spec)
        script_path = (
            self._cfg.run_dir
            / f"sbatch-{timestamp}-p{port_profile.profile_id}-{safe_partition}-{safe_model}-local.sh"
        )

        visible_devices = ",".join(str(idx) for idx in range(partition_spec.gpus_per_node))
        slurm_out = self._cfg.log_dir / "slurm.%j.out"
        slurm_err = self._cfg.log_dir / "slurm.%j.err"

        tmp_root = os.environ.get("TMPDIR", "/tmp")
        user_name = os.environ.get("USER", "user")
        aiter_jit_dir = self._cfg.env.get("AITER_JIT_DIR", f"{tmp_root}/vllm-aiter-jit-{user_name}")
        runtime_root = self._cfg.env.get("VLLM_RUNTIME_ROOT", f"{tmp_root}/vllm-runtime-{user_name}")
        xdg_cache_home = self._cfg.env.get("XDG_CACHE_HOME", f"{runtime_root}/xdg-cache")
        vllm_cache_root = self._cfg.env.get("VLLM_CACHE_ROOT", f"{xdg_cache_home}/vllm")

        effective_extra_args = _effective_vllm_extra_args(
            extra_args=[*model_spec.extra_args, *extra_vllm_args],
            gpus_per_node=partition_spec.gpus_per_node,
        )
        encoded_extra_args = _encode_model_extra_args(effective_extra_args)
        force_seq_trust_remote_code = self._cfg.env.get("VLLM_FORCE_SEQ_TRUST_REMOTE_CODE")
        if force_seq_trust_remote_code is None:
            force_seq_trust_remote_code = "true"
        extra_apptainer_env_flags = _render_apptainer_extra_env_flags(
            extra_env=extra_env,
            indent="              ",
        )
        lmcache_kv_transfer_args = ""
        if lmcache_enabled:
            lmcache_kv_transfer_args = (
                "              --kv-transfer-config\n"
                "              '{\"kv_connector\":\"LMCacheConnectorV1\", \"kv_role\":\"kv_both\"}'\n"
            )
        no_async_scheduling_args = ""
        if no_async_scheduling:
            no_async_scheduling_args = "              --no-async-scheduling\n"

        gateway_config_default = self._cfg.repo_root / "gateway" / "config.toml"
        gateway_config_fallback = self._cfg.repo_root / "gateway" / "config.example.toml"
        gateway_venv_dir_default = self._cfg.repo_root / ".venv"
        service_ready_timeout_seconds = int(max(1, self._cfg.startup_timeout))
        service_ready_poll_interval_seconds = float(max(0.2, self._cfg.wait_up_poll_interval_seconds))

        script = textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            #SBATCH --job-name={_safe_token(f'{self._cfg.job_name_prefix}{port_profile.profile_id}_local')}
            #SBATCH --output={slurm_out}
            #SBATCH --error={slurm_err}
            #SBATCH --nodes={self._cfg.job_nodes}
            #SBATCH --time={partition_spec.max_time}
            #SBATCH --partition={partition_spec.name}

            set -euo pipefail

            echo "Local-mode job ${{SLURM_JOB_ID}} starting on $(hostname) at $(date)"

            REPO_ROOT={shlex.quote(str(self._cfg.repo_root))}
            cd "${{REPO_ROOT}}"

            PORT_PROFILE_ID={port_profile.profile_id}
            VLLM_SERVICE_PORT={port_profile.vllm_port}
            GATEWAY_PORT={gateway_port}
            GATEWAY_PARSE_PORT={gateway_parse_port}
            LMCACHE_INTERNAL_API_SERVER_PORT_START={port_profile.lmcache_port}
            JAEGER_OTLP_LOCAL_PORT={DEFAULT_JAEGER_OTLP_GRPC_PORT}
            JAEGER_UI_LOCAL_PORT={DEFAULT_JAEGER_QUERY_PORT}
            SERVICE_READY_TIMEOUT_SECONDS={service_ready_timeout_seconds}
            SERVICE_READY_POLL_INTERVAL_SECONDS={service_ready_poll_interval_seconds}
            LOCAL_MODE_SCRIPT={shlex.quote(local_mode_script)}

            JAEGER_SIF={shlex.quote(str(self._cfg.jaeger_sif))}
            VLLM_SIF={shlex.quote(str(effective_vllm_sif))}

            VLLM_MODEL_NAME={shlex.quote(model_spec.vllm_model_name)}
            VLLM_SERVED_MODEL_NAME={shlex.quote(model_spec.served_model_name)}
            VLLM_TENSOR_PARALLEL_SIZE={partition_spec.gpus_per_node}
            VLLM_VISIBLE_DEVICES={shlex.quote(visible_devices)}

            VLLM_APPTAINER_HOME={shlex.quote(self._cfg.env.get('VLLM_APPTAINER_HOME', ''))}
            HF_HOME={shlex.quote(self._cfg.env.get('HF_HOME', ''))}
            HF_HUB_CACHE={shlex.quote(self._cfg.env.get('HF_HUB_CACHE', ''))}
            HF_HUB_OFFLINE=1
            TRANSFORMERS_OFFLINE=1
            HF_TOKEN="${{HF_TOKEN:-}}"
            export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

            AITER_JIT_DIR={shlex.quote(aiter_jit_dir)}
            XDG_CACHE_HOME={shlex.quote(xdg_cache_home)}
            VLLM_CACHE_ROOT={shlex.quote(vllm_cache_root)}

            OTEL_SERVICE_NAME={shlex.quote(self._cfg.env.get('OTEL_SERVICE_NAME', 'vllm-server'))}
            OTEL_EXPORTER_OTLP_TRACES_INSECURE={shlex.quote(self._cfg.env.get('OTEL_EXPORTER_OTLP_TRACES_INSECURE', 'true'))}
            OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="grpc://127.0.0.1:${{JAEGER_OTLP_LOCAL_PORT}}"
            VLLM_COLLECT_DETAILED_TRACES={shlex.quote(self._cfg.env.get('VLLM_COLLECT_DETAILED_TRACES', 'all'))}
            VLLM_LOGITS_PROCESSORS={shlex.quote(self._cfg.env.get('VLLM_LOGITS_PROCESSORS', 'forceSeq.force_sequence_logits_processor:ForceSequenceAdapter'))}
            VLLM_MODEL_EXTRA_ARGS_B64={shlex.quote(encoded_extra_args)}
            VLLM_FORCE_SEQ_TRUST_REMOTE_CODE={shlex.quote(force_seq_trust_remote_code)}

            GATEWAY_CONFIG_DEFAULT={shlex.quote(str(gateway_config_default))}
            GATEWAY_CONFIG_FALLBACK={shlex.quote(str(gateway_config_fallback))}
            GATEWAY_VENV_DIR="${{GATEWAY_VENV_DIR:-{shlex.quote(str(gateway_venv_dir_default))}}}"
            GATEWAY_HOST="${{GATEWAY_HOST:-127.0.0.1}}"
            GATEWAY_SKIP_INSTALL="${{GATEWAY_SKIP_INSTALL:-1}}"

            JOB_LOG_DIR={shlex.quote(str(self._cfg.log_dir))}
            mkdir -p "${{JOB_LOG_DIR}}" "${{AITER_JIT_DIR}}" "${{XDG_CACHE_HOME}}" "${{VLLM_CACHE_ROOT}}"

            JAEGER_LOG="${{JOB_LOG_DIR}}/jaeger.${{SLURM_JOB_ID}}.log"
            VLLM_LOG="${{JOB_LOG_DIR}}/vllm.${{SLURM_JOB_ID}}.log"
            GATEWAY_LOG="${{JOB_LOG_DIR}}/gateway.${{SLURM_JOB_ID}}.log"
            LOCAL_MODE_SCRIPT_LOG="${{JOB_LOG_DIR}}/local-mode-script.${{SLURM_JOB_ID}}.log"

            VLLM_READY_URL="http://127.0.0.1:${{VLLM_SERVICE_PORT}}/v1/models"
            JAEGER_READY_URL="http://127.0.0.1:${{JAEGER_UI_LOCAL_PORT}}"

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

            probe_http_url() {{
              local url="$1"
              python3 -c "import sys, urllib.request; req = urllib.request.Request(sys.argv[1], method='GET'); resp = urllib.request.urlopen(req, timeout=3); sys.exit(0 if int(resp.status) == 200 else 1)" "$url" >/dev/null 2>&1
            }}

            probe_tcp_port() {{
              local host="$1"
              local port="$2"
              python3 -c "import socket, sys; sock = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2); sock.close()" "$host" "$port" >/dev/null 2>&1
            }}

            wait_for_service_stack() {{
              local deadline=$((SECONDS + SERVICE_READY_TIMEOUT_SECONDS))
              while (( SECONDS < deadline )); do
                if ! kill -0 "${{JAEGER_PID}}" >/dev/null 2>&1; then
                  echo "Jaeger exited before readiness check completed." >&2
                  return 1
                fi
                if ! kill -0 "${{VLLM_PID}}" >/dev/null 2>&1; then
                  echo "vLLM exited before readiness check completed." >&2
                  return 1
                fi
                if probe_http_url "${{VLLM_READY_URL}}" && probe_http_url "${{JAEGER_READY_URL}}"; then
                  return 0
                fi
                sleep "${{SERVICE_READY_POLL_INTERVAL_SECONDS}}"
              done
              return 1
            }}

            wait_for_gateway_ready() {{
              local deadline=$((SECONDS + SERVICE_READY_TIMEOUT_SECONDS))
              while (( SECONDS < deadline )); do
                if ! kill -0 "${{GATEWAY_PID}}" >/dev/null 2>&1; then
                  echo "Gateway exited before readiness check completed." >&2
                  dump_gateway_diagnostics "gateway process exited before readiness"
                  return 1
                fi
                if [[ "${{GATEWAY_PORT}}" -eq "${{GATEWAY_PARSE_PORT}}" ]]; then
                  if probe_tcp_port "127.0.0.1" "${{GATEWAY_PORT}}"; then
                    return 0
                  fi
                else
                  if probe_tcp_port "127.0.0.1" "${{GATEWAY_PORT}}" && probe_tcp_port "127.0.0.1" "${{GATEWAY_PARSE_PORT}}"; then
                    return 0
                  fi
                fi
                sleep 1
              done
              dump_gateway_diagnostics "gateway readiness timeout"
              return 1
            }}

            dump_gateway_diagnostics() {{
              local reason="${{1:-gateway failure}}"
              echo "Gateway diagnostics: ${{reason}}" >&2
              if [[ -n "${{GATEWAY_PID:-}}" ]]; then
                if kill -0 "${{GATEWAY_PID}}" >/dev/null 2>&1; then
                  echo "Gateway PID ${{GATEWAY_PID}} is still running." >&2
                else
                  local gateway_exit_code=0
                  if wait "${{GATEWAY_PID}}" >/dev/null 2>&1; then
                    gateway_exit_code=0
                  else
                    gateway_exit_code=$?
                  fi
                  echo "Gateway PID ${{GATEWAY_PID}} exited with code ${{gateway_exit_code}}." >&2
                fi
              else
                echo "Gateway PID is not set." >&2
              fi
              echo "Gateway log path: ${{GATEWAY_LOG}}" >&2
              if [[ -f "${{GATEWAY_LOG}}" ]]; then
                echo "----- gateway log tail (last 200 lines) -----" >&2
                tail -n 200 "${{GATEWAY_LOG}}" >&2 || true
                echo "----- end gateway log tail -----" >&2
              else
                echo "Gateway log file not found." >&2
              fi
            }}

            terminate_process() {{
              local name="$1"
              local pid="$2"
              if [[ -z "${{pid}}" ]]; then
                return 0
              fi
              if ! kill -0 "${{pid}}" >/dev/null 2>&1; then
                return 0
              fi
              echo "Stopping ${{name}} (pid=${{pid}})"
              kill "${{pid}}" >/dev/null 2>&1 || true
              local grace_deadline=$((SECONDS + 20))
              while kill -0 "${{pid}}" >/dev/null 2>&1; do
                if (( SECONDS >= grace_deadline )); then
                  echo "${{name}} (pid=${{pid}}) did not exit after SIGTERM; sending SIGKILL."
                  kill -9 "${{pid}}" >/dev/null 2>&1 || true
                  break
                fi
                sleep 1
              done
              wait "${{pid}}" >/dev/null 2>&1 || true
            }}

            cleanup() {{
              set +e
              terminate_process "local script" "${{WORKLOAD_PID:-}}"
              terminate_process "gateway" "${{GATEWAY_PID:-}}"
              terminate_process "vllm" "${{VLLM_PID:-}}"
              terminate_process "jaeger" "${{JAEGER_PID:-}}"
            }}
            trap cleanup EXIT INT TERM

            apptainer run \
              --cleanenv \
              "${{APPTAINER_HOME_ARGS[@]}}" \
              --env COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
              "${{JAEGER_SIF}}" \
              >"${{JAEGER_LOG}}" 2>&1 &
            JAEGER_PID=$!

            VLLM_HIP_VISIBLE_DEVICES="0"
            if [[ "${{VLLM_VISIBLE_DEVICES}}" == *,* ]]; then
              IFS=',' read -r -a VLLM_VISIBLE_DEVICE_LIST <<<"${{VLLM_VISIBLE_DEVICES}}"
              mapped_devices=()
              for i in "${{!VLLM_VISIBLE_DEVICE_LIST[@]}}"; do
                mapped_devices+=("${{i}}")
              done
              IFS=',' VLLM_HIP_VISIBLE_DEVICES="${{mapped_devices[*]}}"
            fi

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
{lmcache_kv_transfer_args}{no_async_scheduling_args}            )

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
              --env HF_HUB_OFFLINE="${{HF_HUB_OFFLINE}}" \
              --env TRANSFORMERS_OFFLINE="${{TRANSFORMERS_OFFLINE}}" \
              --env HF_TOKEN="${{HF_TOKEN}}" \
              --env OTEL_SERVICE_NAME="${{OTEL_SERVICE_NAME}}" \
              --env OTEL_EXPORTER_OTLP_TRACES_INSECURE="${{OTEL_EXPORTER_OTLP_TRACES_INSECURE}}" \
              --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${{OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}}" \
              --env HIP_VISIBLE_DEVICES="${{VLLM_HIP_VISIBLE_DEVICES}}" \
              --env ROCR_VISIBLE_DEVICES="${{VLLM_VISIBLE_DEVICES}}" \
              --env VLLM_MODEL_NAME="${{VLLM_MODEL_NAME}}" \
              --env VLLM_MODEL_EXTRA_ARGS_B64="${{VLLM_MODEL_EXTRA_ARGS_B64}}" \
              --env VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${{VLLM_FORCE_SEQ_TRUST_REMOTE_CODE}}" \
{extra_apptainer_env_flags}              --env LMCACHE_INTERNAL_API_SERVER_ENABLED=1 \
              --env PYTHONHASHSEED=0 \
              --env LMCACHE_INTERNAL_API_SERVER_PORT_START="${{LMCACHE_INTERNAL_API_SERVER_PORT_START}}" \
              "${{VLLM_SIF}}" \
              "${{VLLM_CMD[@]}}" \
              >"${{VLLM_LOG}}" 2>&1 &
            VLLM_PID=$!

            echo "Waiting for Jaeger + vLLM readiness..."
            if ! wait_for_service_stack; then
              echo "Timed out waiting for Jaeger/vLLM readiness." >&2
              exit 73
            fi

            if [[ -n "${{GATEWAY_CONFIG:-}}" ]]; then
              GATEWAY_CONFIG_PATH="${{GATEWAY_CONFIG}}"
            elif [[ -f "${{GATEWAY_CONFIG_DEFAULT}}" ]]; then
              GATEWAY_CONFIG_PATH="${{GATEWAY_CONFIG_DEFAULT}}"
            else
              GATEWAY_CONFIG_PATH="${{GATEWAY_CONFIG_FALLBACK}}"
            fi

            GATEWAY_PYTHON="python3"
            if [[ -x "${{GATEWAY_VENV_DIR}}/bin/python" ]]; then
              GATEWAY_PYTHON="${{GATEWAY_VENV_DIR}}/bin/python"
            fi

            GATEWAY_CMD=(
              "${{GATEWAY_PYTHON}}"
              -m gateway
              start
              --config "${{GATEWAY_CONFIG_PATH}}"
              --host "${{GATEWAY_HOST}}"
              --venv-dir "${{GATEWAY_VENV_DIR}}"
              --port-profile-id "${{PORT_PROFILE_ID}}"
            )
            if [[ "${{GATEWAY_SKIP_INSTALL}}" == "1" ]]; then
              GATEWAY_CMD+=(--skip-install)
            fi

            GATEWAY_JAEGER_API_BASE_URL="http://127.0.0.1:${{JAEGER_UI_LOCAL_PORT}}/api/traces"
            GATEWAY_OTLP_TRACES_ENDPOINT="grpc://127.0.0.1:${{JAEGER_OTLP_LOCAL_PORT}}"

            echo "Starting gateway command: ${{GATEWAY_CMD[*]}}"
            echo "gateway trace endpoints: jaeger_api=${{GATEWAY_JAEGER_API_BASE_URL}} otlp=${{GATEWAY_OTLP_TRACES_ENDPOINT}}"
            GATEWAY_JAEGER_API_BASE_URL_OVERRIDE="${{GATEWAY_JAEGER_API_BASE_URL}}" \
            GATEWAY_OTLP_TRACES_ENDPOINT_OVERRIDE="${{GATEWAY_OTLP_TRACES_ENDPOINT}}" \
              "${{GATEWAY_CMD[@]}}" >"${{GATEWAY_LOG}}" 2>&1 &
            GATEWAY_PID=$!

            echo "Waiting for gateway readiness..."
            if ! wait_for_gateway_ready; then
              echo "Timed out waiting for gateway readiness." >&2
              exit 74
            fi

            export VLLM_BASE_URL="http://127.0.0.1:${{VLLM_SERVICE_PORT}}"
            export GATEWAY_BASE_URL="http://127.0.0.1:${{GATEWAY_PORT}}"
            export GATEWAY_PARSE_BASE_URL="http://127.0.0.1:${{GATEWAY_PARSE_PORT}}"
            export JAEGER_BASE_URL="http://127.0.0.1:${{JAEGER_UI_LOCAL_PORT}}"
            export PORT_PROFILE_ID

            echo "Executing local-mode script: ${{LOCAL_MODE_SCRIPT}}"
            set +e
            bash -lc "${{LOCAL_MODE_SCRIPT}}" >"${{LOCAL_MODE_SCRIPT_LOG}}" 2>&1
            WORKLOAD_EXIT_CODE=$?
            set -e
            echo "Local-mode script finished with exit code ${{WORKLOAD_EXIT_CODE}} at $(date)"
            exit "${{WORKLOAD_EXIT_CODE}}"
            """
        )

        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o750)
        return script_path

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
        port_profile_id: int,
        status: str,
        phase: str | None,
        message: str,
        job_id: str | None,
        started_at: str | None,
        finished_at: str | None,
    ) -> None:
        self._set_progress(
            kind="start",
            port_profile_id=port_profile_id,
            status=status,
            phase=phase,
            message=message,
            job_id=job_id,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _current_start_job_id(self, port_profile_id: int) -> str | None:
        return self._current_progress_job_id("start", port_profile_id)

    def _set_stop_progress(
        self,
        *,
        port_profile_id: int,
        status: str,
        phase: str | None,
        message: str,
        job_id: str | None,
        started_at: str | None,
        finished_at: str | None,
    ) -> None:
        self._set_progress(
            kind="stop",
            port_profile_id=port_profile_id,
            status=status,
            phase=phase,
            message=message,
            job_id=job_id,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _current_stop_job_id(self, port_profile_id: int) -> str | None:
        return self._current_progress_job_id("stop", port_profile_id)

    def _set_progress(
        self,
        *,
        kind: str,
        port_profile_id: int,
        status: str,
        phase: str | None,
        message: str,
        job_id: str | None,
        started_at: str | None,
        finished_at: str | None,
    ) -> None:
        lock, progress_store = self._progress_state(kind)
        with lock:
            progress_store[port_profile_id] = {
                "status": status,
                "phase": phase,
                "message": message,
                "job_id": job_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "updated_at": _utc_now_iso(),
            }

    def _current_progress_job_id(self, kind: str, port_profile_id: int) -> str | None:
        lock, progress_store = self._progress_state(kind)
        with lock:
            progress = progress_store.get(port_profile_id, {})
            value = progress.get("job_id")
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
    repo_root = app_dir.parent.parent

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
    images_table = raw.get("images")
    if images_table is None:
        images_table = {}
    elif not isinstance(images_table, dict):
        raise ControlPlaneError(
            message="images must be a table",
            code=114,
            http_status=500,
        )

    merged_env = dict(os.environ)

    host = str(server_table.get("host", "0.0.0.0"))
    port = int(server_table.get("port", 23971))

    model_config_path = (repo_root / "configs/model_config.toml").resolve()
    run_dir = _resolve_path(repo_root, str(cluster_table.get("run_dir", "servers/servers-amdhpc/run")))
    log_dir = _resolve_path(repo_root, str(cluster_table.get("log_dir", "servers/servers-amdhpc/logs")))
    state_file = _resolve_path(
        repo_root,
        str(cluster_table.get("state_file", "servers/servers-amdhpc/run/control_state.json")),
    )

    login_host = str(cluster_table.get("login_host", "login"))
    job_name_prefix_raw = cluster_table.get("job_name_prefix", cluster_table.get("job_name", "vllm_job_"))
    job_name_prefix = str(job_name_prefix_raw)
    job_nodes = int(cluster_table.get("job_nodes", 1))

    service_port = int(cluster_table.get("service_port", int(merged_env.get("VLLM_SERVICE_PORT", "11451"))))
    jaeger_otlp_port = int(cluster_table.get("jaeger_otlp_port", 4317))
    jaeger_ui_port = int(cluster_table.get("jaeger_ui_port", 16686))
    startup_timeout = int(cluster_table.get("startup_timeout", int(merged_env.get("VLLM_STARTUP_TIMEOUT", "900"))))
    startup_timeout_after_running_raw = cluster_table.get("startup_timeout_after_running", True)
    if not isinstance(startup_timeout_after_running_raw, bool):
        raise ControlPlaneError(
            message="cluster.startup_timeout_after_running must be a boolean",
            code=112,
            http_status=500,
        )
    startup_timeout_after_running = startup_timeout_after_running_raw
    stop_wait_timeout_seconds = int(
        cluster_table.get("stop_wait_timeout_seconds", DEFAULT_STOP_WAIT_TIMEOUT_SECONDS)
    )
    stop_poll_interval_seconds = float(
        cluster_table.get("stop_poll_interval_seconds", DEFAULT_STOP_POLL_INTERVAL_SECONDS)
    )
    wait_up_poll_interval_seconds = float(
        cluster_table.get("wait_up_poll_interval_seconds", DEFAULT_WAIT_UP_POLL_INTERVAL_SECONDS)
    )
    if stop_wait_timeout_seconds <= 0:
        raise ControlPlaneError(
            message="cluster.stop_wait_timeout_seconds must be > 0",
            code=115,
            http_status=500,
        )
    if stop_poll_interval_seconds <= 0:
        raise ControlPlaneError(
            message="cluster.stop_poll_interval_seconds must be > 0",
            code=116,
            http_status=500,
        )
    if wait_up_poll_interval_seconds <= 0:
        raise ControlPlaneError(
            message="cluster.wait_up_poll_interval_seconds must be > 0",
            code=117,
            http_status=500,
        )

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

    apptainer_imgs = _resolve_path(
        repo_root,
        _expand_vars(merged_env.get("APPTAINER_IMGS", f"{Path.home()}/apptainer-images"), merged_env),
    )
    jaeger_image = str(images_table.get("jaeger_image", DEFAULT_JAEGER_IMAGE))
    vllm_image = str(images_table.get("vllm_image", DEFAULT_VLLM_IMAGE))
    if not jaeger_image:
        raise ControlPlaneError(
            message="images.jaeger_image must be non-empty",
            code=118,
            http_status=500,
        )
    if not vllm_image:
        raise ControlPlaneError(
            message="images.vllm_image must be non-empty",
            code=119,
            http_status=500,
        )

    try:
        port_profiles = load_port_profiles()
    except Exception as exc:  # noqa: BLE001
        raise ControlPlaneError(
            message="failed to load configs/port_profiles.toml",
            code=113,
            http_status=500,
            details={"error": str(exc)},
        ) from exc

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
        raw_partition_sif = value.get("sif_img")
        partition_vllm_sif: Path | None = None
        if raw_partition_sif is not None:
            if not isinstance(raw_partition_sif, str):
                raise ControlPlaneError(
                    message=f"partition.{name}.sif_img must be a string",
                    code=124,
                    http_status=500,
                )
            normalized_partition_sif = raw_partition_sif.strip()
            if not normalized_partition_sif:
                raise ControlPlaneError(
                    message=f"partition.{name}.sif_img must be non-empty when provided",
                    code=124,
                    http_status=500,
                )
            partition_vllm_sif = _resolve_partition_sif_path(
                repo_root=repo_root,
                apptainer_imgs=apptainer_imgs,
                raw_path=_expand_vars(normalized_partition_sif, merged_env),
            )
        partitions[name] = PartitionSpec(
            name=name,
            gpus_per_node=int(value["gpus_per_node"]),
            gpu_memory_gb=float(value["gpu_memory_gb"]),
            total_vram_gb=float(value["total_vram_gb"]),
            max_time=str(value.get("max_time", "04:00:00")),
            vllm_sif=partition_vllm_sif,
        )

    if not model_config_path.exists():
        raise ControlPlaneError(
            message=f"model config file not found: {model_config_path}",
            code=106,
            http_status=500,
        )

    try:
        model_raw = tomllib.loads(model_config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ControlPlaneError(
            message=f"failed to parse model config: {model_config_path}",
            code=107,
            http_status=500,
            details={"error": str(exc)},
        ) from exc

    if not isinstance(model_raw, dict):
        raise ControlPlaneError(
            message=f"model config root must be a table: {model_config_path}",
            code=108,
            http_status=500,
        )

    models_table = model_raw.get("models")
    if not isinstance(models_table, dict) or not models_table:
        raise ControlPlaneError(
            message=f"model config must define [models.<name>] tables: {model_config_path}",
            code=109,
            http_status=500,
        )

    models: dict[str, ModelSpec] = {}
    for name, value in models_table.items():
        if not isinstance(value, dict):
            raise ControlPlaneError(
                message=f"models.{name} must be a table in {model_config_path}",
                code=110,
                http_status=500,
            )

        extra_args_raw = value.get("extra_args", [])
        if not isinstance(extra_args_raw, list) or not all(
            isinstance(item, str) for item in extra_args_raw
        ):
            raise ControlPlaneError(
                message=f"models.{name}.extra_args must be a string array in {model_config_path}",
                code=111,
                http_status=500,
            )

        models[name] = ModelSpec(
            name=name,
            vllm_model_name=str(value["vllm_model_name"]),
            served_model_name=str(value["served_model_name"]),
            weight_vram_gb=float(value["weight_vram_gb"]),
            extra_args=list(extra_args_raw),
        )

    jaeger_sif_raw = merged_env.get("JAEGER_SIF", str(apptainer_imgs / _infer_sif_name_from_image(jaeger_image)))
    vllm_sif_raw = merged_env.get("VLLM_SIF", str(apptainer_imgs / _infer_sif_name_from_image(vllm_image)))

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
        job_name_prefix=job_name_prefix,
        job_nodes=job_nodes,
        service_port=service_port,
        jaeger_otlp_port=jaeger_otlp_port,
        jaeger_ui_port=jaeger_ui_port,
        startup_timeout=startup_timeout,
        startup_timeout_after_running=startup_timeout_after_running,
        stop_wait_timeout_seconds=stop_wait_timeout_seconds,
        stop_poll_interval_seconds=stop_poll_interval_seconds,
        wait_up_poll_interval_seconds=wait_up_poll_interval_seconds,
        ssh_options=ssh_options,
        port_profiles=port_profiles,
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


def _is_slurm_running_state(status: str | None) -> bool:
    if status is None:
        return False
    normalized = status.strip().upper()
    return normalized in {"R", "RUNNING", "CG", "COMPLETING"}


def _is_slurm_pending_state(status: str | None) -> bool:
    if status is None:
        return False
    normalized = status.strip().upper()
    return normalized in {"PD", "PENDING"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _resolve_partition_sif_path(
    *,
    repo_root: Path,
    apptainer_imgs: Path,
    raw_path: str,
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return (apptainer_imgs / path.name).resolve()
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
