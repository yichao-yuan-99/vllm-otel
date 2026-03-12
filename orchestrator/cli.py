"""Mini scheduler to run con-driver/replay jobs across port profiles."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence
from urllib import error as url_error
from urllib import request as url_request

try:
    import typer
except ModuleNotFoundError:  # pragma: no cover
    class _FallbackExit(SystemExit):
        def __init__(self, code: int = 0) -> None:
            super().__init__(code)
            self.code = code

    class _FallbackTyperApp:
        def __init__(
            self,
            *,
            add_completion: bool = False,
            no_args_is_help: bool = False,
            invoke_without_command: bool = False,
        ) -> None:
            del add_completion, no_args_is_help, invoke_without_command
            self._callback: Any = None

        def callback(self) -> Any:
            def _register(func: Any) -> Any:
                self._callback = func
                return func

            return _register

        def __call__(self, *, args: list[str], prog_name: str) -> None:
            del args, prog_name
            raise RuntimeError(
                "Typer is required to run the orchestrator CLI. "
                "Install dependencies first (for example: pip install -e ./con-driver)."
            )

    class _FallbackTyperModule:
        Exit = _FallbackExit

        @staticmethod
        def Typer(
            *,
            add_completion: bool = False,
            no_args_is_help: bool = False,
            invoke_without_command: bool = False,
        ) -> _FallbackTyperApp:
            return _FallbackTyperApp(
                add_completion=add_completion,
                no_args_is_help=no_args_is_help,
                invoke_without_command=invoke_without_command,
            )

        @staticmethod
        def Option(default: Any, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            return default

        @staticmethod
        def echo(message: str, err: bool = False) -> None:
            stream = sys.stderr if err else sys.stdout
            print(message, file=stream)

    typer = _FallbackTyperModule()

from gateway.port_profiles import load_port_profile


class JobType(str, Enum):
    CON_DRIVER = "con-driver"
    REPLAY = "replay"


@dataclass
class RunningJob:
    job_index: int
    config_path: Path
    profile_id: int
    command: list[str]
    log_path: Path
    started_at_iso: str
    started_monotonic: float
    process: subprocess.Popen[str]
    log_handle: Any


@dataclass
class JobResult:
    job_index: int
    config_path: str
    port_profile_id: int
    command: list[str]
    log_path: str
    started_at: str
    finished_at: str
    duration_s: float
    exit_code: int | None
    status: str


app = typer.Typer(add_completion=False, no_args_is_help=True, invoke_without_command=True)


def now_iso8601_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def safe_name(value: str) -> str:
    chars: list[str] = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars) or "job"


def parse_port_profile_id_list(raw: str) -> list[int]:
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--port-profile-id-list must be a non-empty comma-separated list")

    parsed: list[int] = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(
                f"--port-profile-id-list contains a non-integer value: {token!r}"
            ) from exc
        if value < 0:
            raise ValueError("--port-profile-id-list values must be >= 0")
        parsed.append(value)

    if len(set(parsed)) != len(parsed):
        raise ValueError("--port-profile-id-list cannot contain duplicate profile IDs")
    return parsed


def discover_job_configs(*, jobs_dir: Path, config_glob: str) -> list[Path]:
    if not jobs_dir.exists() or not jobs_dir.is_dir():
        raise ValueError(f"Invalid --jobs-dir: {jobs_dir}")
    discovered = sorted(path.resolve() for path in jobs_dir.glob(config_glob) if path.is_file())
    if not discovered:
        raise ValueError(
            f"No job configs found in {jobs_dir} with glob pattern {config_glob!r}"
        )
    return discovered


def gateway_health_url(profile_id: int) -> str:
    profile = load_port_profile(profile_id)
    return f"http://127.0.0.1:{profile.gateway_port}/healthz"


def check_gateway_health(*, profile_ids: list[int], timeout_s: float) -> None:
    failures: list[str] = []
    for profile_id in profile_ids:
        healthz_url = gateway_health_url(profile_id)
        req = url_request.Request(url=healthz_url, method="GET")
        try:
            with url_request.urlopen(req, timeout=timeout_s) as response:
                status_code = int(response.getcode())
                raw_body = response.read().decode("utf-8", errors="replace")
        except url_error.URLError as exc:
            failures.append(
                f"profile {profile_id}: failed to reach {healthz_url}: {exc}"
            )
            continue

        if status_code >= 400:
            failures.append(
                f"profile {profile_id}: {healthz_url} returned HTTP {status_code}"
            )
            continue

        parsed: Any
        if raw_body.strip():
            try:
                parsed = json.loads(raw_body)
            except json.JSONDecodeError:
                parsed = {"raw_body": raw_body}
        else:
            parsed = {}
        if not isinstance(parsed, dict) or parsed.get("status") != "ok":
            failures.append(
                f"profile {profile_id}: unexpected health payload from {healthz_url}: {parsed}"
            )

    if failures:
        lines = ["Gateway health checks failed before orchestration starts:"]
        lines.extend(f"- {failure}" for failure in failures)
        raise RuntimeError("\n".join(lines))


def build_job_command(*, job_type: JobType, config_path: Path, profile_id: int) -> list[str]:
    if job_type == JobType.CON_DRIVER:
        return [
            sys.executable,
            "-m",
            "con_driver",
            "--config",
            str(config_path),
            "--port-profile-id",
            str(profile_id),
        ]
    if job_type == JobType.REPLAY:
        return [
            sys.executable,
            "-m",
            "replayer",
            "replay",
            "--config",
            str(config_path),
            "--port-profile-id",
            str(profile_id),
        ]
    raise ValueError(f"Unsupported job type: {job_type!r}")


def build_child_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    pythonpath_parts = [str(repo_root), str(repo_root / "con-driver" / "src")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def run_orchestrator(
    *,
    job_type: JobType,
    jobs_dir: Path,
    config_glob: str,
    profile_ids: list[int],
    output_dir: Path,
    poll_interval_s: float,
    health_timeout_s: float,
    fail_fast: bool,
) -> dict[str, Any]:
    if poll_interval_s <= 0:
        raise ValueError("--poll-interval-s must be > 0")
    if health_timeout_s <= 0:
        raise ValueError("--health-timeout-s must be > 0")

    job_configs = discover_job_configs(jobs_dir=jobs_dir, config_glob=config_glob)
    # Replay always routes via gateway, so enforce health checks up front.
    # Con-driver can run with gateway disabled, so do not hard-require it here.
    if job_type == JobType.REPLAY:
        check_gateway_health(profile_ids=profile_ids, timeout_s=health_timeout_s)

    output_dir.mkdir(parents=True, exist_ok=True)
    env = build_child_env()
    repo_root = Path(__file__).resolve().parents[1]
    started_at_iso = now_iso8601_utc()

    pending = deque(job_configs)
    idle_profiles = list(profile_ids)
    running_by_profile: dict[int, RunningJob] = {}
    completed_results: list[JobResult] = []
    next_job_index = 1
    stop_launching = False
    interrupted = False

    try:
        while pending or running_by_profile:
            while pending and idle_profiles and not stop_launching:
                profile_id = idle_profiles.pop(0)
                config_path = pending.popleft()
                job_index = next_job_index
                next_job_index += 1
                command = build_job_command(
                    job_type=job_type,
                    config_path=config_path,
                    profile_id=profile_id,
                )
                log_path = output_dir / (
                    f"job-{job_index:04d}.profile-{profile_id}.{safe_name(config_path.stem)}.log"
                )
                log_handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=repo_root,
                    env=env,
                )
                running = RunningJob(
                    job_index=job_index,
                    config_path=config_path,
                    profile_id=profile_id,
                    command=command,
                    log_path=log_path,
                    started_at_iso=now_iso8601_utc(),
                    started_monotonic=time.monotonic(),
                    process=process,
                    log_handle=log_handle,
                )
                running_by_profile[profile_id] = running
                typer.echo(
                    f"[launch] job={job_index} profile={profile_id} config={config_path.name}"
                )

            if not running_by_profile:
                break

            time.sleep(poll_interval_s)
            for profile_id, running in list(running_by_profile.items()):
                return_code = running.process.poll()
                if return_code is None:
                    continue

                finished_at_iso = now_iso8601_utc()
                duration_s = round(max(0.0, time.monotonic() - running.started_monotonic), 3)
                with suppress(Exception):
                    running.log_handle.close()

                status = "succeeded" if return_code == 0 else "failed"
                if return_code != 0 and fail_fast:
                    stop_launching = True

                completed_results.append(
                    JobResult(
                        job_index=running.job_index,
                        config_path=str(running.config_path),
                        port_profile_id=profile_id,
                        command=running.command,
                        log_path=str(running.log_path),
                        started_at=running.started_at_iso,
                        finished_at=finished_at_iso,
                        duration_s=duration_s,
                        exit_code=return_code,
                        status=status,
                    )
                )
                typer.echo(
                    f"[done] job={running.job_index} profile={profile_id} "
                    f"status={status} exit_code={return_code}"
                )
                running_by_profile.pop(profile_id, None)
                idle_profiles.append(profile_id)
    except KeyboardInterrupt:
        interrupted = True
        for running in running_by_profile.values():
            with suppress(Exception):
                running.process.terminate()
        for running in running_by_profile.values():
            with suppress(Exception):
                running.process.wait(timeout=10.0)
        for running in running_by_profile.values():
            with suppress(Exception):
                running.log_handle.close()
    finally:
        for running in running_by_profile.values():
            with suppress(Exception):
                running.log_handle.close()

    failed_count = sum(1 for item in completed_results if item.status == "failed")
    succeeded_count = sum(1 for item in completed_results if item.status == "succeeded")
    if interrupted:
        status = "interrupted"
    elif failed_count == 0 and not pending:
        status = "ok"
    else:
        status = "failed"

    summary = {
        "status": status,
        "job_type": job_type.value,
        "jobs_dir": str(jobs_dir),
        "config_glob": config_glob,
        "output_dir": str(output_dir),
        "port_profile_id_list": profile_ids,
        "total_jobs": len(job_configs),
        "launched_jobs": len(completed_results),
        "pending_jobs": len(pending),
        "succeeded": succeeded_count,
        "failed": failed_count,
        "interrupted": interrupted,
        "fail_fast": fail_fast,
        "started_at": started_at_iso,
        "finished_at": now_iso8601_utc(),
        "results": [asdict(item) for item in completed_results],
    }
    return summary


@app.callback()
def run(
    job_type: JobType = typer.Option(
        ...,
        "--job-type",
        help="Job runner type. Use 'con-driver' or 'replay'.",
    ),
    jobs_dir: Path = typer.Option(
        ...,
        "--jobs-dir",
        help="Directory containing job config files to launch.",
    ),
    port_profile_id_list: str = typer.Option(
        ...,
        "--port-profile-id-list",
        help="Comma-separated port profile IDs, for example '0,1,2,3'.",
    ),
    config_glob: str = typer.Option(
        "*.toml",
        "--config-glob",
        help="Glob used to discover config files inside --jobs-dir.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory for orchestrator logs and summary JSON.",
    ),
    poll_interval_s: float = typer.Option(
        1.0,
        "--poll-interval-s",
        help="Polling interval in seconds for child-process completion.",
    ),
    health_timeout_s: float = typer.Option(
        5.0,
        "--health-timeout-s",
        help="Gateway /healthz timeout in seconds per profile.",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast/--no-fail-fast",
        help="Stop launching new jobs after the first job failure.",
    ),
) -> None:
    """Schedule config-driven jobs over idle port profiles."""
    try:
        resolved_jobs_dir = jobs_dir.expanduser().resolve()
        profile_ids = parse_port_profile_id_list(port_profile_id_list)
        if output_dir is not None:
            resolved_output_dir = output_dir.expanduser().resolve()
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            resolved_output_dir = (
                resolved_jobs_dir / f"orchestrator-{job_type.value}-{timestamp}"
            ).resolve()

        summary = run_orchestrator(
            job_type=job_type,
            jobs_dir=resolved_jobs_dir,
            config_glob=config_glob,
            profile_ids=profile_ids,
            output_dir=resolved_output_dir,
            poll_interval_s=poll_interval_s,
            health_timeout_s=health_timeout_s,
            fail_fast=fail_fast,
        )
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    summary_path = resolved_output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=True))

    status = summary.get("status")
    if status == "ok":
        raise typer.Exit(code=0)
    if status == "interrupted":
        raise typer.Exit(code=130)
    raise typer.Exit(code=2)


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    try:
        app(args=raw_args, prog_name="orchestrator")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0
