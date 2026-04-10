#!/usr/bin/env python3
"""Unit tests for the interactive embedded TP2 background controller."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import textwrap
import time
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from client import BackgroundServiceController  # type: ignore[import-not-found]


class BackgroundServiceControllerTest(unittest.TestCase):
    def _write_fake_start_script(self, root: Path) -> Path:
        script_path = root / "fake-start.sh"
        script_path.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                set -euo pipefail
                : "${TEST_MARKER_FILE:?}"
                printf "started\\n" >> "${TEST_MARKER_FILE}"
                echo "fake service booting"
                sleep "${TEST_READY_DELAY_SECONDS:-0.5}"
                echo "Services are ready."
                printf "ready\\n" >> "${TEST_MARKER_FILE}"
                trap 'echo "fake service stopping"; printf "stopped\\n" >> "${TEST_MARKER_FILE}"; exit 0' TERM INT
                while true; do
                  sleep 1
                done
                """
            ),
            encoding="utf-8",
        )
        script_path.chmod(0o755)
        return script_path

    def test_start_is_idempotent_and_stop_terminates_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script_path = self._write_fake_start_script(root)
            log_dir = root / "logs"
            pid_file = log_dir / "background-service.pid.json"
            marker_file = root / "marker.txt"
            controller = BackgroundServiceController(
                start_script=script_path,
                repo_root=root,
                log_dir=log_dir,
                pid_file=pid_file,
            )

            started_at = time.monotonic()
            state, record = controller.start(
                env={
                    "RUN_ID": "test-run",
                    "TEST_MARKER_FILE": str(marker_file),
                    "TEST_READY_DELAY_SECONDS": "0.5",
                }
            )
            elapsed_seconds = time.monotonic() - started_at
            self.assertEqual(state, "started")
            self.assertEqual(record.run_id, "test-run")
            self.assertTrue(pid_file.exists())
            self.assertTrue(Path(record.launcher_log).exists())
            self.assertGreaterEqual(elapsed_seconds, 0.45)
            self.assertIn("ready", marker_file.read_text(encoding="utf-8"))

            status, running_record = controller.status()
            self.assertEqual(status, "running")
            self.assertIsNotNone(running_record)
            self.assertEqual(running_record.pid, record.pid)

            second_state, second_record = controller.start(
                env={"RUN_ID": "ignored", "TEST_MARKER_FILE": str(marker_file)}
            )
            self.assertEqual(second_state, "already_running")
            self.assertEqual(second_record.pid, record.pid)

            stop_state, stopped_record = controller.stop(timeout_seconds=5.0)
            self.assertEqual(stop_state, "stopped")
            self.assertIsNotNone(stopped_record)
            self.assertEqual(stopped_record.pid, record.pid)

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                marker_lines = marker_file.read_text(encoding="utf-8").splitlines()
                if "stopped" in marker_lines:
                    break
                time.sleep(0.1)
            self.assertIn("stopped", marker_file.read_text(encoding="utf-8"))
            self.assertFalse(pid_file.exists())

    def test_status_cleans_stale_pid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script_path = self._write_fake_start_script(root)
            log_dir = root / "logs"
            pid_file = log_dir / "background-service.pid.json"
            log_dir.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(
                textwrap.dedent(
                    f"""\
                    {{
                      "pid": 999999,
                      "run_id": "stale-run",
                      "started_at": "2026-04-07T00:00:00Z",
                      "pid_file": "{pid_file}",
                      "launcher_log": "{log_dir / 'launcher.stale-run.log'}",
                      "start_script": "{script_path}",
                      "repo_root": "{root}"
                    }}
                    """
                ),
                encoding="utf-8",
            )
            controller = BackgroundServiceController(
                start_script=script_path,
                repo_root=root,
                log_dir=log_dir,
                pid_file=pid_file,
            )

            state, record = controller.status(clean_stale=True)
            self.assertEqual(state, "not_running")
            self.assertIsNotNone(record)
            self.assertEqual(record.run_id, "stale-run")
            self.assertFalse(pid_file.exists())

    def test_start_waits_when_service_is_already_starting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script_path = self._write_fake_start_script(root)
            log_dir = root / "logs"
            pid_file = log_dir / "background-service.pid.json"
            marker_file = root / "marker.txt"
            controller = BackgroundServiceController(
                start_script=script_path,
                repo_root=root,
                log_dir=log_dir,
                pid_file=pid_file,
            )

            initial_state, initial_record = controller.start(
                env={
                    "RUN_ID": "test-run-inflight",
                    "TEST_MARKER_FILE": str(marker_file),
                    "TEST_READY_DELAY_SECONDS": "0.8",
                },
                wait_for_ready=False,
            )
            self.assertEqual(initial_state, "started")

            started_at = time.monotonic()
            second_state, second_record = controller.start()
            elapsed_seconds = time.monotonic() - started_at

            self.assertEqual(second_state, "already_running")
            self.assertEqual(second_record.pid, initial_record.pid)
            self.assertGreaterEqual(elapsed_seconds, 0.6)
            self.assertIn("ready", marker_file.read_text(encoding="utf-8"))

            controller.stop(timeout_seconds=5.0)

    def test_start_and_stop_emit_progress_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script_path = self._write_fake_start_script(root)
            log_dir = root / "logs"
            pid_file = log_dir / "background-service.pid.json"
            marker_file = root / "marker.txt"
            controller = BackgroundServiceController(
                start_script=script_path,
                repo_root=root,
                log_dir=log_dir,
                pid_file=pid_file,
            )
            progress_messages: list[str] = []

            controller.start(
                env={
                    "RUN_ID": "test-run-progress",
                    "TEST_MARKER_FILE": str(marker_file),
                    "TEST_READY_DELAY_SECONDS": "0.2",
                },
                progress=progress_messages.append,
            )
            controller.stop(timeout_seconds=5.0, progress=progress_messages.append)

            self.assertTrue(
                any("Starting background service with run_id=test-run-progress" in message for message in progress_messages)
            )
            self.assertTrue(any("Waiting for service startup to complete..." in message for message in progress_messages))
            self.assertTrue(any("launcher: fake service booting" in message for message in progress_messages))
            self.assertTrue(any("launcher: Services are ready." in message for message in progress_messages))
            self.assertTrue(any("Stopping background service pid=" in message for message in progress_messages))
            self.assertTrue(any("launcher: fake service stopping" in message for message in progress_messages))


if __name__ == "__main__":
    unittest.main()
