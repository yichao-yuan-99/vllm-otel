#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Docker multi-backend control."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import client  # type: ignore[import-not-found]


class DockerMultiClientTest(unittest.TestCase):
    def test_compose_up_backends_parallel_launches_all_backends_concurrently(self) -> None:
        contexts = [
            {
                "selection": {
                    "port_profile_id": 0,
                    "launch": {"visible_devices": "0"},
                }
            },
            {
                "selection": {
                    "port_profile_id": 1,
                    "launch": {"visible_devices": "1"},
                }
            },
        ]
        barrier = threading.Barrier(3, timeout=3.0)
        results_holder: dict[str, object] = {}
        exception_holder: dict[str, BaseException] = {}

        def fake_compose_up(context: dict[str, object], *, startup_log_path: Path | None = None) -> object:
            barrier.wait(timeout=3.0)
            return client._single.ExecResult(returncode=0, stdout="", stderr="")

        def run_helper() -> None:
            try:
                results_holder["value"] = client._compose_up_backends_parallel(
                    contexts,
                    startup_log_path=Path("/tmp/startup.log"),
                    compose_log_path=Path("/tmp/compose.log"),
                )
            except BaseException as exc:  # pragma: no cover - surfaced by assertions below
                exception_holder["value"] = exc

        with (
            mock.patch.object(client, "_compose_up_backend", side_effect=fake_compose_up),
            mock.patch.object(client, "_append_compose_logs_for_backend"),
            mock.patch.object(client._single, "_emit_progress"),
        ):
            worker = threading.Thread(target=run_helper)
            worker.start()
            try:
                barrier.wait(timeout=3.0)
            except threading.BrokenBarrierError as exc:
                self.fail(f"compose up calls did not run concurrently: {exc}")
            worker.join(timeout=3.0)

        self.assertFalse(worker.is_alive(), "compose helper did not finish")
        self.assertNotIn("value", exception_holder)
        self.assertIn("value", results_holder)
        results = results_holder["value"]
        assert isinstance(results, list)
        self.assertEqual(
            [item["context"]["selection"]["port_profile_id"] for item in results],
            [0, 1],
        )

    def test_parse_port_profile_ids_accepts_commas_and_repeated_flags(self) -> None:
        parsed = client._parse_port_profile_ids(("0,1", "2"))
        self.assertEqual(parsed, [0, 1, 2])

    def test_stop_impl_keeps_selection_visible_when_residual_services_remain(self) -> None:
        selection = {
            "model_key": "qwen3_coder_30b_fp8",
            "port_profile_ids": [2, 13],
            "launch_profile_key": "h100_nvl_gpu23",
            "backend_selections": [
                {"port_profile_id": 2},
                {"port_profile_id": 13},
            ],
        }
        current_state = {
            "active": True,
            "selection": {
                "model_key": "qwen3_coder_30b_fp8",
                "port_profile_ids": [2, 13],
                "launch_profile_key": "h100_nvl_gpu23",
            },
            "lifecycle_state": "running",
        }
        target_state = json.loads(json.dumps(current_state))
        saved_states: list[dict[str, object]] = []
        residual_snapshot = {
            "ok": False,
            "backends": [
                {
                    "port_profile_id": 2,
                    "alive_services": ["vllm"],
                    "live_containers": [],
                    "live_processes": [{"pid": 1234, "cmdline": "python3 -m vllm.entrypoints.openai.api_server --port 31987"}],
                    "snapshot": {},
                }
            ],
            "gateway_multi": {
                "running": False,
                "alive_services": [],
                "status": {},
                "services": {},
            },
        }

        def fake_save_state(state: dict[str, object], *, update_current_alias: bool = True) -> None:
            saved_states.append(json.loads(json.dumps(state)))

        with (
            mock.patch.object(client, "_load_state", side_effect=[current_state, target_state]),
            mock.patch.object(
                client,
                "_compose_contexts_for_multi_selection",
                return_value=(
                    True,
                    [
                        {"project_name": "p2", "selection": {"port_profile_id": 2}},
                        {"project_name": "p13", "selection": {"port_profile_id": 13}},
                    ],
                    "",
                ),
            ),
            mock.patch.object(client, "_stop_gateway_multi", return_value={"ok": True, "message": "gateway_multi stopped"}),
            mock.patch.object(
                client,
                "_compose_down_backend",
                return_value=client._single.ExecResult(returncode=0, stdout="", stderr=""),
            ),
            mock.patch.object(
                client,
                "_build_multi_stop_residual_snapshot",
                side_effect=[residual_snapshot, residual_snapshot],
            ),
            mock.patch.object(client, "_stop_residual_vllm_processes", return_value=[]),
            mock.patch.object(client, "_save_state", side_effect=fake_save_state),
            mock.patch.object(client, "_write_current_state_alias"),
            mock.patch.object(client._single, "_cleanup_compose_context"),
        ):
            payload = client._stop_impl(selection=selection, reason="stop_blocking")

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["code"], 731)
        self.assertEqual(payload["data"]["residual_services"]["backends"][0]["port_profile_id"], 2)
        self.assertEqual(saved_states[-1]["active"], True)
        self.assertEqual(saved_states[-1]["lifecycle_state"], "degraded")

    def test_stop_impl_succeeds_after_residual_process_cleanup(self) -> None:
        selection = {
            "model_key": "qwen3_coder_30b_fp8",
            "port_profile_ids": [2, 13],
            "launch_profile_key": "h100_nvl_gpu23",
            "backend_selections": [
                {"port_profile_id": 2},
                {"port_profile_id": 13},
            ],
        }
        current_state = {
            "active": True,
            "selection": {
                "model_key": "qwen3_coder_30b_fp8",
                "port_profile_ids": [2, 13],
                "launch_profile_key": "h100_nvl_gpu23",
            },
            "lifecycle_state": "running",
        }
        target_state = json.loads(json.dumps(current_state))
        saved_states: list[dict[str, object]] = []
        residual_before = {
            "ok": False,
            "backends": [
                {
                    "port_profile_id": 2,
                    "alive_services": [],
                    "live_containers": [],
                    "live_processes": [{"pid": 1234, "cmdline": "python3 -m vllm.entrypoints.openai.api_server --port 31987"}],
                    "snapshot": {},
                }
            ],
            "gateway_multi": {
                "running": False,
                "alive_services": [],
                "status": {},
                "services": {},
            },
        }
        residual_after = {
            "ok": True,
            "backends": [],
            "gateway_multi": {
                "running": False,
                "alive_services": [],
                "status": {},
                "services": {},
            },
        }

        def fake_save_state(state: dict[str, object], *, update_current_alias: bool = True) -> None:
            saved_states.append(json.loads(json.dumps(state)))

        with (
            mock.patch.object(client, "_load_state", side_effect=[current_state, target_state]),
            mock.patch.object(
                client,
                "_compose_contexts_for_multi_selection",
                return_value=(
                    True,
                    [
                        {"project_name": "p2", "selection": {"port_profile_id": 2}},
                        {"project_name": "p13", "selection": {"port_profile_id": 13}},
                    ],
                    "",
                ),
            ),
            mock.patch.object(client, "_stop_gateway_multi", return_value={"ok": True, "message": "gateway_multi stopped"}),
            mock.patch.object(
                client,
                "_compose_down_backend",
                return_value=client._single.ExecResult(returncode=0, stdout="", stderr=""),
            ),
            mock.patch.object(
                client,
                "_build_multi_stop_residual_snapshot",
                side_effect=[residual_before, residual_after],
            ),
            mock.patch.object(
                client,
                "_stop_residual_vllm_processes",
                return_value=[
                    {
                        "port_profile_id": 2,
                        "pid": 1234,
                        "cmdline": "python3 -m vllm.entrypoints.openai.api_server --port 31987",
                        "stopped": True,
                    }
                ],
            ),
            mock.patch.object(client, "_save_state", side_effect=fake_save_state),
            mock.patch.object(client, "_write_current_state_alias"),
            mock.patch.object(client._single, "_cleanup_compose_context"),
        ):
            payload = client._stop_impl(selection=selection, reason="stop_blocking")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["code"], 0)
        self.assertEqual(payload["data"]["residual_cleanup"][0]["pid"], 1234)
        self.assertEqual(saved_states[-1]["active"], False)
        self.assertEqual(saved_states[-1]["lifecycle_state"], "inactive")

    def test_state_file_for_port_profile_ids_uses_selection_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            with (
                mock.patch.object(client, "RUNTIME_DIR", runtime_dir),
                mock.patch.object(client, "STATE_FILE", runtime_dir / "state.json"),
            ):
                self.assertEqual(
                    client._state_file_for_port_profile_ids([2, 13]),
                    runtime_dir / "state.2-13.json",
                )

    def test_gateway_multi_runtime_files_use_selection_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            with (
                mock.patch.object(client, "RUNTIME_DIR", runtime_dir),
                mock.patch.object(client, "GATEWAY_MULTI_PID_FILE", runtime_dir / "gateway_multi.pid.json"),
                mock.patch.object(client, "GATEWAY_MULTI_LOG_FILE", runtime_dir / "gateway_multi.log"),
            ):
                self.assertEqual(
                    client._gateway_multi_runtime_files([2, 13]),
                    (
                        runtime_dir / "gateway_multi.2-13.pid.json",
                        runtime_dir / "gateway_multi.2-13.log",
                    ),
                )

    def test_save_state_writes_selection_specific_and_current_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            state_file = runtime_dir / "state.json"
            selection_state_file = runtime_dir / "state.2-13.json"
            state = {
                "active": True,
                "selection": {
                    "model_key": "qwen3_coder_30b",
                    "port_profile_ids": [2, 13],
                    "launch_profile_key": "h100_nvl_gpu23",
                },
            }

            with (
                mock.patch.object(client, "RUNTIME_DIR", runtime_dir),
                mock.patch.object(client, "STATE_FILE", state_file),
                mock.patch.object(client, "_utc_now_iso", return_value="2026-04-07T00:00:00+00:00"),
            ):
                client._save_state(state)

            self.assertTrue(state_file.exists())
            self.assertTrue(selection_state_file.exists())
            self.assertEqual(
                json.loads(selection_state_file.read_text(encoding="utf-8"))["selection"]["port_profile_ids"],
                [2, 13],
            )
            self.assertEqual(
                json.loads(state_file.read_text(encoding="utf-8"))["selection"]["port_profile_ids"],
                [2, 13],
            )

    def test_save_state_can_skip_current_alias_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            state_file = runtime_dir / "state.json"
            current_state = {
                "active": True,
                "selection": {
                    "model_key": "qwen3_coder_30b",
                    "port_profile_ids": [18, 19],
                    "launch_profile_key": "h100_nvl_gpu01",
                },
            }
            target_state = {
                "active": False,
                "selection": {
                    "model_key": "qwen3_coder_30b",
                    "port_profile_ids": [2, 13],
                    "launch_profile_key": "h100_nvl_gpu23",
                },
            }
            state_file.write_text(json.dumps(current_state), encoding="utf-8")

            with (
                mock.patch.object(client, "RUNTIME_DIR", runtime_dir),
                mock.patch.object(client, "STATE_FILE", state_file),
                mock.patch.object(client, "_utc_now_iso", return_value="2026-04-07T00:00:00+00:00"),
            ):
                client._save_state(target_state, update_current_alias=False)

            self.assertEqual(
                json.loads(state_file.read_text(encoding="utf-8"))["selection"]["port_profile_ids"],
                [18, 19],
            )
            self.assertEqual(
                json.loads((runtime_dir / "state.2-13.json").read_text(encoding="utf-8"))["selection"]["port_profile_ids"],
                [2, 13],
            )

    def test_save_state_preserves_legacy_unsuffixed_selection_before_alias_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            state_file = runtime_dir / "state.json"
            legacy_state = {
                "active": True,
                "updated_at": "2026-04-06T04:55:06+00:00",
                "selection": {
                    "model_key": "qwen3_coder_30b",
                    "port_profile_ids": [2, 13],
                    "launch_profile_key": "h100_nvl_gpu23",
                },
            }
            new_state = {
                "active": True,
                "selection": {
                    "model_key": "qwen3_coder_30b",
                    "port_profile_ids": [18, 19],
                    "launch_profile_key": "h100_nvl_gpu01",
                },
            }
            state_file.write_text(json.dumps(legacy_state), encoding="utf-8")

            with (
                mock.patch.object(client, "RUNTIME_DIR", runtime_dir),
                mock.patch.object(client, "STATE_FILE", state_file),
                mock.patch.object(client, "_utc_now_iso", return_value="2026-04-07T00:00:00+00:00"),
            ):
                client._save_state(new_state)

            self.assertEqual(
                json.loads((runtime_dir / "state.2-13.json").read_text(encoding="utf-8"))["selection"]["port_profile_ids"],
                [2, 13],
            )
            self.assertEqual(
                json.loads((runtime_dir / "state.18-19.json").read_text(encoding="utf-8"))["selection"]["port_profile_ids"],
                [18, 19],
            )
            self.assertEqual(
                json.loads(state_file.read_text(encoding="utf-8"))["selection"]["port_profile_ids"],
                [18, 19],
            )

    def test_load_state_migrates_matching_legacy_unsuffixed_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            state_file = runtime_dir / "state.json"
            legacy_state = {
                "active": True,
                "updated_at": "2026-04-06T04:55:06+00:00",
                "selection": {
                    "model_key": "qwen3_coder_30b",
                    "port_profile_ids": [2, 13],
                    "launch_profile_key": "h100_nvl_gpu23",
                },
            }
            state_file.write_text(json.dumps(legacy_state), encoding="utf-8")
            selection = {
                "model_key": "qwen3_coder_30b",
                "port_profile_ids": [2, 13],
                "launch_profile_key": "h100_nvl_gpu23",
            }

            with (
                mock.patch.object(client, "RUNTIME_DIR", runtime_dir),
                mock.patch.object(client, "STATE_FILE", state_file),
                mock.patch.object(client, "_utc_now_iso", return_value="2026-04-07T00:00:00+00:00"),
            ):
                loaded = client._load_state(selection)

            self.assertEqual(loaded["selection"]["port_profile_ids"], [2, 13])
            self.assertEqual(
                json.loads((runtime_dir / "state.2-13.json").read_text(encoding="utf-8"))["selection"]["port_profile_ids"],
                [2, 13],
            )

    def test_load_state_falls_back_to_latest_active_selection_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            state_file = runtime_dir / "state.json"
            state_file.write_text(
                json.dumps({"active": False, "lifecycle_state": "inactive"}),
                encoding="utf-8",
            )
            older_state = {
                "active": True,
                "updated_at": "2026-04-07T00:00:00+00:00",
                "selection": {
                    "model_key": "qwen3_coder_30b",
                    "port_profile_ids": [2, 13],
                    "launch_profile_key": "h100_nvl_gpu23",
                },
            }
            newer_state = {
                "active": True,
                "updated_at": "2026-04-07T01:00:00+00:00",
                "selection": {
                    "model_key": "qwen3_coder_30b",
                    "port_profile_ids": [18, 19],
                    "launch_profile_key": "h100_nvl_gpu01",
                },
            }
            (runtime_dir / "state.2-13.json").write_text(json.dumps(older_state), encoding="utf-8")
            (runtime_dir / "state.18-19.json").write_text(json.dumps(newer_state), encoding="utf-8")

            with (
                mock.patch.object(client, "RUNTIME_DIR", runtime_dir),
                mock.patch.object(client, "STATE_FILE", state_file),
                mock.patch.object(client, "_utc_now_iso", return_value="2026-04-07T02:00:00+00:00"),
            ):
                loaded = client._load_state()

            self.assertEqual(loaded["selection"]["port_profile_ids"], [18, 19])

    def test_split_launch_profile_requires_matching_gpu_count(self) -> None:
        launch = {
            "label": "H100 NVL GPUs 2,3",
            "gpu_type": "NVIDIA H100 NVL",
            "visible_devices": "2,3",
            "visible_device_ids": ["2", "3"],
            "per_gpu_memory_gb": 80.0,
            "total_gpu_memory_gb": 160.0,
            "tensor_parallel_size": 2,
        }
        split = client._split_launch_profile_across_backends(
            launch_profile=launch,
            backend_count=2,
        )
        self.assertEqual(
            split,
            [
                {
                    "label": "H100 NVL GPUs 2,3",
                    "gpu_type": "NVIDIA H100 NVL",
                    "visible_devices": "2",
                    "visible_device_ids": ["2"],
                    "per_gpu_memory_gb": 80.0,
                    "total_gpu_memory_gb": 80.0,
                    "tensor_parallel_size": 1,
                },
                {
                    "label": "H100 NVL GPUs 2,3",
                    "gpu_type": "NVIDIA H100 NVL",
                    "visible_devices": "3",
                    "visible_device_ids": ["3"],
                    "per_gpu_memory_gb": 80.0,
                    "total_gpu_memory_gb": 80.0,
                    "tensor_parallel_size": 1,
                },
            ],
        )

        with self.assertRaisesRegex(ValueError, "GPU count must match"):
            client._split_launch_profile_across_backends(
                launch_profile=launch,
                backend_count=3,
            )

    def test_validate_start_selection_multi_builds_one_backend_per_profile(self) -> None:
        ok, payload = client._validate_start_selection_multi(
            model_key="qwen3_coder_30b",
            port_profile_ids=[0, 1],
            launch_profile_key="h100_nvl_gpu23",
            enforce_weight_limit=False,
        )

        self.assertTrue(ok)
        data = payload["data"]
        self.assertEqual(data["port_profile_ids"], [0, 1])
        self.assertEqual(data["control_port_profile_id"], 0)
        self.assertEqual(data["assignment_policy"], client.DEFAULT_ASSIGNMENT_POLICY)
        self.assertEqual(len(data["backend_selections"]), 2)
        self.assertEqual(
            [item["launch"]["visible_devices"] for item in data["backend_selections"]],
            ["2", "3"],
        )
        self.assertEqual(
            [item["launch"]["tensor_parallel_size"] for item in data["backend_selections"]],
            [1, 1],
        )
        self.assertNotEqual(
            data["backend_selections"][0]["runtime_names"]["compose_project_name"],
            data["backend_selections"][1]["runtime_names"]["compose_project_name"],
        )

    def test_start_gateway_multi_uses_repeated_port_profile_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "gateway_multi.toml"
            config_path.write_text("schema_version = 1\n", encoding="utf-8")
            pid_file = tmp_path / "gateway_multi.pid"
            log_file = tmp_path / "gateway_multi.log"

            class FakeProc:
                pid = 6789

                def poll(self) -> None:
                    return None

            selection = {
                "port_profile_ids": [0, 1],
                "assignment_policy": "round_robin",
            }

            with (
                mock.patch.object(client, "_gateway_multi_runtime_files", return_value=(pid_file, log_file)),
                mock.patch.object(client, "_resolve_gateway_multi_config_path", return_value=config_path),
                mock.patch.object(client, "_ensure_runtime_dir"),
                mock.patch.object(client, "_utc_now_iso", return_value="2026-04-05T19:00:00+00:00"),
                mock.patch.object(client.time, "sleep"),
                mock.patch.object(client.time, "monotonic", side_effect=[0.0, 0.1, 3.1]),
                mock.patch.object(client._single.subprocess, "Popen", return_value=FakeProc()),
            ):
                payload = client._start_gateway_multi(selection)

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["data"]["module"], client.DEFAULT_GATEWAY_MULTI_MODULE_NAME)
            self.assertEqual(payload["data"]["port_profile_ids"], [0, 1])
            self.assertEqual(payload["data"]["command"][2], client.DEFAULT_GATEWAY_MULTI_MODULE_NAME)
            self.assertIn("--port-profile-id", payload["data"]["command"])
            self.assertIn("0", payload["data"]["command"])
            self.assertIn("1", payload["data"]["command"])
            self.assertIn("--policy", payload["data"]["command"])
            self.assertIn("gateway_multi start", log_file.read_text(encoding="utf-8"))

            record = json.loads(pid_file.read_text(encoding="utf-8"))
            self.assertEqual(record["module"], client.DEFAULT_GATEWAY_MULTI_MODULE_NAME)
            self.assertEqual(record["port_profile_ids"], [0, 1])


if __name__ == "__main__":
    unittest.main()
