#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the embedded TP=1 sbatch renderer."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import textwrap
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from embedded_tp1 import EmbeddedTp1Launcher  # type: ignore[import-not-found]
from control_plane import ControlPlaneError  # type: ignore[import-not-found]


class EmbeddedTp1LauncherTest(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        config_path = root / "server_config.toml"
        config_path.write_text(
            textwrap.dedent(
                f"""\
                [server]
                host = "0.0.0.0"
                port = 23971

                [cluster]
                login_host = "login"
                job_name_prefix = "embedded_tp1_test_"
                job_nodes = 1
                startup_timeout = 900
                startup_timeout_after_running = true
                stop_wait_timeout_seconds = 180
                stop_poll_interval_seconds = 2
                wait_up_poll_interval_seconds = 2
                run_dir = "{root / "run"}"
                log_dir = "{root / "logs"}"
                state_file = "{root / "run" / "control_state.json"}"

                [images]
                jaeger_image = "docker://jaegertracing/all-in-one:1.57"
                vllm_image = "docker://example/vllm:test"

                [partition.mi3001x]
                gpus_per_node = 1
                gpu_memory_gb = 192
                total_vram_gb = 192
                max_time = "04:00:00"

                [partition.mi3008x]
                gpus_per_node = 8
                gpu_memory_gb = 192
                total_vram_gb = 1536
                max_time = "12:00:00"
                """
            ),
            encoding="utf-8",
        )
        return config_path

    def _write_experiment_script(self, root: Path) -> Path:
        script_path = root / "experiment.sh"
        script_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        script_path.chmod(0o750)
        return script_path

    def test_mi3001x_renders_single_profile_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(root)
            experiment_script = self._write_experiment_script(root)
            launcher = EmbeddedTp1Launcher(config_path)

            rendered = launcher.render(
                partition="mi3001x",
                model="qwen3_coder_30b",
                experiment_script=experiment_script,
            )

            self.assertEqual(rendered["profile_list"], [0])
            script_text = Path(rendered["sbatch_script"]).read_text(encoding="utf-8")
            self.assertIn("VLLM_TENSOR_PARALLEL_SIZE=1", script_text)
            self.assertIn("launch_vllm 0 ", script_text)
            self.assertNotIn("launch_vllm 1 ", script_text)
            self.assertIn("launch_experiment 0", script_text)

    def test_mi3008x_renders_unrolled_eight_service_stacks_without_srun(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(root)
            experiment_script = self._write_experiment_script(root)
            launcher = EmbeddedTp1Launcher(config_path)

            rendered = launcher.render(
                partition="mi3008x",
                model="qwen3_coder_30b",
                experiment_script=experiment_script,
            )

            self.assertEqual(rendered["profile_list"], list(range(8)))
            script_text = Path(rendered["sbatch_script"]).read_text(encoding="utf-8")
            self.assertNotIn("srun", script_text)
            for profile_id in range(8):
                self.assertIn(f"launch_vllm {profile_id} ", script_text)
                self.assertIn(f"launch_gateway {profile_id} ", script_text)
                self.assertIn(f"launch_experiment {profile_id}", script_text)
            self.assertIn("ROCR_VISIBLE_DEVICES", script_text)

    def test_rejects_models_that_do_not_fit_on_one_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(root)
            experiment_script = self._write_experiment_script(root)
            launcher = EmbeddedTp1Launcher(config_path)

            with self.assertRaisesRegex(ControlPlaneError, "single-GPU VRAM"):
                launcher.render(
                    partition="mi3008x",
                    model="qwen3_235b_fp8",
                    experiment_script=experiment_script,
                )


if __name__ == "__main__":
    unittest.main()
