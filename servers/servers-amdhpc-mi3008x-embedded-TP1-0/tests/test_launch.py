#!/usr/bin/env python3
"""Unit tests for the mi3008x embedded TP1 profile 0 sbatch renderer."""

from __future__ import annotations

from pathlib import Path
import shlex
import sys
import tempfile
import textwrap
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from launch import Mi3008xEmbeddedTp1Profile0Launcher, LaunchError  # type: ignore[import-not-found]


class Mi3008xEmbeddedTp1Profile0LauncherTest(unittest.TestCase):
    def _write_model_config(self, root: Path) -> Path:
        config_path = root / "model_config.toml"
        config_path.write_text(
            textwrap.dedent(
                """\
                schema_version = 1

                [models.qwen3_coder_30b_fp8]
                vllm_model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
                served_model_name = "Qwen3-Coder-30B-A3B-Instruct-FP8"
                weight_vram_gb = 35
                extra_args = []

                [models.qwen3_235b_fp8]
                vllm_model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
                served_model_name = "Qwen3-235B-A22B-Instruct-2507-FP8"
                weight_vram_gb = 240
                extra_args = []
                """
            ),
            encoding="utf-8",
        )
        return config_path

    def _write_start_script(self, root: Path) -> Path:
        path = root / "start-services.sh"
        path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def _write_experiment_script(self, root: Path) -> Path:
        path = root / "experiment.sh"
        path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def test_render_writes_simple_mi3008x_profile0_sbatch_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            start_script = self._write_start_script(root)
            launcher = Mi3008xEmbeddedTp1Profile0Launcher(
                model_config_path=self._write_model_config(root),
                start_script=start_script,
                run_dir=root / "run",
                log_dir=root / "logs",
            )
            experiment_script = self._write_experiment_script(root)

            rendered = launcher.render(
                model="qwen3_coder_30b_fp8",
                experiment_script=experiment_script,
                extra_env={"FOO": "bar"},
                lmcache_max_local_cpu_size="32",
                extra_vllm_args=["--max-num-seqs", "16"],
                no_async_scheduling=True,
                time_limit="1-00:00:00",
            )

            self.assertEqual(rendered["partition"], "mi3008x")
            self.assertEqual(rendered["profile_list"], [0])
            self.assertEqual(rendered["service_count"], 1)
            self.assertEqual(rendered["profile_id"], 0)
            self.assertEqual(rendered["gpu_id"], 0)
            self.assertTrue(rendered["lmcache_enabled"])
            self.assertTrue(rendered["no_async_scheduling"])
            self.assertEqual(rendered["time_limit"], "1-00:00:00")
            self.assertEqual(rendered["extra_env"], {"FOO": "bar", "LMCACHE_MAX_LOCAL_CPU_SIZE": "32"})
            self.assertIn("--max-num-seqs", rendered["effective_model_extra_args"])
            self.assertIn("--no-async-scheduling", rendered["effective_model_extra_args"])
            self.assertIn("--trust-remote-code", rendered["effective_model_extra_args"])
            self.assertIn("--kv-transfer-config", rendered["effective_model_extra_args"])

            script_text = Path(rendered["sbatch_script"]).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --partition=mi3008x", script_text)
            self.assertIn("#SBATCH --time=1-00:00:00", script_text)
            self.assertIn(f"export EXPERIMENT_SCRIPT={shlex.quote(str(experiment_script.resolve()))}", script_text)
            self.assertIn(f"export VLLM_MODEL_KEY={shlex.quote('qwen3_coder_30b_fp8')}", script_text)
            self.assertIn('export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"', script_text)
            self.assertIn(
                'export AMD_SMI_POWER_DAEMON_BIN="${AMD_SMI_POWER_DAEMON_BIN:-'
                f"{shlex.quote(str((MODULE_ROOT.parents[1] / '.venv' / 'bin' / 'amd-smi-power-daemon').resolve()))}"
                '}"',
                script_text,
            )
            self.assertIn(
                'export AMD_POWER_READER_BIN="${AMD_POWER_READER_BIN:-'
                f"{shlex.quote(str((MODULE_ROOT.parents[1] / '.venv' / 'bin' / 'amd-power-reader').resolve()))}"
                '}"',
                script_text,
            )
            self.assertIn('export AMD_SMI_POWER_SOCKET_PATH="${AMD_SMI_POWER_SOCKET_PATH:-/tmp/amdsmi-power-reader.${SLURM_JOB_ID}.sock}"', script_text)
            self.assertIn(f"bash {shlex.quote(str(start_script.resolve()))}", script_text)

    def test_rejects_models_that_do_not_fit_on_one_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            launcher = Mi3008xEmbeddedTp1Profile0Launcher(
                model_config_path=self._write_model_config(root),
                start_script=self._write_start_script(root),
                run_dir=root / "run",
                log_dir=root / "logs",
            )
            experiment_script = self._write_experiment_script(root)

            with self.assertRaisesRegex(LaunchError, "single-GPU VRAM"):
                launcher.render(
                    model="qwen3_235b_fp8",
                    experiment_script=experiment_script,
                )

    def test_extract_sbatch_job_id_handles_cluster_banner_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            launcher = Mi3008xEmbeddedTp1Profile0Launcher(
                model_config_path=self._write_model_config(root),
                start_script=self._write_start_script(root),
                run_dir=root / "run",
                log_dir=root / "logs",
            )

            sbatch_output = textwrap.dedent(
                """\
                Submitted batch job 289656

                sbatch: ---------------------------------------------------------------
                sbatch: AMD AI & HPC Fund Job Submission Filter
                sbatch: ---------------------------------------------------------------
                sbatch: --> ok: runtime limit specified
                sbatch:    --> ok: partition provided mi3008x
                """
            )

            self.assertEqual(launcher._extract_sbatch_job_id(sbatch_output), "289656")


if __name__ == "__main__":
    unittest.main()
