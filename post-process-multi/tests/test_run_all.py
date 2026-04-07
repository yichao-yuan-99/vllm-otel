from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_all.py"
MODULE_NAME = "post_process_multi_run_all"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec for {MODULE_PATH}")
run_all = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = run_all
SPEC.loader.exec_module(run_all)


def _script_ref(command: list[str]) -> str:
    path = Path(command[1]).as_posix()
    if "post-process-multi/" in path:
        return "post-process-multi/" + path.split("post-process-multi/")[-1]
    if "post-process/" in path:
        return "post-process/" + path.split("post-process/")[-1]
    return path


def test_main_run_dir_prefers_multi_overrides(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    commands: list[list[str]] = []

    def fake_run_command(command: list[str]) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(run_all, "_run_command", fake_run_command)

    exit_code = run_all.main(["--run-dir", str(run_dir), "--skip-visualization"])

    assert exit_code == 0
    refs = [_script_ref(command) for command in commands]
    assert refs[0] == "post-process/service-failure/extract_run.py"
    assert "post-process-multi/gateway/ctx-aware-log/extract_run.py" in refs
    assert "post-process/gateway/ctx-aware-log/extract_run.py" not in refs
    for command in commands:
        assert command[2:] == ["--run-dir", str(run_dir.resolve())]


def test_main_run_dir_prefers_multi_visualization_override(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    commands: list[list[str]] = []

    def fake_run_command(command: list[str]) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(run_all, "_run_command", fake_run_command)

    exit_code = run_all.main(["--run-dir", str(run_dir)])

    assert exit_code == 0
    refs = [_script_ref(command) for command in commands]
    assert "post-process-multi/visualization/gateway-ctx-aware/generate_all_figures.py" in refs
    assert "post-process/visualization/gateway-ctx-aware/generate_all_figures.py" not in refs
    assert "post-process/visualization/vllm-metrics/generate_all_figures.py" in refs
