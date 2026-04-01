from __future__ import annotations

import importlib.util
import json
import re
import stat
import sys
import tomllib
from pathlib import Path


def load_generator_module() -> object:
    module_path = (
        Path(__file__).resolve().parents[1] / "local" / "generate_experiment.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "generate_single_maxcon_clean_experiment",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_single_maxcon_clean_experiment"] = module
    spec.loader.exec_module(module)
    return module


def resolve_single_output_batch_dir(output_config_root: Path) -> Path:
    candidates = sorted(path for path in output_config_root.iterdir() if path.is_dir())
    assert len(candidates) == 1
    batch_dir = candidates[0]
    assert re.fullmatch(r"\d{8}T\d{6}Z", batch_dir.name) is not None
    return batch_dir


def write_minimal_model_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "",
                "[models.test_model]",
                'served_model_name = "Test Served Model"',
                'vllm_model_name = "org/test-model"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_main_generates_single_maxcon_clean_bundle(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    results_root = repo_root / "results"
    source_run_dir = results_root / "model-a" / "dataset-a" / "agent-a" / "run-1"
    source_run_dir.mkdir(parents=True)

    plan_payload = {
        "replay_target": {
            "model": "Test Served Model",
        },
        "launch_policy": {
            "strategy": "config_ordered",
            "max_concurrent": 25,
            "pattern": {"name": "eager"},
            "pattern_args": {},
        },
        "workers": [],
    }
    top_plan_path = source_run_dir / "replay-plan.clean.token.top.json"
    rest_plan_path = source_run_dir / "replay-plan.clean.token.rest.json"
    exclude_plan_path = source_run_dir / "replay-plan.clean.token.exclude-unranked.json"
    for plan_path in (top_plan_path, rest_plan_path, exclude_plan_path):
        plan_path.write_text(
            json.dumps(plan_payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    output_config_dir = tmp_path / "generated"
    module.REPO_ROOT = repo_root
    module.RESULTS_ROOT = results_root
    module.MODEL_CONFIG_PATH = repo_root / "configs" / "model_config.toml"
    write_minimal_model_config(module.MODEL_CONFIG_PATH)

    exit_code = module.main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--randomize-seed",
            "11",
            "--max-concurrent",
            "8",
            "--time-constraint-s",
            "600",
            "--target-model",
            "test_model",
            "--port-profile",
            "3",
            "--split",
            "full",
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    batch_dir = resolve_single_output_batch_dir(output_config_dir)
    replay_config = tomllib.loads((batch_dir / "replay.toml").read_text(encoding="utf-8"))
    replay_payload = replay_config["replay"]
    assert replay_payload["randomize_seed"] == 11
    assert replay_payload["time_constraint_s"] == 600.0
    assert replay_payload["port_profile_id"] == 3
    assert replay_payload["launch_policy_override"]["max_concurrent"] == 8
    assert "pattern" not in replay_payload["launch_policy_override"]

    expected_plan_copy = batch_dir / "plan" / exclude_plan_path.name
    assert expected_plan_copy.read_text(encoding="utf-8") == exclude_plan_path.read_text(
        encoding="utf-8"
    )
    assert replay_payload["plan"] == module.path_for_config(expected_plan_copy)

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["split"] == "exclude-unranked"
    assert manifest["split_two_group_metric"] == "token_usage"
    assert manifest["max_concurrent"] == 8
    assert manifest["max_concurrent_slug"] == "c8"
    assert manifest["source_dataset_lineage"] == "dataset-a/agent-a"
    assert manifest["selected_source_plan"] == str(exclude_plan_path)
    assert manifest["selected_plan_copy"] == str(expected_plan_copy)
    assert manifest["related_source_plans"]["top"] == str(top_plan_path)
    assert manifest["related_source_plans"]["rest"] == str(rest_plan_path)
    assert manifest["related_source_plans"]["exclude_unranked"] == str(exclude_plan_path)
    assert manifest["replay_output_dir"].endswith(
        "results/replay/single-maxcon-clean/dataset-a/agent-a/"
        f"split/exclude-unranked/c8/{manifest['batch_timestamp']}"
    )

    run_script_path = batch_dir / "run_replay.sh"
    run_script_text = run_script_path.read_text(encoding="utf-8")
    assert "[single-maxcon-clean]" in run_script_text
    assert "max_concurrent=8" in run_script_text
    assert "--config" in run_script_text
    assert run_script_path.stat().st_mode & stat.S_IXUSR
