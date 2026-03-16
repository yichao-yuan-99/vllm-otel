from __future__ import annotations

import importlib.util
import json
import re
import shlex
import sys
import tomllib
from pathlib import Path


def load_generator_module() -> object:
    module_path = (
        Path(__file__).resolve().parents[1] / "generate_replay_configs.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "generate_replay_qps_local_mix_configs",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_replay_qps_local_mix_configs"] = module
    spec.loader.exec_module(module)
    return module


def resolve_output_batch_dir(output_config_root: Path) -> Path:
    candidates = sorted(
        path
        for path in output_config_root.iterdir()
        if path.is_dir() and re.fullmatch(r"\d{8}T\d{6}Z", path.name) is not None
    )
    assert len(candidates) == 1
    batch_dir = candidates[0]
    return batch_dir


def build_plan_payload(*, worker_prefix: str, worker_count: int, split_group: str) -> dict[str, object]:
    workers: list[dict[str, object]] = []
    for index in range(worker_count):
        workers.append(
            {
                "worker_id": f"{worker_prefix}-{index}",
                "trial_id": f"trial-{worker_prefix}-{index}",
                "source_trail_name": f"trail-{worker_prefix}-{index}",
                "run_offset_s": float(index),
                "launch_priority": index,
                "requests": [
                    {
                        "index": 0,
                        "method": "POST",
                        "path": "v1/chat/completions",
                        "body": {"messages": [{"role": "user", "content": "hello"}]},
                        "response": {"status_code": 200},
                    }
                ],
            }
        )

    return {
        "schema_version": "replay-plan.v1",
        "compile_version": "test",
        "compiled_at": "2026-03-16T00:00:00Z",
        "source_job_dir": "/tmp/fake-job",
        "backend": "harbor",
        "t0": "2026-03-16T00:00:00Z",
        "t0_source": "test",
        "replay_target": {"model": "fake", "deterministic_required": True},
        "launch_policy": {
            "strategy": "config_ordered",
            "source": "test",
            "max_concurrent": 10,
            "seed": None,
            "pattern": {"name": "eager"},
            "pattern_args": {},
        },
        "workers": workers,
        "compile_options": {"exclude_unranked_trails": False},
        "split_two_group": {
            "enabled": True,
            "metric": "token_usage",
            "source_path": "/tmp/fake-split.json",
            "source_selected_p": 0.3,
            "group": split_group,
        },
    }


def test_derive_default_split_plan_paths_detects_additional_suffix(tmp_path: Path) -> None:
    module = load_generator_module()
    source_run_dir = tmp_path / "run-1"
    source_run_dir.mkdir(parents=True)

    top_path = (source_run_dir / "replay-plan.token.top.v2.json").resolve()
    rest_path = (source_run_dir / "replay-plan.token.rest.v2.json").resolve()
    top_path.write_text("{}", encoding="utf-8")
    rest_path.write_text("{}", encoding="utf-8")

    resolved = module.derive_default_split_plan_paths(
        (source_run_dir / "replay-plan.json").resolve()
    )
    assert resolved["top"] == top_path
    assert resolved["rest"] == rest_path


def test_derive_default_split_plan_paths_prefers_explicit_suffix(tmp_path: Path) -> None:
    module = load_generator_module()
    source_run_dir = tmp_path / "run-1"
    source_run_dir.mkdir(parents=True)

    unsuffixed_top_path = (source_run_dir / "replay-plan.token.top.json").resolve()
    unsuffixed_rest_path = (source_run_dir / "replay-plan.token.rest.json").resolve()
    suffixed_top_path = (source_run_dir / "replay-plan.token.top.fp16.json").resolve()
    suffixed_rest_path = (source_run_dir / "replay-plan.token.rest.fp16.json").resolve()

    unsuffixed_top_path.write_text("{}", encoding="utf-8")
    unsuffixed_rest_path.write_text("{}", encoding="utf-8")
    suffixed_top_path.write_text("{}", encoding="utf-8")
    suffixed_rest_path.write_text("{}", encoding="utf-8")

    resolved = module.derive_default_split_plan_paths(
        (source_run_dir / "replay-plan.json").resolve(),
        preferred_suffix="fp16",
    )
    assert resolved["top"] == suffixed_top_path
    assert resolved["rest"] == suffixed_rest_path


def test_derive_default_split_plan_paths_handles_suffixed_base_plan_path(
    tmp_path: Path,
) -> None:
    module = load_generator_module()
    source_run_dir = tmp_path / "run-1"
    source_run_dir.mkdir(parents=True)

    top_path = (source_run_dir / "replay-plan.token.top.v2.json").resolve()
    rest_path = (source_run_dir / "replay-plan.token.rest.v2.json").resolve()
    top_path.write_text("{}", encoding="utf-8")
    rest_path.write_text("{}", encoding="utf-8")

    resolved = module.derive_default_split_plan_paths(
        (source_run_dir / "replay-plan.v2.json").resolve(),
        preferred_suffix="v2",
    )
    assert resolved["top"] == top_path
    assert resolved["rest"] == rest_path


def test_main_generates_local_mode_mix_bundle(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    results_root = repo_root / "results"
    source_run_dir = results_root / "model-a" / "dataset-b" / "agent-c" / "run-1"
    source_run_dir.mkdir(parents=True)

    split_top_plan_path = (source_run_dir / "replay-plan.token.top.json").resolve()
    split_rest_plan_path = (source_run_dir / "replay-plan.token.rest.json").resolve()
    split_top_plan_path.write_text(
        json.dumps(
            build_plan_payload(worker_prefix="top", worker_count=4, split_group="top"),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    split_rest_plan_path.write_text(
        json.dumps(
            build_plan_payload(worker_prefix="rest", worker_count=2, split_group="rest"),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    server_config_path = repo_root / "servers" / "servers-amdhpc" / "server_config.toml"
    server_config_path.parent.mkdir(parents=True)
    server_config_path.write_text("[server]\nport=1\n", encoding="utf-8")

    output_config_dir = tmp_path / "generated"
    rendered_dir = tmp_path / "rendered"
    rendered_dir.mkdir(parents=True)

    module.REPO_ROOT = repo_root
    module.RESULTS_ROOT = results_root

    def fake_build_control_plane(server_config: Path) -> object:
        assert server_config.resolve() == server_config_path.resolve()
        return object()

    def fake_render_local_mode_sbatch(
        *,
        control_plane: object,
        port_profile_id: int,
        partition: str,
        model: str,
        local_mode_script_path: Path,
        check_port_availability: bool,
        lmcache_max_local_cpu_size: str | None,
        no_async_scheduling: bool,
    ) -> Path:
        del control_plane, check_port_availability
        assert port_profile_id == 0
        assert partition == "mi3001x"
        assert model == "qwen3_coder_30b"
        assert lmcache_max_local_cpu_size == "100"
        assert no_async_scheduling is False
        assert local_mode_script_path.exists()
        percent_name = local_mode_script_path.parent.name
        target = rendered_dir / f"{percent_name}.sh"
        target.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "#SBATCH --output=/tmp/original/slurm.%j.out\n"
            "#SBATCH --error=/tmp/original/slurm.%j.err\n"
            "JOB_LOG_DIR=/tmp/original\n"
            "GATEWAY_CONFIG_DEFAULT=/tmp/original-gateway-config.toml\n"
            "echo fake\n",
            encoding="utf-8",
        )
        return target

    module.build_control_plane = fake_build_control_plane
    module.render_local_mode_sbatch = fake_render_local_mode_sbatch

    exit_code = module.main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--qps",
            "0.1",
            "--top-percent-list",
            "0,50,100",
            "--poisson-seed",
            "7",
            "--randomize-seed",
            "11",
            "--time-constraint-s",
            "600",
            "--partition",
            "mi3001x",
            "--model",
            "qwen3_coder_30b",
            "--lmcache",
            "100",
            "--server-config",
            str(server_config_path),
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert exit_code == 0

    batch_dir = resolve_output_batch_dir(output_config_dir)
    p0_dir = batch_dir / "p0"
    p50_dir = batch_dir / "p50"
    p100_dir = batch_dir / "p100"

    for exp_dir in (p0_dir, p50_dir, p100_dir):
        assert (exp_dir / "replay.toml").exists()
        assert (exp_dir / "run_local_replay.sh").exists()
        assert (exp_dir / "gateway-config.toml").exists()
        assert (exp_dir / "sbatch.sh").exists()
        plan_candidates = sorted((exp_dir / "plan").glob("replay-plan.mix.*.json"))
        assert len(plan_candidates) == 1

    mixed_plan_p0_path = sorted((p0_dir / "plan").glob("replay-plan.mix.*.json"))[0]
    mixed_plan_p50_path = sorted((p50_dir / "plan").glob("replay-plan.mix.*.json"))[0]
    mixed_plan_p100_path = sorted((p100_dir / "plan").glob("replay-plan.mix.*.json"))[0]

    mixed_plan_p0 = json.loads(mixed_plan_p0_path.read_text(encoding="utf-8"))
    mixed_plan_p50 = json.loads(mixed_plan_p50_path.read_text(encoding="utf-8"))
    mixed_plan_p100 = json.loads(mixed_plan_p100_path.read_text(encoding="utf-8"))

    assert mixed_plan_p0["mix_top_rest"]["top_workers_selected"] == 0
    assert mixed_plan_p50["mix_top_rest"]["top_workers_selected"] == 2
    assert mixed_plan_p100["mix_top_rest"]["top_workers_selected"] == 4
    assert len(mixed_plan_p0["workers"]) == 2
    assert len(mixed_plan_p50["workers"]) == 4
    assert len(mixed_plan_p100["workers"]) == 6
    assert mixed_plan_p50["split_two_group"]["group"] == "mixed"

    for mixed_plan in (mixed_plan_p0, mixed_plan_p50, mixed_plan_p100):
        launch_priorities = [worker["launch_priority"] for worker in mixed_plan["workers"]]
        assert launch_priorities == list(range(len(launch_priorities)))

    replay_p50 = tomllib.loads((p50_dir / "replay.toml").read_text(encoding="utf-8"))["replay"]
    assert replay_p50["plan"] == module.path_for_config(mixed_plan_p50_path)
    assert replay_p50["launch_policy_override"]["pattern"]["name"] == "poisson"
    assert replay_p50["launch_policy_override"]["pattern_args"]["rate"] == 0.1

    sbatch_p50 = (p50_dir / "sbatch.sh").read_text(encoding="utf-8")
    assert f"GATEWAY_CONFIG_DEFAULT={(p50_dir / 'gateway-config.toml').resolve()}" in sbatch_p50
    expected_replay_dir_p50 = (
        repo_root
        / "results"
        / "replay"
        / batch_dir.name
        / "model-a"
        / "dataset-b"
        / "agent-c"
        / "sweep-qps-local"
        / "mix"
        / "p50"
    ).resolve()
    expected_log_dir_p50 = (expected_replay_dir_p50 / "sbatch-logs").resolve()
    assert f"#SBATCH --output={expected_log_dir_p50}/slurm.%j.out" in sbatch_p50
    assert f"#SBATCH --error={expected_log_dir_p50}/slurm.%j.err" in sbatch_p50
    assert f'JOB_LOG_DIR=\"{expected_log_dir_p50}\"' in sbatch_p50

    submit_all_path = (batch_dir / "submit_all.sh").resolve()
    assert submit_all_path.exists()
    submit_all_content = submit_all_path.read_text(encoding="utf-8")
    assert 'sbatch "${SCRIPT_DIR}/p0/sbatch.sh"' in submit_all_content
    assert 'sbatch "${SCRIPT_DIR}/p50/sbatch.sh"' in submit_all_content
    assert 'sbatch "${SCRIPT_DIR}/p100/sbatch.sh"' in submit_all_content

    expected_replay_timestamp_dir = (
        repo_root / "results" / "replay" / batch_dir.name
    ).resolve()
    expected_generated_copy_dir = (expected_replay_timestamp_dir / "generated").resolve()
    assert (expected_generated_copy_dir / "p0" / "replay.toml").exists()
    assert (expected_generated_copy_dir / "p50" / "replay.toml").exists()
    assert (expected_generated_copy_dir / "p100" / "replay.toml").exists()
    assert (expected_generated_copy_dir / "p50" / "plan").exists()
    assert (expected_generated_copy_dir / "submit_all.sh").exists()
    assert (expected_generated_copy_dir / "manifest.json").exists()

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["partition"] == "mi3001x"
    assert manifest["model"] == "qwen3_coder_30b"
    assert manifest["lmcache"] == 100
    assert manifest["no_async_scheduling"] is False
    assert manifest["qps"] == 0.1
    assert manifest["top_percent_list"] == [0.0, 50.0, 100.0]
    assert manifest["cache_root_dir"] == str((output_config_dir / "cache").resolve())
    assert Path(manifest["cache_dir"]).is_dir()
    assert len(manifest["cache_key"]) == 64
    assert manifest["cache_hit_count"] == 0
    assert manifest["cache_miss_count"] == 3
    assert manifest["generated_batch_copy_dir"] == module.path_for_config(
        expected_generated_copy_dir
    )
    assert manifest["generation_command_raw"] == shlex.join(
        [
            sys.executable,
            str((Path(module.__file__).resolve())),
            "--source-run-dir",
            str(source_run_dir),
            "--qps",
            "0.1",
            "--top-percent-list",
            "0,50,100",
            "--poisson-seed",
            "7",
            "--randomize-seed",
            "11",
            "--time-constraint-s",
            "600",
            "--partition",
            "mi3001x",
            "--model",
            "qwen3_coder_30b",
            "--lmcache",
            "100",
            "--server-config",
            str(server_config_path),
            "--output-config-dir",
            str(output_config_dir),
        ]
    )
    assert manifest["plan_paths"]["top"] == str(split_top_plan_path)
    assert manifest["plan_paths"]["rest"] == str(split_rest_plan_path)
    assert manifest["port_profile"] == 0
    assert manifest["submit_all_script"] == str(submit_all_path)
    assert manifest["submit_all_command"].startswith("bash ")
    assert len(manifest["generated_experiments"]) == 3

    cache_dir = Path(manifest["cache_dir"]).resolve()
    assert (cache_dir / "mix-cache.meta.json").exists()
    for percent_slug in ("p0", "p50", "p100"):
        assert (cache_dir / f"replay-plan.mix.{percent_slug}.json").exists()
        assert (cache_dir / f"replay-plan.mix.{percent_slug}.meta.json").exists()

    generated_by_percent = {
        item["top_percent_slug"]: item for item in manifest["generated_experiments"]
    }
    assert generated_by_percent["p0"]["top_workers_selected"] == 0
    assert generated_by_percent["p50"]["top_workers_selected"] == 2
    assert generated_by_percent["p100"]["top_workers_selected"] == 4
    assert generated_by_percent["p0"]["top_workers_total"] == 4
    assert generated_by_percent["p50"]["top_workers_total"] == 4
    assert generated_by_percent["p100"]["top_workers_total"] == 4
    for experiment in manifest["generated_experiments"]:
        assert experiment["mixed_plan_path"].endswith(".json")
        assert experiment["cached_mixed_plan_path"].endswith(".json")
        assert experiment["cache_hit"] is False
        assert experiment["gateway_config"].endswith("gateway-config.toml")
        assert experiment["sbatch_log_dir"].endswith("/sbatch-logs")


def test_main_reuses_cached_mix_plans_across_runs(tmp_path: Path) -> None:
    module = load_generator_module()
    repo_root = tmp_path / "repo"
    results_root = repo_root / "results"
    source_run_dir = results_root / "model-a" / "dataset-b" / "agent-c" / "run-1"
    source_run_dir.mkdir(parents=True)

    split_top_plan_path = (source_run_dir / "replay-plan.token.top.json").resolve()
    split_rest_plan_path = (source_run_dir / "replay-plan.token.rest.json").resolve()
    split_top_plan_path.write_text(
        json.dumps(
            build_plan_payload(worker_prefix="top", worker_count=4, split_group="top"),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    split_rest_plan_path.write_text(
        json.dumps(
            build_plan_payload(worker_prefix="rest", worker_count=2, split_group="rest"),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    server_config_path = repo_root / "servers" / "servers-amdhpc" / "server_config.toml"
    server_config_path.parent.mkdir(parents=True)
    server_config_path.write_text("[server]\nport=1\n", encoding="utf-8")

    output_config_dir_run1 = tmp_path / "generated-run1"
    output_config_dir_run2 = tmp_path / "generated-run2"
    shared_cache_dir = tmp_path / "shared-cache"
    rendered_dir = tmp_path / "rendered"
    rendered_dir.mkdir(parents=True)

    module.REPO_ROOT = repo_root
    module.RESULTS_ROOT = results_root

    def fake_build_control_plane(server_config: Path) -> object:
        assert server_config.resolve() == server_config_path.resolve()
        return object()

    def fake_render_local_mode_sbatch(
        *,
        control_plane: object,
        port_profile_id: int,
        partition: str,
        model: str,
        local_mode_script_path: Path,
        check_port_availability: bool,
        lmcache_max_local_cpu_size: str | None,
        no_async_scheduling: bool,
    ) -> Path:
        del control_plane, check_port_availability
        assert port_profile_id == 0
        assert partition == "mi3001x"
        assert model == "qwen3_coder_30b"
        assert lmcache_max_local_cpu_size == "100"
        assert no_async_scheduling is False
        assert local_mode_script_path.exists()
        percent_name = local_mode_script_path.parent.name
        target = rendered_dir / f"{percent_name}.sh"
        target.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "#SBATCH --output=/tmp/original/slurm.%j.out\n"
            "#SBATCH --error=/tmp/original/slurm.%j.err\n"
            "JOB_LOG_DIR=/tmp/original\n"
            "GATEWAY_CONFIG_DEFAULT=/tmp/original-gateway-config.toml\n"
            "echo fake\n",
            encoding="utf-8",
        )
        return target

    module.build_control_plane = fake_build_control_plane
    module.render_local_mode_sbatch = fake_render_local_mode_sbatch

    exit_code_run1 = module.main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--qps",
            "0.1",
            "--top-percent-list",
            "25,75",
            "--poisson-seed",
            "7",
            "--randomize-seed",
            "11",
            "--time-constraint-s",
            "600",
            "--partition",
            "mi3001x",
            "--model",
            "qwen3_coder_30b",
            "--lmcache",
            "100",
            "--server-config",
            str(server_config_path),
            "--cache-dir",
            str(shared_cache_dir),
            "--output-config-dir",
            str(output_config_dir_run1),
        ]
    )
    assert exit_code_run1 == 0
    batch_dir_run1 = resolve_output_batch_dir(output_config_dir_run1)
    manifest_run1 = json.loads((batch_dir_run1 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_run1["cache_root_dir"] == str(shared_cache_dir.resolve())
    assert manifest_run1["cache_hit_count"] == 0
    assert manifest_run1["cache_miss_count"] == 2
    assert len(manifest_run1["generated_experiments"]) == 2
    for experiment in manifest_run1["generated_experiments"]:
        assert experiment["cache_hit"] is False

    exit_code_run2 = module.main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--qps",
            "0.1",
            "--top-percent-list",
            "25,75",
            "--poisson-seed",
            "7",
            "--randomize-seed",
            "11",
            "--time-constraint-s",
            "600",
            "--partition",
            "mi3001x",
            "--model",
            "qwen3_coder_30b",
            "--lmcache",
            "100",
            "--server-config",
            str(server_config_path),
            "--cache-dir",
            str(shared_cache_dir),
            "--output-config-dir",
            str(output_config_dir_run2),
        ]
    )
    assert exit_code_run2 == 0
    batch_dir_run2 = resolve_output_batch_dir(output_config_dir_run2)
    manifest_run2 = json.loads((batch_dir_run2 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_run2["cache_root_dir"] == str(shared_cache_dir.resolve())
    assert manifest_run2["cache_hit_count"] == 2
    assert manifest_run2["cache_miss_count"] == 0
    assert len(manifest_run2["generated_experiments"]) == 2
    for experiment in manifest_run2["generated_experiments"]:
        assert experiment["cache_hit"] is True

    experiments_by_percent_run1 = {
        item["top_percent_slug"]: item for item in manifest_run1["generated_experiments"]
    }
    experiments_by_percent_run2 = {
        item["top_percent_slug"]: item for item in manifest_run2["generated_experiments"]
    }
    assert set(experiments_by_percent_run1) == {"p25", "p75"}
    assert set(experiments_by_percent_run2) == {"p25", "p75"}
    for percent_slug in ("p25", "p75"):
        assert (
            experiments_by_percent_run1[percent_slug]["cached_mixed_plan_path"]
            == experiments_by_percent_run2[percent_slug]["cached_mixed_plan_path"]
        )
