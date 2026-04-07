from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import generate_all_figures


def _write_agent_output_summary(run_dir: Path, *, include_replay_status: bool = True) -> Path:
    processed_dir = run_dir / "post-processed" / "agent-output-throughput"
    processed_dir.mkdir(parents=True)
    input_path = processed_dir / "agent-output-throughput.json"
    agents = [
        {
            "gateway_run_id": "run-a",
            "gateway_profile_id": 0,
            "api_token_hash": "hash-a",
            "request_count": 1,
            "requests_with_output_tokens": 1,
            "requests_with_llm_request_duration": 1,
            "requests_with_output_tokens_and_llm_request_duration": 1,
            "output_tokens": 20,
            "completion_tokens": 20,
            "llm_request_duration_s": 1.0,
            "output_throughput_tokens_per_s": 20.0,
        },
        {
            "gateway_run_id": "run-b",
            "gateway_profile_id": 1,
            "api_token_hash": "hash-b",
            "request_count": 2,
            "requests_with_output_tokens": 2,
            "requests_with_llm_request_duration": 2,
            "requests_with_output_tokens_and_llm_request_duration": 2,
            "output_tokens": 15,
            "completion_tokens": 15,
            "llm_request_duration_s": 2.0,
            "output_throughput_tokens_per_s": 7.5,
        },
        {
            "gateway_run_id": "run-c",
            "api_token_hash": "hash-c",
            "gateway_profile_id": 1,
            "request_count": 1,
            "requests_with_output_tokens": 1,
            "requests_with_llm_request_duration": 1,
            "requests_with_output_tokens_and_llm_request_duration": 1,
            "output_tokens": 5,
            "completion_tokens": 5,
            "llm_request_duration_s": 1.0,
            "output_throughput_tokens_per_s": 5.0,
        },
    ]
    if include_replay_status:
        agents[0]["replay_worker_status"] = "completed"
        agents[0]["replay_completed"] = True
        agents[1]["replay_worker_status"] = "completed"
        agents[1]["replay_completed"] = True
        agents[2]["replay_worker_status"] = "timed_out"
        agents[2]["replay_completed"] = False

    base_payload = {
        "source_run_dir": str(run_dir),
        "source_gateway_output_dir": str(run_dir / "gateway-output"),
        "service_failure_detected": False,
        "service_failure_cutoff_time_utc": None,
        "agent_output_throughput_tokens_per_s_histogram": {"bin_size": 1.0},
    }
    payload = generate_all_figures._build_payload_for_agents(  # type: ignore[attr-defined]
        base_payload,
        agents,
        figure_variant_label="fixture",
    )
    payload.pop("_figure_variant_label", None)
    payload["multi_profile"] = True
    payload["port_profile_ids"] = [0, 1]
    payload["series_keys"] = ["profile-0", "profile-1"]

    series_by_profile: dict[str, dict[str, object]] = {}
    for profile_id in payload["port_profile_ids"]:
        profile_agents = [
            agent
            for agent in agents
            if agent.get("gateway_profile_id") == profile_id
        ]
        profile_payload = generate_all_figures._build_payload_for_agents(  # type: ignore[attr-defined]
            base_payload,
            profile_agents,
            figure_variant_label=f"profile-{profile_id}",
        )
        profile_payload.pop("_figure_variant_label", None)
        profile_payload["gateway_profile_id"] = profile_id
        series_by_profile[f"profile-{profile_id}"] = profile_payload
    payload["series_by_profile"] = series_by_profile

    input_path.write_text(json.dumps(payload), encoding="utf-8")
    return input_path


def test_discover_run_dirs_with_agent_output_throughput_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    bad_run = root_dir / "b" / "job-missing-input"

    _write_agent_output_summary(good_run)
    bad_processed_dir = bad_run / "post-processed" / "agent-output-throughput"
    bad_processed_dir.mkdir(parents=True)

    discovered = generate_all_figures.discover_run_dirs_with_agent_output_throughput(root_dir)

    assert discovered == [good_run.resolve()]


def test_generate_figures_for_run_dir_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "job"
    _write_agent_output_summary(run_dir)
    render_calls: list[tuple[str, str | None, int]] = []

    def fake_render_histogram_figure(
        *,
        agent_output_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del image_format, dpi
        render_calls.append(
            (
                "histogram",
                agent_output_payload.get("_figure_variant_label"),
                int(agent_output_payload.get("agent_count") or 0),
            )
        )
        output_path.write_text("fake-histogram", encoding="utf-8")
        return True

    def fake_render_scatter_figure(
        *,
        agent_output_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del image_format, dpi
        render_calls.append(
            (
                "scatter",
                agent_output_payload.get("_figure_variant_label"),
                int(agent_output_payload.get("agent_count") or 0),
            )
        )
        output_path.write_text("fake-scatter", encoding="utf-8")
        return True

    monkeypatch.setattr(
        generate_all_figures,
        "_render_histogram_figure",
        fake_render_histogram_figure,
    )
    monkeypatch.setattr(
        generate_all_figures,
        "_render_scatter_figure",
        fake_render_scatter_figure,
    )

    manifest_path = generate_all_figures.generate_figures_for_run_dir(
        run_dir,
        image_format="png",
        dpi=150,
    )

    assert manifest_path == (
        run_dir
        / "post-processed"
        / "visualization"
        / "agent-output-throughput"
        / "figures-manifest.json"
    ).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["multi_profile"] is True
    assert manifest["port_profile_ids"] == [0, 1]
    assert manifest["series_count"] == 3
    assert manifest["figure_count"] == 12
    assert manifest["requested_figure_count"] == 12
    assert manifest["variant_count"] == 2
    assert manifest["skipped_variant_count"] == 0
    assert manifest["figure_generated"] is True
    assert manifest["figure_file_name"] == "agent-output-throughput-histogram.png"
    assert manifest["image_format"] == "png"
    assert manifest["dpi"] == 150
    assert manifest["agent_count"] == 3
    assert manifest["run_output_throughput_tokens_per_s"] == 10.0
    assert Path(manifest["figure_path"]).is_file()
    assert render_calls == [
        ("histogram", "all agents", 3),
        ("scatter", "all agents", 3),
        ("histogram", "completed replay only", 2),
        ("scatter", "completed replay only", 2),
        ("histogram", "all agents | profile-0", 1),
        ("scatter", "all agents | profile-0", 1),
        ("histogram", "completed replay only | profile-0", 1),
        ("scatter", "completed replay only | profile-0", 1),
        ("histogram", "all agents | profile-1", 2),
        ("scatter", "all agents | profile-1", 2),
        ("histogram", "completed replay only | profile-1", 1),
        ("scatter", "completed replay only | profile-1", 1),
    ]
    assert [figure["series_id"] for figure in manifest["figures"]] == [
        "aggregate",
        "aggregate",
        "aggregate",
        "aggregate",
        "profile-0",
        "profile-0",
        "profile-0",
        "profile-0",
        "profile-1",
        "profile-1",
        "profile-1",
        "profile-1",
    ]
    assert [figure["gateway_profile_id"] for figure in manifest["figures"]] == [
        None,
        None,
        None,
        None,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]
    assert [figure["relative_output_subdir"] for figure in manifest["figures"]] == [
        "",
        "",
        "",
        "",
        "profile-0",
        "profile-0",
        "profile-0",
        "profile-0",
        "profile-1",
        "profile-1",
        "profile-1",
        "profile-1",
    ]
    assert [figure["variant_id"] for figure in manifest["figures"]] == [
        "all-agents",
        "all-agents",
        "completed-replay-only",
        "completed-replay-only",
        "all-agents",
        "all-agents",
        "completed-replay-only",
        "completed-replay-only",
        "all-agents",
        "all-agents",
        "completed-replay-only",
        "completed-replay-only",
    ]
    for figure in manifest["figures"]:
        assert Path(figure["figure_path"]).is_file()


def test_generate_figures_for_run_dir_skips_completed_replay_variant_without_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "job"
    _write_agent_output_summary(run_dir, include_replay_status=False)

    def fake_render_histogram_figure(
        *,
        agent_output_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del agent_output_payload, image_format, dpi
        output_path.write_text("fake-histogram", encoding="utf-8")
        return True

    def fake_render_scatter_figure(
        *,
        agent_output_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del agent_output_payload, image_format, dpi
        output_path.write_text("fake-scatter", encoding="utf-8")
        return True

    monkeypatch.setattr(
        generate_all_figures,
        "_render_histogram_figure",
        fake_render_histogram_figure,
    )
    monkeypatch.setattr(
        generate_all_figures,
        "_render_scatter_figure",
        fake_render_scatter_figure,
    )

    manifest_path = generate_all_figures.generate_figures_for_run_dir(run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["series_count"] == 3
    assert manifest["figure_count"] == 6
    assert manifest["requested_figure_count"] == 6
    assert manifest["variant_count"] == 1
    assert manifest["skipped_variant_count"] == 0
    assert [figure["variant_id"] for figure in manifest["figures"]] == [
        "all-agents",
        "all-agents",
        "all-agents",
        "all-agents",
        "all-agents",
        "all-agents",
    ]


def test_generate_figures_for_run_dir_rejects_missing_input_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        generate_all_figures.generate_figures_for_run_dir(run_dir)
    except ValueError as exc:
        assert "Missing agent-output-throughput file" in str(exc)
    else:
        raise AssertionError("Expected ValueError when input file is missing")


def test_main_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_agent_output_summary(run_a)
    _write_agent_output_summary(run_b)

    processed: list[tuple[Path, str, int]] = []

    def fake_generate_figures_for_run_dir(
        run_dir: Path,
        *,
        agent_output_input_path: Path | None = None,
        output_dir: Path | None = None,
        image_format: str = "png",
        dpi: int = 220,
    ) -> Path:
        del agent_output_input_path, output_dir
        processed.append((run_dir, image_format, dpi))
        return (
            run_dir
            / "post-processed"
            / "visualization"
            / "agent-output-throughput"
            / "figures-manifest.json"
        )

    monkeypatch.setattr(
        generate_all_figures,
        "generate_figures_for_run_dir",
        fake_generate_figures_for_run_dir,
    )

    exit_code = generate_all_figures.main(
        ["--root-dir", str(root_dir), "--max-procs", "1", "--format", "svg", "--dpi", "144"]
    )

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), "svg", 144),
        (run_b.resolve(), "svg", 144),
    ]


def test_main_rejects_dry_run_for_single_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        generate_all_figures.main(["--run-dir", str(run_dir), "--dry-run"])
    except ValueError as exc:
        assert "--dry-run can only be used with --root-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --dry-run is used with --run-dir")


def test_main_rejects_output_dir_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        generate_all_figures.main(
            ["--root-dir", str(root_dir), "--output-dir", str(tmp_path / "figs")]
        )
    except ValueError as exc:
        assert "--output-dir can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output-dir is used with --root-dir")
