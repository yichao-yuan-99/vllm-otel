from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _write_gateway_job(
    run_dir: Path,
    *,
    profile_id: int | None,
    gateway_run_id: str,
    requests: list[tuple[int, int]],
    status_codes: list[int] | None = None,
    run_start_time: str | None = None,
    run_end_time: str | None = None,
    write_lifecycle: bool = True,
) -> None:
    if profile_id is None:
        gateway_run_dir = run_dir / "gateway-output" / gateway_run_id
    else:
        gateway_run_dir = run_dir / "gateway-output" / f"profile-{profile_id}" / gateway_run_id

    records: list[dict[str, object]] = []
    for index, (prompt_tokens, completion_tokens) in enumerate(requests):
        status_code = 200 if status_codes is None else status_codes[index]
        records.append(
            {
                "status_code": status_code,
                "response": {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
                }
            }
        )
    _write_jsonl(gateway_run_dir / "requests" / "model_inference.jsonl", records)

    if run_start_time is not None and run_end_time is not None:
        (gateway_run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_start_time": run_start_time,
                    "run_end_time": run_end_time,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        if write_lifecycle:
            _write_jsonl(
                gateway_run_dir / "events" / "lifecycle.jsonl",
                [
                    {"event_type": "agent_start", "timestamp": run_start_time},
                    {"event_type": "agent_end", "timestamp": run_end_time},
                ],
            )


def test_extract_run_selects_expected_context_percentile_trails(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_gateway_job(
        run_dir,
        profile_id=None,
        gateway_run_id="run_low",
        requests=[(4, 1), (5, 1)],
        run_start_time="2026-03-01T00:00:00Z",
        run_end_time="2026-03-01T00:01:00Z",
    )
    _write_gateway_job(
        run_dir,
        profile_id=2,
        gateway_run_id="run_mid",
        requests=[(10, 5)],
        run_start_time="2026-03-01T00:10:00Z",
        run_end_time="2026-03-01T00:13:00Z",
    )
    _write_gateway_job(
        run_dir,
        profile_id=0,
        gateway_run_id="run_high",
        requests=[(20, 10)],
        run_start_time="2026-03-01T00:20:00Z",
        run_end_time="2026-03-01T00:25:00Z",
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_path = (
        run_dir
        / "original-analysis"
        / "percentiles"
        / extract_run.DEFAULT_OUTPUT_NAME
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["metric"] == "context_usage"
    assert payload["selection_method"] == "nearest_rank_on_ascending_context_length"
    assert payload["trail_count_total"] == 3
    assert payload["trail_count_ranked"] == 3
    assert payload["trail_count_unranked"] == 0
    assert payload["trail_count_with_status_499"] == 0
    assert payload["unranked_mode"] == "strict_499"
    assert payload["percentiles"] == list(range(5, 100, 5))

    ranked_trail_names = [item["source_trail_name"] for item in payload["ranked_trails"]]
    assert ranked_trail_names == [
        "run_low",
        "profile-2/run_mid",
        "profile-0/run_high",
    ]
    assert [item["context_length"] for item in payload["ranked_trails"]] == [6, 15, 30]
    assert [item["trail_duration_s"] for item in payload["ranked_trails"]] == [60.0, 180.0, 300.0]
    assert [item["context_rank_ascending"] for item in payload["ranked_trails"]] == [1, 2, 3]
    assert [item["context_rank_descending"] for item in payload["ranked_trails"]] == [3, 2, 1]

    selected_names = payload["source_trail_names_by_percentile"]
    assert selected_names["5"] == "run_low"
    assert selected_names["30"] == "run_low"
    assert selected_names["35"] == "profile-2/run_mid"
    assert selected_names["65"] == "profile-2/run_mid"
    assert selected_names["70"] == "profile-0/run_high"
    assert selected_names["95"] == "profile-0/run_high"
    assert len(payload["selected_trails"]) == 19
    assert payload["selected_trails"][0]["trail_duration_s"] == 60.0
    assert payload["selected_trails"][-1]["trail_duration_s"] == 300.0


def test_extract_run_keeps_missing_context_trails_unranked(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_gateway_job(
        run_dir,
        profile_id=None,
        gateway_run_id="run_ranked",
        requests=[(2, 3)],
        run_start_time="2026-03-01T00:00:00Z",
        run_end_time="2026-03-01T00:00:05Z",
    )
    _write_jsonl(
        run_dir / "gateway-output" / "run_unranked" / "requests" / "model_inference.jsonl",
        [{"response": {"usage": {}}}],
    )

    payload = extract_run.extract_percentiles_from_run_dir(run_dir)

    assert payload["trail_count_total"] == 2
    assert payload["trail_count_ranked"] == 1
    assert payload["trail_count_unranked"] == 1
    assert payload["trail_count_with_status_499"] == 0
    assert payload["ranked_trails"][0]["source_trail_name"] == "run_ranked"
    assert payload["ranked_trails"][0]["trail_duration_s"] == 5.0
    assert payload["unranked_trails"][0]["source_trail_name"] == "run_unranked"
    assert payload["unranked_trails"][0]["trail_duration_s"] is None
    assert set(payload["source_trail_names_by_percentile"].values()) == {"run_ranked"}


def test_extract_run_keeps_499_trails_unranked(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_gateway_job(
        run_dir,
        profile_id=None,
        gateway_run_id="run_ranked",
        requests=[(2, 3)],
        run_start_time="2026-03-01T00:00:00Z",
        run_end_time="2026-03-01T00:00:05Z",
    )
    _write_gateway_job(
        run_dir,
        profile_id=4,
        gateway_run_id="run_499",
        requests=[(100, 50)],
        status_codes=[499],
        run_start_time="2026-03-01T00:01:00Z",
        run_end_time="2026-03-01T00:01:30Z",
    )

    payload = extract_run.extract_percentiles_from_run_dir(run_dir)

    assert payload["trail_count_total"] == 2
    assert payload["trail_count_ranked"] == 1
    assert payload["trail_count_unranked"] == 1
    assert payload["trail_count_with_status_499"] == 1
    assert payload["ranked_trails"][0]["source_trail_name"] == "run_ranked"
    assert payload["unranked_trails"][0]["source_trail_name"] == "profile-4/run_499"
    assert payload["unranked_trails"][0]["trail_duration_s"] == 30.0
    assert payload["unranked_trails"][0]["has_status_499"] is True
    assert payload["unranked_trails"][0]["requests_with_status_499"] == 1
    assert set(payload["source_trail_names_by_percentile"].values()) == {"run_ranked"}


def test_extract_run_uses_lifecycle_duration_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_gateway_job(
        run_dir,
        profile_id=None,
        gateway_run_id="run_lifecycle_only",
        requests=[(2, 3)],
        run_start_time="2026-03-01T00:00:00Z",
        run_end_time="2026-03-01T00:00:07Z",
    )
    (run_dir / "gateway-output" / "run_lifecycle_only" / "manifest.json").unlink()

    payload = extract_run.extract_percentiles_from_run_dir(run_dir)

    assert payload["ranked_trails"][0]["source_trail_name"] == "run_lifecycle_only"
    assert payload["ranked_trails"][0]["trail_duration_s"] == 7.0


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "nested" / "b"
    (run_a / "gateway-output").mkdir(parents=True)
    (run_b / "gateway-output").mkdir(parents=True)

    processed: list[tuple[Path, Path | None]] = []

    def fake_extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
        processed.append((run_dir, output_path))
        return run_dir / "original-analysis" / "percentiles" / extract_run.DEFAULT_OUTPUT_NAME

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None),
        (run_b.resolve(), None),
    ]
