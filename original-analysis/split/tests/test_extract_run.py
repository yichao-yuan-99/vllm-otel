from __future__ import annotations

import json
import math
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
    requests: list[tuple[int, int, int]],
) -> None:
    if profile_id is None:
        gateway_run_dir = run_dir / "gateway-output" / gateway_run_id
    else:
        gateway_run_dir = (
            run_dir
            / "gateway-output"
            / f"profile-{profile_id}"
            / gateway_run_id
        )

    records: list[dict[str, object]] = []
    for prompt_tokens, completion_tokens, cached_tokens in requests:
        records.append(
            {
                "response": {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "prompt_tokens_details": {
                            "cached_tokens": cached_tokens,
                        },
                    }
                }
            }
        )
    _write_jsonl(gateway_run_dir / "requests" / "model_inference.jsonl", records)


def test_extract_run_generates_expected_top_p_series(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_gateway_job(
        run_dir,
        profile_id=0,
        gateway_run_id="run_high",
        requests=[(20, 10, 5), (8, 2, 1)],
    )
    _write_gateway_job(
        run_dir,
        profile_id=1,
        gateway_run_id="run_mid",
        requests=[(10, 5, 2)],
    )
    _write_gateway_job(
        run_dir,
        profile_id=2,
        gateway_run_id="run_low",
        requests=[(4, 1, 0), (5, 1, 0)],
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_path = (
        run_dir
        / "original-analysis"
        / "split"
        / "top-p-usage-ratio-summary.json"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["trail_count_total"] == 3
    assert payload["trail_count_ranked"] == 3
    assert payload["trail_count_unranked"] == 0

    ranked_ids = [trail["gateway_run_id"] for trail in payload["ranked_trails"]]
    assert ranked_ids == ["run_high", "run_mid", "run_low"]

    token_series = payload["table_2x99"]["top_p_token_usage_ratio"]
    context_series = payload["table_2x99"]["top_p_context_usage_ratio"]
    token_share_series = payload["table_2x99"]["top_p_token_usage_share"]
    context_share_series = payload["table_2x99"]["top_p_context_usage_share"]
    assert len(token_series) == 99
    assert len(context_series) == 99
    assert len(token_share_series) == 99
    assert len(context_share_series) == 99

    # Totals:
    # run_high token=34, context=30
    # run_mid token=13, context=15
    # run_low token=11, context=6
    assert math.isclose(token_series[0], 34 / 24)      # p=1 -> top1/rest2
    assert math.isclose(context_series[0], 30 / 21)
    assert math.isclose(token_series[33], 47 / 11)     # p=34 -> top2/rest1
    assert math.isclose(context_series[33], 45 / 6)
    assert token_series[98] is None                    # p=99 -> rest is empty
    assert context_series[98] is None

    # Shares are still available as top/total.
    assert math.isclose(token_share_series[0], 34 / 58)
    assert math.isclose(context_share_series[0], 30 / 51)
    assert math.isclose(token_share_series[33], 47 / 58)
    assert math.isclose(context_share_series[33], 45 / 51)
    assert math.isclose(token_share_series[98], 1.0)
    assert math.isclose(context_share_series[98], 1.0)

    token_prefix = [value for value in token_series if value is not None]
    context_prefix = [value for value in context_series if value is not None]
    assert all(
        token_prefix[idx] <= token_prefix[idx + 1]
        for idx in range(len(token_prefix) - 1)
    )
    assert all(
        context_prefix[idx] <= context_prefix[idx + 1]
        for idx in range(len(context_prefix) - 1)
    )

    rows = payload["by_percentile"]
    assert rows[0]["p"] == 1
    assert rows[0]["top_trail_count"] == 1
    assert rows[33]["p"] == 34
    assert rows[33]["top_trail_count"] == 2
    assert rows[98]["p"] == 99
    assert rows[98]["top_trail_count"] == 3
    assert rows[98]["rest_trail_count"] == 0
    assert rows[98]["top_token_usage_ratio"] is None
    assert rows[98]["top_context_usage_ratio"] is None


def test_extract_run_keeps_missing_context_trails_unranked(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_gateway_job(
        run_dir,
        profile_id=None,
        gateway_run_id="run_ranked",
        requests=[(2, 3, 0)],
    )
    _write_jsonl(
        run_dir
        / "gateway-output"
        / "run_unranked"
        / "requests"
        / "model_inference.jsonl",
        [{"response": {"usage": {}}}],
    )

    payload = extract_run.extract_split_from_run_dir(run_dir)
    assert payload["trail_count_total"] == 2
    assert payload["trail_count_ranked"] == 1
    assert payload["trail_count_unranked"] == 1
    assert payload["ranked_trails"][0]["gateway_run_id"] == "run_ranked"
    assert payload["unranked_trails"][0]["gateway_run_id"] == "run_unranked"


def test_discover_run_dirs_with_gateway_output_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "nested" / "b"
    (run_a / "gateway-output").mkdir(parents=True)
    (run_b / "gateway-output").mkdir(parents=True)

    discovered = extract_run.discover_run_dirs_with_gateway_output(root_dir)
    assert discovered == [run_a.resolve(), run_b.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    (run_a / "gateway-output").mkdir(parents=True)
    (run_b / "gateway-output").mkdir(parents=True)

    processed: list[tuple[Path, Path | None]] = []

    def fake_extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
        processed.append((run_dir, output_path))
        return run_dir / "original-analysis" / "split" / "top-p-usage-ratio-summary.json"

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None),
        (run_b.resolve(), None),
    ]


def test_extract_run_rejects_dry_run_for_single_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        extract_run.main(["--run-dir", str(run_dir), "--dry-run"])
    except ValueError as exc:
        assert "--dry-run can only be used with --root-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --dry-run is used with --run-dir")


def test_extract_run_rejects_output_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        extract_run.main(
            ["--root-dir", str(root_dir), "--output", str(tmp_path / "out.json")]
        )
    except ValueError as exc:
        assert "--output can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output is used with --root-dir")
