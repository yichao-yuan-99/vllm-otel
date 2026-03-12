from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import aggregate_runs_csv


def _write_split_summary(
    run_dir: Path,
    *,
    token_series: list[float | None],
    context_series: list[float | None],
    include_table: bool = True,
) -> None:
    output_dir = run_dir / "original-analysis" / "split"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {}
    if include_table:
        payload["table_2x99"] = {
            "top_p_token_usage_ratio": token_series,
            "top_p_context_usage_ratio": context_series,
        }
    else:
        payload["by_percentile"] = [
            {
                "p": idx + 1,
                "top_token_usage_ratio": token_series[idx],
                "top_context_usage_ratio": context_series[idx],
            }
            for idx in range(min(len(token_series), len(context_series)))
        ]

    (output_dir / "top-p-usage-ratio-summary.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_discover_split_summary_paths(tmp_path: Path) -> None:
    good = (
        tmp_path
        / "a"
        / "original-analysis"
        / "split"
        / "top-p-usage-ratio-summary.json"
    )
    bad = tmp_path / "b" / "other" / "top-p-usage-ratio-summary.json"
    good.parent.mkdir(parents=True)
    bad.parent.mkdir(parents=True)
    good.write_text("{}", encoding="utf-8")
    bad.write_text("{}", encoding="utf-8")

    discovered = aggregate_runs_csv.discover_split_summary_paths(tmp_path)
    assert discovered == [good.resolve()]


def test_build_rows_sorts_runs_and_infers_percentiles(tmp_path: Path) -> None:
    run_b = tmp_path / "x" / "10"
    run_a = tmp_path / "x" / "2"
    _write_split_summary(
        run_b,
        token_series=[0.3, 0.6],
        context_series=[0.4, 0.7],
    )
    _write_split_summary(
        run_a,
        token_series=[0.1, 0.5],
        context_series=[0.2, 0.8],
    )

    token_rows, context_rows, percentiles = aggregate_runs_csv.build_rows(tmp_path)

    assert percentiles == [1, 2]
    assert [row["run_path"] for row in token_rows] == ["x/2", "x/10"]
    assert [row["run_path"] for row in context_rows] == ["x/2", "x/10"]
    assert token_rows[0]["p1"] == 0.1
    assert token_rows[0]["p2"] == 0.5
    assert context_rows[1]["p1"] == 0.4
    assert context_rows[1]["p2"] == 0.7


def test_build_rows_falls_back_to_by_percentile(tmp_path: Path) -> None:
    run_dir = tmp_path / "demo"
    _write_split_summary(
        run_dir,
        token_series=[0.2, 0.9],
        context_series=[0.3, 1.0],
        include_table=False,
    )

    token_rows, context_rows, percentiles = aggregate_runs_csv.build_rows(tmp_path)

    assert percentiles == [1, 2]
    assert len(token_rows) == 1
    assert len(context_rows) == 1
    assert token_rows[0]["p1"] == 0.2
    assert token_rows[0]["p2"] == 0.9
    assert context_rows[0]["p1"] == 0.3
    assert context_rows[0]["p2"] == 1.0


def test_main_skip_extract_writes_csv_and_manifest(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    _write_split_summary(
        root_dir / "demo",
        token_series=[0.25, 0.75],
        context_series=[0.4, 1.0],
    )

    output_dir = root_dir / "custom-output"
    exit_code = aggregate_runs_csv.main(
        [
            "--root-dir",
            str(root_dir),
            "--skip-extract",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    token_csv = output_dir / "top_p_token_usage_ratio.csv"
    context_csv = output_dir / "top_p_context_usage_ratio.csv"
    manifest_path = output_dir / "split-top-p-ratio-manifest.json"
    assert token_csv.is_file()
    assert context_csv.is_file()
    assert manifest_path.is_file()

    token_rows = _read_csv_rows(token_csv)
    context_rows = _read_csv_rows(context_csv)
    assert len(token_rows) == 1
    assert len(context_rows) == 1
    assert token_rows[0]["run_path"] == "demo"
    assert token_rows[0]["p1"] == "0.25"
    assert token_rows[0]["p2"] == "0.75"
    assert context_rows[0]["p1"] == "0.4"
    assert context_rows[0]["p2"] == "1.0"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["percentiles"] == [1, 2]
    assert "top_p_token_usage_ratio" in manifest["tables"]
    assert "top_p_context_usage_ratio" in manifest["tables"]


def test_main_dry_run_lists_discovered_runs(tmp_path: Path, capsys) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    (run_a / "gateway-output").mkdir(parents=True)

    exit_code = aggregate_runs_csv.main(["--root-dir", str(root_dir), "--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert str(run_a.resolve()) in captured.out
    assert not (root_dir / "split-top-p-ratio-tables").exists()


def test_build_rows_backfills_top_rest_ratio_for_legacy_summaries(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy"
    output_dir = run_dir / "original-analysis" / "split"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "top-p-usage-ratio-summary.json").write_text(
        json.dumps(
            {
                # Legacy shape: no ratio_definition and top_*_ratio was top/total.
                "table_2x99": {
                    "top_p_token_usage_ratio": [0.5, 0.8],
                    "top_p_context_usage_ratio": [0.4, 0.7],
                },
                "by_percentile": [
                    {
                        "p": 1,
                        "top_token_usage_total": 10,
                        "rest_token_usage_total": 5,
                        "top_context_length_total": 12,
                        "rest_context_length_total": 6,
                        "top_token_usage_ratio": 10 / 15,
                        "top_context_usage_ratio": 12 / 18,
                    },
                    {
                        "p": 2,
                        "top_token_usage_total": 18,
                        "rest_token_usage_total": 2,
                        "top_context_length_total": 14,
                        "rest_context_length_total": 1,
                        "top_token_usage_ratio": 18 / 20,
                        "top_context_usage_ratio": 14 / 15,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    token_rows, context_rows, percentiles = aggregate_runs_csv.build_rows(tmp_path)
    assert percentiles == [1, 2]
    assert token_rows[0]["p1"] == 2.0
    assert token_rows[0]["p2"] == 9.0
    assert context_rows[0]["p1"] == 2.0
    assert context_rows[0]["p2"] == 14.0
