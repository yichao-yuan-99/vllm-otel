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


def _write_split_duration_summary(
    run_dir: Path,
    *,
    bin_labels: list[str],
    values_by_metric: dict[str, list[float | None]],
) -> None:
    output_dir = run_dir / "post-processed" / "split" / "duration"
    output_dir.mkdir(parents=True)

    tables: dict[str, dict[str, dict[str, int | float | None]]] = {}
    for metric_name, values in values_by_metric.items():
        metric_table: dict[str, dict[str, int | float | None]] = {}
        for label, value in zip(bin_labels, values):
            metric_table[label] = {
                "avg": value,
                "count": 1 if value is not None else 0,
            }
        tables[metric_name] = metric_table

    (output_dir / "duration-split-summary.json").write_text(
        json.dumps(
            {
                "bin_labels": bin_labels,
                "tables": tables,
            }
        ),
        encoding="utf-8",
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_tables_sorts_rows_by_run_path(tmp_path: Path) -> None:
    run_b = tmp_path / "x" / "10"
    run_a = tmp_path / "x" / "2"
    bin_labels = ["0-50%", "50-100%"]

    _write_split_duration_summary(
        run_b,
        bin_labels=bin_labels,
        values_by_metric={
            "duration_s": [10.0, 20.0],
            "turn_count": [3.0, 4.0],
            "prompt_tokens": [30.0, 40.0],
            "decode_tokens": [5.0, 6.0],
            "cached_prompt_tokens": [2.0, 3.0],
        },
    )
    _write_split_duration_summary(
        run_a,
        bin_labels=bin_labels,
        values_by_metric={
            "duration_s": [1.0, 2.0],
            "turn_count": [1.0, 2.0],
            "prompt_tokens": [10.0, 20.0],
            "decode_tokens": [3.0, 4.0],
            "cached_prompt_tokens": [0.0, 1.0],
        },
    )

    rows_by_metric, ordered_bins = aggregate_runs_csv.build_tables(tmp_path)

    assert ordered_bins == ["0-50%", "50-100%"]
    assert [row["run_path"] for row in rows_by_metric["duration_s"]] == ["x/2", "x/10"]


def test_main_skip_extract_writes_all_metric_tables(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_dir = root_dir / "demo"
    _write_split_duration_summary(
        run_dir,
        bin_labels=["0-50%", "50-100%"],
        values_by_metric={
            "duration_s": [12.0, 18.0],
            "turn_count": [2.0, 4.0],
            "prompt_tokens": [100.0, 200.0],
            "decode_tokens": [30.0, 40.0],
            "cached_prompt_tokens": [5.0, 6.0],
        },
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
    manifest_path = output_dir / "split-duration-tables-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bin_labels"] == ["0-50%", "50-100%"]

    for metric_name in (
        "duration_s",
        "turn_count",
        "prompt_tokens",
        "decode_tokens",
        "cached_prompt_tokens",
    ):
        csv_path = output_dir / f"{metric_name}.csv"
        assert csv_path.is_file()
        rows = _read_csv_rows(csv_path)
        assert len(rows) == 1
        assert rows[0]["run_path"] == "demo"
        assert rows[0]["0-50%"] != ""
        assert rows[0]["50-100%"] != ""


def test_main_dry_run_lists_discovered_runs(tmp_path: Path, capsys) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    (run_a / "gateway-output").mkdir(parents=True)

    exit_code = aggregate_runs_csv.main(["--root-dir", str(root_dir), "--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert str(run_a.resolve()) in captured.out
    assert not (root_dir / "split-duration-tables").exists()
