from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_trial_result(run_dir: Path, trial_id: str, reward: float) -> None:
    _write_json(
        run_dir / "trials" / trial_id / "result.json",
        {
            "trial_id": trial_id,
            "verifier_result": {
                "rewards": {
                    "reward": reward,
                }
            },
        },
    )


def _write_results_json(run_dir: Path, entries: list[tuple[str, str]]) -> None:
    payload = []
    for trial_id, api_token in entries:
        payload.append(
            {
                "trial_id": trial_id,
                "command": [
                    "python3",
                    "-m",
                    "con_driver.gateway_wrapper",
                    "--api-token",
                    api_token,
                ],
            }
        )
    _write_json(run_dir / "meta" / "results.json", payload)


def _write_trail_manifest(run_dir: Path, trail_name: str, api_token_hash: str) -> None:
    trail_dir = run_dir / "gateway-output" / trail_name
    _write_json(
        trail_dir / "manifest.json",
        {
            "api_token_hash": api_token_hash,
        },
    )


def _write_two_group_file(
    run_dir: Path,
    file_name: str,
    *,
    top_names: list[str],
    rest_names: list[str],
    selected_p: int,
) -> None:
    _write_json(
        run_dir / "original-analysis" / "split" / file_name,
        {
            "selection": {
                "selected_p": selected_p,
            },
            "trail_count_ranked": len(top_names) + len(rest_names),
            "group_top": {
                "trail_names": top_names,
            },
            "group_rest": {
                "trail_names": rest_names,
            },
        },
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_demo_run(run_dir: Path) -> None:
    # Trials and their verifier rewards.
    trials = [
        ("trial-0000-a", "token-a", 1.0),
        ("trial-0001-b", "token-b", 0.5),
        ("trial-0002-c", "token-c", 0.0),
    ]

    _write_results_json(run_dir, [(trial_id, token) for trial_id, token, _ in trials])
    for trial_id, token, reward in trials:
        _write_trial_result(run_dir, trial_id, reward)
        # Use the same trail naming scheme produced by split/extract_run.
        trail_name = f"profile-0/run_{trial_id}"
        _write_trail_manifest(run_dir, trail_name, _sha256_hex(token))

    _write_two_group_file(
        run_dir,
        extract_run.NORMAL_TOKEN_TWO_GROUP_NAME,
        top_names=["profile-0/run_trial-0000-a", "profile-0/run_trial-0001-b"],
        rest_names=["profile-0/run_trial-0002-c"],
        selected_p=34,
    )
    _write_two_group_file(
        run_dir,
        extract_run.NORMAL_CONTEXT_TWO_GROUP_NAME,
        top_names=["profile-0/run_trial-0001-b"],
        rest_names=["profile-0/run_trial-0000-a", "profile-0/run_trial-0002-c"],
        selected_p=2,
    )
    _write_two_group_file(
        run_dir,
        extract_run.STRICT_TOKEN_TWO_GROUP_NAME,
        top_names=["profile-0/run_trial-0001-b"],
        rest_names=["profile-0/run_trial-0002-c"],
        selected_p=27,
    )
    _write_two_group_file(
        run_dir,
        extract_run.STRICT_CONTEXT_TWO_GROUP_NAME,
        top_names=["profile-0/run_trial-0000-a"],
        rest_names=["profile-0/run_trial-0002-c"],
        selected_p=11,
    )


def test_extract_run_scores_groups(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _build_demo_run(run_dir)

    output_path = extract_run.extract_run_dir(run_dir)
    assert output_path.is_file()

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["run_trial_count"] == 3
    assert payload["run_trial_with_score_count"] == 3
    assert payload["run_score"] == 1.5
    assert payload["run_score_rate"] == 0.5

    token_normal = payload["split_scores"]["token_usage"]["normal"]
    assert token_normal["selection"]["selected_p"] == 34
    assert token_normal["group_top"]["trail_count"] == 2
    assert token_normal["group_top"]["score"] == 1.5
    assert token_normal["group_top"]["score_rate"] == 0.75
    assert token_normal["group_rest"]["trail_count"] == 1
    assert token_normal["group_rest"]["score"] == 0.0
    assert token_normal["group_rest"]["score_rate"] == 0.0
    assert token_normal["score_gap"] == 1.5
    assert token_normal["score_rate_gap"] == 0.75

    token_strict = payload["split_scores"]["token_usage"]["strict_499"]
    assert token_strict["group_top"]["score"] == 0.5
    assert token_strict["group_top"]["score_rate"] == 0.5
    assert token_strict["group_rest"]["score"] == 0.0
    assert token_strict["group_rest"]["score_rate"] == 0.0


def test_extract_run_tracks_unresolved_trails(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _build_demo_run(run_dir)

    # Override one split file with an unknown trail to exercise unresolved mapping paths.
    _write_two_group_file(
        run_dir,
        extract_run.NORMAL_TOKEN_TWO_GROUP_NAME,
        top_names=["profile-0/run_trial-0000-a"],
        rest_names=["profile-9/run_missing"],
        selected_p=1,
    )

    payload = extract_run.extract_score_by_split_from_run_dir(run_dir)
    token_normal = payload["split_scores"]["token_usage"]["normal"]

    assert token_normal["group_top"]["score"] == 1.0
    assert token_normal["group_top"]["missing_trial_count"] == 0
    assert token_normal["group_rest"]["score"] == 0.0
    assert token_normal["group_rest"]["trail_count"] == 1
    assert token_normal["group_rest"]["missing_trial_count"] == 1
    assert token_normal["group_rest"]["resolved_trial_count"] == 0
    # score_rate uses split trail count denominator, so unresolved trails are still counted.
    assert token_normal["group_rest"]["score_rate"] == 0.0


def test_root_dir_writes_score_tables(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _build_demo_run(run_a)
    _build_demo_run(run_b)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])
    assert exit_code == 0

    token_table_path = root_dir / extract_run.ROOT_TOKEN_TABLE_NAME
    context_table_path = root_dir / extract_run.ROOT_CONTEXT_TABLE_NAME
    strict_token_table_path = root_dir / extract_run.ROOT_STRICT_TOKEN_TABLE_NAME
    strict_context_table_path = root_dir / extract_run.ROOT_STRICT_CONTEXT_TABLE_NAME

    assert token_table_path.is_file()
    assert context_table_path.is_file()
    assert strict_token_table_path.is_file()
    assert strict_context_table_path.is_file()

    token_rows = _read_csv_rows(token_table_path)
    assert len(token_rows) == 2
    assert token_rows[0]["group_top_score"] == "1.5"
    assert token_rows[0]["group_top_score_rate"] == "0.75"
    assert token_rows[0]["group_rest_score"] == "0.0"
    assert token_rows[0]["group_rest_score_rate"] == "0.0"
