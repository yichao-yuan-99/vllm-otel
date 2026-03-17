from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from replayer.cli import build_planned_request
from replayer.cli import cmd_compile
from replayer.cli import cmd_replay
from replayer.cli import discover_gateway_run_dirs
from replayer.cli import discover_profiled_job_dirs
from replayer.cli import extract_response_text as extract_cli_response_text
from replayer.cli import is_acceptable_client_disconnect_early_error
from replayer.cli import parse_launch_policy_override_json
from replayer.cli import REPLAY_PLAN_COMPILE_VERSION
from replayer.cli import resolve_replay_lmcache_log_config
from replayer.cli import resolve_replay_launch_policy
from replayer.cli import resolve_replay_vllm_log_config
from replayer.cli import resolve_compile_target, resolve_replay_target
from replayer.cli import timed_cancel_http_json
from replayer.validate import extract_response_text as extract_validate_response_text


def test_resolve_compile_target_extracts_model_and_profile_tokenize_endpoint() -> None:
    config = {
        "backend": {
            "forwarded_args": [
                "--model",
                "hosted_vllm/Test-Model",
            ]
        },
    }

    configured_model, tokenize_endpoint = resolve_compile_target(
        config=config,
        results_entries=[],
        port_profile_id=1,
    )

    assert configured_model == "hosted_vllm/Test-Model"
    assert tokenize_endpoint == "http://127.0.0.1:24123/tokenize"


def test_discover_gateway_run_dirs_supports_cluster_layout(tmp_path: Path) -> None:
    gateway_output_dir = tmp_path / "gateway-output"
    (gateway_output_dir / "run_root").mkdir(parents=True)
    (gateway_output_dir / "profile-3" / "run_profile_3").mkdir(parents=True)
    (gateway_output_dir / "profile-4" / "run_profile_4").mkdir(parents=True)
    (gateway_output_dir / "profile-x" / "run_invalid").mkdir(parents=True)

    discovered = discover_gateway_run_dirs(gateway_output_dir)
    discovered_pairs = sorted((path.name, profile_id) for path, profile_id in discovered)

    assert discovered_pairs == [
        ("run_profile_3", 3),
        ("run_profile_4", 4),
        ("run_root", None),
    ]


def test_discover_profiled_job_dirs_recurses_and_filters(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    valid_a = root / "team-a" / "job-a"
    valid_b = root / "team-b" / "batch" / "job-b"
    invalid = root / "team-c" / "job-c"

    _write_minimal_compile_job(valid_a)
    _write_minimal_compile_job(valid_b)
    (invalid / "meta").mkdir(parents=True, exist_ok=True)
    (invalid / "meta" / "config.toml").write_text("", encoding="utf-8")

    discovered = discover_profiled_job_dirs(root)

    assert discovered == sorted([valid_a.resolve(), valid_b.resolve()])


def test_resolve_compile_target_uses_selected_port_profile() -> None:
    config = {
        "backend": {
            "forwarded_args": [
                "--model",
                "hosted_vllm/Test-Model",
            ]
        },
    }

    configured_model, tokenize_endpoint = resolve_compile_target(
        config=config,
        results_entries=[],
        port_profile_id=0,
    )

    assert configured_model == "hosted_vllm/Test-Model"
    assert tokenize_endpoint == "http://127.0.0.1:11451/tokenize"


def test_resolve_replay_target_uses_required_port_profile_id() -> None:
    api_base, gateway_url, tokenize_endpoint, port_profile_id = resolve_replay_target(
        replay_target={},
        port_profile_id_override=1,
    )

    assert gateway_url == "http://127.0.0.1:24157"
    assert api_base == "http://127.0.0.1:24157/v1"
    assert tokenize_endpoint == "http://127.0.0.1:24123/tokenize"
    assert port_profile_id == "1"


def test_resolve_replay_target_override_uses_gateway_api_base() -> None:
    api_base, gateway_url, tokenize_endpoint, port_profile_id = resolve_replay_target(
        replay_target={
            "port_profile_id": "0",
            "gateway_url": "http://127.0.0.1:18171",
            "api_base": "http://127.0.0.1:11451/v1",
            "tokenize_endpoint": "http://127.0.0.1:11451/tokenize",
        },
        port_profile_id_override=1,
    )

    assert gateway_url == "http://127.0.0.1:24157"
    assert api_base == "http://127.0.0.1:24157/v1"
    assert tokenize_endpoint == "http://127.0.0.1:24123/tokenize"
    assert port_profile_id == "1"


def test_resolve_replay_target_requires_port_profile_id() -> None:
    with pytest.raises(ValueError, match="Replay requires --port-profile-id"):
        resolve_replay_target(
            replay_target={
                "gateway_url": "http://127.0.0.1:18171",
                "api_base": "http://127.0.0.1:9999/v1",
                "tokenize_endpoint": "http://127.0.0.1:11451/tokenize",
            },
            port_profile_id_override=None,
        )


def test_resolve_replay_vllm_log_config_defaults_from_port_profile() -> None:
    config = resolve_replay_vllm_log_config(
        port_profile_id=1,
        interval_s=1.0,
        timeout_s=5.0,
    )

    assert config.enabled is True
    assert config.endpoint == "http://127.0.0.1:24123/metrics"
    assert config.interval_s == 1.0
    assert config.timeout_s == 5.0


def test_resolve_replay_vllm_log_config_requires_port_profile() -> None:
    with pytest.raises(ValueError, match="requires --port-profile-id"):
        resolve_replay_vllm_log_config(
            port_profile_id=None,
            interval_s=1.0,
            timeout_s=5.0,
        )


def test_resolve_replay_lmcache_log_config_defaults_from_port_profile() -> None:
    config = resolve_replay_lmcache_log_config(
        port_profile_id=1,
        interval_s=1.0,
        timeout_s=5.0,
        probe_timeout_s=0.5,
    )

    assert config.configured is True
    assert config.endpoint == "http://127.0.0.1:29437/metrics"
    assert config.interval_s == 1.0
    assert config.timeout_s == 5.0
    assert config.probe_timeout_s == 0.5


def test_parse_launch_policy_override_json_accepts_top_level_launch_policy() -> None:
    parsed = parse_launch_policy_override_json(
        '{"launch_policy":{"max_concurrent":10,"pattern":{"name":"eager"}}}'
    )
    assert parsed == {
        "max_concurrent": 10,
        "pattern": {"name": "eager"},
    }


def test_resolve_replay_launch_policy_accepts_json_overlay() -> None:
    (
        launch_strategy,
        launch_max_concurrent,
        launch_seed,
        launch_pattern_name,
        launch_pattern_rate_per_second,
        _next_launch_delay_s,
        overrides,
        effective_launch_policy,
    ) = resolve_replay_launch_policy(
        launch_policy_payload={
            "strategy": "config_ordered",
            "max_concurrent": 5,
            "seed": 7,
            "pattern": {"name": "eager"},
            "pattern_args": {},
        },
        launch_policy_override={
            "max_concurrent": 10,
            "seed": 11,
            "pattern": {"name": "eager"},
            "pattern_args": {},
        },
    )

    assert launch_strategy == "config_ordered"
    assert launch_max_concurrent == 10
    assert launch_seed == 11
    assert launch_pattern_name == "eager"
    assert launch_pattern_rate_per_second is None
    assert overrides == {
        "max_concurrent": 10,
        "seed": 11,
        "pattern": {"name": "eager"},
        "pattern_args": {},
    }
    assert effective_launch_policy["max_concurrent"] == 10
    assert effective_launch_policy["seed"] == 11


def test_resolve_replay_launch_policy_reads_poisson_rate_from_pattern_args() -> None:
    (
        _launch_strategy,
        launch_max_concurrent,
        _launch_seed,
        launch_pattern_name,
        launch_pattern_rate_per_second,
        _next_launch_delay_s,
        _overrides,
        effective_launch_policy,
    ) = resolve_replay_launch_policy(
        launch_policy_payload={
            "strategy": "config_ordered",
            "max_concurrent": 5,
            "pattern": {"name": "eager"},
            "pattern_args": {},
        },
        launch_policy_override={
            "pattern": {"name": "poisson"},
            "pattern_args": {"rate": 2.0},
        },
    )

    assert launch_max_concurrent is None
    assert launch_pattern_name == "poisson"
    assert launch_pattern_rate_per_second == 2.0
    assert "max_concurrent" not in effective_launch_policy
    assert effective_launch_policy["pattern_args"] == {"rate": 2.0}


def test_resolve_replay_launch_policy_poisson_override_keeps_explicit_max_concurrent() -> None:
    (
        _launch_strategy,
        launch_max_concurrent,
        _launch_seed,
        launch_pattern_name,
        launch_pattern_rate_per_second,
        _next_launch_delay_s,
        _overrides,
        effective_launch_policy,
    ) = resolve_replay_launch_policy(
        launch_policy_payload={
            "strategy": "config_ordered",
            "max_concurrent": 5,
            "pattern": {"name": "eager"},
            "pattern_args": {},
        },
        launch_policy_override={
            "max_concurrent": 3,
            "pattern": {"name": "poisson"},
            "pattern_args": {"rate": 2.0},
        },
    )

    assert launch_max_concurrent == 3
    assert launch_pattern_name == "poisson"
    assert launch_pattern_rate_per_second == 2.0
    assert effective_launch_policy["max_concurrent"] == 3


def test_extract_response_text_rejects_reasoning_only_shape() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning_content": "{\"ok\":true}",
                    "reasoning": "fallback",
                }
            }
        ]
    }

    with pytest.raises(ValueError, match="Unable to extract response text"):
        extract_cli_response_text(payload)
    with pytest.raises(ValueError, match="Unable to extract response text"):
        extract_validate_response_text(payload)


def test_cmd_compile_updates_progress_bar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    meta_dir = job_dir / "meta"
    run_dir = job_dir / "gateway-output" / "run_20260301T000000Z_test_hash"
    requests_dir = run_dir / "requests"
    events_dir = run_dir / "events"
    trace_dir = run_dir / "trace"
    meta_dir.mkdir(parents=True)
    requests_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)

    (meta_dir / "config.toml").write_text(
        "\n".join(
            [
                '[gateway]',
                'enabled = true',
                'url = "http://127.0.0.1:9999"',
                '',
                '[backend]',
                'name = "harbor"',
                'forwarded_args = ["--model", "Test-Model", "--agent-kwarg", "api_base=http://127.0.0.1:9999/v1"]',
                '',
                '[run]',
                'max_concurrent = 1',
                'pattern = "eager"',
                '',
            ]
        ),
        encoding="utf-8",
    )
    (meta_dir / "run_manifest.json").write_text(
        json.dumps({"started_at": "2026-03-01T00:00:00.000Z"}),
        encoding="utf-8",
    )
    (meta_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "trial_id": "trial-1",
                    "command": ["harbor", "--api-token", "token-1"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (meta_dir / "events.jsonl").write_text("", encoding="utf-8")

    api_token_hash = hashlib.sha256(b"token-1").hexdigest()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "api_token_hash": api_token_hash,
                "trace_id": "trace-1",
                "run_start_time": "2026-03-01T00:00:01.000Z",
                "request_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (events_dir / "lifecycle.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "agent_start",
                        "timestamp": "2026-03-01T00:00:02.000Z",
                    }
                ),
                json.dumps(
                    {
                        "event_type": "agent_end",
                        "timestamp": "2026-03-01T00:00:05.000Z",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (requests_dir / "model_inference.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_id": "req-1",
                        "request_start_time": "2026-03-01T00:00:02.100Z",
                        "request_end_time": "2026-03-01T00:00:02.200Z",
                        "request": {"model": "Test-Model", "messages": []},
                        "response": {"choices": [{"message": {"content": "ok-1"}}]},
                    }
                ),
                json.dumps(
                    {
                        "request_id": "req-2",
                        "request_start_time": "2026-03-01T00:00:03.100Z",
                        "request_end_time": "2026-03-01T00:00:03.200Z",
                        "request": {"model": "Test-Model", "messages": []},
                        "response": {"choices": [{"message": {"content": "ok-2"}}]},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (trace_dir / "jaeger_trace.json").write_text(json.dumps({"data": []}), encoding="utf-8")

    def fake_build_planned_request(**kwargs: object) -> dict[str, object]:
        record = kwargs["record"]
        assert isinstance(record, dict)
        index = kwargs["index"]
        assert isinstance(index, int)
        delta = kwargs["delta_agent_action_after_s"]
        return {
            "index": index,
            "method": "POST",
            "path": "v1/chat/completions",
            "body": {"model": "Test-Model", "messages": []},
            "expected_response_text": f"ok-{index + 1}",
            "delta_agent_action_after_s": delta,
        }

    class FakeProgress:
        def __init__(self) -> None:
            self.added: list[dict[str, object]] = []
            self.updates: list[dict[str, object]] = []

        def __enter__(self) -> "FakeProgress":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def add_task(self, description: str, *, total: int, **fields: object) -> int:
            self.added.append(
                {
                    "description": description,
                    "total": total,
                    "fields": fields,
                }
            )
            return 1

        def update(self, task_id: int, *, advance: int = 0, **fields: object) -> None:
            self.updates.append(
                {
                    "task_id": task_id,
                    "advance": advance,
                    "fields": fields,
                }
            )

    fake_progress = FakeProgress()

    monkeypatch.setattr("replayer.cli.build_planned_request", fake_build_planned_request)
    monkeypatch.setattr("replayer.cli.create_compile_progress", lambda: fake_progress)

    exit_code = cmd_compile(
        argparse.Namespace(
            job_dir=str(job_dir),
            plan_out=str(tmp_path / "plan.json"),
            backend=None,
            port_profile_id=1,
        )
    )

    assert exit_code == 0
    plan_payload = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert "agent_timeout_s" not in plan_payload
    assert plan_payload["replay_target"] == {
        "model": "Test-Model",
        "deterministic_required": True,
    }
    assert fake_progress.added == [
        {
            "description": "compiling replay plan",
            "total": 2,
            "fields": {
                "workers_completed": 0,
                "workers_total": 1,
            },
        }
    ]
    assert sum(1 for update in fake_progress.updates if update["advance"] == 1) == 2
    assert fake_progress.updates[-1] == {
        "task_id": 1,
        "advance": 0,
        "fields": {
            "total": 2,
            "workers_completed": 1,
            "workers_total": 1,
        },
    }


def test_cmd_compile_omits_agent_timeout_when_not_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    meta_dir = job_dir / "meta"
    run_dir = job_dir / "gateway-output" / "run_001"
    requests_dir = run_dir / "requests"
    events_dir = run_dir / "events"
    trace_dir = run_dir / "trace"
    meta_dir.mkdir(parents=True)
    requests_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)

    (meta_dir / "config.toml").write_text(
        """
[backend]
name = "harbor"

[run]
pattern = "eager"
max_concurrent = 1

[gateway]
enabled = true
url = "http://127.0.0.1:9999"
""".strip(),
        encoding="utf-8",
    )
    (meta_dir / "run_manifest.json").write_text(
        json.dumps({"started_at": "2026-03-01T00:00:00Z"}),
        encoding="utf-8",
    )
    api_token = "token-1"
    (meta_dir / "results.json").write_text(
        json.dumps(
            [{"trial_id": "trial-1", "command": ["harbor", "--api-token", api_token]}]
        ),
        encoding="utf-8",
    )
    (meta_dir / "events.jsonl").write_text(
        json.dumps({"event_type": "job_start", "timestamp": "2026-03-01T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "api_token_hash": hashlib.sha256(api_token.encode()).hexdigest(),
                "run_start_time": "2026-03-01T00:00:01Z",
                "request_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (events_dir / "lifecycle.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event_type": "agent_start", "timestamp": "2026-03-01T00:00:02Z"}),
                json.dumps({"event_type": "agent_end", "timestamp": "2026-03-01T00:00:03Z"}),
            ]
        ),
        encoding="utf-8",
    )
    (requests_dir / "model_inference.jsonl").write_text("", encoding="utf-8")
    (trace_dir / "jaeger_trace.json").write_text(json.dumps({"data": []}), encoding="utf-8")

    monkeypatch.setattr(
        "replayer.cli.resolve_compile_target",
        lambda **_: (
            "Test-Model",
            "http://127.0.0.1:9998/tokenize",
        ),
    )

    plan_path = tmp_path / "plan-no-timeout.json"
    exit_code = cmd_compile(
        argparse.Namespace(
            job_dir=str(job_dir),
            plan_out=str(plan_path),
            backend=None,
            port_profile_id=1,
        )
    )

    assert exit_code == 0
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert "agent_timeout_s" not in plan_payload


def test_cmd_compile_reads_cluster_gateway_output_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    meta_dir = job_dir / "meta"
    run_dir = job_dir / "gateway-output" / "profile-3" / "run_001"
    requests_dir = run_dir / "requests"
    events_dir = run_dir / "events"
    trace_dir = run_dir / "trace"
    meta_dir.mkdir(parents=True)
    requests_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)

    (meta_dir / "config.toml").write_text(
        """
[backend]
name = "harbor"

[run]
pattern = "eager"
max_concurrent = 1

[gateway]
enabled = true
url = "http://127.0.0.1:9999"
""".strip(),
        encoding="utf-8",
    )
    (meta_dir / "run_manifest.json").write_text(
        json.dumps({"started_at": "2026-03-01T00:00:00Z"}),
        encoding="utf-8",
    )
    api_token = "token-1"
    (meta_dir / "results.json").write_text(
        json.dumps(
            [{"trial_id": "trial-1", "command": ["harbor", "--api-token", api_token]}]
        ),
        encoding="utf-8",
    )
    (meta_dir / "events.jsonl").write_text(
        json.dumps({"event_type": "job_start", "timestamp": "2026-03-01T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "api_token_hash": hashlib.sha256(api_token.encode()).hexdigest(),
                "run_start_time": "2026-03-01T00:00:01Z",
                "request_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (events_dir / "lifecycle.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event_type": "agent_start", "timestamp": "2026-03-01T00:00:02Z"}),
                json.dumps({"event_type": "agent_end", "timestamp": "2026-03-01T00:00:03Z"}),
            ]
        ),
        encoding="utf-8",
    )
    (requests_dir / "model_inference.jsonl").write_text("", encoding="utf-8")
    (trace_dir / "jaeger_trace.json").write_text(json.dumps({"data": []}), encoding="utf-8")

    monkeypatch.setattr(
        "replayer.cli.resolve_compile_target",
        lambda **_: (
            "Test-Model",
            "http://127.0.0.1:9998/tokenize",
        ),
    )

    plan_path = tmp_path / "plan-cluster-layout.json"
    exit_code = cmd_compile(
        argparse.Namespace(
            job_dir=str(job_dir),
            plan_out=str(plan_path),
            backend=None,
            port_profile_id=1,
        )
    )

    assert exit_code == 0
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(plan_payload["workers"]) == 1
    assert plan_payload["workers"][0]["worker_id"] == "trial-1"


def test_cmd_compile_reads_options_from_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    meta_dir = job_dir / "meta"
    run_dir = job_dir / "gateway-output" / "run_001"
    requests_dir = run_dir / "requests"
    events_dir = run_dir / "events"
    trace_dir = run_dir / "trace"
    meta_dir.mkdir(parents=True)
    requests_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)

    (meta_dir / "config.toml").write_text(
        """
[backend]
name = "harbor"

[run]
pattern = "eager"
max_concurrent = 1

[gateway]
enabled = true
url = "http://127.0.0.1:9999"
""".strip(),
        encoding="utf-8",
    )
    (meta_dir / "run_manifest.json").write_text(
        json.dumps({"started_at": "2026-03-01T00:00:00Z"}),
        encoding="utf-8",
    )
    api_token = "token-1"
    (meta_dir / "results.json").write_text(
        json.dumps(
            [{"trial_id": "trial-1", "command": ["harbor", "--api-token", api_token]}]
        ),
        encoding="utf-8",
    )
    (meta_dir / "events.jsonl").write_text(
        json.dumps({"event_type": "job_start", "timestamp": "2026-03-01T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "api_token_hash": hashlib.sha256(api_token.encode()).hexdigest(),
                "run_start_time": "2026-03-01T00:00:01Z",
                "request_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (events_dir / "lifecycle.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event_type": "agent_start", "timestamp": "2026-03-01T00:00:02Z"}),
                json.dumps({"event_type": "agent_end", "timestamp": "2026-03-01T00:00:03Z"}),
            ]
        ),
        encoding="utf-8",
    )
    (requests_dir / "model_inference.jsonl").write_text("", encoding="utf-8")
    (trace_dir / "jaeger_trace.json").write_text(json.dumps({"data": []}), encoding="utf-8")

    monkeypatch.setattr(
        "replayer.cli.resolve_compile_target",
        lambda **_: (
            "Test-Model",
            "http://127.0.0.1:9998/tokenize",
        ),
    )

    plan_path = tmp_path / "plan-from-config.json"
    config_path = tmp_path / "replayer-config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[compile]",
                f'job_dir = "{job_dir}"',
                f'plan_out = "{plan_path}"',
                "port_profile_id = 1",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cmd_compile(
        argparse.Namespace(
            config=str(config_path),
            job_dir=None,
            plan_out=None,
            backend=None,
            port_profile_id=None,
        )
    )

    assert exit_code == 0
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan_payload["replay_target"]["model"] == "Test-Model"


def test_cmd_compile_model_override_updates_replay_target_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    _write_minimal_compile_job(job_dir)
    plan_path = tmp_path / "plan-model-override.json"

    monkeypatch.setattr(
        "replayer.cli.resolve_compile_target",
        lambda **_: (
            "qwen3_coder_30b",
            "http://127.0.0.1:9998/tokenize",
        ),
    )

    exit_code = cmd_compile(
        argparse.Namespace(
            job_dir=str(job_dir),
            plan_out=str(plan_path),
            backend=None,
            port_profile_id=1,
            model="qwen3_coder_30b_fp8",
        )
    )

    assert exit_code == 0
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan_payload["replay_target"]["model"] == "Qwen3-Coder-30B-A3B-Instruct-FP8"
    assert (
        plan_payload["compile_options"]["model_override"]
        == "Qwen3-Coder-30B-A3B-Instruct-FP8"
    )
    assert plan_payload["compile_options"]["model_override_key"] == "qwen3_coder_30b_fp8"


def test_cmd_compile_model_override_forces_recompile_instead_of_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    job_dir = tmp_path / "job"
    _write_minimal_compile_job(job_dir)
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "compile_version": REPLAY_PLAN_COMPILE_VERSION,
                "backend": "harbor",
                "replay_target": {"model": "qwen3_coder_30b"},
                "launch_policy": {"strategy": "config_ordered"},
                "workers": [],
                "compile_options": {"exclude_unranked_trails": False},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "replayer.cli.resolve_compile_target",
        lambda **_: (
            "qwen3_coder_30b",
            "http://127.0.0.1:9998/tokenize",
        ),
    )

    exit_code = cmd_compile(
        argparse.Namespace(
            job_dir=str(job_dir),
            plan_out=str(plan_path),
            backend=None,
            port_profile_id=1,
            model="qwen3_coder_30b_fp8",
        )
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["reused_existing_plan"] is False
    assert summary["model_override"] == "qwen3_coder_30b_fp8"
    assert summary["model_override_key"] == "qwen3_coder_30b_fp8"
    assert summary["model_override_resolved"] == "Qwen3-Coder-30B-A3B-Instruct-FP8"

    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan_payload["replay_target"]["model"] == "Qwen3-Coder-30B-A3B-Instruct-FP8"


def test_cmd_compile_model_override_rejects_unknown_config_model(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    _write_minimal_compile_job(job_dir)
    plan_path = tmp_path / "replay-plan.json"

    with pytest.raises(
        ValueError,
        match="--model must match a name configured in configs/model_config.toml",
    ):
        cmd_compile(
            argparse.Namespace(
                job_dir=str(job_dir),
                plan_out=str(plan_path),
                backend=None,
                port_profile_id=1,
                model="not_a_configured_model",
            )
        )


def test_cmd_compile_rejects_backend_override_in_config(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True)

    config_path = tmp_path / "replayer-config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[compile]",
                f'job_dir = "{job_dir}"',
                "port_profile_id = 1",
                'backend = "harbor"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Compile backend override is no longer supported",
    ):
        cmd_compile(
            argparse.Namespace(
                config=str(config_path),
                job_dir=None,
                plan_out=None,
                port_profile_id=None,
            )
        )


def _write_minimal_compile_job(job_dir: Path, *, api_token: str = "token-1") -> None:
    meta_dir = job_dir / "meta"
    run_dir = job_dir / "gateway-output" / "run_001"
    requests_dir = run_dir / "requests"
    events_dir = run_dir / "events"
    trace_dir = run_dir / "trace"

    meta_dir.mkdir(parents=True)
    requests_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)

    (meta_dir / "config.toml").write_text(
        """
[backend]
name = "harbor"
forwarded_args = ["--model", "Test-Model"]

[run]
pattern = "eager"
max_concurrent = 1

[gateway]
enabled = true
url = "http://127.0.0.1:9999"
""".strip(),
        encoding="utf-8",
    )
    (meta_dir / "run_manifest.json").write_text(
        json.dumps({"started_at": "2026-03-01T00:00:00Z"}),
        encoding="utf-8",
    )
    (meta_dir / "results.json").write_text(
        json.dumps(
            [{"trial_id": "trial-1", "command": ["harbor", "--api-token", api_token]}]
        ),
        encoding="utf-8",
    )
    (meta_dir / "events.jsonl").write_text(
        json.dumps({"event_type": "job_start", "timestamp": "2026-03-01T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "api_token_hash": hashlib.sha256(api_token.encode()).hexdigest(),
                "run_start_time": "2026-03-01T00:00:01Z",
                "request_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (events_dir / "lifecycle.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event_type": "agent_start", "timestamp": "2026-03-01T00:00:02Z"}),
                json.dumps({"event_type": "agent_end", "timestamp": "2026-03-01T00:00:03Z"}),
            ]
        ),
        encoding="utf-8",
    )
    (requests_dir / "model_inference.jsonl").write_text("", encoding="utf-8")
    (trace_dir / "jaeger_trace.json").write_text(json.dumps({"data": []}), encoding="utf-8")


def test_cmd_compile_job_root_compiles_all_discovered_jobs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs_root = tmp_path / "jobs-root"
    job_a = jobs_root / "job-a"
    job_b = jobs_root / "team-x" / "job-b"
    _write_minimal_compile_job(job_a, api_token="token-a")
    _write_minimal_compile_job(job_b, api_token="token-b")

    exit_code = cmd_compile(
        argparse.Namespace(
            config=None,
            job_dir=None,
            job_root=str(jobs_root),
            plan_out=None,
            backend=None,
            port_profile_id=1,
            request_timeout_s=None,
            split_two_group_plans=None,
            split_two_group_metric=None,
        )
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["mode"] == "job_root"
    assert summary["job_count_total"] == 2
    assert summary["job_count_succeeded"] == 2
    assert summary["job_count_failed"] == 0
    assert {item["job_dir"] for item in summary["jobs"]} == {
        str(job_a.resolve()),
        str(job_b.resolve()),
    }
    assert all(item["status"] == "ok" for item in summary["jobs"])
    assert (job_a / "replay-plan.json").is_file()
    assert (job_b / "replay-plan.json").is_file()


def test_cmd_compile_job_root_rejects_plan_out(
    tmp_path: Path,
) -> None:
    jobs_root = tmp_path / "jobs-root"
    jobs_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="--job-root cannot be combined with --plan-out"):
        cmd_compile(
            argparse.Namespace(
                config=None,
                job_dir=None,
                job_root=str(jobs_root),
                plan_out=str(tmp_path / "plan.json"),
                backend=None,
                port_profile_id=1,
                request_timeout_s=None,
                split_two_group_plans=None,
                split_two_group_metric=None,
            )
        )


def _write_split_two_group_compile_job(job_dir: Path) -> None:
    meta_dir = job_dir / "meta"
    run_alpha = job_dir / "gateway-output" / "run_alpha"
    run_beta = job_dir / "gateway-output" / "profile-2" / "run_beta"
    split_dir = job_dir / "original-analysis" / "split"

    for run_dir in [run_alpha, run_beta]:
        (run_dir / "requests").mkdir(parents=True, exist_ok=True)
        (run_dir / "events").mkdir(parents=True, exist_ok=True)
        (run_dir / "trace").mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    token_alpha = "token-alpha"
    token_beta = "token-beta"
    hash_alpha = hashlib.sha256(token_alpha.encode()).hexdigest()
    hash_beta = hashlib.sha256(token_beta.encode()).hexdigest()

    (meta_dir / "config.toml").write_text(
        """
[backend]
name = "harbor"
forwarded_args = ["--model", "Test-Model"]

[run]
pattern = "eager"
max_concurrent = 2
""".strip(),
        encoding="utf-8",
    )
    (meta_dir / "run_manifest.json").write_text(
        json.dumps({"started_at": "2026-03-01T00:00:00Z"}),
        encoding="utf-8",
    )
    (meta_dir / "results.json").write_text(
        json.dumps(
            [
                {"trial_id": "trial-alpha", "command": ["harbor", "--api-token", token_alpha]},
                {"trial_id": "trial-beta", "command": ["harbor", "--api-token", token_beta]},
            ]
        ),
        encoding="utf-8",
    )
    (meta_dir / "events.jsonl").write_text("", encoding="utf-8")

    (run_alpha / "manifest.json").write_text(
        json.dumps(
            {
                "api_token_hash": hash_alpha,
                "run_start_time": "2026-03-01T00:00:01Z",
                "request_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_beta / "manifest.json").write_text(
        json.dumps(
            {
                "api_token_hash": hash_beta,
                "run_start_time": "2026-03-01T00:00:02Z",
                "request_count": 0,
            }
        ),
        encoding="utf-8",
    )

    lifecycle_payload = "\n".join(
        [
            json.dumps({"event_type": "agent_start", "timestamp": "2026-03-01T00:00:03Z"}),
            json.dumps({"event_type": "agent_end", "timestamp": "2026-03-01T00:00:04Z"}),
        ]
    )
    for run_dir in [run_alpha, run_beta]:
        (run_dir / "events" / "lifecycle.jsonl").write_text(lifecycle_payload, encoding="utf-8")
        (run_dir / "requests" / "model_inference.jsonl").write_text("", encoding="utf-8")
        (run_dir / "trace" / "jaeger_trace.json").write_text(
            json.dumps({"data": []}),
            encoding="utf-8",
        )

    (split_dir / "top-p-token-usage-two-groups.json").write_text(
        json.dumps(
            {
                "selection": {"selected_p": 1},
                "group_top": {"trail_names": ["run_alpha"]},
                "group_rest": {"trail_names": ["profile-2/run_beta"]},
            }
        ),
        encoding="utf-8",
    )
    (split_dir / "top-p-context-usage-two-groups.json").write_text(
        json.dumps(
            {
                "selection": {"selected_p": 1},
                "group_top": {"trail_names": ["profile-2/run_beta"]},
                "group_rest": {"trail_names": ["run_alpha"]},
            }
        ),
        encoding="utf-8",
    )


def test_cmd_compile_split_two_group_plans_writes_top_and_rest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    job_dir = tmp_path / "job"
    _write_split_two_group_compile_job(job_dir)
    plan_path = tmp_path / "replay-plan.json"

    exit_code = cmd_compile(
        argparse.Namespace(
            job_dir=str(job_dir),
            plan_out=str(plan_path),
            backend=None,
            port_profile_id=1,
            split_two_group_plans=True,
            split_two_group_metric=None,
        )
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["split_two_group_plans"] is True
    assert summary["split_two_group_metric"] == "token_usage"

    top_path = tmp_path / "replay-plan.token.top.json"
    rest_path = tmp_path / "replay-plan.token.rest.json"
    assert summary["plan_paths"]["top"] == str(top_path.resolve())
    assert summary["plan_paths"]["rest"] == str(rest_path.resolve())
    assert top_path.is_file()
    assert rest_path.is_file()
    assert not plan_path.exists()

    top_payload = json.loads(top_path.read_text(encoding="utf-8"))
    rest_payload = json.loads(rest_path.read_text(encoding="utf-8"))
    assert len(top_payload["workers"]) == 1
    assert len(rest_payload["workers"]) == 1
    assert top_payload["workers"][0]["source_trail_name"] == "run_alpha"
    assert rest_payload["workers"][0]["source_trail_name"] == "profile-2/run_beta"
    assert top_payload["workers"][0]["launch_priority"] == 0
    assert rest_payload["workers"][0]["launch_priority"] == 0
    assert top_payload["split_two_group"]["group"] == "top"
    assert rest_payload["split_two_group"]["group"] == "rest"


def test_cmd_compile_split_two_group_plans_uses_context_metric(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    _write_split_two_group_compile_job(job_dir)
    plan_path = tmp_path / "replay-plan.json"

    exit_code = cmd_compile(
        argparse.Namespace(
            job_dir=str(job_dir),
            plan_out=str(plan_path),
            backend=None,
            port_profile_id=1,
            split_two_group_plans=True,
            split_two_group_metric="context_usage",
        )
    )

    assert exit_code == 0

    top_payload = json.loads(
        (tmp_path / "replay-plan.context.top.json").read_text(encoding="utf-8")
    )
    rest_payload = json.loads(
        (tmp_path / "replay-plan.context.rest.json").read_text(encoding="utf-8")
    )
    assert top_payload["workers"][0]["source_trail_name"] == "profile-2/run_beta"
    assert rest_payload["workers"][0]["source_trail_name"] == "run_alpha"


def test_cmd_compile_reuses_existing_plan_when_compile_version_matches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True)
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "compile_version": REPLAY_PLAN_COMPILE_VERSION,
                "backend": "harbor",
                "launch_policy": {"strategy": "config_ordered"},
                "workers": [{"requests": [{}, {}]}],
            }
        ),
        encoding="utf-8",
    )

    exit_code = cmd_compile(
        argparse.Namespace(
            job_dir=str(job_dir),
            plan_out=str(plan_path),
            backend=None,
            port_profile_id=1,
        )
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["reused_existing_plan"] is True
    assert summary["compile_version"] == REPLAY_PLAN_COMPILE_VERSION
    assert summary["worker_count"] == 1
    assert summary["request_count"] == 2


def test_cmd_compile_reuses_existing_plan_when_compile_version_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True)
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "backend": "harbor",
                "launch_policy": {"strategy": "config_ordered"},
                "workers": [],
            }
        ),
        encoding="utf-8",
    )

    exit_code = cmd_compile(
        argparse.Namespace(
            job_dir=str(job_dir),
            plan_out=str(plan_path),
            backend=None,
            port_profile_id=1,
        )
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["reused_existing_plan"] is True
    assert summary["compile_version"] == REPLAY_PLAN_COMPILE_VERSION


def test_cmd_compile_reuses_existing_plan_when_compile_version_empty(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True)
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "compile_version": "",
                "backend": "harbor",
                "launch_policy": {"strategy": "config_ordered"},
                "workers": [],
            }
        ),
        encoding="utf-8",
    )

    exit_code = cmd_compile(
        argparse.Namespace(
            job_dir=str(job_dir),
            plan_out=str(plan_path),
            backend=None,
            port_profile_id=1,
        )
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["reused_existing_plan"] is True
    assert summary["compile_version"] == REPLAY_PLAN_COMPILE_VERSION


def test_cmd_compile_recompiles_when_compile_version_mismatches(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_minimal_compile_job(job_dir)
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "compile_version": "stale-version",
                "backend": "harbor",
                "legacy_marker": True,
                "workers": [],
            }
        ),
        encoding="utf-8",
    )

    exit_code = cmd_compile(
        argparse.Namespace(
            job_dir=str(job_dir),
            plan_out=str(plan_path),
            backend=None,
            port_profile_id=1,
        )
    )

    assert exit_code == 0
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan_payload["compile_version"] == REPLAY_PLAN_COMPILE_VERSION
    assert "legacy_marker" not in plan_payload
    assert plan_payload["source_job_dir"] == str(job_dir.resolve())


def test_build_planned_request_for_client_disconnected_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_tokenize_response_text(**_: object) -> list[int]:
        raise AssertionError("tokenize_response_text should not be called")

    monkeypatch.setattr("replayer.cli.tokenize_response_text", fail_tokenize_response_text)

    planned = build_planned_request(
        record={
            "request_id": "req-1",
            "http_method": "POST",
            "http_path": "v1/chat/completions",
            "request_duration_ms": 600068.796,
            "request": {
                "model": "Test-Model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 32,
                "stop": ["DONE"],
                "stop_token_ids": [2],
                "vllm_xargs": {
                    "forced_token_ids": [1, 2, 3],
                    "force_eos_after_sequence": True,
                    "temperature": 0.0,
                },
            },
            "response": {
                "error": "client_disconnected",
                "detail": "downstream client disconnected before the upstream response completed",
            },
            "status_code": 499,
        },
        index=0,
        configured_model="Configured-Model",
        tokenize_endpoint="http://127.0.0.1:9998/tokenize",
        request_timeout_s=5.0,
        delta_agent_action_after_s=1.25,
    )

    assert planned["replay_mode"] == "client_disconnect_after_duration"
    assert planned["cancel_after_s"] == 600.068796
    assert planned["expected_status_code"] == 499
    assert planned["expected_error"] == "client_disconnected"
    assert planned["expected_response_text"] is None
    assert planned["force_eos_after_sequence"] is False
    assert planned["body"]["ignore_eos"] is True
    assert "max_tokens" not in planned["body"]
    assert "stop" not in planned["body"]
    assert "stop_token_ids" not in planned["body"]
    assert planned["body"]["vllm_xargs"] == {"temperature": 0.0}


def test_build_planned_request_model_override_rewrites_body_and_tokenize_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_tokenize_response_text(
        *,
        tokenize_endpoint: str,
        model_name: str,
        text: str | None,
        timeout_s: float,
    ) -> list[int]:
        observed["tokenize_endpoint"] = tokenize_endpoint
        observed["model_name"] = model_name
        observed["text"] = text
        observed["timeout_s"] = timeout_s
        return [1, 2, 3]

    monkeypatch.setattr("replayer.cli.tokenize_response_text", fake_tokenize_response_text)

    planned = build_planned_request(
        record={
            "request_id": "req-override-1",
            "model": "source-model-record",
            "http_method": "POST",
            "http_path": "v1/chat/completions",
            "request": {
                "model": "source-model-body",
                "messages": [{"role": "user", "content": "hi"}],
            },
            "response": {"choices": [{"message": {"content": "ok"}}]},
            "status_code": 200,
        },
        index=0,
        configured_model="source-model-config",
        model_override="target-model-override",
        tokenize_endpoint="http://127.0.0.1:9998/tokenize",
        request_timeout_s=5.0,
        delta_agent_action_after_s=0.0,
    )

    assert planned["model_for_tokenize"] == "target-model-override"
    assert planned["body"]["model"] == "target-model-override"
    assert observed["model_name"] == "target-model-override"


def test_cmd_replay_updates_progress_bar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "replay_target": {
                    "port_profile_id": "1",
                    "api_base": "http://127.0.0.1:9999/v1",
                    "gateway_url": "http://127.0.0.1:9999",
                    "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 1,
                    "pattern": {"name": "eager"},
                },
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {"model": "Test-Model", "messages": []},
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeProgress:
        def __init__(self) -> None:
            self.added: list[dict[str, object]] = []
            self.updates: list[dict[str, object]] = []

        def __enter__(self) -> "FakeProgress":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def add_task(self, description: str, *, total: int, **fields: object) -> int:
            self.added.append(
                {
                    "description": description,
                    "total": total,
                    "fields": fields,
                }
            )
            return 1

        def update(self, task_id: int, *, advance: int = 0, **fields: object) -> None:
            self.updates.append(
                {
                    "task_id": task_id,
                    "advance": advance,
                    "fields": fields,
                }
            )

    fake_progress = FakeProgress()

    def fake_create_replay_progress() -> FakeProgress:
        return fake_progress

    def fake_http_json(
        *,
        method: str,
        url: str,
        payload: object,
        timeout_s: float | None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, object]:
        assert timeout_s is None
        return (
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "ok",
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("replayer.cli.create_replay_progress", fake_create_replay_progress)
    monkeypatch.setattr("replayer.cli.http_json", fake_http_json)

    exit_code = cmd_replay(
        argparse.Namespace(
            plan=str(plan_path),
            output_dir=str(tmp_path / "replay-output"),
            port_profile_id=1,
            launch_policy_override_json='{"max_concurrent": 1}',
            vllm_log=None,
            vllm_log_interval_s=1.0,
            vllm_log_timeout_s=5.0,
        )
    )

    assert exit_code == 0
    assert fake_progress.added == [
        {
            "description": "replaying",
            "total": 1,
            "fields": {"launched": 0, "active": 0, "failed": 0},
        }
    ]
    assert {
        "task_id": 1,
        "advance": 0,
        "fields": {"launched": 1, "active": 1, "failed": 0},
    } in fake_progress.updates
    assert {
        "task_id": 1,
        "advance": 1,
        "fields": {"launched": 1, "active": 0, "failed": 0},
    } in fake_progress.updates


def test_cmd_replay_enables_lmcache_log_when_probe_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "replay-plan.json"
    output_dir = tmp_path / "replay-output"
    plan_path.write_text(
        json.dumps(
            {
                "replay_target": {
                    "port_profile_id": "1",
                    "api_base": "http://127.0.0.1:9999/v1",
                    "gateway_url": "http://127.0.0.1:9999",
                    "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 1,
                    "pattern": {"name": "eager"},
                },
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "api_token": "token-1",
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {"model": "Test-Model", "messages": []},
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_http_json(
        *,
        method: str,
        url: str,
        payload: object,
        timeout_s: float | None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, object]:
        del method, payload, headers
        assert timeout_s is None
        if url.endswith("/v1/chat/completions"):
            return (
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": "ok",
                            }
                        }
                    ]
                },
            )
        return 200, {}

    def fake_vllm_monitor(*, output_dir: Path, config: object) -> object:
        del config
        return type(
            "FakeMonitor",
            (),
            {
                "stdout_log": output_dir / "vllm-log" / "monitor.stdout.log",
                "stderr_log": output_dir / "vllm-log" / "monitor.stderr.log",
            },
        )()

    def fake_lmcache_monitor(*, output_dir: Path, config: object) -> object:
        del config
        return type(
            "FakeMonitor",
            (),
            {
                "stdout_log": output_dir / "lmcache-log" / "monitor.stdout.log",
                "stderr_log": output_dir / "lmcache-log" / "monitor.stderr.log",
            },
        )()

    monkeypatch.setattr("replayer.cli.http_json", fake_http_json)
    monkeypatch.setattr("replayer.cli.start_replay_vllm_monitor", fake_vllm_monitor)
    monkeypatch.setattr("replayer.cli.start_replay_lmcache_monitor", fake_lmcache_monitor)
    monkeypatch.setattr("replayer.cli.stop_replay_vllm_monitor", lambda monitor: 0)
    monkeypatch.setattr(
        "replayer.cli.probe_metrics_endpoint",
        lambda *, endpoint, timeout_s: (True, None),
    )

    exit_code = cmd_replay(
        argparse.Namespace(
            plan=str(plan_path),
            output_dir=str(output_dir),
            port_profile_id=1,
            launch_policy_override_json='{"max_concurrent": 1}',
            vllm_log=None,
            vllm_log_interval_s=1.0,
            vllm_log_timeout_s=5.0,
        )
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "replay" / "summary.json").read_text(encoding="utf-8"))
    assert summary["lmcache_log_configured"] is True
    assert summary["lmcache_log_probe_success"] is True
    assert summary["lmcache_log_enabled"] is True
    assert summary["lmcache_log_dir"] == str(output_dir / "lmcache-log")
    assert summary["lmcache_log_monitor_return_code"] == 0


def test_cmd_replay_num_tasks_truncates_to_first_launch_order_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "replay_target": {
                    "port_profile_id": "1",
                    "api_base": "http://127.0.0.1:9999/v1",
                    "gateway_url": "http://127.0.0.1:9999",
                    "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 1,
                    "pattern": {"name": "eager"},
                },
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "launch_priority": 0,
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "marker-1"}],
                                },
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    },
                    {
                        "worker_id": "worker-2",
                        "launch_priority": 1,
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "marker-2"}],
                                },
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    },
                    {
                        "worker_id": "worker-3",
                        "launch_priority": 2,
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "marker-3"}],
                                },
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    observed_markers: list[str] = []

    def fake_http_json(
        *,
        method: str,
        url: str,
        payload: object,
        timeout_s: float | None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, object]:
        del method, timeout_s, headers
        if url.endswith("/job/start") or url.endswith("/agent/start") or url.endswith("/agent/end") or url.endswith("/job/end"):
            return 200, {}
        assert isinstance(payload, dict)
        messages = payload.get("messages")
        assert isinstance(messages, list) and messages
        first_message = messages[0]
        assert isinstance(first_message, dict)
        content = first_message.get("content")
        assert isinstance(content, str)
        observed_markers.append(content)
        return 200, {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("replayer.cli.http_json", fake_http_json)

    output_dir = tmp_path / "replay-output"
    exit_code = cmd_replay(
        argparse.Namespace(
            plan=str(plan_path),
            output_dir=str(output_dir),
            num_tasks=2,
            port_profile_id=1,
            launch_policy_override_json=None,
            vllm_log=None,
            vllm_log_interval_s=1.0,
            vllm_log_timeout_s=5.0,
        )
    )

    assert exit_code == 0
    assert observed_markers == ["marker-1", "marker-2"]

    summary = json.loads((output_dir / "replay" / "summary.json").read_text(encoding="utf-8"))
    assert summary["workers_total"] == 2
    assert summary["workers_completed"] == 2
    assert summary["num_tasks_requested"] == 2
    assert summary["workers_in_plan"] == 3
    assert summary["tasks_wrapped"] is False

    worker_log_names = sorted(path.name for path in (output_dir / "replay" / "workers").glob("*.json"))
    assert worker_log_names == ["worker-1.json", "worker-2.json"]


def test_cmd_replay_num_tasks_wraps_when_requested_tasks_exceed_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "replay_target": {
                    "port_profile_id": "1",
                    "api_base": "http://127.0.0.1:9999/v1",
                    "gateway_url": "http://127.0.0.1:9999",
                    "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 1,
                    "pattern": {"name": "eager"},
                },
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "launch_priority": 0,
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "marker-1"}],
                                },
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    },
                    {
                        "worker_id": "worker-2",
                        "launch_priority": 1,
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "marker-2"}],
                                },
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    observed_markers: list[str] = []

    def fake_http_json(
        *,
        method: str,
        url: str,
        payload: object,
        timeout_s: float | None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, object]:
        del method, timeout_s, headers
        if url.endswith("/job/start") or url.endswith("/agent/start") or url.endswith("/agent/end") or url.endswith("/job/end"):
            return 200, {}
        assert isinstance(payload, dict)
        messages = payload.get("messages")
        assert isinstance(messages, list) and messages
        first_message = messages[0]
        assert isinstance(first_message, dict)
        content = first_message.get("content")
        assert isinstance(content, str)
        observed_markers.append(content)
        return 200, {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("replayer.cli.http_json", fake_http_json)

    output_dir = tmp_path / "replay-output"
    exit_code = cmd_replay(
        argparse.Namespace(
            plan=str(plan_path),
            output_dir=str(output_dir),
            num_tasks=5,
            port_profile_id=1,
            launch_policy_override_json=None,
            vllm_log=None,
            vllm_log_interval_s=1.0,
            vllm_log_timeout_s=5.0,
        )
    )

    assert exit_code == 0
    assert observed_markers == ["marker-1", "marker-2", "marker-1", "marker-2", "marker-1"]

    summary = json.loads((output_dir / "replay" / "summary.json").read_text(encoding="utf-8"))
    assert summary["workers_total"] == 5
    assert summary["workers_completed"] == 5
    assert summary["num_tasks_requested"] == 5
    assert summary["workers_in_plan"] == 2
    assert summary["tasks_wrapped"] is True

    worker_log_names = sorted(path.name for path in (output_dir / "replay" / "workers").glob("*.json"))
    assert worker_log_names == [
        "worker-1.json",
        "worker-1__wrap2__task3.json",
        "worker-1__wrap3__task5.json",
        "worker-2.json",
        "worker-2__wrap2__task4.json",
    ]


def test_cmd_replay_reads_config_and_cli_overrides_take_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "replay_target": {
                    "port_profile_id": "1",
                    "api_base": "http://127.0.0.1:9999/v1",
                    "gateway_url": "http://127.0.0.1:9999",
                    "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 2,
                    "pattern": {"name": "eager"},
                },
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "launch_priority": 0,
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "marker-1"}],
                                },
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    },
                    {
                        "worker_id": "worker-2",
                        "launch_priority": 1,
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "marker-2"}],
                                },
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "replay-output"
    config_path = tmp_path / "replayer-config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[replay]",
                f'plan = "{plan_path}"',
                f'output_dir = "{output_dir}"',
                "num_tasks = 1",
                "vllm_log_interval_s = 1.0",
                "vllm_log_timeout_s = 5.0",
                "",
                "[replay.launch_policy_override]",
                "max_concurrent = 1",
            ]
        ),
        encoding="utf-8",
    )

    observed_markers: list[str] = []

    def fake_http_json(
        *,
        method: str,
        url: str,
        payload: object,
        timeout_s: float | None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, object]:
        del method, timeout_s, headers
        if url.endswith("/job/start") or url.endswith("/agent/start") or url.endswith("/agent/end") or url.endswith("/job/end"):
            return 200, {}
        assert isinstance(payload, dict)
        messages = payload.get("messages")
        assert isinstance(messages, list) and messages
        first_message = messages[0]
        assert isinstance(first_message, dict)
        content = first_message.get("content")
        assert isinstance(content, str)
        observed_markers.append(content)
        return 200, {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("replayer.cli.http_json", fake_http_json)

    exit_code = cmd_replay(
        argparse.Namespace(
            config=str(config_path),
            plan=None,
            output_dir=None,
            num_tasks=2,
            port_profile_id=1,
            launch_policy_override_json=None,
            vllm_log=None,
            vllm_log_interval_s=None,
            vllm_log_timeout_s=None,
        )
    )

    assert exit_code == 0
    assert observed_markers == ["marker-1", "marker-2"]

    summary = json.loads((output_dir / "replay" / "summary.json").read_text(encoding="utf-8"))
    assert summary["workers_total"] == 2
    assert summary["num_tasks_requested"] == 2
    assert summary["launch_policy_overrides"] == {"max_concurrent": 1}


def test_cmd_replay_poisson_override_drops_plan_max_concurrent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "replay_target": {
                    "port_profile_id": "1",
                    "api_base": "http://127.0.0.1:9999/v1",
                    "gateway_url": "http://127.0.0.1:9999",
                    "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 1,
                    "pattern": {"name": "eager"},
                },
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "launch_priority": 0,
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "marker-1"}],
                                },
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    },
                    {
                        "worker_id": "worker-2",
                        "launch_priority": 1,
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "marker-2"}],
                                },
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "replay-output"
    config_path = tmp_path / "replayer-config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[replay]",
                f'plan = "{plan_path}"',
                f'output_dir = "{output_dir}"',
                "num_tasks = 2",
                "vllm_log_interval_s = 1.0",
                "vllm_log_timeout_s = 5.0",
                "",
                "[replay.launch_policy_override]",
                "seed = 11",
                "",
                "[replay.launch_policy_override.pattern]",
                'name = "poisson"',
                "",
                "[replay.launch_policy_override.pattern_args]",
                "rate = 100.0",
            ]
        ),
        encoding="utf-8",
    )

    observed_markers: list[str] = []

    def fake_http_json(
        *,
        method: str,
        url: str,
        payload: object,
        timeout_s: float | None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, object]:
        del method, timeout_s, headers
        if (
            url.endswith("/job/start")
            or url.endswith("/agent/start")
            or url.endswith("/agent/end")
            or url.endswith("/job/end")
        ):
            return 200, {}
        assert isinstance(payload, dict)
        messages = payload.get("messages")
        assert isinstance(messages, list) and messages
        first_message = messages[0]
        assert isinstance(first_message, dict)
        content = first_message.get("content")
        assert isinstance(content, str)
        observed_markers.append(content)
        return 200, {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("replayer.cli.http_json", fake_http_json)
    monkeypatch.setattr("replayer.cli.sleep_with_stop", lambda stop_event, seconds: True)

    exit_code = cmd_replay(
        argparse.Namespace(
            config=str(config_path),
            plan=None,
            output_dir=None,
            num_tasks=None,
            port_profile_id=1,
            launch_policy_override_json=None,
            vllm_log=None,
            vllm_log_interval_s=None,
            vllm_log_timeout_s=None,
        )
    )

    assert exit_code == 0
    assert observed_markers == ["marker-1", "marker-2"]

    summary = json.loads((output_dir / "replay" / "summary.json").read_text(encoding="utf-8"))
    assert summary["launch_pattern"] == "poisson"
    assert summary["launch_max_concurrent"] is None
    assert summary["launch_seed"] == 11
    assert "max_concurrent" not in summary["effective_launch_policy"]


def test_cmd_replay_rejects_vllm_log_false_in_config(tmp_path: Path) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "replay_target": {
                    "model": "Test-Model",
                    "deterministic_required": True,
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 1,
                    "pattern": {"name": "eager"},
                },
                "workers": [],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "replayer-config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[replay]",
                f'plan = "{plan_path}"',
                "vllm_log = false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no longer supports disabling vLLM logging"):
        cmd_replay(
            argparse.Namespace(
                config=str(config_path),
                plan=None,
                output_dir=str(tmp_path / "replay-output"),
                num_tasks=None,
                port_profile_id=1,
                launch_policy_override_json=None,
                vllm_log=None,
                vllm_log_interval_s=1.0,
                vllm_log_timeout_s=5.0,
                agent_timeout_s=None,
            )
        )


def test_cmd_replay_handles_client_disconnect_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "agent_timeout_s": 3000.0,
                "replay_target": {
                    "port_profile_id": "1",
                    "api_base": "http://127.0.0.1:9999/v1",
                    "gateway_url": "http://127.0.0.1:9999",
                    "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 1,
                    "pattern": {"name": "eager"},
                },
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "hi"}],
                                    "ignore_eos": True,
                                },
                                "replay_mode": "client_disconnect_after_duration",
                                "cancel_after_s": 600.0,
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_timed_cancel_http_json(**_: object) -> dict[str, object]:
        return {
            "outcome": "cancelled",
            "response_status": None,
            "response_payload": None,
            "error": None,
        }

    monkeypatch.setattr("replayer.cli.timed_cancel_http_json", fake_timed_cancel_http_json)
    monkeypatch.setattr("replayer.cli.http_json", lambda **_: (200, {}))

    output_dir = tmp_path / "replay-output"
    exit_code = cmd_replay(
        argparse.Namespace(
            plan=str(plan_path),
            output_dir=str(output_dir),
            port_profile_id=1,
            agent_timeout_s=3000.0,
            launch_policy_override_json=None,
            vllm_log=None,
            vllm_log_interval_s=1.0,
            vllm_log_timeout_s=5.0,
        )
    )

    assert exit_code == 0

    summary = json.loads((output_dir / "replay" / "summary.json").read_text(encoding="utf-8"))
    assert summary["workers_completed"] == 1
    assert summary["workers_failed"] == 0
    assert summary["workers_timed_out"] == 0
    assert summary["requests_sent"] == 1
    assert summary["requests_failed"] == 0

    worker_log = json.loads(
        (output_dir / "replay" / "workers" / "worker-1.json").read_text(encoding="utf-8")
    )
    assert worker_log["status"] == "completed"
    assert worker_log["requests_succeeded"] == 1


def test_cmd_replay_accepts_long_running_client_disconnect_gateway_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "replay_target": {
                    "port_profile_id": "1",
                    "api_base": "http://127.0.0.1:9999/v1",
                    "gateway_url": "http://127.0.0.1:9999",
                    "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 1,
                    "pattern": {"name": "eager"},
                },
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "hi"}],
                                    "ignore_eos": True,
                                },
                                "replay_mode": "client_disconnect_after_duration",
                                "cancel_after_s": 600.0,
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_timed_cancel_http_json(**_: object) -> dict[str, object]:
        return {
            "outcome": "completed_early",
            "elapsed_s": 61.0,
            "response_status": 400,
            "response_payload": {"error": {"message": "mock gateway error"}},
            "error": None,
        }

    monkeypatch.setattr("replayer.cli.timed_cancel_http_json", fake_timed_cancel_http_json)
    monkeypatch.setattr("replayer.cli.http_json", lambda **_: (200, {}))

    output_dir = tmp_path / "replay-output"
    exit_code = cmd_replay(
        argparse.Namespace(
            plan=str(plan_path),
            output_dir=str(output_dir),
            port_profile_id=1,
            launch_policy_override_json=None,
            vllm_log=None,
            vllm_log_interval_s=1.0,
            vllm_log_timeout_s=5.0,
        )
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "replay" / "summary.json").read_text(encoding="utf-8"))
    assert summary["workers_completed"] == 1
    assert summary["workers_failed"] == 0
    assert summary["requests_sent"] == 1
    assert summary["requests_failed"] == 0

    worker_log = json.loads(
        (output_dir / "replay" / "workers" / "worker-1.json").read_text(encoding="utf-8")
    )
    assert worker_log["status"] == "completed"
    assert worker_log["requests_succeeded"] == 1


def test_is_acceptable_client_disconnect_early_error_requires_error_and_one_minute() -> None:
    assert (
        is_acceptable_client_disconnect_early_error(
            {
                "response_status": 400,
                "elapsed_s": 61.0,
            }
        )
        is True
    )
    assert (
        is_acceptable_client_disconnect_early_error(
            {
                "response_status": 400,
                "elapsed_s": 59.0,
            }
        )
        is False
    )
    assert (
        is_acceptable_client_disconnect_early_error(
            {
                "response_status": 200,
                "elapsed_s": 120.0,
            }
        )
        is False
    )


def test_timed_cancel_http_json_returns_completed_early_when_response_finishes() -> None:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                self.rfile.read(length)
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        start = time.monotonic()
        result = timed_cancel_http_json(
            method="POST",
            url=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            payload={"model": "Test-Model", "messages": []},
            cancel_after_s=600.0,
            connect_timeout_s=None,
            stop_event=threading.Event(),
        )
        elapsed = time.monotonic() - start
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)

    assert result["outcome"] == "completed_early"
    assert result["response_status"] == 200
    assert result["response_payload"] == {"ok": True}
    assert isinstance(result["elapsed_s"], float)
    assert result["elapsed_s"] >= 0.0
    assert elapsed < 2.0


def test_cmd_replay_marks_worker_timed_out_at_agent_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "agent_timeout_s": 3000.0,
                "replay_target": {
                    "port_profile_id": "1",
                    "api_base": "http://127.0.0.1:9999/v1",
                    "gateway_url": "http://127.0.0.1:9999",
                    "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 1,
                    "pattern": {"name": "eager"},
                },
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {"model": "Test-Model", "messages": []},
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_timed_cancel_http_json(**kwargs: object) -> dict[str, object]:
        cancel_after_s = kwargs["cancel_after_s"]
        assert isinstance(cancel_after_s, float)
        assert 0.0 < cancel_after_s <= 3000.0
        return {
            "outcome": "cancelled",
            "response_status": None,
            "response_payload": None,
            "error": None,
        }

    monkeypatch.setattr("replayer.cli.timed_cancel_http_json", fake_timed_cancel_http_json)
    monkeypatch.setattr("replayer.cli.http_json", lambda **_: (200, {}))

    output_dir = tmp_path / "replay-output"
    exit_code = cmd_replay(
        argparse.Namespace(
            plan=str(plan_path),
            output_dir=str(output_dir),
            port_profile_id=1,
            launch_policy_override_json=None,
            vllm_log=None,
            vllm_log_interval_s=1.0,
            vllm_log_timeout_s=5.0,
        )
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "replay" / "summary.json").read_text(encoding="utf-8"))
    assert summary["workers_failed"] == 0
    assert summary["workers_timed_out"] == 1
    assert summary["requests_sent"] == 1
    assert summary["requests_failed"] == 1

    worker_log = json.loads(
        (output_dir / "replay" / "workers" / "worker-1.json").read_text(encoding="utf-8")
    )
    assert worker_log["status"] == "timed_out"
    assert worker_log["requests_failed"] == 1


def test_cmd_replay_gateway_calls_have_no_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "agent_timeout_s": 3000.0,
                "replay_target": {
                    "port_profile_id": "1",
                    "api_base": "http://127.0.0.1:9999/v1",
                    "gateway_url": "http://127.0.0.1:9999",
                    "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 1,
                    "pattern": {"name": "eager"},
                },
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "api_token": "token-1",
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    observed: list[tuple[str, str | None, float | None]] = []

    def fake_http_json(
        *,
        method: str,
        url: str,
        payload: object,
        timeout_s: float | None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, object]:
        observed.append((method, url, timeout_s))
        return 200, {}

    monkeypatch.setattr("replayer.cli.http_json", fake_http_json)

    output_dir = tmp_path / "replay-output"
    exit_code = cmd_replay(
        argparse.Namespace(
            plan=str(plan_path),
            output_dir=str(output_dir),
            port_profile_id=1,
            launch_policy_override_json=None,
            vllm_log=None,
            vllm_log_interval_s=1.0,
            vllm_log_timeout_s=5.0,
        )
    )

    assert exit_code == 0
    assert observed == [
        ("POST", "http://127.0.0.1:24157/job/start", None),
        ("POST", "http://127.0.0.1:24157/agent/start", None),
        ("POST", "http://127.0.0.1:24157/agent/end", None),
        ("POST", "http://127.0.0.1:24157/job/end", None),
    ]
    summary = json.loads((output_dir / "replay" / "summary.json").read_text(encoding="utf-8"))
    worker_result = summary["worker_results"]["worker-1"]
    assert worker_result["api_token"] == "token-1"


def test_cmd_replay_randomize_seed_shuffles_launch_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "replay_target": {
                    "port_profile_id": "1",
                    "api_base": "http://127.0.0.1:9999/v1",
                    "gateway_url": "http://127.0.0.1:9999",
                    "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 1,
                    "pattern": {"name": "eager"},
                },
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "api_token": "token-1",
                        "launch_priority": 0,
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "marker-1"}],
                                },
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    },
                    {
                        "worker_id": "worker-2",
                        "api_token": "token-2",
                        "launch_priority": 1,
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "marker-2"}],
                                },
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    },
                    {
                        "worker_id": "worker-3",
                        "api_token": "token-3",
                        "launch_priority": 2,
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "marker-3"}],
                                },
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    observed_markers: list[str] = []

    def fake_http_json(
        *,
        method: str,
        url: str,
        payload: object,
        timeout_s: float | None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, object]:
        del method, timeout_s, headers
        if (
            url.endswith("/job/start")
            or url.endswith("/agent/start")
            or url.endswith("/agent/end")
            or url.endswith("/job/end")
        ):
            return 200, {}
        assert isinstance(payload, dict)
        messages = payload.get("messages")
        assert isinstance(messages, list) and messages
        first_message = messages[0]
        assert isinstance(first_message, dict)
        content = first_message.get("content")
        assert isinstance(content, str)
        observed_markers.append(content)
        return 200, {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("replayer.cli.http_json", fake_http_json)

    output_dir = tmp_path / "replay-output"
    exit_code = cmd_replay(
        argparse.Namespace(
            plan=str(plan_path),
            output_dir=str(output_dir),
            num_tasks=3,
            randomize_seed=7,
            time_constraint_s=None,
            port_profile_id=1,
            launch_policy_override_json=None,
            vllm_log=None,
            vllm_log_interval_s=1.0,
            vllm_log_timeout_s=5.0,
        )
    )

    assert exit_code == 0
    assert observed_markers == ["marker-3", "marker-1", "marker-2"]

    summary = json.loads((output_dir / "replay" / "summary.json").read_text(encoding="utf-8"))
    assert summary["workers_total"] == 3
    assert summary["workers_completed"] == 3
    assert summary["randomize_seed"] == 7


def test_cmd_replay_time_constraint_marks_unfinished_workers_time_bound_finished(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "replay_target": {
                    "port_profile_id": "1",
                    "api_base": "http://127.0.0.1:9999/v1",
                    "gateway_url": "http://127.0.0.1:9999",
                    "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 1,
                    "pattern": {"name": "eager"},
                },
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "api_token": "token-1",
                        "launch_priority": 0,
                        "delta_agent_start_s": 0.0,
                        "delta_first_request_s": 0.0,
                        "requests": [
                            {
                                "method": "POST",
                                "path": "v1/chat/completions",
                                "body": {
                                    "model": "Test-Model",
                                    "messages": [{"role": "user", "content": "marker-1"}],
                                },
                                "expected_response_text": "ok",
                                "delta_agent_action_after_s": 0.0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_http_json(
        *,
        method: str,
        url: str,
        payload: object,
        timeout_s: float | None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, object]:
        del method, payload, timeout_s, headers
        if (
            url.endswith("/job/start")
            or url.endswith("/agent/start")
            or url.endswith("/agent/end")
            or url.endswith("/job/end")
        ):
            return 200, {}
        raise AssertionError("Deterministic request path should use timed_cancel_http_json")

    def fake_timed_cancel_http_json(**kwargs: object) -> dict[str, object]:
        stop_event = kwargs.get("stop_event")
        assert isinstance(stop_event, threading.Event)
        stop_event.wait(timeout=1.0)
        return {
            "outcome": "stopped",
            "elapsed_s": 0.1,
            "response_status": None,
            "response_payload": None,
            "error": None,
        }

    monkeypatch.setattr("replayer.cli.http_json", fake_http_json)
    monkeypatch.setattr("replayer.cli.timed_cancel_http_json", fake_timed_cancel_http_json)

    output_dir = tmp_path / "replay-output"
    exit_code = cmd_replay(
        argparse.Namespace(
            plan=str(plan_path),
            output_dir=str(output_dir),
            num_tasks=None,
            randomize_seed=None,
            time_constraint_s=0.05,
            port_profile_id=1,
            launch_policy_override_json=None,
            vllm_log=None,
            vllm_log_interval_s=1.0,
            vllm_log_timeout_s=5.0,
        )
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "replay" / "summary.json").read_text(encoding="utf-8"))
    assert summary["workers_failed"] == 0
    assert summary["workers_timed_out"] == 0
    assert summary["workers_time_bound_finished"] == 1
    assert summary["time_constraint_reached"] is True
    assert summary["workers_total"] == 1
    assert summary["requests_sent"] == 1
    assert summary["requests_failed"] == 0

    worker_log = json.loads(
        (output_dir / "replay" / "workers" / "worker-1.json").read_text(encoding="utf-8")
    )
    assert worker_log["status"] == "time_bound_finished"


def test_cmd_replay_rejects_time_constraint_with_num_tasks(tmp_path: Path) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "replay_target": {
                    "port_profile_id": "1",
                    "api_base": "http://127.0.0.1:9999/v1",
                    "gateway_url": "http://127.0.0.1:9999",
                    "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
                },
                "launch_policy": {
                    "strategy": "config_ordered",
                    "max_concurrent": 1,
                    "pattern": {"name": "eager"},
                },
                "workers": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot be combined with --num-tasks"):
        cmd_replay(
            argparse.Namespace(
                plan=str(plan_path),
                output_dir=str(tmp_path / "replay-output"),
                num_tasks=1,
                randomize_seed=None,
                time_constraint_s=30.0,
                port_profile_id=1,
                launch_policy_override_json=None,
                vllm_log=None,
                vllm_log_interval_s=1.0,
                vllm_log_timeout_s=5.0,
            )
        )
