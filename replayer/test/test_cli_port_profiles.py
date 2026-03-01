from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from replayer.cli import cmd_replay
from replayer.cli import extract_response_text as extract_cli_response_text
from replayer.cli import parse_launch_policy_override_json
from replayer.cli import resolve_replay_launch_policy
from replayer.cli import resolve_compile_target, resolve_replay_target
from replayer.validate import extract_response_text as extract_validate_response_text


def test_resolve_compile_target_uses_runtime_port_profile_id() -> None:
    config = {
        "runtime": {"port_profile_id": "1"},
        "gateway": {"enabled": True, "url": "http://127.0.0.1:9999"},
        "backend": {
            "forwarded_args": [
                "--model",
                "hosted_vllm/Test-Model",
                "--agent-kwarg",
                "api_base=http://127.0.0.1:9999/v1",
            ]
        },
    }

    gateway_url, api_base, configured_model, tokenize_endpoint, port_profile_id = (
        resolve_compile_target(
            config=config,
            results_entries=[],
            port_profile_id_override=None,
            tokenize_endpoint_override=None,
        )
    )

    assert gateway_url == "http://127.0.0.1:24157"
    assert api_base == "http://127.0.0.1:24157/v1"
    assert configured_model == "hosted_vllm/Test-Model"
    assert tokenize_endpoint == "http://127.0.0.1:24123/tokenize"
    assert port_profile_id == 1


def test_resolve_compile_target_override_wins() -> None:
    config = {
        "runtime": {"port_profile_id": "0"},
        "gateway": {"enabled": True, "url": "http://127.0.0.1:9999"},
        "backend": {
            "forwarded_args": [
                "--model",
                "hosted_vllm/Test-Model",
                "--agent-kwarg",
                "api_base=http://127.0.0.1:9999/v1",
            ]
        },
    }

    gateway_url, api_base, _, tokenize_endpoint, port_profile_id = resolve_compile_target(
        config=config,
        results_entries=[],
        port_profile_id_override=1,
        tokenize_endpoint_override=None,
    )

    assert gateway_url == "http://127.0.0.1:24157"
    assert api_base == "http://127.0.0.1:24157/v1"
    assert tokenize_endpoint == "http://127.0.0.1:24123/tokenize"
    assert port_profile_id == 1


def test_resolve_compile_target_uses_direct_vllm_api_base_when_gateway_disabled() -> None:
    config = {
        "runtime": {"port_profile_id": "1"},
        "gateway": {"enabled": False, "url": "http://127.0.0.1:9999"},
        "backend": {
            "forwarded_args": [
                "--model",
                "hosted_vllm/Test-Model",
                "--agent-kwarg",
                "api_base=http://127.0.0.1:9999/v1",
            ]
        },
    }

    gateway_url, api_base, _, tokenize_endpoint, port_profile_id = resolve_compile_target(
        config=config,
        results_entries=[],
        port_profile_id_override=None,
        tokenize_endpoint_override=None,
    )

    assert gateway_url == "http://127.0.0.1:24157"
    assert api_base == "http://127.0.0.1:24123/v1"
    assert tokenize_endpoint == "http://127.0.0.1:24123/tokenize"
    assert port_profile_id == 1


def test_resolve_replay_target_uses_plan_port_profile_id() -> None:
    api_base, gateway_url, tokenize_endpoint, port_profile_id = resolve_replay_target(
        replay_target={
            "port_profile_id": "1",
            "gateway_url": "http://127.0.0.1:9999",
            "api_base": "http://127.0.0.1:9999/v1",
            "tokenize_endpoint": "http://127.0.0.1:9998/tokenize",
        },
        port_profile_id_override=None,
    )

    assert gateway_url == "http://127.0.0.1:24157"
    assert api_base == "http://127.0.0.1:24157/v1"
    assert tokenize_endpoint == "http://127.0.0.1:24123/tokenize"
    assert port_profile_id == "1"


def test_resolve_replay_target_override_preserves_direct_vllm_mode() -> None:
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
    assert api_base == "http://127.0.0.1:24123/v1"
    assert tokenize_endpoint == "http://127.0.0.1:24123/tokenize"
    assert port_profile_id == "1"


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

    assert launch_max_concurrent == 5
    assert launch_pattern_name == "poisson"
    assert launch_pattern_rate_per_second == 2.0
    assert effective_launch_policy["pattern_args"] == {"rate": 2.0}


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


def test_cmd_replay_updates_progress_bar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_path = tmp_path / "replay-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "replay_target": {
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
        timeout_s: float,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, object]:
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
            request_timeout_s=5.0,
            port_profile_id=None,
            gateway_lifecycle="off",
            launch_policy_override_json='{"max_concurrent": 1}',
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
