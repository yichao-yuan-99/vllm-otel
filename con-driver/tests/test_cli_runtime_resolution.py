from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CON_DRIVER_SRC = PROJECT_ROOT / "con-driver" / "src"
if str(CON_DRIVER_SRC) not in sys.path:
    sys.path.insert(0, str(CON_DRIVER_SRC))

from con_driver import cli
from con_driver.backends.harbor import runtime as harbor_runtime


def test_main_synthesizes_runtime_from_port_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[driver]",
                'driver_backend = "harbor"',
                'pool = "terminal-bench@2.0"',
                'pattern = "eager"',
                "max_concurrent = 1",
                "n_task = 1",
                'results_dir = "out"',
                "port_profile_id = 1",
                'agent = "terminus-2"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_probe_single_served_model_name(*, base_url: str, timeout_s: float) -> str:
        assert base_url == "http://127.0.0.1:24123"
        assert timeout_s == 3600.0
        return "Qwen3-Coder-30B-A3B-Instruct"

    def fake_run_driver(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(
        harbor_runtime,
        "probe_single_served_model_name",
        fake_probe_single_served_model_name,
    )
    monkeypatch.setattr(cli, "_run_driver", fake_run_driver)

    exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == 0
    assert captured["gateway_url"] == "http://127.0.0.1:28171"
    assert captured["vllm_log_enabled"] is True
    assert captured["vllm_log_endpoint"] == "http://127.0.0.1:24123/metrics"
    assert captured["resolved_agent_name"] == "terminus-2"
    assert captured["resolved_model_name"] == "Qwen3-Coder-30B-A3B-Instruct"
    assert captured["resolved_model_context_window"] == 262144
    assert captured["agent_base_url"] == "http://127.0.0.1:28171/v1"
    assert captured["trial_env"] == {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:28171/v1",
        "OPENAI_BASE_URL": "http://127.0.0.1:28171/v1",
        "LLM_BASE_URL": "http://127.0.0.1:28171/v1",
        "BASE_URL": "http://127.0.0.1:28171/v1",
        "OPENAI_API_BASE": "http://127.0.0.1:28171/v1",
        "HOSTED_VLLM_API_BASE": "http://127.0.0.1:28171/v1",
        "VLLM_API_BASE": "http://127.0.0.1:28171/v1",
    }
    assert captured["forwarded_args"] == [
        "--agent",
        "terminus-2",
        "--model",
        "hosted_vllm/Qwen3-Coder-30B-A3B-Instruct",
        "--agent-kwarg",
        "api_base=http://127.0.0.1:28171/v1",
        "--agent-kwarg",
        "base_url=http://127.0.0.1:28171/v1",
        "--agent-kwarg",
        'model_info={"max_input_tokens":262144,"max_output_tokens":262144,"input_cost_per_token":0.0,"output_cost_per_token":0.0}',
        "--agent-kwarg",
        'trajectory_config={"linear_history":true}',
    ]


def test_main_rejects_managed_model_override_in_simplified_mode(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[driver]",
                'driver_backend = "harbor"',
                'pool = "terminal-bench@2.0"',
                'pattern = "eager"',
                "max_concurrent = 1",
                "n_task = 1",
                'results_dir = "out"',
                "port_profile_id = 1",
                'agent = "terminus-2"',
                'forwarded_args = ["--model", "hosted_vllm/manual"]',
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        harbor_runtime,
        "probe_single_served_model_name",
        lambda **_: (_ for _ in ()).throw(AssertionError("probe should not run")),
    )

    exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == 1
    assert "Do not pass '--model'" in capsys.readouterr().err


def test_main_preserves_manual_forwarded_args_without_port_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[driver]",
                'driver_backend = "harbor"',
                'pool = "terminal-bench@2.0"',
                'pattern = "eager"',
                "max_concurrent = 1",
                "n_task = 1",
                'results_dir = "out"',
                'gateway_url = "http://127.0.0.1:11457"',
                'forwarded_args = ["--agent", "terminus-2", "--model", "hosted_vllm/manual"]',
                "",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run_driver(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "_run_driver", fake_run_driver)

    exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == 0
    assert captured["forwarded_args"] == [
        "--agent",
        "terminus-2",
        "--model",
        "hosted_vllm/manual",
    ]
    assert captured["trial_env"] == {}
    assert captured["resolved_agent_name"] == "terminus-2"
    assert captured["resolved_model_name"] is None
    assert captured["vllm_log_enabled"] is False
