from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CON_DRIVER_SRC = PROJECT_ROOT / "con-driver" / "src"
if str(CON_DRIVER_SRC) not in sys.path:
    sys.path.insert(0, str(CON_DRIVER_SRC))

from con_driver import cli
from con_driver.backends.harbor import runtime as harbor_runtime
from con_driver.vllm_metrics_monitor import _build_raw_record


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


def test_main_synthesizes_hosted_vllm_for_mini_swe_agent(
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
                "port_profile_id = 0",
                'agent = "mini-swe-agent"',
                "gateway = false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_probe_single_served_model_name(*, base_url: str, timeout_s: float) -> str:
        assert base_url == "http://127.0.0.1:11451"
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
    assert captured["agent_base_url"] == "http://192.168.5.1:11451/v1"
    assert captured["forwarded_args"] == [
        "--agent",
        "mini-swe-agent",
        "--model",
        "hosted_vllm/Qwen3-Coder-30B-A3B-Instruct",
        "--agent-kwarg",
        "api_base=http://192.168.5.1:11451/v1",
        "--agent-kwarg",
        "base_url=http://192.168.5.1:11451/v1",
        "--agent-kwarg",
        'model_info={"max_input_tokens":262144,"max_output_tokens":262144,"input_cost_per_token":0.0,"output_cost_per_token":0.0}',
    ]
    assert captured["trial_env"] == {
        "ANTHROPIC_BASE_URL": "http://192.168.5.1:11451/v1",
        "OPENAI_BASE_URL": "http://192.168.5.1:11451/v1",
        "LLM_BASE_URL": "http://192.168.5.1:11451/v1",
        "BASE_URL": "http://192.168.5.1:11451/v1",
        "OPENAI_API_BASE": "http://192.168.5.1:11451/v1",
        "HOSTED_VLLM_API_BASE": "http://192.168.5.1:11451/v1",
        "VLLM_API_BASE": "http://192.168.5.1:11451/v1",
    }


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


def test_main_rejects_manual_vllm_log_endpoint(
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
                'vllm_log_endpoint = "http://127.0.0.1:24123/metrics"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == 1
    assert "vllm_log_endpoint" in capsys.readouterr().err


def test_vllm_metrics_monitor_keeps_raw_metrics_text() -> None:
    record = _build_raw_record("# HELP test\nvllm:num_requests_running 3\n")

    assert isinstance(record["timestamp"], int)
    assert isinstance(record["captured_at"], str)
    assert record["content"] == "# HELP test\nvllm:num_requests_running 3\n"
    assert "families" not in record


def test_main_passes_task_subset_and_cli_overrides_config(
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
                "task_subset_start = 10",
                "task_subset_end = 20",
                'results_dir = "out"',
                "gateway = false",
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

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--task-subset-start",
            "30",
            "--task-subset-end",
            "40",
        ]
    )

    assert exit_code == 0
    assert captured["task_subset_start"] == 30
    assert captured["task_subset_end"] == 40


def test_main_cluster_mode_uses_port_profile_list_and_defaults_max_concurrent(
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
                "n_task = 2",
                'results_dir = "out"',
                "sample_without_replacement = true",
                'port_profile_id_list = "2,1"',
                'max_concurrent_list = "3,4"',
                'agent = "terminus-2"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_probe_single_served_model_name(*, base_url: str, timeout_s: float) -> str:
        assert base_url in {"http://127.0.0.1:24123", "http://127.0.0.1:31987"}
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
    assert captured["max_concurrent"] == 7
    assert captured["port_profile_id"] is None
    assert captured["vllm_log_enabled"] is True
    assert captured["vllm_log_endpoint"] == "http://127.0.0.1:24123/metrics"
    launch_profiles = captured["launch_profiles"]
    assert isinstance(launch_profiles, list)
    assert [entry.port_profile_id for entry in launch_profiles] == [1, 2]
    assert [entry.max_concurrent for entry in launch_profiles] == [4, 3]
    assert [entry.vllm_log_endpoint for entry in launch_profiles] == [
        "http://127.0.0.1:24123/metrics",
        "http://127.0.0.1:31987/metrics",
    ]


def test_main_rejects_mixing_port_profile_id_and_port_profile_id_list(
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
                "n_task = 1",
                "max_concurrent = 1",
                'results_dir = "out"',
                "sample_without_replacement = true",
                'port_profile_id_list = "1,2"',
                'max_concurrent_list = "1,1"',
                'agent = "terminus-2"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        harbor_runtime,
        "probe_single_served_model_name",
        lambda **_: "Qwen3-Coder-30B-A3B-Instruct",
    )

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--port-profile-id",
            "3",
        ]
    )

    assert exit_code == 1
    assert "--port-profile-id cannot be used with --port-profile-id-list" in capsys.readouterr().err


def test_main_rejects_port_profile_id_list_without_max_concurrent_list(
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
                "n_task = 1",
                "max_concurrent = 1",
                'results_dir = "out"',
                'port_profile_id_list = "1,2"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == 1
    assert "--max-concurrent-list is required when --port-profile-id-list is set" in capsys.readouterr().err


def test_main_rejects_max_concurrent_list_without_port_profile_id_list(
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
                "n_task = 1",
                "max_concurrent = 1",
                'results_dir = "out"',
                'max_concurrent_list = "1,1"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == 1
    assert "--max-concurrent-list requires --port-profile-id-list" in capsys.readouterr().err
