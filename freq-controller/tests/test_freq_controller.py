from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PROJECT_ROOT / "freq-controller"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from freq_controller import (
    FrequencyController,
    GatewayContextSnapshot,
    GatewayIPCConfig,
    FrequencyControllerConfig,
    MovingAverageWindow,
    ZeusdConfig,
    _default_gateway_ipc_socket_path,
    choose_next_frequency_index,
    load_controller_config,
)


class FakeGatewayClient:
    def __init__(self, snapshots: list[GatewayContextSnapshot | Exception]) -> None:
        self._snapshots = list(snapshots)
        self._index = 0

    def read_context_snapshot(self) -> GatewayContextSnapshot:
        if self._index >= len(self._snapshots):
            value = self._snapshots[-1]
        else:
            value = self._snapshots[self._index]
            self._index += 1
        if isinstance(value, Exception):
            raise value
        return value

    def read_total_context_usage(self) -> float:
        return self.read_context_snapshot().total_context_tokens


class FakeGPUController:
    def __init__(self) -> None:
        self.set_calls: list[int] = []
        self.reset_calls = 0

    def set_frequency(self, frequency_mhz: int) -> None:
        self.set_calls.append(frequency_mhz)

    def reset_frequency(self) -> None:
        self.reset_calls += 1


class FakeClock:
    def __init__(self) -> None:
        self.now_s = 0.0

    def monotonic(self) -> float:
        return self.now_s

    def sleep(self, seconds: float) -> None:
        self.now_s += max(0.0, seconds)


DEFAULT_SHARED_FREQUENCY_MHZ_LEVELS = tuple(range(345, 1800, 15))


def gateway_snapshot(
    total_context_tokens: float,
    *,
    job_active: bool = True,
    agent_count: int = 1,
    job_started_at: str | None = "2026-04-01T00:00:00Z",
) -> GatewayContextSnapshot:
    return GatewayContextSnapshot(
        job_active=job_active,
        job_started_at=job_started_at if job_active else None,
        agent_count=agent_count,
        total_context_tokens=total_context_tokens,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_load_controller_config_applies_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "frequency_mhz_levels = [1200, 810, 1005]",
                "target_context_usage_lower_bound = 5000",
                "target_context_usage_upper_bound = 15000",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_controller_config(config_path)
    assert config.frequency_mhz_levels == (810, 1005, 1200)
    assert config.control_interval_s == 5.0
    assert config.context_query_hz == 5.0
    assert config.gateway.resolved_socket_path(0) == Path(
        "/tmp/vllm-gateway-profile-0.sock"
    )
    assert config.zeusd.socket_path == "/var/run/zeusd.sock"


def test_load_controller_config_supports_no_config_with_cli_bounds() -> None:
    config = load_controller_config(
        None,
        target_context_usage_lower_bound=12000,
        target_context_usage_upper_bound=22000,
    )

    assert config.frequency_mhz_levels == DEFAULT_SHARED_FREQUENCY_MHZ_LEVELS
    assert config.control_interval_s == 5.0
    assert config.context_query_hz == 5.0
    assert config.target_context_usage_lower_bound == 12000.0
    assert config.target_context_usage_upper_bound == 22000.0
    assert config.gateway.resolved_socket_path(0) == Path(
        "/tmp/vllm-gateway-profile-0.sock"
    )
    assert config.zeusd.socket_path == "/var/run/zeusd.sock"


def test_load_controller_config_supports_optional_tables(tmp_path: Path) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "[controller]",
                "frequencies_mhz = [600, 900, 1200]",
                "target_context_tokens_lower_bound = 50",
                "target_context_tokens_upper_bound = 150",
                "control_interval = 20",
                "context_query_frequency_hz = 2",
                "",
                "[gateway]",
                'ipc_socket_path = "/tmp/gateway.sock"',
                "",
                "[zeusd]",
                'socket_path = "/tmp/zeusd.sock"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_controller_config(config_path)
    assert config.frequency_mhz_levels == (600, 900, 1200)
    assert config.control_interval_s == 20.0
    assert config.context_query_hz == 2.0
    assert config.gateway.resolved_socket_path(9) == Path("/tmp/gateway.sock")
    assert config.zeusd.socket_path == "/tmp/zeusd.sock"


def test_load_controller_config_allows_cli_bounds_without_controller_table(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "[gateway]",
                'ipc_socket_path = "/tmp/gateway.sock"',
                "",
                "[zeusd]",
                'socket_path = "/tmp/zeusd.sock"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_controller_config(
        config_path,
        target_context_usage_lower_bound=500.0,
        target_context_usage_upper_bound=1500.0,
    )

    assert config.frequency_mhz_levels == DEFAULT_SHARED_FREQUENCY_MHZ_LEVELS
    assert config.target_context_usage_lower_bound == 500.0
    assert config.target_context_usage_upper_bound == 1500.0
    assert config.gateway.resolved_socket_path(2) == Path("/tmp/gateway.sock")
    assert config.zeusd.socket_path == "/tmp/zeusd.sock"


def test_load_controller_config_cli_bounds_override_config_bounds(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "frequency_mhz_levels = [600, 900, 1200]",
                "target_context_usage_lower_bound = 50",
                "target_context_usage_upper_bound = 150",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_controller_config(
        config_path,
        target_context_usage_lower_bound=75.0,
        target_context_usage_upper_bound=175.0,
    )

    assert config.target_context_usage_lower_bound == 75.0
    assert config.target_context_usage_upper_bound == 175.0


def test_load_controller_config_rejects_port_profile_id_in_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "frequency_mhz_levels = [600, 900, 1200]",
                "target_context_usage_lower_bound = 50",
                "target_context_usage_upper_bound = 150",
                "",
                "[gateway]",
                "port_profile_id = 3",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="use --port-profile-id instead",
    ):
        load_controller_config(config_path)


def test_load_controller_config_rejects_gpu_index_in_config(tmp_path: Path) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "frequency_mhz_levels = [600, 900, 1200]",
                "target_context_usage_lower_bound = 50",
                "target_context_usage_upper_bound = 150",
                "",
                "[zeusd]",
                "gpu_index = 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="use --gpu-index instead",
    ):
        load_controller_config(config_path)


def test_load_controller_config_merges_relative_shared_config(tmp_path: Path) -> None:
    shared_path = tmp_path / "script-shared.toml"
    shared_path.write_text(
        "\n".join(
            [
                "frequency_mhz_levels = [600, 900, 1200]",
                "control_interval_s = 45",
                "context_query_hz = 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "[shared]",
                'config_path = "./script-shared.toml"',
                "",
                "[controller]",
                "target_context_usage_lower_bound = 100",
                "target_context_usage_upper_bound = 200",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_controller_config(config_path)
    assert config.frequency_mhz_levels == (600, 900, 1200)
    assert config.control_interval_s == 45.0
    assert config.context_query_hz == 3.0
    assert config.target_context_usage_lower_bound == 100.0
    assert config.target_context_usage_upper_bound == 200.0


def test_default_gateway_ipc_socket_path_uses_profile_id() -> None:
    assert _default_gateway_ipc_socket_path(0) == Path(
        "/tmp/vllm-gateway-profile-0.sock"
    )
    assert _default_gateway_ipc_socket_path(5) == Path(
        "/tmp/vllm-gateway-profile-5.sock"
    )


def test_moving_average_window_prunes_old_samples() -> None:
    window = MovingAverageWindow(10.0)
    window.add(0.0, 10.0)
    window.add(5.0, 30.0)
    window.add(12.0, 50.0)

    assert window.average(12.0) == pytest.approx(40.0)
    assert window.sample_count == 2


def test_choose_next_frequency_index_respects_bounds_and_clamps() -> None:
    assert choose_next_frequency_index(
        current_index=1,
        moving_average_context_usage=20.0,
        lower_bound=50.0,
        upper_bound=150.0,
        max_index=2,
    ) == (0, "decrease")
    assert choose_next_frequency_index(
        current_index=1,
        moving_average_context_usage=200.0,
        lower_bound=50.0,
        upper_bound=150.0,
        max_index=2,
    ) == (2, "increase")
    assert choose_next_frequency_index(
        current_index=0,
        moving_average_context_usage=10.0,
        lower_bound=50.0,
        upper_bound=150.0,
        max_index=2,
    ) == (0, "decrease")
    assert choose_next_frequency_index(
        current_index=1,
        moving_average_context_usage=100.0,
        lower_bound=50.0,
        upper_bound=150.0,
        max_index=2,
    ) == (1, "hold")


def test_frequency_controller_runs_policy_and_resets_on_exit(tmp_path: Path) -> None:
    config = FrequencyControllerConfig(
        frequency_mhz_levels=(600, 900, 1200),
        target_context_usage_lower_bound=60.0,
        target_context_usage_upper_bound=140.0,
        control_interval_s=2.0,
        context_query_hz=1.0,
        gateway=GatewayIPCConfig(),
        zeusd=ZeusdConfig(socket_path="/tmp/zeusd.sock"),
    )
    fake_gateway = FakeGatewayClient(
        [
            gateway_snapshot(50.0),
            gateway_snapshot(50.0),
            gateway_snapshot(50.0),
            gateway_snapshot(50.0),
            gateway_snapshot(200.0),
            gateway_snapshot(200.0),
        ]
    )
    fake_gpu = FakeGPUController()
    fake_clock = FakeClock()

    controller = FrequencyController(
        config,
        tmp_path,
        port_profile_id=7,
        gpu_index=3,
        gateway_client=fake_gateway,
        gpu_controller=fake_gpu,
        monotonic=fake_clock.monotonic,
        sleep_func=fake_clock.sleep,
    )

    log_paths = controller.run(max_control_decisions=2)

    assert fake_gpu.set_calls == [900, 600, 900]
    assert fake_gpu.reset_calls == 1
    assert log_paths.query_path.exists()
    assert log_paths.decision_path.exists()
    assert ".query." in log_paths.query_path.name
    assert ".decision." in log_paths.decision_path.name

    query_records = read_jsonl(log_paths.query_path)
    decision_records = read_jsonl(log_paths.decision_path)

    assert query_records[0]["phase"] == "pending"
    assert query_records[0]["job_active"] is True
    assert query_records[0]["context_usage"] == 50.0
    assert [record["phase"] for record in query_records[1:]] == [
        "active",
        "active",
        "active",
        "active",
        "active",
    ]
    assert [record["context_usage"] for record in query_records[1:]] == [
        50.0,
        50.0,
        50.0,
        200.0,
        200.0,
    ]

    assert len(decision_records) == 2
    assert decision_records[0]["action"] == "decrease"
    assert decision_records[0]["changed"] is True
    assert decision_records[0]["current_frequency_mhz"] == 900
    assert decision_records[0]["target_frequency_mhz"] == 600
    assert decision_records[0]["window_context_usage"] == pytest.approx(50.0)
    assert decision_records[0]["sample_count"] == 3
    assert decision_records[0]["lower_bound"] == 60.0
    assert decision_records[0]["upper_bound"] == 140.0

    assert decision_records[1]["action"] == "increase"
    assert decision_records[1]["changed"] is True
    assert decision_records[1]["current_frequency_mhz"] == 600
    assert decision_records[1]["target_frequency_mhz"] == 900
    assert decision_records[1]["window_context_usage"] == pytest.approx(150.0)
    assert decision_records[1]["sample_count"] == 3


def test_frequency_controller_waits_for_job_start_before_control(
    tmp_path: Path,
) -> None:
    config = FrequencyControllerConfig(
        frequency_mhz_levels=(600, 900, 1200),
        target_context_usage_lower_bound=60.0,
        target_context_usage_upper_bound=140.0,
        control_interval_s=1.0,
        context_query_hz=1.0,
        gateway=GatewayIPCConfig(),
        zeusd=ZeusdConfig(socket_path="/tmp/zeusd.sock"),
    )
    fake_gateway = FakeGatewayClient(
        [
            gateway_snapshot(0.0, job_active=False, agent_count=0, job_started_at=None),
            gateway_snapshot(0.0, job_active=False, agent_count=0, job_started_at=None),
            gateway_snapshot(100.0, job_active=True, agent_count=1),
            gateway_snapshot(100.0, job_active=True, agent_count=1),
            gateway_snapshot(100.0, job_active=True, agent_count=1),
        ]
    )
    fake_gpu = FakeGPUController()
    fake_clock = FakeClock()

    controller = FrequencyController(
        config,
        tmp_path,
        gateway_client=fake_gateway,
        gpu_controller=fake_gpu,
        monotonic=fake_clock.monotonic,
        sleep_func=fake_clock.sleep,
    )

    log_paths = controller.run(max_control_decisions=1)

    assert fake_clock.now_s >= 2.0
    assert fake_gpu.set_calls == [900]
    assert fake_gpu.reset_calls == 1

    query_records = read_jsonl(log_paths.query_path)
    decision_records = read_jsonl(log_paths.decision_path)

    assert [record["phase"] for record in query_records[:3]] == [
        "pending",
        "pending",
        "pending",
    ]
    assert [record["job_active"] for record in query_records[:3]] == [
        False,
        False,
        True,
    ]
    assert [record["context_usage"] for record in query_records[:3]] == [
        0.0,
        0.0,
        100.0,
    ]
    assert [record["phase"] for record in query_records[3:]] == [
        "active",
        "active",
    ]
    assert [record["context_usage"] for record in query_records[3:]] == [
        100.0,
        100.0,
    ]

    assert len(decision_records) == 1
    assert decision_records[0]["action"] == "hold"
    assert decision_records[0]["changed"] is False
    assert decision_records[0]["current_frequency_mhz"] == 900
    assert decision_records[0]["target_frequency_mhz"] == 900
    assert decision_records[0]["window_context_usage"] == pytest.approx(100.0)
    assert decision_records[0]["sample_count"] == 2


def test_frequency_controller_reuses_last_snapshot_on_active_read_error(
    tmp_path: Path,
) -> None:
    config = FrequencyControllerConfig(
        frequency_mhz_levels=(600, 900, 1200),
        target_context_usage_lower_bound=60.0,
        target_context_usage_upper_bound=140.0,
        control_interval_s=2.0,
        context_query_hz=1.0,
        gateway=GatewayIPCConfig(),
        zeusd=ZeusdConfig(socket_path="/tmp/zeusd.sock"),
    )
    fake_gateway = FakeGatewayClient(
        [
            gateway_snapshot(100.0),
            gateway_snapshot(100.0),
            ConnectionResetError(104, "Connection reset by peer"),
            gateway_snapshot(200.0),
        ]
    )
    fake_gpu = FakeGPUController()
    fake_clock = FakeClock()

    controller = FrequencyController(
        config,
        tmp_path,
        gateway_client=fake_gateway,
        gpu_controller=fake_gpu,
        monotonic=fake_clock.monotonic,
        sleep_func=fake_clock.sleep,
    )

    log_paths = controller.run(max_control_decisions=1)

    assert fake_gpu.set_calls == [900]
    assert fake_gpu.reset_calls == 1

    query_records = read_jsonl(log_paths.query_path)
    decision_records = read_jsonl(log_paths.decision_path)

    assert [record["phase"] for record in query_records] == [
        "pending",
        "active",
        "active",
        "active",
    ]
    assert [record["context_usage"] for record in query_records] == [
        100.0,
        100.0,
        100.0,
        200.0,
    ]
    assert query_records[1].get("error") is None
    assert query_records[2]["error"] == "[Errno 104] Connection reset by peer"
    assert query_records[2]["job_active"] is True
    assert query_records[2]["agent_count"] == 1
    assert query_records[2]["sample_count_window"] == 2

    assert len(decision_records) == 1
    assert decision_records[0]["action"] == "hold"
    assert decision_records[0]["changed"] is False
    assert decision_records[0]["window_context_usage"] == pytest.approx(133.33333333333334)
    assert decision_records[0]["sample_count"] == 3
