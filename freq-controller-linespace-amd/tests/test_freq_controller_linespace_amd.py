from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PROJECT_ROOT / "freq-controller-linespace-amd"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from freq_controller_linespace_amd import (
    AmdClockConfig,
    AmdMaxGPUFrequencyController,
    FrequencyController,
    FrequencyControllerConfig,
    GatewayContextSnapshot,
    GatewayIPCConfig,
    choose_next_frequency_index,
    choose_target_frequency_index,
    discover_supported_frequency_mhz_levels,
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


class FakeGPUController:
    def __init__(self) -> None:
        self.set_calls: list[int] = []
        self.reset_calls = 0
        self.failures_by_frequency: dict[int, int] = {}

    def set_frequency(self, frequency_mhz: int) -> None:
        self.set_calls.append(frequency_mhz)
        remaining_failures = self.failures_by_frequency.get(frequency_mhz, 0)
        if remaining_failures > 0:
            self.failures_by_frequency[frequency_mhz] = remaining_failures - 1
            raise ConnectionResetError(104, "Connection reset by peer")

    def reset_frequency(self) -> None:
        self.reset_calls += 1


class FakeClock:
    def __init__(self) -> None:
        self.now_s = 0.0

    def monotonic(self) -> float:
        return self.now_s

    def sleep(self, seconds: float) -> None:
        self.now_s += max(0.0, seconds)


class FakeClkType:
    GFX = "gfx"


class FakeAmdSmi:
    AmdSmiClkType = FakeClkType

    def __init__(self) -> None:
        self.init_calls = 0
        self.shutdown_calls = 0
        self.handles = ["gpu0", "gpu1"]

    def amdsmi_init(self) -> None:
        self.init_calls += 1

    def amdsmi_shut_down(self) -> None:
        self.shutdown_calls += 1

    def amdsmi_get_processor_handles(self) -> list[str]:
        return list(self.handles)

    def amdsmi_get_clk_freq(self, handle: str, clock_type: str) -> dict[str, object]:
        assert clock_type == "gfx"
        return {
            "gpu0": {"frequency": [500_000_000, 800_000_000, 1_700_000_000]},
            "gpu1": {"frequency": [500_000_000, 670_000_000]},
        }[handle]


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


def test_discover_supported_frequency_mhz_levels() -> None:
    backend = FakeAmdSmi()

    levels = discover_supported_frequency_mhz_levels(
        gpu_index=0,
        amdsmi_module=backend,
    )

    assert backend.init_calls == 1
    assert backend.shutdown_calls == 1
    assert levels == (500, 800, 1700)


def test_load_controller_config_applies_linespace_fields_and_defaults(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "frequency_mhz_levels = [1200, 810, 1005]",
                "target_context_usage_threshold = 5000",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_controller_config(config_path, gpu_index=1)

    assert config.frequency_mhz_levels == (810, 1005, 1200)
    assert config.target_context_usage_threshold == 5000.0
    assert config.segment_count == 2
    assert config.segment_width_context_usage == 2500.0
    assert config.control_interval_s == 5.0
    assert config.context_query_hz == 5.0
    assert config.amd.resolved_reset_max_frequency_mhz(config.frequency_mhz_levels) == 1200


def test_load_controller_config_discovers_frequency_levels_without_config(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "freq_controller_linespace_amd._load_shared_controller_table",
        lambda path: {
            "control_interval_s": 5.0,
            "context_query_hz": 5.0,
        },
    )
    monkeypatch.setattr(
        "freq_controller_linespace_amd.discover_supported_frequency_mhz_levels",
        lambda *, gpu_index: (500, 800, 1700),
    )

    config = load_controller_config(
        None,
        target_context_usage_threshold=12000,
        gpu_index=2,
    )

    assert config.frequency_mhz_levels == (500, 800, 1700)
    assert config.target_context_usage_threshold == 12000.0


def test_load_controller_config_uses_shared_threshold_default(monkeypatch) -> None:
    monkeypatch.setattr(
        "freq_controller_linespace_amd._load_shared_controller_table",
        lambda path: {
            "threshold": 395784,
            "control_interval_s": 5.0,
            "context_query_hz": 5.0,
        },
    )
    monkeypatch.setattr(
        "freq_controller_linespace_amd.discover_supported_frequency_mhz_levels",
        lambda *, gpu_index: (500, 800, 1700),
    )

    config = load_controller_config(None, gpu_index=1)

    assert config.frequency_mhz_levels == (500, 800, 1700)
    assert config.target_context_usage_threshold == 395784.0


def test_gateway_ipc_config_prefers_gateway_ctx_socket(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "freq_controller_linespace_amd.DEFAULT_GATEWAY_IPC_SOCKET_DIR",
        tmp_path,
    )
    socket_path = tmp_path / "vllm-gateway-ctx-profile-3.sock"
    socket_path.touch()

    resolved = GatewayIPCConfig().resolved_socket_path(3)

    assert resolved == socket_path.resolve()


def test_gateway_ipc_config_falls_back_to_legacy_socket(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "freq_controller_linespace_amd.DEFAULT_GATEWAY_IPC_SOCKET_DIR",
        tmp_path,
    )
    socket_path = tmp_path / "vllm-gateway-profile-4.sock"
    socket_path.touch()

    resolved = GatewayIPCConfig().resolved_socket_path(4)

    assert resolved == socket_path.resolve()


def test_gateway_ipc_config_defaults_to_gateway_ctx_socket_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "freq_controller_linespace_amd.DEFAULT_GATEWAY_IPC_SOCKET_DIR",
        tmp_path,
    )

    resolved = GatewayIPCConfig().resolved_socket_path(5)

    assert resolved == (tmp_path / "vllm-gateway-ctx-profile-5.sock").resolve()


def test_load_controller_config_rejects_non_positive_threshold(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "frequency_mhz_levels = [600, 900, 1200]",
                "threshold = 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="target_context_usage_threshold must be > 0",
    ):
        load_controller_config(config_path, gpu_index=0)


def test_choose_target_frequency_index_uses_linear_segments() -> None:
    assert choose_target_frequency_index(
        moving_average_context_usage=-5.0,
        target_context_usage_threshold=40.0,
        max_index=3,
    ) == 0
    assert choose_target_frequency_index(
        moving_average_context_usage=0.0,
        target_context_usage_threshold=40.0,
        max_index=3,
    ) == 0
    assert choose_target_frequency_index(
        moving_average_context_usage=5.0,
        target_context_usage_threshold=40.0,
        max_index=3,
    ) == 0
    assert choose_target_frequency_index(
        moving_average_context_usage=15.0,
        target_context_usage_threshold=40.0,
        max_index=3,
    ) == 1
    assert choose_target_frequency_index(
        moving_average_context_usage=30.0,
        target_context_usage_threshold=40.0,
        max_index=3,
    ) == 2
    assert choose_target_frequency_index(
        moving_average_context_usage=39.999,
        target_context_usage_threshold=40.0,
        max_index=3,
    ) == 2
    assert choose_target_frequency_index(
        moving_average_context_usage=40.0,
        target_context_usage_threshold=40.0,
        max_index=3,
    ) == 3


def test_choose_next_frequency_index_moves_directly_to_target_segment() -> None:
    assert choose_next_frequency_index(
        current_index=2,
        moving_average_context_usage=1.0,
        target_context_usage_threshold=40.0,
        max_index=3,
    ) == (0, "decrease")
    assert choose_next_frequency_index(
        current_index=0,
        moving_average_context_usage=40.0,
        target_context_usage_threshold=40.0,
        max_index=3,
    ) == (3, "increase")
    assert choose_next_frequency_index(
        current_index=1,
        moving_average_context_usage=15.0,
        target_context_usage_threshold=40.0,
        max_index=3,
    ) == (1, "hold")


def test_amd_max_gpu_frequency_controller_updates_only_max(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert check is False
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("freq_controller_linespace_amd.subprocess.run", fake_run)

    controller = AmdMaxGPUFrequencyController(
        command_path="amd-set-gpu-core-freq",
        gpu_index=1,
        script_path="/usr/local/bin/set_gpu_clockfreq.sh",
        reset_max_frequency_mhz=1700,
    )
    controller.set_frequency(1200)
    controller.reset_frequency()

    assert calls == [
        [
            "amd-set-gpu-core-freq",
            "--gpu-index",
            "1",
            "--max-mhz",
            "1200",
            "--script-path",
            "/usr/local/bin/set_gpu_clockfreq.sh",
        ],
        [
            "amd-set-gpu-core-freq",
            "--gpu-index",
            "1",
            "--max-mhz",
            "1700",
            "--script-path",
            "/usr/local/bin/set_gpu_clockfreq.sh",
        ],
    ]


def test_frequency_controller_uses_linespace_policy_and_resets_on_exit(
    tmp_path: Path,
) -> None:
    config = FrequencyControllerConfig(
        frequency_mhz_levels=(600, 900, 1200, 1500),
        target_context_usage_threshold=30.0,
        control_interval_s=2.0,
        context_query_hz=1.0,
        gateway=GatewayIPCConfig(),
        amd=AmdClockConfig(
            script_path="/usr/local/bin/set_gpu_clockfreq.sh",
            reset_max_frequency_mhz=1500,
        ),
    )
    fake_gateway = FakeGatewayClient(
        [
            gateway_snapshot(1.0),
            gateway_snapshot(1.0),
            gateway_snapshot(1.0),
            gateway_snapshot(1.0),
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

    log_paths = controller.run(max_control_decisions=1)

    assert fake_gpu.set_calls == [1200, 600]
    assert fake_gpu.reset_calls == 1
    assert log_paths.query_path.exists()
    assert log_paths.decision_path.exists()
    assert log_paths.control_error_path.exists()
    assert log_paths.query_path.name.startswith("freq-controller-ls-amd.query.")
    assert log_paths.decision_path.name.startswith("freq-controller-ls-amd.decision.")
    assert log_paths.control_error_path.name.startswith(
        "freq-controller-ls-amd.control-error."
    )

    query_records = read_jsonl(log_paths.query_path)
    decision_records = read_jsonl(log_paths.decision_path)
    control_error_records = read_jsonl(log_paths.control_error_path)

    assert query_records[0]["phase"] == "pending"
    assert [record["context_usage"] for record in query_records[1:]] == [
        1.0,
        1.0,
        1.0,
    ]

    assert len(decision_records) == 1
    assert decision_records[0]["action"] == "decrease"
    assert decision_records[0]["changed"] is True
    assert decision_records[0]["current_frequency_mhz"] == 1200
    assert decision_records[0]["target_frequency_mhz"] == 600
    assert decision_records[0]["target_context_usage_threshold"] == 30.0
    assert decision_records[0]["segment_count"] == 3
    assert decision_records[0]["segment_width_context_usage"] == 10.0
    assert decision_records[0]["target_frequency_index"] == 0
    assert control_error_records == []


def test_frequency_controller_logs_control_failures_and_retries(
    tmp_path: Path,
) -> None:
    config = FrequencyControllerConfig(
        frequency_mhz_levels=(600, 900, 1200, 1500),
        target_context_usage_threshold=30.0,
        control_interval_s=2.0,
        context_query_hz=1.0,
        gateway=GatewayIPCConfig(),
        amd=AmdClockConfig(
            script_path="/usr/local/bin/set_gpu_clockfreq.sh",
            reset_max_frequency_mhz=1500,
        ),
    )
    fake_gateway = FakeGatewayClient(
        [
            gateway_snapshot(1.0),
            gateway_snapshot(1.0),
            gateway_snapshot(1.0),
            gateway_snapshot(1.0),
            gateway_snapshot(1.0),
            gateway_snapshot(1.0),
        ]
    )
    fake_gpu = FakeGPUController()
    fake_gpu.failures_by_frequency[600] = 1
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

    assert fake_gpu.set_calls == [1200, 600, 600]
    assert fake_gpu.reset_calls == 1

    decision_records = read_jsonl(log_paths.decision_path)
    control_error_records = read_jsonl(log_paths.control_error_path)

    assert len(decision_records) == 2
    assert [record["action"] for record in decision_records] == [
        "decrease",
        "decrease",
    ]
    assert len(control_error_records) == 1
    assert control_error_records[0]["reason"] == "control_decision"
    assert control_error_records[0]["action"] == "decrease"
    assert control_error_records[0]["error"] == "[Errno 104] Connection reset by peer"
    assert control_error_records[0]["attempted_frequency_index"] == 0
    assert control_error_records[0]["attempted_frequency_mhz"] == 600
    assert control_error_records[0]["current_frequency_index"] == 2
    assert control_error_records[0]["current_frequency_mhz"] == 1200
