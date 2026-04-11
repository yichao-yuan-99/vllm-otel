from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PROJECT_ROOT / "freq-controller-linespace-slo-amd"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from freq_controller_linespace_slo_amd import (
    AmdClockConfig,
    FrequencyController,
    FrequencyControllerConfig,
    GatewayContextSnapshot,
    GatewayIPCConfig,
    GatewayOutputThroughputSnapshot,
    choose_next_frequency_index,
    discover_supported_frequency_mhz_levels,
    load_controller_config,
    parse_gpu_index_tokens,
)


class FakeGatewayClient:
    def __init__(
        self,
        context_snapshots: list[GatewayContextSnapshot | Exception],
        *,
        throughput_snapshots: list[GatewayOutputThroughputSnapshot | Exception] | None = None,
    ) -> None:
        self._context_snapshots = list(context_snapshots)
        self._context_index = 0
        self._throughput_snapshots = (
            []
            if throughput_snapshots is None
            else list(throughput_snapshots)
        )
        self._throughput_index = 0

    @staticmethod
    def _next_value(values: list[object], index: int) -> tuple[object, int]:
        if not values:
            raise RuntimeError("no fake gateway values configured")
        if index >= len(values):
            value = values[-1]
        else:
            value = values[index]
            index += 1
        if isinstance(value, Exception):
            raise value
        return value, index

    def read_context_snapshot(self) -> GatewayContextSnapshot:
        value, self._context_index = self._next_value(
            self._context_snapshots,
            self._context_index,
        )
        assert isinstance(value, GatewayContextSnapshot)
        return value

    def read_output_throughput_snapshot(self) -> GatewayOutputThroughputSnapshot:
        value, self._throughput_index = self._next_value(
            self._throughput_snapshots,
            self._throughput_index,
        )
        assert isinstance(value, GatewayOutputThroughputSnapshot)
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


def throughput_snapshot(
    min_output_tokens_per_s: float | None,
    *,
    max_output_tokens_per_s: float | None = None,
    avg_output_tokens_per_s: float | None = None,
    job_active: bool = True,
    agent_count: int = 1,
    throughput_agent_count: int | None = None,
    job_started_at: str | None = "2026-04-01T00:00:00Z",
) -> GatewayOutputThroughputSnapshot:
    if throughput_agent_count is None:
        resolved_throughput_agent_count = 0 if min_output_tokens_per_s is None else 1
    else:
        resolved_throughput_agent_count = throughput_agent_count
    return GatewayOutputThroughputSnapshot(
        job_active=job_active,
        job_started_at=job_started_at if job_active else None,
        agent_count=agent_count,
        throughput_agent_count=resolved_throughput_agent_count,
        min_output_tokens_per_s=min_output_tokens_per_s,
        max_output_tokens_per_s=(
            None
            if min_output_tokens_per_s is None
            else min_output_tokens_per_s
            if max_output_tokens_per_s is None
            else max_output_tokens_per_s
        ),
        avg_output_tokens_per_s=(
            None
            if min_output_tokens_per_s is None
            else min_output_tokens_per_s
            if avg_output_tokens_per_s is None
            else avg_output_tokens_per_s
        ),
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_parse_gpu_index_tokens_accepts_single_and_comma_separated_values() -> None:
    assert parse_gpu_index_tokens(["3"]) == (3,)
    assert parse_gpu_index_tokens(["2,3"]) == (2, 3)


def test_discover_supported_frequency_mhz_levels() -> None:
    backend = FakeAmdSmi()

    levels = discover_supported_frequency_mhz_levels(
        gpu_index=0,
        amdsmi_module=backend,
    )

    assert backend.init_calls == 1
    assert backend.shutdown_calls == 1
    assert levels == (500, 800, 1700)


def test_load_controller_config_applies_linespace_amd_slo_fields_and_defaults(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "frequency_mhz_levels = [1200, 810, 1005]",
                "target_context_usage_threshold = 5000",
                "target_output_throughput_tokens_per_s = 24",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_controller_config(config_path, gpu_index=1)

    assert config.frequency_mhz_levels == (810, 1005, 1200)
    assert config.target_context_usage_threshold == 5000.0
    assert config.target_output_throughput_tokens_per_s == 24.0
    assert config.aggressive_slo_control is False
    assert config.segment_count == 2
    assert config.segment_width_context_usage == 2500.0
    assert config.control_interval_s == 5.0
    assert config.context_query_hz == 5.0
    assert config.amd.resolved_reset_max_frequency_mhz(config.frequency_mhz_levels) == 1200


def test_load_controller_config_supports_shared_defaults_with_cli_slo_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "freq_controller_linespace_slo_amd._load_shared_controller_table",
        lambda path: {
            "frequency_mhz_levels": [500, 800, 1700],
            "threshold": 395784,
            "control_interval_s": 5.0,
            "context_query_hz": 5.0,
        },
    )

    config = load_controller_config(
        None,
        target_output_throughput_tokens_per_s=18.5,
        gpu_index=1,
    )

    assert config.frequency_mhz_levels == (500, 800, 1700)
    assert config.target_context_usage_threshold == 395784.0
    assert config.target_output_throughput_tokens_per_s == 18.5
    assert config.aggressive_slo_control is False


def test_load_controller_config_supports_aggressive_slo_cli_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "freq_controller_linespace_slo_amd._load_shared_controller_table",
        lambda path: {
            "frequency_mhz_levels": [500, 800, 1700],
            "threshold": 395784,
            "control_interval_s": 5.0,
            "context_query_hz": 5.0,
        },
    )

    config = load_controller_config(
        None,
        target_output_throughput_tokens_per_s=18.5,
        aggressive_slo_control=True,
        gpu_index=1,
    )

    assert config.aggressive_slo_control is True


def test_load_controller_config_requires_output_throughput_target() -> None:
    with pytest.raises(
        ValueError,
        match="target_output_throughput_tokens_per_s is required",
    ):
        load_controller_config(None, gpu_index=0)


def test_amd_clock_config_resolves_sibling_command_when_not_on_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    helper_path = bin_dir / "amd-set-gpu-core-freq"
    helper_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    helper_path.chmod(0o755)

    launcher_path = bin_dir / "freq-controller-linespace-slo-amd"
    launcher_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    monkeypatch.setattr("freq_controller_linespace_slo_amd.shutil.which", lambda _: None)
    monkeypatch.setattr(sys, "argv", [str(launcher_path)])

    config = AmdClockConfig(command_path="amd-set-gpu-core-freq")

    assert config.command_path == str(helper_path)


def test_gateway_ipc_config_prefers_gateway_ctx_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "freq_controller_linespace_slo_amd.DEFAULT_GATEWAY_IPC_SOCKET_DIR",
        tmp_path,
    )
    socket_path = tmp_path / "vllm-gateway-ctx-profile-3.sock"
    socket_path.touch()

    resolved = GatewayIPCConfig().resolved_socket_path(3)

    assert resolved == socket_path.resolve()


def test_choose_next_frequency_index_prioritizes_slo_increase() -> None:
    assert choose_next_frequency_index(
        current_index=1,
        moving_average_context_usage=1.0,
        target_context_usage_threshold=40.0,
        max_index=3,
        moving_average_min_output_tokens_per_s=8.0,
        target_output_throughput_tokens_per_s=10.0,
    ) == (2, "increase_for_slo")


def test_choose_next_frequency_index_aggresive_sets_max_for_slo() -> None:
    assert choose_next_frequency_index(
        current_index=1,
        moving_average_context_usage=1.0,
        target_context_usage_threshold=40.0,
        max_index=4,
        moving_average_min_output_tokens_per_s=8.0,
        target_output_throughput_tokens_per_s=10.0,
        aggressive_slo_control=True,
    ) == (4, "increase_for_slo")


def test_frequency_controller_uses_context_linespace_policy_when_slo_met(
    tmp_path: Path,
) -> None:
    config = FrequencyControllerConfig(
        frequency_mhz_levels=(600, 900, 1200, 1500),
        target_context_usage_threshold=30.0,
        target_output_throughput_tokens_per_s=8.0,
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
        ],
        throughput_snapshots=[
            throughput_snapshot(12.0),
            throughput_snapshot(12.0),
            throughput_snapshot(12.0),
        ],
    )
    fake_gpu = FakeGPUController()
    fake_clock = FakeClock()

    controller = FrequencyController(
        config,
        tmp_path,
        port_profile_id=7,
        gpu_index=(2, 3),
        gateway_client=fake_gateway,
        gpu_controller=fake_gpu,
        monotonic=fake_clock.monotonic,
        sleep_func=fake_clock.sleep,
    )

    log_paths = controller.run(max_control_decisions=1)

    assert fake_gpu.set_calls == [1200, 600]
    assert fake_gpu.reset_calls == 1
    assert log_paths.slo_decision_path.exists()

    query_records = read_jsonl(log_paths.query_path)
    decision_records = read_jsonl(log_paths.decision_path)
    slo_decision_records = read_jsonl(log_paths.slo_decision_path)

    assert query_records[0]["phase"] == "pending"
    assert query_records[0]["gpu_indices"] == [2, 3]
    assert [record["min_output_tokens_per_s"] for record in query_records[1:]] == [
        12.0,
        12.0,
        12.0,
    ]
    assert len(decision_records) == 1
    assert decision_records[0]["action"] == "decrease"
    assert decision_records[0]["decision_policy"] == "context_linespace"
    assert decision_records[0]["slo_override_applied"] is False
    assert decision_records[0]["target_frequency_mhz"] == 600
    assert decision_records[0]["context_target_frequency_index"] == 0
    assert decision_records[0]["gpu_indices"] == [2, 3]
    assert slo_decision_records == []


def test_frequency_controller_prioritizes_slo_increase_and_logs_slo_decision(
    tmp_path: Path,
) -> None:
    config = FrequencyControllerConfig(
        frequency_mhz_levels=(600, 900, 1200, 1500),
        target_context_usage_threshold=30.0,
        target_output_throughput_tokens_per_s=8.0,
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
        ],
        throughput_snapshots=[
            throughput_snapshot(4.0),
            throughput_snapshot(4.0),
            throughput_snapshot(4.0),
        ],
    )
    fake_gpu = FakeGPUController()
    fake_clock = FakeClock()

    controller = FrequencyController(
        config,
        tmp_path,
        port_profile_id=7,
        gpu_index=(2, 3),
        gateway_client=fake_gateway,
        gpu_controller=fake_gpu,
        monotonic=fake_clock.monotonic,
        sleep_func=fake_clock.sleep,
    )

    log_paths = controller.run(max_control_decisions=1)

    assert fake_gpu.set_calls == [1200, 1500]
    assert fake_gpu.reset_calls == 1

    decision_records = read_jsonl(log_paths.decision_path)
    slo_decision_records = read_jsonl(log_paths.slo_decision_path)

    assert len(decision_records) == 1
    assert decision_records[0]["action"] == "increase_for_slo"
    assert decision_records[0]["decision_policy"] == "throughput_slo_precedence"
    assert decision_records[0]["slo_override_applied"] is True
    assert decision_records[0]["target_frequency_mhz"] == 1500
    assert decision_records[0]["window_min_output_tokens_per_s"] == 4.0
    assert decision_records[0]["gpu_indices"] == [2, 3]
    assert len(slo_decision_records) == 1
    assert slo_decision_records[0]["action"] == "increase_for_slo"
    assert slo_decision_records[0]["gpu_indices"] == [2, 3]


def test_frequency_controller_aggresive_slo_sets_max_frequency(
    tmp_path: Path,
) -> None:
    config = FrequencyControllerConfig(
        frequency_mhz_levels=(600, 900, 1200, 1500, 1800),
        target_context_usage_threshold=30.0,
        target_output_throughput_tokens_per_s=8.0,
        aggressive_slo_control=True,
        control_interval_s=2.0,
        context_query_hz=1.0,
        gateway=GatewayIPCConfig(),
        amd=AmdClockConfig(
            script_path="/usr/local/bin/set_gpu_clockfreq.sh",
            reset_max_frequency_mhz=1800,
        ),
    )
    fake_gateway = FakeGatewayClient(
        [
            gateway_snapshot(1.0),
            gateway_snapshot(1.0),
            gateway_snapshot(1.0),
            gateway_snapshot(1.0),
        ],
        throughput_snapshots=[
            throughput_snapshot(4.0),
            throughput_snapshot(4.0),
            throughput_snapshot(4.0),
        ],
    )
    fake_gpu = FakeGPUController()
    fake_clock = FakeClock()

    controller = FrequencyController(
        config,
        tmp_path,
        port_profile_id=7,
        gpu_index=(2, 3),
        gateway_client=fake_gateway,
        gpu_controller=fake_gpu,
        monotonic=fake_clock.monotonic,
        sleep_func=fake_clock.sleep,
    )

    log_paths = controller.run(max_control_decisions=1)

    assert fake_gpu.set_calls == [1200, 1800]
    assert fake_gpu.reset_calls == 1

    decision_records = read_jsonl(log_paths.decision_path)
    slo_decision_records = read_jsonl(log_paths.slo_decision_path)

    assert len(decision_records) == 1
    assert decision_records[0]["action"] == "increase_for_slo"
    assert decision_records[0]["decision_policy"] == "throughput_slo_precedence"
    assert decision_records[0]["slo_override_applied"] is True
    assert decision_records[0]["aggressive_slo_control"] is True
    assert decision_records[0]["target_frequency_mhz"] == 1800
    assert decision_records[0]["target_frequency_index"] == 4
    assert decision_records[0]["context_target_frequency_index"] == 0
    assert len(slo_decision_records) == 1
    assert slo_decision_records[0]["aggressive_slo_control"] is True
    assert slo_decision_records[0]["target_frequency_mhz"] == 1800


def test_frequency_controller_logs_control_failures_and_retries(
    tmp_path: Path,
) -> None:
    config = FrequencyControllerConfig(
        frequency_mhz_levels=(600, 900, 1200, 1500),
        target_context_usage_threshold=30.0,
        target_output_throughput_tokens_per_s=8.0,
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
        ],
        throughput_snapshots=[
            throughput_snapshot(12.0),
            throughput_snapshot(12.0),
            throughput_snapshot(12.0),
            throughput_snapshot(12.0),
            throughput_snapshot(12.0),
        ],
    )
    fake_gpu = FakeGPUController()
    fake_gpu.failures_by_frequency[600] = 1
    fake_clock = FakeClock()

    controller = FrequencyController(
        config,
        tmp_path,
        port_profile_id=7,
        gpu_index=(2, 3),
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
    slo_decision_records = read_jsonl(log_paths.slo_decision_path)

    assert len(decision_records) == 2
    assert [record["action"] for record in decision_records] == [
        "decrease",
        "decrease",
    ]
    assert slo_decision_records == []
    assert len(control_error_records) == 1
    assert control_error_records[0]["reason"] == "control_decision"
    assert control_error_records[0]["action"] == "decrease"
    assert control_error_records[0]["error"] == "[Errno 104] Connection reset by peer"
    assert control_error_records[0]["attempted_frequency_index"] == 0
    assert control_error_records[0]["attempted_frequency_mhz"] == 600
    assert control_error_records[0]["current_frequency_index"] == 2
    assert control_error_records[0]["current_frequency_mhz"] == 1200
    assert control_error_records[0]["gpu_indices"] == [2, 3]
