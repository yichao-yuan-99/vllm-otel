from __future__ import annotations

import json
from pathlib import Path
import socket
import sys
from typing import TypeVar

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PROJECT_ROOT / "freq-controller-linespace-instance-slo"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import freq_controller_linespace_instance_slo as controller_module

from freq_controller_linespace_instance_slo import (
    FrequencyController,
    FrequencyControllerConfig,
    FrequencyControllerLogPaths,
    GatewayContextSnapshot,
    GatewayIPCConfig,
    GatewayOutputThroughputSnapshot,
    ZeusdConfig,
    ZeusdGPUFrequencyController,
    choose_next_frequency_index,
    choose_target_frequency_index,
    load_controller_config,
    parse_gpu_index_tokens,
)

SnapshotT = TypeVar("SnapshotT")


class FakeGatewayClient:
    def __init__(
        self,
        context_snapshots: list[GatewayContextSnapshot | Exception],
        *,
        throughput_snapshots: list[GatewayOutputThroughputSnapshot | Exception]
        | None = None,
    ) -> None:
        self._context_snapshots = list(context_snapshots)
        self._context_index = 0
        self._throughput_snapshots = (
            [] if throughput_snapshots is None else list(throughput_snapshots)
        )
        self._throughput_index = 0

    @staticmethod
    def _next_value(
        values: list[SnapshotT | Exception],
        index: int,
    ) -> tuple[SnapshotT, int]:
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
        return value

    def read_output_throughput_snapshot(self) -> GatewayOutputThroughputSnapshot:
        value, self._throughput_index = self._next_value(
            self._throughput_snapshots,
            self._throughput_index,
        )
        return value

    def read_total_context_usage(self) -> float:
        return self.read_context_snapshot().total_context_tokens


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


class FakeZeusBackend:
    def __init__(self) -> None:
        self.set_calls: list[tuple[int, int, int]] = []
        self.reset_calls: list[int] = []

    def set_gpu_locked_clocks(
        self,
        *,
        gpu_index: int,
        min_clock_mhz: int,
        max_clock_mhz: int,
    ) -> None:
        self.set_calls.append((gpu_index, min_clock_mhz, max_clock_mhz))

    def reset_gpu_locked_clocks(self, *, gpu_index: int) -> None:
        self.reset_calls.append(gpu_index)


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


def test_parse_gpu_index_tokens_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate value: 3"):
        parse_gpu_index_tokens(["2,3,3"])


def test_load_controller_config_applies_instance_slo_fields_and_defaults(
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

    config = load_controller_config(config_path)

    assert config.frequency_mhz_levels == (810, 1005, 1200)
    assert config.target_context_usage_threshold == 5000.0
    assert config.target_output_throughput_tokens_per_s == 24.0
    assert config.segment_count == 2
    assert config.segment_width_context_usage == 2500.0
    assert config.control_interval_s == 5.0
    assert config.context_query_hz == 5.0


def test_load_controller_config_supports_no_config_with_cli_override() -> None:
    config = load_controller_config(
        None,
        target_context_usage_threshold=12000,
        target_output_throughput_tokens_per_s=18.5,
    )

    assert config.frequency_mhz_levels == DEFAULT_SHARED_FREQUENCY_MHZ_LEVELS
    assert config.target_context_usage_threshold == 12000.0
    assert config.target_output_throughput_tokens_per_s == 18.5


def test_load_controller_config_filters_frequency_list_geq() -> None:
    config = load_controller_config(
        None,
        target_output_throughput_tokens_per_s=18.5,
        only_freq_list_geq=915,
    )

    assert config.frequency_mhz_levels == tuple(
        frequency_mhz
        for frequency_mhz in DEFAULT_SHARED_FREQUENCY_MHZ_LEVELS
        if frequency_mhz >= 915
    )


def test_load_controller_config_rejects_empty_frequency_list_after_geq_filter() -> None:
    with pytest.raises(
        ValueError,
        match="--only-freq-list-geq filtered frequency_mhz_levels to an empty list",
    ):
        load_controller_config(
            None,
            target_output_throughput_tokens_per_s=18.5,
            only_freq_list_geq=999999,
        )


def test_load_controller_config_uses_shared_context_threshold_default() -> None:
    config = load_controller_config(
        None,
        target_output_throughput_tokens_per_s=11.0,
    )

    assert config.frequency_mhz_levels == DEFAULT_SHARED_FREQUENCY_MHZ_LEVELS
    assert config.target_context_usage_threshold == 395784.0
    assert config.target_output_throughput_tokens_per_s == 11.0


def test_gateway_ipc_config_prefers_active_gateway_ctx_socket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        controller_module,
        "DEFAULT_GATEWAY_IPC_SOCKET_DIR",
        tmp_path,
    )
    gateway_ctx_socket_path = tmp_path / "vllm-gateway-ctx-profile-2.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(gateway_ctx_socket_path))
    listener.listen(1)
    try:
        assert (
            GatewayIPCConfig().resolved_socket_path(2)
            == gateway_ctx_socket_path.resolve()
        )
    finally:
        listener.close()
        if gateway_ctx_socket_path.exists():
            gateway_ctx_socket_path.unlink()


def test_load_controller_config_requires_output_throughput_target() -> None:
    with pytest.raises(
        ValueError,
        match="target_output_throughput_tokens_per_s is required",
    ):
        load_controller_config(None)


def test_load_controller_config_rejects_non_positive_threshold(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "frequency_mhz_levels = [600, 900, 1200]",
                "threshold = 0",
                "target_output_throughput_tokens_per_s = 12",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="target_context_usage_threshold must be > 0",
    ):
        load_controller_config(config_path)


def test_load_controller_config_rejects_gpu_targeting_in_toml(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "frequency_mhz_levels = [600, 900, 1200]",
                "target_context_usage_threshold = 30",
                "target_output_throughput_tokens_per_s = 12",
                "[zeusd]",
                "gpu_indices = [2, 3]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="use --gpu-index instead"):
        load_controller_config(config_path)


def test_main_passes_slo_and_only_freq_list_geq_to_run_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fake_log_paths = FrequencyControllerLogPaths(
        query_path=tmp_path / "query.jsonl",
        decision_path=tmp_path / "decision.jsonl",
        control_error_path=tmp_path / "control-error.jsonl",
        slo_decision_path=tmp_path / "slo-decision.jsonl",
    )

    def fake_run_controller(
        config_path: Path | None,
        log_dir: Path,
        *,
        target_context_usage_threshold: float | None = None,
        target_output_throughput_tokens_per_s: float | None = None,
        only_freq_list_geq: int | None = None,
        gateway_ipc_socket_path: Path | None = None,
        port_profile_id: int = controller_module.DEFAULT_GATEWAY_PORT_PROFILE_ID,
        gpu_index: int | tuple[int, ...] | str = controller_module.DEFAULT_GPU_INDEX,
    ) -> controller_module.FrequencyControllerLogPaths:
        captured["config_path"] = config_path
        captured["log_dir"] = log_dir
        captured["target_context_usage_threshold"] = target_context_usage_threshold
        captured["target_output_throughput_tokens_per_s"] = (
            target_output_throughput_tokens_per_s
        )
        captured["only_freq_list_geq"] = only_freq_list_geq
        captured["gateway_ipc_socket_path"] = gateway_ipc_socket_path
        captured["port_profile_id"] = port_profile_id
        captured["gpu_index"] = gpu_index
        return fake_log_paths

    monkeypatch.setattr(controller_module, "run_controller", fake_run_controller)

    exit_code = controller_module.main(
        [
            "--log-dir",
            str(tmp_path),
            "--threshold",
            "395784",
            "--throughput-target",
            "12",
            "--only-freq-list-geq",
            "915",
            "--gpu-index",
            "2,3",
        ]
    )

    assert exit_code == 0
    assert captured["config_path"] is None
    assert captured["log_dir"] == tmp_path
    assert captured["target_context_usage_threshold"] == 395784.0
    assert captured["target_output_throughput_tokens_per_s"] == 12.0
    assert captured["only_freq_list_geq"] == 915
    assert captured["gateway_ipc_socket_path"] is None
    assert captured["port_profile_id"] == 0
    assert captured["gpu_index"] == (2, 3)


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


def test_choose_next_frequency_index_moves_directly_to_target_segment_when_slo_met() -> None:
    assert choose_next_frequency_index(
        current_index=2,
        moving_average_context_usage=1.0,
        target_context_usage_threshold=40.0,
        max_index=3,
        moving_average_min_output_tokens_per_s=12.0,
        target_output_throughput_tokens_per_s=10.0,
    ) == (0, "decrease")
    assert choose_next_frequency_index(
        current_index=0,
        moving_average_context_usage=40.0,
        target_context_usage_threshold=40.0,
        max_index=3,
        moving_average_min_output_tokens_per_s=12.0,
        target_output_throughput_tokens_per_s=10.0,
    ) == (3, "increase")
    assert choose_next_frequency_index(
        current_index=1,
        moving_average_context_usage=15.0,
        target_context_usage_threshold=40.0,
        max_index=3,
        moving_average_min_output_tokens_per_s=12.0,
        target_output_throughput_tokens_per_s=10.0,
    ) == (1, "hold")


def test_choose_next_frequency_index_sets_max_for_slo_violation() -> None:
    assert choose_next_frequency_index(
        current_index=1,
        moving_average_context_usage=1.0,
        target_context_usage_threshold=40.0,
        max_index=4,
        moving_average_min_output_tokens_per_s=8.0,
        target_output_throughput_tokens_per_s=10.0,
    ) == (4, "increase_for_slo")


def test_zeusd_gpu_frequency_controller_applies_and_resets_each_gpu() -> None:
    controller = ZeusdGPUFrequencyController(
        socket_path="/tmp/zeusd.sock",
        gpu_index=(2, 3),
    )
    fake_backend = FakeZeusBackend()
    controller._gpus = fake_backend

    controller.set_frequency(1200)
    controller.reset_frequency()

    assert fake_backend.set_calls == [
        (2, 1200, 1200),
        (3, 1200, 1200),
    ]
    assert fake_backend.reset_calls == [2, 3]


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
        zeusd=ZeusdConfig(socket_path="/tmp/zeusd.sock"),
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
    assert log_paths.query_path.exists()
    assert log_paths.decision_path.exists()
    assert log_paths.control_error_path.exists()
    assert log_paths.slo_decision_path.exists()
    assert log_paths.query_path.name.startswith(
        "freq-controller-ls-instance-slo.query."
    )
    assert log_paths.decision_path.name.startswith(
        "freq-controller-ls-instance-slo.decision."
    )
    assert log_paths.control_error_path.name.startswith(
        "freq-controller-ls-instance-slo.control-error."
    )
    assert log_paths.slo_decision_path.name.startswith(
        "freq-controller-ls-instance-slo.slo-decision."
    )

    query_records = read_jsonl(log_paths.query_path)
    decision_records = read_jsonl(log_paths.decision_path)
    control_error_records = read_jsonl(log_paths.control_error_path)
    slo_decision_records = read_jsonl(log_paths.slo_decision_path)

    assert query_records[0]["phase"] == "pending"
    assert query_records[0]["gpu_indices"] == [2, 3]
    assert [record["context_usage"] for record in query_records[1:]] == [
        1.0,
        1.0,
        1.0,
    ]
    assert [record["min_output_tokens_per_s"] for record in query_records[1:]] == [
        12.0,
        12.0,
        12.0,
    ]

    assert len(decision_records) == 1
    assert decision_records[0]["action"] == "decrease"
    assert decision_records[0]["changed"] is True
    assert decision_records[0]["decision_policy"] == "context_linespace"
    assert decision_records[0]["slo_override_applied"] is False
    assert decision_records[0]["current_frequency_mhz"] == 1200
    assert decision_records[0]["target_frequency_mhz"] == 600
    assert decision_records[0]["target_context_usage_threshold"] == 30.0
    assert decision_records[0]["window_min_output_tokens_per_s"] == 12.0
    assert decision_records[0]["target_output_throughput_tokens_per_s"] == 8.0
    assert decision_records[0]["segment_count"] == 3
    assert decision_records[0]["segment_width_context_usage"] == 10.0
    assert decision_records[0]["target_frequency_index"] == 0
    assert decision_records[0]["context_target_frequency_index"] == 0
    assert decision_records[0]["gpu_indices"] == [2, 3]
    assert "aggressive_slo_control" not in decision_records[0]
    assert control_error_records == []
    assert slo_decision_records == []


def test_frequency_controller_prioritizes_slo_max_and_resets_on_exit(
    tmp_path: Path,
) -> None:
    config = FrequencyControllerConfig(
        frequency_mhz_levels=(600, 900, 1200, 1500, 1800),
        target_context_usage_threshold=30.0,
        target_output_throughput_tokens_per_s=8.0,
        control_interval_s=2.0,
        context_query_hz=1.0,
        gateway=GatewayIPCConfig(),
        zeusd=ZeusdConfig(socket_path="/tmp/zeusd.sock"),
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

    query_records = read_jsonl(log_paths.query_path)
    decision_records = read_jsonl(log_paths.decision_path)
    control_error_records = read_jsonl(log_paths.control_error_path)
    slo_decision_records = read_jsonl(log_paths.slo_decision_path)

    assert [record["min_output_tokens_per_s"] for record in query_records[1:]] == [
        4.0,
        4.0,
        4.0,
    ]
    assert len(decision_records) == 1
    assert decision_records[0]["action"] == "increase_for_slo"
    assert decision_records[0]["changed"] is True
    assert decision_records[0]["decision_policy"] == "throughput_slo_precedence"
    assert decision_records[0]["slo_override_applied"] is True
    assert decision_records[0]["target_frequency_mhz"] == 1800
    assert decision_records[0]["window_min_output_tokens_per_s"] == 4.0
    assert decision_records[0]["target_frequency_index"] == 4
    assert decision_records[0]["context_target_frequency_index"] == 0
    assert decision_records[0]["gpu_indices"] == [2, 3]
    assert "aggressive_slo_control" not in decision_records[0]
    assert len(slo_decision_records) == 1
    assert slo_decision_records[0]["action"] == "increase_for_slo"
    assert slo_decision_records[0]["decision_policy"] == "throughput_slo_precedence"
    assert slo_decision_records[0]["slo_override_applied"] is True
    assert slo_decision_records[0]["window_min_output_tokens_per_s"] == 4.0
    assert slo_decision_records[0]["gpu_indices"] == [2, 3]
    assert control_error_records == []


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
        zeusd=ZeusdConfig(socket_path="/tmp/zeusd.sock"),
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
