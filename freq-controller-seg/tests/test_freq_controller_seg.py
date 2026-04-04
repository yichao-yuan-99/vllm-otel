from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PROJECT_ROOT / "freq-controller-seg"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from freq_controller_seg import (
    FrequencyController,
    FrequencyControllerConfig,
    GatewayContextSnapshot,
    GatewayIPCConfig,
    ZeusdConfig,
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


def test_load_controller_config_applies_segment_fields_and_defaults(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "frequency_mhz_levels = [1200, 810, 1005]",
                "target_context_usage_lower_bound = 5000",
                "target_context_usage_upper_bound = 15000",
                "low_freq_threshold = 2000",
                "low_freq_cap_mhz = 1005",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_controller_config(config_path)

    assert config.frequency_mhz_levels == (810, 1005, 1200)
    assert config.target_context_usage_lower_bound == 5000.0
    assert config.target_context_usage_upper_bound == 15000.0
    assert config.low_freq_threshold == 2000.0
    assert config.low_freq_cap_mhz == 1005
    assert config.low_freq_cap_index == 1
    assert config.control_interval_s == 5.0
    assert config.context_query_hz == 5.0


def test_load_controller_config_supports_no_config_with_cli_overrides() -> None:
    config = load_controller_config(
        None,
        target_context_usage_lower_bound=12000,
        target_context_usage_upper_bound=22000,
        low_freq_threshold=7000,
        low_freq_cap_mhz=1005,
    )

    assert config.frequency_mhz_levels == DEFAULT_SHARED_FREQUENCY_MHZ_LEVELS
    assert config.target_context_usage_lower_bound == 12000.0
    assert config.target_context_usage_upper_bound == 22000.0
    assert config.low_freq_threshold == 7000.0
    assert config.low_freq_cap_mhz == 1005


def test_load_controller_config_uses_shared_segment_defaults() -> None:
    config = load_controller_config(
        None,
        target_context_usage_lower_bound=12000,
        target_context_usage_upper_bound=22000,
    )

    assert config.frequency_mhz_levels == DEFAULT_SHARED_FREQUENCY_MHZ_LEVELS
    assert config.target_context_usage_lower_bound == 12000.0
    assert config.target_context_usage_upper_bound == 22000.0
    assert config.low_freq_threshold == 6000.0
    assert config.low_freq_cap_mhz == 900


def test_load_controller_config_rejects_low_freq_threshold_above_lower_bound(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "frequency_mhz_levels = [600, 900, 1200]",
                "target_context_usage_lower_bound = 50",
                "target_context_usage_upper_bound = 150",
                "low_freq_threshold = 75",
                "low_freq_cap_mhz = 900",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="low_freq_threshold must be <= target_context_usage_lower_bound",
    ):
        load_controller_config(config_path)


def test_load_controller_config_rejects_low_freq_cap_not_in_levels(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "controller.toml"
    config_path.write_text(
        "\n".join(
            [
                "frequency_mhz_levels = [600, 900, 1200]",
                "target_context_usage_lower_bound = 50",
                "target_context_usage_upper_bound = 150",
                "low_freq_threshold = 25",
                "low_freq_cap_mhz = 750",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="low_freq_cap_mhz must be one of frequency_mhz_levels",
    ):
        load_controller_config(config_path)


def test_choose_next_frequency_index_enforces_segmented_floor() -> None:
    assert choose_next_frequency_index(
        current_index=1,
        moving_average_context_usage=30.0,
        lower_bound=80.0,
        upper_bound=140.0,
        low_freq_threshold=40.0,
        low_freq_cap_index=1,
        max_index=2,
    ) == (0, "decrease")
    assert choose_next_frequency_index(
        current_index=1,
        moving_average_context_usage=70.0,
        lower_bound=80.0,
        upper_bound=140.0,
        low_freq_threshold=40.0,
        low_freq_cap_index=1,
        max_index=2,
    ) == (1, "decrease")
    assert choose_next_frequency_index(
        current_index=0,
        moving_average_context_usage=70.0,
        lower_bound=80.0,
        upper_bound=140.0,
        low_freq_threshold=40.0,
        low_freq_cap_index=1,
        max_index=2,
    ) == (1, "increase_to_low_freq_cap")
    assert choose_next_frequency_index(
        current_index=1,
        moving_average_context_usage=100.0,
        lower_bound=80.0,
        upper_bound=140.0,
        low_freq_threshold=40.0,
        low_freq_cap_index=1,
        max_index=2,
    ) == (1, "hold")
    assert choose_next_frequency_index(
        current_index=1,
        moving_average_context_usage=200.0,
        lower_bound=80.0,
        upper_bound=140.0,
        low_freq_threshold=40.0,
        low_freq_cap_index=1,
        max_index=2,
    ) == (2, "increase")


def test_frequency_controller_uses_segmented_policy_and_resets_on_exit(
    tmp_path: Path,
) -> None:
    config = FrequencyControllerConfig(
        frequency_mhz_levels=(600, 900, 1200),
        target_context_usage_lower_bound=80.0,
        target_context_usage_upper_bound=140.0,
        low_freq_threshold=40.0,
        low_freq_cap_mhz=900,
        control_interval_s=2.0,
        context_query_hz=1.0,
        gateway=GatewayIPCConfig(),
        zeusd=ZeusdConfig(socket_path="/tmp/zeusd.sock"),
    )
    fake_gateway = FakeGatewayClient(
        [
            gateway_snapshot(30.0),
            gateway_snapshot(30.0),
            gateway_snapshot(30.0),
            gateway_snapshot(30.0),
            gateway_snapshot(70.0),
            gateway_snapshot(70.0),
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
    assert log_paths.query_path.name.startswith("freq-controller.query.")
    assert log_paths.decision_path.name.startswith("freq-controller.decision.")

    query_records = read_jsonl(log_paths.query_path)
    decision_records = read_jsonl(log_paths.decision_path)

    assert query_records[0]["phase"] == "pending"
    assert [record["context_usage"] for record in query_records[1:]] == [
        30.0,
        30.0,
        30.0,
        70.0,
        70.0,
    ]

    assert len(decision_records) == 2
    assert decision_records[0]["action"] == "decrease"
    assert decision_records[0]["changed"] is True
    assert decision_records[0]["current_frequency_mhz"] == 900
    assert decision_records[0]["target_frequency_mhz"] == 600
    assert decision_records[0]["window_context_usage"] == pytest.approx(30.0)
    assert decision_records[0]["effective_min_frequency_mhz"] == 600
    assert decision_records[0]["low_freq_threshold"] == 40.0
    assert decision_records[0]["low_freq_cap_mhz"] == 900

    assert decision_records[1]["action"] == "increase_to_low_freq_cap"
    assert decision_records[1]["changed"] is True
    assert decision_records[1]["current_frequency_mhz"] == 600
    assert decision_records[1]["target_frequency_mhz"] == 900
    assert decision_records[1]["window_context_usage"] == pytest.approx(
        56.666666666666664
    )
    assert decision_records[1]["effective_min_frequency_mhz"] == 900
