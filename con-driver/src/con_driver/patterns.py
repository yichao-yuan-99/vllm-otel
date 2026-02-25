"""Arrival pattern implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from random import Random


class ArrivalPattern(ABC):
    """Controls delay between successive launches."""

    @abstractmethod
    def next_delay_s(self) -> float:
        pass

    @abstractmethod
    def describe(self) -> dict[str, float | str]:
        pass


class EagerArrivalPattern(ArrivalPattern):
    """Launch as soon as slots are available."""

    def next_delay_s(self) -> float:
        return 0.0

    def describe(self) -> dict[str, float | str]:
        return {"name": "eager"}


class PoissonArrivalPattern(ArrivalPattern):
    """Launches with exponentially distributed inter-arrival times."""

    def __init__(self, *, rate_per_second: float, rng: Random):
        if rate_per_second <= 0:
            raise ValueError("Poisson rate must be > 0")
        self._rate_per_second = rate_per_second
        self._rng = rng

    def next_delay_s(self) -> float:
        return self._rng.expovariate(self._rate_per_second)

    def describe(self) -> dict[str, float | str]:
        return {
            "name": "poisson",
            "rate_per_second": self._rate_per_second,
            "mean_interval_s": 1.0 / self._rate_per_second,
        }


def build_arrival_pattern(
    *, name: str, pattern_args: dict[str, str], rng: Random
) -> ArrivalPattern:
    """Build an arrival pattern from CLI inputs."""
    normalized = name.strip().lower()
    if normalized == "eager":
        return EagerArrivalPattern()

    if normalized in {"poisson", "possion"}:
        rate_value = (
            pattern_args.get("rate")
            or pattern_args.get("arrival-rate")
            or pattern_args.get("arrival_rate")
            or pattern_args.get("lambda")
        )

        if rate_value is None:
            mean_value = (
                pattern_args.get("mean-interval-s")
                or pattern_args.get("mean_interval_s")
                or pattern_args.get("interval-s")
                or pattern_args.get("interval_s")
            )
            if mean_value is None:
                raise ValueError(
                    "Poisson pattern requires --pattern-args with one of: "
                    "--rate=<arrivals_per_second> or --mean-interval-s=<seconds>."
                )
            mean_interval = float(mean_value)
            if mean_interval <= 0:
                raise ValueError("Poisson mean interval must be > 0")
            rate = 1.0 / mean_interval
        else:
            rate = float(rate_value)

        return PoissonArrivalPattern(rate_per_second=rate, rng=rng)

    raise ValueError(
        f"Unsupported pattern '{name}'. Supported patterns: eager, poisson."
    )
