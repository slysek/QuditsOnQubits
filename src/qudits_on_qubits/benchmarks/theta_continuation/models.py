"""Validated configuration for reproducible theta continuation benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from importlib.resources import files
import re
from typing import Any, Mapping

from .encoding import _validate_integer, theta_grid


_DEFAULT_BASELINE_QPY = str(
    files("qudits_on_qubits.benchmarks.theta_continuation.data").joinpath("CZ3_W.qpy")
)
_DEFAULT_BASELINE_SHA256 = "a84ec3bfb10c03e65942fc653c7ba485ffc62a8a85756fa3f62aec8102b5e791"
_KNOWN_STATES = ("two_qutrit", "ghz3", "ame43")


def _positive_tolerance(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonempty_tuple(value: object, name: str) -> tuple:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a nonempty tuple or list")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return tuple(value)


@dataclass(frozen=True)
class ThetaBenchmarkConfig:
    """Immutable numerical and compilation settings; output handling belongs to the runner."""

    theta_points: int = 41
    limit_points: int | None = None
    max_nfev: int = 3000
    max_subdivisions: int = 4
    f3_tolerance: float = 1e-10
    cz3_tolerance: float = 1e-5
    transpiler_seeds: tuple[int, ...] = (0, 1, 2)
    optimization_level: int = 3
    baseline_qpy: str = _DEFAULT_BASELINE_QPY
    baseline_sha256: str = _DEFAULT_BASELINE_SHA256
    states: tuple[str, ...] = _KNOWN_STATES

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("theta_points", 2, None),
            ("max_nfev", 1, None),
            ("max_subdivisions", 0, 4),
            ("optimization_level", 0, 3),
        ):
            object.__setattr__(self, name, _validate_integer(getattr(self, name), name, minimum, maximum))
        if self.limit_points is not None:
            object.__setattr__(
                self, "limit_points", _validate_integer(self.limit_points, "limit_points", 1, self.theta_points),
            )
        for name in ("f3_tolerance", "cz3_tolerance"):
            tolerance = _positive_tolerance(getattr(self, name), name)
            if name == "f3_tolerance" and tolerance > 1e-10:
                raise ValueError("f3_tolerance must not exceed 1e-10; existing synthesis uses this limit")
            object.__setattr__(self, name, tolerance)

        seeds = tuple(
            _validate_integer(seed, "transpiler_seeds", 0, 2**32 - 1)
            for seed in _nonempty_tuple(self.transpiler_seeds, "transpiler_seeds")
        )
        if len(set(seeds)) != len(seeds):
            raise ValueError("transpiler_seeds must not contain duplicates")
        object.__setattr__(self, "transpiler_seeds", seeds)

        states = _nonempty_tuple(self.states, "states")
        if any(not isinstance(state, str) or state not in _KNOWN_STATES for state in states):
            raise ValueError(f"states must contain only {', '.join(_KNOWN_STATES)}")
        if len(set(states)) != len(states):
            raise ValueError("states must not contain duplicates")
        object.__setattr__(self, "states", states)

        if not isinstance(self.baseline_qpy, str) or not self.baseline_qpy.strip():
            raise ValueError("baseline_qpy must be a nonempty path string")
        if not isinstance(self.baseline_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", self.baseline_sha256):
            raise ValueError("baseline_sha256 must contain exactly 64 hexadecimal characters")
        object.__setattr__(self, "baseline_sha256", self.baseline_sha256.lower())

    def grid(self) -> tuple[float, ...]:
        """Return the configured full grid or its unchanged prefix."""
        return theta_grid(self.theta_points, self.limit_points)

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON-safe configuration values."""
        values = asdict(self)
        values["transpiler_seeds"] = list(self.transpiler_seeds)
        values["states"] = list(self.states)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ThetaBenchmarkConfig:
        """Construct and validate configuration values decoded from JSON."""
        if not isinstance(values, Mapping):
            raise TypeError("configuration must be a mapping")
        return cls(**dict(values))
