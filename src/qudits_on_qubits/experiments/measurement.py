"""Validated measurement policies for independent local Bell sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, ClassVar

from .errors import ExperimentValidationError


@dataclass(frozen=True)
class RandomizedBlocks:
    """Draw local settings independently, retaining a fixed shot block per draw.

    ``setting_draws`` is a minimum; drawing continues until every required Bell
    pattern is covered, subject to ``max_setting_draws``.
    """

    setting_draws: int
    shots_per_draw: int
    max_setting_draws: int
    max_circuits_per_job: int = 100
    confidence_level: float = 0.95
    mode: ClassVar[str] = "independent_local_uniform_blocks"
    version: ClassVar[int] = 1

    def __post_init__(self) -> None:
        for name in ("setting_draws", "shots_per_draw", "max_setting_draws", "max_circuits_per_job"):
            value = getattr(self, name)
            minimum = 1
            if type(value) is not int or value < minimum:
                raise ExperimentValidationError(f"{name} must be an integer of at least {minimum}")
        if self.max_setting_draws < self.setting_draws:
            raise ExperimentValidationError("max_setting_draws must be at least setting_draws")
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not 0 < self.confidence_level < 1
            or not math.isfinite(self.confidence_level)
        ):
            raise ExperimentValidationError("confidence_level must be finite and between 0 and 1")

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "version": self.version,
            "setting_draws": self.setting_draws,
            "shots_per_draw": self.shots_per_draw,
            "max_setting_draws": self.max_setting_draws,
            "max_circuits_per_job": self.max_circuits_per_job,
            "confidence_level": self.confidence_level,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "RandomizedBlocks":
        if not isinstance(data, Mapping):
            raise ExperimentValidationError("measurement must be a mapping")
        if data.get("mode") != cls.mode or type(data.get("version")) is not int or data["version"] != cls.version:
            raise ExperimentValidationError("unsupported measurement mode or version")
        values = {key: value for key, value in data.items() if key not in {"mode", "version"}}
        try:
            return cls(**values)
        except TypeError as error:
            raise ExperimentValidationError("invalid randomized measurement fields") from error
