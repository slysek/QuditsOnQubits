"""Numerical acceptance and trial budgets for the unified benchmark."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
import math


@dataclass(frozen=True)
class BenchmarkConfig:
    transpiler_seeds: tuple[int, ...] = (0, 1, 2)
    f3_tolerance: float = 1e-10
    cz3_tolerance: float = 1e-5
    minimum_fidelity: float = 1 - 1e-6
    maximum_leakage: float = 1e-6
    max_validation_qubits: int = 12

    def __post_init__(self):
        seeds = tuple(self.transpiler_seeds)
        if (len(seeds) < 2 or len(set(seeds)) != len(seeds)
                or any(isinstance(s, bool) or not isinstance(s, Integral) or not 0 <= s < 2**32 for s in seeds)):
            raise ValueError("transpiler_seeds must contain at least two unique uint32 integers")
        object.__setattr__(self, "transpiler_seeds", tuple(int(s) for s in seeds))
        for name in ("f3_tolerance", "cz3_tolerance", "minimum_fidelity", "maximum_leakage"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
            if name.endswith("tolerance") and value <= 0:
                raise ValueError(f"{name} must be positive")
            if not name.endswith("tolerance") and not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.f3_tolerance > 1e-10:
            raise ValueError("f3_tolerance cannot exceed 1e-10")
        limit = self.max_validation_qubits
        if isinstance(limit, bool) or not isinstance(limit, Integral) or not 2 <= limit <= 16:
            raise ValueError("max_validation_qubits must be an integer between 2 and 16")
        object.__setattr__(self, "max_validation_qubits", int(limit))

    def to_dict(self):
        result = asdict(self)
        result["transpiler_seeds"] = list(self.transpiler_seeds)
        return result
