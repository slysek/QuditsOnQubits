"""The theta encoding family in the |00>, |01>, |10>, |11> basis order."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate


def _validate_integer(value: object, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        bounds = f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise ValueError(f"{name} must be {bounds}")
    return result


def _validate_theta(theta: object) -> float:
    if isinstance(theta, bool) or not isinstance(theta, Real):
        raise TypeError("theta must be a finite real number")
    if not 0 <= theta <= np.pi / 4:
        raise ValueError("theta must be finite and between 0 and pi/4 radians")
    return float(theta)


def theta_embedding(theta: float) -> np.ndarray:
    """Return the complex 4x3 isometry with encoded zero cos(theta)|00>-sin(theta)|11>."""
    angle = _validate_theta(theta)
    return np.array(
        [[math.cos(angle), 0, 0], [0, 1, 0], [0, 0, 1], [-math.sin(angle), 0, 0]],
        dtype=complex,
    )


def theta_leakage(theta: float) -> np.ndarray:
    """Return the normalized leakage vector sin(theta)|00>+cos(theta)|11>."""
    angle = _validate_theta(theta)
    return np.array([math.sin(angle), 0, 0, math.cos(angle)], dtype=complex)


def theta_grid(point_count: int = 41, limit_points: int | None = None) -> tuple[float, ...]:
    """Return a uniform grid including both endpoints, optionally limited to its prefix."""
    count = _validate_integer(point_count, "point_count", 2)
    limit = count if limit_points is None else _validate_integer(limit_points, "limit_points", 1, count)
    return tuple(float(theta) for theta in np.linspace(0, np.pi / 4, count)[:limit])


def make_theta_candidates(
    point_count: int = 41, limit_points: int | None = None,
) -> list[DirectBasisCandidate]:
    """Return candidates named by their full-grid indices, preserving prefix identities."""
    return [
        DirectBasisCandidate(
            name=f"theta_{index:05d}",
            candidate_type="theta_continuation",
            matrix=theta_embedding(theta),
            notes=f"theta={theta:.17g} rad; grid_index={index}; point_count={point_count}",
        )
        for index, theta in enumerate(theta_grid(point_count, limit_points))
    ]
