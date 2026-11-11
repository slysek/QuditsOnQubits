"""Logical targets in the computational basis, using positive Fourier phases."""

import numpy as np

from .models import _integer_at_least


def _phase_table(d: int) -> np.ndarray:
    levels = np.arange(d)
    return np.exp((2j * np.pi / d) * (np.outer(levels, levels) % d))


def fourier_d(d: int) -> np.ndarray:
    """The normalized, positive-sign d-dimensional Fourier transform."""
    d = _integer_at_least(d, "d", 2)
    return _phase_table(d) / np.sqrt(d)


def controlled_z_d(d: int) -> np.ndarray:
    """Apply exp(2πi*j*k/d) to the two-qudit basis vector of index j*d+k."""
    d = _integer_at_least(d, "d", 2)
    return np.diag(_phase_table(d).ravel())
