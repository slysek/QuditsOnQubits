"""Deterministic encoding families, independent of synthesis policy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol

import numpy as np

from qudits_on_qubits.benchmarks.direct_basis.math_utils import canonical_qutrit_embedding
from qudits_on_qubits.benchmarks.models import EncodingCandidate
from qudits_on_qubits.benchmarks.theta_continuation.encoding import theta_embedding, theta_grid


class EncodingFamily(Protocol):
    """A family supplies candidates and a JSON description of its configuration."""

    def generate(self) -> Iterable[EncodingCandidate]: ...

    def to_dict(self) -> dict[str, Any]: ...


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def canonical_candidate() -> EncodingCandidate:
    """Return the explicit baseline, without adding it to any selected family."""
    return EncodingCandidate("canonical", "canonical", canonical_qutrit_embedding())


def _haar_su2(rng: np.random.Generator) -> np.ndarray:
    gaussian = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    unitary, triangular = np.linalg.qr(gaussian)
    diagonal = np.diag(triangular)
    unitary = unitary * (diagonal / np.abs(diagonal))
    return unitary / np.sqrt(np.linalg.det(unitary))


@dataclass(frozen=True)
class LocalSU2:
    """Sample (U1 tensor U2) E0 using Haar QR and determinant correction.

    The seed and sample index reproduce each candidate. Increasing ``samples``
    preserves the existing prefix; a finite sample does not cover the full family.
    """

    samples: int = 20
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", _integer(self.samples, "samples", 1))
        object.__setattr__(self, "seed", _integer(self.seed, "seed", 0))

    def generate(self) -> Iterable[EncodingCandidate]:
        rng = np.random.default_rng(self.seed)
        canonical = canonical_qutrit_embedding()
        for index in range(self.samples):
            first, second = _haar_su2(rng), _haar_su2(rng)
            yield EncodingCandidate(
                candidate_id=f"local_su2_s{self.seed}_{index:05d}",
                family="local_su2",
                encoding=np.kron(first, second) @ canonical,
                parameters={"seed": self.seed, "sample_index": index, "sampler": "haar_qr"},
            )

    def to_dict(self) -> dict[str, Any]:
        return {"family": "local_su2", "samples": self.samples, "seed": self.seed, "sampler": "haar_qr"}


@dataclass(frozen=True)
class SchmidtTheta:
    """Use the existing uniform theta grid and its fixed encoding basis."""

    points: int = 41

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _integer(self.points, "points", 2))

    def generate(self) -> Iterable[EncodingCandidate]:
        for index, theta in enumerate(theta_grid(self.points)):
            yield EncodingCandidate(
                candidate_id=f"schmidt_theta_p{self.points}_{index:05d}",
                family="schmidt_theta",
                encoding=theta_embedding(theta),
                parameters={"theta": theta, "grid_index": index, "points": self.points},
            )

    def to_dict(self) -> dict[str, Any]:
        return {"family": "schmidt_theta", "points": self.points}
