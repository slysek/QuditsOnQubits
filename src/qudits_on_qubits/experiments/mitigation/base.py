"""Protocols shared by experiment mitigation implementations."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class ZNEStrategy(Protocol):
    """Circuit folding and zero-noise extrapolation strategy contract."""

    def fold(self, circuits: Sequence[Any], factor: int) -> tuple[Any, ...]:
        """Return freshly allocated circuits folded by ``factor``."""

    def extrapolate(
        self, factors: Sequence[int], values: Sequence[complex]
    ) -> tuple[complex, Any]:
        """Extrapolate values to zero noise and return immutable fit evidence."""


@runtime_checkable
class ReadoutMitigationStrategy(Protocol):
    """Small subset of M3 API used by experiment runner."""

    def cals_from_matrices(self, matrices: Sequence[Any]) -> None:
        """Configure assignment matrices."""

    def apply_correction(
        self, counts: Mapping[str, int], qubits: Sequence[int]
    ) -> Mapping[str, float]:
        """Correct one setting's counts using its physical-qubit mapping."""
