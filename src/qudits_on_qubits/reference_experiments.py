from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

import numpy as np


FrozenMatrix: TypeAlias = tuple[tuple[complex, ...], ...]
WeightedEdge: TypeAlias = tuple[int, int, int]
Outcome: TypeAlias = int | None

_ATOL = 1e-10


def _freeze_matrix(value: Any, *, name: str) -> FrozenMatrix:
    try:
        return tuple(tuple(complex(entry) for entry in row) for row in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric matrix") from error


def _matrix_array(value: FrozenMatrix, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=complex)
    except ValueError as error:
        raise ValueError(f"{name} must be a rectangular numeric matrix") from error
    if array.ndim != 2:
        raise ValueError(f"{name} must be a rectangular numeric matrix")
    return array


@dataclass(frozen=True)
class EncodingSpec:
    encoding_id: str
    logical_dimension: int
    physical_qubits_per_qutrit: int
    isometry: FrozenMatrix
    leakage_basis: FrozenMatrix

    def __post_init__(self) -> None:
        if not isinstance(self.encoding_id, str) or not self.encoding_id.strip():
            raise ValueError("encoding_id must be nonempty")
        if not isinstance(self.logical_dimension, int) or self.logical_dimension < 1:
            raise ValueError("logical_dimension must be a positive integer")
        if (
            not isinstance(self.physical_qubits_per_qutrit, int)
            or self.physical_qubits_per_qutrit < 1
        ):
            raise ValueError("physical_qubits_per_qutrit must be a positive integer")

        frozen_isometry = _freeze_matrix(self.isometry, name="isometry")
        frozen_leakage = _freeze_matrix(self.leakage_basis, name="leakage_basis")
        object.__setattr__(self, "isometry", frozen_isometry)
        object.__setattr__(self, "leakage_basis", frozen_leakage)

        isometry = _matrix_array(frozen_isometry, name="isometry")
        leakage = _matrix_array(frozen_leakage, name="leakage_basis")
        physical_dimension = 2**self.physical_qubits_per_qutrit
        leakage_dimension = physical_dimension - self.logical_dimension
        if leakage_dimension < 0:
            raise ValueError("logical_dimension cannot exceed physical dimension")
        if isometry.shape != (physical_dimension, self.logical_dimension):
            raise ValueError(
                "isometry must have shape "
                f"({physical_dimension}, {self.logical_dimension})"
            )
        if leakage.shape != (physical_dimension, leakage_dimension):
            raise ValueError(
                "leakage_basis must have shape "
                f"({physical_dimension}, {leakage_dimension})"
            )
        if not np.allclose(
            isometry.conj().T @ isometry,
            np.eye(self.logical_dimension),
            rtol=0,
            atol=_ATOL,
        ):
            raise ValueError("isometry must satisfy E^dagger E = I")
        if not np.allclose(
            leakage.conj().T @ leakage,
            np.eye(leakage_dimension),
            rtol=0,
            atol=_ATOL,
        ):
            raise ValueError("leakage_basis must be orthonormal")
        if not np.allclose(
            isometry.conj().T @ leakage,
            np.zeros((self.logical_dimension, leakage_dimension)),
            rtol=0,
            atol=_ATOL,
        ):
            raise ValueError("isometry and leakage_basis must be orthogonal")

    def as_array(self) -> np.ndarray:
        return np.array(self.isometry, dtype=complex, copy=True)

    def leakage_array(self) -> np.ndarray:
        return np.array(self.leakage_basis, dtype=complex, copy=True)


@dataclass(frozen=True)
class LogicalStateSpec:
    state_id: str
    local_dimension: int
    num_parties: int
    party_order: tuple[int, ...]
    weighted_edges: tuple[WeightedEdge, ...]

    def __post_init__(self) -> None:
        try:
            frozen_party_order = tuple(int(party) for party in self.party_order)
        except (TypeError, ValueError) as error:
            raise ValueError("party_order must contain integers") from error
        try:
            frozen_weighted_edges = tuple(
                tuple(int(value) for value in edge) for edge in self.weighted_edges
            )
        except (TypeError, ValueError) as error:
            raise ValueError("weighted_edges must contain integer triples") from error
        object.__setattr__(self, "party_order", frozen_party_order)
        object.__setattr__(self, "weighted_edges", frozen_weighted_edges)

        if self.local_dimension != 3:
            raise ValueError("local_dimension must be exactly 3")
        if not isinstance(self.num_parties, int) or self.num_parties < 2:
            raise ValueError("num_parties must be at least 2")
        if self.party_order != tuple(range(self.num_parties)):
            raise ValueError("party_order must equal tuple(range(num_parties))")

        seen_edges: set[tuple[int, int]] = set()
        for edge in self.weighted_edges:
            if len(edge) != 3:
                raise ValueError("each weighted edge must contain (u, v, weight)")
            u, v, weight = edge
            if not 0 <= u < v < self.num_parties:
                raise ValueError(
                    "weighted edge endpoints must satisfy 0 <= u < v < num_parties"
                )
            if weight not in {1, 2}:
                raise ValueError("weighted edge weight must be 1 or 2")
            pair = (u, v)
            if pair in seen_edges:
                raise ValueError(f"duplicate weighted edge: {pair}")
            seen_edges.add(pair)

    def statevector(self) -> np.ndarray:
        dimension = self.local_dimension**self.num_parties
        normalization = math.sqrt(dimension)
        omega = np.exp(2j * np.pi / self.local_dimension)
        state = np.empty(dimension, dtype=complex)
        shape = (self.local_dimension,) * self.num_parties
        for index in range(dimension):
            digits = np.unravel_index(index, shape)
            exponent = sum(
                weight * digits[u] * digits[v]
                for u, v, weight in self.weighted_edges
            ) % self.local_dimension
            state[index] = omega**exponent / normalization
        if not np.isclose(np.linalg.norm(state), 1, rtol=0, atol=_ATOL):
            raise ValueError("generated statevector is not normalized")
        return state

    def legacy_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (u, v)
            for u, v, weight in self.weighted_edges
            for _ in range(weight)
        )


CANONICAL_EZ = EncodingSpec(
    encoding_id="canonical_ez",
    logical_dimension=3,
    physical_qubits_per_qutrit=2,
    isometry=(
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 0, 0),
    ),
    leakage_basis=((0,), (0,), (0,), (1,)),
)

ENCODINGS: Mapping[str, EncodingSpec] = MappingProxyType(
    {"canonical_ez": CANONICAL_EZ}
)


def get_encoding(encoding_id: str) -> EncodingSpec:
    normalized_id = encoding_id.strip() if isinstance(encoding_id, str) else ""
    try:
        return ENCODINGS[normalized_id]
    except KeyError as error:
        accepted = ", ".join(ENCODINGS)
        raise ValueError(
            f"unknown encoding ID {encoding_id!r}; accepted IDs: {accepted}"
        ) from error
