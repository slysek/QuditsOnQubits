from __future__ import annotations

import math
import hashlib
import itertools
import json
from dataclasses import dataclass, fields, is_dataclass
from numbers import Complex, Real
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

import numpy as np


FrozenMatrix: TypeAlias = tuple[tuple[complex, ...], ...]
WeightedEdge: TypeAlias = tuple[int, int, int]
Outcome: TypeAlias = int | None

_ATOL = 1e-10


def _is_metadata_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


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
        if (
            not _is_metadata_integer(self.logical_dimension)
            or self.logical_dimension < 1
        ):
            raise ValueError("logical_dimension must be a positive integer")
        if (
            not _is_metadata_integer(self.physical_qubits_per_qutrit)
            or self.physical_qubits_per_qutrit < 1
        ):
            raise ValueError("physical_qubits_per_qutrit must be a positive integer")
        object.__setattr__(self, "logical_dimension", int(self.logical_dimension))
        object.__setattr__(
            self,
            "physical_qubits_per_qutrit",
            int(self.physical_qubits_per_qutrit),
        )

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
            party_order = tuple(self.party_order)
        except TypeError as error:
            raise ValueError("party_order must be an iterable of integers") from error
        try:
            weighted_edges = tuple(
                tuple(edge) for edge in self.weighted_edges
            )
        except TypeError as error:
            raise ValueError("weighted_edges must contain integer triples") from error

        if not isinstance(self.state_id, str) or not self.state_id.strip():
            raise ValueError("state_id must be a nonempty string")
        if not _is_metadata_integer(self.local_dimension):
            raise ValueError("local_dimension must be exactly 3")
        local_dimension = int(self.local_dimension)
        if local_dimension != 3:
            raise ValueError("local_dimension must be exactly 3")
        if not _is_metadata_integer(self.num_parties) or self.num_parties < 2:
            raise ValueError("num_parties must be at least 2")
        num_parties = int(self.num_parties)
        if not all(_is_metadata_integer(party) for party in party_order):
            raise ValueError("party_order values must be integers")
        frozen_party_order = tuple(int(party) for party in party_order)
        if frozen_party_order != tuple(range(num_parties)):
            raise ValueError("party_order must equal tuple(range(num_parties))")

        seen_edges: set[tuple[int, int]] = set()
        frozen_weighted_edges: list[WeightedEdge] = []
        for edge in weighted_edges:
            if len(edge) != 3:
                raise ValueError("each weighted edge must contain (u, v, weight)")
            if not all(_is_metadata_integer(value) for value in edge):
                raise ValueError("weighted edge values must be integers")
            u, v, weight = (int(value) for value in edge)
            if not 0 <= u < v < num_parties:
                raise ValueError(
                    "weighted edge endpoints must satisfy 0 <= u < v < num_parties"
                )
            if weight not in {1, 2}:
                raise ValueError("weighted edge weight must be 1 or 2")
            pair = (u, v)
            if pair in seen_edges:
                raise ValueError(f"duplicate weighted edge: {pair}")
            seen_edges.add(pair)
            frozen_weighted_edges.append((u, v, weight))

        object.__setattr__(self, "local_dimension", local_dimension)
        object.__setattr__(self, "num_parties", num_parties)
        object.__setattr__(self, "party_order", frozen_party_order)
        object.__setattr__(self, "weighted_edges", tuple(frozen_weighted_edges))

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


def _omega(d: int = 3) -> complex:
    if not _is_metadata_integer(d) or d < 1:
        raise ValueError("d must be a positive integer")
    return complex(np.exp(2j * np.pi / d))


def _finite_complex(value: Any, *, name: str) -> complex:
    if not isinstance(value, Complex) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonempty_string(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _ordered_eigenbasis(matrix: Any) -> tuple[np.ndarray, complex]:
    try:
        array = np.asarray(matrix, dtype=complex)
    except (TypeError, ValueError) as error:
        raise ValueError("matrix must be a numeric 3x3 matrix") from error
    if array.shape != (3, 3):
        raise ValueError("matrix must be a numeric 3x3 matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must contain finite values")

    eigenvalues, eigenvectors = np.linalg.eig(array)
    roots = np.asarray([_omega() ** outcome for outcome in range(3)])
    candidates: list[tuple[float, float, tuple[int, ...], complex]] = []
    for permutation in itertools.permutations(range(3)):
        ordered_values = eigenvalues[list(permutation)]
        ratios = ordered_values / roots
        mean_ratio = complex(np.mean(ratios))
        if abs(mean_ratio) <= 1e-12:
            continue
        gamma = mean_ratio / abs(mean_ratio)
        error = float(np.max(np.abs(ordered_values - gamma * roots)))
        if error <= 1e-7:
            candidates.append(
                (abs(float(np.angle(gamma))), error, permutation, gamma)
            )
    if not candidates:
        raise ValueError("matrix must have a valid ordered qutrit root spectrum")

    _, error, permutation, gamma = min(candidates, key=lambda item: item[:3])
    if error > 1e-7:
        raise ValueError("matrix must have a valid ordered qutrit root spectrum")

    basis = np.array(eigenvectors[:, list(permutation)], dtype=complex, copy=True)
    for column_index in range(3):
        vector = basis[:, column_index]
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("matrix eigenvectors must be normalizable")
        vector = vector / norm
        pivot = int(np.argmax(np.abs(vector)))
        vector = vector * np.exp(-1j * np.angle(vector[pivot]))
        vector[pivot] = complex(abs(vector[pivot]), 0.0)
        basis[:, column_index] = vector

    if not np.allclose(basis.conj().T @ basis, np.eye(3), rtol=0, atol=1e-8):
        raise ValueError("ordered eigenbasis must be orthonormal")
    return basis, complex(gamma)


@dataclass(frozen=True)
class LocalObservableSpec:
    label: str
    matrix: FrozenMatrix

    def __post_init__(self) -> None:
        _nonempty_string(self.label, name="label")
        frozen_matrix = _freeze_matrix(self.matrix, name="matrix")
        object.__setattr__(self, "matrix", frozen_matrix)
        array = _matrix_array(frozen_matrix, name="matrix")
        if array.shape != (3, 3):
            raise ValueError("matrix must have shape 3x3")
        if not np.all(np.isfinite(array)):
            raise ValueError("matrix must contain finite values")
        if not np.allclose(array.conj().T @ array, np.eye(3), rtol=0, atol=1e-8):
            raise ValueError("matrix must be unitary")
        _ordered_eigenbasis(array)

    def as_array(self) -> np.ndarray:
        return np.array(self.matrix, dtype=complex, copy=True)

    def ordered_eigenbasis(self) -> tuple[np.ndarray, complex]:
        return _ordered_eigenbasis(self.matrix)


@dataclass(frozen=True)
class BellFactorSpec:
    party: int
    setting_label: str
    outcome_power: int
    operator_scale: complex = 1 + 0j

    def __post_init__(self) -> None:
        if not _is_metadata_integer(self.party) or self.party < 0:
            raise ValueError("party must be a nonnegative integer")
        _nonempty_string(self.setting_label, name="setting_label")
        if not _is_metadata_integer(self.outcome_power):
            raise ValueError("outcome_power must be an integer")
        object.__setattr__(self, "party", int(self.party))
        object.__setattr__(self, "outcome_power", int(self.outcome_power))
        object.__setattr__(
            self,
            "operator_scale",
            _finite_complex(self.operator_scale, name="operator_scale"),
        )

    def logical_operator(self, observable: LocalObservableSpec) -> np.ndarray:
        if not isinstance(observable, LocalObservableSpec):
            raise ValueError("observable must be a LocalObservableSpec")
        if observable.label != self.setting_label:
            raise ValueError(
                f"factor setting {self.setting_label!r} does not match "
                f"observable {observable.label!r}"
            )
        basis, _ = observable.ordered_eigenbasis()
        diagonal = np.diag(
            [
                _omega() ** ((self.outcome_power * outcome) % 3)
                for outcome in range(3)
            ]
        )
        return self.operator_scale * basis @ diagonal @ basis.conj().T


@dataclass(frozen=True)
class BellTermSpec:
    coefficient: complex
    factors: tuple[BellFactorSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficient",
            _finite_complex(self.coefficient, name="coefficient"),
        )
        try:
            frozen_factors = tuple(self.factors)
        except TypeError as error:
            raise ValueError("factors must be an iterable of BellFactorSpec") from error
        if not frozen_factors:
            raise ValueError("factors must be nonempty")
        if not all(isinstance(factor, BellFactorSpec) for factor in frozen_factors):
            raise ValueError("factors must contain BellFactorSpec values")
        object.__setattr__(self, "factors", frozen_factors)

    def sampling_coefficient(self) -> complex:
        result = self.coefficient
        for factor in self.factors:
            result *= factor.operator_scale
        return complex(result)


@dataclass(frozen=True)
class BellFunctionalSpec:
    functional_id: str
    normalization: str
    terms: tuple[BellTermSpec, ...]
    classical_bound: float
    classical_bound_source: str

    def __post_init__(self) -> None:
        _nonempty_string(self.functional_id, name="functional_id")
        _nonempty_string(self.normalization, name="normalization")
        _nonempty_string(self.classical_bound_source, name="classical_bound_source")
        try:
            frozen_terms = tuple(self.terms)
        except TypeError as error:
            raise ValueError("terms must be an iterable of BellTermSpec") from error
        if not frozen_terms or not all(
            isinstance(term, BellTermSpec) for term in frozen_terms
        ):
            raise ValueError("terms must contain at least one BellTermSpec")
        object.__setattr__(self, "terms", frozen_terms)
        if not isinstance(self.classical_bound, Real) or isinstance(
            self.classical_bound, (bool, np.bool_)
        ):
            raise ValueError("classical_bound must be finite")
        bound = float(self.classical_bound)
        if not math.isfinite(bound):
            raise ValueError("classical_bound must be finite")
        object.__setattr__(self, "classical_bound", bound)


@dataclass(frozen=True)
class OutcomeConventionSpec:
    local_dimension: int
    logical_outcomes: tuple[int, ...]
    leakage_outcome: Outcome = None
    measurement_basis_index_map: tuple[tuple[int, Outcome], ...] = ()
    root_phase_sign: int = 1

    def __post_init__(self) -> None:
        if not _is_metadata_integer(self.local_dimension) or self.local_dimension < 1:
            raise ValueError("local_dimension must be a positive integer")
        local_dimension = int(self.local_dimension)
        try:
            logical_outcomes = tuple(self.logical_outcomes)
        except TypeError as error:
            raise ValueError("logical_outcomes must be an iterable of integers") from error
        if (
            len(logical_outcomes) != local_dimension
            or not all(_is_metadata_integer(value) for value in logical_outcomes)
            or len(set(logical_outcomes)) != len(logical_outcomes)
        ):
            raise ValueError(
                "logical_outcomes must contain local_dimension unique integers"
            )
        frozen_logical_outcomes = tuple(int(value) for value in logical_outcomes)
        if self.leakage_outcome is not None and not _is_metadata_integer(
            self.leakage_outcome
        ):
            raise ValueError("leakage_outcome must be an integer or None")
        leakage_outcome = (
            None if self.leakage_outcome is None else int(self.leakage_outcome)
        )

        source = self.measurement_basis_index_map
        try:
            entries = tuple(source.items()) if isinstance(source, Mapping) else tuple(source)
            map_entries = tuple(tuple(entry) for entry in entries)
        except TypeError as error:
            raise ValueError(
                "measurement_basis_index_map must contain index/outcome pairs"
            ) from error
        seen_indices: set[int] = set()
        frozen_map: list[tuple[int, Outcome]] = []
        for entry in map_entries:
            if len(entry) != 2:
                raise ValueError(
                    "measurement_basis_index_map must contain index/outcome pairs"
                )
            index, outcome = entry
            if not _is_metadata_integer(index) or index < 0:
                raise ValueError("measurement basis indices must be nonnegative integers")
            index = int(index)
            if index in seen_indices:
                raise ValueError("measurement basis indices must be unique")
            if outcome is not None and not _is_metadata_integer(outcome):
                raise ValueError("measurement basis outcomes must use declared outcomes")
            outcome = None if outcome is None else int(outcome)
            if (
                outcome not in frozen_logical_outcomes
                and outcome != leakage_outcome
            ):
                raise ValueError("measurement basis outcomes must use declared outcomes")
            seen_indices.add(index)
            frozen_map.append((index, outcome))
        if not _is_metadata_integer(self.root_phase_sign):
            raise ValueError("root_phase_sign must be -1 or 1")
        root_phase_sign = int(self.root_phase_sign)
        if root_phase_sign not in {-1, 1}:
            raise ValueError("root_phase_sign must be -1 or 1")

        object.__setattr__(self, "local_dimension", local_dimension)
        object.__setattr__(self, "logical_outcomes", frozen_logical_outcomes)
        object.__setattr__(self, "leakage_outcome", leakage_outcome)
        object.__setattr__(self, "measurement_basis_index_map", tuple(frozen_map))
        object.__setattr__(self, "root_phase_sign", root_phase_sign)


@dataclass(frozen=True)
class LeakagePolicy:
    report_rate: bool
    compute_unconditional: bool
    compute_conditional: bool
    leakage_contribution: complex

    def __post_init__(self) -> None:
        for name in (
            "report_rate",
            "compute_unconditional",
            "compute_conditional",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        object.__setattr__(
            self,
            "leakage_contribution",
            _finite_complex(
                self.leakage_contribution,
                name="leakage_contribution",
            ),
        )


@dataclass(frozen=True)
class ExpectedValueSpec:
    ideal_bell_value: float
    absolute_tolerance: float

    def __post_init__(self) -> None:
        for name in ("ideal_bell_value", "absolute_tolerance"):
            value = getattr(self, name)
            if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be finite")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, normalized)
        if self.absolute_tolerance < 0:
            raise ValueError("absolute_tolerance must be nonnegative")


def _canonical_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("canonical serialization rejects nonfinite floats")
    normalized = 0.0 if value == 0 else value
    return format(normalized, ".17g")


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, (complex, np.complexfloating)):
        number = complex(value)
        return [_canonical_float(number.real), _canonical_float(number.imag)]
    if isinstance(value, (float, np.floating)):
        return _canonical_float(float(value))
    if isinstance(value, np.integer) and not isinstance(value, np.bool_):
        return int(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Mapping):
        entries = sorted(value.items(), key=lambda item: str(item[0]))
        keys = [str(key) for key, _ in entries]
        if len(keys) != len(set(keys)):
            raise ValueError("canonical serialization rejects duplicate string keys")
        return {
            str(key): _canonical_value(item_value)
            for key, item_value in entries
        }
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    raise ValueError(
        f"canonical serialization does not support {type(value).__name__}"
    )


@dataclass(frozen=True)
class ReferenceExperimentSpec:
    schema_version: str
    experiment_id: str
    state: LogicalStateSpec
    default_encoding_id: str
    observables: tuple[LocalObservableSpec, ...]
    bell_functional: BellFunctionalSpec
    outcome_convention: OutcomeConventionSpec
    leakage_policy: LeakagePolicy
    expected: ExpectedValueSpec
    expected_unique_measurement_settings: int

    def __post_init__(self) -> None:
        if self.schema_version != "reference-experiment-v1":
            raise ValueError("schema_version must be exactly reference-experiment-v1")
        _nonempty_string(self.experiment_id, name="experiment_id")
        if not isinstance(self.state, LogicalStateSpec):
            raise ValueError("state must be a LogicalStateSpec")
        if not isinstance(self.default_encoding_id, str) or (
            self.default_encoding_id not in ENCODINGS
        ):
            accepted = ", ".join(ENCODINGS)
            raise ValueError(
                "default_encoding_id must identify a known encoding; "
                f"accepted IDs: {accepted}"
            )
        try:
            frozen_observables = tuple(self.observables)
        except TypeError as error:
            raise ValueError(
                "observables must be an iterable of LocalObservableSpec"
            ) from error
        if not frozen_observables or not all(
            isinstance(observable, LocalObservableSpec)
            for observable in frozen_observables
        ):
            raise ValueError("observables must contain LocalObservableSpec values")
        labels = tuple(observable.label for observable in frozen_observables)
        if len(labels) != len(set(labels)):
            raise ValueError("observables must have unique observable labels")
        object.__setattr__(self, "observables", frozen_observables)
        if not isinstance(self.bell_functional, BellFunctionalSpec):
            raise ValueError("bell_functional must be a BellFunctionalSpec")
        if not isinstance(self.outcome_convention, OutcomeConventionSpec):
            raise ValueError("outcome_convention must be an OutcomeConventionSpec")
        if self.outcome_convention.local_dimension != self.state.local_dimension:
            raise ValueError("outcome convention and state dimensions must match")
        if not isinstance(self.leakage_policy, LeakagePolicy):
            raise ValueError("leakage_policy must be a LeakagePolicy")
        if not isinstance(self.expected, ExpectedValueSpec):
            raise ValueError("expected must be an ExpectedValueSpec")
        if (
            not _is_metadata_integer(self.expected_unique_measurement_settings)
            or self.expected_unique_measurement_settings < 1
        ):
            raise ValueError(
                "expected_unique_measurement_settings must be a positive integer"
            )
        object.__setattr__(
            self,
            "expected_unique_measurement_settings",
            int(self.expected_unique_measurement_settings),
        )

        known_labels = set(labels)
        valid_parties = set(self.state.party_order)
        for term in self.bell_functional.terms:
            seen_parties: set[int] = set()
            for factor in term.factors:
                if factor.party not in valid_parties:
                    raise ValueError(
                        f"factor party {factor.party} is not in state party_order"
                    )
                if factor.party in seen_parties:
                    raise ValueError("each term may have at most one factor per party")
                if factor.setting_label not in known_labels:
                    raise ValueError(
                        f"unknown observable {factor.setting_label!r} in Bell term"
                    )
                if factor.outcome_power % self.state.local_dimension == 0:
                    raise ValueError(
                        "factor outcome_power must be nonzero modulo local dimension"
                    )
                seen_parties.add(factor.party)

        actual_settings = len(self.measurement_settings())
        if actual_settings != self.expected_unique_measurement_settings:
            raise ValueError(
                "actual unique measurement setting count "
                f"{actual_settings} does not match expected "
                f"{self.expected_unique_measurement_settings}"
            )

    def observable(self, label: str) -> LocalObservableSpec:
        for observable in self.observables:
            if observable.label == label:
                return observable
        accepted = ", ".join(observable.label for observable in self.observables)
        raise ValueError(
            f"unknown observable label {label!r}; accepted labels: {accepted}"
        )

    def setting_for_term(self, term: BellTermSpec) -> tuple[str | None, ...]:
        if not isinstance(term, BellTermSpec):
            raise ValueError("term must be a BellTermSpec")
        settings: list[str | None] = [None] * self.state.num_parties
        for factor in term.factors:
            if factor.party not in self.state.party_order:
                raise ValueError(f"factor party {factor.party} is outside party_order")
            if settings[factor.party] is not None:
                raise ValueError("each term may have at most one factor per party")
            settings[factor.party] = factor.setting_label
        return tuple(settings)

    def powers_for_term(self, term: BellTermSpec) -> tuple[int, ...]:
        if not isinstance(term, BellTermSpec):
            raise ValueError("term must be a BellTermSpec")
        powers = [0] * self.state.num_parties
        for factor in term.factors:
            if factor.party not in self.state.party_order:
                raise ValueError(f"factor party {factor.party} is outside party_order")
            if powers[factor.party] != 0:
                raise ValueError("each term may have at most one factor per party")
            powers[factor.party] = factor.outcome_power
        return tuple(powers)

    def measurement_settings(self) -> tuple[tuple[str | None, ...], ...]:
        settings = {
            self.setting_for_term(term) for term in self.bell_functional.terms
        }
        return tuple(
            sorted(
                settings,
                key=lambda setting: tuple(
                    "" if label is None else label for label in setting
                ),
            )
        )

    def logical_bell_operator(self) -> np.ndarray:
        dimension = self.state.local_dimension**self.state.num_parties
        bell_operator = np.zeros((dimension, dimension), dtype=complex)
        identity = np.eye(self.state.local_dimension, dtype=complex)
        for term in self.bell_functional.terms:
            factors_by_party = {factor.party: factor for factor in term.factors}
            term_operator = np.asarray([[1 + 0j]])
            for party in self.state.party_order:
                factor = factors_by_party.get(party)
                local_operator = (
                    identity
                    if factor is None
                    else factor.logical_operator(
                        self.observable(factor.setting_label)
                    )
                )
                term_operator = np.kron(term_operator, local_operator)
            bell_operator += term.coefficient * term_operator
        return bell_operator

    def to_dict(self) -> dict[str, Any]:
        result = _canonical_value(self)
        if not isinstance(result, dict):
            raise ValueError("reference experiment must serialize as an object")
        return result

    def stable_hash(self) -> str:
        serialized = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _make_xz() -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((3, 3), dtype=complex)
    for column in range(3):
        x[(column + 1) % 3, column] = 1
    z = np.diag([_omega() ** outcome for outcome in range(3)])
    return x, z


def _lambda(power: int) -> complex:
    if power == 1:
        return complex(np.exp(1j * np.pi / 18))
    if power == 2:
        return complex(np.exp(-1j * np.pi / 18))
    raise ValueError("power must be 1 or 2")


def _measurement_observables(power: int) -> tuple[np.ndarray, ...]:
    if power not in {1, 2} or isinstance(power, (bool, np.bool_)):
        raise ValueError("power must be 1 or 2")
    x, z = _make_xz()
    omega = _omega()
    result: list[np.ndarray] = []
    for setting in range(3):
        observable = np.zeros((3, 3), dtype=complex)
        for k in range(3):
            phase = omega ** (power * setting * k)
            phase *= omega ** (power * k * (k + 1))
            xz = x @ np.linalg.matrix_power(z, k)
            observable += phase * np.linalg.matrix_power(xz, power)
        result.append(_lambda(power) * observable / math.sqrt(3))
    return tuple(result)


def _root_expectation_scale(
    observable: LocalObservableSpec,
    desired_operator: Any,
    outcome_power: int,
) -> complex:
    if not isinstance(observable, LocalObservableSpec):
        raise ValueError("observable must be a LocalObservableSpec")
    if not _is_metadata_integer(outcome_power):
        raise ValueError("outcome_power must be an integer")
    try:
        desired = np.asarray(desired_operator, dtype=complex)
    except (TypeError, ValueError) as error:
        raise ValueError("desired_operator must be a numeric 3x3 matrix") from error
    if desired.shape != (3, 3) or not np.all(np.isfinite(desired)):
        raise ValueError("desired_operator must be a finite numeric 3x3 matrix")

    basis, _ = observable.ordered_eigenbasis()
    in_measurement_basis = basis.conj().T @ desired @ basis
    diagonal = np.diag(in_measurement_basis)
    if not np.allclose(
        in_measurement_basis,
        np.diag(diagonal),
        rtol=0,
        atol=1e-7,
    ):
        raise ValueError("desired operator must be diagonal in measurement eigenbasis")
    roots = np.asarray(
        [
            _omega() ** ((outcome_power * outcome) % 3)
            for outcome in range(3)
        ]
    )
    scales = diagonal / roots
    common_scale = complex(np.mean(scales))
    if not np.allclose(scales, common_scale, rtol=0, atol=1e-7):
        raise ValueError(
            "desired diagonal must equal one common scale times qutrit roots"
        )
    return common_scale


def _factor(
    party: int,
    label: str,
    observable: LocalObservableSpec,
    desired_operator: np.ndarray,
    outcome_power: int,
) -> BellFactorSpec:
    return BellFactorSpec(
        party=party,
        setting_label=label,
        outcome_power=outcome_power,
        operator_scale=_root_expectation_scale(
            observable,
            desired_operator,
            outcome_power,
        ),
    )


_OUTCOME_CONVENTION = OutcomeConventionSpec(
    local_dimension=3,
    logical_outcomes=(0, 1, 2),
    leakage_outcome=None,
    measurement_basis_index_map=((0, 0), (1, 1), (2, 2), (3, None)),
    root_phase_sign=1,
)

_LEAKAGE_POLICY = LeakagePolicy(
    report_rate=True,
    compute_unconditional=True,
    compute_conditional=True,
    leakage_contribution=0j,
)


def _observable_map(candidate: str) -> dict[str, LocalObservableSpec]:
    x, z = _make_xz()
    matrices: dict[str, np.ndarray] = {
        f"A{setting}": matrix
        for setting, matrix in enumerate(_measurement_observables(1))
    }
    matrices.update(
        {
            f"B{setting}": z @ np.linalg.matrix_power(x, setting)
            for setting in range(3)
        }
    )
    if candidate == "ghz3":
        matrices.update({"C0": z, "C1": z @ x})
    elif candidate == "ame43":
        matrices.update({"C0": z, "C1": x, "D0": z, "D1": z @ x})
    elif candidate != "two_qutrit":
        raise ValueError(
            f"unknown reference experiment candidate {candidate!r}"
        )
    return {
        label: LocalObservableSpec(label, matrix)
        for label, matrix in matrices.items()
    }


def _measured_factor(
    observables: Mapping[str, LocalObservableSpec],
    party: int,
    label: str,
    outcome_power: int,
) -> BellFactorSpec:
    observable = observables[label]
    desired = np.linalg.matrix_power(observable.as_array(), outcome_power)
    return _factor(
        party,
        label,
        observable,
        desired,
        outcome_power,
    )


def _two_qutrit_terms(
    observables: Mapping[str, LocalObservableSpec],
) -> tuple[BellTermSpec, ...]:
    omega = _omega()
    terms: list[BellTermSpec] = []
    for power in (1, 2):
        a_power = _measurement_observables(power)
        lam = _lambda(power)
        term_specs = (
            (0, 1 / (lam * math.sqrt(3)), lambda _a: 1),
            (
                1,
                1 / (lam * omega ** (2 * power) * math.sqrt(3)),
                lambda a: omega ** (-power * a),
            ),
            (
                2,
                1 / (lam * math.sqrt(3)),
                lambda a: omega ** (-2 * power * a),
            ),
        )
        for b, coefficient, phase in term_specs:
            for a in range(3):
                a_label = f"A{a}"
                b_label = f"B{b}"
                terms.append(
                    BellTermSpec(
                        coefficient * phase(a),
                        (
                            _factor(
                                0,
                                a_label,
                                observables[a_label],
                                a_power[a],
                                power,
                            ),
                            _measured_factor(
                                observables,
                                1,
                                b_label,
                                power,
                            ),
                        ),
                    )
                )
    return tuple(terms)


def _ghz3_terms(
    observables: Mapping[str, LocalObservableSpec],
) -> tuple[BellTermSpec, ...]:
    omega = _omega()
    terms: list[BellTermSpec] = []
    for power in (1, 2):
        a_power = _measurement_observables(power)
        lam = _lambda(power)
        first = 1 / (lam * math.sqrt(3))
        second = omega**power / (2 * lam * math.sqrt(3))
        term_specs = (
            (0, 0, first, lambda _a: 1),
            (1, 0, second, lambda a: omega ** (-power * a)),
            (2, 0, first, lambda a: omega ** (-2 * power * a)),
            (0, 1, second, lambda a: omega ** (-power * a)),
        )
        for b, c, coefficient, phase in term_specs:
            for a in range(3):
                a_label = f"A{a}"
                b_label = f"B{b}"
                c_label = f"C{c}"
                terms.append(
                    BellTermSpec(
                        coefficient * phase(a),
                        (
                            _factor(
                                0,
                                a_label,
                                observables[a_label],
                                a_power[a],
                                power,
                            ),
                            _measured_factor(
                                observables,
                                1,
                                b_label,
                                power,
                            ),
                            _measured_factor(
                                observables,
                                2,
                                c_label,
                                power,
                            ),
                        ),
                    )
                )
    return tuple(terms)


def _ame43_terms(
    observables: Mapping[str, LocalObservableSpec],
) -> tuple[BellTermSpec, ...]:
    omega = _omega()
    terms: list[BellTermSpec] = []

    def measured(party: int, label: str, power: int) -> BellFactorSpec:
        return _measured_factor(observables, party, label, power)

    for power in (1, 2):
        a_power = _measurement_observables(power)
        lam = _lambda(power)
        first = 1 / (math.sqrt(3) * lam)
        second = 1 / (
            2 * math.sqrt(3) * lam * omega ** (2 * power)
        )

        for a in range(3):
            a_label = f"A{a}"
            a_factor = _factor(
                0,
                a_label,
                observables[a_label],
                a_power[a],
                power,
            )
            terms.append(
                BellTermSpec(
                    first,
                    (
                        a_factor,
                        measured(1, "B0", power),
                        measured(3, "D0", power),
                    ),
                )
            )

        for a, phase in enumerate(
            (1, omega ** (-2 * power), omega ** (-power))
        ):
            a_label = f"A{a}"
            terms.append(
                BellTermSpec(
                    first * phase,
                    (
                        _factor(
                            0,
                            a_label,
                            observables[a_label],
                            a_power[a],
                            power,
                        ),
                        measured(1, "B2", power),
                        measured(2, "C0", 2 * power),
                        measured(3, "D0", power),
                    ),
                )
            )

        for a, phase in enumerate(
            (1, omega ** (-power), omega ** (-2 * power))
        ):
            a_label = f"A{a}"
            a_factor = _factor(
                0,
                a_label,
                observables[a_label],
                a_power[a],
                power,
            )
            terms.extend(
                (
                    BellTermSpec(
                        second * phase,
                        (
                            a_factor,
                            measured(1, "B1", power),
                            measured(2, "C0", power),
                            measured(3, "D0", power),
                        ),
                    ),
                    BellTermSpec(
                        second * phase,
                        (
                            a_factor,
                            measured(1, "B0", power),
                            measured(2, "C0", 2 * power),
                            measured(3, "D1", power),
                        ),
                    ),
                )
            )

        terms.append(
            BellTermSpec(
                1,
                (
                    measured(1, "B0", power),
                    measured(2, "C1", power),
                    measured(3, "D0", 2 * power),
                ),
            )
        )
    return tuple(terms)


def _reference_spec(candidate: str) -> ReferenceExperimentSpec:
    observables = _observable_map(candidate)
    if candidate == "two_qutrit":
        state = LogicalStateSpec(
            "two_qutrit", 3, 2, (0, 1), ((0, 1, 1),)
        )
        terms = _two_qutrit_terms(observables)
        functional_id = "two_qutrit_bell_v1"
        normalization = "two_qutrit_pdf_v1"
        classical_bound = 6 * math.cos(math.pi / 9)
        classical_bound_source = "analytic"
        ideal_value = 6.0
        expected_settings = 9
    elif candidate == "ghz3":
        state = LogicalStateSpec(
            "ghz3", 3, 3, (0, 1, 2), ((0, 1, 1), (0, 2, 1))
        )
        terms = _ghz3_terms(observables)
        functional_id = "ghz3_bell_v1"
        normalization = "ghz3_graph_v1"
        classical_bound = 6 * math.cos(math.pi / 9)
        classical_bound_source = "numeric_bruteforce"
        ideal_value = 6.0
        expected_settings = 12
    elif candidate == "ame43":
        state = LogicalStateSpec(
            "ame43",
            3,
            4,
            (0, 1, 2, 3),
            ((0, 1, 1), (0, 3, 1), (1, 2, 1), (2, 3, 2)),
        )
        terms = _ame43_terms(observables)
        functional_id = "ame43_bell_v1"
        normalization = "ame43_pdf_v1"
        classical_bound = 7.63816
        classical_bound_source = "pdf"
        ideal_value = 8.0
        expected_settings = 13
    else:
        raise ValueError(
            f"unknown reference experiment candidate {candidate!r}"
        )

    return ReferenceExperimentSpec(
        schema_version="reference-experiment-v1",
        experiment_id=candidate,
        state=state,
        default_encoding_id="canonical_ez",
        observables=tuple(observables.values()),
        bell_functional=BellFunctionalSpec(
            functional_id=functional_id,
            normalization=normalization,
            terms=terms,
            classical_bound=classical_bound,
            classical_bound_source=classical_bound_source,
        ),
        outcome_convention=_OUTCOME_CONVENTION,
        leakage_policy=_LEAKAGE_POLICY,
        expected=ExpectedValueSpec(
            ideal_bell_value=ideal_value,
            absolute_tolerance=1e-10,
        ),
        expected_unique_measurement_settings=expected_settings,
    )


REFERENCE_EXPERIMENTS: Mapping[str, ReferenceExperimentSpec] = MappingProxyType(
    {
        "two_qutrit": _reference_spec("two_qutrit"),
        "ghz3": _reference_spec("ghz3"),
        "ame43": _reference_spec("ame43"),
    }
)

_REFERENCE_EXPERIMENT_ALIASES: Mapping[str, str] = MappingProxyType(
    {"2qutrit": "two_qutrit"}
)


def list_reference_experiments() -> tuple[str, ...]:
    return tuple(REFERENCE_EXPERIMENTS)


def get_reference_experiment(experiment_id: str) -> ReferenceExperimentSpec:
    normalized_id = experiment_id.strip() if isinstance(experiment_id, str) else ""
    canonical_id = _REFERENCE_EXPERIMENT_ALIASES.get(normalized_id, normalized_id)
    try:
        return REFERENCE_EXPERIMENTS[canonical_id]
    except KeyError as error:
        accepted = ", ".join(REFERENCE_EXPERIMENTS)
        raise ValueError(
            f"unknown reference experiment ID {experiment_id!r}; "
            f"accepted IDs: {accepted}"
        ) from error
