"""Adapters from audited Bell references to vertical-slice contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

from ..bell_functionals import candidate_statevector
from ..bell_measurements import (
    build_sampler_circuits_for_candidate,
    evaluate_reference_bell_values_from_counts,
)
from ..reference_experiments import get_encoding, get_reference_experiment
from .models import (
    IsometricQuditEncoding,
    JsonValue,
    LogicalOutcome,
    PreparedExperiment,
    QuditEncoding,
    SpecValidationError,
)

_CIRCUIT_SCHEMA = "bell-reference-circuit-v1"
_POSTPROCESSOR_SCHEMA = "bell-postprocessor-v1"

Setting = tuple[str | None, ...]
QutritBitIndices = tuple[tuple[int, int], ...]


def _reference(reference_id: object):
    try:
        return get_reference_experiment(reference_id)  # type: ignore[arg-type]
    except ValueError as error:
        raise SpecValidationError(str(error)) from error


def _stable_hash(payload: Mapping[str, JsonValue]) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def canonical_qutrit_encoding() -> IsometricQuditEncoding:
    """Adapt canonical registry encoding and its frozen outcome convention."""

    registry_encoding = get_encoding("canonical_ez")
    reference = get_reference_experiment("two_qutrit")
    physical_dimension = 2**registry_encoding.physical_qubits_per_qutrit
    outcome_map = dict(reference.outcome_convention.measurement_basis_index_map)
    if set(outcome_map) != set(range(physical_dimension)):
        raise SpecValidationError(
            "reference outcome convention must map every physical basis index"
        )
    decode_table = tuple(
        LogicalOutcome(value=outcome_map[index], leaked=outcome_map[index] is None)
        for index in range(physical_dimension)
    )
    return IsometricQuditEncoding(
        encoding_id=registry_encoding.encoding_id,
        logical_dimension=registry_encoding.logical_dimension,
        physical_qubits=registry_encoding.physical_qubits_per_qutrit,
        matrix=np.asarray(registry_encoding.isometry, dtype=complex),
        decode_table=decode_table,
    )


@dataclass(frozen=True)
class BellPostprocessorSpec:
    """Ordered Bell counts adapter with immutable JSON-safe metadata."""

    reference_id: str
    settings: tuple[Setting, ...]
    qutrit_bit_indices: tuple[QutritBitIndices, ...]
    reference_spec_hash: str = field(init=False)

    @property
    def kind(self) -> str:
        return "bell.reference"

    def __post_init__(self) -> None:
        reference = _reference(self.reference_id)
        object.__setattr__(self, "reference_id", reference.experiment_id)
        object.__setattr__(self, "reference_spec_hash", reference.stable_hash())
        try:
            settings = tuple(tuple(setting) for setting in self.settings)
            indices_by_circuit = tuple(
                tuple(tuple(pair) for pair in indices)
                for indices in self.qutrit_bit_indices
            )
        except TypeError as error:
            raise SpecValidationError(
                "settings and qutrit_bit_indices must be iterable"
            ) from error
        if len(settings) != len(indices_by_circuit) or not settings:
            raise SpecValidationError(
                "settings and qutrit_bit_indices must have the same nonzero length"
            )
        if len(set(settings)) != len(settings):
            raise SpecValidationError("settings must be unique")
        if settings != reference.measurement_settings():
            raise SpecValidationError(
                "settings must match the ordered reference measurement settings"
            )
        for setting, indices in zip(settings, indices_by_circuit):
            if len(setting) != reference.state.num_parties:
                raise SpecValidationError(
                    "each setting must contain one label per reference party"
                )
            if any(label is not None and not isinstance(label, str) for label in setting):
                raise SpecValidationError("setting labels must be strings or None")
            if len(indices) != reference.state.num_parties:
                raise SpecValidationError(
                    "qutrit bit indices must contain one pair per reference party"
                )
            if any(
                len(pair) != 2
                or any(
                    isinstance(index, bool) or not isinstance(index, int) or index < 0
                    for index in pair
                )
                for pair in indices
            ):
                raise SpecValidationError(
                    "qutrit bit indices must be pairs of non-negative integers"
                )
        object.__setattr__(self, "settings", settings)
        object.__setattr__(self, "qutrit_bit_indices", indices_by_circuit)

    def evaluate(
        self, counts_by_circuit: Sequence[Mapping[str, int]]
    ) -> Mapping[str, JsonValue]:
        if isinstance(counts_by_circuit, (str, bytes)) or not isinstance(
            counts_by_circuit, Sequence
        ):
            raise SpecValidationError("counts_by_circuit must be a sequence")
        if len(counts_by_circuit) != len(self.settings):
            raise SpecValidationError(
                f"counts_by_circuit must contain exactly {len(self.settings)} mappings"
            )
        copied_counts: list[dict[str, int]] = []
        for counts in counts_by_circuit:
            if not isinstance(counts, Mapping):
                raise SpecValidationError("each counts entry must be a mapping")
            copied: dict[str, int] = {}
            for bitstring, count in counts.items():
                if not isinstance(bitstring, str):
                    raise SpecValidationError("count bitstrings must be strings")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise SpecValidationError("counts must be non-negative integers")
                copied[bitstring] = count
            copied_counts.append(copied)

        evaluated = evaluate_reference_bell_values_from_counts(
            self.reference_id,
            dict(zip(self.settings, copied_counts)),
            dict(zip(self.settings, self.qutrit_bit_indices)),
        )
        values = (evaluated.unconditional, evaluated.conditional)
        if any(
            not math.isfinite(value.real) or not math.isfinite(value.imag)
            for value in values
        ):
            raise SpecValidationError("Bell evaluation returned non-finite values")
        return {
            "benchmark": self.reference_id,
            "bell_unconditional": {
                "real": float(evaluated.unconditional.real),
                "imag": float(evaluated.unconditional.imag),
            },
            "bell_conditional": {
                "real": float(evaluated.conditional.real),
                "imag": float(evaluated.conditional.imag),
            },
            "leakage_rate": float(evaluated.leakage_rate),
            "total_shots": evaluated.total_shots,
            "accepted_shots": evaluated.accepted_shots,
            "circuit_count": len(self.settings),
        }

    def to_manifest_dict(self) -> Mapping[str, JsonValue]:
        return {
            "kind": self.kind,
            "schema_version": _POSTPROCESSOR_SCHEMA,
            "reference_id": self.reference_id,
            "settings": [list(setting) for setting in self.settings],
            "qutrit_bit_indices": [
                [list(pair) for pair in indices]
                for indices in self.qutrit_bit_indices
            ],
            "reference_spec_hash": self.reference_spec_hash,
        }

    def stable_hash(self) -> str:
        return _stable_hash(self.to_manifest_dict())


@dataclass(frozen=True)
class BellReferenceCircuitSpec:
    """Vertical-slice circuit spec backed by audited Bell registry entries."""

    reference_id: str

    @property
    def kind(self) -> str:
        return "bell.reference"

    @property
    def circuit_id(self) -> str:
        return self.reference_id

    @property
    def logical_dimensions(self) -> tuple[int, ...]:
        reference = get_reference_experiment(self.reference_id)
        return (reference.state.local_dimension,) * reference.state.num_parties

    def __post_init__(self) -> None:
        reference = _reference(self.reference_id)
        object.__setattr__(self, "reference_id", reference.experiment_id)

    def prepare(self, encoding: QuditEncoding) -> PreparedExperiment:
        if not isinstance(encoding, QuditEncoding):
            raise SpecValidationError("encoding must implement QuditEncoding")
        reference = get_reference_experiment(self.reference_id)
        if encoding.logical_dimension != reference.state.local_dimension:
            raise SpecValidationError(
                "encoding logical dimension must match reference local dimension"
            )
        if encoding.physical_qubits != 2:
            raise SpecValidationError(
                "audited Bell measurement adapter requires two physical qubits per qutrit"
            )
        try:
            isometry = np.asarray(encoding.isometry(), dtype=complex)
        except (TypeError, ValueError, OverflowError) as error:
            raise SpecValidationError("encoding isometry must be numeric") from error
        expected_shape = (2**encoding.physical_qubits, encoding.logical_dimension)
        if isometry.shape != expected_shape:
            raise SpecValidationError(
                f"encoding isometry must have shape {expected_shape}"
            )

        from qiskit import QuantumCircuit

        state = candidate_statevector(self.reference_id, isometry)
        qubit_count = reference.state.num_parties * encoding.physical_qubits
        source = QuantumCircuit(qubit_count, name=f"{self.reference_id}_source")
        source.initialize(state.data, tuple(range(qubit_count)))
        executable, metadata = build_sampler_circuits_for_candidate(
            self.reference_id,
            source,
            isometry,
            d=reference.state.local_dimension,
            add_measurements=True,
            sort_settings=True,
        )
        settings = tuple(
            tuple(setting) for setting in metadata["setting_by_circuit_index"]
        )
        indices = tuple(
            tuple(
                tuple(pair)
                for pair in metadata["qutrit_bit_indices_by_setting"][setting]
            )
            for setting in settings
        )
        postprocessor = BellPostprocessorSpec(self.reference_id, settings, indices)
        provenance: Mapping[str, JsonValue] = {
            "adapter": "bell.reference",
            "reference_id": self.reference_id,
            "reference_spec_hash": reference.stable_hash(),
            "encoding_id": encoding.encoding_id,
            "encoding_hash": encoding.stable_hash(),
            "state_preparation": "candidate_statevector.initialize",
            "source_circuit_count": 1,
            "executable_circuit_count": len(executable),
        }
        return PreparedExperiment(
            source_circuits=(source,),
            executable_circuits=tuple(executable),
            postprocessor=postprocessor,
            provenance=provenance,
        )

    def to_manifest_dict(self) -> Mapping[str, JsonValue]:
        reference = get_reference_experiment(self.reference_id)
        return {
            "kind": self.kind,
            "schema_version": _CIRCUIT_SCHEMA,
            "reference_id": self.reference_id,
            "logical_dimensions": list(self.logical_dimensions),
            "reference_spec_hash": reference.stable_hash(),
        }

    def stable_hash(self) -> str:
        return _stable_hash(self.to_manifest_dict())


__all__ = [
    "BellPostprocessorSpec",
    "BellReferenceCircuitSpec",
    "canonical_qutrit_encoding",
]
