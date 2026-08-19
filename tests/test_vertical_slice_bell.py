"""Bell-reference adapter tests for generic vertical-slice contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np
import pytest

from qudits_on_qubits.reference_experiments import (
    get_encoding,
    get_reference_experiment,
)
from qudits_on_qubits.vertical_slice import (
    BellPostprocessorSpec,
    BellReferenceCircuitSpec,
    IsometricQuditEncoding,
    LogicalOutcome,
    SpecValidationError,
    canonical_qutrit_encoding,
)


def _dimension_four_encoding() -> IsometricQuditEncoding:
    return IsometricQuditEncoding(
        encoding_id="identity-d4",
        logical_dimension=4,
        physical_qubits=2,
        matrix=np.eye(4, dtype=complex),
        decode_table=tuple(LogicalOutcome(index, False) for index in range(4)),
    )


def test_canonical_qutrit_encoding_adapts_frozen_registry_convention() -> None:
    registry_encoding = get_encoding("canonical_ez")
    convention = get_reference_experiment("two_qutrit").outcome_convention

    first = canonical_qutrit_encoding()
    second = canonical_qutrit_encoding()

    assert first.encoding_id == registry_encoding.encoding_id
    assert first.logical_dimension == registry_encoding.logical_dimension
    assert first.physical_qubits == registry_encoding.physical_qubits_per_qutrit
    np.testing.assert_array_equal(first.isometry(), np.asarray(registry_encoding.isometry))
    assert first.decode_table == tuple(
        LogicalOutcome(value, value is None)
        for _, value in convention.measurement_basis_index_map
    )
    assert first == second
    assert first.stable_hash() == second.stable_hash()
    assert len(first.stable_hash()) == 64


def test_reference_spec_normalizes_alias_and_has_stable_manifest() -> None:
    spec = BellReferenceCircuitSpec("2qutrit")
    payload = spec.to_manifest_dict()

    assert spec.kind == "bell.reference"
    assert spec.reference_id == "two_qutrit"
    assert spec.circuit_id == "two_qutrit"
    assert spec.logical_dimensions == (3, 3)
    assert payload == {
        "kind": "bell.reference",
        "schema_version": "bell-reference-circuit-v1",
        "reference_id": "two_qutrit",
        "logical_dimensions": [3, 3],
        "reference_spec_hash": get_reference_experiment("two_qutrit").stable_hash(),
    }
    assert spec.stable_hash() == BellReferenceCircuitSpec("two_qutrit").stable_hash()
    assert len(spec.stable_hash()) == 64
    with pytest.raises(FrozenInstanceError):
        spec.reference_id = "ghz3"  # type: ignore[misc]


def test_reference_spec_rejects_unsupported_reference() -> None:
    with pytest.raises(SpecValidationError, match="unknown reference experiment"):
        BellReferenceCircuitSpec("missing")


def test_two_qutrit_preparation_builds_source_and_measured_executables() -> None:
    prepared = BellReferenceCircuitSpec("two_qutrit").prepare(
        canonical_qutrit_encoding()
    )

    assert len(prepared.source_circuits) == 1
    source = prepared.source_circuits[0]
    assert source.num_qubits == 4
    assert source.num_clbits == 0
    assert any(item.operation.name in {"initialize", "state_preparation"} for item in source.data)
    assert len(prepared.executable_circuits) == 9
    assert all(circuit.num_qubits == 4 for circuit in prepared.executable_circuits)
    assert all(circuit.num_clbits == 4 for circuit in prepared.executable_circuits)
    assert prepared.postprocessor.kind == "bell.reference"
    json.dumps(dict(prepared.provenance), allow_nan=False)


def test_preparation_rejects_wrong_dimension_before_circuit_construction() -> None:
    with pytest.raises(SpecValidationError, match="logical dimension"):
        BellReferenceCircuitSpec("two_qutrit").prepare(_dimension_four_encoding())


@pytest.mark.parametrize(
    ("reference_id", "parties", "circuit_count"),
    (("ghz3", 3, 12), ("ame43", 4, 13)),
)
def test_preparation_keeps_other_registry_references_structural(
    reference_id: str, parties: int, circuit_count: int
) -> None:
    prepared = BellReferenceCircuitSpec(reference_id).prepare(
        canonical_qutrit_encoding()
    )

    assert prepared.source_circuits[0].num_qubits == 2 * parties
    assert len(prepared.executable_circuits) == circuit_count
    assert all(
        circuit.num_clbits == 2 * parties
        for circuit in prepared.executable_circuits
    )


def test_postprocessor_evaluates_ordered_counts_as_json_safe_result() -> None:
    postprocessor = BellReferenceCircuitSpec("two_qutrit").prepare(
        canonical_qutrit_encoding()
    ).postprocessor
    counts = tuple({"0000": 8, "0011": 2} for _ in postprocessor.settings)

    result = postprocessor.evaluate(counts)

    assert result["benchmark"] == "two_qutrit"
    assert set(result) == {
        "benchmark",
        "bell_unconditional",
        "bell_conditional",
        "leakage_rate",
        "total_shots",
        "accepted_shots",
        "circuit_count",
    }
    assert set(result["bell_unconditional"]) == {"real", "imag"}
    assert set(result["bell_conditional"]) == {"real", "imag"}
    assert result["total_shots"] == 90
    assert result["accepted_shots"] == 72
    assert result["leakage_rate"] == pytest.approx(0.2)
    assert result["circuit_count"] == 9
    assert result["bell_unconditional"]["real"] == pytest.approx(
        0.8 * result["bell_conditional"]["real"]
    )
    json.dumps(result, allow_nan=False)


def test_postprocessor_validates_counts_and_manifest_is_stable() -> None:
    postprocessor = BellReferenceCircuitSpec("two_qutrit").prepare(
        canonical_qutrit_encoding()
    ).postprocessor

    assert isinstance(postprocessor, BellPostprocessorSpec)
    with pytest.raises(SpecValidationError, match="exactly 9"):
        postprocessor.evaluate(({"0000": 1},))
    bad_counts = [{"0000": 1} for _ in postprocessor.settings]
    bad_counts[0] = {"0000": -1}
    with pytest.raises(SpecValidationError, match="non-negative integers"):
        postprocessor.evaluate(bad_counts)

    payload = postprocessor.to_manifest_dict()
    assert payload["schema_version"] == "bell-postprocessor-v1"
    assert payload["reference_spec_hash"] == get_reference_experiment(
        "two_qutrit"
    ).stable_hash()
    assert payload["settings"] == [list(setting) for setting in postprocessor.settings]
    assert payload["qutrit_bit_indices"] == [
        [list(pair) for pair in indices]
        for indices in postprocessor.qutrit_bit_indices
    ]
    assert payload == postprocessor.to_manifest_dict()
    json.dumps(payload, allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        postprocessor.reference_id = "ghz3"  # type: ignore[misc]


def test_generic_contract_models_do_not_import_bell_domain_modules() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "qudits_on_qubits"
        / "vertical_slice"
        / "models.py"
    ).read_text(encoding="utf-8")

    assert "bell_measurements" not in source
    assert "bell_functionals" not in source
    assert "reference_experiments" not in source
