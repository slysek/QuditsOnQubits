from __future__ import annotations

import json

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qudits_on_qubits.bell_measurements import canonical_Ez
from qudits_on_qubits.experiments.artifacts import BasisArtifacts
from qudits_on_qubits.experiments.preparation import metadata_summary, prepare_measurements


@pytest.mark.parametrize(
    ("state", "qubits", "pairs"),
    [
        ("two_qutrit", 4, ((0, 1), (2, 3))),
        ("ghz3", 6, ((0, 1), (2, 3), (4, 5))),
        ("ame43", 8, ((0, 1), (2, 3), (4, 5), (6, 7))),
    ],
)
def test_prepares_measured_circuits_and_json_safe_summary(state, qubits, pairs, tmp_path):
    artifacts = BasisArtifacts(
        directory=tmp_path,
        state=state,
        state_circuit=QuantumCircuit(qubits),
        encoding=canonical_Ez(),
        source_paths={},
        source_hashes={},
        provenance={"candidate": state},
    )

    prepared = prepare_measurements(artifacts)
    summary = metadata_summary(prepared.metadata)

    assert prepared.circuits
    assert len(prepared.circuits) == len(prepared.metadata["setting_by_circuit_index"])
    assert all(circuit.num_clbits > 0 for circuit in prepared.circuits)
    assert summary["state"] == state
    assert summary["candidate"] == state
    assert summary["qutrit_qubits"] == [list(pair) for pair in pairs]
    assert summary["circuit_count"] == len(prepared.circuits)
    json.dumps(summary)


def test_prepared_measurements_are_immutable(tmp_path):
    artifacts = BasisArtifacts(
        directory=tmp_path,
        state="two_qutrit",
        state_circuit=QuantumCircuit(4),
        encoding=np.eye(4, 3),
        source_paths={},
        source_hashes={},
        provenance={},
    )

    prepared = prepare_measurements(artifacts)

    with pytest.raises(TypeError):
        prepared.metadata["candidate"] = "changed"
