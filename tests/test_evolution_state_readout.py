"""Check offline decoding and routing for encoded graph states."""

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from qudits_on_qubits.benchmarks.theta_continuation.hardware_compilation import _probabilities
from qudits_on_qubits.evolution.encodings import canonical_encoding
from qudits_on_qubits.evolution.state_readout import compile_readout
from qudits_on_qubits.evolution.state_workloads import build_workload


def _workload():
    diagonal = np.ones(16, dtype=complex)
    for a in range(3):
        for b in range(3):
            diagonal[a + 4*b] = np.exp(2j * np.pi * a*b/3)
    gate = QuantumCircuit(4)
    gate.unitary(np.diag(diagonal), range(4))
    gate = transpile(gate, basis_gates=["u", "cz"], optimization_level=0,
                     seed_transpiler=0, num_processes=1)
    encoding = canonical_encoding(3)
    return build_workload("two_qutrit", encoding, gate), encoding


def test_readout_preserves_state_and_terminal_probabilities():
    circuit, encoding = _workload()
    result = compile_readout("two_qutrit", circuit, encoding, [(0, 1), (1, 2), (2, 3)])
    measured = result["circuits"]["routed_with_measurement"]
    expected = Statevector(result["circuits"]["decoded"]).probabilities()
    np.testing.assert_allclose(_probabilities(measured), expected, atol=1e-10)
    assert result["metadata"]["decoded_metrics"]["is_valid"]
    assert result["metadata"]["preparation"]["n_cz"] == circuit.count_ops().get("cz", 0)
    assert measured.count_ops()["measure"] == 4
    assert not result["metadata"]["physical_backend_validated"]
    allowed = {(0, 1), (1, 2), (2, 3)}
    for item in measured.data:
        if len(item.qubits) == 2:
            assert tuple(sorted(measured.find_bit(q).index for q in item.qubits)) in allowed


def test_readout_rejects_invalid_source_and_graph():
    circuit, encoding = _workload()
    with pytest.raises(ValueError, match="source state"):
        compile_readout("two_qutrit", QuantumCircuit(4), encoding, [(0, 1), (1, 2), (2, 3)])
    with pytest.raises(ValueError, match="connect"):
        compile_readout("two_qutrit", circuit, encoding, [(0, 1), (2, 3)])


def test_noncanonical_encoding_decodes_to_canonical_state():
    circuit, canonical = _workload()
    unitary = np.array([[1, 1, 0, 0], [1, -1, 0, 0],
                        [0, 0, 1, 1], [0, 0, 1, -1]], dtype=complex) / np.sqrt(2)
    encoding = unitary @ canonical
    encoded = circuit.copy()
    encoded.unitary(unitary, [0, 1])
    encoded.unitary(unitary, [2, 3])
    encoded = transpile(encoded, basis_gates=["u", "cz"], optimization_level=0,
                        seed_transpiler=0, num_processes=1)
    result = compile_readout("two_qutrit", encoded, encoding, [(0, 1), (1, 2), (2, 3)])
    decoded = Statevector(result["circuits"]["decoded"])
    assert decoded.equiv(Statevector(circuit))
    np.testing.assert_allclose(_probabilities(result["circuits"]["routed_with_measurement"]),
                               decoded.probabilities(), atol=1e-10)


@pytest.mark.parametrize("name,count", [("ghz3", 9), ("ame43", 19)])
def test_saved_state_readout(name, count):
    from pathlib import Path
    from qiskit import qpy

    folder = Path(__file__).resolve().parents[1] / "artifacts/evolution/openevolve" / f"{name}_mock_20261102"
    if not (folder / "best.qpy").exists() or not (folder / "encoding.npy").exists():
        pytest.skip("The saved offline baseline is not available")
    with (folder / "best.qpy").open("rb") as stream:
        circuit, = qpy.load(stream)
    encoding = np.load(folder / "encoding.npy", allow_pickle=False)
    edges = [(i, i + 1) for i in range(circuit.num_qubits - 1)]
    result = compile_readout(name, circuit, encoding, edges)
    assert result["metadata"]["preparation"]["n_cz"] == count
    decoded = Statevector(result["circuits"]["decoded"])
    np.testing.assert_allclose(_probabilities(result["circuits"]["routed_with_measurement"]),
                               decoded.probabilities(), atol=1e-10)


def test_readout_accepts_one_encoding_per_qutrit():
    from qudits_on_qubits.evolution.skeleton_fit import fit_skeleton
    fit = fit_skeleton("two_qutrit", ((0, 2), (0, 3), (1, 3)), restarts=16)
    assert fit.encodings.shape == (2, 4, 3)
    result = compile_readout("two_qutrit", fit.circuit, fit.encodings, [(0, 1), (1, 2), (2, 3)])
    assert result["metadata"]["decoded_metrics"]["is_valid"]
    assert len(result["metadata"]["decoder_per_party"]) == 2
    assert result["metadata"]["decoded"]["n_cz"] == 3 + fit.metrics["decoder_cz_count"]
