"""Runtime assets must work independently of a source checkout."""

from importlib.resources import files

import numpy as np
import pytest
from qiskit import qpy
from qiskit.quantum_info import Operator, Statevector

from qudits_on_qubits import create_ame_circuit


@pytest.mark.parametrize("dimension", [3, 4])
def test_bundled_gates_and_core_imports(dimension, monkeypatch, tmp_path):
    from qudits_on_qubits.core import project_paths

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project_paths, "_REPO_ROOT", tmp_path)
    directory = files("qudits_on_qubits").joinpath("quantum_circuits")
    names = [f"Fgate{dimension}.qpy"]
    names += ["CZgate3.qpy" if dimension == 3 else "CZgate4cor.qpy"]
    names += [
        f"{gate}gate{dimension}{suffix}.qpy"
        for gate in "XZ" for suffix in ("", "dag")
    ]
    for name in names:
        with directory.joinpath(name).open("rb") as stream:
            assert len(qpy.load(stream)) == 1
        assert project_paths.quantum_circuits_path(name) == str(directory.joinpath(name))

    circuit, graph = create_ame_circuit(2, dimension)
    assert graph.vcount() == 2
    assert circuit.num_qubits == 4
    levels = np.arange(dimension)
    expected = np.zeros((4, 4), dtype=complex)
    expected[:dimension, :dimension] = (
        np.exp(2j * np.pi * np.outer(levels, levels) / dimension) / dimension
    )
    assert Statevector.from_instruction(circuit).equiv(expected.ravel())

    # This module eagerly loads all eight X/Z gates, including their adjoints.
    from qudits_on_qubits.core.generate_b_ame import generate_b_ame

    assert callable(generate_b_ame)


@pytest.mark.parametrize("strategy", ["append_w", "prepared_w_then_conjugated_entanglers"])
def test_encoding_change_uses_package_imports(strategy):
    encoding = np.eye(4, dtype=complex)[:, [0, 2, 3]] * [1, 1j, -1]
    circuit, _ = create_ame_circuit(2, 3, E_new=encoding, encoding_strategy=strategy)
    levels = np.arange(3)
    logical = np.exp(2j * np.pi * np.outer(levels, levels) / 3).ravel() / 3
    expected = np.kron(encoding, encoding) @ logical
    assert Statevector.from_instruction(circuit).equiv(expected)


@pytest.mark.parametrize("dimension", [3, 4])
@pytest.mark.parametrize("dagger", [False, True])
def test_bundled_stabilizer_gates_match_logical_paulis(dimension, dagger):
    from qudits_on_qubits.core.generate_b_ame import create_stab_from_str

    shift = np.roll(np.eye(dimension), 1, axis=0)
    phase = np.diag(np.exp(2j * np.pi * np.arange(dimension) / dimension))
    for word, logical in (("X", shift), ("Z", phase), ("XZ", phase @ shift)):
        expected = np.eye(4, dtype=complex)
        expected[:dimension, :dimension] = logical
        if dagger:
            expected = expected.conj().T
        actual = Operator(create_stab_from_str(word, dimension, dag=dagger)).data
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
