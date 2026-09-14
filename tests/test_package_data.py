"""Runtime assets must work independently of a source checkout."""

from importlib.resources import files

import numpy as np
import pytest
from qiskit import qpy
from qiskit.quantum_info import Statevector

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
    assert np.isclose(np.linalg.norm(Statevector.from_instruction(circuit).data), 1)

    # This module eagerly loads all eight X/Z gates, including their adjoints.
    from qudits_on_qubits.core.generate_b_ame import generate_b_ame

    assert callable(generate_b_ame)


@pytest.mark.parametrize("strategy", ["append_w", "prepared_w_then_conjugated_entanglers"])
def test_encoding_change_uses_package_imports(strategy):
    encoding = np.eye(4, 3, dtype=complex)
    circuit, _ = create_ame_circuit(2, 3, E_new=encoding, encoding_strategy=strategy)
    baseline, _ = create_ame_circuit(2, 3)
    assert Statevector.from_instruction(circuit).equiv(Statevector.from_instruction(baseline))
