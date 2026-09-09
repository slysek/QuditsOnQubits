from __future__ import annotations

import pytest


@pytest.fixture
def exact_cz3_synthesis(monkeypatch, tmp_path):
    """Test benchmark plumbing without launching a costly BQSKit search.

    This exact, generally expensive CZ decomposition is ONLY a test double.
    Production synthesis is separately exercised by the opt-in integration test.
    All real gate validation, F3 synthesis, caching and exports still run.
    """
    from qiskit import QuantumCircuit, transpile
    from qudits_on_qubits.benchmarks.direct_basis import optimized_gates as gates
    from qudits_on_qubits.benchmarks.direct_basis.math_utils import (
        physical_two_qutrit_gate_in_encoding, qutrit_cz,
    )

    calls = []

    def compile_exact(embedding):
        calls.append(embedding.copy())
        circuit = QuantumCircuit(4)
        circuit.unitary(physical_two_qutrit_gate_in_encoding(qutrit_cz(), embedding), range(4))
        return transpile(circuit, basis_gates=["u", "cz"], seed_transpiler=0, optimization_level=1)

    monkeypatch.setattr(gates, "_compile_cz3", compile_exact)
    monkeypatch.setattr(gates, "repo_path", lambda *args: str(tmp_path / "gate_cache"))
    return calls
