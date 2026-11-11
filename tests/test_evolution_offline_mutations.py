"""Check state mutations with an independent statevector reference."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest
from qiskit import qpy
from qiskit.quantum_info import Statevector

from qudits_on_qubits.evolution.oe_candidates import candidate_from_circuit
from qudits_on_qubits.evolution.oe_evaluator import _limits, assess_candidate

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/evolution/openevolve"
STATES = {
    "ghz3": (9, ((0, 1, 1), (0, 2, 1))),
    "ame43": (19, ((0, 1, 1), (0, 3, 1), (1, 2, 1), (2, 3, 2))),
}


def reference_state(encoding, edges, n):
    state = np.zeros(4**n, dtype=complex)
    for digits in product(range(3), repeat=n):
        vector = np.array([1.0], dtype=complex)
        for digit in reversed(digits):
            vector = np.kron(vector, encoding[:, digit])
        phase = sum(weight * digits[a] * digits[b] for a, b, weight in edges)
        state += np.exp(2j * np.pi * phase / 3) * vector / np.sqrt(3**n)
    return state


@pytest.fixture(params=["ghz3", "ame43"])
def baseline(request):
    state = request.param
    folder = RUNS / f"{state}_mock_20261102"
    if not (folder / "best.qpy").exists() or not (folder / "encoding.npy").exists():
        pytest.skip("The saved offline baseline is not available")
    with (folder / "best.qpy").open("rb") as stream:
        circuit, = qpy.load(stream)
    encoding = np.load(folder / "encoding.npy", allow_pickle=False)
    count, edges = STATES[state]
    assert circuit.count_ops().get("cz", 0) == count
    expected = reference_state(encoding, edges, circuit.num_qubits // 2)
    assert abs(np.vdot(expected, Statevector.from_instruction(circuit).data)) == pytest.approx(1, abs=1e-12)
    return state, circuit, encoding, expected


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("mutation", ["remove_cz", "change_angle", "add_cz_pair", "periodic_angle"])
def test_mutations_match_independent_statevector(tmp_path, baseline, seed, mutation):
    state, original, encoding, reference = baseline
    candidate = candidate_from_circuit(original)
    operations = candidate["operations"]
    rng = np.random.default_rng(seed)
    if mutation in ("remove_cz", "add_cz_pair"):
        indices = [i for i, op in enumerate(operations) if op[0] == "cz"]
        index = int(rng.choice(indices))
        if mutation == "remove_cz":
            del operations[index]
        else:
            operations[index:index] = [operations[index].copy(), operations[index].copy()]
    else:
        indices = [i for i, op in enumerate(operations) if op[0] in ("u", "u3")]
        index = int(rng.choice(indices))
        operations[index][3] += 0.2 if mutation == "change_angle" else 2 * np.pi
    path = tmp_path / "candidate.py"
    path.write_text("CANDIDATE = " + repr(candidate), encoding="utf-8")
    circuit, metrics = assess_candidate(path, state=state, encoding=encoding)
    actual = Statevector.from_instruction(circuit).data
    overlap = np.vdot(reference, actual)
    fidelity = abs(overlap)**2
    error = np.linalg.norm(actual - np.exp(1j * np.angle(overlap)) * reference)
    projector = encoding @ encoding.conj().T
    for _ in range(circuit.num_qubits // 2 - 1):
        projector = np.kron(projector, encoding @ encoding.conj().T)
    leakage = np.linalg.norm(actual - projector @ actual)
    assert metrics["F"] == pytest.approx(fidelity, abs=1e-12)
    assert metrics["error_norm"] == pytest.approx(error, abs=1e-12)
    assert metrics["leakage_norm"] == pytest.approx(leakage, abs=1e-12)
    assert metrics["is_valid"] == (error <= 1e-10 and leakage <= 1e-10)
    assert (metrics["combined_score"] >= 1) == metrics["is_valid"]
    if mutation in ("add_cz_pair", "periodic_angle"):
        assert metrics["is_valid"]
    else:
        assert not metrics["is_valid"]
    assert metrics["cz_count"] == STATES[state][0] + {"remove_cz": -1, "add_cz_pair": 2}.get(mutation, 0)


@pytest.mark.parametrize("limits", [(0.001, 256, 1), (600, 4096, 600)])
def test_resource_limit_boundaries_accept(limits):
    _limits(*limits)


@pytest.mark.parametrize("limits", [
    (0, 2048, 30), (600.1, 2048, 30), (float("nan"), 2048, 30),
    (30, 255, 30), (30, 4097, 30), (30, True, 30),
    (30, 2048, 0), (30, 2048, 601), (30, 2048, 1.5),
])
def test_resource_limit_boundaries_reject(limits):
    with pytest.raises(ValueError):
        _limits(*limits)
