"""Check the classical depth polish: exact CZ swaps keep the circuit and lower the two-qubit depth."""
from time import perf_counter

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from qudits_on_qubits.evolution import polish
from qudits_on_qubits.evolution.polish import exact_swaps, shallowest, two_qubit_depth, variants
from qudits_on_qubits.evolution.skeleton_fit import _circuits

WIDTH = 4
CHAIN = [(0, 1), (1, 2), (2, 3)]


def _operator(pairs, x):
    return Operator(_circuits(WIDTH, pairs, (), x, 0.0)[0])


def _angles(theta_after_first_cz_on_qubit_1):
    x = np.random.default_rng(7).uniform(0, 2 * np.pi, 3 * WIDTH + 6 * len(CHAIN))
    x[3 * WIDTH + 3] = theta_after_first_cz_on_qubit_1  # U on qubit 1 after CZ (0, 1)
    return x


def test_two_qubit_depth_counts_parallel_layers():
    assert two_qubit_depth([]) == 0
    assert two_qubit_depth([(0, 1), (2, 3)]) == 1
    assert two_qubit_depth(CHAIN) == 3


@pytest.mark.parametrize("theta", [0.0, np.pi], ids=["diagonal", "anti-diagonal"])
def test_exact_swaps_keep_the_circuit(theta):
    x = _angles(theta)
    swaps = exact_swaps(CHAIN, x, WIDTH)
    assert swaps
    for pairs, y in swaps:
        assert sorted(pairs) == sorted(CHAIN) and pairs != CHAIN
        assert _operator(pairs, y).equiv(_operator(CHAIN, x))


def test_general_u_gates_give_no_swap():
    assert exact_swaps(CHAIN, _angles(1.0), WIDTH) == []


def test_shallowest_lowers_the_chain_depth():
    x = _angles(0.0)
    depth, pairs, y = shallowest(CHAIN, x, WIDTH, deadline=perf_counter() + 10)
    assert depth == two_qubit_depth(pairs) == 2
    assert _operator(pairs, y).equiv(_operator(CHAIN, x))


def test_routed_polish_takes_the_variant_with_the_fewest_routed_gates(monkeypatch):
    x = _angles(0.0)
    found = variants(CHAIN, x, WIDTH, deadline=perf_counter() + 10)
    assert found[0][1] == CHAIN and 1 < len(found) <= polish.ROUTED_VARIANTS + 1
    target = found[-1][1]

    def costs(circuit, _name, **_):
        pairs = [tuple(circuit.find_bit(q).index for q in item.qubits)
                 for item in circuit.data if item.operation.name == "cz"]
        count = 3 if pairs == list(target) else 9
        return {"routed_two_qubit_count": count, "routed_two_qubit_depth": count, "routed_depth": 10,
                "routed_success_estimate": 0.5}

    monkeypatch.setattr(polish, "routed_costs", costs)
    base = {"is_valid": True, "cz_count": 3, "two_qubit_depth": 3, "depth": 5, "one_qubit_gate_count": 9,
            "decoder_cz_count": 0, "F": 1.0}
    _, pairs, y = polish._best_routed(found, base, WIDTH, "iqm_garnet", "preparation", perf_counter() + 10)
    assert pairs == target and _operator(pairs, y).equiv(_operator(CHAIN, x))
