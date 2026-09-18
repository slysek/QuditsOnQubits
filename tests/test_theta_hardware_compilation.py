"""The hardware layer preserves supplied programs and complete Bell histograms."""
from importlib import import_module, util
import json

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qudits_on_qubits.benchmarks.theta_continuation.encoding import theta_embedding


MODULE = "qudits_on_qubits.benchmarks.theta_continuation.hardware_compilation"


def test_hardware_compiler_api_exists():
    assert util.find_spec(MODULE) is not None, "Add isolated hardware compilation layer"


@pytest.fixture
def compiler():
    return import_module(MODULE)


def _line(n):
    return [(q, q + 1) for q in range(n - 1)]


def _source(n):
    circuit = QuantumCircuit(n)
    for q in range(n):
        circuit.ry(0.13 * (q + 1), q)
    for q in range(0, n - 2, 2):
        circuit.cx(q, q + 2)
    for q in range(0, n, 2):
        circuit.cz(q, q + 1)
        circuit.rz(0.071, q + 1)
    return circuit


def test_pair_repair_retains_nonidentity_adjacent_pairs(compiler):
    permutation = (2, 3, 0, 1)
    final, swaps = compiler.shortest_party_repair(permutation, _line(4))
    assert tuple(final) == permutation
    assert swaps == []


def test_pair_repair_finds_shortest_path_without_restoring_identity(compiler):
    permutation = (0, 2, 3, 1)
    final, swaps = compiler.shortest_party_repair(permutation, _line(4))
    assert len(swaps) == 1
    assert tuple(final) != tuple(range(4))
    for a, b in ((0, 1), (2, 3)):
        assert abs(final[a] - final[b]) == 1
    replayed = list(permutation)
    for a, b in swaps:
        replayed = [b if q == a else a if q == b else q for q in replayed]
    assert replayed == list(final)


def test_gate_compilation_certifies_full_operator_and_counts(compiler):
    source = _source(4)
    source.cz(0, 3)
    result = compiler.compile_gate(source, _line(4), seeds=(2, 3, 4))
    meta = result["metadata"]
    assert meta["full_operator_error"] <= 1e-10
    assert meta["routing"]["seed"] in (2, 3, 4)
    for name in ("native", "routed"):
        circuit = result["circuits"][name + ".qpy"]
        assert set(circuit.count_ops()) <= {"r", "cz"}
        assert meta[name]["size"] == circuit.size()
        assert meta[name]["n_cz"] == circuit.count_ops().get("cz", 0)
    routed = result["circuits"]["routed.qpy"]
    assert all(abs(routed.find_bit(i.qubits[0]).index - routed.find_bit(i.qubits[1]).index) == 1
               for i in routed.data if i.operation.name == "cz")
    json.dumps(meta, allow_nan=False)


def test_gate_validation_does_not_only_check_zero_state(compiler, monkeypatch):
    circuit = QuantumCircuit(2)
    circuit.cz(0, 1)
    monkeypatch.setattr(compiler, "transpile", lambda source, **kwargs: QuantumCircuit(2))
    with pytest.raises(ValueError, match="exact|preserv|candidate"):
        compiler.compile_gate(circuit, [(0, 1)])


def test_small_rotations_are_not_silently_dropped(compiler):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.rz(3e-8, 0)
    circuit.cx(0, 1)
    result = compiler.compile_gate(circuit, [(0, 1)])
    assert result["metadata"]["full_operator_error"] <= 1e-10


@pytest.mark.parametrize("name,n,count", [("two_qutrit", 4, 9), ("ghz3", 6, 12), ("ame43", 8, 13)])
def test_complete_bell_histograms_weights_and_locality(compiler, name, n, count):
    source = _source(n)
    before = Operator(source).data.copy()
    result = compiler.compile_case(name, source, theta_embedding(0.23), _line(n), seeds=(1,))
    meta, circuits, arrays = result["metadata"], result["circuits"], result["arrays"]
    assert len(meta["settings"]) == count
    assert meta["max_probability_error"] <= 1e-10
    assert meta["shared_preparation_verified"]
    assert meta["independent_local_settings_verified"]
    np.testing.assert_allclose(Operator(source).data, before, atol=0, rtol=0)
    weights = arrays["weights.npy"]
    assert weights.shape == (count, 2**n)
    leakage = [k for k in range(2**n) if any((k >> q) & 3 == 3 for q in range(0, n, 2))]
    assert not np.any(weights[:, leakage])
    assert meta["ideal_bell"] == pytest.approx(float(np.sum(weights * arrays["probabilities_native.npy"])))
    for kind in ("native", "routed"):
        np.testing.assert_allclose(arrays[f"probabilities_{kind}.npy"], arrays["probabilities_source.npy"], atol=1e-10, rtol=0)
        prefix = circuits[f"bell_prefix_{kind}.qpy"]
        permutation = list(range(n)) if kind == "native" else meta["routing"]["bell"]["final_permutation"]
        physical_party = {permutation[q]: q // 2 for q in range(n)}
        for index, row in enumerate(meta["per_setting"]):
            full = circuits[f"bell_{kind}_{index:03d}.qpy"]
            assert full.count_ops()["measure"] == n
            assert row[kind]["size"] == full.size()
            assert row[kind]["depth"] == full.depth()
            assert len(full.data) >= len(prefix.data)
            assert full.data[:len(prefix.data)] == prefix.data
            for instruction in full.data[len(prefix.data):]:
                assert len({physical_party[full.find_bit(q).index] for q in instruction.qubits}) <= 1
    if name == "ame43":
        assert any(None in row for row in meta["settings"])
    json.dumps(meta, allow_nan=False)


@pytest.mark.parametrize("edges", [[], [(0, 0)], [(0, 2)], [(0, 1.0)], [(False, 1)]])
def test_bad_edges_rejected(compiler, edges):
    with pytest.raises((TypeError, ValueError)):
        compiler.compile_gate(QuantumCircuit(2), edges)


@pytest.mark.parametrize("seeds", [(), (True,), (-1,), (1.5,), (np.nan,), "012"])
def test_bad_seeds_rejected(compiler, seeds):
    with pytest.raises((TypeError, ValueError)):
        compiler.compile_gate(QuantumCircuit(2), [(0, 1)], seeds=seeds)


@pytest.mark.parametrize("encoding", [np.eye(3), np.ones((4, 3)), np.full((4, 3), np.nan)])
def test_bad_encoding_rejected(compiler, encoding):
    with pytest.raises(ValueError, match="encoding|isometry"):
        compiler.compile_case("two_qutrit", QuantumCircuit(4), encoding, _line(4))


def test_disconnected_graph_and_nonunitary_source_rejected(compiler):
    with pytest.raises(ValueError, match="connect"):
        compiler.compile_gate(QuantumCircuit(4), [(0, 1), (2, 3)])
    measured = QuantumCircuit(2, 2)
    measured.measure([0, 1], [0, 1])
    with pytest.raises(ValueError, match="unitary|bound"):
        compiler.compile_gate(measured, [(0, 1)])


def test_measurement_synthesis_preserves_every_row_projector(compiler):
    rng = np.random.default_rng(76)
    unitary, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    shortened, evidence = compiler._synthesize_measurement(unitary)
    actual = Operator(shortened).data
    assert shortened.count_ops().get("cz", 0) <= 2
    assert evidence["row_projector_error"] <= 1e-10
    for wanted, got in zip(unitary, actual):
        np.testing.assert_allclose(np.outer(wanted.conj(), wanted), np.outer(got.conj(), got), atol=1e-10, rtol=0)


@pytest.mark.parametrize("cheaper_level", [3, 1])
def test_routing_compares_both_valid_optimization_levels(compiler, monkeypatch, cheaper_level):
    source = QuantumCircuit(2)
    source.r(0.23, 0.4, 0)
    source.cz(0, 1)
    real_transpile = compiler.transpile
    trials = []

    def controlled_transpile(circuit, **kwargs):
        if kwargs.get("coupling_map") is None:
            return real_transpile(circuit, **kwargs)
        seed, level = kwargs["seed_transpiler"], kwargs["optimization_level"]
        trials.append((seed, level))
        assert kwargs["initial_layout"] == [0, 1]
        candidate = circuit.copy()
        if level != cheaper_level:
            candidate.cz(0, 1)
            candidate.cz(0, 1)
        return candidate

    monkeypatch.setattr(compiler, "transpile", controlled_transpile)
    result, metadata = compiler._route(source, ((0, 1),), (8, 9), full_operator=True)
    assert trials == [(8, 3), (8, 1), (9, 3), (9, 1)]
    assert metadata["optimization_level"] == cheaper_level
    assert result.count_ops()["cz"] == 1
    assert len(metadata["candidates"]) == 4
    assert all(record["accepted"] for record in metadata["candidates"])


def test_routing_falls_back_to_level_zero_after_both_drop_tiny_rotation(compiler, monkeypatch):
    source = QuantumCircuit(2)
    source.r(3e-8, 0.2, 0)
    real_transpile = compiler.transpile
    trials = []

    def controlled_transpile(circuit, **kwargs):
        if kwargs.get("coupling_map") is None:
            return real_transpile(circuit, **kwargs)
        level = kwargs["optimization_level"]
        trials.append(level)
        return circuit.copy() if level == 0 else QuantumCircuit(2)

    monkeypatch.setattr(compiler, "transpile", controlled_transpile)
    result, metadata = compiler._route(source, ((0, 1),), (5,), full_operator=True)
    assert trials == [3, 1, 0]
    assert metadata["optimization_level"] == 0
    assert [trial["accepted"] for trial in metadata["candidates"]] == [False, False, True]
    np.testing.assert_allclose(Operator(result).data, Operator(source).data, atol=1e-10, rtol=0)
