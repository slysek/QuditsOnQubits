import json

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qudits_on_qubits.benchmarks.targets import BackendTarget


def test_local_profile_is_explicit_and_snapshot_round_trips():
    target = BackendTarget.local()
    manifest = target.to_dict()
    assert manifest["provider"] == "local"
    assert manifest["synthetic"] is True
    assert manifest["topology"] == "all_to_all"
    assert target.num_qubits == 4
    assert manifest["compilation"]["routing_method"] == "sabre"
    assert manifest["versions"]["qiskit"]
    assert len(manifest["target_hash"]) == 64
    restored = BackendTarget.from_snapshot(json.loads(json.dumps(manifest)))
    assert restored.to_dict() == manifest
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    restored.validate_native(restored.compile(circuit, seed=3))


def test_routed_non_neighbor_entangler_preserves_logical_state_and_layout():
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.ry(0.47, 2)
    circuit.cx(0, 2)
    target = BackendTarget.local(
        num_qubits=3, coupling_map=[(0, 1), (1, 0), (1, 2), (2, 1)],
        initial_layout=[0, 1, 2], optimization_level=0,
    )
    compiled = target.compile(circuit, seed=8)
    assert compiled.layout is not None
    target.validate_native(compiled)
    assert compiled.count_ops()["cz"] > 1
    final = compiled.layout.final_index_layout()
    physical = Statevector.from_instruction(compiled).data
    logical = np.array([
        physical[sum(((index >> logical_bit) & 1) << physical_bit
                     for logical_bit, physical_bit in enumerate(final))]
        for index in range(8)
    ])
    assert abs(np.vdot(Statevector.from_instruction(circuit).data, logical)) ** 2 == pytest.approx(1.0)


def test_native_validation_rejects_bad_connectivity_and_non_native_gate():
    target = BackendTarget.local(num_qubits=3, coupling_map=[(0, 1), (1, 2)])
    disconnected = QuantumCircuit(3)
    disconnected.cz(0, 2)
    with pytest.raises(ValueError, match="unsupported|Unsupported"):
        target.validate_native(disconnected)
    non_native = QuantumCircuit(3)
    non_native.cx(0, 1)
    with pytest.raises(ValueError, match="unsupported|Unsupported"):
        target.validate_native(non_native)


def test_snapshot_fingerprint_and_configuration_are_immutable():
    edges = [(0, 1), (1, 0)]
    layout = [1, 0]
    target = BackendTarget.local(num_qubits=2, coupling_map=edges, initial_layout=layout)
    before = target.to_dict()
    edges.clear()
    layout.reverse()
    manifest = target.to_dict()
    manifest["compilation"]["initial_layout"].reverse()
    assert target.to_dict() == before
    assert target.to_dict()["target_hash"] != BackendTarget.local(num_qubits=2).to_dict()["target_hash"]
    tampered = target.to_dict()
    tampered["target"]["num_qubits"] = 8
    with pytest.raises(ValueError, match="hash|fingerprint"):
        BackendTarget.from_snapshot(tampered)


@pytest.mark.parametrize("options", [
    {"num_qubits": 0}, {"optimization_level": 4}, {"initial_layout": [0, 0]},
    {"coupling_map": [(0, 4)]}, {"basis_gates": ()},
    {"approximation_degree": 0.5},
])
def test_invalid_local_configuration_fails_early(options):
    with pytest.raises(ValueError):
        BackendTarget.local(**options)


def test_compile_rejects_invalid_seed_and_too_wide_circuit():
    target = BackendTarget.local(num_qubits=2)
    with pytest.raises(ValueError, match="seed"):
        target.compile(QuantumCircuit(1), seed=-1)
    with pytest.raises(ValueError, match="qubits|width"):
        target.compile(QuantumCircuit(3), seed=0)


def test_sabre_budget_is_captured_and_environment_drift_rejected(monkeypatch):
    from qiskit.transpiler.preset_passmanagers import builtin_plugins
    monkeypatch.setattr(builtin_plugins, "_get_trial_count", lambda default: default)
    target = BackendTarget.local(optimization_level=2)
    budget = target.to_dict()["compilation"]["layout_search_budget"]
    assert budget["layout_trials"] == 20
    assert budget["swap_trials"] == 20
    assert budget["max_iterations"] == 2
    monkeypatch.setattr(builtin_plugins, "_get_trial_count", lambda default: default + 1)
    with pytest.raises(ValueError, match="budget|environment"):
        target.compile(QuantumCircuit(2), seed=1)


def test_hardware_snapshot_never_claims_lossy_offline_replay():
    with pytest.raises(ValueError, match="Only synthetic local"):
        BackendTarget.from_snapshot({"provider": "iqm"})


def test_native_validation_checks_gate_class_not_only_its_name():
    from qiskit.circuit import Gate
    target = BackendTarget.local(num_qubits=2)
    spoofed = QuantumCircuit(2)
    spoofed.append(Gate("cz", 2, []), [0, 1])
    with pytest.raises(ValueError, match="Unsupported"):
        target.validate_native(spoofed)
