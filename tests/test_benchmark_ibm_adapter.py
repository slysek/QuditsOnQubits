import pytest
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import InstructionProperties

from qudits_on_qubits.benchmarks.targets import BackendTarget


@pytest.mark.parametrize("entangler", ["cx", "ecr"])
def test_ibm_uses_backendv2_native_isa_with_non_cz_entangler(entangler):
    backend = GenericBackendV2(
        num_qubits=3, basis_gates=["id", "rz", "sx", "x", entangler],
        coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1]], seed=3,
    )
    target = BackendTarget.ibm(backend.name, backend=backend,
                               initial_layout=[0, 1, 2], optimization_level=1)
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.rz(0.21, 2)
    compiled = target.compile(circuit, seed=5)
    target.validate_native(compiled)
    assert compiled.count_ops()[entangler] >= 1
    assert compiled.layout is not None
    assert "cz" not in compiled.count_ops()
    assert target.to_dict()["provider"] == "ibm"


def test_ibm_calibration_is_frozen_and_contributes_to_target_identity():
    backend = GenericBackendV2(2, basis_gates=["cx", "rz", "sx", "x"], seed=1)
    original = BackendTarget.ibm(backend.name, backend=backend)
    old_manifest = original.to_dict()
    backend.target.update_instruction_properties("cx", (0, 1), InstructionProperties(duration=1.0, error=0.25))
    updated = BackendTarget.ibm(backend.name, backend=backend)
    assert old_manifest == original.to_dict()
    assert old_manifest["target_hash"] != updated.to_dict()["target_hash"]
    original.validate_native(original.compile(QuantumCircuit(2), seed=1))


def test_ibm_rejects_mismatching_backend_identity():
    backend = GenericBackendV2(2)
    with pytest.raises(ValueError, match="identity|name"):
        BackendTarget.ibm("some_other_backend", backend=backend)


def test_native_fixed_parameters_are_validated_and_fingerprinted():
    from types import SimpleNamespace
    from qiskit.circuit.library import RZGate
    from qiskit.transpiler import Target
    def target_for(angle):
        target = Target(num_qubits=1)
        target.add_instruction(RZGate(angle), {(0,): None})
        return BackendTarget.ibm("fixed_parameter_fixture", backend=SimpleNamespace(name="fixed_parameter_fixture", target=target))
    first, second = target_for(0.5), target_for(0.7)
    assert first.to_dict()["target_hash"] != second.to_dict()["target_hash"]
    valid = QuantumCircuit(1)
    valid.rz(0.5, 0)
    first.validate_native(valid)
    invalid = QuantumCircuit(1)
    invalid.rz(0.7, 0)
    with pytest.raises(ValueError, match="Unsupported"):
        first.validate_native(invalid)


def test_ibm_read_only_loader_is_called_once(monkeypatch):
    from qudits_on_qubits.benchmarks.direct_basis import ibm_backend
    backend = GenericBackendV2(2)
    calls = []
    def load(name, **kwargs):
        calls.append(name)
        return backend
    monkeypatch.setattr(ibm_backend, "load_ibm_backend", load)
    target = BackendTarget.ibm(backend.name, optimization_level=0)
    for seed in (1, 2):
        target.compile(QuantumCircuit(1), seed=seed)
    assert calls == [backend.name]
