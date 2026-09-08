from __future__ import annotations

from importlib import import_module
from dataclasses import asdict
import json
import traceback
from types import SimpleNamespace

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Parameter
from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers.sampler_pub import SamplerPub

from qudits_on_qubits.experiments.backends.base import BackendAdapter, SubmittedJob
from qudits_on_qubits.experiments.backends.ibm import IBMAdapter
from qudits_on_qubits.experiments.errors import (
    BackendCompatibilityError,
    BackendUnavailableError,
    ExperimentValidationError,
    JobResultError,
    JobSubmissionError,
)
from qudits_on_qubits.experiments.execution import ExecutionMode
from qudits_on_qubits.experiments.ibm_spec import IBMHardware
from qudits_on_qubits.experiments.models import TranspilationConfig


class Backend:
    name = "ibm_test"
    backend_version = "1.2.3"
    num_qubits = 5
    max_circuits = 4

    def status(self):
        return SimpleNamespace(operational=True)

    def configuration(self):
        return SimpleNamespace(max_shots=1000, dynamic_reprate_enabled=True, rep_delay_range=[0.0, 0.01])


class Job:
    primitive_id = "sampler"

    def __init__(self, backend, result=None, inputs=None):
        self._backend = backend
        self._result = result
        self.inputs = inputs
        self.result_calls = []

    def job_id(self):
        return "ibm-job-123"

    def backend(self):
        return self._backend

    def result(self, **kwargs):
        self.result_calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def status(self):
        return "DONE"


class Service:
    def __init__(self, backend, job=None):
        self._backend = backend
        self._job = job or Job(backend)
        self.backend_calls = []
        self.job_calls = []

    def backend(self, name):
        self.backend_calls.append(name)
        return self._backend

    def job(self, job_id):
        self.job_calls.append(job_id)
        return self._job


class SamplerFactory:
    def __init__(self, job):
        self.job = job
        self.calls = []
        self.pubs = []

    def __call__(self, **options):
        self.calls.append(options)
        return self

    def run(self, pubs):
        self.pubs.append(pubs)
        return self.job


def circuit():
    result = QuantumCircuit(2, 2)
    result.h(0)
    result.cx(0, 1)
    result.measure([0, 1], [0, 1])
    return result


def fixture(result=None, inputs=None):
    backend = Backend()
    job = Job(backend, result, inputs)
    service = Service(backend, job)
    factory = SamplerFactory(job)
    adapter = IBMAdapter(IBMHardware("ibm_test"), service=service, sampler_factory=factory)
    return adapter, service, factory


def pub_result(**registers):
    return SamplerPubResult(DataBin(**{
        name: BitArray.from_samples(strings, num_bits=len(strings[0])) for name, strings in registers.items()
    }))


def test_spec_roundtrip_and_no_connection_on_construction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("account lookup on import or construction")

    monkeypatch.setattr(import_module("qudits_on_qubits._ibm_runtime"), "runtime_account_options", forbidden)
    spec = IBMHardware("ibm_test", account_name="research", instance="crn:v1:resource")
    assert IBMHardware.from_safe_dict(spec.to_safe_dict()) == spec
    assert spec.execution_mode is ExecutionMode.HARDWARE
    adapter = IBMAdapter(spec)
    assert isinstance(adapter, BackendAdapter)
    assert adapter._service is None


@pytest.mark.parametrize("field", ["device", "account_name", "instance"])
def test_spec_rejects_credential_material(field):
    with pytest.raises(ExperimentValidationError):
        IBMHardware(**{"device": "ibm_test", field: "token=private-secret"})


def test_spec_rejects_secret_fields_and_wrong_mode():
    with pytest.raises(ExperimentValidationError):
        IBMHardware.from_safe_dict({"device": "ibm_test", "token": "unlabelled-secret"})
    with pytest.raises(ExperimentValidationError):
        IBMHardware.from_safe_dict({"device": "ibm_test", "execution_mode": "ideal_simulator"})


def test_resolve_capabilities_and_metadata_are_safe():
    adapter, service, _ = fixture()
    assert service.backend_calls == []
    assert adapter.resolve().kind == "ibm"
    assert adapter.resolve().name == "ibm_test"
    assert service.backend_calls == ["ibm_test"]
    assert adapter.capabilities().max_circuits == 4
    assert adapter.capabilities().supports_resume
    assert not adapter.capabilities().local
    assert adapter.availability().available
    assert "service" not in repr(adapter.metadata())
    assert "account" not in repr(adapter.metadata())


@pytest.mark.parametrize("shots", [True, 0, -1, 1.5, 1001])
def test_preflight_rejects_invalid_or_unsupported_shots(shots):
    adapter, _, factory = fixture()
    with pytest.raises((BackendCompatibilityError, ExperimentValidationError)):
        adapter.preflight([circuit()], shots)
    assert factory.calls == []


def test_preflight_rejects_size_and_parameters():
    adapter, _, _ = fixture()
    with pytest.raises(BackendCompatibilityError, match="at most"):
        adapter.preflight([circuit()] * 5, 10)
    too_large = QuantumCircuit(6, 6)
    too_large.measure(range(6), range(6))
    with pytest.raises(BackendCompatibilityError, match="capacity"):
        adapter.preflight([too_large], 10)
    parameterized = circuit()
    parameterized.rz(Parameter("theta"), 0)
    with pytest.raises(BackendCompatibilityError, match="bound"):
        adapter.preflight([parameterized], 10)
    with pytest.raises(BackendCompatibilityError, match="measurements"):
        adapter.preflight([QuantumCircuit(2)], 10)


def test_compile_preserves_mapping_and_honors_transpilation_config():
    adapter, _, _ = fixture()
    seen = {}

    def fake_transpiler(circuits, *, backend, **options):
        seen.update(circuits=circuits, backend=backend, options=options)
        return [item.copy() for item in circuits]

    adapter._transpiler = fake_transpiler
    source = circuit()
    compiled = adapter.compile([source], TranspilationConfig(optimization_level=1, seed_transpiler=4, initial_layout=(3, 2)))
    assert seen["options"] == {"optimization_level": 1, "seed_transpiler": 4, "initial_layout": [3, 2]}
    assert seen["backend"] is adapter.backend
    assert compiled.circuits[0] == source
    assert compiled.metadata["register_layouts"][0][0]["clbits"] == (0, 1)
    assert len(compiled.metadata["circuit_digests"][0]) == 64
    adapter._transpiler = lambda *args, **kwargs: [QuantumCircuit(2)]
    with pytest.raises(BackendCompatibilityError):
        adapter.compile([source], TranspilationConfig())


def test_compile_real_qiskit_backend_locally():
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(3, seed=2)
    adapter = IBMAdapter(IBMHardware(backend.name), backend=backend)
    compiled = adapter.compile([circuit()], TranspilationConfig(seed_transpiler=12, initial_layout=(2, 1)))
    assert compiled.circuits[0].num_qubits == 3
    assert compiled.circuits[0].num_parameters == 0
    assert compiled.circuits[0].layout is not None
    assert compiled.metadata["register_layouts"][0][0]["clbits"] == (0, 1)


def test_submit_enforces_raw_options_and_exact_separate_pub_shots():
    adapter, _, factory = fixture()
    source = circuit()
    submitted = adapter.submit([source, source], 37, {"max_execution_time": 123, "execution": {"rep_delay": 0.002}})
    assert submitted.circuit_count == 2
    assert submitted.shots == 37
    assert factory.pubs == [[(source, None, 37), (source, None, 37)]]
    options = factory.calls[0]["options"]
    assert options["dynamical_decoupling"] == {"enable": False}
    assert options["twirling"] == {"enable_gates": False, "enable_measure": False}
    assert options["execution"] == {"meas_type": "classified", "init_qubits": True, "rep_delay": 0.002}
    assert "default_shots" not in options
    from qiskit_ibm_runtime.options import SamplerOptions

    sdk_options = SamplerOptions(**options)
    assert sdk_options.dynamical_decoupling.enable is False
    assert sdk_options.twirling.enable_measure is False


@pytest.mark.parametrize("options", [
    {"shots": 5}, {"default_shots": 5}, {"resilience_level": 1},
    {"experimental": {}}, {"dynamical_decoupling": {"enable": True}},
    {"twirling": {"enable_measure": True}}, {"twirling": {"num_randomizations": 2}},
    {"execution": {"meas_type": "kerneled"}}, {"execution": {"init_qubits": False}},
    {"execution": {"rep_delay": 0.1}}, {"execution": {"rep_delay": float("nan")}},
    {"max_execution_time": True}, {"token": "private"},
])
def test_unsafe_options_fail_before_sampler_submission(options):
    adapter, _, factory = fixture()
    with pytest.raises(BackendCompatibilityError):
        adapter.submit([circuit()], 4, options)
    assert factory.pubs == []
    assert factory.calls == []


@pytest.mark.parametrize("right,expected", [
    (["0", "0", "1", "1"], {"00": 2, "11": 2}),
    (["1", "1", "0", "0"], {"10": 2, "01": 2}),
])
def test_multiple_registers_preserve_shot_correlations(right, expected):
    left_register, right_register = ClassicalRegister(1, "alice"), ClassicalRegister(1, "bob")
    source = QuantumCircuit(QuantumRegister(2), left_register, right_register)
    source.measure([0, 1], [0, 1])
    # Reverse data-bin order to ensure circuit provenance controls decoding.
    raw = PrimitiveResult([pub_result(bob=right, alice=["0", "0", "1", "1"])])
    adapter, _, _ = fixture(raw)
    submitted = adapter.submit([source], 4)
    result = adapter.result(submitted)
    assert dict(result.counts[0]) == expected
    assert sum(result.counts[0].values()) == 4
    assert result.metadata["joint_register_counts"]


def test_register_order_maps_to_actual_global_classical_bit_indices():
    first, second, third = Clbit(), Clbit(), Clbit()
    source = QuantumCircuit(QuantumRegister(3))
    source.add_bits([first, second, third])
    source.add_register(ClassicalRegister(bits=[third, first], name="outer"))
    source.add_register(ClassicalRegister(bits=[second], name="middle"))
    source.measure(range(3), range(3))
    # outer[1]=c0=1, outer[0]=c2=0, middle[0]=c1=1 -> c2c1c0 = 011.
    adapter, _, _ = fixture(PrimitiveResult([pub_result(outer=["10", "10"], middle=["1", "1"])]))
    assert dict(adapter.result(adapter.submit([source], 2)).counts[0]) == {"011": 2}


def test_single_register_and_repeated_circuit_results_stay_separate():
    raw = PrimitiveResult([pub_result(c=["00", "11"]), pub_result(c=["01", "10"])])
    adapter, _, _ = fixture(raw)
    submitted = adapter.submit([circuit(), circuit()], 2)
    assert [dict(item) for item in adapter.result(submitted).counts] == [{"00": 1, "11": 1}, {"01": 1, "10": 1}]


def test_multiple_marginal_counts_are_rejected():
    source = QuantumCircuit(QuantumRegister(2), ClassicalRegister(1, "a"), ClassicalRegister(1, "b"))
    source.measure([0, 1], [0, 1])
    marginal = SimpleNamespace(num_bits=1, get_counts=lambda: {"0": 1, "1": 1})
    adapter, _, _ = fixture([SimpleNamespace(data=SimpleNamespace(a=marginal, b=marginal))])
    with pytest.raises(JobResultError, match="per-shot"):
        adapter.result(adapter.submit([source], 2))


@pytest.mark.parametrize("raw,match", [
    (PrimitiveResult([pub_result(c=["00"])]), "sum"),
    (PrimitiveResult([pub_result(c=["0", "1"])]), "width"),
    (PrimitiveResult([pub_result(other=["00", "11"])]), "registers"),
    (PrimitiveResult([pub_result(c=["00", "11"]), pub_result(c=["00", "11"])]), "PUB count"),
])
def test_result_rejects_malformed_counts(raw, match):
    adapter, _, _ = fixture(raw)
    with pytest.raises(JobResultError, match=match):
        adapter.result(adapter.submit([circuit()], 2))


def test_wrong_shot_count_error_retains_joint_counts_without_sdk_metadata():
    source = QuantumCircuit(QuantumRegister(2), ClassicalRegister(1, "a"), ClassicalRegister(1, "b"))
    source.measure([0, 1], [0, 1])
    raw = PrimitiveResult(
        [pub_result(a=["0", "1"], b=["1", "0"])],
        metadata={"token": "private-secret", "opaque": object()},
    )
    adapter, _, factory = fixture(raw)
    submitted = adapter.submit([source], 5)
    with pytest.raises(JobResultError, match="sum") as caught:
        adapter.result(submitted)
    evidence = caught.value.raw_evidence
    assert evidence["counts"] == [{"10": 1, "01": 1}]
    assert evidence["job_id"] == submitted.job_id
    assert evidence["target_identity"]["kind"] == "ibm"
    rendered = json.dumps(evidence, allow_nan=False)
    assert "private-secret" not in rendered
    assert "opaque" not in rendered
    assert len(factory.pubs) == 1


@pytest.mark.parametrize("returned_pubs", [0, 1, 3])
def test_wrong_pub_count_error_retains_every_available_count_set(returned_pubs):
    raw = PrimitiveResult([pub_result(c=["00", "11"])] * returned_pubs)
    adapter, _, _ = fixture(raw)
    with pytest.raises(JobResultError, match="PUB count") as caught:
        adapter.result(adapter.submit([circuit(), circuit()], 2))
    assert caught.value.raw_evidence["counts"] == [{"00": 1, "11": 1}] * returned_pubs
    json.dumps(caught.value.raw_evidence, allow_nan=False)


def test_unaligned_registers_preserve_separate_bitstrings_and_later_pub_positions():
    source = QuantumCircuit(QuantumRegister(2), ClassicalRegister(1, "a"), ClassicalRegister(1, "b"))
    source.measure([0, 1], [0, 1])
    good = pub_result(a=["0", "1"], b=["0", "1"])
    unaligned = pub_result(a=["0", "1"], b=["1"])
    adapter, _, _ = fixture(PrimitiveResult([good, unaligned, good]))
    with pytest.raises(JobResultError, match="inconsistent shot counts") as caught:
        adapter.result(adapter.submit([source, source, source], 2))
    evidence = caught.value.raw_evidence
    assert evidence["counts"] == [{"00": 1, "11": 1}, None, {"00": 1, "11": 1}]
    diagnostics = evidence["metadata"]["diagnostics"]
    record = diagnostics["pubs"][0]
    assert record["pub_index"] == 1
    registers = {item["name"]: item["bitstrings"] for item in record["registers"]}
    assert registers == {"a": ["0", "1"], "b": ["1"]}


def test_raw_register_diagnostics_are_bounded_and_omit_nonbinary_content():
    source = QuantumCircuit(QuantumRegister(2), ClassicalRegister(1, "a"), ClassicalRegister(1, "b"))
    source.measure([0, 1], [0, 1])
    register_a = SimpleNamespace(num_bits=1, num_shots=2001, get_bitstrings=lambda: ["token=private-secret", *(["0"] * 2000)])
    register_b = SimpleNamespace(num_bits=1, num_shots=2000, get_bitstrings=lambda: ["1"] * 2000)
    adapter, _, _ = fixture([SimpleNamespace(data=SimpleNamespace(a=register_a, b=register_b))])
    with pytest.raises(JobResultError, match="malformed") as caught:
        adapter.result(adapter.submit([source], 2))
    evidence = caught.value.raw_evidence
    assert evidence["counts"] == [None]
    rendered = json.dumps(evidence, allow_nan=False)
    assert "private-secret" not in rendered
    assert len(rendered) < 10000
    records = evidence["metadata"]["diagnostics"]["pubs"][0]["registers"]
    assert all(len(record["bitstrings"]) <= 256 for record in records)
    assert all(record["truncated"] for record in records)


def test_restore_validates_backend_shots_count_and_circuit_provenance():
    source = circuit()
    adapter, service, factory = fixture(
        PrimitiveResult([pub_result(c=["00", "11"])]),
        {"pubs": [SamplerPub(source, shots=2)]},
    )
    restored = adapter.restore_job("ibm-job-123", circuit_count=1, shots=2, circuits=[source])
    assert service.job_calls == ["ibm-job-123"]
    assert dict(adapter.result(restored).counts[0]) == {"00": 1, "11": 1}
    assert factory.calls == []
    with pytest.raises(JobResultError, match="shots"):
        adapter.restore_job("ibm-job-123", shots=3)
    with pytest.raises(JobResultError, match="PUB count"):
        adapter.restore_job("ibm-job-123", circuit_count=2)
    changed = source.copy()
    changed.x(0)
    with pytest.raises(JobResultError, match="circuit provenance"):
        adapter.restore_job("ibm-job-123", circuits=[changed])
    service._job._backend = SimpleNamespace(name="ibm_other")
    with pytest.raises(JobResultError, match="backend"):
        adapter.restore_job("ibm-job-123")


def test_restore_accepts_pub_tuples_and_reports_unavailable_provenance():
    adapter, service, _ = fixture(inputs={"pubs": [(circuit(), None, 5)]})
    assert adapter.restore_job("ibm-job-123").shots == 5
    service._job.inputs = None
    restored = adapter.restore_job("ibm-job-123", circuit_count=1, shots=5, circuits=[circuit()])
    assert restored.metadata["input_provenance_available"] is False
    assert restored.metadata["register_layouts"]
    service._job.inputs = {"pubs": [(circuit(), None, None)]}
    with pytest.raises(JobResultError, match="explicit shot"):
        adapter.restore_job("ibm-job-123", shots=5)


def test_unknown_job_recovery_requires_remote_raw_options_and_circuit_provenance():
    source = circuit()
    adapter, service, factory = fixture(inputs={"pubs": [(source, None, 5)]})
    submitted = adapter.restore_job("ibm-job-123", circuits=[source], shots=5)
    with pytest.raises(BackendCompatibilityError, match="provider evidence"):
        adapter.verify_restored_job(submitted, circuits=[source], shots=5)
    service._job.inputs["options"] = {
        "dynamical_decoupling": {"enable": False},
        "twirling": {"enable_gates": False, "enable_measure": False},
        "execution": {"meas_type": "classified", "init_qubits": True},
    }
    adapter.verify_restored_job(submitted, circuits=[source], shots=5)
    assert factory.calls == []
    service._job.inputs["options"]["twirling"]["enable_measure"] = True
    with pytest.raises(JobResultError, match="incompatible"):
        adapter.restore_job("ibm-job-123", circuits=[source], shots=5)


def test_restore_accepts_actual_runtime_serialization_roundtrip():
    from qiskit_ibm_runtime.options import SamplerOptions
    from qiskit_ibm_runtime.utils.json import RuntimeDecoder, RuntimeEncoder

    source = circuit()
    options = SamplerOptions(
        dynamical_decoupling={"enable": False},
        twirling={"enable_gates": False, "enable_measure": False},
        execution={"meas_type": "classified", "init_qubits": True},
    )
    inputs = {"pubs": [SamplerPub(source, shots=5)], **SamplerOptions._get_program_inputs(asdict(options))}
    decoded = json.loads(json.dumps(inputs, cls=RuntimeEncoder), cls=RuntimeDecoder)
    adapter, _, factory = fixture(inputs=decoded)
    restored = adapter.restore_job("ibm-job-123", circuits=[source], shots=5)
    adapter.verify_restored_job(restored, circuits=[source], shots=5)
    assert restored.metadata["circuit_provenance_verified"]
    assert restored.metadata["raw_options_verified"]
    assert factory.calls == []


def test_restore_rejects_populated_parameter_values_for_bound_circuit():
    adapter, _, _ = fixture(inputs={"pubs": [(circuit(), [1.0], 5)]})
    with pytest.raises(JobResultError, match="invalid"):
        adapter.restore_job("ibm-job-123", circuits=[circuit()], shots=5)


def test_provider_timeout_is_sanitized_and_never_resubmits():
    adapter, _, factory = fixture(TimeoutError("token=private-secret"))
    submitted = adapter.submit([circuit()], 2)
    with pytest.raises(JobResultError) as caught:
        adapter.result(submitted, timeout=0.2)
    assert "private-secret" not in "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert submitted.handle.result_calls == [{"timeout": 0.2}]
    assert len(factory.pubs) == 1


def test_submission_exception_is_sanitized():
    adapter, _, factory = fixture()

    def failure(*args, **kwargs):
        raise RuntimeError("token=private-secret")

    factory.run = failure
    with pytest.raises(JobSubmissionError) as caught:
        adapter.submit([circuit()], 2)
    assert "private-secret" not in "".join(traceback.format_exception(caught.type, caught.value, caught.tb))


def test_service_credentials_are_used_only_for_connection(monkeypatch):
    import qiskit_ibm_runtime

    adapter_module = import_module("qudits_on_qubits.experiments.backends.ibm")
    seen = []
    service = Service(Backend())
    monkeypatch.setattr(adapter_module, "runtime_account_options", lambda **kwargs: {"token": "opaque-key", "name": kwargs["account_name"]})
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", lambda **kwargs: (seen.append(kwargs), service)[1])
    adapter = IBMAdapter(IBMHardware("ibm_test", account_name="research"))
    assert adapter.resolve().name == "ibm_test"
    assert seen == [{"token": "opaque-key", "name": "research"}]
    assert "opaque-key" not in repr(adapter.metadata())
    assert "token" not in repr(adapter.metadata())
