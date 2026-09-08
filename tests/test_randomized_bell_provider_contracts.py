"""Randomized runner contracts exercised through the actual hardware adapters."""

from __future__ import annotations

from copy import deepcopy
import itertools
import json
import math
import socket
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPubResult

from qudits_on_qubits.experiments import (
    ExperimentSpec, IBMHardware, IQMHardware, PathBasis, PiastQHardware, RandomizedBlocks,
)
from qudits_on_qubits.experiments.backends import IBMAdapter, IQMAdapter, PiastQAdapter
from qudits_on_qubits.experiments.block_runner import run_randomized_experiment, resume_randomized_experiment
from qudits_on_qubits.experiments.errors import BackendCompatibilityError


ALPHABETS = {"two_qutrit": (3, 3), "ghz3": (3, 3, 2), "ame43": (3, 3, 2, 2)}


@pytest.fixture(autouse=True)
def no_provider_network(monkeypatch):
    def reject(*_args, **_kwargs):
        raise AssertionError("provider contract tests must not use network connections")

    monkeypatch.setattr(socket, "getaddrinfo", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def draws(state):
    values = itertools.cycle(itertools.chain.from_iterable(
        itertools.product(*(range(size) for size in ALPHABETS[state]))
    ))
    return lambda size: next(values)


def make_spec(tmp_path, provider, state="two_qutrit", *, shots=7, limit=5):
    directory = tmp_path / "basis"
    directory.mkdir()
    with (directory / "graph_state_direct_basis.qpy").open("wb") as stream:
        qpy.dump(QuantumCircuit(2 * len(ALPHABETS[state])), stream)
    np.save(directory / "E.npy", np.eye(4, 3), allow_pickle=False)
    backend = {
        "ibm": IBMHardware("ibm_contract"),
        "iqm": IQMHardware("garnet"),
        "piastq": PiastQHardware("managed", "test-team"),
    }[provider]
    count = math.prod(ALPHABETS[state]) + 2
    return ExperimentSpec(
        state=state, basis=PathBasis(directory), backend=backend,
        measurement=RandomizedBlocks(count, shots, count, limit), output_root=tmp_path / "runs",
    )


def expected_counts(circuit, shots):
    block_id = int(circuit.name.removeprefix("block_"))
    width = circuit.num_clbits
    ones = block_id % shots
    values = {"0" * width: shots - ones}
    if ones:
        values["0" * (width - 1) + "1"] = ones
    return values


class ProviderJob:
    primitive_id = "sampler"

    def __init__(self, probe, index, circuits, shots, options):
        self.probe = probe
        self.index = index
        self.circuits = tuple(circuits)
        self.shots = shots
        self.inputs = {"pubs": [(c, None, shots) for c in circuits], "options": deepcopy(options)}
        self.histograms = tuple(expected_counts(c, shots) for c in circuits)
        self.result_calls = []

    def job_id(self):
        return f"{self.probe.kind}-contract-{self.index}"

    def backend(self):
        return self.probe.backend

    def status(self):
        return "DONE"

    def get_counts(self, index=None):
        if index is None:
            return self.histograms[0] if len(self.histograms) == 1 else self.histograms
        return self.histograms[index]

    def counts(self):
        return self.histograms

    def result(self, **kwargs):
        self.result_calls.append(kwargs)
        if self.index == self.probe.fail_result_job:
            raise TimeoutError("simulated provider result timeout")
        if self.probe.kind != "ibm":
            return self
        pubs = []
        for circuit, histogram in zip(self.circuits, self.histograms):
            joint_shots = [bits for bits, count in histogram.items() for _ in range(count)]
            registers = {}
            for register in circuit.cregs:
                indices = [circuit.find_bit(bit).index for bit in register]
                strings = ["".join(bits[-1 - index] for index in reversed(indices)) for bits in joint_shots]
                registers[register.name] = BitArray.from_samples(strings, num_bits=len(register))
            pubs.append(SamplerPubResult(DataBin(**registers)))
        return PrimitiveResult(pubs)


class ProviderProbe:
    def __init__(self, kind):
        self.kind = kind
        self.jobs = {}
        self.submissions = []
        self.restores = []
        self.options = []
        self.compiled = []
        self.fail_result_job = None
        self.backend = SimpleNamespace(
            name={"ibm": "ibm_contract", "iqm": "garnet", "piastq": "piastq-main"}[kind],
            num_qubits=64, max_circuits=4, backend_version="contract-1",
            calibration_set_id="contract-calibration",
            status=lambda: SimpleNamespace(operational=True),
            configuration=lambda: SimpleNamespace(max_shots=1000, dynamic_reprate_enabled=True, rep_delay_range=[0.0, 0.01]),
            run=self.iqm_run, retrieve_job=self.restore,
        )

    def submit(self, circuits, shots, options):
        job = ProviderJob(self, len(self.submissions), circuits, shots, options)
        self.submissions.append((tuple(circuits), shots))
        self.options.append(deepcopy(options))
        self.jobs[job.job_id()] = job
        return job

    def iqm_run(self, circuits, *, shots, **options):
        return self.submit(circuits, shots, options)

    def restore(self, job_id):
        self.restores.append(job_id)
        return self.jobs[job_id]

    def adapter(self, spec):
        probe = self
        if self.kind == "ibm":
            class Sampler:
                def __init__(self, *, mode, options):
                    assert mode is probe.backend
                    self.options = options

                def run(self, pubs):
                    assert all(parameters is None for _, parameters, _ in pubs)
                    assert len({shots for _, _, shots in pubs}) == 1
                    return probe.submit([c for c, _, _ in pubs], pubs[0][2], self.options)

            def transpiler(circuits, *, backend, **options):
                assert backend is probe.backend
                probe.compiled.extend(circuits)
                return [circuit.copy() for circuit in circuits]

            service = SimpleNamespace(backend=lambda name: self.backend, job=self.restore)
            return IBMAdapter(spec.backend, service=service, sampler_factory=Sampler, transpiler=transpiler)
        if self.kind == "iqm":
            def transpiler(circuit, backend, **options):
                assert backend is probe.backend
                probe.compiled.append(circuit)
                return circuit.copy()

            return IQMAdapter(spec.backend, backend=self.backend, transpiler=transpiler)

        class Client:
            def __init__(self, **kwargs):
                self.backend = probe.backend

            def retrieve_job(self, job_id):
                return probe.restore(job_id)

        class Sampler:
            def __init__(self, backend, *, options):
                assert backend is probe.backend
                self.options = options

            def run(self, circuits, *, shots):
                return probe.submit(circuits, shots, self.options)

        return PiastQAdapter(spec.backend, client_type=Client, sampler_type=Sampler, env_loader=lambda _: {}, poll_interval=0.01)


@pytest.mark.parametrize("provider", ["ibm", "iqm", "piastq"])
@pytest.mark.parametrize("state", ["two_qutrit", "ghz3", "ame43"])
def test_hardware_adapters_keep_each_randomized_block_and_zero_contribution_context(tmp_path, provider, state):
    spec = make_spec(tmp_path, provider, state)
    probe = ProviderProbe(provider)
    result = run_randomized_experiment(spec, adapter=probe.adapter(spec), _randbelow=draws(state), _source="test_sequence")
    assert result.status.value == "completed", result.values
    count = spec.measurement.setting_draws
    assert sum(len(circuits) for circuits, _ in probe.submissions) == count
    assert [len(circuits) for circuits, _ in probe.submissions] == [
        min(4, count - start) for start in range(0, count, 4)
    ]
    assert all(len(circuits) <= 4 and shots == 7 for circuits, shots in probe.submissions)
    sent = [circuit for circuits, _ in probe.submissions for circuit in circuits]
    assert [circuit.name for circuit in sent] == [f"block_{index:08d}" for index in range(count)]
    assert len({id(circuit) for circuit in sent}) == count
    assert all(circuit.num_clbits == 2 * len(ALPHABETS[state]) for circuit in sent)
    assert all(circuit.num_parameters == 0 for circuit in sent)
    schedule = read_json(result.artifact_dir / "schedule.json")
    assert schedule["blocks"][0]["settings"] == schedule["blocks"][-2]["settings"]
    assert schedule["blocks"][1]["settings"] == schedule["blocks"][-1]["settings"]
    if provider != "piastq":
        assert len(probe.compiled) == math.prod(ALPHABETS[state])
        assert all(not circuit.name.startswith("block_") for circuit in probe.compiled)
    evidence = [
        histogram
        for path in sorted((result.artifact_dir / "batches").glob("*/counts.json"))
        for histogram in read_json(path)["counts"]
    ]
    assert evidence == [expected_counts(circuit, 7) for circuit in sent]
    assert evidence[0] != evidence[-2]
    assert result.values["budget"]["completed_shots"] == count * 7
    zero_blocks = [block["block_id"] for block in schedule["blocks"] if not block["matched_pattern_indices"]]
    assert len(zero_blocks) == {"two_qutrit": 0, "ghz3": 6, "ame43": 21}[state]
    assert all(sum(evidence[index].values()) == 7 for index in zero_blocks)
    leakage_blocks = {record["block_id"]: record for record in result.values["leakage"]["blocks"]}
    assert all(
        leakage_blocks[index]["status"] == "completed" and leakage_blocks[index]["total_shots"] == 7
        for index in zero_blocks
    )
    if provider == "ibm":
        assert all(options["dynamical_decoupling"] == {"enable": False} for options in probe.options)
        assert all(options["twirling"] == {"enable_gates": False, "enable_measure": False} for options in probe.options)
        assert all(options["execution"] == {"meas_type": "classified", "init_qubits": True} for options in probe.options)
    else:
        assert all(options == {} for options in probe.options)
    document = read_json(result.artifact_dir / "experiment.json")
    assert document["schema_version"] == 4
    assert document["spec"]["backend"]["execution_mode"] == "hardware"
    assert document["backend"]["kind"] == provider
    assert "token" not in json.dumps(document).lower()
    before = len(probe.submissions)
    loaded = resume_randomized_experiment(result.artifact_dir, adapter=object())
    assert loaded.to_safe_dict() == result.to_safe_dict()
    assert len(probe.submissions) == before


@pytest.mark.parametrize("provider", ["ibm", "iqm", "piastq"])
def test_confirmed_job_resume_fetches_existing_remote_job_without_replaying_completed_blocks(tmp_path, provider):
    spec = make_spec(tmp_path, provider, limit=3)
    probe = ProviderProbe(provider)
    probe.fail_result_job = 1
    result = run_randomized_experiment(spec, adapter=probe.adapter(spec), _randbelow=draws("two_qutrit"), _source="test_sequence")
    assert result.values["execution"]["status"] == "partial"
    assert len(probe.submissions) == 2
    first_counts = (result.artifact_dir / "batches/0000/counts.json").read_bytes()
    schedule = (result.artifact_dir / "schedule.json").read_bytes()
    expected_job_id = f"{provider}-contract-1"
    assert read_json(result.artifact_dir / "batches/0001/submission.json")["job_id"] == expected_job_id
    assert not (result.artifact_dir / "batches/0001/counts.json").exists()
    compilation_count = len(probe.compiled)
    probe.fail_result_job = None
    resumed = resume_randomized_experiment(result.artifact_dir, adapter=probe.adapter(spec))
    assert resumed.status.value == "completed", resumed.values
    assert probe.restores == [expected_job_id]
    assert len(probe.submissions) == 4
    assert len(probe.compiled) == compilation_count
    assert (result.artifact_dir / "batches/0000/counts.json").read_bytes() == first_counts
    assert (result.artifact_dir / "schedule.json").read_bytes() == schedule
    sent = [circuit.name for circuits, _ in probe.submissions for circuit in circuits]
    assert sent == [f"block_{index:08d}" for index in range(11)]
    assert resumed.values["budget"]["completed_shots"] == 11 * 7


def test_ibm_safe_execution_options_reach_runtime_and_resume(tmp_path):
    spec = make_spec(tmp_path, "ibm")
    probe = ProviderProbe("ibm")
    options = {"max_execution_time": 60, "execution": {"rep_delay": 0.002, "init_qubits": True}}
    result = run_randomized_experiment(spec, adapter=probe.adapter(spec), _randbelow=draws("two_qutrit"), run_options=options)
    assert result.status.value == "completed", result.values
    assert all(item["max_execution_time"] == 60 and item["execution"]["rep_delay"] == 0.002 for item in probe.options)
    assert resume_randomized_experiment(result.artifact_dir, adapter=object(), run_options=options).status.value == "completed"


def test_ibm_invalid_execution_options_fail_before_creating_submission_attempt(tmp_path):
    spec = make_spec(tmp_path, "ibm")
    probe = ProviderProbe("ibm")
    adapter = probe.adapter(spec)
    with pytest.raises(BackendCompatibilityError, match="outside"):
        adapter.validate_run_options({"execution": {"rep_delay": 0.1}})
    result = run_randomized_experiment(
        spec, adapter=adapter, _randbelow=draws("two_qutrit"),
        run_options={"execution": {"rep_delay": 0.1}},
    )
    assert result.status.value == "failed"
    assert probe.submissions == []
    assert not list(result.artifact_dir.glob("batches/*/attempt.json"))


@pytest.mark.parametrize("failure", ["shots", "pub_count"])
def test_ibm_invalid_provider_results_are_persisted_as_evidence_and_stop(tmp_path, failure):
    spec = make_spec(tmp_path, "ibm")
    probe = ProviderProbe("ibm")
    submit = probe.submit

    def malformed_submit(circuits, shots, options):
        job = submit(circuits, shots, options)
        if failure == "shots":
            job.histograms = tuple({"0" * circuit.num_clbits: shots - 1} for circuit in circuits)
        else:
            job.histograms = job.histograms[:-1]
        return job

    probe.submit = malformed_submit
    result = run_randomized_experiment(spec, adapter=probe.adapter(spec), _randbelow=draws("two_qutrit"))
    assert result.status.value == "failed"
    assert len(probe.submissions) == 1
    counts_path = result.artifact_dir / "batches/0000/counts.json"
    assert counts_path.is_file()
    evidence = read_json(counts_path)
    assert evidence["counts"] == [dict(histogram) for histogram in probe.jobs["ibm-contract-0"].histograms]
    assert read_json(result.artifact_dir / "batches/0000/receipt.json")["valid"] is False
    assert result.values["budget"]["completed_shots"] == 0
    assert not (result.artifact_dir / "batches/0001/attempt.json").exists()
