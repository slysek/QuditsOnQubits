from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import numpy as np
from qiskit import QuantumCircuit, qpy


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _write_two_qutrit_basis(directory: Path) -> None:
    directory.mkdir()
    source = QuantumCircuit(4, name="provider_contract_source")
    with (directory / "graph_state_direct_basis.qpy").open("wb") as handle:
        qpy.dump((source,), handle)
    np.save(directory / "E.npy", np.eye(4, 3, dtype=float), allow_pickle=False)


def _reject_network(*_args, **_kwargs):
    raise AssertionError("provider contract tests must not use the network")


class _ProviderResult:
    def __init__(self, circuits, shots):
        self._counts = [
            {"0" * max(1, circuit.num_clbits): shots}
            for circuit in circuits
        ]
        self.status = "DONE"
        self.time_taken = 0.1

    def get_counts(self, index=None):
        if index is None:
            return self._counts[0] if len(self._counts) == 1 else self._counts
        return self._counts[index]


class _IQMJob:
    def __init__(self, circuits, shots, job_id="iqm-contract-job"):
        self._job_id = job_id
        self._result = _ProviderResult(circuits, shots)

    def job_id(self):
        return self._job_id

    def status(self):
        return "DONE"

    def result(self, **_kwargs):
        return self._result


class _IQMBackend:
    name = "garnet"
    num_qubits = 64
    calibration_set_id = "contract-calibration"
    backend_version = "contract-1"

    def __init__(self):
        self.jobs = {}

    def run(self, circuits, **options):
        job = _IQMJob(circuits, options["shots"])
        self.jobs[job.job_id()] = job
        return job

    def retrieve_job(self, job_id):
        return self.jobs[job_id]

    def status(self):
        return SimpleNamespace(operational=True)


class _PiastJob:
    def __init__(self, circuits, shots, job_id="piastq-contract-job"):
        self._job_id = job_id
        self._counts = tuple(
            {"0" * max(1, circuit.num_clbits): shots}
            for circuit in circuits
        )

    def job_id(self):
        return self._job_id

    def status(self):
        return "DONE"

    def result(self, **_kwargs):
        return SimpleNamespace(status="DONE", time_taken=0.1)

    def counts(self):
        return self._counts


class _PiastBackend:
    name = "piastq-main"
    num_qubits = 64
    backend_version = "contract-1"

    def status(self):
        return SimpleNamespace(operational=True)


def _piast_types():
    jobs = {}
    backend = _PiastBackend()

    class Client:
        def __init__(self, **_kwargs):
            self.backend = backend

        def retrieve_job(self, job_id):
            return jobs[job_id]

    class Sampler:
        def __init__(self, actual_backend, *, options):
            assert actual_backend is backend
            assert options == {}

        def run(self, circuits, *, shots):
            job = _PiastJob(circuits, shots)
            jobs[job.job_id()] = job
            return job

    return Client, Sampler


def test_one_scientific_spec_uses_aer_iqm_and_piastq_provider_contracts(
    tmp_path, monkeypatch
):
    from qudits_on_qubits.experiments.backends import IQMAdapter, PiastQAdapter
    import qudits_on_qubits.experiments.backends.piastq as piastq_module
    from qudits_on_qubits.experiments.execution import ExecutionMode
    from qudits_on_qubits.experiments.models import (
        AerIdeal,
        BootstrapConfig,
        ExperimentSpec,
        ExperimentStatus,
        IQMHardware,
        PathBasis,
        PiastQHardware,
    )
    from qudits_on_qubits.experiments.runner import resume_experiment, run_experiment

    monkeypatch.setattr(socket, "getaddrinfo", _reject_network)
    monkeypatch.setattr(socket, "create_connection", _reject_network)
    monkeypatch.setattr(
        piastq_module,
        "transpile",
        lambda circuits, *, backend, **options: list(circuits),
    )

    basis = tmp_path / "basis"
    output_root = tmp_path / "runs"
    _write_two_qutrit_basis(basis)
    base_spec = ExperimentSpec(
        state="two_qutrit",
        basis=PathBasis(basis),
        backend=AerIdeal(seed_simulator=11),
        shots=64,
        bootstrap=BootstrapConfig(samples=2, seed=7),
        output_root=output_root,
        tags={"purpose": "provider-contract"},
    )
    iqm_spec = replace(base_spec, backend=IQMHardware("garnet"))
    piastq_spec = replace(base_spec, backend=PiastQHardware("managed", "team"))

    iqm_adapter = IQMAdapter(
        iqm_spec.backend,
        backend=_IQMBackend(),
        transpiler=lambda circuit, _backend, **_options: circuit,
    )
    client_type, sampler_type = _piast_types()
    piastq_adapter = PiastQAdapter(
        piastq_spec.backend,
        client_type=client_type,
        sampler_type=sampler_type,
        env_loader=lambda _path: {},
        poll_interval=0.01,
    )

    cases = (
        (base_spec, None, ExecutionMode.IDEAL_SIMULATOR, "aer_ideal"),
        (iqm_spec, iqm_adapter, ExecutionMode.HARDWARE, "iqm"),
        (piastq_spec, piastq_adapter, ExecutionMode.HARDWARE, "piastq"),
    )
    documents = []
    setting_orders = []
    for spec, adapter, expected_mode, expected_identity in cases:
        result = run_experiment(spec, adapter=adapter, _sleep=lambda _: None)
        document = json.loads(
            (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
        )

        assert result.status is ExperimentStatus.COMPLETED
        assert document["schema_version"] == 3
        assert document["status"] == "completed"
        assert document["backend"] == result.backend
        assert document["backend"]["kind"] == expected_identity
        assert document["spec"]["backend"]["execution_mode"] == expected_mode.value
        assert document["result"] == result.values
        assert set(document["result"]) == {
            "raw",
            "raw_conditional",
            "raw_unconditional",
            "raw_invalid_codeword_rate",
            "raw_invalid_codeword_shots",
            "config",
            "diagnostics",
        }
        assert document["result"]["raw"] == document["result"]["raw_conditional"]
        assert set(document["result"]["raw"]["estimate"]) == {"real", "imag"}
        assert {path.name for path in result.artifact_dir.iterdir()} == {
            "experiment.json"
        }
        assert "sha256" not in json.dumps(document).lower()
        assert "token" not in repr(document).lower()

        assert list(document["counts_by_factor"]) == ["1"]
        factor_counts = document["counts_by_factor"]["1"]
        assert factor_counts
        assert all(
            sum(item["counts"].values()) == spec.shots
            for item in factor_counts
        )
        setting_orders.append(
            tuple(tuple(item["setting"]) for item in factor_counts)
        )

        loaded = resume_experiment(result.artifact_dir, adapter=object())
        assert loaded.to_safe_dict() == result.to_safe_dict()
        documents.append(document)

    assert setting_orders[0] == setting_orders[1] == setting_orders[2]
    assert {frozenset(document) for document in documents} == {
        frozenset(documents[0])
    }
    scientific_specs = []
    for document in documents:
        serialized_spec = deepcopy(document["spec"])
        serialized_spec.pop("backend")
        scientific_specs.append(serialized_spec)
    assert scientific_specs[0] == scientific_specs[1] == scientific_specs[2]
