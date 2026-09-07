"""Offline contract tests using the real optional client and only a fake HTTP boundary."""
from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter

from qudits_on_qubits import (
    BootstrapConfig, ExperimentSpec, PathBasis, PiastQHardware,
    TranspilationConfig, run_experiment,
)
from qudits_on_qubits.experiments.backends import PiastQAdapter
from qudits_on_qubits.experiments.errors import BackendCompatibilityError


@pytest.fixture
def managed():
    cft = pytest.importorskip("cft_piastq")
    httpx = pytest.importorskip("httpx")
    requests, circuits = [], []
    state = {"polls": 0, "shots": 0}

    def transport(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/api/runner/health":
            return httpx.Response(200, json={"runner_available": True})
        if request.method == "POST" and request.url.path == "/api/runner/jobs":
            payload = json.loads(request.content)
            assert payload["owner"] == "offline-test"
            state["shots"] = payload["shots"]
            for entry in payload["circuits"]:
                encoded = base64.b64decode(entry["qpy_base64"])
                assert encoded[6] == 13, "managed runner requires Qiskit 1.4-compatible QPY"
                circuits.extend(qpy.load(io.BytesIO(encoded)))
            return httpx.Response(200, json={"server_job_id": "managed-test"})
        if request.url.path == "/api/runner/jobs/managed-test":
            state["polls"] += 1
            status = "running" if state["polls"] == 1 else "succeeded"
            return httpx.Response(200, json={"status": status})
        if request.url.path == "/api/runner/jobs/managed-test/result":
            # Distinct outcomes verify ordering and little-endian decoding.
            return httpx.Response(200, json={
                "quasi_dists": [{str(i % 3): 1.0} for i in range(len(circuits))],
                "shots": state["shots"],
            })
        raise AssertionError(f"Unexpected request: {request.method} {request.url.path}")

    with httpx.Client(transport=httpx.MockTransport(transport)) as http_client:
        def client_type(**kwargs):
            return cft.PiastQClient(**kwargs, http_client=http_client)

        adapter = PiastQAdapter(
            PiastQHardware(owner="offline-test"), client_type=client_type,
            sampler_type=cft.PiastQSampler,
            env_loader=lambda _: {"dashboard_api_url": "https://dashboard.invalid"},
            poll_interval=0.001,
        )
        yield adapter, requests, circuits


def test_real_managed_handle_accepts_logical_circuits(managed):
    adapter, requests, _ = managed
    source = QuantumCircuit(4, 4)
    source.h(0)
    source.cx(0, 3)
    source.measure(range(4), range(4))
    compiled = adapter.compile([source], TranspilationConfig())
    assert compiled.circuits == (source,)
    assert compiled.metadata["transpilation"]["compilation_owner"] == "managed_runner"
    assert not any(method == "POST" for method, _ in requests)


@pytest.mark.parametrize("config", [
    TranspilationConfig(optimization_level=1),
    TranspilationConfig(seed_transpiler=7),
    TranspilationConfig(initial_layout=(0, 1, 2, 3)),
    TranspilationConfig(layout_method="dense"),
    TranspilationConfig(routing_method="sabre"),
    TranspilationConfig(scheduling_method="aqt"),
])
def test_managed_rejects_unforwarded_transpilation_options(managed, config):
    adapter, requests, _ = managed
    with pytest.raises(BackendCompatibilityError, match="managed runner"):
        adapter.compile([QuantumCircuit(4)], config)
    assert requests == []


def test_managed_rejects_unbound_parameters_before_http(managed):
    adapter, requests, _ = managed
    source = QuantumCircuit(1)
    source.rx(Parameter("theta"), 0)
    with pytest.raises(BackendCompatibilityError, match="bound"):
        adapter.compile([source], TranspilationConfig())
    assert requests == []


def test_managed_experiment_qpy_polling_counts_and_bell_result(managed, tmp_path):
    from qudits_on_qubits.bell_measurements import (
        build_sampler_circuits_for_candidate, canonical_Ez, compute_bell_value_from_counts,
    )
    from qudits_on_qubits.bell_measurements.sampler_circuits import decoding_kwargs_from_metadata

    adapter, requests, received = managed
    basis = tmp_path / "basis"
    basis.mkdir()
    source = QuantumCircuit(4)
    encoding = canonical_Ez()
    with (basis / "graph_state_direct_basis.qpy").open("wb") as handle:
        qpy.dump([source], handle)
    np.save(basis / "E.npy", encoding)
    expected, metadata = build_sampler_circuits_for_candidate(
        candidate="two_qutrit", state_circuit=source, E=encoding,
        qutrit_qubits=((2, 3), (0, 1)),
    )
    result = run_experiment(ExperimentSpec(
        state="two_qutrit", basis=PathBasis(basis), backend=PiastQHardware(owner="offline-test"),
        shots=32, uncertainty=BootstrapConfig(samples=10, seed=7), output_root=tmp_path / "runs",
    ), adapter=adapter)
    document = json.loads((result.artifact_dir / "experiment.json").read_text())
    assert document["status"] == "completed"
    assert document["transpilation"] == {
        "compilation_owner": "managed_runner", "circuit_representation": "logical",
    }
    assert document["job_ids"] == ["managed-test"]
    assert len(received) == len(expected)
    for actual, original in zip(received, expected, strict=True):
        assert actual == original
    counts = [{format(i % 3, "04b"): 32} for i in range(len(expected))]
    assert [entry["counts"] for entry in document["counts_by_factor"]["1"]] == counts
    by_setting = dict(zip(map(tuple, metadata["setting_by_circuit_index"]), counts, strict=True))
    bell = compute_bell_value_from_counts(
        by_setting, metadata["terms"], metadata["qutrit_bit_indices_by_setting"],
        **decoding_kwargs_from_metadata(metadata),
    )
    estimate = document["result"]["raw"]["estimate"]
    assert complex(estimate["real"], estimate["imag"]) == pytest.approx(bell)
    assert requests.count(("POST", "/api/runner/jobs")) == 1
    assert requests.count(("GET", "/api/runner/jobs/managed-test")) >= 2
    assert requests.count(("GET", "/api/runner/jobs/managed-test/result")) == 1
    from qudits_on_qubits import resume_experiment

    before_resume = list(requests)
    restored = resume_experiment(result.artifact_dir)
    assert restored.values == result.values
    assert requests == before_resume
