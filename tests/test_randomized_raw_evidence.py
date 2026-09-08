"""Strict legacy adapter errors retain safe provider evidence for block runs."""

import json
from types import SimpleNamespace

import pytest

from qudits_on_qubits.experiments.backends.base import (
    BackendIdentity, BaseBackendAdapter, SubmittedJob,
)
from qudits_on_qubits.experiments.errors import JobResultError
from qudits_on_qubits.experiments.raw_evidence import attach_raw_evidence
from test_experiment_piastq_adapter import _PiastJob, _adapter


@pytest.mark.parametrize("counts", [
    [{"0000": 7}, {"0000": 8}], [{"0000": 8}],
    [{"0000": -1}, {"0000": 8}], [{"0000": True}, {"0000": 8}],
    [{"0000": 8.0}, {"0000": 8}],
])
def test_base_result_keeps_extracted_invalid_counts_while_raising(counts):
    identity = BackendIdentity("aer_ideal", "local-test")
    result = SimpleNamespace(get_counts=lambda index=None: counts if index is None else counts[index])
    job = SubmittedJob("job-1", SimpleNamespace(result=lambda: result), identity, 2, 8)
    adapter = SimpleNamespace(resolve=lambda: identity)
    with pytest.raises(JobResultError) as caught:
        BaseBackendAdapter.result(adapter, job)
    evidence = caught.value.raw_evidence
    assert evidence["counts"] == counts
    assert evidence["job_id"] == "job-1"
    assert evidence["target_identity"] == identity.to_safe_dict()
    assert evidence["metadata"]["validation"] == "invalid_provider_raw"
    json.dumps(evidence, allow_nan=False)


def test_base_partial_extraction_retains_successful_prefix_and_missing_position():
    identity = BackendIdentity("aer_ideal", "local-test")

    def get_counts(index=None):
        if index == 0:
            return {"0000": 8}
        raise RuntimeError("unavailable second circuit")

    job = SubmittedJob("job-1", SimpleNamespace(result=lambda: SimpleNamespace(get_counts=get_counts)),
                       identity, 2, 8)
    with pytest.raises(JobResultError) as caught:
        BaseBackendAdapter.result(SimpleNamespace(resolve=lambda: identity), job)
    assert caught.value.raw_evidence["counts"] == [{"0000": 8}, None]


@pytest.mark.parametrize("counts", [
    [{"0000": 7}, {"0000": 8}], [{"0000": 8}],
    [{"0000": -1}, {"0000": 8}], [{"0000": True}, {"0000": 8}],
])
def test_piast_result_keeps_extracted_invalid_counts_while_raising(counts):
    adapter, *_ = _adapter(job=_PiastJob(counts=counts))
    job = adapter.submit((object(), object()), 8)
    with pytest.raises(JobResultError) as caught:
        adapter.result(job)
    assert caught.value.raw_evidence["counts"] == counts
    assert caught.value.raw_evidence["job_id"] == job.job_id
    json.dumps(caught.value.raw_evidence, allow_nan=False)


def test_helper_keeps_missing_pub_positions_and_register_diagnostics():
    identity = BackendIdentity("ibm", "ibm-test")
    counts = [{"001": 7}, None, {"101": 8}]
    diagnostics = {"pubs": [{"pub_index": 1, "registers": [
        {"name": "readout", "bitstrings": ["00", "11"]},
    ]}]}
    error = JobResultError("invalid joint mapping")
    assert attach_raw_evidence(error, job_id="job-1", target_identity=identity,
                               counts=counts, diagnostics=diagnostics) is error
    assert error.raw_evidence["counts"] == counts
    assert error.raw_evidence["metadata"]["diagnostics"] == diagnostics
    json.dumps(error.raw_evidence, allow_nan=False)


def test_helper_does_not_serialize_secrets_objects_or_nonfinite_numbers():
    identity = BackendIdentity("ibm", "ibm-test", metadata={"api_token": "synthetic-secret"})
    cycle = {}
    cycle["self"] = cycle
    error = attach_raw_evidence(JobResultError("invalid counts"), job_id="job-1",
        target_identity=identity,
        counts=[{"0000": float("nan"), "1111": float("inf"), "api_token": "synthetic-secret", "0101": object()}],
        diagnostics={"password": "synthetic-secret", "unsafe": "token=synthetic-secret", "cycle": cycle})
    serialized = json.dumps(error.raw_evidence, allow_nan=False)
    assert "synthetic-secret" not in serialized
    assert "api_token" not in serialized
    assert error.raw_evidence["metadata"]["evidence_incomplete"] is True


def test_helper_bounds_large_register_diagnostics(monkeypatch):
    import qudits_on_qubits.experiments.raw_evidence as module
    monkeypatch.setattr(module, "MAX_EVIDENCE_ITEMS", 20)
    identity = BackendIdentity("ibm", "ibm-test")
    error = attach_raw_evidence(JobResultError("invalid PUB"), job_id="job-1", target_identity=identity,
                               counts=[None], diagnostics={"bitstrings": ["00"]*1000})
    assert len(error.raw_evidence["metadata"]["diagnostics"]["bitstrings"]) < 1000
    assert error.raw_evidence["metadata"]["evidence_incomplete"] is True


def test_helper_bounds_malformed_histogram_entries(monkeypatch):
    import qudits_on_qubits.experiments.raw_evidence as module
    monkeypatch.setattr(module, "MAX_EVIDENCE_ITEMS", 20)
    error = attach_raw_evidence(JobResultError("invalid histogram"), job_id="job-1",
        target_identity=BackendIdentity("ibm", "ibm-test"),
        counts=[{f"malformed-{i}": object() for i in range(1000)}])
    assert len(error.raw_evidence["counts"][0]) < 1000
    assert error.raw_evidence["metadata"]["evidence_incomplete"] is True
    assert error.raw_evidence["job_id"] == "job-1"
    assert error.raw_evidence["target_identity"]["kind"] == "ibm"


def test_bounded_evidence_fits_strict_store_container_and_numeric_limits():
    from qudits_on_qubits.experiments.store import ExperimentStore

    error = attach_raw_evidence(JobResultError("invalid raw"), job_id="job-1",
        target_identity=BackendIdentity("ibm", "ibm-test"),
        counts=[{"0000": 2**2000}],
        diagnostics={"bitstrings": ["01"]*20000})
    ExperimentStore.validate_plain_json(error.raw_evidence)
    assert error.raw_evidence["metadata"]["evidence_incomplete"] is True
