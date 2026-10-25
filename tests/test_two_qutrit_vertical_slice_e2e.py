"""Real ideal-Aer acceptance tests for the public two-qutrit vertical slice."""

from __future__ import annotations

from hashlib import sha256
import json
from unittest.mock import Mock

import pytest

from qudits_on_qubits import AerIdeal
from qudits_on_qubits.experiments.backends import create_backend_adapter
from qudits_on_qubits.vertical_slice import (
    ArtifactIntegrityError,
    BellReferenceCircuitSpec,
    ExecutionSpec,
    QuditExperimentSpec,
    canonical_qutrit_encoding,
    load_run_manifest,
    run_vertical_slice,
)


def _spec(tmp_path, *, shots: int = 2048) -> QuditExperimentSpec:
    return QuditExperimentSpec(
        circuit=BellReferenceCircuitSpec("two_qutrit"),
        encoding=canonical_qutrit_encoding(),
        backend=AerIdeal(seed_simulator=42),
        execution=ExecutionSpec(shots=shots, seed=42),
        output_root=tmp_path,
    )


def test_two_qutrit_bell_runs_end_to_end_on_ideal_aer(tmp_path) -> None:
    completed = run_vertical_slice(_spec(tmp_path))

    assert completed.manifest.status == "completed"
    assert completed.result["benchmark"] == "two_qutrit"
    assert completed.result["bell_unconditional"]["real"] == pytest.approx(
        6.0, abs=0.15
    )
    assert completed.result["bell_conditional"]["real"] == pytest.approx(
        6.0, abs=0.15
    )
    assert completed.result["leakage_rate"] == 0.0
    assert completed.result["circuit_count"] == 9
    assert completed.result["total_shots"] == 9 * 2048

    manifest_path = completed.artifact_dir / "run-manifest.json"
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "run-manifest-v1"
    assert on_disk["experiment_spec"]["circuit"]["reference_id"] == "two_qutrit"
    assert on_disk["encoding"]["encoding_id"] == "canonical_ez"
    assert on_disk["backend"]["execution_mode"] == "ideal_simulator"
    assert on_disk["software"]["package_version"]

    loaded = load_run_manifest(completed.artifact_dir)
    assert loaded == completed.manifest
    roles = {artifact.role for artifact in loaded.artifacts}
    assert roles == {
        "source-circuits",
        "encoding",
        "logical-measurements",
        "postprocessing",
        "compiled-circuits",
        "counts",
        "result",
    }
    for artifact in loaded.artifacts:
        path = completed.artifact_dir / artifact.path
        assert sha256(path.read_bytes()).hexdigest() == artifact.sha256


def test_manifest_loader_rejects_tampered_artifact(tmp_path) -> None:
    completed = run_vertical_slice(_spec(tmp_path, shots=128))
    counts = completed.artifact_dir / "counts.json"
    counts.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="counts artifact hash mismatch"):
        load_run_manifest(completed.artifact_dir)


@pytest.mark.parametrize("method", ["compile", "submit", "result"])
def test_failed_run_preserves_evidence_without_resubmission(tmp_path, method) -> None:
    spec = _spec(tmp_path, shots=16)
    adapter = Mock(wraps=create_backend_adapter(spec.backend))
    getattr(adapter, method).side_effect = RuntimeError("The provider failed.")

    with pytest.raises(RuntimeError, match="The provider failed") as failure:
        run_vertical_slice(spec, adapter=adapter)

    manifest = load_run_manifest(failure.value.__qoq_artifact_dir__)
    assert manifest.status == "failed"
    assert manifest.failure["message"] == "The provider failed."
    assert adapter.submit.call_count == (0 if method == "compile" else 1)
    assert bool(manifest.jobs) is (method == "result")
    if method == "result":
        assert manifest.jobs["main"]["job_id"]


def test_public_top_level_exports_vertical_slice() -> None:
    from qudits_on_qubits import (
        BellReferenceCircuitSpec as PublicBellReferenceCircuitSpec,
        QuditExperimentSpec as PublicQuditExperimentSpec,
        RunManifest,
        run_vertical_slice as public_run_vertical_slice,
    )

    assert PublicBellReferenceCircuitSpec is BellReferenceCircuitSpec
    assert PublicQuditExperimentSpec is QuditExperimentSpec
    assert RunManifest.__name__ == "RunManifest"
    assert public_run_vertical_slice is run_vertical_slice
