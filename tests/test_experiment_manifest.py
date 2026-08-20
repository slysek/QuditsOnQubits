from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.experiments.backends import (
    Availability,
    BackendCapabilities,
    BackendIdentity,
)
from qudits_on_qubits.experiments.errors import (
    ExperimentPersistenceError,
    ExperimentValidationError,
)
from qudits_on_qubits.experiments.execution import ExecutionMode
from qudits_on_qubits.experiments.models import AerIdeal, ExperimentSpec, PathBasis


TIMESTAMP = "2026-08-20T12:00:00.000000Z"


def resolved_backend(kind: str = "aer_ideal") -> dict[str, object]:
    identity = BackendIdentity(kind, "target", provider="test-provider")
    capabilities = BackendCapabilities(local=True, supports_resume=False)
    return {
        "identity": identity.to_safe_dict(),
        "capabilities": capabilities.to_safe_dict(),
        "metadata": {
            "identity": identity.to_safe_dict(),
            "capabilities": capabilities.to_safe_dict(),
        },
        "availability": Availability(True).to_safe_dict(),
    }


def manifest_document(
    *,
    status: str = "created",
    backend: dict[str, object] | None = None,
) -> dict[str, object]:
    spec = ExperimentSpec(
        state="two_qutrit",
        basis=PathBasis("basis"),
        backend=AerIdeal(seed_simulator=11),
        shots=64,
        output_root="runs",
        tags={"purpose": "contract"},
    )
    history = [{"status": "created", "timestamp": TIMESTAMP}]
    timestamps = {"created": TIMESTAMP, "updated": TIMESTAMP}
    if status != "created":
        history.append({"status": status, "timestamp": TIMESTAMP})
        timestamps[status] = TIMESTAMP
    return {
        "schema_version": 2,
        "experiment_id": "20260820T120000.000000Z-contract-abcdef123456",
        "spec": spec.to_safe_dict(),
        "status": status,
        "timestamps": timestamps,
        "status_history": history,
        "attempts": [],
        "backend": backend,
        "jobs": {},
        "job_ids": [],
        "source": None,
        "circuits": {"source": None, "logical": None, "factors": {}},
        "counts": {},
        "postprocessing": None,
        "calibration": None,
        "result": None,
        "result_artifact": None,
        "failure": None,
    }


def test_manifest_round_trip_is_deeply_immutable_and_returns_fresh_copies():
    from qudits_on_qubits.experiments.manifest import RunManifest

    document = manifest_document()
    document["circuits"]["factors"]["1"] = {
        "artifact": "compiled-factor-1.qpy",
        "sha256": "a" * 64,
        "circuit_count": 1,
    }
    manifest = RunManifest.from_safe_dict(document)
    document["circuits"]["factors"]["1"]["artifact"] = "changed.qpy"
    copy = manifest.to_safe_dict()
    copy["circuits"]["factors"]["1"]["artifact"] = "copy.qpy"

    assert manifest.schema_version == 2
    assert manifest.execution_mode is ExecutionMode.IDEAL_SIMULATOR
    assert manifest.circuits["factors"]["1"]["artifact"] == "compiled-factor-1.qpy"
    assert manifest.to_safe_dict()["circuits"]["factors"]["1"]["artifact"] == "compiled-factor-1.qpy"
    with pytest.raises(TypeError):
        manifest.circuits["factors"]["1"]["artifact"] = "changed.qpy"


@pytest.mark.parametrize(
    ("backend_kind", "expected_mode"),
    [
        ("aer_ideal", "ideal_simulator"),
        ("noisy_simulator", "noisy_simulator"),
        ("iqm_hardware", "hardware"),
        ("piastq_hardware", "hardware"),
    ],
)
def test_schema_v1_builtin_backend_normalizes_in_memory(backend_kind, expected_mode):
    from qudits_on_qubits.experiments.manifest import RunManifest

    document = manifest_document()
    document["schema_version"] = 1
    document["spec"]["backend"] = {"kind": backend_kind}
    manifest = RunManifest.from_safe_dict(document)

    assert manifest.schema_version == 2
    assert manifest.to_safe_dict()["spec"]["backend"]["execution_mode"] == expected_mode
    assert "execution_mode" not in document["spec"]["backend"]


def test_schema_v1_custom_backend_is_rejected_without_guessing():
    from qudits_on_qubits.experiments.manifest import RunManifest

    document = manifest_document()
    document["schema_version"] = 1
    document["spec"]["backend"] = {
        "kind": "custom",
        "identity": "legacy",
        "supports_resume": True,
    }
    with pytest.raises(ExperimentPersistenceError, match="custom"):
        RunManifest.from_safe_dict(document)


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        (lambda item: item.update(schema_version=99), ExperimentPersistenceError, "schema"),
        (lambda item: item.update(status="unknown"), ExperimentValidationError, "status"),
        (
            lambda item: item["spec"]["backend"].update(execution_mode="hardware"),
            ExperimentValidationError,
            "execution_mode",
        ),
        (
            lambda item: item["spec"]["tags"].update(purpose="token=manifest-secret"),
            ExperimentValidationError,
            "unsafe",
        ),
        (
            lambda item: item["circuits"]["factors"].update(
                {"1": {"artifact": "../escape.qpy", "sha256": "a" * 64}}
            ),
            ExperimentValidationError,
            "artifact",
        ),
    ],
)
def test_manifest_rejects_invalid_schema_state_mode_secret_and_artifact(
    mutation,
    error_type,
    message,
):
    from qudits_on_qubits.experiments.manifest import RunManifest

    document = manifest_document()
    mutation(document)
    with pytest.raises(error_type, match=message) as caught:
        RunManifest.from_safe_dict(document)
    assert "manifest-secret" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_manifest_rejects_resolved_identity_that_disagrees_with_spec():
    from qudits_on_qubits.experiments.manifest import RunManifest

    document = manifest_document(status="validated", backend=resolved_backend("custom"))
    with pytest.raises(ExperimentValidationError, match="identity"):
        RunManifest.from_safe_dict(document)


def test_manifest_does_not_mutate_caller_during_validation():
    from qudits_on_qubits.experiments.manifest import RunManifest

    document = manifest_document()
    before = deepcopy(document)
    RunManifest.from_safe_dict(document)
    assert document == before
