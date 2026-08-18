from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from qiskit import QuantumCircuit, qpy


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_two_qutrit_basis(directory: Path) -> None:
    directory.mkdir()
    source = QuantumCircuit(4, name="minimal_two_qutrit_source")
    with (directory / "graph_state_direct_basis.qpy").open("wb") as handle:
        qpy.dump((source,), handle)
    np.save(directory / "E.npy", np.eye(4, 3, dtype=float), allow_pickle=False)


def _assert_finite_estimate(estimate: dict[str, object]) -> None:
    point = estimate["estimate"]
    standard_error = estimate["standard_error"]
    confidence_interval = estimate["confidence_interval"]
    assert isinstance(point, dict)
    assert isinstance(standard_error, dict)
    assert isinstance(confidence_interval, dict)
    assert all(math.isfinite(point[component]) for component in ("real", "imag"))
    assert all(
        math.isfinite(standard_error[component]) and standard_error[component] >= 0
        for component in ("real", "imag")
    )
    for component in ("real", "imag"):
        interval = confidence_interval[component]
        assert isinstance(interval, dict)
        assert math.isfinite(interval["low"])
        assert math.isfinite(interval["high"])
        assert interval["low"] <= interval["high"]


def test_real_ideal_aer_run_is_durable_reproducible_and_resumable(tmp_path: Path) -> None:
    from qudits_on_qubits.experiments import (
        AerIdeal,
        BootstrapConfig,
        ExperimentSpec,
        ExperimentStatus,
        PathBasis,
        resume_experiment,
        run_experiment,
    )

    basis = tmp_path / "basis"
    output_root = tmp_path / "runs"
    _write_two_qutrit_basis(basis)
    spec = ExperimentSpec(
        state="two_qutrit",
        basis=PathBasis(basis),
        backend=AerIdeal(seed_simulator=11),
        shots=64,
        bootstrap=BootstrapConfig(samples=30, seed=7),
        output_root=output_root,
    )

    first = run_experiment(spec)
    document = json.loads(
        (first.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )

    assert first.status is ExperimentStatus.COMPLETED
    assert first.backend["kind"] == "aer_ideal"
    assert first.backend["provider"] == "qiskit-aer"
    assert first.backend["metadata"] == {"method": "statevector", "noise_model": None}
    assert document["backend"]["capabilities"]["local"] is True
    assert document["backend"]["capabilities"]["supports_resume"] is False
    assert [entry["status"] for entry in document["status_history"]] == [
        "created",
        "validated",
        "compiled",
        "submitted",
        "running",
        "postprocessing",
        "completed",
    ]

    timestamps = document["timestamps"]
    assert document["spec"] == spec.to_safe_dict()
    assert document["status"] == "completed"
    assert document["result"] == first.values
    assert timestamps["created"] == document["status_history"][0]["timestamp"]
    assert timestamps["updated"] == timestamps["completed"]
    for name in ("created", "validated", "compiled", "submitted", "running", "postprocessing", "completed"):
        assert datetime.fromisoformat(timestamps[name].replace("Z", "+00:00")).tzinfo is not None

    expected_artifacts = {
        "experiment.json",
        "source-state.qpy",
        "source-encoding.json",
        "logical-measurements.qpy",
        "postprocessing.json",
        "compiled-factor-1.qpy",
        "counts-factor-1.json",
        "result.json",
    }
    assert expected_artifacts == {path.name for path in first.artifact_dir.iterdir()}
    assert document["calibration"] is None
    assert not any(
        marker in path.name.lower()
        for path in first.artifact_dir.iterdir()
        for marker in ("remote", "dashboard", "web")
    )

    circuits = document["circuits"]
    assert _sha256(first.artifact_dir / circuits["source"]["artifact"]) == circuits["source"]["sha256"]
    assert _sha256(first.artifact_dir / circuits["logical"]["artifact"]) == circuits["logical"]["sha256"]
    factor = circuits["factors"]["1"]
    assert factor["artifact"] == "compiled-factor-1.qpy"
    assert factor["circuit_count"] > 0
    assert _sha256(first.artifact_dir / factor["artifact"]) == factor["sha256"]
    assert document["source"]["hashes"] == {
        "state": _sha256(basis / "graph_state_direct_basis.qpy"),
        "encoding": _sha256(basis / "E.npy"),
    }
    assert _sha256(first.artifact_dir / "source-encoding.json") == document["source"]["encoding_sha256"]
    assert _sha256(first.artifact_dir / "postprocessing.json") == document["postprocessing"]["sha256"]
    assert _sha256(first.artifact_dir / "result.json") == document["result_artifact"]["sha256"]

    counts_path = first.artifact_dir / document["counts"]["1"]["artifact"]
    assert _sha256(counts_path) == document["counts"]["1"]["sha256"]
    saved_counts = json.loads(counts_path.read_text(encoding="utf-8"))
    assert len(saved_counts["settings"]) == factor["circuit_count"]
    assert all(sum(entry["counts"].values()) == 64 for entry in saved_counts["settings"])

    assert set(document["result"]) == {"raw", "config", "diagnostics"}
    _assert_finite_estimate(document["result"]["raw"])

    resumed = resume_experiment(first.artifact_dir, adapter=object())
    assert resumed.to_safe_dict() == first.to_safe_dict()

    second = run_experiment(spec)
    assert second.experiment_id != first.experiment_id
    assert second.artifact_dir != first.artifact_dir
    assert json.loads((second.artifact_dir / "counts-factor-1.json").read_text(encoding="utf-8")) == saved_counts
    assert second.values == first.values


def test_experiments_public_surface_is_explicit() -> None:
    import qudits_on_qubits.experiments as experiments

    expected = {
        "AerIdeal",
        "Availability",
        "Backend",
        "BackendAdapter",
        "BackendCapabilities",
        "BackendCompatibilityError",
        "BackendIdentity",
        "BackendStatus",
        "BackendUnavailableError",
        "Basis",
        "BellEstimate",
        "BenchmarkBasis",
        "BootstrapBellResults",
        "BootstrapConfig",
        "BootstrapDiagnostics",
        "BootstrapInputs",
        "ComplexComponents",
        "ComplexConfidenceInterval",
        "ConfidenceInterval",
        "CustomBackend",
        "ExperimentError",
        "ExperimentPersistenceError",
        "ExperimentResult",
        "ExperimentSpec",
        "ExperimentStatus",
        "ExperimentStore",
        "ExperimentValidationError",
        "IQMHardware",
        "JobResultError",
        "JobSubmissionError",
        "LinearZNEFit",
        "MitigationConfig",
        "NoisySimulator",
        "OptionalDependencyError",
        "PathBasis",
        "PiastQHardware",
        "ReadoutBootstrapStrategy",
        "ReadoutCalibration",
        "ReadoutMitigationStrategy",
        "RetryConfig",
        "TranspilationConfig",
        "ZNEBootstrapStrategy",
        "ZNEStrategy",
        "apply_readout_mitigation",
        "assignment_matrices_from_counts",
        "bootstrap_bell_results",
        "build_m3_mitigation",
        "build_readout_calibration_circuits",
        "calibration_cache_is_valid",
        "create_backend_adapter",
        "fold_cz_batch",
        "linear_zne_extrapolate",
        "resume_experiment",
        "run_experiment",
        "run_experiments",
        "validate_zne_factors",
    }

    assert set(experiments.__all__) == expected
    assert all(getattr(experiments, name) is not None for name in expected)
    assert {"SubmittedJob", "ExecutionResult", "CompiledBatch", "backend_registry"}.isdisjoint(
        experiments.__all__
    )


def test_top_level_experiment_api_is_lazy_without_optional_dependencies_or_network() -> None:
    code = """
import builtins
import sys

def reject_network(event, _args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise AssertionError(f'network access attempted: {event}')
sys.addaudithook(reject_network)

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'mthree' or name.startswith('cft_piastq'):
        raise AssertionError(f'optional dependency imported: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

import qudits_on_qubits as qoq
assert 'qudits_on_qubits.experiments' not in sys.modules
uncertainty_names = (
    'BootstrapBellResults', 'BootstrapDiagnostics', 'BootstrapInputs',
    'ReadoutBootstrapStrategy', 'ZNEBootstrapStrategy', 'bootstrap_bell_results',
)
assert set(uncertainty_names).issubset(qoq.__all__)
for name in (
    'ExperimentSpec', 'PathBasis', 'BenchmarkBasis', 'AerIdeal', 'NoisySimulator',
    'IQMHardware', 'PiastQHardware', 'CustomBackend', 'MitigationConfig',
    'BootstrapConfig', 'TranspilationConfig', 'RetryConfig', 'ExperimentResult',
    'BellEstimate', 'ComplexConfidenceInterval', 'run_experiment',
    'run_experiments', 'resume_experiment',
) + uncertainty_names:
    assert getattr(qoq, name) is not None
import qudits_on_qubits.experiments as experiments
for name in uncertainty_names:
    assert getattr(qoq, name) is getattr(experiments, name)
assert 'mthree' not in sys.modules
assert 'cft_piastq' not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(SRC), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_readme_documents_library_runner_contracts() -> None:
    readme_source = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    readme = readme_source.lower()

    required = (
        "python library",
        "no dashboard",
        "no web",
        "no server",
        "benchmarkbasis",
        "aerideal",
        "noisysimulator",
        "iqmhardware",
        "piastqhardware",
        "custombackend",
        "run_experiments",
        "resume_experiment",
        "2000 local resamples",
        "not 2000 backend experiments",
        "raw",
        "readout_mitigated",
        "zne_readout_mitigated",
        "no silent fallback",
        "retrieve_job",
        "environment variables",
        "drift",
        "model bias",
    )
    assert all(statement in readme for statement in required)
    assert "uncertainty=BootstrapConfig(" in readme_source
    assert "bootstrap=BootstrapConfig(" not in readme_source
