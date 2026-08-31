from __future__ import annotations

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


def test_real_ideal_aer_run_writes_single_reproducible_completed_result(
    tmp_path: Path,
) -> None:
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
    assert first.backend["metadata"] == {
        "method": "statevector",
        "noise_model": None,
    }
    assert document["schema_version"] == 3
    assert document["status"] == "completed"
    expected_spec = spec.to_safe_dict()
    expected_spec.pop("workload_optimization")
    assert document["spec"] == expected_spec
    assert "workload_optimization" not in document["spec"]
    assert document["backend"] == first.backend
    assert document["result"] == first.values
    assert document["calibration"] is None
    assert {path.name for path in first.artifact_dir.iterdir()} == {
        "experiment.json"
    }
    assert "sha256" not in json.dumps(document).lower()

    assert list(document["counts_by_factor"]) == ["1"]
    saved_counts = document["counts_by_factor"]["1"]
    assert saved_counts
    assert all(isinstance(entry["setting"], list) for entry in saved_counts)
    assert all(sum(entry["counts"].values()) == 64 for entry in saved_counts)
    assert set(document["result"]) == {
        "raw",
        "raw_conditional",
        "raw_unconditional",
        "raw_invalid_codeword_rate",
        "raw_invalid_codeword_shots",
        "config",
        "diagnostics",
    }
    _assert_finite_estimate(document["result"]["raw"])

    resumed = resume_experiment(first.artifact_dir, adapter=object())
    assert resumed.to_safe_dict() == first.to_safe_dict()

    second = run_experiment(spec)
    second_document = json.loads(
        (second.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    assert second.experiment_id != first.experiment_id
    assert second.artifact_dir != first.artifact_dir
    assert {path.name for path in second.artifact_dir.iterdir()} == {
        "experiment.json"
    }
    assert second_document["counts_by_factor"]["1"] == saved_counts
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
        "ExperimentDurabilityError",
        "ExperimentError",
        "ExecutionMode",
        "ExperimentPersistenceError",
        "ExperimentResult",
        "ExperimentSpec",
        "ExperimentStatus",
        "ExperimentStore",
        "ExperimentValidationError",
        "IQMHardware",
        "IQMQubitSelectorConfig",
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
        "RunManifest",
        "ScalarEstimate",
        "TranspilationConfig",
        "WorkloadMetrics",
        "WorkloadOptimizationConfig",
        "ZNEBootstrapStrategy",
        "ZNEStrategy",
        "apply_readout_mitigation",
        "assignment_matrices_from_counts",
        "bootstrap_bell_results",
        "build_m3_mitigation",
        "build_readout_calibration_circuits",
        "calibration_cache_is_valid",
        "choose_workload_ranking_basis",
        "create_backend_adapter",
        "fold_cz_batch",
        "linear_zne_extrapolate",
        "resume_experiment",
        "run_experiment",
        "run_experiments",
        "summarize_compiled_workload",
        "validate_zne_factors",
        "workload_rank_key",
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
    'BellEstimate', 'ComplexConfidenceInterval', 'ExecutionMode', 'RunManifest',
    'run_experiment',
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
        "transpile_to_iqm",
        "backend.run",
        "schema-v3",
        "compiled qpy",
        "sha-256 manifests",
        "unfinished runs",
        "no silent fallback",
        "environment variables",
        "drift",
        "model bias",
    )
    assert all(statement in readme for statement in required)
    assert "uncertainty=BootstrapConfig(" in readme_source
    assert "ExecutionMode" in readme_source
    assert "RunManifest" in readme_source
    assert "legacy schema-v1/schema-v2" in readme_source
    assert "resume_experiment(results[0].artifact_dir)" in readme_source
    assert "execution_mode=ExecutionMode.HARDWARE" in readme_source
    assert "bootstrap=BootstrapConfig(" not in readme_source
