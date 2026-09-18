from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qudits_on_qubits.benchmarks.config import BenchmarkConfig
from qudits_on_qubits.benchmarks.families import LocalSU2, SchmidtTheta
from qudits_on_qubits.benchmarks.workloads import two_qutrit_graph_circuit
from qudits_on_qubits.benchmarks.targets import BackendTarget
from qudits_on_qubits.benchmarks.synthesis import ExactSynthesis
from qudits_on_qubits.benchmarks.runner import run_benchmark
from qudits_on_qubits.benchmarks.artifacts import load_benchmark


@pytest.fixture
def config():
    return BenchmarkConfig(transpiler_seeds=(0, 1))


def test_both_families_routed_and_artifacts_reanalyze(tmp_path, config):
    result = run_benchmark(two_qutrit_graph_circuit(),
        families=[LocalSU2(samples=1), SchmidtTheta(points=2)],
        backend=BackendTarget.local(num_qubits=4, coupling_map=[(0,1),(1,0),(1,2),(2,1),(2,3),(3,2)]),
        synthesis=ExactSynthesis(), config=config, output_dir=tmp_path / "run")
    assert len(result.trials) == 8
    assert result.trials.success.all()
    assert (result.trials.fidelity > 1 - 1e-10).all()
    assert set(result.statistics.class_name) == {"canonical", "local_su2", "schmidt_theta"}
    assert not result.pareto_front.empty
    assert (result.trials.two_qubit_gate_count >= 0).all()
    restored = load_benchmark(result.output_dir)
    assert restored.statistics.to_json() == result.statistics.to_json()
    assert (result.output_dir / "report.md").is_file()
    assert (result.output_dir / "pareto.png").is_file()
    trial = result.trials[result.trials.success].iloc[0]
    path = result.output_dir / trial.graph_state_transpiled_qpy
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash"):
        load_benchmark(result.output_dir)


class BrokenSynthesis:
    def to_dict(self):
        return {"method": "broken"}

    def prepare(self, candidates, output_dir, progress=None):
        pass

    def build(self, candidate):
        raise ValueError("controlled synthesis failure")


def test_synthesis_failure_is_diagnostic_not_zero_cost(tmp_path, config):
    result = run_benchmark(two_qutrit_graph_circuit(),
        families=[LocalSU2(samples=1)], backend=BackendTarget.local(),
        synthesis=BrokenSynthesis(), config=config, output_dir=tmp_path / "failed")
    assert not result.trials.success.any()
    assert result.trials.two_qubit_gate_count.isna().all()
    assert result.pareto_front.empty
    assert set(result.trials.status) == {"synthesis_failed"}
    assert len(result.trials) == 4


@pytest.mark.parametrize("kwargs", [
    {"transpiler_seeds": (0,)}, {"transpiler_seeds": (1,1)},
    {"transpiler_seeds": (False,1)}, {"f3_tolerance": 1e-8},
    {"cz3_tolerance": float("nan")}, {"minimum_fidelity": 2},
    {"maximum_leakage": -1}, {"max_validation_qubits": 30},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        BenchmarkConfig(**kwargs)


def test_backend_required_and_existing_outputs_untouched(tmp_path, config):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    marker = run_dir / "keep"
    marker.write_text("unchanged")
    with pytest.raises((TypeError, ValueError)):
        run_benchmark(two_qutrit_graph_circuit(), families=[LocalSU2(samples=1)], backend=None,
                      synthesis=ExactSynthesis(), config=config, output_dir=tmp_path / "none")
    assert not (tmp_path / "none").exists()
    with pytest.raises(FileExistsError):
        run_benchmark(two_qutrit_graph_circuit(), families=[LocalSU2(samples=1)],
                      backend=BackendTarget.local(), synthesis=ExactSynthesis(),
                      config=config, output_dir=run_dir)
    assert marker.read_text() == "unchanged"

def test_provider_compilation_error_is_recorded(tmp_path, config, monkeypatch):
    from qiskit.exceptions import QiskitError
    from qudits_on_qubits.benchmarks.workloads import LogicalCircuit
    def fail(*args, **kwargs):
        raise QiskitError("controlled transpilation failure")
    monkeypatch.setattr(BackendTarget, "compile", fail)
    result = run_benchmark(LogicalCircuit(1, ()), families=[LocalSU2(samples=1)],
        backend=BackendTarget.local(), synthesis=ExactSynthesis(),
        config=config, output_dir=tmp_path / "compile-failure")
    assert set(result.trials.status) == {"compilation_failed"}
    assert not result.trials.success.any()


def test_artifact_corruption_is_fatal_not_a_scientific_failure(tmp_path, config):
    from qudits_on_qubits.benchmarks.synthesis import SynthesisArtifactError
    class CorruptSynthesis(BrokenSynthesis):
        def build(self, candidate):
            raise SynthesisArtifactError("artifact hash mismatch")
    with pytest.raises(SynthesisArtifactError, match="hash"):
        run_benchmark(two_qutrit_graph_circuit(), families=[LocalSU2(samples=1)],
            backend=BackendTarget.local(), synthesis=CorruptSynthesis(),
            config=config, output_dir=tmp_path / "corrupt")


def test_compilation_protocol_changes_comparison_identity(tmp_path, config):
    from qudits_on_qubits.benchmarks.workloads import LogicalCircuit
    ids = []
    for level in (0, 1):
        result = run_benchmark(LogicalCircuit(1, ()), families=[LocalSU2(samples=1)],
            backend=BackendTarget.local(optimization_level=level), synthesis=ExactSynthesis(),
            config=config, output_dir=tmp_path / str(level))
        ids.append(result.trials.comparison_id.iloc[0])
    assert ids[0] != ids[1]


@pytest.mark.parametrize("matching_workload", [True, False])
def test_saved_preparation_reused_with_independent_baseline(tmp_path, config, matching_workload):
    from qiskit import qpy
    from qudits_on_qubits.benchmarks.synthesis import SavedGateSynthesis, saved_candidate
    from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore
    from qudits_on_qubits.benchmarks.workloads import LogicalCircuit

    workload = two_qutrit_graph_circuit()
    candidate = next(LocalSU2(samples=1).generate())
    library = ExactSynthesis().build(candidate)
    source = workload.build(candidate.encoding, library)
    source.barrier()  # Marker proves that the original preparation survives import.
    bundle = tmp_path / "saved"
    bundle.mkdir()
    np.save(bundle / "E.npy", candidate.encoding)
    for name, circuit in (("F3_W.qpy", library.f3), ("CZ3_W.qpy", library.cz3),
                          ("graph_state_direct_basis.qpy", source)):
        with (bundle / name).open("wb") as handle:
            qpy.dump(circuit, handle)
    bound_hash = workload.stable_hash() if matching_workload else LogicalCircuit(2, ()).stable_hash()
    result = run_benchmark(workload, families=[], backend=BackendTarget.local(),
        synthesis=SavedGateSynthesis(bundle, workload_hash=bound_hash),
        baseline_synthesis=ExactSynthesis(), saved_candidates=[saved_candidate(bundle)],
        config=config, output_dir=tmp_path / "result")
    baseline = result.trials[result.trials.candidate_name == "canonical"]
    replay = result.trials[result.trials.candidate_name == "saved"]
    assert baseline.success.all()
    if matching_workload:
        assert replay.success.all()
        saved = RunStore.open(result.output_dir).read_bundle(replay.iloc[0].candidate_bundle)
        assert saved["circuits"]["source.qpy"] == source
        assert saved["metadata"]["synthesis_provenance"]["parameter_source"] == "saved_qpy"
    else:
        assert not replay.success.any()
        assert replay.error.str.contains("workload hash mismatch").all()
        assert "saved" not in set(result.pareto_front.candidate_name)
