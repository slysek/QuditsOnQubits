import numpy as np
import pytest

from qudits_on_qubits.benchmarks.theta_continuation import library
from qudits_on_qubits.benchmarks.theta_continuation.compilation import compile_cz3_candidate
from qudits_on_qubits.benchmarks.theta_continuation.cz3_template import load_cz3_template
from qudits_on_qubits.benchmarks.theta_continuation.f3 import synthesize_theta_f3
from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig


@pytest.fixture
def bundle():
    encoding = np.eye(4, 3)
    f3 = synthesize_theta_f3(encoding, transpiler_seeds=(0,))
    cz3 = compile_cz3_candidate(load_cz3_template(ThetaBenchmarkConfig().baseline_qpy).original_circuit, encoding, seeds=(0,))
    return {"metadata": {"correct": True, "selected_method": "continuation", "f3_alpha": f3.alpha,
                         "f3_metrics": f3.optimal_metrics, "continuation_seconds": 1.2,
                         "fallback_seconds": 0.0},
            "circuits": {"f3_optimal.qpy": f3.optimal, "cz3_selected.qpy": cz3.circuit}}


def test_adapter_recomputes_metrics_and_restores_metadata(bundle):
    bundle["circuits"]["cz3_selected.qpy"].metadata = {"E_norm": 99}
    gates = library.ThetaGateLibrary.from_bundle(bundle, np.eye(4, 3))
    metrics = gates.benchmark_metrics()
    assert metrics["gate_library"] == "theta_continuation_v1"
    assert metrics["E_norm"] < 1e-5
    assert metrics["N_2q"] == gates.cz3.count_ops()["cz"]
    assert metrics["f3_leakage_phase"] == bundle["metadata"]["f3_alpha"]
    assert metrics["cz3_synthesis_seed"] is None
    assert metrics["cz3_synthesis_epsilon"] is None
    assert gates.cz3 is not bundle["circuits"]["cz3_selected.qpy"]


def test_bqskit_provenance_only_for_bqskit_selection(bundle):
    bundle["metadata"]["selected_method"] = "bqskit"
    metrics = library.ThetaGateLibrary.from_bundle(bundle, np.eye(4, 3)).benchmark_metrics()
    assert metrics["cz3_synthesis_seed"] == 0
    assert metrics["cz3_synthesis_epsilon"] == 1e-8


def test_adapter_rejects_bad_gates_and_wrong_phase(bundle):
    bundle["metadata"]["f3_alpha"] += 0.3
    with pytest.raises(ValueError, match="F3"):
        library.ThetaGateLibrary.from_bundle(bundle, np.eye(4, 3))


def test_adapter_rejects_unsuccessful_point(bundle):
    bundle["metadata"]["correct"] = False
    with pytest.raises(ValueError, match="correct"):
        library.ThetaGateLibrary.from_bundle(bundle, np.eye(4, 3))
