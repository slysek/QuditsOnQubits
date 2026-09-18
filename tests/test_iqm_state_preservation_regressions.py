"""Regressions for phase-preserving state preparation and IQM QPY exports."""

import json

import numpy as np
import pytest
from qiskit import qpy

from scripts.run_direct_basis_benchmarks import (
    _iqm_strategy_names_from_args, _validate_cli_selection_args, build_parser,
)
from qudits_on_qubits.benchmarks.direct_basis.benchmark import (
    _safe_fidelity,
    _save_qpy,
    benchmark_direct_basis,
)
from qudits_on_qubits.benchmarks.direct_basis.circuits import build_direct_basis_graph_state_circuit
from qudits_on_qubits.benchmarks.direct_basis.circuit_serialization import normalize_circuit_bit_indices
from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import build_iqm_pass_manager
from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_harness import (
    _export_trial_transpiled_circuit,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_strategies import (
    get_iqm_transpiler_strategy,
)


def test_state_preparation_cli_defaults_preserve_relative_phases():
    args = build_parser().parse_args(["--state", "ghz3", "--iqm-backend", "garnet"])
    names = _iqm_strategy_names_from_args(args)
    assert names
    assert all(not get_iqm_transpiler_strategy(name).remove_final_rzs for name in names)


def test_cli_rejects_phase_dropping_before_loading_backend():
    args = build_parser().parse_args([
        "--state", "ghz3", "--iqm-backend", "garnet", "--iqm-strategy", "preset_default",
    ])
    with pytest.raises(ValueError, match="preserve relative phases"):
        _validate_cli_selection_args(args)


def test_bell_measurement_cli_retains_probability_preserving_strategies():
    args = build_parser().parse_args([
        "--state", "ghz3", "--iqm-backend", "garnet", "--ranking-workload", "bell_measurements",
    ])
    assert "preset_default" in _iqm_strategy_names_from_args(args)


@pytest.mark.parametrize("strategy", ["preset_default", "transpile_to_iqm_default"])
def test_state_preparation_rejects_phase_dropping_even_without_fidelity(strategy):
    with pytest.raises(ValueError, match="preserv.*phase|phase.*preserv"):
        benchmark_direct_basis(
            state_name="ghz3", n_qutrits=3, basis_matrix=np.eye(3),
            basis_candidate_name="E_old", basis_candidate_type="identity",
            transpiler_backend=object(), iqm_strategy_names=(strategy,),
            compute_fidelity=False, n_transpile_runs=1,
        )


@pytest.fixture
def compiled_ghz3(exact_cz3_synthesis):
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    reference = build_direct_basis_graph_state_circuit("ghz3", np.eye(3), n_qutrits=3)
    circuit = build_iqm_pass_manager(IQMFakeGarnet(), seed_transpiler=18).run(reference)
    return reference, circuit


@pytest.mark.parametrize("exporter", ["benchmark", "harness"])
def test_exported_ghz3_preserves_logical_state_and_layout(compiled_ghz3, tmp_path, exporter):
    reference, circuit = compiled_ghz3
    layout = circuit.layout.final_index_layout()
    before, notes = _safe_fidelity(reference, circuit, max_qubits=10)
    assert before == pytest.approx(1, abs=1e-10), notes
    if exporter == "benchmark":
        path = _save_qpy(circuit, str(tmp_path / "state.qpy"))
    else:
        path = _export_trial_transpiled_circuit(
            {"quantum_circuit_dir": str(tmp_path)}, circuit,
            strategy_name="preset_exact", seed_transpiler=18,
        )
    with open(path, "rb") as handle:
        loaded = qpy.load(handle)[0]
    fidelity, notes = _safe_fidelity(reference, loaded, max_qubits=10)
    assert fidelity == pytest.approx(1, abs=1e-10), notes
    assert loaded.layout.final_index_layout() == layout
    assert loaded.count_ops() == circuit.count_ops()


def test_normalization_preserves_measurements_conditions_and_source(compiled_ghz3):
    _, source = compiled_ghz3
    circuit = source.copy()
    circuit.measure_all()
    circuit.metadata = {"diagnostic": [18]}
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(circuit.qubits[0])
    before_indices = [circuit.find_bit(bit).index for bit in circuit.qubits]

    def instructions(qc):
        positions = {bit: index for index, bit in enumerate(qc.qubits)}
        return [(inst.operation.name, tuple(positions[b] for b in inst.qubits), inst.clbits)
                for inst in qc.data]

    normalized = normalize_circuit_bit_indices(circuit)
    assert instructions(normalized) == instructions(circuit)
    assert normalized.cregs == circuit.cregs
    assert normalized.metadata == circuit.metadata
    assert [normalized.find_bit(bit).index for bit in normalized.qubits] == list(range(circuit.num_qubits))
    assert [circuit.find_bit(bit).index for bit in circuit.qubits] == before_indices
    assert normalize_circuit_bit_indices(normalized) is normalized


def test_benchmark_csv_mapping_matches_exported_circuit(tmp_path, exact_cz3_synthesis):
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    row = benchmark_direct_basis(
        state_name="ghz3", n_qutrits=3, basis_matrix=np.eye(3),
        basis_candidate_name="E_old", basis_candidate_type="identity",
        transpiler_backend=IQMFakeGarnet(), iqm_strategy_names=("preset_exact",),
        quantum_circuits_dir=str(tmp_path), n_transpile_runs=1,
    )
    assert row["success"], row["error_message"]
    assert row["fidelity"] == pytest.approx(1, abs=1e-10)
    with open(row["graph_state_transpiled_qpy"], "rb") as handle:
        circuit = qpy.load(handle)[0]
    active = sorted({circuit.find_bit(bit).index for inst in circuit.data for bit in inst.qubits})
    assert json.loads(row["active_physical_qubits"]) == active
    assert json.loads(row["final_index_layout"]) == circuit.layout.final_index_layout()
